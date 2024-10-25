import random
from collections import defaultdict
from typing import List, Any, Dict
from tasks.fl.fl_user_ours import FLUserOurs

import numpy as np
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split
import torch
from torchvision.utils import save_image

from tasks.mnist_task import MNISTTask
from tasks.fl.fl_task import FederatedLearningTask


class MNIST_OursTask(FederatedLearningTask, MNISTTask):

    def load_data(self) -> None:
        self.load_mnist_data()
        train_loaders, test_loaders = self.assign_data(bias=self.params.fl_q)
        self.fl_train_loaders = train_loaders
        self.fl_test_loaders = test_loaders
        return

    def assign_data(self, bias=1, p=0.1):
        num_labels = len(self.classes)
        num_workers = self.params.fl_total_participants
        server_pc = 0

        # assign data to the clients
        other_group_size = (1 - bias) / (num_labels - 1)
        worker_per_group = num_workers / num_labels

        #assign training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]   
        server_data = []
        server_label = [] 
        
        # compute the labels needed for each class
        real_dis = [1. / num_labels for _ in range(num_labels)]
        samp_dis = [0 for _ in range(num_labels)]
        num1 = int(server_pc * p)
        samp_dis[1] = num1
        average_num = (server_pc - num1) / (num_labels - 1)
        resid = average_num - np.floor(average_num)
        sum_res = 0.
        for other_num in range(num_labels - 1):
            if other_num == 1:
                continue
            samp_dis[other_num] = int(average_num)
            sum_res += resid
            if sum_res >= 1.0:
                samp_dis[other_num] += 1
                sum_res -= 1
        samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

        # randomly assign the data points based on the labels
        server_counter = [0 for _ in range(num_labels)]
        for x, y in self.train_dataset:
            upper_bound = y * (1. - bias) / (num_labels - 1) + bias
            lower_bound = y * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()
            
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y
            
            if server_counter[y] < samp_dis[y]:
                server_data.append(x)
                server_label.append(y)
                server_counter[y] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)
        
        random_order = np.random.RandomState(seed=self.params.random_seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]

        train_loaders, test_loaders = [], []
        for i in range(len(each_worker_data)):
            train_set = ClientDataset(each_worker_data[i], each_worker_label[i])
            tot = len(train_set)
            train_size = int(tot * self.params.attacker_train_ratio)
            test_size = tot - train_size
            train_set, test_set = random_split(train_set,
                                               lengths=[train_size, test_size], 
                                               generator=torch.Generator().manual_seed(self.params.random_seed))

            train_loader = DataLoader(train_set,
                                      batch_size=self.params.batch_size,
                                      shuffle=True)
            test_loader = DataLoader(test_set,
                                      batch_size=self.params.batch_size,
                                      shuffle=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
        
        return train_loaders, test_loaders

    def accumulate_weights_weighted(self, weight_accumulator, local_updates, genuine_scores):
        gs_sum = sum(genuine_scores.values())
        for user_id, local_update in local_updates.items():
            for name, value in local_update.items():
                weight_accumulator[name].add_(value * (genuine_scores[user_id] / (gs_sum + 1e-9)) * self.params.fl_total_participants)

    @torch.no_grad()
    def compute_genuine_score(self, model, dataloader, synthesizer):
        model.eval()
        correct = 0
        total = 0
        for i, data in enumerate(dataloader):
            batch = self.get_batch(i, data)
            batch = synthesizer.make_backdoor_batch(batch, test=True, attack=True)
            outputs = model(batch.inputs)
            
            pred_class_idx = torch.argmax(outputs, dim=1)
            correct += pred_class_idx[pred_class_idx==batch.labels].shape[0]
            total += batch.inputs.shape[0]
        
        return 1 - correct / total

    @torch.no_grad()
    def compute_genuine_score_global(self, model, dataloader, triggers, masks, target_cls):
        model.eval()
        correct = 0
        total = 0
        for i, data in enumerate(dataloader):
            batch = self.get_batch(i, data)
            images = batch.inputs
            trigger, mask = triggers[target_cls], masks[target_cls]

            triggerh = self.tanh_trigger(trigger)
            maskh = self.tanh_mask(mask)
            trojan_images = (1 - torch.unsqueeze(maskh, dim=0)) * images + torch.unsqueeze(maskh, dim=0) * triggerh
            outputs = model(trojan_images)
            labels = torch.tensor([target_cls] * batch.inputs.size(0)).to(self.params.device)

            pred_class_idx = torch.argmax(outputs, dim=1)
            correct += pred_class_idx[pred_class_idx==labels].shape[0]
            total += batch.inputs.shape[0]
        
        return 1 - correct / total

    def tanh_mask(self, vector):
        return torch.tanh(vector) / 2 + 0.5

    def tanh_trigger(self, vector):
        return (torch.tanh(vector) - 0.1307) / 0.3081

    def reverse_engineer_per_class(self, model, target_label, dataloader):
        model.eval()
        width, height = self.params.input_shape[1], self.params.input_shape[2]
        trigger = torch.randn((1, width, height))
        trigger = trigger.to(self.params.device).detach().requires_grad_(True)
        mask = torch.zeros((width, height))
        mask = mask.to(self.params.device).detach().requires_grad_(True)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([{"params": trigger}, {"params": mask}], lr=0.005)

        min_norm = np.inf
        min_norm_count = 0
        for epoch in range(self.params.nc_steps):
            norm = 0.0
            for i, data in enumerate(dataloader):
                batch = self.get_batch(i, data)
                optimizer.zero_grad()
                images = batch.inputs

                triggerh = self.tanh_trigger(trigger)
                maskh = self.tanh_mask(mask)
                trojan_images = (1 - torch.unsqueeze(maskh, dim=0)) * images + torch.unsqueeze(maskh, dim=0) * triggerh
                y_pred = model(trojan_images)
                y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(self.params.device)
                loss = criterion(y_pred, y_target) + 0.01 * torch.sum(maskh)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    norm = torch.sum(maskh)

            # early stopping
            if norm < min_norm:
                min_norm = norm
                min_norm_count = 0
            else: min_norm_count += 1
            if min_norm_count > 30: break

        return trigger, mask

    def reverse_engineer_trigger(self, model, dataloader):
        triggers, masks, norm_list = [], [], []
        for cls in range(len(self.classes)):
            trigger, mask = self.reverse_engineer_per_class(model, cls, dataloader)
            triggers.append(trigger)
            masks.append(mask)
            norm_list.append(torch.sum(self.tanh_mask(mask)).item())

            # visualize for debugging
            # batch = self.get_batch(0, next(iter(dataloader)))
            # images = batch.inputs

            # triggerh = self.tanh_trigger(trigger)
            # maskh = self.tanh_mask(mask)
            # trojan_images = (1 - torch.unsqueeze(maskh, dim=0)) * images + torch.unsqueeze(maskh, dim=0) * triggerh

            # save_image(images, 'runs/images_{}.png'.format(cls))
            # save_image(triggerh, 'runs/trigger_{}.png'.format(cls))
            # save_image(maskh, 'runs/mask_{}.png'.format(cls))
            # save_image(trojan_images, 'runs/trojan_images_{}.png'.format(cls))
            
        return triggers, masks, norm_list

    def sample_users_for_round(self, epoch) -> List[FLUserOurs]:
        sampled_ids = random.sample(
            range(self.params.fl_total_participants),
            self.params.fl_no_models)
        # sampled_ids = range(self.params.fl_total_participants)
        sampled_users = []
        for pos, user_id in enumerate(sampled_ids):
            train_loader = self.fl_train_loaders[user_id]
            test_loader = self.fl_test_loaders[user_id]
            compromised = self.check_user_compromised(epoch, pos, user_id)
            user = FLUserOurs(user_id, compromised=compromised,
                          train_loader=train_loader, test_loader=test_loader)
            sampled_users.append(user)

        return sampled_users


class ClientDataset(Dataset):
    def __init__(self, data_list, label_list):
        super().__init__()
        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        return self.data_list[index], self.label_list[index]