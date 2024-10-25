from cProfile import label
import random
from collections import defaultdict

import numpy as np
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from tasks.cifar10_task import Cifar10Task
from tasks.fl.fl_task import FederatedLearningTask


class Cifar10_FedAvgTask(FederatedLearningTask, Cifar10Task):

    def load_data(self) -> None:
        self.load_cifar_data()
        train_loaders = self.assign_data(bias=self.params.fl_q)
        self.fl_train_loaders = train_loaders
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

        train_loaders = []
        for i in range(len(each_worker_data)):
            train_set = ClientDataset(each_worker_data[i], each_worker_label[i])
            train_loader = DataLoader(train_set,
                                      batch_size=self.params.batch_size,
                                      shuffle=True)
            train_loaders.append(train_loader)
        
        return train_loaders 


class ClientDataset(Dataset):
    def __init__(self, data_list, label_list):
        super().__init__()
        self.data_list = data_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        return self.data_list[index], self.label_list[index]
