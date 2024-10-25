import argparse
import shutil
from datetime import datetime
import copy
from threading import Thread, Lock
from collections import defaultdict

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
from helper import Helper
from utils.utils import *

logger = logging.getLogger('logger')


def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True, ratio=None, report=True):
    criterion = hlpr.task.criterion
    model.train()

    for i, data in enumerate(train_loader):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, ratio)
        loss.backward()
        optimizer.step()

        if report:
            hlpr.report_training_losses_scales(i, epoch)
        if i == hlpr.params.max_batch_id:
            break

    return


def test(hlpr: Helper, epoch, backdoor=False):
    model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()

    with torch.no_grad():
        for i, data in enumerate(hlpr.task.test_loader):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric


def run(hlpr):
    acc = test(hlpr, 0, backdoor=False)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        acc = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)
        if hlpr.task.scheduler is not None:
            hlpr.task.scheduler.step(epoch)


def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        if epoch < hlpr.params.attack_start_epoch:
            run_fl_round_benign(hlpr, epoch)
        elif hlpr.params.ours:
            run_fl_round_ours_parallel(hlpr, epoch)
        elif hlpr.params.fltrust:
            run_fl_round_fltrust(hlpr, epoch)
        elif hlpr.params.defense == 'krum' or hlpr.params.defense == 'median':
            run_fl_round_byzantine(hlpr, epoch)
        else:
            run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)

        hlpr.save_model(hlpr.task.model, epoch, metric)


def run_fl_round_byzantine(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)

    local_updates = []
    for user in round_participants:
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True, report=False)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False, report=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        local_updates.append(local_update)
    
    local_update_final = globals()[hlpr.params.defense](local_updates, hlpr)
    for name, value in local_update_final.items():
        global_model.state_dict()[name].add_(value * hlpr.params.fl_eta)


def krum(w, hlpr):
    distances = defaultdict(dict)
    non_malicious_count = hlpr.params.fl_total_participants - hlpr.params.fl_number_of_adversaries
    num = 0
    for k in w[0].keys():
        if num == 0:
            for i in range(len(w)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
            num = 1
        else:
            for i in range(len(w)):
                for j in range(i):
                    distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
                    distances[i][j] += distances[j][i]
    minimal_error = 1e20
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user
    return w[minimal_error_index]


def median(w, hlpr):
    number_to_consider = hlpr.params.fl_total_participants
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        tmp = []
        for i in range(len(w)):
            tmp.append(w[i][k].cpu().numpy())
        tmp = np.array(tmp)
        med = np.median(tmp, axis=0)
        new_tmp = []
        for i in range(len(tmp)):
            new_tmp.append(tmp[i] - med)
        new_tmp = np.array(new_tmp)
        good_vals = np.argsort(abs(new_tmp), axis=0)[:number_to_consider]
        good_vals = np.take_along_axis(new_tmp, good_vals, axis=0)
        k_weight = np.array(np.mean(good_vals) + med)
        w_avg[k] = torch.from_numpy(k_weight).to(hlpr.params.device)
    return w_avg


def run_fl_round_benign(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in round_participants:
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(hlpr.params.fl_local_epochs):
            train(hlpr, local_epoch, local_model, optimizer,
                    user.train_loader, attack=False, report=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)

    hlpr.task.update_global_model(weight_accumulator, global_model)


def run_fl_round(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in round_participants:
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True, report=False)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False, report=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.accumulate_weights(weight_accumulator, local_update)

    hlpr.task.update_global_model(weight_accumulator, global_model)


def run_fl_round_fltrust(hlpr, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    ref_global_model = hlpr.task.build_model().to(hlpr.params.device)
    hlpr.task.copy_params(global_model, ref_global_model)
    optimizer = hlpr.task.make_optimizer(ref_global_model)
    for local_epoch in range(hlpr.params.fl_local_epochs):
        train(hlpr, local_epoch, ref_global_model, optimizer,
                        hlpr.task.clean_loader, attack=False, report=False)
    global_update = hlpr.task.get_fl_update(ref_global_model, global_model)

    benign_ids, malicious_ids = [], []
    for user in round_participants:
        if user.compromised:
            malicious_ids.append(user.user_id)
        else:
            benign_ids.append(user.user_id)

    local_updates = {}
    trust_scores = {}
    for user in round_participants:
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        for local_epoch in range(hlpr.params.fl_local_epochs):
            if user.compromised:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=True, report=False)
            else:
                train(hlpr, local_epoch, local_model, optimizer,
                      user.train_loader, attack=False, report=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised:
            hlpr.attack.fl_scale_update(local_update)

        # compute trust score, normalize magnitude of local model updates
        trust_score, norm_scale = ts_and_norm_scale(global_update, local_update)
        # update local update with norm sacle
        hlpr.attack.fl_scale_update(local_update, scale=norm_scale)

        local_updates[user.user_id] = local_update
        trust_scores[user.user_id] = trust_score
        
    benign_average = [trust_scores[i] for i in benign_ids]
    malicious_average = [trust_scores[i] for i in malicious_ids]
    benign_average = sum(benign_average) / (len(benign_average) + 1e-9)
    malicious_average = sum(malicious_average) / (len(malicious_average) + 1e-9)
    logger.warning('Trust Scores: Benign Average: {:.5f}, Malicious Average: {:.5f}'.format(benign_average, malicious_average))

    # compute the final update as weighted average over local updates with genuine scores
    weight_accumulator = hlpr.task.get_empty_accumulator()
    hlpr.task.accumulate_weights_weighted(weight_accumulator, local_updates, trust_scores)
    hlpr.task.update_global_model(weight_accumulator, global_model)


def run_fl_round_ours_parallel(hlpr, epoch):
    global_model = hlpr.task.model

    # Client Update
    round_participants = hlpr.task.sample_users_for_round(epoch)
    local_updates = {}

    benign_users, malicious_users = [], []
    benign_ids, malicious_ids = [], []
    for user in round_participants:
        if user.compromised:
            malicious_users.append(user)
            malicious_ids.append(user.user_id)
        else:
            benign_users.append(user)
            benign_ids.append(user.user_id)

    start = time.time()
    remaining_clients = len(benign_users)
    while remaining_clients > 0:
        thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
        threads = []
        for user in benign_users[len(benign_users) - remaining_clients: \
                                 len(benign_users) - remaining_clients + thread_pool_size]:
            thread = ClientThreadBenign(user, hlpr, global_model)
            threads.append(thread)
            thread.start()
        for thread in threads:
            update = thread.join()
            local_updates.update(update)

        remaining_clients -= thread_pool_size
    end = time.time()
    logger.info('Client time Bni: {}'.format(end - start))

    genuine_scores_approx = {}
    r_all_clients = {}

    start = time.time()
    remaining_clients = len(malicious_users)
    while remaining_clients > 0:
        thread_pool_size = min(remaining_clients, hlpr.params.max_threads)
        threads = []
        for user in malicious_users[len(malicious_users) - remaining_clients: \
                                    len(malicious_users) - remaining_clients + thread_pool_size]:
            thread = ClientThreadMalicious(user, hlpr, global_model)
            threads.append(thread)
            thread.start()

        for thread in threads:
            update, key, p_local_final, r_final = thread.join()
            local_updates.update(update)
            genuine_scores_approx[key] = p_local_final
            r_all_clients[key] = r_final

        remaining_clients -= thread_pool_size
    end = time.time()
    logger.info('Client time Mal: {}'.format(end - start))

    if hlpr.tb_writer is not None:
        hlpr.tb_writer.add_scalars('Client/Genuine_Scores_Approx', genuine_scores_approx, global_step=epoch)
        hlpr.tb_writer.add_scalars('Client/r', r_all_clients, global_step=epoch)
        hlpr.flush_writer()

    # Server Update
    start = time.time()
    # get reference model
    ref_global_model = hlpr.task.build_model().to(hlpr.params.device)
    hlpr.task.copy_params(global_model, ref_global_model)
    ref_weight_accumulator = hlpr.task.get_empty_accumulator()
    for local_update in local_updates.values():
        hlpr.task.accumulate_weights(ref_weight_accumulator, local_update)
    hlpr.task.update_global_model(ref_weight_accumulator, ref_global_model)

    # reverse engineer trigger
    triggers, masks, norm_list = hlpr.task.reverse_engineer_trigger(ref_global_model, hlpr.task.clean_loader)
    logger.warning(norm_list)
    target_cls = int(torch.argmin(torch.tensor(norm_list)))

    # compute genuine scores for each client
    genuine_scores_output = {}
    genuine_scores = {}
    for user_id, local_update in local_updates.items():
        # recover local model
        local_model = hlpr.task.build_model().to(hlpr.params.device)
        hlpr.task.copy_params(global_model, local_model)
        for name, update in local_update.items():
            model_weight = local_model.state_dict()[name]
            model_weight.add_(update)

        # compute genuine score for this local model
        p_global = hlpr.task.compute_genuine_score_global(local_model, 
                                                          hlpr.task.clean_loader,
                                                          triggers,
                                                          masks,
                                                          target_cls)
        
        genuine_scores[user_id] = p_global

        # Plotting (Part 1)
        if user_id in malicious_ids:
            key = 'Client {} (Malicious)'.format(user_id)
        else:
            key = 'Client {} (Benign)'.format(user_id)
        genuine_scores_output[key] = p_global
    
    # Plotting (Part 2) -- x-axis: global step, y-axis: genuine scores of all clients
    if hlpr.tb_writer is not None:
        hlpr.tb_writer.add_scalars('Server/Genuine_Scores', genuine_scores_output, global_step=epoch)
        hlpr.flush_writer()

    benign_average = [genuine_scores[i] for i in benign_ids]
    malicious_average = [genuine_scores[i] for i in malicious_ids]
    benign_average = sum(benign_average) / (len(benign_average) + 1e-9)
    malicious_average = sum(malicious_average) / (len(malicious_average) + 1e-9)
    logger.warning('Genuine Scores: Benign Average: {:.5f}, Malicious Average: {:.5f}'.format(benign_average, malicious_average))

    # compute the final update as weighted average over local updates with genuine scores
    weight_accumulator = hlpr.task.get_empty_accumulator()
    hlpr.task.accumulate_weights_weighted(weight_accumulator, local_updates, genuine_scores)
    hlpr.task.update_global_model(weight_accumulator, global_model)

    end = time.time()
    logger.info('Server time: {}'.format(end - start))


class ClientThreadBenign(Thread):
    def __init__(self, user, hlpr, global_model):
        super().__init__()
        self.user = user
        self.hlpr = hlpr
        self.global_model = global_model
        self.local_model = hlpr.task.build_model().to(hlpr.task.params.device)
        self._return = None

    def run(self):
        # print('This is Client {}'.format(self.user.user_id))
        self.hlpr.task.copy_params(self.global_model, self.local_model)
        optimizer = self.hlpr.task.make_optimizer(self.local_model)
        for local_epoch in range(self.hlpr.params.fl_local_epochs):
            train(self.hlpr, local_epoch, self.local_model, optimizer,
                    self.user.train_loader, attack=False, report=False)
            # print('Client {} Epoch {}'.format(self.user.user_id, local_epoch))
        local_update = self.hlpr.task.get_fl_update(self.local_model, self.global_model)
        self._return = {self.user.user_id: local_update}

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class ClientThreadMalicious(Thread):
    def __init__(self, user, hlpr, global_model):
        super().__init__()
        self.user = user
        self.hlpr = hlpr
        self.global_model = global_model
        self.local_model = hlpr.task.build_model().to(hlpr.task.params.device)
        self._return = None

    def run(self):
        # print('This is Client {}'.format(self.user.user_id))
        self.hlpr.task.copy_params(self.global_model, self.local_model)
        optimizer = self.hlpr.task.make_optimizer(self.local_model)
        
        pr_sum_max = 0
        p_local_final, r_final = 0, 0
        r = 0
        local_model_best = self.hlpr.task.build_model().to(self.hlpr.params.device)
        if self.hlpr.params.static:
            r = 1  # do not optimize r in static attack, always set to 1
        while r <= 1:
            for local_epoch in range(self.hlpr.params.fl_local_epochs):
                train(self.hlpr, local_epoch, self.local_model, optimizer,
                        self.user.train_loader, attack=True, ratio=r, report=False)
                # print('Client {} Epoch {}'.format(self.user.user_id, local_epoch))

            p_local = self.hlpr.task.compute_genuine_score(self.local_model, 
                                                        self.user.test_loader, 
                                                        self.hlpr.attack.synthesizer)
            pr_sum = p_local + self.hlpr.params.ours_lbd * r
            if pr_sum > pr_sum_max:
                pr_sum_max = pr_sum
                p_local_final, r_final = p_local, r
                self.hlpr.task.copy_params(self.local_model, local_model_best)
            if r == 1:
                break
            r = min(r + self.hlpr.params.r_interval, 1)
            self.hlpr.task.copy_params(self.global_model, self.local_model)
        self.hlpr.task.copy_params(local_model_best, self.local_model)
        
        key = 'Client {} (Malicious)'.format(self.user.user_id)
        local_update = self.hlpr.task.get_fl_update(self.local_model, self.global_model)
        self.hlpr.attack.fl_scale_update(local_update)

        self._return = {self.user.user_id: local_update}, key, p_local_final, r_final

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True, help='Tensorboard name')
    parser.add_argument('--commit', dest='commit',
                        default=get_current_git_hash())

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

    helper = Helper(params)
    logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)
        else:
            run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
