import logging
from typing import Dict

import torch
from copy import deepcopy
import numpy as np
from models.model import Model
from synthesizers.synthesizer import Synthesizer
from losses.loss_functions import compute_all_losses_and_grads
from utils.min_norm_solvers import MGDASolver
from utils.parameters import Params

logger = logging.getLogger('logger')


class Attack:
    params: Params
    synthesizer: Synthesizer
    nc_model: Model
    nc_optim: torch.optim.Optimizer
    loss_hist = list()
    # fixed_model: Model

    def __init__(self, params, synthesizer):
        self.params = params
        self.synthesizer = synthesizer

    def compute_blind_loss(self, model, criterion, batch, attack, ratio=None):
        """

        :param model:
        :param criterion:
        :param batch:
        :param attack: Do not attack at all. Ignore all the parameters
        :return:
        """
        batch = batch.clip(self.params.clip_batch)
        loss_tasks = self.params.loss_tasks.copy() if attack else ['normal']
        batch_back = self.synthesizer.make_backdoor_batch(batch, attack=attack, ratio=ratio)
        scale = dict()

        if self.params.loss_threshold and (np.mean(self.loss_hist) >= self.params.loss_threshold
                                           or len(self.loss_hist) < 1000):
            loss_tasks = ['normal']

        if len(loss_tasks) == 1:
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=False
            )
        elif self.params.loss_balance == 'MGDA':

            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=True)
            if len(loss_tasks) > 1:
                scale = MGDASolver.get_scales(grads, loss_values,
                                              self.params.mgda_normalize,
                                              loss_tasks)
        elif self.params.loss_balance == 'fixed':
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, compute_grad=False)

            for t in loss_tasks:
                scale[t] = self.params.fixed_scales[t]
        else:
            raise ValueError(f'Please choose between `MGDA` and `fixed`.')

        if len(loss_tasks) == 1:
            scale = {loss_tasks[0]: 1.0}
        self.loss_hist.append(loss_values[list(loss_values.keys())[0]].item())
        self.loss_hist = self.loss_hist[-1000:]
        blind_loss = self.scale_losses(loss_tasks, loss_values, scale)

        return blind_loss

    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for it, t in enumerate(loss_tasks):
            self.params.running_losses[t].append(loss_values[t].item())
            self.params.running_scales[t].append(scale[t])
            if it == 0:
                blind_loss = scale[t] * loss_values[t]
            else:
                blind_loss += scale[t] * loss_values[t]
        self.params.running_losses['total'].append(blind_loss.item())
        return blind_loss

    def fl_scale_update(self, local_update: Dict[str, torch.Tensor], scale=None):
        for name, value in local_update.items():
            if scale is None:
                value.mul_(self.params.fl_weight_scale)
            else:
                value.mul_(scale)
