# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import numpy as np
from sklearn.metrics import recall_score, f1_score

from timm.data import Mixup
from timm.utils import accuracy

# import util.misc as misc
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:   # 检查当前迭代步骤是否是积累梯度的起始步骤，如果是起始步骤则调整学习率
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        # print(samples.shape)
        targets = targets.to(device, non_blocking=True)
        # print(targets.shape)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():  # 这是一个自动混合精度上下文管理器，用于支持半精度训练（Mixed Precision Training），加速训练过程。
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class Meter(object):

    def __init__(self):
        """
        To record the measure the performance.
        """
        self.Y_true = np.array([], dtype=np.int)
        self.Y_pred = np.array([], dtype=np.int)

    def add(self, y_true, y_pred, verbose=False):
        if len(self.Y_true.shape) != len(y_true.shape):
            print('shape self.Y_true', self.Y_true.shape)
            print('y_true', y_true.shape)

        self.Y_true = np.concatenate((self.Y_true, y_true))
        self.Y_pred = np.concatenate((self.Y_pred, y_pred))

    def reset(self):
        self.Y_true = np.array([], dtype=np.int)
        self.Y_pred = np.array([], dtype=np.int)

    def value(self):
        # Changed from 3 to 5 classes
        eye = np.eye(5, dtype=np.int)
        Y_true = eye[self.Y_true]
        Y_pred = eye[self.Y_pred]
        uar = recall_score(Y_true, Y_pred, average=None)
        uf1 = f1_score(Y_true, Y_pred, average=None)
        return uar, uf1


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    Y_true = np.array([], dtype=np.int)
    Y_pred = np.array([], dtype=np.int)

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        # Changed from 3 to 5 classes
        target_one_hot = torch.zeros(target.size(0), 5).scatter_(1, target.view(-1, 1), 1.)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target_one_hot = target_one_hot.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # acc1, _ = accuracy(output, target, topk=(1,))

        y_pred = output.data.max(1)[1]  # 返回最大可能的那一类
        y_true = target_one_hot.data.max(1)[1]

        Y_true = np.concatenate((Y_true,y_true.cpu().numpy()))
        Y_pred = np.concatenate((Y_pred,y_pred.cpu().numpy()))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # metric_logger.meters['uf1'].update(uf1.mean(), n=batch_size)
        # metric_logger.meters['uar'].update(uar.mean(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()  # 没分布式环境用不着

    # Changed from 3 to 5 classes
    eye = np.eye(5, dtype=np.int)
    Y_true_eye = eye[Y_true]
    Y_pred_eye = eye[Y_pred]
    uar = recall_score(Y_true_eye, Y_pred_eye, average=None)
    uf1 = f1_score(Y_true_eye, Y_pred_eye, average=None)
    # print("---------受试者的每个epoch",Y_true)
    # print("---------受试者的每个epoch", Y_pred)

    print('* Acc@1 {top1.global_avg:.3f} UF1 {uf1:.3f} UAR {uar:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, uf1=np.mean(uf1), uar=np.mean(uar), losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, uf1.mean(), uar.mean(), Y_true, Y_pred
