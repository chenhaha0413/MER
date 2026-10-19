# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.9.7" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_transform
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from sklearn.utils import shuffle
import itertools
import pandas as pd
import models_vit
import models_mae
from util.datasets import data_split, sample_data, get_triple_meta_data_f, Dataset_flow
from engine_finetune_real import train_one_epoch, evaluate
from sklearn.metrics import recall_score, f1_score


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to val')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='', type=str,
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_csv', required=True, type=str,
                        help='CSV containing data_sub, clip, label, and frame paths')
    parser.add_argument('--data_root', default='.', type=str,
                        help='Root directory used to resolve relative image paths')
    parser.add_argument('--class_names', nargs='+',
                        default=['negative', 'positive', 'surprise'],
                        help='Class names in label order')
    parser.add_argument('--no_oversample', action='store_false', dest='oversample',
                        help='Disable class-balanced training oversampling')
    parser.set_defaults(oversample=True)
    parser.add_argument('--nb_classes', default=3, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./outputs/finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./outputs/finetune',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')      # 微调模型的路径

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args, subject_out_idx):
    misc.init_distributed_mode(args)    # 与分布式训练相关的

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))     # 打印工作目录和args的参数

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset_train = build_dataset(is_train=True, args=args)   # 训练时强数据增强
    # dataset_val = build_dataset(is_train=False, args=args)    # 测试时弱数据增强

    df_train, df_val = data_split(args.data_csv, subject_out_idx)
    df_four = pd.read_csv(args.data_csv)
    df_train_sampled = sample_data(df_train, df_four) if args.oversample else df_train
    df_train_sampled = shuffle(df_train_sampled, random_state=args.seed)
    train_paths, train_labels = get_triple_meta_data_f(df_train_sampled)

    dataset_train = Dataset_flow(root=args.data_root,
                                 img_paths=train_paths,
                                 img_labels=train_labels,
                                 transform=build_transform(is_train=True, args=args))

    val_paths, val_labels = get_triple_meta_data_f(df_val)

    dataset_val = Dataset_flow(root=args.data_root,
                               img_paths=val_paths,
                               img_labels=val_labels,
                               transform=build_transform(is_train=False, args=args))

    if True:  # args.distributed:   # 分布式训练
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        # print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None   # mixup 用于cv领域的数据增强操作
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )


    # if args.finetune and not args.eval:
    #     checkpoint = torch.load(args.finetune, map_location='cpu')
    #     print(f"Loaded checkpoint keys: {checkpoint.keys()}")
    #
    #     if 'model' in checkpoint:
    #         checkpoint_model = checkpoint['model']
    #     elif 'state_dict' in checkpoint:
    #         checkpoint_model = checkpoint['state_dict']
    #     else:
    #         raise KeyError(f"Checkpoint does not contain 'model' or 'state_dict'. Keys: {checkpoint.keys()}")
    #
    #     print("Load pre-trained checkpoint from: %s" % args.finetune)
    #     state_dict = model.state_dict()
    #
    #     for k in ['head.weight', 'head.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        # print(checkpoint.keys())
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        # checkpoint_model = checkpoint['model']
        checkpoint_model = checkpoint.get('model', checkpoint)  # 如果没有'model'键，则直接取整个checkpoint
        # print(checkpoint_model.keys())
        state_dict = model.state_dict()
        # print(state_dict)
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]             # 如果分类头的分类数不等于预训练模型的分类数，就去掉分类头。

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        # print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))   # 打印出模型和模型需要的参数

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()  # 参数decay 和优化器及损失函数  loss——scaler 等价于 求梯度回传 并且更新参数

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    # start_time = time.time()

    max_accuracy = 0.0
    best_val_uar = 0.0
    best_var_uf1 = 0.0
    best_true_y = np.array([], dtype=int)
    best_pred_y = np.array([], dtype=int)
    best_uf1_score = float('-inf')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        test_stats, uf1, uar, true_y, pred_y = evaluate(data_loader_val, model,
                                              device)  # 每一轮之后紧跟测试状态 acc/uf1/uar  和用于计算UF1/UAR的标签值和预测值

        if uf1 >= best_var_uf1:  # 找出30个epoch中最好的uf1和uar
            best_val_uar = uar
            best_var_uf1 = uf1
            best_true_y = true_y
            best_pred_y = pred_y

            if best_var_uf1 >= best_uf1_score:
                best_uf1_score = best_var_uf1
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

                if log_writer is not None:  # 得到指标曲线
                    log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                    # log_writer.add_scalar('perf/test_acc15', test_stats['acc5'], epoch)
                    log_writer.add_scalars('perf/test_uf1_uar', {'uf1': uf1, 'uar': uar},epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'uf1': best_var_uf1,
                             'uar': best_val_uar,
                             'subject_idx': subject_out_idx,
                             'epoch': epoch,
                             'n_parameters': n_parameters}

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")

    print(f'Max uf1: %.4f' % (best_var_uf1))   # 找到30个epoch中最大的那一个
    print(f'Max uar: %.4f' % (best_val_uar))
    # print("**************每个受试者最好的epoch",best_true_y)
    # print("**************每个受试者最好的epoch",best_pred_y)
    return best_var_uf1, best_val_uar, best_true_y, best_pred_y   # 找到30个epoch中最大的那一个用于后续的UF1和UAR计算

def plot_confusion_matrix(true, pred, classes_name, output_dir, title='Confusion Matrix',
                          cmap=plt.cm.Blues, normalize=True):
    # classes = list(set(true))
    # 排序，准确对上分类结果
    # classes.sort()
    # 对比，得到混淆矩阵
    confusion = confusion_matrix(true, pred)
    print(confusion)

    if normalize:
        confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        print('normalized confusion matrix')
        print(confusion)
    else:
        print('confusion matrix without normalization')
    plt.figure()
    plt.title(title)
    # plt.figure(figsize=(5, 4), dpi=120)
    indices = np.arange(len(classes_name))
    plt.xticks(indices, classes_name, rotation=45)
    # plt.xticks(indices, classes_name, rotation=45)
    plt.yticks(indices, classes_name, rotation=45)
    plt.rc('font', size='13')
    plt.imshow(confusion, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    # indices = range(len(confusion)), fontsize=15


    thresh = confusion.max() / 1.5 if normalize else confusion.max() / 2
    for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
        if normalize:
            plt.text(j, i, '{:0.2f}'.format(confusion[i, j]),
                     horizontalalignment="center", va='center', fontsize=14,
                     color="white" if confusion[i, j] > thresh else "black")
        else:
            plt.text(j, i, confusion[i, j],
                     horizontalalignment="center", va='center', fontsize=14,
                     color="white" if confusion[i, j] > thresh else "black")

    plt.tight_layout()  # 会自动调整子图参数
    plt.xlabel('Prediction labels')
    plt.ylabel('True labels')

    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import copy

    args = get_args_parser().parse_args()
    if len(args.class_names) != args.nb_classes:
        raise ValueError('--class_names must contain exactly --nb_classes entries')

    metadata = pd.read_csv(args.data_csv)
    if 'data_sub' not in metadata.columns:
        raise ValueError("The dataset CSV must contain a 'data_sub' column")

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    subject_count = metadata['data_sub'].nunique()

    all_true = []
    all_pred = []
    fold_metrics = []

    for subject_idx in range(subject_count):
        fold_args = copy.deepcopy(args)
        fold_args.output_dir = str(base_output_dir / f'subject_{subject_idx:03d}')
        fold_args.log_dir = str(Path(fold_args.output_dir) / 'tensorboard')
        Path(fold_args.output_dir).mkdir(parents=True, exist_ok=True)

        uf1, uar, true_y, pred_y = main(fold_args, subject_out_idx=subject_idx)
        all_true.extend(np.asarray(true_y, dtype=int).tolist())
        all_pred.extend(np.asarray(pred_y, dtype=int).tolist())
        fold_metrics.append({'subject_idx': subject_idx, 'uf1': float(uf1), 'uar': float(uar)})

    all_true = np.asarray(all_true, dtype=int)
    all_pred = np.asarray(all_pred, dtype=int)
    labels = list(range(args.nb_classes))
    class_f1 = f1_score(all_true, all_pred, labels=labels, average=None, zero_division=0)
    class_recall = recall_score(all_true, all_pred, labels=labels, average=None, zero_division=0)

    plot_confusion_matrix(
        all_true,
        all_pred,
        classes_name=args.class_names,
        output_dir=str(base_output_dir),
    )

    summary = {
        'num_subjects': int(subject_count),
        'uf1': float(class_f1.mean()),
        'uar': float(class_recall.mean()),
        'class_f1': class_f1.tolist(),
        'class_recall': class_recall.tolist(),
        'folds': fold_metrics,
    }
    with open(base_output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
