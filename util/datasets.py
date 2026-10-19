# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from PIL import Image
from torchvision import datasets, transforms

import cv2
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.utils.data as data
import pandas as pd
import numpy as np


class Dataset(data.Dataset):

    def __init__(self, root, img_paths, img_labels, transform=None, get_aux=True, aux=None):
        """Load image paths and labels from gt_file"""
        self.root = root
        self.transform = transform
        self.get_aux = get_aux
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.aux = aux

    def __getitem__(self, idx):
        """Load image.

        Args:
            idx (int): image idx.

        Returns:
            img (tensor): image tensor

        """
        img_path1 = self.img_paths[idx]
        if idx < (len(self.img_paths) - 1):
            img_path2 = self.img_paths[idx + 1]
        img = Image.open(os.path.join(self.root, img_path1)).convert('RGB')
        label = self.img_labels[idx]

        if self.transform:
            img = self.transform(img)

        if self.get_aux:
            return img, label

    # return img, label, self.aux[idx]

    def __len__(self):
        return len(self.img_paths)


class Dataset_flow(Dataset):
    def __init__(self, root, img_paths, img_labels, transform=None, get_aux=True, aux=None):
        """Load image paths and labels from gt_file"""
        self.root = root
        self.transform = transform
        self.get_aux = get_aux
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.aux = aux

    def __getitem__(self, idx):
        """Load image and compute optical flow and strain.

        Args:
            idx (int): image idx.

        Returns:
            img (tensor): concatenated tensor of horizontal flow, vertical flow, and optical strain.
            label (int): corresponding label.
        """
        img_path1 = self.img_paths[idx][1]
        img_path2 = self.img_paths[idx][0]

        # Read images
        img1 = cv2.imread(os.path.join(self.root, img_path1), 0)
        img2 = cv2.imread(os.path.join(self.root, img_path2), 0)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x = flow[..., 0]  # Horizontal flow
        flow_y = flow[..., 1]  # Vertical flow

        # Calculate optical strain (gradient magnitude of flow fields)
        grad_x = cv2.Sobel(flow_x, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(flow_y, cv2.CV_64F, 0, 1, ksize=3)
        strain = np.sqrt(grad_x**2 + grad_y**2)  # Magnitude of the gradient

        # Normalize the channels to range [0, 255]
        flow_x_norm = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
        flow_y_norm = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)
        strain_norm = cv2.normalize(strain, None, 0, 255, cv2.NORM_MINMAX)

        # Stack the flow and strain channels
        combined = np.stack([flow_x_norm, flow_y_norm, strain_norm], axis=-1).astype(np.uint8)

        # Convert to PIL image
        img = Image.fromarray(combined)
        label = self.img_labels[idx]

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        if self.get_aux:
            return img, label

        return img, label

    def __len__(self):
        return len(self.img_paths)



class Dataset_flow_2(data.Dataset):

    def __init__(self, root, img_paths, img_labels, transform=None, get_aux=True, aux=None):
        """Initialize dataset with image paths and labels."""
        self.root = root
        self.transform = transform
        self.get_aux = get_aux
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.aux = aux

    def __getitem__(self, idx):
        """Load image and process based on type.

        Args:
            idx (int): Image index.

        Returns:
            img (tensor): Processed image tensor.
            label (int): Image label.
        """
        img_path1 = self.img_paths[idx][1]
        img_path2 = self.img_paths[idx][0]

        # Load images
        img1 = cv2.imread(os.path.join(self.root, img_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(self.root, img_path2), cv2.IMREAD_GRAYSCALE)
        img3 = cv2.imread(os.path.join(self.root, img_path2), cv2.IMREAD_COLOR)
        PIL_image = img3

        # # Determine if input is already an optical flow image
        # if img3 is not None and img3.shape[2] == 3:  # RGB or potential flow image
        #     # Additional checks to determine if it's RGB or flow
        #     if np.mean(img3[..., 1]) == 255 and np.min(img3[..., 0]) >= 0 and np.max(img3[..., 0]) <= 180:
        #         print(f"Index {idx}: Input image is an optical flow image.")
        #         PIL_image = img3
        #     else:
        #         print(f"Index {idx}: Input image is an RGB image. Computing optical flow.")
        #
        #         # Compute optical flow
        #         hsv = np.zeros_like(img3)
        #         hsv[..., 1] = 255
        #         flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #         mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #         hsv[..., 0] = ang * 180 / np.pi / 2
        #         hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        #         PIL_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # else:
        #     raise ValueError(f"Index {idx}: Invalid image format or failed to load image.")

        img = Image.fromarray(PIL_image)
        label = self.img_labels[idx]

        if self.transform:
            img = self.transform(img)

        if self.get_aux:
            return img, label

        return img, label, self.aux[idx]

    def __len__(self):
        return len(self.img_paths)

#
# def data_split(file_path, subject_out_idx=0):
#     """Split dataset into train set and validation set
#     """
#     # data, subject, clipID, label, apex_frame, apex_frame_path
#     data_sub_column = 'data_sub'
#
#     df = pd.read_csv(file_path)
#     # print(type(df))
#     subject_list = list(df[data_sub_column].unique())
#     # print(subject_list)
#     subject_out = subject_list[subject_out_idx]
#     # print('subject_out', subject_out)
#     df_train = df[df[data_sub_column] != subject_out]
#     df_val = df[df[data_sub_column] == subject_out]
#
#     return df_train, df_val
def data_split(file_path, subject_out_idx=0):
    """Split dataset into train set and validation set"""
    data_sub_column = 'data_sub'

    df = pd.read_csv(file_path)
    if data_sub_column not in df.columns:
        raise ValueError(f"Column '{data_sub_column}' not found in the dataset.")

    subject_list = list(df[data_sub_column].unique())
    print(f"Unique subjects: {subject_list}, Length: {len(subject_list)}")
    print(len(subject_list))
    print(subject_out_idx)

    if subject_out_idx < 0 or subject_out_idx >= len(subject_list):
        raise IndexError(f"subject_out_idx ({subject_out_idx}) is out of range. "
                         f"Valid range: 0 to {len(subject_list) - 1}")

    subject_out = subject_list[subject_out_idx]
    print(f"Selected subject for validation: {subject_out}")

    df_train = df[df[data_sub_column] != subject_out]
    df_val = df[df[data_sub_column] == subject_out]

    return df_train, df_val


def sample_data(df, df_four, random_state=0):
    """Balance classes using alternate frames from training clips only.

    Rows are drawn only from subject/clip pairs already present in ``df`` so
    that LOSO validation subjects cannot enter the training set.
    """
    if df.empty:
        return df.copy()

    required = {'data_sub', 'clip', 'label'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    target_size = int(df['label'].value_counts().max())
    parts = []
    for label, group in df.groupby('label', sort=True):
        group = group.copy()
        needed = target_size - len(group)
        if needed > 0:
            clip_keys = group[['data_sub', 'clip']].drop_duplicates()
            candidates = df_four.merge(clip_keys, on=['data_sub', 'clip'], how='inner')
            candidates = candidates[candidates['label'] == label]
            if candidates.empty:
                candidates = group
            extra = candidates.sample(
                n=needed,
                replace=needed > len(candidates),
                random_state=random_state,
            )
            group = pd.concat([group, extra], ignore_index=True)
        parts.append(group)

    return pd.concat(parts, ignore_index=True)


# def sample_data(df, df_four):
#     df_neg = df[df.label == 0]
#     df_pos = df[df.label == 1]
#     df_sur = df[df.label == 2]
#
#     # 获取各个类别的初始样本数
#     num_sur = len(df_sur)
#     num_pos = len(df_pos)
#     num_neg = len(df_neg)
#
#     # 目标样本数为原始样本数最多的类别的样本数
#     target_size = max(num_sur, num_pos, num_neg)  # 目标样本数是当前最大类别的样本数
#
#     # 计算每个类别需要上采样的倍数
#     num_sur = max(1, int(target_size / num_sur) - 1)
#     num_pos = max(1, int(target_size / num_pos) - 1)
#     num_neg = max(0, int(target_size / num_neg) - 1)
#
#     # 对各类别进行上采样
#     df_neg = upsample_subdata(df_neg, df_four, num_neg)
#     df_pos = upsample_subdata(df_pos, df_four, num_pos)
#     df_sur = upsample_subdata(df_sur, df_four, num_sur)
#
#     # 合并数据
#     df = pd.concat([df_neg, df_pos, df_sur])
#     return df



def get_triple_meta_data(file_path):
    df = pd.read_csv(file_path)
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)
    off_paths = list(df.offset_frame_path)

    paths = [(on, apex, off) for (on, apex, off) in zip(on_paths, apex_paths, off_paths)]
    labels = list(df.label)
    return paths, labels


def get_triple_meta_data_f(df):
    # df = pd.read_csv(file_path)
    # paths = list(df.apex_frame_path)
    on_paths = list(df.onset_frame_path)
    apex_paths = list(df.apex_frame_path)
    off_paths = list(df.offset_frame_path)

    paths = [(on, apex, off) for (on, apex, off) in zip(on_paths, apex_paths, off_paths)]
    labels = list(df.label)
    return paths, labels


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # val transform     # 对imagefolder里的图片做增广处理并返回dataset
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
