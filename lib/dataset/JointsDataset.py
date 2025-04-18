# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random

from utils.transforms import get_affine_transform, affine_transform, get_scale
from utils.cameras import project_pose_cpu

logger = logging.getLogger(__name__)

class JointsDataset(Dataset):
    """
    用于处理3D姿态估计任务的数据集类
    主要功能包括：
    1. 加载和处理多视角图像数据
    2. 生成2D和3D热图
    3. 处理数据增强
    4. 管理相机参数和空间变换
    """
    def __init__(self, cfg, is_train=True, transform=None):
        """
        初始化数据集
        Args:
            cfg: 配置文件对象，包含所有必要的参数
            is_train: 是否为训练模式
            transform: 图像变换函数
        """
        self.cfg = cfg
        self.root_id = cfg.DATASET.ROOT_JOINT_ID  # 根关节ID（通常是骨盆）
        self.max_people = cfg.CAPTURE_SPEC.MAX_PEOPLE  # 最大人数
        self.num_views = cfg.DATASET.CAMERA_NUM  # 相机视角数量
        self.color_rgb = cfg.DATASET.COLOR_RGB  # 是否使用RGB颜色空间
        self.dataset_dir = cfg.DATASET.DATADIR  # 数据集目录
        self.ori_image_size = np.array(cfg.DATASET.ORI_IMAGE_SIZE)  # 原始图像尺寸
        self.image_size = np.array(cfg.DATASET.IMAGE_SIZE)  # 目标图像尺寸
        self.heatmap_size = np.array(cfg.DATASET.HEATMAP_SIZE)  # 热图尺寸

        # 相机标定相关参数
        self.sigma = cfg.NETWORK.SIGMA  # 高斯分布的标准差
    
        # 3D空间参数
        self.space_size = np.array(cfg.CAPTURE_SPEC.SPACE_SIZE)  # 3D空间大小
        self.space_center = np.array(cfg.CAPTURE_SPEC.SPACE_CENTER)  # 3D空间中心点
        self.voxels_per_axis = np.array(cfg.CAPTURE_SPEC.VOXELS_PER_AXIS)  # 每个轴上的体素数量
        self.individual_space_size = np.array(cfg.INDIVIDUAL_SPEC.SPACE_SIZE)  # 个体空间大小

        # 根据训练/测试模式选择热图源
        if is_train:
            self.input_heatmap_src = cfg.DATASET.TRAIN_HEATMAP_SRC
        else:
            self.input_heatmap_src = cfg.DATASET.TEST_HEATMAP_SRC
        
        self.data_augmentation = cfg.DATASET.DATA_AUGMENTATION  # 是否进行数据增强
        self.transform = transform  # 图像变换函数
        self.resize_transform = self._get_resize_transform()  # 获取图像缩放变换矩阵
        self.cameras = None  # 相机参数
        self.db = []  # 数据集存储列表
    
    def _get_resize_transform(self):
        """
        计算图像缩放变换矩阵
        Returns:
            trans: 仿射变换矩阵
        """
        r = 0  # 旋转角度
        c = np.array([self.ori_image_size[0] / 2.0, self.ori_image_size[1] / 2.0])  # 中心点
        s = get_scale((self.ori_image_size[0], self.ori_image_size[1]), self.image_size)  # 缩放比例
        trans = get_affine_transform(c, s, r, self.image_size)  # 计算仿射变换矩阵
        return trans

    def _rebuild_db(self):
        """
        重建数据集数据库
        处理每个数据样本，包括：
        1. 3D关节位置
        2. 可见性标记
        3. 边界框
        4. 热图生成
        """
        for idx in range(len(self.db)):
            db_rec = self.db[idx]

            # 处理测试集数据（无3D姿态真值）
            if 'joints_3d' not in db_rec:
                meta = {
                    'seq': db_rec['seq'],
                    'all_image_path': db_rec['all_image_path'],
                }
                target = np.zeros((1, 1, 1), dtype=np.float32)
                target = torch.from_numpy(target)
                self.db[idx] = {
                    'target': target,
                    'meta': meta
                }
                continue

            # 处理训练集数据
            joints_3d = db_rec['joints_3d']  # 3D关节位置
            joints_3d_vis = db_rec['joints_3d_vis']  # 3D关节可见性
            nposes = len(joints_3d)
            assert nposes <= self.max_people, 'too many persons'

            # 初始化统一大小的数组
            joints_3d_u = np.zeros((self.max_people, self.num_joints, 3))
            joints_3d_vis_u = np.zeros((self.max_people, self.num_joints))
            for i in range(nposes):
                joints_3d_u[i] = joints_3d[i][:, 0:3]
                joints_3d_vis_u[i] = joints_3d_vis[i]

            # 计算根关节位置
            if isinstance(self.root_id, int):
                roots_3d = joints_3d_u[:, self.root_id]
            elif isinstance(self.root_id, list):
                roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
            
            # 生成目标热图
            target = self.generate_target(joints_3d, joints_3d_vis)

            # 构建元数据
            meta = {
                'num_person': nposes,
                'joints_3d': joints_3d_u,
                'joints_3d_vis': joints_3d_vis_u,
                'roots_3d': roots_3d,
                'bbox': target['bbox'],
                'seq': db_rec['seq'],
            }
            if 'all_image_path' in db_rec.keys():
                meta['all_image_path'] = db_rec['all_image_path']
            
            # 更新数据库记录
            self.db[idx] = {
                'target': target,
                'meta': meta
            }
        
            # 如果有预测的2D姿态，也保存下来
            if 'pred_pose2d' in db_rec.keys():
                self.db[idx]['pred_pose2d'] = db_rec['pred_pose2d']

        return

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """
        评估函数（需要子类实现）
        """
        raise NotImplementedError

    def __len__(self,):
        """
        返回数据集大小
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        获取单个数据样本
        Args:
            idx: 数据索引
        Returns:
            all_input: 多视角输入图像，维度为(num_views, 3, image_size[1], image_size[0])
            target: 目标热图，包含各种目标信息，包含：
                index: 目标索引，维度为(max_people,)
                offset: 目标偏移，维度为(max_people, 2)
                bbox: 目标边界框，维度为(max_people, 2)
                target_2d: 2D目标热图，维度为(voxels_per_axis[0], voxels_per_axis[1])
                target_1d: 1D目标热图，维度为(max_people, voxels_per_axis[2])
                mask: 目标掩码，维度为(max_people,)
            meta: 元数据，包含：
                num_person: 人数
                joints_3d: 3D关节位置，维度为(max_people, num_joints, 3)
                joints_3d_vis: 3D关节可见性，维度为(max_people, num_joints)
                roots_3d: 根关节位置，维度为(3,)
                bbox: 边界框，维度为(max_people, 2)
                seq: 序列号
            input_heatmaps: 输入热图，维度为(num_joints, heatmap_size[1], heatmap_size[0])
        """
        db_rec = self.db[idx]

        # 读取输入图像
        if 'all_image_path' in db_rec['meta'].keys():
            all_image_path = db_rec['meta']['all_image_path']
            all_input = []
            for image_path in all_image_path:
                input = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if self.color_rgb:
                    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                if self.transform:
                    input = self.transform(input)
                all_input.append(input)
            all_input = torch.stack(all_input, dim=0)
        else:
            all_input = torch.zeros((1, 1, 1, 1))

        # 生成输入热图
        if self.input_heatmap_src == 'image':
            input_heatmaps = torch.zeros((1, 1, 1))

        elif self.input_heatmap_src == 'pred':
            # 使用预测的2D姿态生成热图
            assert 'pred_pose2d' in db_rec.keys() and db_rec['pred_pose2d'] is not None, 'Dataset must provide pred_pose2d'
            all_preds = db_rec['pred_pose2d']
            input_heatmaps = []
            for preds in all_preds:
                for n in range(len(preds)):
                    for i in range(len(preds[n])):
                        preds[n][i, :2] = affine_transform(preds[n][i, :2], self.resize_transform)
                input_heatmap = torch.from_numpy(self.generate_input_heatmap(preds))
                input_heatmaps.append(input_heatmap)
            input_heatmaps = torch.stack(input_heatmaps, dim=0)
            
        elif self.input_heatmap_src == 'gt':
            # 使用真实2D姿态生成热图
            assert 'joints_3d' in db_rec['meta'], 'Dataset must provide gt joints_3d'
            joints_3d = db_rec['meta']['joints_3d']
            joints_3d_vis = db_rec['meta']['joints_3d_vis']
            seq = db_rec['meta']['seq']
            nposes = len(joints_3d)
            input_heatmaps = []
            
            # 获取2D投影姿态
            for c in range(self.num_views):
                joints_2d = []
                joints_vis = []
                for n in range(nposes):
                    pose = project_pose_cpu(joints_3d[n], self.cameras[seq][c])

                    # 检查关节是否在图像范围内
                    x_check = np.bitwise_and(pose[:, 0] >= 0,
                                                pose[:, 0] <= self.ori_image_size[0] - 1)
                    y_check = np.bitwise_and(pose[:, 1] >= 0,
                                                pose[:, 1] <= self.ori_image_size[1] - 1)
                    check = np.bitwise_and(x_check, y_check)
                    vis = joints_3d_vis[n] > 0
                    vis[np.logical_not(check)] = 0
                    
                    # 应用仿射变换
                    for i in range(len(pose)):
                        pose[i] = affine_transform(pose[i], self.resize_transform)
                        if (np.min(pose[i]) < 0 or pose[i, 0] >= self.image_size[0]
                            or pose[i, 1] >= self.image_size[1]):
                                vis[i] = 0
                    
                    joints_2d.append(pose)
                    joints_vis.append(vis)

                input_heatmap = self.generate_input_heatmap(joints_2d, joints_vis=joints_vis)
                input_heatmap = torch.from_numpy(input_heatmap)
                input_heatmaps.append(input_heatmap)
            input_heatmaps = torch.stack(input_heatmaps, dim=0)

        target = db_rec["target"]
        meta = db_rec["meta"]
        return all_input, target, meta, input_heatmaps

    def compute_human_scale(self, pose, joints_vis):
        """
        计算人体尺度
        Args:
            pose: 2D姿态
            joints_vis: 关节可见性
        Returns:
            人体尺度值
        """
        idx = (joints_vis > 0.1)
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        return np.clip(np.maximum(maxy - miny, maxx - minx)**2,  1.0 / 4 * 96**2, 4 * 96**2)

    def generate_target(self, joints_3d, joints_3d_vis):
        """
        生成目标热图
        Args:
            joints_3d: 3D关节位置，维度为(max_people, num_joints, 3)
            joints_3d_vis: 3D关节可见性，维度为(max_people, num_joints)
        Returns:
            包含各种目标信息的字典，
            index: 目标索引，维度为(max_people,)
            offset: 目标偏移，维度为(max_people, 2)
            bbox: 目标边界框，维度为(max_people, 2)
            target_2d: 2D目标热图，维度为(voxels_per_axis[0], voxels_per_axis[1])
            target_1d: 1D目标热图，维度为(max_people, voxels_per_axis[2])
            mask: 目标掩码，维度为(max_people,)
        """
        num_people = len(joints_3d)
        space_size = np.array(self.space_size)
        space_center = np.array(self.space_center)
        individual_space_size = np.array(self.individual_space_size)
        voxels_per_axis = np.array(self.voxels_per_axis)
        voxel_size = space_size / (voxels_per_axis - 1)

        # 创建3D空间网格
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, voxels_per_axis[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, voxels_per_axis[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, voxels_per_axis[2]) + space_center[2]

        # 初始化目标数组
        target_index = np.zeros((self.max_people))
        target_2d = np.zeros((voxels_per_axis[0], voxels_per_axis[1]), dtype=np.float32)
        target_1d = np.zeros((self.max_people, voxels_per_axis[2]), dtype=np.float32)
        target_bbox = np.zeros((self.max_people, 2), dtype=np.float32)
        target_offset = np.zeros((self.max_people, 2), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # 根关节（通常是骨盆）
            idx = (joints_3d_vis[n] > 0.1)
            if isinstance(joint_id, int):
                center_pos = joints_3d[n][joint_id]
            elif isinstance(joint_id, list):
                center_pos = (joints_3d[n][joint_id[0]] + joints_3d[n][joint_id[1]]) / 2.0
            
            # 计算目标索引、偏移和边界框大小
            loc = (center_pos - space_center + 0.5 * space_size) / voxel_size # 计算目标在voxel中的位置
            assert np.sum(loc < 0) == 0 and np.sum(loc > voxels_per_axis) == 0, "human centers out of bound!" 
            target_index[n] = (loc // 1)[0] * voxels_per_axis[1] + (loc // 1)[1] # 计算目标在voxel中的哪一格
            target_offset[n] = (loc % 1)[:2] # 计算目标在格子中的偏移
            target_bbox[n] = ((2 * np.abs(center_pos - joints_3d[n][idx]).max(axis = 0) + 200.0) / individual_space_size)[:2]
            if np.sum(target_bbox[n] > 1) > 0:
                print("Warning: detected an instance where the size of the bounding box is {:.2f}m, larger than 2m".format(np.max(target_bbox[n]) * 2.0))

            # 生成高斯分布
            mu_x, mu_y, mu_z = center_pos[0], center_pos[1], center_pos[2]
            i_x = [np.searchsorted(grid1Dx,  mu_x - 3 * cur_sigma),
                       np.searchsorted(grid1Dx,  mu_x + 3 * cur_sigma, 'right')]
            i_y = [np.searchsorted(grid1Dy,  mu_y - 3 * cur_sigma),
                       np.searchsorted(grid1Dy,  mu_y + 3 * cur_sigma, 'right')]
            i_z = [np.searchsorted(grid1Dz,  mu_z - 3 * cur_sigma),
                       np.searchsorted(grid1Dz,  mu_z + 3 * cur_sigma, 'right')]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            # 生成2D目标热图
            gridx, gridy = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], indexing='ij')
            g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2) / (2 * cur_sigma ** 2))
            target_2d[i_x[0]:i_x[1], i_y[0]:i_y[1]] = np.maximum(target_2d[i_x[0]:i_x[1], i_y[0]:i_y[1]], g)

            # 生成1D目标热图
            gridz = grid1Dz[i_z[0]:i_z[1]]
            g = np.exp(-(gridz - mu_z) ** 2 / (2 * cur_sigma ** 2))
            target_1d[n, i_z[0]:i_z[1]] = np.maximum(target_1d[n, i_z[0]:i_z[1]], g)
            
        # 裁剪热图值到[0,1]范围
        target_2d = np.clip(target_2d, 0, 1)
        target_1d = np.clip(target_1d, 0, 1)
        mask = (np.arange(self.max_people) <= num_people)
        # target_index的维度：(max_people,)
        # target_offset的维度：(max_people, 2)
        # target_bbox的维度：(max_people, 2)
        # target_2d的维度：(voxels_per_axis[0], voxels_per_axis[1])
        # target_1d的维度：(max_people, voxels_per_axis[2])
        # mask的维度：(max_people,)
        target = {'index': target_index, 'offset': target_offset, 'bbox': target_bbox,
                  '2d_heatmaps': target_2d, '1d_heatmaps': target_1d, 'mask':mask}
        return target

    def generate_input_heatmap(self, joints, joints_vis=None):
        """
        生成输入热图
        Args:
            joints: 2D关节位置
            joints_vis: 关节可见性
        Returns:
            输入热图，维度为(num_joints, heatmap_size[1], heatmap_size[0])
        """
        num_joints = joints[0].shape[0]
        target = np.zeros((num_joints, self.heatmap_size[1],\
                           self.heatmap_size[0]), dtype=np.float32)
        feat_stride = self.image_size / self.heatmap_size

        for n in range(len(joints)):
            # 计算人体尺度
            human_scale = 2 * self.compute_human_scale(
                    joints[n][:, :2] / feat_stride, np.ones(num_joints))
            if human_scale == 0:
                continue

            cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
            tmp_size = cur_sigma * 3
            for joint_id in range(num_joints):
                if joints_vis is not None and joints_vis[n][joint_id] == 0:
                    continue

                # 计算热图中心点
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1]\
                        or br[0] < 0 or br[1] < 0:
                    continue

                # 生成高斯分布
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2

                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                # 数据增强
                if self.data_augmentation:
                    # 随机缩放
                    scale = 0.9 + np.random.randn(1) * 0.03 if random.random() < 0.6 else 1.0
                    if joint_id in [7, 8]:  # 手腕
                        scale = scale * 0.5 if random.random() < 0.1 else scale
                    elif joint_id in [9, 10]:  # 脚踝
                        scale = scale * 0.2 if random.random() < 0.1 else scale
                    else:
                        scale = scale * 0.5 if random.random() < 0.05 else scale
                    g *= scale

                    # 随机遮挡
                    start = [int(np.random.uniform(0, self.heatmap_size[1] -1)),
                                int(np.random.uniform(0, self.heatmap_size[0] -1))]
                    end = [int(min(start[0] + np.random.uniform(self.heatmap_size[1] / 4, 
                            self.heatmap_size[1] * 0.75), self.heatmap_size[1])),
                            int(min(start[1] + np.random.uniform(self.heatmap_size[0] / 4,
                            self.heatmap_size[0] * 0.75), self.heatmap_size[0]))]
                    g[start[0]:end[0], start[1]:end[1]] = 0.0

                # 将高斯分布应用到热图上
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                target[joint_id][img_y[0]:img_y[1],
                                    img_x[0]:img_x[1]] = np.maximum(
                                        target[joint_id][img_y[0]:img_y[1],
                                                        img_x[0]:img_x[1]],
                                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
            target = np.clip(target, 0, 1)

        return target