# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnns_2d import P2PNet
from models.weight_net import WeightNet
from models.project_individual_inference import ProjectLayerInference


class SoftArgmaxLayer(nn.Module):
    """
    软性坐标回归层：将热图转换为精确坐标值
    
    通过对热图应用softmax并计算期望坐标，实现亚像素精度的坐标回归
    不同于直接取热图最大值位置，这种方法可以得到连续的坐标值，提高精度
    """
    def __init__(self, cfg):
        """
        初始化软性坐标回归层
        
        Args:
            cfg: 配置对象，包含beta参数(控制softmax的温度)
        """
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA  # 控制softmax的"温度"，值越大输出越尖锐

    def forward(self, x, grids):
        """
        前向传播
        
        Args:
            x: 热图特征，形状为[3, batch_size, channel, height*width, 1]，3表示三个正交平面
            grids: 坐标网格，包含每个像素的x,y坐标
            
        Returns:
            x: 回归的精确坐标，形状为[3, batch_size, channel, 2]
            confs: 置信度分数，形状为[batch_size]
        """
        batch_size = x.size(1)
        channel = x.size(2)
        x = x.reshape(3, batch_size, channel, -1, 1)  # 调整形状，方便进行softmax
        x = F.softmax(self.beta * x, dim=3)  # 对空间维度应用softmax，beta控制温度

        # 计算置信度分数：热图中每个关节特征图的最大值的平均值
        confs, _ = torch.max(x, dim=3)  # 找出每个特征图的最大概率值
        confs = torch.mean(confs.squeeze(3), dim=(0, 2))  # 对所有关节和平面取平均，得到每个人的总体置信度

        grids = grids.reshape(3, 1, 1, -1, 2)  # 调整坐标网格的形状，匹配softmax后的热图
        x = torch.mul(x, grids)  # 将概率与坐标相乘
        x = torch.sum(x, dim=3)  # 求和得到期望坐标(soft-argmax)
        return x, confs


class JointLocalizationNetInference(nn.Module):
    """
    关节定位网络：预测每个检测到的人体的关节3D位置
    
    工作流程：
    1. 为每个检测到的人体构建个体特征体素
    2. 将体素投影到三个正交平面(xy, xz, yz)
    3. 使用2D CNN提取关节特征
    4. 通过软性坐标回归得到每个平面上的关节位置
    5. 融合三个平面的预测，得到最终的3D关节位置
    """
    def __init__(self, cfg):
        """
        初始化关节定位网络
        
        Args:
            cfg: 配置对象
        """
        super(JointLocalizationNetInference, self).__init__()
        self.conv_net = P2PNet(cfg.DATASET.NUM_JOINTS, cfg.DATASET.NUM_JOINTS)  # 2D卷积网络，用于处理投影特征
        self.weight_net = WeightNet(cfg)  # 权重网络，用于预测融合权重
        self.project_layer = ProjectLayerInference(cfg)  # 投影层，用于构建个体特征体素
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)  # 软性坐标回归层

    def fuse_pose_preds(self, pose_preds, weights):
        """
        融合三个正交平面上的关节位置预测
        
        使用权重网络预测的权重，将三个平面(xy, xz, yz)上的预测融合为最终的3D位置
        x坐标由xy平面和xz平面的预测加权得到
        y坐标由xy平面和yz平面的预测加权得到
        z坐标由xz平面和yz平面的预测加权得到
        
        Args:
            pose_preds: 三个平面上的位置预测，形状为[3, batch_size, num_joints, 2]
            weights: 融合权重，形状为[batch_size, num_joints, 6]
            
        Returns:
            pred: 融合后的3D位置预测，形状为[batch_size, num_joints, 3]
        """
        # 将权重分解为三部分，每部分对应一个平面
        weights = torch.chunk(weights, 3)
        xy_weight, xz_weight, yz_weight = weights[0], weights[1], weights[2]
        # 获取三个平面的位置预测
        xy_pred, xz_pred, yz_pred = pose_preds[0], pose_preds[1], pose_preds[2]

        # 对权重进行归一化，确保每个坐标轴的权重和为1
        x_weight = torch.cat([xy_weight, xz_weight], dim=2)  # x坐标来自xy平面和xz平面
        y_weight = torch.cat([xy_weight, yz_weight], dim=2)  # y坐标来自xy平面和yz平面
        z_weight = torch.cat([xz_weight, yz_weight], dim=2)  # z坐标来自xz平面和yz平面
        x_weight = x_weight / torch.sum(x_weight, dim=2).unsqueeze(2)  # 归一化x权重
        y_weight = y_weight / torch.sum(y_weight, dim=2).unsqueeze(2)  # 归一化y权重
        z_weight = z_weight / torch.sum(z_weight, dim=2).unsqueeze(2)  # 归一化z权重
        
        # 计算加权融合的3D坐标
        x_pred = x_weight[:, :, :1] * xy_pred[:, :, :1] + x_weight[:, :, 1:] * xz_pred[:, :, :1]  # 融合x坐标
        y_pred = y_weight[:, :, :1] * xy_pred[:, :, 1:] + y_weight[:, :, 1:] * yz_pred[:, :, :1]  # 融合y坐标
        z_pred = z_weight[:, :, :1] * xz_pred[:, :, 1:] + z_weight[:, :, 1:] * yz_pred[:, :, 1:]  # 融合z坐标
        
        # 拼接x,y,z坐标得到最终的3D位置
        pred = torch.cat([x_pred, y_pred, z_pred], dim=2)
        return pred

    def forward(self, heatmaps, proposal_centers, mask, cam_params, sample_grid_fine, resize_transform):
        """
        前向传播函数
        
        Args:
            heatmaps: 多视角输入热图，形状为[batch_size, num_views, num_joints, height, width]
            proposal_centers: 人体中心位置，形状为[batch_size, max_proposals, 7]
            mask: 有效提案的掩码，形状为[batch_size, max_proposals]
            cam_params: 相机参数
            resize_transform: 图像缩放变换矩阵
            
        Returns:
            all_fused_pose_preds: 融合后的3D关节位置，形状为[batch_size, max_proposals, num_joints, 3]
            all_pose_preds: 三个平面上的关节位置，形状为[3, batch_size, max_proposals, num_joints, 2]
        """
        device = heatmaps.device
        batch_size = proposal_centers.shape[0]
        max_proposals = proposal_centers.shape[1]
        num_joints = heatmaps.shape[2]
        
        # 初始化输出张量
        all_fused_pose_preds = torch.zeros((batch_size, max_proposals, num_joints, 3), device=device)  # 融合3D位置
        all_pose_preds = torch.zeros((3, batch_size, max_proposals, num_joints, 2), device=device)  # 三个平面的位置
        
        # 逐个批次处理
        for i in range(batch_size):
            # 如果当前批次没有有效提案，则跳过
            if torch.sum(mask[i]) == 0:
                continue
            
            # 构建个体特定的特征体素：为每个检测到的人体单独构建一个特征体素
            cubes, offset = self.project_layer(heatmaps, i, proposal_centers[i, mask[i]], sample_grid_fine)
            
            # 将体素投影到三个正交平面(xy, xz, yz)并提取关节特征
            # 通过在三个维度上分别做最大池化，得到三个平面的投影
            input = torch.cat([torch.max(cubes, dim=4)[0],  # xy平面：沿z轴最大池化
                               torch.max(cubes, dim=3)[0],  # xz平面：沿y轴最大池化
                               torch.max(cubes, dim=2)[0]])  # yz平面：沿x轴最大池化
            # 使用卷积网络提取关节特征，并分割为三部分，对应三个平面
            joint_features = torch.stack(torch.chunk(self.conv_net(input), 3), dim=0)
            
            # 使用软性坐标回归将特征图转换为精确坐标
            pose_preds, confs = self.soft_argmax_layer(joint_features, center_grid)

            # 添加偏移量，将相对坐标转换为全局坐标
            offset = offset.reshape(-1, 1, 3)  # 调整形状
            pose_preds[0] += offset[:, :, :2]  # xy平面添加x,y偏移
            pose_preds[1] += offset[:, :, ::2]  # xz平面添加x,z偏移
            pose_preds[2] += offset[:, :, 1:]  # yz平面添加y,z偏移

            # 计算融合权重并获得最终预测
            weights = self.weight_net(joint_features)  # 预测融合权重
            fused_pose_preds = self.fuse_pose_preds(pose_preds, weights)  # 融合三个平面的预测

            # 将当前批次的结果保存到输出张量中
            all_fused_pose_preds[i, mask[i]] = fused_pose_preds
            all_pose_preds[:, i, mask[i]] = pose_preds
            proposal_centers[i, mask[i], 4] = confs  # 更新置信度

        return all_fused_pose_preds, all_pose_preds