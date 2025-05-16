# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

# 测试时需要添加
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from models.cnns_2d import P2PNet
from models.weight_net_ts import WeightNetTS
from models.project_individual_ts import ProjectLayerTS


class SoftArgmaxLayerTS(nn.Module):
    """
    软性坐标回归层：将热图转换为精确坐标值
    
    通过对热图应用softmax并计算期望坐标，实现亚像素精度的坐标回归
    不同于直接取热图最大值位置，这种方法可以得到连续的坐标值，提高精度
    """
    beta: float
    
    def __init__(self, beta: float):
        """
        初始化软性坐标回归层
        
        Args:
            beta: softmax的温度参数，控制输出的尖锐程度
        """
        super(SoftArgmaxLayerTS, self).__init__()
        self.beta = beta  # 控制softmax的"温度"，值越大输出越尖锐

    def forward(self, x: torch.Tensor, grids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


class JointLocalizationNetTS(nn.Module):
    """
    关节定位网络：预测每个检测到的人体的关节3D位置
    
    工作流程：
    1. 为每个检测到的人体构建个体特征体素
    2. 将体素投影到三个正交平面(xy, xz, yz)
    3. 使用2D CNN提取关节特征
    4. 通过软性坐标回归得到每个平面上的关节位置
    5. 融合三个平面的预测，得到最终的3D关节位置
    """
    num_joints: int
    voxels_per_axis: List[int]
    
    def __init__(self, num_joints: int, voxels_per_axis: List[int], 
                 num_channel_joint_feat: int, num_channel_joint_hidden: int, 
                 image_size: List[int], heatmap_size: List[int], ori_image_size: List[int], 
                 space_center: List[float], space_size: List[float], 
                 individual_space_size: List[float], beta: float):
        """
        初始化关节定位网络
        
        Args:
            num_joints: 关节点数量
            voxels_per_axis: 每个轴上的体素数量 [x, y, z]
            num_channel_joint_feat: 关节特征通道数
            num_channel_joint_hidden: 隐藏层通道数
            image_size: 输入图像尺寸 [width, height]
            heatmap_size: 热图尺寸 [width, height]
            ori_image_size: 原始图像尺寸 [width, height]
            space_center: 全局空间中心 [x, y, z]
            space_size: 全局空间大小 [x, y, z]
            individual_space_size: 个体空间大小 [x, y, z]
            beta: softmax温度参数
        """
        super(JointLocalizationNetTS, self).__init__()
        self.num_joints = num_joints
        self.voxels_per_axis = voxels_per_axis
        
        self.conv_net = P2PNet(num_joints, num_joints)  # 2D卷积网络，用于处理投影特征
        self.weight_net = WeightNetTS(voxels_per_axis, num_joints, num_channel_joint_feat, num_channel_joint_hidden)  # 权重网络，用于预测融合权重
        self.project_layer = ProjectLayerTS(image_size, heatmap_size, ori_image_size, 
                 space_center, space_size, 
                 individual_space_size, voxels_per_axis)  # 投影层，用于构建个体特征体素
        self.soft_argmax_layer = SoftArgmaxLayerTS(beta)  # 软性坐标回归层

    def fuse_pose_preds(self, pose_preds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
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
        weights_chunks = torch.chunk(weights, 3)
        xy_weight, xz_weight, yz_weight = weights_chunks[0], weights_chunks[1], weights_chunks[2]
        
        # 获取三个平面的位置预测
        xy_pred, xz_pred, yz_pred = pose_preds[0], pose_preds[1], pose_preds[2]

        # 对权重进行归一化，确保每个坐标轴的权重和为1
        x_weight = torch.cat([xy_weight, xz_weight], dim=2)  # x坐标来自xy平面和xz平面
        y_weight = torch.cat([xy_weight, yz_weight], dim=2)  # y坐标来自xy平面和yz平面
        z_weight = torch.cat([xz_weight, yz_weight], dim=2)  # z坐标来自xz平面和yz平面
        
        # 避免除零错误
        x_sum = torch.sum(x_weight, dim=2, keepdim=True)
        y_sum = torch.sum(y_weight, dim=2, keepdim=True)
        z_sum = torch.sum(z_weight, dim=2, keepdim=True)
        
        # 使用 where 代替条件判断，确保 TorchScript 兼容性
        x_weight = torch.where(x_sum > 0, x_weight / x_sum, x_weight)
        y_weight = torch.where(y_sum > 0, y_weight / y_sum, y_weight)
        z_weight = torch.where(z_sum > 0, z_weight / z_sum, z_weight)
        
        # 计算加权融合的3D坐标
        x_pred = x_weight[:, :, :1] * xy_pred[:, :, :1] + x_weight[:, :, 1:] * xz_pred[:, :, :1]  # 融合x坐标
        y_pred = y_weight[:, :, :1] * xy_pred[:, :, 1:] + y_weight[:, :, 1:] * yz_pred[:, :, :1]  # 融合y坐标
        z_pred = z_weight[:, :, :1] * xz_pred[:, :, 1:] + z_weight[:, :, 1:] * yz_pred[:, :, 1:]  # 融合z坐标
        
        # 拼接x,y,z坐标得到最终的3D位置
        pred = torch.cat([x_pred, y_pred, z_pred], dim=2)
        return pred

    def forward(self, heatmaps: torch.Tensor, fine_sample_grids: torch.Tensor, 
               proposal_centers: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数
        
        Args:
            heatmaps: 多视角输入热图，形状为[batch_size, num_views, num_joints, height, width]
            fine_sample_grids: 精细采样网格，形状为[batch_size, num_views, fine_voxels_x, fine_voxels_y, fine_voxels_z, 2]
            proposal_centers: 人体中心位置，形状为[batch_size, max_proposals, 7]
            mask: 有效提案的掩码，形状为[batch_size, max_proposals]
            
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
            # 计算当前批次有效提案的数量
            valid_count = torch.sum(mask[i]).item()
            
            # 如果当前批次没有有效提案，则跳过
            if valid_count == 0:
                continue
            
            # 构建个体特定的特征体素：为每个检测到的人体单独构建一个特征体素
            cubes, offset = self.project_layer(heatmaps, fine_sample_grids, i, proposal_centers[i, mask[i]])
            
            # 将体素投影到三个正交平面(xy, xz, yz)并提取关节特征
            # 通过在三个维度上分别做最大池化，得到三个平面的投影
            input = torch.cat([torch.max(cubes, dim=4)[0],  # xy平面：沿z轴最大池化
                               torch.max(cubes, dim=3)[0],  # xz平面：沿y轴最大池化
                               torch.max(cubes, dim=2)[0]])  # yz平面：沿x轴最大池化
            # 使用卷积网络提取关节特征，并分割为三部分，对应三个平面
            joint_features = torch.stack(torch.chunk(self.conv_net(input), 3), dim=0)
            
            # 使用软性坐标回归将特征图转换为精确坐标
            pose_preds, confs = self.soft_argmax_layer(joint_features, self.project_layer.center_grid)

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


# 测试代码
if __name__ == "__main__":
    print("=== 测试 JointLocalizationNetTS 模型 ===")
    # 测试参数
    num_joints = 17
    voxels_per_axis = [32, 32, 32]
    num_channel_joint_feat = 32
    num_channel_joint_hidden = 64
    image_size = [256, 256]
    heatmap_size = [64, 64]
    ori_image_size = [1280, 720]
    space_center = [0.0, 0.0, 1000.0]
    space_size = [2000.0, 2000.0, 2000.0]
    individual_space_size = [1000.0, 1000.0, 1000.0]
    beta = 100.0
    
    # 创建模型实例
    model = JointLocalizationNetTS(
        num_joints=num_joints,
        voxels_per_axis=voxels_per_axis,
        num_channel_joint_feat=num_channel_joint_feat,
        num_channel_joint_hidden=num_channel_joint_hidden,
        image_size=image_size,
        heatmap_size=heatmap_size,
        ori_image_size=ori_image_size,
        space_center=space_center,
        space_size=space_size,
        individual_space_size=individual_space_size,
        beta=beta
    )
    
    print("模型创建成功!")
    
    # 创建测试输入
    batch_size = 2
    num_views = 3
    max_proposals = 5
    h, w = 64, 64
    
    # 模拟热图输入
    heatmaps = torch.rand(batch_size, num_views, num_joints, h, w)
    
    # 模拟采样网格
    fine_voxels_x = model.project_layer.fine_voxels_per_axis[0].item()
    fine_voxels_y = model.project_layer.fine_voxels_per_axis[1].item()
    fine_voxels_z = model.project_layer.fine_voxels_per_axis[2].item()
    fine_sample_grids = torch.rand(batch_size, num_views, fine_voxels_x, fine_voxels_y, fine_voxels_z, 2) * 2 - 1
    
    # 模拟人体中心位置
    proposal_centers = torch.zeros(batch_size, max_proposals, 7)
    for i in range(batch_size):
        valid_count = min(3, max_proposals)  # 每个批次最多3个有效提案
        proposal_centers[i, :valid_count, :3] = torch.tensor([0.0, 0.0, 1000.0]) + torch.randn(valid_count, 3) * 100
        proposal_centers[i, :valid_count, 5:7] = torch.rand(valid_count, 2) * 0.3 + 0.5  # 边界框大小
    
    # 创建掩码，指示哪些提案是有效的
    mask = torch.zeros(batch_size, max_proposals, dtype=torch.bool)
    for i in range(batch_size):
        valid_count = min(3, max_proposals)
        mask[i, :valid_count] = True
    
    print("测试输入准备完成!")
    
    # 测试前向传播
    outputs = model.forward(heatmaps, fine_sample_grids, proposal_centers, mask)
    
    print(f"输出形状:")
    print(f"- fused_pose_preds: {outputs[0].shape}")
    print(f"- pose_preds: {outputs[1].shape}")
    
    # 尝试导出为 TorchScript
    print("\n尝试导出为 TorchScript...")
    
    # 尝试使用 torch.jit.script
    model_path = "joint_localization_net_ts.pt"
    scripted_model = torch.jit.script(model)
    print("模型成功通过 torch.jit.script 导出!")
    
    # 保存模型
    scripted_model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 加载导出的模型
    print("\n加载导出的模型并验证...")
    loaded_model = torch.jit.load(model_path)
    
    # 使用相同的输入测试原始模型和加载的模型
    print("运行原始模型...")
    original_outputs = model.forward(heatmaps, fine_sample_grids, proposal_centers.clone(), mask)
    
    print("运行转换后的模型...")
    converted_outputs = loaded_model.forward(heatmaps, fine_sample_grids, proposal_centers.clone(), mask)
    
    # 验证输出是否一致
    fused_match = torch.allclose(original_outputs[0], converted_outputs[0], rtol=1e-5, atol=1e-5)
    pose_match = torch.allclose(original_outputs[1], converted_outputs[1], rtol=1e-5, atol=1e-5)
    
    print("\n模型输出验证结果:")
    print(f"- fused_pose_preds 一致性: {'通过' if fused_match else '失败'}")
    print(f"- pose_preds 一致性: {'通过' if pose_match else '失败'}")
    
    if fused_match and pose_match:
        print("\n模型已成功导出为 TorchScript 格式并通过验证。")
    else:
        print("\n警告: 转换后的模型输出与原始模型不一致，可能存在兼容性问题。")
            