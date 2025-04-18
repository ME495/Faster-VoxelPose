# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

def get_index2D(indices, shape):
    """
    将一维展平索引转换为二维坐标索引
    
    Args:
        indices: 展平后的一维索引, 形状为 [batch_size, num_people]
        shape: 原始2D热图的形状
        
    Returns:
        二维坐标索引, 形状为 [batch_size, num_people, 2]，每个点包含 [x, y] 坐标
    """
    batch_size = indices.shape[0]
    num_people = indices.shape[1]
    # 计算x坐标（行索引）：通过整除得到
    indices_x = torch.div(indices, shape[1], rounding_mode='trunc').reshape(batch_size, num_people, -1)
    # 计算y坐标（列索引）：通过取余得到
    indices_y = (indices % shape[1]).reshape(batch_size, num_people, -1)
    # 拼接x和y坐标
    indices = torch.cat([indices_x, indices_y], dim=2)
    return indices

def max_pool2D(inputs, kernel=3):
    """
    执行最大池化操作，用于非极大值抑制
    
    通过与局部最大值比较，保留局部最大值点，抑制非最大值点
    
    Args:
        inputs: 输入热图，形状为 [batch_size, channels, height, width]
        kernel: 池化核大小，默认为3
        
    Returns:
        经过非极大值抑制后的热图，只保留局部最大值点
    """
    padding = (kernel - 1) // 2  # 计算填充大小，保持输出尺寸不变
    # 执行最大池化，得到每个点的局部最大值
    max = F.max_pool2d(inputs, kernel_size=kernel, stride=1, padding=padding)
    # 比较原始值和局部最大值，只保留等于局部最大值的点（即局部最大值点）
    keep = (inputs == max).float()
    return keep * inputs  # 只保留局部最大值点的原始值

def nms2D(prob_map, max_num):
    """
    对概率热图执行2D非极大值抑制并选取前K个点
    
    Args:
        prob_map: 概率热图，形状为 [batch_size, channels, height, width]
        max_num: 需要选取的最大点数
        
    Returns:
        topk_values: 前K个点的概率值，形状为 [batch_size, max_num]
        topk_index: 前K个点的2D坐标索引，形状为 [batch_size, max_num, 2]
        topk_flatten_index: 前K个点的1D展平索引，形状为 [batch_size, max_num]
    """
    batch_size = prob_map.shape[0]
    # 步骤1: 执行非极大值抑制，只保留局部最大值点
    prob_map_nms = max_pool2D(prob_map)
    # 步骤2: 将热图展平为1D张量
    prob_map_nms_reshape = prob_map_nms.reshape(batch_size, -1)
    # 步骤3: 选取前K个最大值及其索引
    topk_values, topk_flatten_index = prob_map_nms_reshape.topk(max_num)
    # 步骤4: 将1D索引转换回2D坐标
    topk_index = get_index2D(topk_flatten_index, prob_map[0].shape)
    return topk_values, topk_index, topk_flatten_index