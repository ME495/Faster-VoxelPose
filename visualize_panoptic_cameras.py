#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
import json_tricks as json
import pickle
import argparse

# Panoptic 数据集的相机列表
CAM_LIST = [(0, 3), (0, 6), (0, 12), (0, 13), (0, 23)]

def visualize_cameras(dataset_dir, sequence, num_views=5):
    """
    可视化 Panoptic 数据集的世界坐标系和相机位置
    
    Args:
        dataset_dir: 数据集根目录
        sequence: 序列名称，如 '160906_pizza1'
        num_views: 要可视化的相机数量（最多5个）
    """
    # 坐标转换矩阵
    M = np.array([[1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0],
                 [0.0, 1.0, 0.0]])
    
    # 加载相机参数
    cam_file = osp.join(dataset_dir, sequence, "calibration_{:s}.json".format(sequence))
    with open(cam_file, "r") as f:
        calib = json.load(f)
    
    # 创建3D图形
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制世界坐标系
    origin = np.zeros(3)
    ax.quiver(origin[0], origin[1], origin[2], 1, 0, 0, color='r', length=500, arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2], 0, 1, 0, color='g', length=500, arrow_length_ratio=0.1)
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, color='b', length=500, arrow_length_ratio=0.1)
    
    # 提取并可视化相机位置
    cam_list = CAM_LIST[:num_views]
    camera_positions = []
    camera_orientations = []
    
    for cam_idx, (panel, node) in enumerate(cam_list):
        # 找到对应的相机
        for cam in calib["cameras"]:
            if cam['panel'] == panel and cam['node'] == node:
                # 提取相机参数
                R = np.array(cam['R']).dot(M) # 世界坐标系到相机坐标系的旋转矩阵
                t = np.array(cam['t']).reshape((3, 1))*10.0 # 从厘米转换为毫米
                
                # 计算相机世界坐标位置 (相机中心在世界坐标系中的位置)
                # C = -R^T * t
                C = -np.dot(R.T, t).flatten()
                
                # 计算相机朝向 (z轴)
                z_axis = R[2, :] * 300  # 放大以便可视化
                
                camera_positions.append(C)
                camera_orientations.append(z_axis)
                
                # 将相机位置存储为变量
                print(f"相机 {panel}_{node} 位置: {C * 0.001} 米")
                
                # 绘制相机位置
                ax.scatter(C[0], C[1], C[2], color='black', s=100, label=f'Camera {panel}_{node}')
                
                # 绘制相机朝向
                ax.quiver(C[0], C[1], C[2], z_axis[0], z_axis[1], z_axis[2], color='purple', length=1.0)
                
                break
    
    # 设置图形属性
    ax.set_xlabel('X 轴', fontsize=14)
    ax.set_ylabel('Y 轴', fontsize=14)
    ax.set_zlabel('Z 轴', fontsize=14)
    ax.set_title(f'Panoptic 数据集 {sequence} 相机位置可视化', fontsize=16)
    
    # 设置坐标轴范围，确保所有相机都可见
    camera_positions = np.array(camera_positions)
    min_pos = np.min(camera_positions, axis=0) - 1000
    max_pos = np.max(camera_positions, axis=0) + 1000
    mid_pos = (np.min(camera_positions, axis=0) + np.max(camera_positions, axis=0)) / 2
    print(mid_pos)
    
    ax.set_xlim(min_pos[0], max_pos[0])
    ax.set_ylim(min_pos[1], max_pos[1])
    # ax.set_zlim(min_pos[2], max_pos[2])
    ax.set_zlim(0, max_pos[2])
    
    # 添加图例
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{sequence}_cameras.png', dpi=300)
    plt.show()
    
    print(f"图像已保存为 {sequence}_cameras.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化 Panoptic 数据集的相机配置')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='数据集根目录路径')
    parser.add_argument('--sequence', type=str, default='160906_pizza1',
                        help='要可视化的序列名称')
    parser.add_argument('--num_views', type=int, default=5, 
                        help='要可视化的相机数量 (最多5个)')
    
    args = parser.parse_args()
    
    visualize_cameras(args.dataset_dir, args.sequence, args.num_views) 