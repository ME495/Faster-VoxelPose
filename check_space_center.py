import os
import numpy as np
import pickle
import json
import glob
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Check if SPACE_CENTER is the mean of all human centers')
    parser.add_argument('--config', default='configs/panoptic/jln64.yaml', help='config file path')
    parser.add_argument('--datadir', default='data/Panoptic', help='dataset directory')
    args = parser.parse_args()
    return args

def load_config(config_file):
    """加载配置文件"""
    import yaml
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # 获取配置中的SPACE_CENTER
    space_center = cfg['CAPTURE_SPEC']['SPACE_CENTER']
    print(f"配置文件中的SPACE_CENTER: {space_center}")
    
    # 定义训练序列列表 (与Panoptic数据集一致)
    TRAIN_LIST = [
        '160422_ultimatum1',
        '160224_haggling1',
        '160226_haggling1',
        '161202_haggling1',
        '160906_ian1',
        '160906_ian2',
        '160906_ian3',
        '160906_band1',
        '160906_band2'
    ]
    
    VAL_LIST = [
        '160906_pizza1',
        '160422_haggling1',
        '160906_ian5',
        '160906_band4',
    ]
    
    # 合并所有序列
    ALL_SEQUENCES = TRAIN_LIST + VAL_LIST
    
    # 定义根关节ID (通常是骨盆或mid-hip)
    ROOT_JOINT_ID = cfg['DATASET']['ROOT_JOINT_ID']  # 注意，有些数据集可能是列表
    
    # 存储所有人体的中心坐标
    all_centers = []
    
    # 遍历所有序列
    for seq in tqdm(ALL_SEQUENCES, desc="处理序列"):
        # 确定序列数据目录
        curr_anno = os.path.join(args.datadir, seq, 'hdPose3d_stage1_coco19')
        if not os.path.exists(curr_anno):
            print(f"警告: 找不到序列 {seq} 的数据目录")
            continue
            
        # 获取所有标注文件
        anno_files = sorted(glob.glob(f'{curr_anno}/*.json'))
        
        # 设置采样间隔
        interval = 3  # 用于训练集的采样间隔
        
        # 遍历标注文件
        for i, anno_file in enumerate(anno_files):
            if i % interval == 0:  # 按间隔采样
                try:
                    with open(anno_file, "r") as f:
                        bodies = json.load(f)["bodies"]
                    
                    if len(bodies) == 0:
                        continue
                        
                    # 处理每个人体
                    for body in bodies:
                        pose3d = np.array(body['joints19']).reshape((-1, 4))
                        
                        # 根关节为单一ID
                        if isinstance(ROOT_JOINT_ID, int):
                            # 检查可见性
                            if pose3d[ROOT_JOINT_ID, -1] > 0.1:
                                # 坐标变换 (与原代码一致)
                                M = np.array([[1.0, 0.0, 0.0],
                                              [0.0, 0.0, -1.0],
                                              [0.0, 1.0, 0.0]])
                                pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
                                
                                # 获取根关节坐标作为人体中心 (乘以10以匹配原代码处理)
                                center = pose3d[ROOT_JOINT_ID, 0:3] * 10.0
                                all_centers.append(center)
                        
                        # 如果根关节是列表 (如骨盆由两个关节点表示)
                        elif isinstance(ROOT_JOINT_ID, list):
                            # 检查可见性 (至少一个关节点可见)
                            if any(pose3d[j, -1] > 0.1 for j in ROOT_JOINT_ID):
                                # 坐标变换
                                M = np.array([[1.0, 0.0, 0.0],
                                              [0.0, 0.0, -1.0],
                                              [0.0, 1.0, 0.0]])
                                pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)
                                
                                # 计算多个关节点的平均值作为中心
                                visible_joints = [j for j in ROOT_JOINT_ID if pose3d[j, -1] > 0.1]
                                if visible_joints:
                                    center = np.mean([pose3d[j, 0:3] for j in visible_joints], axis=0) * 10.0
                                    all_centers.append(center)
                                
                except Exception as e:
                    print(f"处理文件 {anno_file} 时出错: {e}")
    
    # 计算所有人体中心的均值
    if all_centers:
        all_centers = np.array(all_centers)
        mean_center = np.mean(all_centers, axis=0)
        std_center = np.std(all_centers, axis=0)
        print(f"处理了 {len(all_centers)} 个人体")
        print(f"人体中心坐标均值: {mean_center}")
        print(f"人体中心坐标标准差: {std_center}")
        print(f"坐标分布范围:")
        print(f"X: 最小值 {np.min(all_centers[:, 0]):.2f}, 最大值 {np.max(all_centers[:, 0]):.2f}")
        print(f"Y: 最小值 {np.min(all_centers[:, 1]):.2f}, 最大值 {np.max(all_centers[:, 1]):.2f}")
        print(f"Z: 最小值 {np.min(all_centers[:, 2]):.2f}, 最大值 {np.max(all_centers[:, 2]):.2f}")
        
        # 比较与配置中SPACE_CENTER的差异
        print("\n与配置中SPACE_CENTER的比较:")
        diff = mean_center - np.array(space_center)
        print(f"差异 (均值 - 配置值): {diff}")
        print(f"相对误差 (%): {100 * np.abs(diff / mean_center)}")
        
        # 结论
        threshold = 100  # 100mm误差阈值
        if np.all(np.abs(diff) < threshold):
            print("\n结论: SPACE_CENTER与人体中心均值非常接近，可能是基于数据集均值设置的。")
        else:
            print("\n结论: SPACE_CENTER与人体中心均值存在较大差异，可能不是直接基于数据集均值设置的。")
            print("可能的原因:")
            print("1. SPACE_CENTER可能是手动设置的经验值")
            print("2. 可能基于其他因素如相机位置或捕获空间设置")
            print("3. 可能基于不同的子集或采样方法计算得到")
    else:
        print("未找到有效的人体数据")

if __name__ == "__main__":
    main() 