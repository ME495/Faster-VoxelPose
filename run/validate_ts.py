# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
from tqdm import tqdm
import time
import numpy as np

import _init_paths
from core.config import config, update_config
from utils.utils import create_logger
from utils.vis import test_vis_all
from utils.cameras import project_pose
from utils.transforms import affine_transform_pts_cuda as do_transform

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', 
                        required=True, type=str)

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args


def compute_grid(boxSize, boxCenter, nBins, device):
    """
    计算3D体素空间中的网格点坐标
    
    Args:
        boxSize: 3D空间的大小
        boxCenter: 3D空间的中心点
        nBins: 每个轴上的体素数量
        device: 计算设备
        
    Returns:
        grid: 形状为(N, 3)的张量，表示所有体素中心点的3D坐标
    """
    if isinstance(boxSize, int) or isinstance(boxSize, float):
        boxSize = [boxSize, boxSize, boxSize]
    if isinstance(nBins, int):
        nBins = [nBins, nBins, nBins]

    # 创建x, y, z轴上的一维网格
    grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
    grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
    grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
    
    # 使用meshgrid创建3D网格
    gridx, gridy, gridz = torch.meshgrid(
        grid1Dx + boxCenter[0],
        grid1Dy + boxCenter[1],
        grid1Dz + boxCenter[2],
        indexing='ij'
    )
    
    # 将三个坐标轴的网格点整合为一个网格
    gridx = gridx.contiguous().view(-1, 1)
    gridy = gridy.contiguous().view(-1, 1)
    gridz = gridz.contiguous().view(-1, 1)
    grid = torch.cat([gridx, gridy, gridz], dim=1)
    return grid
    
    
def project_grid(grid, ori_image_size, image_size, camera, w, h, nbins, resize_transform, device):
    """
    将3D网格点投影到2D图像平面上
    
    Args:
        camera: 相机参数
        w: 热图宽度
        h: 热图高度
        nbins: 体素总数
        resize_transform: 图像缩放变换矩阵
        device: 计算设备
        
    Returns:
        sample_grid: 用于网格采样的坐标网格，范围在[-1.1, 1.1]之间
    """
    # 使用相机参数将3D点投影到2D平面
    xy = project_pose(grid, camera)
    
    # 裁剪坐标范围
    xy = torch.clamp(xy, -1.0, max(ori_image_size[0], ori_image_size[1]))
    
    # 应用仿射变换（调整图像尺寸）
    xy = do_transform(xy, resize_transform)
    
    # 将坐标缩放到热图尺寸
    xy = xy * torch.tensor(
        [w, h], dtype=torch.float, device=device) / torch.tensor(
        image_size, dtype=torch.float, device=device)
        
    # 将坐标归一化到[-1, 1]范围，用于grid_sample函数
    sample_grid = xy / torch.tensor(
        [w - 1, h - 1], dtype=torch.float,
        device=device) * 2.0 - 1.0
        
    # 调整尺寸并裁剪坐标范围
    sample_grid = torch.clamp(sample_grid.view(1, 1, nbins, 2), -1.1, 1.1)
    return sample_grid


def main():
    args = parse_args()
    logger, final_output_dir, _ = create_logger(config, args.cfg, 'validate')

    print('=> loading data...')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    model = models.faster_voxelpose_ts.FasterVoxelPoseNetTS(config)

    if config.NETWORK.PRETRAINED_BACKBONE:
        backbone = eval('models.' + config.BACKBONE + '.get')(config)
        print('=> loading weights of the backbone')
        backbone.load_state_dict(torch.load(config.NETWORK.PRETRAINED_BACKBONE))
        backbone.eval()   # freeze the backbone
        scripted_backbone = torch.jit.script(backbone)
        scripted_backbone = scripted_backbone.to(config.DEVICE)
    else:
        raise ValueError('pretrained backbone must be specified!')

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load model state {}'.format(test_model_file))
        model.load_state_dict(torch.load(test_model_file), strict=False)
    else:
        raise ValueError('check model file for testing!')

    print("=> validating...")
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model = scripted_model.to(config.DEVICE)

    # loading constants of the dataset
    cameras = test_loader.dataset.cameras
    resize_transform = torch.as_tensor(test_loader.dataset.resize_transform, dtype=torch.float, device=config.DEVICE)
    
    with torch.no_grad():
        all_fused_poses = []
        # 用于统计FPS的变量
        inference_times = []
        grid = compute_grid(config.CAPTURE_SPEC.SPACE_SIZE, config.CAPTURE_SPEC.SPACE_CENTER, config.CAPTURE_SPEC.VOXELS_PER_AXIS, config.DEVICE)
        nbins = config.CAPTURE_SPEC.VOXELS_PER_AXIS[0] * config.CAPTURE_SPEC.VOXELS_PER_AXIS[1] * config.CAPTURE_SPEC.VOXELS_PER_AXIS[2]
        sample_grids_dict = {}
        for seq in cameras.keys():
            sample_grids = torch.zeros(config.DATASET.CAMERA_NUM, 1, nbins, 2, device=config.DEVICE)
            for c in range(config.DATASET.CAMERA_NUM):
                sample_grids[c] = project_grid(grid, 
                                            config.DATASET.ORI_IMAGE_SIZE, 
                                            config.DATASET.IMAGE_SIZE, 
                                            cameras[seq][c], 
                                            config.DATASET.HEATMAP_SIZE[0], config.DATASET.HEATMAP_SIZE[1], 
                                            nbins, resize_transform, config.DEVICE)
            sample_grids_dict[seq] = sample_grids.view(config.DATASET.CAMERA_NUM, config.CAPTURE_SPEC.VOXELS_PER_AXIS[0], 
                                                       config.CAPTURE_SPEC.VOXELS_PER_AXIS[1], config.CAPTURE_SPEC.VOXELS_PER_AXIS[2], 2)
        world_space_size = torch.tensor(config.CAPTURE_SPEC.SPACE_SIZE, device=config.DEVICE)
        world_space_center = torch.tensor(config.CAPTURE_SPEC.SPACE_CENTER, device=config.DEVICE)
        ind_space_size = torch.tensor(config.INDIVIDUAL_SPEC.SPACE_SIZE, device=config.DEVICE)
        ind_voxels_per_axis = torch.tensor(config.INDIVIDUAL_SPEC.VOXELS_PER_AXIS, device=config.DEVICE)
        fine_voxels_per_axis = (world_space_size / ind_space_size * (ind_voxels_per_axis - 1)).int() + 1
        fine_grid = compute_grid(world_space_size, world_space_center, fine_voxels_per_axis, device=config.DEVICE)
        fine_nbins = fine_voxels_per_axis[0] * fine_voxels_per_axis[1] * fine_voxels_per_axis[2]
        fine_sample_grids_dict = {}
        for seq in cameras.keys():
            fine_sample_grids = torch.zeros(config.DATASET.CAMERA_NUM, 1, fine_nbins, 2, device=config.DEVICE)  
            for c in range(config.DATASET.CAMERA_NUM):
                fine_sample_grids[c] = project_grid(fine_grid, 
                                                       config.DATASET.ORI_IMAGE_SIZE, 
                                                       config.DATASET.IMAGE_SIZE, 
                                                       cameras[seq][c], 
                                                       config.DATASET.HEATMAP_SIZE[0], config.DATASET.HEATMAP_SIZE[1], 
                                                       fine_nbins, resize_transform, config.DEVICE)
            fine_sample_grids_dict[seq] = fine_sample_grids.view(config.DATASET.CAMERA_NUM, fine_voxels_per_axis[0], 
                                                                 fine_voxels_per_axis[1], fine_voxels_per_axis[2], 2)
        
        for i, (inputs, _, meta, _) in enumerate(tqdm(test_loader)):
            # 记录开始时间
            start_time = time.time()
            
            if config.DATASET.TEST_HEATMAP_SRC == 'image':
                inputs = inputs.to(config.DEVICE)
                
                with torch.amp.autocast('cuda'):
                    input_heatmaps = scripted_backbone(inputs.view(-1, 3, inputs.shape[3], inputs.shape[4]))
                    input_heatmaps = input_heatmaps.view(-1, config.DATASET.CAMERA_NUM, input_heatmaps.shape[1], input_heatmaps.shape[2], input_heatmaps.shape[3])
                    sample_grids = [sample_grids_dict[meta['seq'][j]] for j in range(input_heatmaps.shape[0])]
                    sample_grids = torch.cat(sample_grids, dim=0)
                    fine_sample_grids = [fine_sample_grids_dict[meta['seq'][j]] for j in range(input_heatmaps.shape[0])]
                    fine_sample_grids = torch.cat(fine_sample_grids, dim=0)
                    fused_poses = scripted_model(input_heatmaps, sample_grids, fine_sample_grids)
            else:
                raise ValueError('test heatmap source must be image!')
            
            # 确保GPU操作完成
            torch.cuda.synchronize()
            
            # 记录结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            all_fused_poses.append(fused_poses)

            # visualization
            if config.TEST.VISUALIZATION:
                prefix = '{}_{:08}'.format(os.path.join(final_output_dir, 'validation'), i)
                test_vis_all(config, meta, cameras, resize_transform, inputs, input_heatmaps, fused_poses, None, None, prefix)
        
        all_fused_poses = torch.cat(all_fused_poses, dim=0)
        
        # 计算FPS统计信息
        inference_times = np.array(inference_times)
        mean_inference_time = np.mean(inference_times)
        mean_fps = 1.0 / mean_inference_time
        
        # 忽略前10次迭代的预热时间，重新计算
        if len(inference_times) > 10:
            inference_times_no_first = inference_times[10:]
            mean_inference_time_no_first = np.mean(inference_times_no_first)
            mean_fps_no_first = 1.0 / mean_inference_time_no_first
            logger.info(f'平均推理时间(忽略前10次): {mean_inference_time_no_first:.4f}秒, FPS: {mean_fps_no_first:.2f}')
        
        logger.info(f'平均推理时间: {mean_inference_time:.4f}秒, FPS: {mean_fps:.2f}')
        logger.info(f'最快推理时间: {np.min(inference_times):.4f}秒, 最慢: {np.max(inference_times):.4f}秒')
        
        # 按批次大小计算每个人的FPS
        persons_per_batch = config.TEST.BATCH_SIZE * config.DATASET.CAMERA_NUM  # 假设每个视角一个人
        person_fps = mean_fps * persons_per_batch
        logger.info(f'每批次处理 {persons_per_batch} 个人体姿态, 每人每秒处理速度: {person_fps:.2f}')

    if test_dataset.has_evaluate_function:
        metric, msg = test_loader.dataset.evaluate(all_fused_poses)
        logger.info(msg)

if __name__ == "__main__":
    main()