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

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', 
                        required=True, type=str)

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    return args


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

    print('=> constructing models...')
    model = eval('models.' + config.MODEL + '.get')(config)
    model = model.to(config.DEVICE)

    if config.NETWORK.PRETRAINED_BACKBONE:
        backbone = eval('models.' + config.BACKBONE + '.get')(config)
        print('=> loading weights of the backbone')
        backbone.load_state_dict(torch.load(config.NETWORK.PRETRAINED_BACKBONE))
        backbone = backbone.to(config.DEVICE)
        backbone.eval()   # freeze the backbone
    else:
        backbone = None

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load model state {}'.format(test_model_file))
        model.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('check model file for testing!')

    print("=> validating...")
    model.eval()

    # loading constants of the dataset
    cameras = test_loader.dataset.cameras
    resize_transform = torch.as_tensor(test_loader.dataset.resize_transform, dtype=torch.float, device=config.DEVICE)

    with torch.no_grad():
        all_fused_poses = []
        # 用于统计FPS的变量
        inference_times = []
        
        for i, (inputs, _, meta, input_heatmaps) in enumerate(tqdm(test_loader)):
            # 记录开始时间
            start_time = time.time()
            
            if config.DATASET.TEST_HEATMAP_SRC == 'image':
                inputs = inputs.to(config.DEVICE)
                fused_poses, plane_poses, proposal_centers, input_heatmaps, _ = model(backbone=backbone, views=inputs, 
                                                                                      meta=meta, cameras=cameras, 
                                                                                      resize_transform=resize_transform)
            else:
                input_heatmaps = input_heatmaps.to(config.DEVICE)
                fused_poses, plane_poses, proposal_centers, _, _  = model(backbone=backbone, meta=meta, 
                                                                          input_heatmaps=input_heatmaps, 
                                                                          cameras=cameras, 
                                                                          resize_transform=resize_transform)
            
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
                test_vis_all(config, meta, cameras, resize_transform, inputs, input_heatmaps, fused_poses, plane_poses, proposal_centers, prefix)
        
        all_fused_poses = torch.cat(all_fused_poses, dim=0)
        
        # 计算FPS统计信息
        inference_times = np.array(inference_times)
        mean_inference_time = np.mean(inference_times)
        mean_fps = 1.0 / mean_inference_time
        
        # 忽略第一次迭代的预热时间，重新计算
        if len(inference_times) > 1:
            inference_times_no_first = inference_times[1:]
            mean_inference_time_no_first = np.mean(inference_times_no_first)
            mean_fps_no_first = 1.0 / mean_inference_time_no_first
            logger.info(f'平均推理时间(忽略第一次): {mean_inference_time_no_first:.4f}秒, FPS: {mean_fps_no_first:.2f}')
        
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