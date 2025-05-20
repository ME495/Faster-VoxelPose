import torch
import argparse
import os

import _init_paths
from core.config import config, update_config
from utils.utils import create_logger
import models

def main():
    parser = argparse.ArgumentParser(description='Export Faster-VoxelPose model to TorchScript')
    parser.add_argument('--cfg', help='experiment configure file name',
                        required=True, type=str)
    args, _ = parser.parse_known_args()
    update_config(args.cfg)
    
    logger, final_output_dir, _ = create_logger(config, args.cfg, 'export')

    # 1. 加载和处理骨干网络
    print('=> loading and scripting the backbone...')
    backbone = eval('models.' + config.BACKBONE + '.get')(config)
    if config.NETWORK.PRETRAINED_BACKBONE and os.path.isfile(config.NETWORK.PRETRAINED_BACKBONE):
        print(f'=> loading backbone weights from {config.NETWORK.PRETRAINED_BACKBONE}')
        backbone.load_state_dict(torch.load(config.NETWORK.PRETRAINED_BACKBONE))
    else:
        print('=> WARNING: No pretrained backbone found or specified. Using randomly initialized backbone.')
    backbone.eval()
    scripted_backbone = torch.jit.script(backbone)
    backbone_output_path = os.path.join(final_output_dir, "scripted_backbone.pt")
    scripted_backbone.save(backbone_output_path)
    print(f'=> Scripted backbone saved to {backbone_output_path}')

    # 2. 创建并加载主模型
    print('=> creating FasterVoxelPoseNetTS model...')
    model = models.faster_voxelpose_ts.FasterVoxelPoseNetTS(config)

    model_weights_path = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(model_weights_path):
        print(f'=> loading model weights from {model_weights_path}')
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')), strict=False)
    else:
        print(f"=> WARNING: No model file specified or found at {model_weights_path}. Using randomly initialized model.")

    model.eval()
    model = model.to(torch.device('cpu')) # 导出时通常使用CPU，除非特定需要

    # 3. 转换主模型为TorchScript
    print('=> scripting the main model...')
    try:
        # 使用 torch.jit.script 转换模型
        scripted_model = torch.jit.script(model)
        model_output_path = os.path.join(final_output_dir, "scripted_faster_voxelpose_ts.pt")
        scripted_model.save(model_output_path)
        print(f'=> Scripted model saved to {model_output_path}')
    except Exception as e:
        print(f"Error during model scripting: {e}")
        print("Please check the model definition for TorchScript compatibility.")

if __name__ == '__main__':
    main()
