import torch
import argparse
import os
import onnxruntime as ort

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
    
    onnx_backbone_path = os.path.join(final_output_dir,"backbone.onnx")
    torch.onnx.export(backbone.half(), torch.randn(4, 3, 784, 1024, dtype=torch.float16), onnx_backbone_path, 
                      input_names=["input"], output_names=["output"], do_constant_folding=True)
    print(f'=> ONNX backbone saved to {onnx_backbone_path}')

    # # ONNX图优化
    # print(f'=> Optimizing ONNX backbone model: {onnx_backbone_path}')
    # optimized_onnx_backbone_path = os.path.join(final_output_dir, "backbone_optimized.onnx")
    # try:
    #     sess_options = ort.SessionOptions()
    #     sess_options.optimized_model_filepath = optimized_onnx_backbone_path
    #     sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # 或者 ORT_ENABLE_EXTENDED
    #     ort.InferenceSession(onnx_backbone_path, sess_options, providers=['CPUExecutionProvider'])
    #     print(f'=> Optimized ONNX backbone saved to {optimized_onnx_backbone_path}')
    #     # 后续在validate_onnx_ts.py中应该加载这个优化后的模型
    # except Exception as e:
    #     print(f"Error during ONNX model optimization: {e}")
    #     print("Please ensure onnxruntime is correctly installed and the model is valid.")

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
