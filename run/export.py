import torch
import argparse
import os

import _init_paths
from core.config import config, update_config
import models

parser = argparse.ArgumentParser(description='Train keypoints network')
parser.add_argument('--cfg', help='experiment configure file name', 
                        required=True, type=str)
args, _ = parser.parse_known_args()
update_config(args.cfg)

backbone = eval('models.' + config.BACKBONE + '.get')(config)
print('=> loading weights of the backbone')
backbone.load_state_dict(torch.load(config.NETWORK.PRETRAINED_BACKBONE))
backbone.eval()
scripted_backbone = torch.jit.script(backbone)
scripted_backbone.save("backbone.pt")

# 创建模型
model = models.faster_voxelpose.get(config)
dataset = config.DATASET.TEST_DATASET
cfg_name = os.path.basename(args.cfg).split('.')[0]
final_output_dir = os.path.join(config.OUTPUT_DIR, dataset, cfg_name)
test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
model.load_state_dict(torch.load(test_model_file))
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("faster_voxelpose.pt")
