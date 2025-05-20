#pragma once

#include <vector>
#include <string>
#include <torch/torch.h>
#include "types.h"

// 从JSON文件加载相机参数
std::vector<CameraParams> load_cameras_from_json(
    const std::string& json_path, 
    torch::Device device); 