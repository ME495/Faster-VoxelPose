#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>

// 加载并预处理图像
torch::Tensor load_and_preprocess_image(
    const std::string& image_path, 
    const std::vector<int>& target_image_size, 
    torch::Device device); 