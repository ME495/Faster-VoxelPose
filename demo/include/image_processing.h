#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 加载并预处理图像
cv::Mat load_and_preprocess_image(
    const std::string& image_path, 
    const std::vector<int>& target_image_size); 