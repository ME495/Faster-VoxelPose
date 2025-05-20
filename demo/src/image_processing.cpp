#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

cv::Mat load_and_preprocess_image(
    const std::string& image_path, 
    const std::vector<int>& target_image_size) {
    
    // 读取图像
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "错误：无法加载图像: " << image_path << std::endl;
        return cv::Mat();
    }

    // 调整图像大小到target_image_size（这是主干网络期望的输入尺寸）
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(target_image_size[0], target_image_size[1]));

    // 转换为RGB（OpenCV默认是BGR）
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

    // 归一化到[0,1]范围并减均值除以方差（ImageNet标准）
    resized_img.convertTo(resized_img, CV_32FC3, 1.0f / 255.0f);
    std::vector<cv::Mat> channels(3);
    cv::split(resized_img, channels);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean[c]) / std[c];
    }
    cv::merge(channels, resized_img);

    return resized_img;
} 