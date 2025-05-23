#include "utils.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <filesystem>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 定义可视化颜色(BGR格式，用于OpenCV) - 扩展调色板以支持更多人
const std::vector<cv::Scalar> VIS_COLORS = {
    cv::Scalar(255, 0, 0),     // 蓝色
    cv::Scalar(0, 255, 0),     // 绿色  
    cv::Scalar(0, 0, 255),     // 红色
    cv::Scalar(255, 255, 0),   // 青色
    cv::Scalar(255, 0, 255),   // 品红
    cv::Scalar(0, 255, 255),   // 黄色
    cv::Scalar(128, 0, 128),   // 紫色
    cv::Scalar(255, 165, 0),   // 橙色 (BGR格式)
    cv::Scalar(0, 128, 255),   // 橙红色
    cv::Scalar(147, 20, 255),  // 深粉色
    cv::Scalar(0, 255, 127),   // 春绿色
    cv::Scalar(255, 20, 147),  // 深粉红
    cv::Scalar(30, 144, 255),  // 道奇蓝
    cv::Scalar(50, 205, 50),   // 酸橙绿
    cv::Scalar(255, 69, 0),    // 红橙色
    cv::Scalar(138, 43, 226),  // 蓝紫色
    cv::Scalar(0, 191, 255),   // 深天蓝
    cv::Scalar(127, 255, 0),   // 查特利绿
    cv::Scalar(255, 105, 180), // 热粉色
    cv::Scalar(32, 178, 170),  // 浅海绿
};

// 列出目录中的图像文件，按字母顺序排序
std::vector<std::string> list_image_files(const std::string& dir_path) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".tiff") {
                    files.push_back(entry.path().string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "文件系统错误: " << e.what() << " 访问路径: " << dir_path << std::endl;
    }
    std::sort(files.begin(), files.end());
    return files;
}

// 获取调整大小的仿射变换矩阵
torch::Tensor get_resize_affine_transform_cpp(
    const std::vector<int>& ori_size, 
    const std::vector<int>& new_size, 
    torch::Device device) {
    
    // ori_size: 原始图像的{宽, 高}
    // new_size: 目标图像的{宽, 高}

    // 1. 计算原始图像的中心
    std::vector<float> center = {
        static_cast<float>(ori_size[0]) / 2.0f,
        static_cast<float>(ori_size[1]) / 2.0f
    };

    // 2. 根据Python中get_scale的逻辑计算缩放比例
    float w = static_cast<float>(ori_size[0]);
    float h = static_cast<float>(ori_size[1]);
    float w_resized = static_cast<float>(new_size[0]);
    float h_resized = static_cast<float>(new_size[1]);
    
    float w_pad, h_pad;
    if (w / w_resized < h / h_resized) {
        w_pad = h / h_resized * w_resized;
        h_pad = h;
    } else {
        w_pad = w;
        h_pad = w / w_resized * h_resized;
    }
    
    std::vector<float> scale = {w_pad / 200.0f, h_pad / 200.0f};
    
    // 定义输出大小和其他参数，与Python实现对齐
    float rot = 0.0f; // 调整大小操作不需要旋转
    std::vector<float> shift = {0.0f, 0.0f}; // 不需要偏移
    bool inv = false;

    // 根据Python实现的逻辑
    std::vector<float> scale_tmp = {scale[0] * 200.0f, scale[1] * 200.0f};
    float src_w = scale_tmp[0];
    float src_h = scale_tmp[1];
    float dst_w = static_cast<float>(new_size[0]);
    float dst_h = static_cast<float>(new_size[1]);

    float rot_rad = M_PI * rot / 180.0f;
    std::vector<float> src_dir, dst_dir;
    
    if (src_w >= src_h) {
        // 基于旋转计算方向向量
        src_dir = {
            static_cast<float>(0.0f - std::sin(rot_rad) * src_w * 0.5f),
            static_cast<float>(0.0f - std::cos(rot_rad) * src_w * 0.5f)
        };
        dst_dir = {0.0f, dst_w * -0.5f};
    } else {
        src_dir = {
            static_cast<float>(std::cos(rot_rad) * src_h * 0.5f),
            static_cast<float>(-std::sin(rot_rad) * src_h * 0.5f)
        };
        dst_dir = {dst_h * -0.5f, 0.0f};
    }

    // 设置仿射变换的源点和目标点
    std::vector<cv::Point2f> src_pts(3);
    std::vector<cv::Point2f> dst_pts(3);

    // 点1：中心点
    src_pts[0] = cv::Point2f(center[0] + scale_tmp[0] * shift[0], center[1] + scale_tmp[1] * shift[1]);
    dst_pts[0] = cv::Point2f(dst_w * 0.5f, dst_h * 0.5f);

    // 点2：中心 + 方向
    src_pts[1] = cv::Point2f(center[0] + src_dir[0] + scale_tmp[0] * shift[0], 
                            center[1] + src_dir[1] + scale_tmp[1] * shift[1]);
    dst_pts[1] = cv::Point2f(dst_w * 0.5f + dst_dir[0], dst_h * 0.5f + dst_dir[1]);

    // 点3：形成三角形的第三个点(类似Python中的get_3rd_point)
    src_pts[2] = cv::Point2f(src_pts[0].x - src_pts[1].y + src_pts[0].y,
                            src_pts[0].y + src_pts[1].x - src_pts[0].x);
    dst_pts[2] = cv::Point2f(dst_pts[0].x - dst_pts[1].y + dst_pts[0].y,
                            dst_pts[0].y + dst_pts[1].x - dst_pts[0].x);

    // 使用OpenCV的getAffineTransform(相当于Python中的cv2.getAffineTransform)
    cv::Mat trans_cv;
    if (inv) {
        trans_cv = cv::getAffineTransform(dst_pts, src_pts);
    } else {
        trans_cv = cv::getAffineTransform(src_pts, dst_pts);
    }

    // 将OpenCV的Mat转换为torch::Tensor
    torch::Tensor trans = torch::zeros({2, 3}, torch::kFloat32);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            trans[i][j] = trans_cv.at<double>(i, j);
        }
    }

    trans = trans.to(device);
    return trans;
} 