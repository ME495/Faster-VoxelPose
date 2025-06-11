#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <future>
#include <algorithm>
#include <iomanip>
#include <ATen/ATen.h>
#include <ATen/cudnn/Handle.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <memory>
#include <opencv2/opencv.hpp>

// 流水线数据结构
struct FrameData {
    int frame_id;
    long long timestamp;
    std::vector<cv::Mat> camera_images;
    torch::Tensor poses;
    std::chrono::high_resolution_clock::time_point capture_time;
    std::chrono::high_resolution_clock::time_point processing_start_time;
    std::chrono::high_resolution_clock::time_point processing_end_time;
    bool valid;
    
    FrameData() : frame_id(-1), timestamp(0), valid(false) {}
};

// 线程安全队列
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    size_t max_size_;
    std::atomic<long long> dropped_count_{0};
    
public:
    ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}
    
    bool push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        bool dropped = false;
        if (queue_.size() >= max_size_) {
            // 队列满，丢弃最旧的帧
            queue_.pop();
            dropped_count_++;
            dropped = true;
        }
        queue_.push(item);
        condition_.notify_one();
        return !dropped; // 返回true表示没有丢弃，false表示丢弃了旧帧
    }
    
    long long get_dropped_count() const {
        return dropped_count_.load();
    }
    
    bool pop(T& item, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (timeout_ms < 0) {
            condition_.wait(lock, [this] { return !queue_.empty(); });
        } else {
            if (!condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                                   [this] { return !queue_.empty(); })) {
                return false;
            }
        }
        item = queue_.front();
        queue_.pop();
        return true;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
    }
};

// GigE Camera includes
#include <stdio.h>
#include <conio.h>
#include "GigELib.h"
#pragma comment(lib, "ws2_32.lib")
#include <windows.h>
#include <omp.h>

// VoxelPose includes
#include "types.h"
#include "camera.h"
#include "projection.h"
#include "utils.h"
#include "visualization.h"
#include "image_processing.h"
#include "tensorrt_inference.h"
#include "person_tracker.h"
#include "license_auth.h"  // 添加授权验证头文件

using namespace QUICKCAM;
using namespace cv;
using namespace std;

// 流水线全局变量
std::atomic<bool> pipeline_running{true};
std::atomic<int> global_frame_id{0};
ThreadSafeQueue<std::shared_ptr<FrameData>> capture_queue(5);
ThreadSafeQueue<std::shared_ptr<FrameData>> processing_queue(5);

// 主线程可视化所需全局变量
std::mutex viz_mutex;
std::condition_variable viz_cv;
cv::Mat visualization_img_global;
std::atomic<bool> new_frame_for_visualization{false};

// 性能统计
std::atomic<long long> total_frames_captured{0};
std::atomic<long long> total_frames_processed{0};
std::atomic<long long> total_frames_visualized{0};
std::atomic<long long> total_capture_time_ms{0};
std::atomic<long long> total_processing_time_ms{0};
std::atomic<long long> total_visualization_time_ms{0};

// 图像捕获线程函数
void capture_thread_func(GigECamera** cam, Image* img, int camNum, 
                        const std::vector<int>& ori_image_size_cfg,
                        const std::vector<int>& image_size_cfg,
                        torch::Device device, unsigned int framerate) {
        std::cout << "图像捕获和预处理线程启动" << std::endl;
    
    while (pipeline_running) {
        auto capture_start = std::chrono::high_resolution_clock::now();
        
        auto frame_data = std::make_shared<FrameData>();
        frame_data->frame_id = global_frame_id++;
        frame_data->capture_time = capture_start;
        frame_data->camera_images.resize(camNum);
        
        long long timestamp = 0x7FFFFFFFFFFFFFFF;
        bool frame_valid = true;
        
        // 1. 获取所有相机图像
        for(int i = 0; i < camNum; ++i) {
            if(cam[i]->RetrieveBuffer(&img[i]) != GEV_STATUS_SUCCESS || img[i].GetRows() == 0) {
                frame_valid = false;
                break;
            }
            
            // 按时间戳分组帧
            if(timestamp > img[i].GetEmbeddedTimestamp())
                timestamp = img[i].GetEmbeddedTimestamp();
        }
        
        if (!frame_valid) {
            continue;
        }
        
        // 2. 取消较新的帧
        for(int i = 0; i < camNum; ++i) {
            if(img[i].GetEmbeddedTimestamp() - timestamp > 1000000/framerate/2) {
                cam[i]->CancelLastRetrievedBuffer();
                frame_valid = false;
            }
        }
        
        if (!frame_valid) {
            continue;
        }

        //for (int i = 0; i < camNum; ++i)
        //{
        //    printf("T:%fs #:%08d F:%d Res:%dx%d Format:0x%x\n",
        //        img[i].GetEmbeddedTimestamp() / 1000000., img[i].GetEmbeddedSerial(), img[i].GetEmbeddedFramecounter(),
        //        img[i].GetCols(), img[i].GetRows(), img[i].GetPixelFormat());
        //}
        
        // 3. 复制图像数据并进行预处理
        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < camNum; ++i) {
            if(img[i].GetData() && img[i].GetDataSize()) {
                int width = img[i].GetCols();
                int height = img[i].GetRows();
                
                if(width == ori_image_size_cfg[0] && height == ori_image_size_cfg[1]) {
                    Mat temp(height, width, CV_8UC3, img[i].GetData(), img[i].GetStride());
                    
                    // 图像预处理
                    // 等比例缩放到目标尺寸，不足部分用0填充
                    Mat resized_img = Mat::zeros(image_size_cfg[1], image_size_cfg[0], CV_8UC3);
                    
                    // 计算缩放比例
                    double scale_x = static_cast<double>(image_size_cfg[0]) / temp.cols;
                    double scale_y = static_cast<double>(image_size_cfg[1]) / temp.rows;
                    double scale = min(scale_x, scale_y);
                    
                    int new_width = static_cast<int>(temp.cols * scale);
                    int new_height = static_cast<int>(temp.rows * scale);
                    
                    Mat scaled_img;
                    resize(temp, scaled_img, Size(new_width, new_height));
                    
                    int x_offset = (image_size_cfg[0] - new_width) / 2;
                    int y_offset = (image_size_cfg[1] - new_height) / 2;
                    
                    scaled_img.copyTo(resized_img(Rect(x_offset, y_offset, new_width, new_height)));

                    frame_data->camera_images[i] = resized_img.clone(); // 保留原始图像用于可视化
                    
                    // // 归一化并转换为tensor
                    // Mat normalized_img;
                    // resized_img.convertTo(normalized_img, CV_32FC3, 1.0f / 255.0f);
                    // std::vector<cv::Mat> channels(3);
                    // cv::split(normalized_img, channels);
                    // const float mean[3] = {0.485f, 0.456f, 0.406f};
                    // const float std[3] = {0.229f, 0.224f, 0.225f};
                    // for (int c = 0; c < 3; ++c) {
                    //     channels[c] = (channels[c] - mean[c]) / std[c];
                    // }
                    // cv::merge(channels, normalized_img);
                    
                    // torch::Tensor img_tensor = torch::from_blob(
                    //     normalized_img.data,
                    //     {normalized_img.rows, normalized_img.cols, 3},
                    //     torch::kFloat32
                    // ).clone().permute({2, 0, 1}).unsqueeze(0).to(device);
                    
                    // frame_data->processed_tensors[i] = img_tensor;
                } else {
                    frame_valid = false;
                    break;
                }
            } else {
                frame_valid = false;
                break;
            }
        }
        
        frame_data->timestamp = timestamp;
        frame_data->valid = frame_valid;
        
        if (frame_valid) {
            capture_queue.push(frame_data);
            auto capture_end = std::chrono::high_resolution_clock::now();
            auto capture_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                capture_end - capture_start).count();
            total_capture_time_ms += capture_duration;
            total_frames_captured++;
        }
        
        // 简单的帧率控制
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
            std::cout << "图像捕获和预处理线程退出" << std::endl;
}

// 图像处理和姿态估计线程函数
void processing_thread_func(std::unique_ptr<TensorRTInference>& backbone_trt,
                           torch::jit::script::Module& model_module,
                           const std::vector<int>& ori_image_size_cfg,
                           const std::vector<int>& image_size_cfg,
                           const std::vector<int>& heatmap_size_cfg,
                           const torch::Tensor& final_sample_grid_coarse,
                           const torch::Tensor& final_sample_grid_fine,
                           PersonTracker& person_tracker,
                           torch::Device device) {
    std::cout << "姿态估计线程启动" << std::endl;
    
    while (pipeline_running || !capture_queue.empty()) {
        std::shared_ptr<FrameData> frame_data;
        if (!capture_queue.pop(frame_data, 100)) { // 100ms超时
            continue;
        }
        
        if (!frame_data || !frame_data->valid) {
            continue;
        }
        
        auto processing_start = std::chrono::high_resolution_clock::now();
        frame_data->processing_start_time = processing_start;
        
        // // 使用已预处理的张量
        // int camNum = frame_data->processed_tensors.size();
        
        // // 检查预处理的张量是否都有效
        // if (camNum == 0 || frame_data->processed_tensors.empty()) {
        //     continue;
        // }

        // 使用原始图像
        int camNum = frame_data->camera_images.size();
        
        if (camNum == 0 || frame_data->camera_images.empty()) {
            continue;
        }

        std::vector<torch::Tensor> processed_tensors;
        processed_tensors.resize(camNum);

        for (int i = 0; i < camNum; ++i) {
            Mat resized_img = frame_data->camera_images[i];
            processed_tensors[i] = torch::from_blob(
                resized_img.data,
                {resized_img.rows, resized_img.cols, 3},
                torch::kUInt8
            ).to(torch::kFloat).to(device).permute({ 2, 0, 1 }).unsqueeze(0);
        }
        torch::Tensor batch_input = torch::cat(processed_tensors, 0);

        torch::NoGradGuard no_grad;

        // 归一化并转换为tensor
        batch_input = batch_input / 255.0f;
        torch::Tensor mean = torch::tensor({ 0.485f, 0.456f, 0.406f }, device).reshape({1, 3, 1, 1});
        torch::Tensor standard_deviation = torch::tensor({0.229f, 0.224f, 0.225f}, device).reshape({ 1, 3, 1, 1 });
        batch_input = (batch_input - mean) / standard_deviation;
        batch_input = batch_input.to(torch::kHalf);

        // 姿态估计
        // Heatmap提取
        torch::Tensor batch_heatmaps = backbone_trt->forward(batch_input);
        batch_heatmaps = batch_heatmaps.to(torch::kFloat);
        batch_heatmaps = batch_heatmaps.unsqueeze(0);
        
        // 姿态估计
        std::vector<torch::jit::IValue> model_inputs;
        model_inputs.push_back(batch_heatmaps);
        model_inputs.push_back(final_sample_grid_coarse);
        model_inputs.push_back(final_sample_grid_fine);
        torch::Tensor fused_poses = model_module.forward(model_inputs).toTensor();
        
        // 人体跟踪
        if (fused_poses.defined() && fused_poses.numel() > 0) {
            fused_poses = person_tracker.update(fused_poses, frame_data->frame_id);
        }
        
        frame_data->poses = fused_poses;
        // processed_tensors已经在捕获线程中设置
        
        auto processing_end = std::chrono::high_resolution_clock::now();
        frame_data->processing_end_time = processing_end;
        
        if (device.type() == torch::kCUDA) {
            torch::cuda::synchronize();
            frame_data->processing_end_time = std::chrono::high_resolution_clock::now();
        }
        
        auto processing_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            frame_data->processing_end_time - processing_start).count();
        total_processing_time_ms += processing_duration;
        total_frames_processed++;
        
        // 发送到可视化队列
        processing_queue.push(frame_data);
    }
    
    std::cout << "姿态估计线程退出" << std::endl;
}

// 可视化线程函数
void visualization_thread_func(const std::vector<CameraParams>& cameras,
                              const torch::Tensor& resize_transform_tensor,
                              const std::vector<int>& ori_image_size_cfg,
                              bool enable_visualization) {
    std::cout << "可视化线程启动" << std::endl;
    
    // FPS计算变量
    int frames = 0;
    double fps = 0.0;
    double freq = getTickFrequency();
    int64 lastTime = getTickCount();
    
    while (pipeline_running || !processing_queue.empty()) {
        std::shared_ptr<FrameData> frame_data;
        if (!processing_queue.pop(frame_data, 100)) { // 100ms超时
            continue;
        }
        
        if (!frame_data || !frame_data->valid || frame_data->camera_images.empty()) {
            continue;
        }
        
        auto visualization_start = std::chrono::high_resolution_clock::now();
        
        if (enable_visualization) {
            Mat visualization_img = frame_data->camera_images[0].clone();
            
            if (frame_data->poses.defined() && frame_data->poses.numel() > 0 && frame_data->poses.dim() == 4) {
                // 获取关节点定义
                int num_joints_from_pose = frame_data->poses.size(2);
                std::vector<std::pair<int, int>> current_limbs = get_limbs_definition(num_joints_from_pose);
                
                // 3D姿态可视化
                visualize_3d_pose(
                    visualization_img,
                    frame_data->poses,
                    cameras[0],
                    resize_transform_tensor,
                    current_limbs,
                    ori_image_size_cfg
                );
            }
            
            // 添加FPS和统计信息
            frames++;
            if (frames % 10 == 0) {
                int64 now = getTickCount();
                double seconds = (now - lastTime) / freq;
                fps = 10.0 / seconds;
                lastTime = now;
            }
            
            std::string fps_text = "FPS: " + std::to_string((int)fps);
            std::string frame_text = "Frame: " + std::to_string(frame_data->frame_id);
            std::string queue_text = "Queue: C" + std::to_string(capture_queue.size()) + 
                                   " P" + std::to_string(processing_queue.size());
            
            putText(visualization_img, fps_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
            putText(visualization_img, frame_text, Point(10, 70), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
            putText(visualization_img, queue_text, Point(10, 110), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 2);
            
            // 将图像发送到主线程进行显示
            {
                std::lock_guard<std::mutex> lock(viz_mutex);
                visualization_img_global = visualization_img;
                new_frame_for_visualization = true;
            }
            viz_cv.notify_one();
        }
        
        auto visualization_end = std::chrono::high_resolution_clock::now();
        auto visualization_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            visualization_end - visualization_start).count();
        total_visualization_time_ms += visualization_duration;
        total_frames_visualized++;
    }
    
    std::cout << "可视化线程退出" << std::endl;
}

int main(int argc, const char* argv[]) {
    if (argc < 5) {
        std::cerr << "用法: capture_from_cameras <path_to_scripted_backbone.pt> "
                     "<path_to_scripted_model.pt> <path_to_calibration.json> "
                     "<device (cpu/cuda)> [bind_ip_address] [enable_visualization]" << std::endl;
        std::cerr << "参数说明:" << std::endl;
        std::cerr << "  bind_ip_address: 可选，默认为 192.168.2.250" << std::endl;
        std::cerr << "  enable_visualization: 可选，0=关闭可视化(默认)，1=开启可视化" << std::endl;
        return -1;
    }

    // --- 授权验证 ---
    std::cout << "正在进行授权验证..." << std::endl;
    if (!verifyLicense()) {
        std::cerr << "\n===============================================" << std::endl;
        std::cerr << "授权验证失败！程序将退出。" << std::endl;
        std::cerr << "请联系技术支持获取有效的授权码。" << std::endl;
        std::cerr << "===============================================" << std::endl;
        system("pause");
        return -1;
    }
    std::cout << "授权验证通过，程序继续运行...\n" << std::endl;

    std::string backbone_path = argv[1];
    std::string model_path = argv[2];
    std::string calib_json_path = argv[3];
    std::string device_str = argv[4];
    std::string bind_ip = (argc > 5) ? argv[5] : "192.168.2.250";
    bool enable_visualization = false;  // 默认关闭可视化
    if (argc > 6) {
        int viz_flag = std::atoi(argv[6]);
        enable_visualization = (viz_flag != 0);
    }

    // 设置设备(CPU/CUDA)
    torch::DeviceType device_type;
    if (device_str == "cuda" && torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "使用CUDA设备." << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "使用CPU设备." << std::endl;
    }
    torch::Device device(device_type);
    
    // 显示可视化设置
    std::cout << "可视化设置: " << (enable_visualization ? "开启" : "关闭") << std::endl;

    // 设置torch线程数
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads > 8) {
        torch::set_num_threads(8);
        torch::set_num_interop_threads(8);
        std::cout << "已设置torch线程数: " << 8 << std::endl;
    }

    // --- 配置参数 ---
    std::vector<int> ori_image_size_cfg = {2048, 1544};      
    std::vector<int> image_size_cfg = {1024, 784};           
    std::vector<int> heatmap_size_cfg = {256, 200};          
    const int NUM_CAMERAS = 4;
    unsigned int framerate = 30;

    std::vector<float> space_size_cfg = {8000.0f, 8000.0f, 2000.0f}; 
    std::vector<float> space_center_cfg = {0.0f, -300.0f, 800.0f};  
    std::vector<int> voxels_per_axis_cfg = {80, 80, 20};           
    std::vector<float> individual_space_size_cfg = {2000.0f, 2000.0f, 2000.0f};
    std::vector<int> ind_voxels_per_axis_cfg = {64, 64, 64};      

    // 设置OpenMP线程数
    int openmp_threads = min(NUM_CAMERAS, (int)std::thread::hardware_concurrency());
    omp_set_num_threads(openmp_threads);
    std::cout << "已设置OpenMP线程数: " << openmp_threads << std::endl;

    // 加载相机参数
    std::vector<CameraParams> cameras = load_cameras_from_json(calib_json_path, device);
    if (cameras.size() < NUM_CAMERAS) {
        std::cerr << "错误: 加载了 " << cameras.size() << " 个相机, 需要 " << NUM_CAMERAS << std::endl;
        return -1;
    }
    if (cameras.size() > NUM_CAMERAS) {
         cameras.resize(NUM_CAMERAS);
    }

    // 加载模型
    std::unique_ptr<TensorRTInference> backbone_trt;
    torch::jit::script::Module model_module;
    try {
        if (backbone_path.substr(backbone_path.find_last_of(".") + 1) == "trt") {
            std::cout << "加载 TensorRT backbone 引擎: " << backbone_path << std::endl;
            backbone_trt = std::make_unique<TensorRTInference>(backbone_path, device);
            if (!backbone_trt->isInitialized()) {
                throw std::runtime_error("TensorRT backbone 初始化失败");
            }
        } else {
            throw std::runtime_error("backbone 文件必须是 .trt 格式");
        }
		
        model_module = torch::jit::load(model_path, device);
        model_module.eval();
    } catch (const std::exception& e) {
        std::cerr << "加载模型时出错:\n" << e.what() << std::endl;
        return -1;
    }

    // --- 预计算网格 ---
    torch::Tensor resize_transform_tensor = get_resize_affine_transform_cpp(ori_image_size_cfg, image_size_cfg, device);
    torch::Tensor grid_coarse = compute_grid_cpp(space_size_cfg, space_center_cfg, voxels_per_axis_cfg, device);
    int nbins_coarse = voxels_per_axis_cfg[0] * voxels_per_axis_cfg[1] * voxels_per_axis_cfg[2];
    std::vector<torch::Tensor> sample_grids_coarse_list_static;

    torch::Tensor world_space_size_tensor = torch::tensor(space_size_cfg, device);
    torch::Tensor ind_space_size_tensor = torch::tensor(individual_space_size_cfg, device);
    torch::Tensor ind_voxels_per_axis_tensor = torch::tensor(ind_voxels_per_axis_cfg, torch::kInt).to(device);
    torch::Tensor fine_voxels_per_axis_float = (world_space_size_tensor / ind_space_size_tensor * 
                                             (ind_voxels_per_axis_tensor.to(torch::kFloat) - 1.0f)) + 1.0f;
    std::vector<int> fine_voxels_per_axis_cfg(3);
    fine_voxels_per_axis_cfg[0] = static_cast<int>(fine_voxels_per_axis_float[0].item<float>());
    fine_voxels_per_axis_cfg[1] = static_cast<int>(fine_voxels_per_axis_float[1].item<float>());
    fine_voxels_per_axis_cfg[2] = static_cast<int>(fine_voxels_per_axis_float[2].item<float>());
    torch::Tensor grid_fine = compute_grid_cpp(space_size_cfg, space_center_cfg, fine_voxels_per_axis_cfg, device);
    int nbins_fine = fine_voxels_per_axis_cfg[0] * fine_voxels_per_axis_cfg[1] * fine_voxels_per_axis_cfg[2];
    std::vector<torch::Tensor> sample_grids_fine_list_static;

    for (int i = 0; i < NUM_CAMERAS; ++i) {
        torch::Tensor sg_coarse_cam = project_grid_cpp(grid_coarse, ori_image_size_cfg, image_size_cfg, 
                                                        cameras[i], heatmap_size_cfg[0], heatmap_size_cfg[1],
                                                        nbins_coarse, resize_transform_tensor, device);
        sample_grids_coarse_list_static.push_back(sg_coarse_cam.view({voxels_per_axis_cfg[0], 
                                                                     voxels_per_axis_cfg[1], 
                                                                     voxels_per_axis_cfg[2], 2}));

        torch::Tensor sg_fine_cam = project_grid_cpp(grid_fine, ori_image_size_cfg, image_size_cfg, 
                                                      cameras[i], heatmap_size_cfg[0], heatmap_size_cfg[1],
                                                      nbins_fine, resize_transform_tensor, device);
        sample_grids_fine_list_static.push_back(sg_fine_cam.view({fine_voxels_per_axis_cfg[0], 
                                                                 fine_voxels_per_axis_cfg[1], 
                                                                 fine_voxels_per_axis_cfg[2], 2}));
    }
    torch::Tensor final_sample_grid_coarse = torch::stack(sample_grids_coarse_list_static, 0).unsqueeze(0);
    torch::Tensor final_sample_grid_fine = torch::stack(sample_grids_fine_list_static, 0).unsqueeze(0);

    // --- 初始化人体跟踪器 ---
    PersonTracker person_tracker(10, 500.0f);
    std::cout << "已初始化人体跟踪器 (最大丢失帧数: 10, 距离阈值: 500.0)" << std::endl;

    // --- 初始化GigE相机 ---
    QCGuid guid;
    GigECamera *cam[10] = {NULL};
    BusManager busMgr;
    Property ppt;
    Image img[10];

    unsigned int camNum = 0, R3Num = 0;
    
    // 启动相机管理器
    BusManager::StartManager(htonl(inet_addr(bind_ip.c_str())));
    busMgr.RequestEnumerateCameras();
    busMgr.GetNumOfCameras(&camNum);
    
    if(camNum == 0) {
        printf("没有找到ChingMU GigE相机!\n");
        goto camera_cleanup;
    } else {
        printf("找到 %d 个相机:\n", camNum);
        for(int i=0; i<camNum; ++i) {
            uint32_t camSerial = 0;
            busMgr.GetCameraSerialNumberFromIndex(i, &camSerial);
            printf("%08d\n", camSerial);
        }
    }

    // 连接R3相机
    printf("连接R3相机:\n");
    for(int i=0; i<camNum && i<200; ++i) {
        DeviceInfo devInfo;
        busMgr.GetCameraFromIndex(i, &guid);
        busMgr.GetDeviceInfo(&guid, &devInfo);
        if(!strcmp((char*)devInfo.user_defined_name, "R3"))
            cam[R3Num] = new GigECamera();
        else
            continue;
    
        if(cam[R3Num]->Connect(&guid) != GEV_STATUS_SUCCESS) {
            printf("连接相机[%08s] 失败!\n", devInfo.serial_number);
            goto camera_cleanup;
        }
    
        // 设置相机帧率
        ppt.type = PT_FRAME_RATE;
        ppt.valueA = framerate;
        if(cam[R3Num]->SetProperty(&ppt) != GEV_STATUS_SUCCESS) {
            printf("设置相机[%08s] fps[%d] 失败!\n", devInfo.serial_number, ppt.valueA);
            goto camera_cleanup;
        }

        // 设置相机传输图像
        bool lock = true;
        cam[R3Num]->SetTransmitImageLock(&lock);

        // 开始捕获图像
        if(cam[R3Num]->StartCapture()!= GEV_STATUS_SUCCESS) {
            printf("启动相机[%08s] 捕获图像失败\n", devInfo.serial_number);
            goto camera_cleanup;
        }

        R3Num++;
        if(R3Num >= NUM_CAMERAS) break; // 只使用需要的相机数量
    }

    if(R3Num < NUM_CAMERAS) {
        printf("错误: 只找到 %d 个R3相机, 需要 %d 个\n", R3Num, NUM_CAMERAS);
        goto camera_cleanup;
    }

        camNum = R3Num;
    
    // 根据设置决定是否创建OpenCV窗口
    if (enable_visualization) {
        namedWindow("Pose Estimation Results", WINDOW_NORMAL);
        resizeWindow("Pose Estimation Results", 1024, 784);
        std::cout << "已创建可视化窗口" << std::endl;
    }
    
    std::cout << "==================================================" << std::endl;
    std::cout << "启动流水线并行处理系统" << std::endl;
    std::cout << "图像捕获&预处理 -> 姿态估计 -> 可视化 (并行)" << std::endl;
    if (enable_visualization) {
        std::cout << "可视化模式: 启用，在窗口中按 Q/ESC 键退出" << std::endl;
    } else {
        std::cout << "可视化模式: 禁用，按 Q 键退出" << std::endl;
    }
    std::cout << "队列大小: 捕获=" << capture_queue.size() 
              << ", 处理=" << processing_queue.size() << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // --- 流水线处理作用域 (避免goto跳过变量初始化) ---
    {
        // --- 启动流水线线程 ---
        pipeline_running = true;
        
        // 启动图像捕获线程
        std::thread capture_thread(capture_thread_func, cam, img, camNum, 
                                  std::ref(ori_image_size_cfg), std::ref(image_size_cfg), device, framerate);
        
        // 启动姿态估计线程
        std::thread processing_thread(processing_thread_func, 
                                     std::ref(backbone_trt), std::ref(model_module),
                                     std::ref(ori_image_size_cfg), std::ref(image_size_cfg), std::ref(heatmap_size_cfg),
                                     std::ref(final_sample_grid_coarse), std::ref(final_sample_grid_fine),
                                     std::ref(person_tracker), device);
        
        // 启动可视化线程
        std::thread visualization_thread(visualization_thread_func,
                                       std::ref(cameras), std::ref(resize_transform_tensor),
                                       std::ref(ori_image_size_cfg), enable_visualization);
        
        // 监控线程 - 定期输出统计信息
        std::thread monitor_thread([&]() {
            std::cout << "监控线程启动" << std::endl;
            int last_captured = 0, last_processed = 0, last_visualized = 0;
            auto last_time = std::chrono::high_resolution_clock::now();
            
            while (pipeline_running) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                
                auto current_time = std::chrono::high_resolution_clock::now();
                int current_captured = total_frames_captured.load();
                int current_processed = total_frames_processed.load();
                int current_visualized = total_frames_visualized.load();
                
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - last_time).count();
                if (duration > 0) {
                    double capture_fps = (current_captured - last_captured) / static_cast<double>(duration);
                    double processing_fps = (current_processed - last_processed) / static_cast<double>(duration);
                    double visualization_fps = (current_visualized - last_visualized) / static_cast<double>(duration);
                    
                                    std::cout << "=== 流水线状态统计 ===" << std::endl;
                std::cout << "捕获FPS: " << std::fixed << std::setprecision(1) << capture_fps 
                         << " | 处理FPS: " << processing_fps 
                         << " | 可视化FPS: " << visualization_fps << std::endl;
                std::cout << "队列状态: 捕获=" << capture_queue.size() 
                         << ", 处理=" << processing_queue.size() << std::endl;
                std::cout << "总帧数: 捕获=" << current_captured 
                         << ", 处理=" << current_processed 
                         << ", 可视化=" << current_visualized << std::endl;
                std::cout << "丢弃帧数: 捕获队列=" << capture_queue.get_dropped_count() 
                         << ", 处理队列=" << processing_queue.get_dropped_count() << std::endl;
                    
                    if (current_processed > 0) {
                        double avg_capture_ms = static_cast<double>(total_capture_time_ms.load()) / current_captured;
                        double avg_processing_ms = static_cast<double>(total_processing_time_ms.load()) / current_processed;
                                                double avg_visualization_ms = static_cast<double>(total_visualization_time_ms.load()) / 
                                                       max(1LL, static_cast<long long>(current_visualized));
                        
                        std::cout << "平均耗时: 捕获=" << std::fixed << std::setprecision(2) << avg_capture_ms 
                                 << "ms, 处理=" << avg_processing_ms 
                                 << "ms, 可视化=" << avg_visualization_ms << "ms" << std::endl;
                    }
                    std::cout << "=========================" << std::endl;
                    
                    last_captured = current_captured;
                    last_processed = current_processed;
                    last_visualized = current_visualized;
                    last_time = current_time;
                }
            }
            std::cout << "监控线程退出" << std::endl;
        });
        
        // 主线程等待退出信号或处理可视化
        if (enable_visualization) {
            cv::Mat local_viz_img; // 用于在主线程中显示
            while (pipeline_running) {
                {
                    std::unique_lock<std::mutex> lock(viz_mutex);
                    if (viz_cv.wait_for(lock, std::chrono::milliseconds(100), [&]{ return new_frame_for_visualization.load(); })) {
                        local_viz_img = visualization_img_global.clone();
                        new_frame_for_visualization = false;
                    }
                }

                if (!local_viz_img.empty()) {
                    imshow("Pose Estimation Results", local_viz_img);
                }

                int key = waitKey(1);
                if (key == 'q' || key == 'Q' || key == 27) {
                    pipeline_running = false;
                }
            }
        } else {
            std::cout << "流水线运行中，按 Q 键退出..." << std::endl;
            while (pipeline_running) {
                if (_kbhit() && _getch() == 'q') {
                    pipeline_running = false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        
        // 等待所有线程完成
        std::cout << "正在停止流水线..." << std::endl;
        
        if (capture_thread.joinable()) {
            capture_thread.join();
        }
        if (processing_thread.joinable()) {
            processing_thread.join();
        }
        if (visualization_thread.joinable()) {
            visualization_thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
        
        // 输出最终统计
        std::cout << "\n==================== 流水线最终统计 ====================" << std::endl;
        
        long long final_captured = total_frames_captured.load();
        long long final_processed = total_frames_processed.load();
        long long final_visualized = total_frames_visualized.load();
        
        std::cout << "总处理帧数:" << std::endl;
        std::cout << "  - 图像捕获: " << final_captured << " 帧" << std::endl;
        std::cout << "  - 姿态估计: " << final_processed << " 帧" << std::endl;
        std::cout << "  - 可视化显示: " << final_visualized << " 帧" << std::endl;
        
        std::cout << "丢弃帧统计:" << std::endl;
        std::cout << "  - 捕获队列丢弃: " << capture_queue.get_dropped_count() << " 帧" << std::endl;
        std::cout << "  - 处理队列丢弃: " << processing_queue.get_dropped_count() << " 帧" << std::endl;
        
        if (final_captured > 0) {
            double avg_capture_ms = static_cast<double>(total_capture_time_ms.load()) / final_captured;
            std::cout << "平均图像捕获时间: " << std::fixed << std::setprecision(2) 
                     << avg_capture_ms << " 毫秒/帧" << std::endl;
        }
        
        if (final_processed > 0) {
            double avg_processing_ms = static_cast<double>(total_processing_time_ms.load()) / final_processed;
            std::cout << "平均姿态估计时间: " << std::fixed << std::setprecision(2) 
                     << avg_processing_ms << " 毫秒/帧" << std::endl;
        }
        
        if (final_visualized > 0) {
            double avg_visualization_ms = static_cast<double>(total_visualization_time_ms.load()) / final_visualized;
            std::cout << "平均可视化时间: " << std::fixed << std::setprecision(2) 
                     << avg_visualization_ms << " 毫秒/帧" << std::endl;
        }
        
        // 计算流水线效率
        double capture_efficiency = final_captured > 0 ? (static_cast<double>(final_processed) / final_captured * 100.0) : 0.0;
        double visualization_efficiency = final_processed > 0 ? (static_cast<double>(final_visualized) / final_processed * 100.0) : 0.0;
        
        std::cout << "流水线效率:" << std::endl;
        std::cout << "  - 处理效率: " << std::fixed << std::setprecision(1) << capture_efficiency << "%" << std::endl;
        std::cout << "  - 可视化效率: " << std::fixed << std::setprecision(1) << visualization_efficiency << "%" << std::endl;
        
        std::cout << "OpenMP线程数: " << omp_get_max_threads() << std::endl;
        std::cout << "========================================================" << std::endl;
    } // 流水线处理作用域结束

camera_cleanup:
    // 清理相机资源
    for(int i=0; i<camNum; ++i) {
        if(cam[i] && cam[i]->IsConnected()) {
            bool lock = false;
            cam[i]->SetTransmitImageLock(&lock);
            cam[i]->StopCapture();
            cam[i]->Disconnect();
            delete cam[i];
            cam[i] = NULL;
        }
    }
    
    // 释放OpenCV资源
    if (enable_visualization) {
        destroyAllWindows();
    }
    
    std::cout << "程序结束." << std::endl;
    
    return 0;
}
