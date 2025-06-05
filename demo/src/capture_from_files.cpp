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
#include <opencv2/opencv.hpp>

#include "types.h"
#include "camera.h"
#include "projection.h"
#include "utils.h"
#include "visualization.h"
#include "image_processing.h"
#include "tensorrt_inference.h"
#include "person_tracker.h"
#include "license_auth.h"

int main(int argc, const char* argv[]) {
    if (argc < 6) { // Prog_name, backbone, model, calib.json, image_base_dir, device
        std::cerr << "用法: capture_from_files <path_to_scripted_backbone.pt> \
                     <path_to_scripted_model.pt> <path_to_calibration.json> \
                     <image_sequence_base_dir> <device (cpu/cuda)> [num_frames_to_process]" << std::endl;
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
    std::string image_base_dir = argv[4];
    std::string device_str = argv[5];
    int num_frames_to_process_arg = -1; // 不指定则处理所有帧
    if (argc > 6) {
        try {
            num_frames_to_process_arg = std::stoi(argv[6]);
        } catch (const std::exception& e) {
            std::cerr << "无效的帧数参数: " << argv[6] << std::endl;
            return -1;
        }
    }

    // 设置设备(CPU/CUDA)
    torch::DeviceType device_type;
    if (device_str == "cuda" && torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "使用CUDA设备." << std::endl;
        // // 设置cudnn参数（仅在CUDA下有效）
        // torch::globalContext().setBenchmarkCuDNN(true);
        // torch::globalContext().setDeterministicCuDNN(false);
        // std::cout << "已开启cuDNN benchmark, 关闭deterministic." << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "使用CPU设备." << std::endl;
    }
    torch::Device device(device_type);

    // 设置torch线程数（可根据实际CPU核心数调整）
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads > 8) {
        torch::set_num_threads(8);
        torch::set_num_interop_threads(8);
        std::cout << "已设置torch线程数: " << 8 << std::endl;
    }

    // --- 配置参数 (来自configs/custom/jln64.yaml) --- 
    std::vector<int> ori_image_size_cfg = {2048, 1544};      
    std::vector<int> image_size_cfg = {1024, 784};           
    std::vector<int> heatmap_size_cfg = {256, 200};          
    const int NUM_CAMERAS = 4; // 需求定义

    std::vector<float> space_size_cfg = {8000.0f, 8000.0f, 2000.0f}; 
    std::vector<float> space_center_cfg = {0.0f, -300.0f, 800.0f};  
    std::vector<int> voxels_per_axis_cfg = {80, 80, 20};           
    std::vector<float> individual_space_size_cfg = {2000.0f, 2000.0f, 2000.0f};
    std::vector<int> ind_voxels_per_axis_cfg = {64, 64, 64};      
    // --- 配置结束 ---

    // 加载相机参数
    std::vector<CameraParams> cameras = load_cameras_from_json(calib_json_path, device);
    if (cameras.size() < NUM_CAMERAS) {
        std::cerr << "错误: 加载了 " << cameras.size() << " 个相机, 需要 " << NUM_CAMERAS << std::endl;
        return -1;
    }
    // 确保只使用NUM_CAMERAS个相机
    if (cameras.size() > NUM_CAMERAS) {
         cameras.resize(NUM_CAMERAS);
    }

    // 列出每个相机的图像文件
    std::vector<std::vector<std::string>> camera_image_files(NUM_CAMERAS);
    size_t min_frames = std::string::npos;

    for (int i = 0; i < NUM_CAMERAS; ++i) {
        fs::path cam_dir = fs::path(image_base_dir) / cameras[i].id;
        camera_image_files[i] = list_image_files(cam_dir.string());
        if (camera_image_files[i].empty()) {
            std::cerr << "错误: 在目录中没有找到图像文件: " << cam_dir.string() << std::endl;
            return -1;
        }
        if (min_frames == std::string::npos || camera_image_files[i].size() < min_frames) {
            min_frames = camera_image_files[i].size();
        }
        std::cout << "为相机 " << cameras[i].id << " 找到 " << camera_image_files[i].size() << " 张图像" << std::endl;
    }

    if (min_frames == 0 || min_frames == std::string::npos) {
        std::cerr << "错误: 没有可处理的帧或各相机之间的帧数不一致." << std::endl;
        return -1;
    }
    std::cout << "处理 " << min_frames << " 帧同步序列." << std::endl;

    int frames_to_run = (num_frames_to_process_arg > 0 && num_frames_to_process_arg < min_frames) ? 
                         num_frames_to_process_arg : min_frames;
    std::cout << "将处理 " << frames_to_run << " 帧." << std::endl;

    // 加载模型
    std::unique_ptr<TensorRTInference> backbone_trt;
    torch::jit::script::Module model_module;
    try {
        // 检查 backbone 文件扩展名
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

    // --- 预计算网格（这些在整个序列中是静态的） ---
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
    // --- 预计算网格结束 ---

    // --- 线程安全队列和同步变量 ---
    std::queue<std::vector<cv::Mat>> image_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool finished = false;
    bool image_error = false;

    // --- 图片读取线程函数 ---
    auto image_loader_thread_func = [&](int frames_to_run) {
        for (int frame_idx = 0; frame_idx < frames_to_run; ++frame_idx) {
            std::vector<cv::Mat> batch_images;
            bool has_error = false;
            // 并行读取所有相机图片
            std::vector<std::future<cv::Mat>> image_futures;
            for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
                image_futures.push_back(std::async(std::launch::async, load_and_preprocess_image,
                    camera_image_files[cam_idx][frame_idx], image_size_cfg));
            }
            for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
                cv::Mat norm_img = image_futures[cam_idx].get();
                if (norm_img.empty()) {
                    std::cerr << "由于相机 " << cam_idx << " 的图像加载错误，跳过帧处理" << std::endl;
                    has_error = true;
                    break;
                }
                batch_images.push_back(norm_img);
            }
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                // 如果队列元素大于10则等待，避免消耗过多内存
                queue_cv.wait(lock, [&]{ return image_queue.size() <= 10; });
                if (has_error || batch_images.size() != NUM_CAMERAS) {
                    image_error = true;
                    break;
                }
                image_queue.push(std::move(batch_images));
                queue_cv.notify_one();
            }
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            finished = true;
        }
        queue_cv.notify_all();
    };

    // 启动图片读取线程
    std::thread image_loader_thread(image_loader_thread_func, frames_to_run);

    // --- 初始化人体跟踪器 ---
    PersonTracker person_tracker(10, 500.0f); // 最大丢失帧数10，距离阈值500.0
    std::cout << "已初始化人体跟踪器 (最大丢失帧数: 10, 距离阈值: 500.0)" << std::endl;

    // --- 处理循环 ---
    torch::NoGradGuard no_grad;
    long long total_duration_ms = 0;
    long long total_image_processing_ms = 0;
    long long total_heatmap_extraction_ms = 0;
    long long total_pose_estimation_ms = 0;
    int frames_processed_for_fps = 0;
    int warmup_frames = 10;

    std::cout << "开始处理 " << frames_to_run << " 帧..." << std::endl;

    for (int frame_idx = 0; frame_idx < frames_to_run; ++frame_idx) {
        if (frame_idx % 50 == 0 || frame_idx == frames_to_run -1) {
             std::cout << "处理帧 " << frame_idx + 1 << "/" << frames_to_run << std::endl;
        }

        auto overall_frame_start_time = std::chrono::high_resolution_clock::now();

        // --- 1. 从队列获取图片张量 ---
        auto image_processing_start_time = std::chrono::high_resolution_clock::now();
        std::vector<torch::Tensor> batch_tensors;
        std::vector<cv::Mat> batch_images;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [&]{ return !image_queue.empty() || finished || image_error; });
            if (image_error) {
                std::cerr << "图片读取线程发生错误，终止主循环。" << std::endl;
                break;
            }
            if (!image_queue.empty()) {
                batch_images = std::move(image_queue.front());
                image_queue.pop();
                queue_cv.notify_one();
            } else if (finished) {
                break;
            }
        }
        if (batch_images.size() != NUM_CAMERAS) {
            std::cerr << "错误: 为帧 " << frame_idx << " 加载的图像不足" << std::endl;
            continue;
        }
        auto image_processing_end_time = std::chrono::high_resolution_clock::now();
        if (device_type == torch::kCUDA) {
            torch::cuda::synchronize();
            image_processing_end_time = std::chrono::high_resolution_clock::now();
        }

        // --- 2. Heatmap提取 ---
        // 主线程将cv::Mat转为Tensor
        for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
            const cv::Mat& norm_img = batch_images[cam_idx];
            torch::Tensor img_tensor = torch::from_blob(
                norm_img.data,
                {norm_img.rows, norm_img.cols, 3},
                torch::kFloat32
            ).clone().permute({2, 0, 1}).unsqueeze(0).to(device);
            batch_tensors.push_back(img_tensor);
        }
        
        auto heatmap_extraction_start_time = std::chrono::high_resolution_clock::now();
        torch::Tensor batch_input = torch::cat(batch_tensors, 0);  // [NUM_CAMERAS, 3, H, W]
        batch_input = batch_input.to(torch::kHalf);
        torch::Tensor batch_heatmaps = backbone_trt->forward(batch_input);
        batch_heatmaps = batch_heatmaps.to(torch::kFloat);
        batch_heatmaps = batch_heatmaps.unsqueeze(0);
        auto heatmap_extraction_end_time = std::chrono::high_resolution_clock::now();
        if (device_type == torch::kCUDA) {
            torch::cuda::synchronize();
            heatmap_extraction_end_time = std::chrono::high_resolution_clock::now();
        }

        // --- 3. 姿态估计 ---
        auto pose_estimation_start_time = std::chrono::high_resolution_clock::now();
        std::vector<torch::jit::IValue> model_inputs;
        model_inputs.push_back(batch_heatmaps);
        model_inputs.push_back(final_sample_grid_coarse); 
        model_inputs.push_back(final_sample_grid_fine);
        torch::Tensor fused_poses = model_module.forward(model_inputs).toTensor();
        
        // --- 4. 人体跟踪 ---
        if (fused_poses.defined() && fused_poses.numel() > 0) {
            fused_poses = person_tracker.update(fused_poses, frame_idx);
            
            // 输出跟踪信息（每50帧或最后一帧）
            if (frame_idx % 50 == 0 || frame_idx == frames_to_run - 1) {
                std::cout << "  帧 " << frame_idx << ": 当前跟踪 " 
                         << person_tracker.get_tracked_person_count() << " 个人体" << std::endl;
            }
        }
        
        auto pose_estimation_end_time = std::chrono::high_resolution_clock::now();
        if (device_type == torch::kCUDA) {
            torch::cuda::synchronize(); 
            pose_estimation_end_time = std::chrono::high_resolution_clock::now();
        }
        auto overall_frame_end_time = std::chrono::high_resolution_clock::now();
        // 计时统计
        if (frame_idx >= warmup_frames) {
            auto image_processing_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                image_processing_end_time - image_processing_start_time);
            total_image_processing_ms += image_processing_duration.count();
            auto heatmap_extraction_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                heatmap_extraction_end_time - heatmap_extraction_start_time);
            total_heatmap_extraction_ms += heatmap_extraction_duration.count();
            auto pose_estimation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                pose_estimation_end_time - pose_estimation_start_time);
            total_pose_estimation_ms += pose_estimation_duration.count();
            auto overall_frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                overall_frame_end_time - overall_frame_start_time);
            total_duration_ms += overall_frame_duration.count();
            frames_processed_for_fps++;
        }
        // --- 可视化（示例：热身后每20帧保存一次）---
        if (frame_idx >= warmup_frames && frame_idx % 20 == 0) {
            if (fused_poses.defined() && fused_poses.numel() > 0 && 
                fused_poses.dim() == 4 && batch_tensors.size() == NUM_CAMERAS) {
                int num_joints_from_pose = fused_poses.size(2);
                std::vector<std::pair<int, int>> current_limbs = get_limbs_definition(num_joints_from_pose);
                for (int view_idx = 0; view_idx < NUM_CAMERAS; ++view_idx) {
                    std::string vis_output_prefix = "output_frame_" + std::to_string(frame_idx);
                    if (batch_tensors[view_idx].size(0) == 1) {
                        save_image_with_poses_cpp(
                            vis_output_prefix,
                            batch_tensors[view_idx].squeeze(0),
                            fused_poses,
                            cameras[view_idx],
                            resize_transform_tensor,
                            view_idx,
                            0.2f,
                            current_limbs,
                            ori_image_size_cfg
                        );
                    }
                }
            }
        }
    }
    // 等待图片读取线程结束
    if (image_loader_thread.joinable()) {
        image_loader_thread.join();
    }

    // 输出性能统计信息
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "完成图像序列处理." << std::endl;
    if (frames_processed_for_fps > 0) {
        double avg_overall_time_per_frame_ms = static_cast<double>(total_duration_ms) / frames_processed_for_fps;
        double avg_image_processing_ms = static_cast<double>(total_image_processing_ms) / frames_processed_for_fps;
        double avg_heatmap_extraction_ms = static_cast<double>(total_heatmap_extraction_ms) / frames_processed_for_fps;
        double avg_pose_estimation_ms = static_cast<double>(total_pose_estimation_ms) / frames_processed_for_fps;
        double fps = 1000.0 / avg_overall_time_per_frame_ms;

        std::cout << "用于计算FPS的处理帧数: " << frames_processed_for_fps << std::endl;
        std::cout << "总处理时间(热身后): " << total_duration_ms << " 毫秒" << std::endl;
        std::cout << "平均每帧处理时间(总体): " << std::fixed << std::setprecision(2) 
                 << avg_overall_time_per_frame_ms << " 毫秒" << std::endl;
        std::cout << "  平均图像处理时间: " << std::fixed << std::setprecision(2) 
                 << avg_image_processing_ms << " 毫秒" << std::endl;
        std::cout << "  平均热图提取时间: " << std::fixed << std::setprecision(2) 
                 << avg_heatmap_extraction_ms << " 毫秒" << std::endl;
        std::cout << "  平均姿态估计时间: " << std::fixed << std::setprecision(2) 
                 << avg_pose_estimation_ms << " 毫秒" << std::endl;
        std::cout << "FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
    } else {
        std::cout << "热身后处理的帧数不足，无法计算FPS." << std::endl;
    }
    std::cout << "-------------------------------------" << std::endl;

    return 0;
}