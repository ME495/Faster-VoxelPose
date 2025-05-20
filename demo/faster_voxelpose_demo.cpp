#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>   // For std::ifstream
#include <chrono>    // For timing
#include <filesystem> // For directory iteration (C++17)
#include <algorithm> // For std::sort, std::min
#include <iomanip>   // For std::fixed, std::setprecision

namespace fs = std::filesystem;
using json = nlohmann::json;

// Helper struct for Camera Parameters
struct CameraParams {
    torch::Tensor R; // Rotation matrix (3x3), world to camera
    torch::Tensor T; // Translation vector (3x1), camera center in world coordinates (C_w)
                     // such that x_cam = R @ (x_world - T)
    torch::Tensor K_intrinsic; // Intrinsics matrix (3x3)
    torch::Tensor f; // Focal length (fx, fy) (2x1)
    torch::Tensor c; // Principal point (cx, cy) (2x1)
    torch::Tensor k_dist; // Radial distortion coefficients (k1, k2, k3) (3x1)
    torch::Tensor p_dist; // Tangential distortion coefficients (p1, p2) (2x1)
    std::string id; // camera id like "44310001"

    CameraParams() :
        R(torch::eye(3, torch::kFloat32)),
        T(torch::zeros({3,1}, torch::kFloat32)),
        K_intrinsic(torch::eye(3, torch::kFloat32)),
        f(torch::ones({2,1}, torch::kFloat32) * 1000.0),
        c(torch::zeros({2,1}, torch::kFloat32)),
        k_dist(torch::zeros({3,1}, torch::kFloat32)),
        p_dist(torch::zeros({2,1}, torch::kFloat32)),
        id("unknown")
    {}
};

// Function to list image files in a directory, sorted alphabetically
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
        std::cerr << "Filesystem error: " << e.what() << " while accessing path: " << dir_path << std::endl;
    }
    std::sort(files.begin(), files.end());
    return files;
}

// Function to get affine transform for resizing (mimics utils.transforms.get_affine_transform behavior for r=0, shift=0)
torch::Tensor get_resize_affine_transform_cpp(const std::vector<int>& ori_size, 
                                              const std::vector<int>& new_size, 
                                              torch::Device device) {
    // For r=0, shift=0, this simplifies significantly.
    // The Python version uses a scale factor of 200 for intermediate `scale_tmp`,
    // which is a convention in that codebase. We need to be careful if it affects the matrix ratios.
    // Let's assume direct scaling based on original and new size for simplicity in a demo.
    // If `get_affine_transform` in Python produces a matrix not equivalent to simple scaling + translation 
    // due to its internal logic (like `scale_tmp`), this C++ version might need refinement.

    float scale_x = static_cast<float>(new_size[0]) / static_cast<float>(ori_size[0]);
    float scale_y = static_cast<float>(new_size[1]) / static_cast<float>(ori_size[1]);

    // Create a 2x3 affine matrix for cv::warpAffine or similar PyTorch operations
    // [ scale_x, 0,       translation_x ]
    // [ 0,       scale_y, translation_y ]
    // For simple resize, translation is 0 if scaling from origin.
    // PyTorch affine_grid and grid_sample expect normalized coordinates, so the transform should map pixel coords.
    // However, the `resize_transform` in `validate_ts.py` is used by `affine_transform_pts_cuda` 
    // which expects a 2x3 matrix.

    // The Python's `get_affine_transform` with rot=0, shift=[0,0]
    // src[0,:] = center
    // dst[0,:] = [dst_w * 0.5, dst_h * 0.5]
    // Effectively, it maps the center of the source to the center of the destination,
    // and scales based on `scale_tmp` then maps points.
    // For simple resize from (0,0) to (new_w, new_h) from (0,0) to (ori_w, ori_h):
    // x' = x * new_w / ori_w
    // y' = y * new_h / ori_h
    // This corresponds to an affine matrix:
    // [[new_w/ori_w, 0, 0],
    //  [0, new_h/ori_h, 0]]

    // torch::Tensor trans = torch::zeros({2, 3}, torch::kFloat32, device);
    // Fix for the torch::zeros error by explicitly specifying the options using torch::TensorOptions.  
    torch::Tensor trans = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    trans[0][0] = scale_x;
    trans[1][1] = scale_y;
    return trans;
}

std::vector<CameraParams> load_cameras_from_json(const std::string& json_path, torch::Device device) {
    std::vector<CameraParams> cameras;
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        std::cerr << "Error opening calibration file: " << json_path << std::endl;
        return cameras;
    }
    json calib_data;
    try {
        calib_data = json::parse(ifs);
    } catch (json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
        return cameras;
    }

    for (auto& [cam_id_str, cam_json] : calib_data.items()) {
        CameraParams cam_p;
        cam_p.id = cam_id_str;
        try {
            std::vector<float> k_vec = cam_json["k"].get<std::vector<float>>();
            cam_p.f = torch::tensor({k_vec[0], k_vec[1]}, torch::kFloat32).to(device).view({2,1});
            cam_p.c = torch::tensor({k_vec[2], k_vec[3]}, torch::kFloat32).to(device).view({2,1});
            cam_p.K_intrinsic = torch::eye(3, torch::kFloat32).to(device);
            cam_p.K_intrinsic[0][0] = k_vec[0];
            cam_p.K_intrinsic[1][1] = k_vec[1];
            cam_p.K_intrinsic[0][2] = k_vec[2];
            cam_p.K_intrinsic[1][2] = k_vec[3];

            std::vector<float> d_vec = cam_json["d"].get<std::vector<float>>();
            cam_p.k_dist = torch::tensor({d_vec[0], d_vec[1], d_vec[4]}, torch::kFloat32).to(device).view({3,1});
            cam_p.p_dist = torch::tensor({d_vec[2], d_vec[3]}, torch::kFloat32).to(device).view({2,1});

            std::vector<float> p_mat_vec = cam_json["p"].get<std::vector<float>>();
            torch::Tensor P_proj = torch::tensor(p_mat_vec, torch::kFloat32).to(device).view({3, 4});
            torch::Tensor K_inv = torch::inverse(cam_p.K_intrinsic);
            torch::Tensor RT_compound = torch::mm(K_inv, P_proj);
            cam_p.R = RT_compound.slice(1, 0, 3);
            torch::Tensor t_world_to_cam = RT_compound.slice(1, 3, 4);
            cam_p.T = -torch::mm(cam_p.R.transpose(0,1), t_world_to_cam);
            cameras.push_back(cam_p);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing camera " << cam_id_str << ": " << e.what() << std::endl;
        }
    }
    // Sort cameras by ID string to ensure consistent order for processing
    std::sort(cameras.begin(), cameras.end(), [](const CameraParams& a, const CameraParams& b) {
        return a.id < b.id;
    });
    return cameras;
}

// Forward declarations (some already exist)
torch::Tensor compute_grid_cpp(const std::vector<float>& boxSize, const std::vector<float>& boxCenter, const std::vector<int>& nBins, torch::Device device);
torch::Tensor project_point_cpp(torch::Tensor x, const torch::Tensor& R, const torch::Tensor& T, const torch::Tensor& f, const torch::Tensor& c, const torch::Tensor& k, const torch::Tensor& p);
torch::Tensor affine_transform_pts_cpp(torch::Tensor pts, torch::Tensor t);
torch::Tensor project_grid_cpp(torch::Tensor grid, const std::vector<int>& ori_image_size_vec, const std::vector<int>& image_size_vec, const CameraParams& camera, int heatmap_w, int heatmap_h, int nbins, torch::Tensor resize_transform, torch::Device device);
torch::Tensor load_and_preprocess_image(const std::string& image_path, const std::vector<int>& target_image_size, torch::Device device);

int main(int argc, const char* argv[]) {
    if (argc < 6) { // Prog_name, backbone, model, calib.json, image_base_dir, device
        std::cerr << "Usage: faster_voxelpose_demo <path_to_scripted_backbone.pt> \
                     <path_to_scripted_model.pt> <path_to_calibration.json> \
                     <image_sequence_base_dir> <device (cpu/cuda)> [num_frames_to_process]" << std::endl;
        return -1;
    }

    std::string backbone_path = argv[1];
    std::string model_path = argv[2];
    std::string calib_json_path = argv[3];
    std::string image_base_dir = argv[4];
    std::string device_str = argv[5];
    int num_frames_to_process_arg = -1; // Process all if not specified
    if (argc > 6) {
        try {
            num_frames_to_process_arg = std::stoi(argv[6]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid number for num_frames_to_process: " << argv[6] << std::endl;
            return -1;
        }
    }

    torch::DeviceType device_type;
    if (device_str == "cuda" && torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        std::cout << "Using CUDA device." << std::endl;
    } else {
        device_type = torch::kCPU;
        std::cout << "Using CPU device." << std::endl;
    }
    torch::Device device(device_type);

    // --- Configuration values from configs/custom/jln64.yaml --- 
    std::vector<int> ori_image_size_cfg = {2048, 1544};      
    std::vector<int> image_size_cfg = {1024, 784};           
    std::vector<int> heatmap_size_cfg = {256, 200};          
    const int NUM_CAMERAS = 4; // Defined by requirement

    std::vector<float> space_size_cfg = {8000.0f, 8000.0f, 2000.0f}; 
    std::vector<float> space_center_cfg = {0.0f, -300.0f, 800.0f};  
    std::vector<int> voxels_per_axis_cfg = {80, 80, 20};           
    std::vector<float> individual_space_size_cfg = {2000.0f, 2000.0f, 2000.0f};
    std::vector<int> ind_voxels_per_axis_cfg = {64, 64, 64};      
    // --- End Configuration ---

    // Load Camera Parameters
    std::vector<CameraParams> cameras = load_cameras_from_json(calib_json_path, device);
    if (cameras.size() < NUM_CAMERAS) {
        std::cerr << "Error: Loaded " << cameras.size() << " cameras, expected " << NUM_CAMERAS << std::endl;
        return -1;
    }
    // Ensure we only use NUM_CAMERAS, in case json has more but sorted by ID.
    if (cameras.size() > NUM_CAMERAS) {
         cameras.resize(NUM_CAMERAS);
    }

    // List image files for each camera
    std::vector<std::vector<std::string>> camera_image_files(NUM_CAMERAS);
    size_t min_frames = std::string::npos;

    for (int i = 0; i < NUM_CAMERAS; ++i) {
        fs::path cam_dir = fs::path(image_base_dir) / cameras[i].id;
        camera_image_files[i] = list_image_files(cam_dir.string());
        if (camera_image_files[i].empty()) {
            std::cerr << "Error: No image files found in directory: " << cam_dir.string() << std::endl;
            return -1;
        }
        if (min_frames == std::string::npos || camera_image_files[i].size() < min_frames) {
            min_frames = camera_image_files[i].size();
        }
        std::cout << "Found " << camera_image_files[i].size() << " images for camera " << cameras[i].id << std::endl;
    }

    if (min_frames == 0 || min_frames == std::string::npos) {
        std::cerr << "Error: No frames to process or inconsistent frame counts across cameras." << std::endl;
        return -1;
    }
    std::cout << "Processing a sequence of " << min_frames << " synchronized frames." << std::endl;

    int frames_to_run = (num_frames_to_process_arg > 0 && num_frames_to_process_arg < min_frames) ? num_frames_to_process_arg : min_frames;
    std::cout << "Will run for " << frames_to_run << " frames." << std::endl;

    torch::jit::script::Module backbone_module, model_module;
    try {
        backbone_module = torch::jit::load(backbone_path, device);
        backbone_module.eval();
        model_module = torch::jit::load(model_path, device);
        model_module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model(s):\n" << e.what() << std::endl;
        return -1;
    }

    // --- Precompute Grids (these are static for the sequence) ---
    torch::Tensor resize_transform_tensor = get_resize_affine_transform_cpp(ori_image_size_cfg, image_size_cfg, device);
    torch::Tensor grid_coarse = compute_grid_cpp(space_size_cfg, space_center_cfg, voxels_per_axis_cfg, device);
    int nbins_coarse = voxels_per_axis_cfg[0] * voxels_per_axis_cfg[1] * voxels_per_axis_cfg[2];
    std::vector<torch::Tensor> sample_grids_coarse_list_static;

    torch::Tensor world_space_size_tensor = torch::tensor(space_size_cfg, device);
    torch::Tensor ind_space_size_tensor = torch::tensor(individual_space_size_cfg, device);
    torch::Tensor ind_voxels_per_axis_tensor = torch::tensor(ind_voxels_per_axis_cfg, torch::kInt).to(device);
    torch::Tensor fine_voxels_per_axis_float = (world_space_size_tensor / ind_space_size_tensor * (ind_voxels_per_axis_tensor.to(torch::kFloat) - 1.0f)) + 1.0f;
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
        sample_grids_coarse_list_static.push_back(sg_coarse_cam.view({voxels_per_axis_cfg[0], voxels_per_axis_cfg[1], voxels_per_axis_cfg[2], 2}));

        torch::Tensor sg_fine_cam = project_grid_cpp(grid_fine, ori_image_size_cfg, image_size_cfg, 
                                                      cameras[i], heatmap_size_cfg[0], heatmap_size_cfg[1],
                                                      nbins_fine, resize_transform_tensor, device);
        sample_grids_fine_list_static.push_back(sg_fine_cam.view({fine_voxels_per_axis_cfg[0], fine_voxels_per_axis_cfg[1], fine_voxels_per_axis_cfg[2], 2}));
    }
    torch::Tensor final_sample_grid_coarse = torch::stack(sample_grids_coarse_list_static, 0).unsqueeze(0);
    torch::Tensor final_sample_grid_fine = torch::stack(sample_grids_fine_list_static, 0).unsqueeze(0);
    // --- End Precompute Grids ---

    torch::NoGradGuard no_grad;
    long long total_duration_ms = 0;
    int frames_processed_for_fps = 0;
    int warmup_frames = 10;

    std::cout << "Starting processing loop for " << frames_to_run << " frames..." << std::endl;

    for (int frame_idx = 0; frame_idx < frames_to_run; ++frame_idx) {
        if (frame_idx % 50 == 0 || frame_idx == frames_to_run -1) {
             std::cout << "Processing frame " << frame_idx + 1 << "/" << frames_to_run << std::endl;
        }

        auto frame_start_time = std::chrono::high_resolution_clock::now();

        std::vector<torch::Tensor> current_frame_heatmaps_list;
        for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
            torch::Tensor input_tensor = load_and_preprocess_image(camera_image_files[cam_idx][frame_idx], image_size_cfg, device);
            if (input_tensor.numel() == 0) { std::cerr << "Skipping frame due to image load error." << std::endl; continue; }
            torch::Tensor heatmaps_view = backbone_module.forward({input_tensor}).toTensor();
            current_frame_heatmaps_list.push_back(heatmaps_view);
        }
        std::cout << "current_frame_heatmaps_list: " << current_frame_heatmaps_list.size() << std::endl;
        if(current_frame_heatmaps_list.size() != NUM_CAMERAS) { 
            std::cerr << "Error: Not enough heatmaps generated for frame " << frame_idx << std::endl; continue; 
        }
        torch::Tensor input_heatmaps_for_model = torch::cat(current_frame_heatmaps_list, 0).unsqueeze(0);

        std::vector<torch::jit::IValue> model_inputs;
        model_inputs.push_back(input_heatmaps_for_model);
        model_inputs.push_back(final_sample_grid_coarse); 
        model_inputs.push_back(final_sample_grid_fine);

        torch::Tensor fused_poses = model_module.forward(model_inputs).toTensor();
        
        if (device_type == torch::kCUDA) {
            torch::cuda::synchronize(); // Ensure all CUDA ops are done before timing
        }
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        
        if (frame_idx >= warmup_frames) { // Start timing after warmup
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end_time - frame_start_time);
            total_duration_ms += frame_duration.count();
            frames_processed_for_fps++;
        }
        // Optionally, do something with fused_poses here (e.g., visualization, saving)
    }

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Finished processing image sequence." << std::endl;
    if (frames_processed_for_fps > 0) {
        double avg_time_per_frame_ms = static_cast<double>(total_duration_ms) / frames_processed_for_fps;
        double fps = 1000.0 / avg_time_per_frame_ms;
        std::cout << "Frames processed for FPS: " << frames_processed_for_fps << std::endl;
        std::cout << "Total processing time (after warmup): " << total_duration_ms << " ms" << std::endl;
        std::cout << "Average time per frame: " << std::fixed << std::setprecision(2) << avg_time_per_frame_ms << " ms" << std::endl;
        std::cout << "FPS: " << std::fixed << std::setprecision(2) << fps << std::endl;
    } else {
        std::cout << "Not enough frames processed after warmup to calculate FPS." << std::endl;
    }
    std::cout << "-------------------------------------" << std::endl;

    return 0;
}

// --- Implementations for Helper Functions --- 
// (compute_grid_cpp, project_point_cpp, affine_transform_pts_cpp, project_grid_cpp as previously defined)

torch::Tensor compute_grid_cpp(const std::vector<float>& boxSize_vec, 
                               const std::vector<float>& boxCenter_vec, 
                               const std::vector<int>& nBins_vec, 
                               torch::Device device) {
    torch::Tensor boxSize = torch::tensor(boxSize_vec, device);
    torch::Tensor boxCenter = torch::tensor(boxCenter_vec, device);
    torch::Tensor grid1Dx = torch::linspace(-boxSize[0].item<float>() / 2.0f, boxSize[0].item<float>() / 2.0f, nBins_vec[0], device);
    torch::Tensor grid1Dy = torch::linspace(-boxSize[1].item<float>() / 2.0f, boxSize[1].item<float>() / 2.0f, nBins_vec[1], device);
    torch::Tensor grid1Dz = torch::linspace(-boxSize[2].item<float>() / 2.0f, boxSize[2].item<float>() / 2.0f, nBins_vec[2], device);
    std::vector<torch::Tensor> grids_v = torch::meshgrid({grid1Dx + boxCenter[0].item<float>(), 
                                                          grid1Dy + boxCenter[1].item<float>(), 
                                                          grid1Dz + boxCenter[2].item<float>()}, "ij");
    torch::Tensor gridx = grids_v[0].contiguous().view({-1, 1});
    torch::Tensor gridy = grids_v[1].contiguous().view({-1, 1});
    torch::Tensor gridz = grids_v[2].contiguous().view({-1, 1});
    return torch::cat({gridx, gridy, gridz}, 1);
}

torch::Tensor project_point_cpp(torch::Tensor x, 
                              const torch::Tensor& R, const torch::Tensor& T, 
                              const torch::Tensor& f, const torch::Tensor& c, 
                              const torch::Tensor& k_dist, const torch::Tensor& p_dist
                              ) {
    torch::Tensor x_minus_T = x.transpose(0, 1) - T;
    torch::Tensor xcam = torch::mm(R, x_minus_T);
    torch::Tensor y = xcam.slice(0, 0, 2) / (xcam.slice(0, 2, 3) + 1e-5);
    torch::Tensor r_sq = torch::sum(y * y, 0);
    torch::Tensor d_radial = 1.0f + k_dist[0] * r_sq + k_dist[1] * r_sq * r_sq + k_dist[2] * r_sq * r_sq * r_sq;
    torch::Tensor y0 = y.slice(0, 0, 1);
    torch::Tensor y1 = y.slice(0, 1, 2);
    torch::Tensor u_tangential = 2 * p_dist[0] * y0 * y1 + p_dist[1] * (r_sq + 2 * y0 * y0);
    torch::Tensor v_tangential = 2 * p_dist[1] * y0 * y1 + p_dist[0] * (r_sq + 2 * y1 * y1);
    torch::Tensor u = y0 * d_radial + u_tangential;
    torch::Tensor v = y1 * d_radial + v_tangential;
    torch::Tensor y_distorted = torch::cat({u, v}, 0);
    torch::Tensor ypixel = f * y_distorted + c;
    return ypixel.transpose(0, 1);
}

torch::Tensor affine_transform_pts_cpp(torch::Tensor pts, torch::Tensor t) {
    int64_t npts = pts.size(0);
    torch::Tensor ones = torch::ones({npts, 1}, pts.options());
    torch::Tensor pts_homo = torch::cat({pts, ones}, 1);
    torch::Tensor out_homo = torch::mm(t, pts_homo.transpose(0, 1)); 
    return out_homo.slice(0,0,2).transpose(0,1);
}

torch::Tensor project_grid_cpp(torch::Tensor grid, 
                             const std::vector<int>& ori_image_size_vec,
                             const std::vector<int>& image_size_vec,
                             const CameraParams& camera, 
                             int heatmap_w, int heatmap_h, 
                             int nbins, 
                             torch::Tensor resize_transform, 
                             torch::Device device) {
    torch::Tensor xy = project_point_cpp(grid, camera.R, camera.T, camera.f, camera.c, camera.k_dist, camera.p_dist);
    float clamp_max = static_cast<float>(std::max(ori_image_size_vec[0], ori_image_size_vec[1]));
    xy = torch::clamp(xy, -1.0f, clamp_max);
    xy = affine_transform_pts_cpp(xy, resize_transform);
    torch::Tensor wh_heatmap = torch::tensor({static_cast<float>(heatmap_w), static_cast<float>(heatmap_h)}, device);
    torch::Tensor wh_image_size = torch::tensor({static_cast<float>(image_size_vec[0]), static_cast<float>(image_size_vec[1])}, device);
    xy = xy * wh_heatmap / wh_image_size;
    torch::Tensor wh_minus_1 = torch::tensor({static_cast<float>(heatmap_w - 1), static_cast<float>(heatmap_h - 1)}, device);
    torch::Tensor sample_grid_normalized = (xy / wh_minus_1) * 2.0f - 1.0f;
    sample_grid_normalized = sample_grid_normalized.view({1, 1, nbins, 2}); // This shape is for F.grid_sample compatibility
    sample_grid_normalized = torch::clamp(sample_grid_normalized, -1.1f, 1.1f);
    return sample_grid_normalized;
}

// load_and_preprocess_image function remains as previously defined.

torch::Tensor load_and_preprocess_image(const std::string& image_path, 
                                        const std::vector<int>& target_image_size, // {width, height}
                                        torch::Device device) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error loading image: " << image_path << std::endl;
        return torch::Tensor();
    }

    // Resize image to target_image_size (this is what backbone expects)
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(target_image_size[0], target_image_size[1]));

    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
    resized_img.convertTo(resized_img, CV_32FC3, 1.0f / 255.0f);

    torch::Tensor img_tensor = torch::from_blob(resized_img.data, {resized_img.rows, resized_img.cols, 3}, torch::kFloat32).to(device);
    img_tensor = img_tensor.permute({2, 0, 1}); // HWC to CHW

    // Normalization (values from validate_ts.py)
    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}, device).view({3, 1, 1});
    torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}, device).view({3, 1, 1});
    img_tensor = (img_tensor - mean) / std;

    return img_tensor.unsqueeze(0); // Add batch dimension
} 