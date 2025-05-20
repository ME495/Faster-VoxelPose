#include "projection.h"
#include <cmath>

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
                              const torch::Tensor& k_dist, const torch::Tensor& p_dist) {
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
    sample_grid_normalized = sample_grid_normalized.view({1, 1, nbins, 2}); // grid_sample兼容格式
    sample_grid_normalized = torch::clamp(sample_grid_normalized, -1.1f, 1.1f);
    return sample_grid_normalized;
}

torch::Tensor project_pose_cpp(const torch::Tensor& poses_3d_person, const CameraParams& camera) {
    // 批量处理所有关节点，poses_3d_person形状为(NumJoints, 3)
    return project_point_cpp(
        poses_3d_person, camera.R, camera.T, camera.f, camera.c, camera.k_dist, camera.p_dist);
}

bool is_valid_coord_cpp(const torch::Tensor& pt, int width, int height) {
    float x = pt[0].item<float>();
    float y = pt[1].item<float>();
    return x >= 0 && x < width && y >= 0 && y < height;
} 