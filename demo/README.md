# Faster VoxelPose 推理代码

本目录包含使用 LibTorch 实现的 Faster VoxelPose 推理代码，可以从多视角图像中进行3D人体姿态估计。

## 依赖项

- LibTorch (C++ PyTorch API) >= 1.7.0
- OpenCV >= 4.2.0
- CMake >= 3.10
- C++14 兼容编译器

## 编译

1. 确保已安装 LibTorch 和 OpenCV

2. 创建构建目录并编译

```bash
mkdir build
cd build
cmake ..
make
```

## 模型准备

在使用推理代码前，需要将 PyTorch 模型导出为 TorchScript 格式。可以使用以下 Python 代码：

```python
import torch
from models.faster_voxelpose import get
from core.config import config

# 加载配置
config.defrost()
config.DEVICE = "cuda:0"  # 或 "cpu"
config.freeze()

# 创建模型
model = get(config)

# 加载预训练权重
checkpoint = torch.load("path/to/checkpoint.pth.tar")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# 导出主模型
scripted_model = torch.jit.script(model)
scripted_model.save("faster_voxelpose.pt")

# 导出骨干网络
backbone = model.backbone
scripted_backbone = torch.jit.script(backbone)
scripted_backbone.save("backbone.pt")
```

## 使用方法

```bash
./faster_voxelpose_inference <模型路径> <骨干网络路径> <相机参数文件> <图像目录> [帧索引] [输出文件]
```

参数说明：

- `<模型路径>`: TorchScript 格式的 Faster VoxelPose 模型文件路径
- `<骨干网络路径>`: TorchScript 格式的骨干网络模型文件路径
- `<相机参数文件>`: 相机参数文件路径（YAML 或 JSON 格式）
- `<图像目录>`: 多视角图像所在目录
- `[帧索引]`: 可选，要处理的帧索引，默认为 0
- `[输出文件]`: 可选，输出结果文件路径，默认为 "poses.csv"

## 相机参数文件格式

相机参数文件应为 YAML 或 JSON 格式，包含以下内容：

```yaml
num_cameras: 5
camera_0_K: !!opencv-matrix
  rows: 3
  cols: 3
  dt: f
  data: [fx_0, 0, cx_0, 0, fy_0, cy_0, 0, 0, 1]
camera_0_R: !!opencv-matrix
  rows: 3
  cols: 3
  dt: f
  data: [r11_0, r12_0, r13_0, r21_0, r22_0, r23_0, r31_0, r32_0, r33_0]
camera_0_t: !!opencv-matrix
  rows: 3
  cols: 1
  dt: f
  data: [tx_0, ty_0, tz_0]
camera_0_distCoeffs: !!opencv-matrix
  rows: 1
  cols: 5
  dt: f
  data: [k1_0, k2_0, p1_0, p2_0, k3_0]
# ... 其他相机参数
```

## 图像目录结构

图像目录应包含多视角图像，文件名格式为：

```
camera_<camera_idx>_<frame_idx>.jpg
```

例如：
- camera_0_0.jpg
- camera_1_0.jpg
- camera_2_0.jpg
- ...

## 输出格式

输出文件为 CSV 格式，包含以下列：

- person_id: 人体ID
- joint_id: 关节ID
- x: X坐标
- y: Y坐标
- z: Z坐标

## 示例

```bash
./faster_voxelpose_inference models/faster_voxelpose.pt models/backbone.pt configs/cameras.yaml data/images 0 results.csv
```

## 自定义配置

如需修改配置参数（如最大人数、关节数量等），可以在 `faster_voxelpose_inference.cpp` 文件中修改 `FasterVoxelPoseConfig` 结构体的默认值。 