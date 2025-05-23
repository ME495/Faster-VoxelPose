# TensorRT 版本使用说明

本版本已将 backbone 模型从 TorchScript 格式改为 TensorRT 格式以提高推理速度。

## 依赖项

1. **TensorRT**: 需要安装 NVIDIA TensorRT 8.x
2. **CUDA**: 需要 CUDA 11.8
3. **LibTorch**: PyTorch C++ 库
4. **OpenCV**: 图像处理库

## 模型准备

1. 确保您的 backbone 模型已转换为 TensorRT 格式（.trt 文件）
2. 您可以使用项目中的 `run/trtexec.sh` 脚本来转换 ONNX 模型到 TensorRT：
   ```bash
   trtexec --onnx=backbone.onnx --saveEngine=backbone.trt --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16
   ```

## 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 使用方法

```bash
./faster_voxelpose_demo <backbone.trt> <model.pt> <calibration.json> <image_dir> <device> [num_frames]
```

### 参数说明

- `backbone.trt`: TensorRT 引擎文件路径（必须是 .trt 格式）
- `model.pt`: 主模型的 TorchScript 文件
- `calibration.json`: 相机标定文件
- `image_dir`: 图像序列目录
- `device`: 设备类型（cpu/cuda）
- `num_frames`: 可选，要处理的帧数

### 示例

```bash
./faster_voxelpose_demo \
    models/backbone.trt \
    models/faster_voxelpose.pt \
    calibration/calib.json \
    data/images \
    cuda \
    100
```

## 性能改进

使用 TensorRT 后，您应该看到以下性能改进：
- 更快的 backbone 推理速度（通常 2-5 倍加速）
- 更低的 GPU 内存使用
- 更高的整体 FPS

## 注意事项

1. TensorRT 引擎文件是特定于 GPU 架构的，在不同的 GPU 上可能需要重新生成
2. 确保输入图像的尺寸与生成 TensorRT 引擎时使用的尺寸一致
3. 程序现在要求 backbone 文件必须是 .trt 格式，不再支持 TorchScript backbone

## 故障排除

如果遇到问题：

1. 检查 TensorRT 是否正确安装
2. 确认 CUDA 版本兼容性
3. 验证 .trt 文件是否有效
4. 检查 GPU 内存是否足够
