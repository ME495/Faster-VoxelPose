# VoxelPose 安装指南

## 概述

VoxelPose 系统包含两个主要组件：

1. **VoxelPose 主程序** - 人体姿态估计应用
2. **License Generator** - 授权码生成工具

## 系统要求

- Windows 10/11 (x64)
- NVIDIA GPU (支持CUDA 11.8)
- Visual Studio 2019/2022 运行时库

## 安装方法

### 方法1: 使用 CMake 安装

```bash
# 构建项目
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build . --config Release

# 安装 VoxelPose 主程序
cmake --build . --target install_voxelpose --config Release

# 安装 License Generator
cmake --build . --target install_license_gen --config Release
```

### 方法2: 手动安装

1. 编译项目
2. 复制可执行文件和依赖DLL到目标目录

## 安装目录结构

```
install/
├── VoxelPose/                  # VoxelPose 主程序目录
│   ├── capture_from_files.exe
│   ├── capture_from_cameras.exe
│   ├── *.dll                   # 所有依赖库
│   └── install_voxelpose.bat   # 安装说明脚本
└── LicenseGenerator/           # 授权码生成器目录
    ├── license_generator.exe
    ├── libssl-3-x64.dll
    ├── libcrypto-3-x64.dll
    └── install_license_generator.bat
```

## 依赖库清单

### VoxelPose 主程序依赖：

- **LibTorch DLLs:**
  - torch.dll, torch_cpu.dll, torch_cuda.dll
  - c10.dll, c10_cuda.dll
  - asmjit.dll, fbgemm.dll, fbjni.dll
  - libiomp5md.dll, libiompstubs5md.dll
  - caffe2_nvrtc.dll, uv.dll

- **OpenCV DLLs:**
  - opencv_world480.dll (Release)
  - opencv_world480d.dll (Debug)

- **TensorRT DLLs:**
  - nvinfer.dll, nvinfer_plugin.dll
  - nvonnxparser.dll, nvparsers.dll

- **CUDA Runtime DLLs:**
  - cudart64_118.dll
  - cublas64_11.dll, cublasLt64_11.dll
  - cusparse64_11.dll, cusolver64_11.dll
  - curand64_10.dll, cufft64_10.dll
  - nvrtc64_118_0.dll, nvrtc-builtins64_118.dll

- **OpenSSL DLLs:**
  - libssl-3-x64.dll
  - libcrypto-3-x64.dll

- **GigE Camera DLLs:**
  - GigELib_v141.dll

### License Generator 依赖：

- **OpenSSL DLLs:**
  - libssl-3-x64.dll
  - libcrypto-3-x64.dll

## 使用方法

### 1. 生成授权码

```bash
cd install/LicenseGenerator

# 生成30天授权码
license_generator.exe 30

# 生成1年授权码
license_generator.exe 365

# 验证授权码
license_generator.exe verify
```

### 2. 运行 VoxelPose

```bash
cd install/VoxelPose

# 确保 license.dat 文件存在
copy ..\LicenseGenerator\license.dat .

# 从文件处理
capture_from_files.exe backbone.trt model.pt calibration.json images_dir cuda

# 从相机实时处理
capture_from_cameras.exe backbone.trt model.pt calibration.json cuda
```

## 故障排除

### 常见问题：

1. **缺少 DLL 错误**
   - 确保所有依赖 DLL 都在可执行文件同一目录
   - 检查系统是否安装了 Visual C++ Redistributable

2. **CUDA 错误**
   - 确保系统安装了正确版本的 CUDA (11.8)
   - 检查 NVIDIA 驱动版本

3. **授权失败**
   - 确保 license.dat 文件存在于程序目录
   - 检查授权码是否过期
   - 验证设备 ID 是否匹配

### 环境变量设置：

如果遇到 DLL 加载问题，可以将依赖库目录添加到 PATH：

```batch
set PATH=%PATH%;C:\path\to\cuda\bin
set PATH=%PATH%;C:\path\to\tensorrt\lib
```

## 技术支持

如有问题，请联系技术支持并提供：
- 错误信息截图
- 系统环境信息
- 使用的命令行参数 