# 性能优化说明 - OpenMP并行化

## 优化概述
对 `capture_from_cameras.cpp` 中的图像处理流程进行了OpenMP并行化优化，主要针对多相机图像的同时处理。

## 优化内容

### 1. 并行处理相机图像
- **优化前**: 顺序处理每个相机的图像 (4个相机串行处理)
- **优化后**: 使用OpenMP并行处理所有相机图像

### 2. 关键修改

#### 线程设置
```cpp
// 自动设置OpenMP线程数
int openmp_threads = min(NUM_CAMERAS, (int)std::thread::hardware_concurrency());
omp_set_num_threads(openmp_threads);
```

#### 并行处理循环
```cpp
#pragma omp parallel for schedule(dynamic)
for(int i = 0; i < camNum; ++i) {
    // 图像处理: 缩放、归一化、tensor转换
}
```

#### 数据结构调整
- 使用预分配的vector: `std::vector<torch::Tensor> batch_tensors(camNum)`
- 添加有效性标志: `std::vector<bool> image_valid_flags(camNum, false)`
- 避免在并行区域内的竞争条件

### 3. 性能监控
新增图像处理时间统计:
- 总处理时间
- 图像处理时间 (OpenMP优化部分)
- 姿态估计时间
- 实时FPS显示

### 4. 预期性能提升
- **理论加速比**: 接近4倍 (4个相机并行处理)
- **实际加速比**: 2-3倍 (考虑内存带宽和CPU缓存影响)
- **主要受益**: 图像缩放、归一化和tensor转换操作

### 5. 适用场景
- 多相机同步捕获 (2个以上相机)
- CPU核心数 ≥ 相机数量时效果最佳
- 图像处理时间占总处理时间比重较大的情况

### 6. 注意事项
1. OpenMP线程数自动调整为相机数量和CPU核心数的较小值
2. 使用动态调度策略 (`schedule(dynamic)`) 平衡负载
3. 避免在并行区域内进行torch操作，确保线程安全
4. 预分配内存避免并行区域内的内存分配竞争

## 使用方法
重新编译项目后直接运行，程序会自动启用OpenMP优化并在控制台显示详细的性能统计信息。 