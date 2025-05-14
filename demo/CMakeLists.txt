cmake_minimum_required(VERSION 3.10)
project(faster_voxelpose_inference)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找必要的包
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(faster_voxelpose_inference faster_voxelpose_inference.cpp)

# 链接库
target_link_libraries(faster_voxelpose_inference ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# 设置编译选项
if(MSVC)
  target_compile_options(faster_voxelpose_inference PRIVATE /W4)
else()
  target_compile_options(faster_voxelpose_inference PRIVATE -Wall -Wextra -pedantic)
endif()

# 设置包含目录
target_include_directories(faster_voxelpose_inference PRIVATE ${OpenCV_INCLUDE_DIRS})

# 安装目标
install(TARGETS faster_voxelpose_inference DESTINATION bin)

# 打印信息
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}") 