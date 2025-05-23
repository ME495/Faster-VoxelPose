cmake_minimum_required(VERSION 3.10)
project(FasterVoxelPoseDemo)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Ensure compile_commands.json is generated for IDEs

# Add compiler optimization flags based on build type
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # For GCC and Clang
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O2 -g -DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os -DNDEBUG")
elseif(MSVC)
  # For MSVC
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2 /Zi /DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /O1 /DNDEBUG")
endif()

# Enable link-time optimization for Release builds
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_CHECK_OUTPUT)
  if(LTO_SUPPORTED)
    message(STATUS "IPO / LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "IPO / LTO not supported: ${LTO_CHECK_OUTPUT}")
  endif()
endif()

# 源文件
set(SOURCES
    src/main.cpp
    src/types.cpp
    src/camera.cpp
    src/projection.cpp
    src/utils.cpp
    src/visualization.cpp
    src/image_processing.cpp
    src/tensorrt_inference.cpp
)

# set(CAFFE2_USE_CUDNN ON) # This line is generally for Caffe2, LibTorch handles its own cuDNN

# --- TensorRT ---
set(TENSORRT_ROOT "D:/opt/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/TensorRT-8.6.1.6")
# 查找 TensorRT 库
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_LIBRARY_RUNTIME nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY_INFER)
    message(STATUS "Found TensorRT: ${TENSORRT_INCLUDE_DIR}")
    set(TENSORRT_FOUND TRUE)
else()
    message(WARNING "TensorRT not found. You may need to set TENSORRT_ROOT or install TensorRT.")
    set(TENSORRT_FOUND FALSE)
endif()

# --- CUDA ---
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    enable_language(CUDA)
    message(STATUS "Found CUDA: ${CUDA_VERSION}")
endif()

# --- LibTorch --- 
# 根据构建类型选择不同的libtorch路径
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_PREFIX_PATH "D:/opt/libtorch-win-shared-with-deps-debug-2.5.1+cu118/libtorch")
else()
    set(CMAKE_PREFIX_PATH "D:/opt/libtorch-win-shared-with-deps-2.5.1+cu118/libtorch")
endif()
find_package(Torch REQUIRED)

# --- OpenCV --- 
# Ensure OpenCV is found. You might need to set OpenCV_DIR.
# export OpenCV_DIR=/path/to/opencv/build or pass -DOpenCV_DIR=/path/to/opencv/build
set(OpenCV_DIR "D:/opt/opencv-4.8.0/build")
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "LibTorch not found. Please set CMAKE_PREFIX_PATH correctly.")
endif()

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please set OpenCV_DIR correctly.")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

add_executable(faster_voxelpose_demo ${SOURCES})

target_link_libraries(faster_voxelpose_demo
    "${TORCH_LIBRARIES}"
    "${OpenCV_LIBS}"
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_RUNTIME}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

set_property(TARGET faster_voxelpose_demo PROPERTY CXX_STANDARD 17)

# Apply link-time optimization to the target if supported
if(LTO_SUPPORTED AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_property(TARGET faster_voxelpose_demo PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if (MSVC)
  # For Windows, need to copy Torch DLLs to the executable directory for execution
  # or ensure they are in the PATH.
  # This example does not automate DLL copying.
  
  # Enable multi-core compilation for MSVC
  target_compile_options(faster_voxelpose_demo PRIVATE /MP)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "nlohmann_json include dir: ${nlohmann_json_SOURCE_DIR}/include")