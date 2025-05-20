cmake_minimum_required(VERSION 3.10)
project(FasterVoxelPoseDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Ensure compile_commands.json is generated for IDEs

set(CMAKE_BUILD_TYPE "Debug")

# set(CAFFE2_USE_CUDNN ON) # This line is generally for Caffe2, LibTorch handles its own cuDNN

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

include_directories(${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(faster_voxelpose_demo faster_voxelpose_demo.cpp)

target_include_directories(faster_voxelpose_demo PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    include
)

target_link_libraries(faster_voxelpose_demo
    "${TORCH_LIBRARIES}"
    "${OpenCV_LIBS}"
)

set_property(TARGET faster_voxelpose_demo PROPERTY CXX_STANDARD 17)

if (MSVC)
  # For Windows, need to copy Torch DLLs to the executable directory for execution
  # or ensure they are in the PATH.
  # This example does not automate DLL copying.
endif()

message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "nlohmann_json include dir: ${nlohmann_json_SOURCE_DIR}/include") 