cmake_minimum_required(VERSION 3.10)
project(FasterVoxelPoseDemo)

# set(CMAKE_BUILD_TYPE Debug)

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
  
  # Enable OpenMP for GCC and Clang
  if(OPENMP_ENABLED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
elseif(MSVC)
  # For MSVC
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2 /Zi /DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /O1 /DNDEBUG")
  
  # Enable OpenMP for MSVC (will be handled per-target)
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
    src/types.cpp
    src/camera.cpp
    src/projection.cpp
    src/utils.cpp
    src/visualization.cpp
    src/image_processing.cpp
    src/tensorrt_inference.cpp
    src/person_tracker.cpp
    src/license_auth.cpp
)

# set(CAFFE2_USE_CUDNN ON) # This line is generally for Caffe2, LibTorch handles its own cuDNN

set(GIGELIB_ROOT "D:/opt/GIGECA/GigELib")
set(GIGELIB_INCLUDE_DIR "${GIGELIB_ROOT}/include")
set(GIGELIB_LIB_DIR ${GIGELIB_ROOT}/lib)

# 根据构建类型选择不同的库名
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug 版本
    set(GIGELIB_LIBS "${GIGELIB_LIB_DIR}/GigELibd_v141.lib")
else()
    # Release 版本
    set(GIGELIB_LIBS "${GIGELIB_LIB_DIR}/GigELib_v141.lib")
endif()

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

# --- OpenMP ---
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP C++: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
    set(OPENMP_ENABLED TRUE)
else()
    message(WARNING "OpenMP C++ support not found. Building without OpenMP optimization.")
    set(OPENMP_ENABLED FALSE)
endif()

# --- OpenSSL ---
set(OPENSSL_ROOT_DIR "D:/Program Files/OpenSSL-Win64")
set(OPENSSL_INCLUDE_DIR "${OPENSSL_ROOT_DIR}/include")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(OPENSSL_LIB_DIR "${OPENSSL_ROOT_DIR}/lib/VC/x64/MTd")
else()
    set(OPENSSL_LIB_DIR "${OPENSSL_ROOT_DIR}/lib/VC/x64/MT")
endif()
set(OPENSSL_LIBS 
    ${OPENSSL_LIB_DIR}/libssl.lib
    ${OPENSSL_LIB_DIR}/libcrypto.lib
)


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
    ${GIGELIB_INCLUDE_DIR}
    ${OPENSSL_INCLUDE_DIR}
)

add_executable(capture_from_files src/capture_from_files.cpp ${SOURCES})

target_link_libraries(capture_from_files
    "${TORCH_LIBRARIES}"
    "${OpenCV_LIBS}"
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_RUNTIME}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${OPENSSL_LIBS}
)

# 条件性添加OpenMP支持
if(OPENMP_ENABLED)
    target_link_libraries(capture_from_files OpenMP::OpenMP_CXX)
endif()

set_property(TARGET capture_from_files PROPERTY CXX_STANDARD 17)

# Apply link-time optimization to the target if supported
if(LTO_SUPPORTED AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_property(TARGET capture_from_files PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if (MSVC)
  # For Windows, need to copy Torch DLLs to the executable directory for execution
  # or ensure they are in the PATH.
  # This example does not automate DLL copying.
  
  # Enable multi-core compilation for MSVC
  target_compile_options(capture_from_files PRIVATE /MP)
  
  # Enable OpenMP for MSVC
  if(OPENMP_ENABLED)
    target_compile_options(capture_from_files PRIVATE /openmp)
  endif()
endif()

add_executable(capture_from_cameras src/capture_from_cameras.cpp ${SOURCES})

target_link_libraries(capture_from_cameras
    "${TORCH_LIBRARIES}"
    "${OpenCV_LIBS}"
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_RUNTIME}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${GIGELIB_LIBS}
    ${OPENSSL_LIBS}
)

# 条件性添加OpenMP支持
if(OPENMP_ENABLED)
    target_link_libraries(capture_from_cameras OpenMP::OpenMP_CXX)
endif()

set_property(TARGET capture_from_cameras PROPERTY CXX_STANDARD 17)

# Apply link-time optimization to the target if supported
if(LTO_SUPPORTED AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_property(TARGET capture_from_cameras PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if (MSVC)
  # For Windows, need to copy Torch DLLs to the executable directory for execution
  # or ensure they are in the PATH.
  # This example does not automate DLL copying.
  
  # Enable multi-core compilation for MSVC
  target_compile_options(capture_from_cameras PRIVATE /MP)
  
  # Enable OpenMP for MSVC
  if(OPENMP_ENABLED)
    target_compile_options(capture_from_cameras PRIVATE /openmp)
  endif()
endif()

# 授权码生成器
add_executable(license_generator src/license_generator.cpp)

target_link_libraries(license_generator ${OPENSSL_LIBS})

set_property(TARGET license_generator PROPERTY CXX_STANDARD 17)

if (MSVC)
  target_compile_options(license_generator PRIVATE /MP)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenMP enabled: ${OPENMP_ENABLED}")
if(OPENMP_ENABLED)
    message(STATUS "OpenMP version: ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
endif()
message(STATUS "nlohmann_json include dir: ${nlohmann_json_SOURCE_DIR}/include")
message(STATUS "OpenSSL include dir: ${OPENSSL_INCLUDE_DIR}")
message(STATUS "OpenSSL libraries: ${OPENSSL_LIBS}")

# ============================================
# 安装配置
# ============================================

# 设置安装目录
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

# 定义安装组件
set(VOXELPOSE_COMPONENT "VoxelPose")
set(LICENSE_COMPONENT "LicenseGenerator")

# ============================================
# VoxelPose 程序安装 (capture_from_files + capture_from_cameras)
# ============================================

# 安装可执行文件
install(TARGETS capture_from_files capture_from_cameras
    RUNTIME DESTINATION ${VOXELPOSE_COMPONENT}
    COMPONENT ${VOXELPOSE_COMPONENT}
)

# 安装LibTorch DLLs
if(WIN32)
    # 获取LibTorch DLL路径
    get_target_property(TORCH_DLLS torch IMPORTED_LOCATION_RELEASE)
    if(NOT TORCH_DLLS)
        get_target_property(TORCH_DLLS torch IMPORTED_LOCATION_DEBUG)
    endif()
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(TORCH_DLL_DIR "${CMAKE_PREFIX_PATH}/lib")
    else()
        set(TORCH_DLL_DIR "${CMAKE_PREFIX_PATH}/lib")
    endif()
    
    # 安装LibTorch DLLs
    install(FILES
        "${TORCH_DLL_DIR}/asmjit.dll"
        "${TORCH_DLL_DIR}/c10.dll"
        "${TORCH_DLL_DIR}/c10_cuda.dll"
        "${TORCH_DLL_DIR}/caffe2_nvrtc.dll"
        "${TORCH_DLL_DIR}/fbgemm.dll"
        "${TORCH_DLL_DIR}/fbjni.dll"
        "${TORCH_DLL_DIR}/libiomp5md.dll"
        "${TORCH_DLL_DIR}/libiompstubs5md.dll"
        "${TORCH_DLL_DIR}/torch.dll"
        "${TORCH_DLL_DIR}/torch_cpu.dll"
        "${TORCH_DLL_DIR}/torch_cuda.dll"
        "${TORCH_DLL_DIR}/uv.dll"
        DESTINATION ${VOXELPOSE_COMPONENT}
        COMPONENT ${VOXELPOSE_COMPONENT}
        OPTIONAL
    )
    
    # 安装OpenCV DLLs
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OPENCV_DLL_SUFFIX "d")
    else()
        set(OPENCV_DLL_SUFFIX "")
    endif()
    
    install(FILES
        "${OpenCV_DIR}/x64/vc16/bin/opencv_world480${OPENCV_DLL_SUFFIX}.dll"
        DESTINATION ${VOXELPOSE_COMPONENT}
        COMPONENT ${VOXELPOSE_COMPONENT}
        OPTIONAL
    )
    
    # 安装TensorRT DLLs
    install(FILES
        "${TENSORRT_ROOT}/lib/nvinfer.dll"
        "${TENSORRT_ROOT}/lib/nvinfer_plugin.dll"
        "${TENSORRT_ROOT}/lib/nvonnxparser.dll"
        "${TENSORRT_ROOT}/lib/nvparsers.dll"
        DESTINATION ${VOXELPOSE_COMPONENT}
        COMPONENT ${VOXELPOSE_COMPONENT}
        OPTIONAL
    )
    
    # 安装CUDA Runtime DLLs (需要根据实际CUDA版本调整)
    set(CUDA_BIN_DIR "${CUDA_TOOLKIT_ROOT_DIR}/bin")
    install(FILES
        "${CUDA_BIN_DIR}/cudart64_118.dll"
        "${CUDA_BIN_DIR}/curand64_10.dll"
        "${CUDA_BIN_DIR}/cusparse64_11.dll"
        "${CUDA_BIN_DIR}/cublas64_11.dll"
        "${CUDA_BIN_DIR}/cublasLt64_11.dll"
        "${CUDA_BIN_DIR}/cusolver64_11.dll"
        "${CUDA_BIN_DIR}/cusolverMg64_11.dll"
        "${CUDA_BIN_DIR}/cufft64_10.dll"
        "${CUDA_BIN_DIR}/nvrtc64_118_0.dll"
        "${CUDA_BIN_DIR}/nvrtc-builtins64_118.dll"
        DESTINATION ${VOXELPOSE_COMPONENT}
        COMPONENT ${VOXELPOSE_COMPONENT}
        OPTIONAL
    )
    
    # 安装OpenSSL DLLs
    install(FILES
        "${OPENSSL_ROOT_DIR}/bin/libssl-3-x64.dll"
        "${OPENSSL_ROOT_DIR}/bin/libcrypto-3-x64.dll"
        DESTINATION ${VOXELPOSE_COMPONENT}
        COMPONENT ${VOXELPOSE_COMPONENT}
        OPTIONAL
    )
    
    # 安装GigE Library DLLs (仅capture_from_cameras需要)
    install(FILES
        "${GIGELIB_LIB_DIR}/GigELib_v141.dll"
        DESTINATION ${VOXELPOSE_COMPONENT}
        COMPONENT ${VOXELPOSE_COMPONENT}
        OPTIONAL
    )
endif()

# ============================================
# License Generator 安装
# ============================================

# 安装可执行文件
install(TARGETS license_generator
    RUNTIME DESTINATION ${LICENSE_COMPONENT}
    COMPONENT ${LICENSE_COMPONENT}
)

# 安装OpenSSL DLLs (license_generator只需要OpenSSL)
if(WIN32)
    install(FILES
        "${OPENSSL_ROOT_DIR}/bin/libssl-3-x64.dll"
        "${OPENSSL_ROOT_DIR}/bin/libcrypto-3-x64.dll"
        DESTINATION ${LICENSE_COMPONENT}
        COMPONENT ${LICENSE_COMPONENT}
        OPTIONAL
    )
endif()

# ============================================
# 安装脚本和说明文件
# ============================================

# 创建批处理脚本
configure_file("${CMAKE_SOURCE_DIR}/scripts/install_voxelpose.bat.in" 
               "${CMAKE_BINARY_DIR}/install_voxelpose.bat" @ONLY)
configure_file("${CMAKE_SOURCE_DIR}/scripts/install_license_generator.bat.in" 
               "${CMAKE_BINARY_DIR}/install_license_generator.bat" @ONLY)

# 安装批处理脚本
install(FILES "${CMAKE_BINARY_DIR}/install_voxelpose.bat"
        DESTINATION ${VOXELPOSE_COMPONENT}
        COMPONENT ${VOXELPOSE_COMPONENT})

install(FILES "${CMAKE_BINARY_DIR}/install_license_generator.bat"
        DESTINATION ${LICENSE_COMPONENT}
        COMPONENT ${LICENSE_COMPONENT})

# 安装说明文档
install(FILES "${CMAKE_SOURCE_DIR}/README_INSTALL.md"
        DESTINATION .
        COMPONENT ${VOXELPOSE_COMPONENT})

install(FILES "${CMAKE_SOURCE_DIR}/README_INSTALL.md"
        DESTINATION .
        COMPONENT ${LICENSE_COMPONENT})

# 安装主安装脚本（在根目录）
install(FILES 
        "${CMAKE_SOURCE_DIR}/install.bat"
        "${CMAKE_SOURCE_DIR}/uninstall.bat"
        DESTINATION .
        COMPONENT ${VOXELPOSE_COMPONENT})

# ============================================
# 创建安装包配置
# ============================================

# 设置 CPack 变量
set(CPACK_PACKAGE_NAME "VoxelPose")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "VoxelPose 3D Human Pose Estimation System")
set(CPACK_PACKAGE_VENDOR "VoxelPose Team")

# 创建两个不同的安装包
set(CPACK_COMPONENTS_ALL ${VOXELPOSE_COMPONENT} ${LICENSE_COMPONENT})
set(CPACK_COMPONENT_VOXELPOSE_DISPLAY_NAME "VoxelPose Main Application")
set(CPACK_COMPONENT_VOXELPOSE_DESCRIPTION "VoxelPose pose estimation applications")
set(CPACK_COMPONENT_LICENSEGENERATOR_DISPLAY_NAME "License Generator")
set(CPACK_COMPONENT_LICENSEGENERATOR_DESCRIPTION "License generation utility")

# Windows平台使用NSIS或ZIP
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "VoxelPose ${CPACK_PACKAGE_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "VoxelPose")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
endif()

include(CPack)

# 自定义安装目标
add_custom_target(install_voxelpose
    COMMAND ${CMAKE_COMMAND} --build . --target install --config $<CONFIG> -- /p:Component=${VOXELPOSE_COMPONENT}
    DEPENDS capture_from_files capture_from_cameras
    COMMENT "Installing VoxelPose applications and dependencies"
)

add_custom_target(install_license_gen
    COMMAND ${CMAKE_COMMAND} --build . --target install --config $<CONFIG> -- /p:Component=${LICENSE_COMPONENT}
    DEPENDS license_generator
    COMMENT "Installing License Generator and dependencies"
)

message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "VoxelPose will be installed to: ${CMAKE_INSTALL_PREFIX}/${VOXELPOSE_COMPONENT}")
message(STATUS "License Generator will be installed to: ${CMAKE_INSTALL_PREFIX}/${LICENSE_COMPONENT}")
