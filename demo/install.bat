@echo off
echo ========================================
echo VoxelPose Automatic Installation Script
echo ========================================
echo.

:: 检查是否在正确的目录
if not exist "CMakeLists.txt" (
    echo Error: CMakeLists.txt not found!
    echo Please run this script from the project root directory.
    pause
    exit /b 1
)

:: 设置构建类型
set BUILD_TYPE=Release
if "%1"=="debug" set BUILD_TYPE=Debug
if "%1"=="Debug" set BUILD_TYPE=Debug

echo Build Type: %BUILD_TYPE%
echo.

:: 创建构建目录
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

:: 配置项目
echo Configuring project...
cmake .. -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if errorlevel 1 (
    echo Configuration failed!
    pause
    exit /b 1
)

:: 编译项目
echo.
echo Building project...
cmake --build . --config %BUILD_TYPE%
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

:: 询问安装选项
echo.
echo ========================================
echo Installation Options:
echo 1. Install VoxelPose Main Applications
echo 2. Install License Generator
echo 3. Install Both (Recommended)
echo 4. Skip Installation
echo ========================================
set /p choice="Please choose (1-4): "

if "%choice%"=="1" goto install_voxelpose
if "%choice%"=="2" goto install_license
if "%choice%"=="3" goto install_both
if "%choice%"=="4" goto skip_install
goto invalid_choice

:install_voxelpose
echo.
echo Installing VoxelPose Main Applications...
cmake --build . --target install_voxelpose --config %BUILD_TYPE%
goto installation_complete

:install_license
echo.
echo Installing License Generator...
cmake --build . --target install_license_gen --config %BUILD_TYPE%
goto installation_complete

:install_both
echo.
echo Installing VoxelPose Main Applications...
cmake --build . --target install_voxelpose --config %BUILD_TYPE%
echo.
echo Installing License Generator...
cmake --build . --target install_license_gen --config %BUILD_TYPE%
goto installation_complete

:skip_install
echo Installation skipped.
goto end

:invalid_choice
echo Invalid choice! Please run the script again.
pause
exit /b 1

:installation_complete
echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Installation Directory: %CD%\install
echo.
echo VoxelPose Applications: install\VoxelPose\
echo License Generator: install\LicenseGenerator\
echo.
echo Next Steps:
echo 1. Generate a license using License Generator
echo 2. Copy license.dat to VoxelPose directory
echo 3. Run VoxelPose applications
echo.
echo For detailed instructions, see README_INSTALL.md
echo.

:: 询问是否打开安装目录
set /p open_dir="Open installation directory? (y/n): "
if /i "%open_dir%"=="y" (
    explorer install
)

:end
echo.
pause 