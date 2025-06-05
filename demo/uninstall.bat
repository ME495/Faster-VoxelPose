@echo off
echo ========================================
echo VoxelPose Uninstallation Script
echo ========================================
echo.

:: 检查安装目录是否存在
if not exist "build\install" (
    echo No installation found in build\install directory.
    echo Nothing to uninstall.
    pause
    exit /b 0
)

:: 显示将要删除的内容
echo The following will be removed:
echo - build\install\VoxelPose\
echo - build\install\LicenseGenerator\
echo - All installed files and dependencies
echo.

:: 确认删除
set /p confirm="Are you sure you want to uninstall VoxelPose? (y/n): "
if /i not "%confirm%"=="y" (
    echo Uninstallation cancelled.
    pause
    exit /b 0
)

:: 删除安装目录
echo.
echo Removing installation files...

if exist "build\install\VoxelPose" (
    echo Removing VoxelPose applications...
    rmdir /s /q "build\install\VoxelPose"
)

if exist "build\install\LicenseGenerator" (
    echo Removing License Generator...
    rmdir /s /q "build\install\LicenseGenerator"
)

if exist "build\install\README_INSTALL.md" (
    del "build\install\README_INSTALL.md"
)

:: 检查是否还有其他文件
if exist "build\install" (
    dir /b "build\install" > nul 2>&1
    if not errorlevel 1 (
        echo Removing remaining installation directory...
        rmdir "build\install" 2>nul
    )
)

echo.
echo ========================================
echo Uninstallation Complete!
echo ========================================
echo.
echo All VoxelPose files have been removed.
echo.
echo Note: Build cache in 'build' directory is preserved.
echo To completely clean the project, you can manually delete the 'build' folder.
echo.

pause 