#ifndef LICENSE_AUTH_H
#define LICENSE_AUTH_H

#include <string>

// 授权验证函数
bool verifyLicense(const std::string& licenseFile = "license.dat");

// 获取设备ID
std::string getDeviceId();

// 内部函数（可选导出）
std::string getMachineGuid();
std::string getDiskId();

#endif // LICENSE_AUTH_H 