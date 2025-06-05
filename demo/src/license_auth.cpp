#include "license_auth.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <windows.h>
#include <winioctl.h>  // 添加存储设备控制接口
#include <intrin.h>
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>

// 密钥 - 与生成器保持一致
const std::string SECRET_KEY = "VoxelPoseAuthKey2024SecretKey32B";

// 获取机器GUID
std::string getMachineGuid() {
    HKEY hKey;
    std::string machineGuid;
    
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, 
                     "SOFTWARE\\Microsoft\\Cryptography", 
                     0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        
        char buffer[256];
        DWORD bufferSize = sizeof(buffer);
        DWORD type;
        
        if (RegQueryValueExA(hKey, "MachineGuid", NULL, &type, 
                            (LPBYTE)buffer, &bufferSize) == ERROR_SUCCESS) {
            machineGuid = std::string(buffer);
        }
        RegCloseKey(hKey);
    }
    return machineGuid;
}

// 获取磁盘序列号
std::string getDiskId() {
    std::string diskSerial;
    
    // 获取系统盘（C:）的序列号
    DWORD volumeSerialNumber = 0;
    if (GetVolumeInformationA("C:\\", NULL, 0, &volumeSerialNumber, NULL, NULL, NULL, 0)) {
        std::stringstream ss;
        ss << std::hex << volumeSerialNumber;
        diskSerial = ss.str();
    }
    
    // 如果获取C盘序列号失败，尝试获取第一个物理磁盘的序列号
    if (diskSerial.empty()) {
        HANDLE hDevice = CreateFileA("\\\\.\\PhysicalDrive0", 
                                    0, 
                                    FILE_SHARE_READ | FILE_SHARE_WRITE, 
                                    NULL, 
                                    OPEN_EXISTING, 
                                    0, 
                                    NULL);
        
        if (hDevice != INVALID_HANDLE_VALUE) {
            STORAGE_PROPERTY_QUERY query;
            STORAGE_DEVICE_DESCRIPTOR* descriptor = NULL;
            DWORD bytesReturned;
            
            query.PropertyId = StorageDeviceProperty;
            query.QueryType = PropertyStandardQuery;
            
            // 先获取需要的缓冲区大小
            STORAGE_DESCRIPTOR_HEADER header;
            if (DeviceIoControl(hDevice, IOCTL_STORAGE_QUERY_PROPERTY, 
                               &query, sizeof(query), 
                               &header, sizeof(header), 
                               &bytesReturned, NULL)) {
                
                // 分配缓冲区并获取完整描述符
                DWORD bufferSize = header.Size;
                descriptor = (STORAGE_DEVICE_DESCRIPTOR*)malloc(bufferSize);
                
                if (descriptor && DeviceIoControl(hDevice, IOCTL_STORAGE_QUERY_PROPERTY, 
                                                 &query, sizeof(query), 
                                                 descriptor, bufferSize, 
                                                 &bytesReturned, NULL)) {
                    
                    // 获取序列号
                    if (descriptor->SerialNumberOffset != 0) {
                        char* serialNumber = (char*)descriptor + descriptor->SerialNumberOffset;
                        diskSerial = std::string(serialNumber);
                        
                        // 移除空格和非打印字符
                        diskSerial.erase(std::remove_if(diskSerial.begin(), diskSerial.end(), 
                                        [](char c) { return std::isspace(c) || !std::isprint(c); }), 
                                        diskSerial.end());
                    }
                }
                
                if (descriptor) {
                    free(descriptor);
                }
            }
            CloseHandle(hDevice);
        }
    }
    
    // 如果仍然为空，使用一个基于系统信息的备用ID
    if (diskSerial.empty()) {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        std::stringstream ss;
        ss << "BACKUP_" << std::hex << sysInfo.dwProcessorType << "_" << sysInfo.dwNumberOfProcessors;
        diskSerial = ss.str();
    }
    
    return diskSerial;
}

// 获取设备ID
std::string getDeviceId() {
    std::string machineGuid = getMachineGuid();
    std::string diskId = getDiskId();
    return machineGuid + "_" + diskId;
}

// Base64解码
std::vector<unsigned char> base64Decode(const std::string& encoded) {
    BIO* bio = BIO_new_mem_buf(encoded.c_str(), encoded.length());
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64, bio);
    
    std::vector<unsigned char> result(encoded.length());
    int length = BIO_read(bio, result.data(), encoded.length());
    BIO_free_all(bio);
    
    result.resize(length);
    return result;
}

// AES解密
std::string aesDecrypt(const std::string& ciphertext, const std::string& key) {
    try {
        std::vector<unsigned char> data = base64Decode(ciphertext);
        
        if (data.size() < AES_BLOCK_SIZE) {
            return "";
        }
        
        // 提取IV
        unsigned char iv[AES_BLOCK_SIZE];
        std::copy(data.begin(), data.begin() + AES_BLOCK_SIZE, iv);
        
        // 提取密文
        std::vector<unsigned char> encrypted_data(data.begin() + AES_BLOCK_SIZE, data.end());
        
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        
        // 初始化解密
        EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, 
                          (unsigned char*)key.c_str(), iv);
        
        // 解密数据
        std::vector<unsigned char> plaintext(encrypted_data.size());
        int len;
        int plaintext_len;
        
        EVP_DecryptUpdate(ctx, plaintext.data(), &len, 
                         encrypted_data.data(), encrypted_data.size());
        plaintext_len = len;
        
        EVP_DecryptFinal_ex(ctx, plaintext.data() + len, &len);
        plaintext_len += len;
        
        EVP_CIPHER_CTX_free(ctx);
        
        return std::string((char*)plaintext.data(), plaintext_len);
    } catch (...) {
        return "";
    }
}

// 获取当前时间戳
int64_t getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

// 验证授权码
bool verifyLicense(const std::string& licenseFile) {
    try {
        // 读取授权码文件
        std::ifstream file(licenseFile);
        if (!file.is_open()) {
            std::cerr << "授权验证失败: 无法找到授权码文件 " << licenseFile << std::endl;
            return false;
        }
        
        std::string license;
        std::getline(file, license);
        file.close();
        
        if (license.empty()) {
            std::cerr << "授权验证失败: 授权码文件为空" << std::endl;
            return false;
        }
        
        // 解密
        std::string decryptedData = aesDecrypt(license, SECRET_KEY);
        
        if (decryptedData.empty()) {
            std::cerr << "授权验证失败: 无法解密授权码" << std::endl;
            return false;
        }
        
        // 解析数据
        std::vector<std::string> parts;
        std::stringstream ss(decryptedData);
        std::string item;
        
        while (std::getline(ss, item, '|')) {
            parts.push_back(item);
        }
        
        // 检查数据格式 - 必须包含3个部分：设备ID|时间戳|授权天数
        if (parts.size() != 3) {
            std::cerr << "授权验证失败: 授权码格式无效" << std::endl;
            return false;
        }
        
        std::string storedDeviceId = parts[0];
        int64_t storedTimestamp = std::stoll(parts[1]);
        int authorizedDays = std::stoi(parts[2]);
        
        // 验证设备ID
        std::string currentDeviceId = getDeviceId();
        if (storedDeviceId != currentDeviceId) {
            std::cerr << "授权验证失败: 设备ID不匹配" << std::endl;
            std::cerr << "当前设备ID: " << currentDeviceId << std::endl;
            std::cerr << "授权设备ID: " << storedDeviceId << std::endl;
            return false;
        }
        
        // 验证时间
        int64_t currentTimestamp = getCurrentTimestamp();
        int64_t timeDiff = currentTimestamp - storedTimestamp;
        int64_t maxValidTime = static_cast<int64_t>(authorizedDays) * 24 * 60 * 60; // 转换为秒数
        
        if (timeDiff > maxValidTime) {
            std::cerr << "授权验证失败: 授权已过期" << std::endl;
            std::cerr << "授权时间: " << storedTimestamp << std::endl;
            std::cerr << "当前时间: " << currentTimestamp << std::endl;
            std::cerr << "已过期: " << (timeDiff / (24 * 60 * 60)) << " 天" << std::endl;
            std::cerr << "授权期限: " << authorizedDays << " 天" << std::endl;
            return false;
        }
        
        if (timeDiff < 0) {
            std::cerr << "授权验证失败: 系统时间异常" << std::endl;
            return false;
        }
        
        // 授权验证成功
        int64_t remainingDays = (maxValidTime - timeDiff) / (24 * 60 * 60);
        std::cout << "授权验证成功!" << std::endl;
        std::cout << "授权期限: " << authorizedDays << " 天" << std::endl;
        std::cout << "剩余有效期: " << remainingDays << " 天" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "授权验证失败: " << e.what() << std::endl;
        return false;
    }
} 