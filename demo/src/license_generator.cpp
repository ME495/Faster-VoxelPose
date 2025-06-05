#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <windows.h>
#include <winioctl.h>
#include <intrin.h>
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>

#pragma comment(lib, "libcrypto.lib")
#pragma comment(lib, "libssl.lib")

// 密钥 - 在实际应用中应该更复杂和安全
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

// Base64编码
std::string base64Encode(const std::vector<unsigned char>& data) {
    BIO* bio = BIO_new(BIO_s_mem());
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64, bio);
    
    BIO_write(bio, data.data(), data.size());
    BIO_flush(bio);
    
    BUF_MEM* bufferPtr;
    BIO_get_mem_ptr(bio, &bufferPtr);
    
    std::string result(bufferPtr->data, bufferPtr->length);
    BIO_free_all(bio);
    
    return result;
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

// AES加密
std::string aesEncrypt(const std::string& plaintext, const std::string& key) {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    
    // 生成随机IV
    unsigned char iv[AES_BLOCK_SIZE];
    RAND_bytes(iv, AES_BLOCK_SIZE);
    
    // 初始化加密
    EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, 
                      (unsigned char*)key.c_str(), iv);
    
    // 加密数据
    std::vector<unsigned char> ciphertext(plaintext.length() + AES_BLOCK_SIZE);
    int len;
    int ciphertext_len;
    
    EVP_EncryptUpdate(ctx, ciphertext.data(), &len, 
                     (unsigned char*)plaintext.c_str(), plaintext.length());
    ciphertext_len = len;
    
    EVP_EncryptFinal_ex(ctx, ciphertext.data() + len, &len);
    ciphertext_len += len;
    
    EVP_CIPHER_CTX_free(ctx);
    
    // 将IV和密文合并
    std::vector<unsigned char> result;
    result.insert(result.end(), iv, iv + AES_BLOCK_SIZE);
    result.insert(result.end(), ciphertext.begin(), ciphertext.begin() + ciphertext_len);
    
    return base64Encode(result);
}

// AES解密
std::string aesDecrypt(const std::string& ciphertext, const std::string& key) {
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
}

// 获取当前时间戳
int64_t getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(duration).count();
}

// 生成授权码
std::string generateLicense(int authorizedDays) {
    std::string deviceId = getDeviceId();
    int64_t timestamp = getCurrentTimestamp();
    
    std::cout << "设备ID: " << deviceId << std::endl;
    std::cout << "时间戳: " << timestamp << std::endl;
    std::cout << "授权天数: " << authorizedDays << " 天" << std::endl;
    
    // 组合数据：设备ID|时间戳|授权天数
    std::string data = deviceId + "|" + std::to_string(timestamp) + "|" + std::to_string(authorizedDays);
    
    // 加密
    std::string encryptedData = aesEncrypt(data, SECRET_KEY);
    
    return encryptedData;
}

// 验证授权码
bool verifyLicense(const std::string& license) {
    try {
        // 解密
        std::string decryptedData = aesDecrypt(license, SECRET_KEY);
        
        // 解析数据
        std::vector<std::string> parts;
        std::stringstream ss(decryptedData);
        std::string item;
        
        while (std::getline(ss, item, '|')) {
            parts.push_back(item);
        }
        
        // 检查数据格式 - 必须包含3个部分：设备ID|时间戳|授权天数
        if (parts.size() != 3) {
            std::cout << "授权码格式无效!" << std::endl;
            return false;
        }
        
        std::string storedDeviceId = parts[0];
        int64_t storedTimestamp = std::stoll(parts[1]);
        int authorizedDays = std::stoi(parts[2]);
        
        std::cout << "存储的设备ID: " << storedDeviceId << std::endl;
        std::cout << "存储的时间戳: " << storedTimestamp << std::endl;
        std::cout << "授权天数: " << authorizedDays << " 天" << std::endl;
        
        // 验证设备ID
        std::string currentDeviceId = getDeviceId();
        if (storedDeviceId != currentDeviceId) {
            std::cout << "设备ID不匹配!" << std::endl;
            return false;
        }
        
        // 验证时间
        int64_t currentTimestamp = getCurrentTimestamp();
        int64_t timeDiff = currentTimestamp - storedTimestamp;
        int64_t maxValidTime = static_cast<int64_t>(authorizedDays) * 24 * 60 * 60; // 转换为秒数
        
        std::cout << "当前时间戳: " << currentTimestamp << std::endl;
        std::cout << "时间差: " << timeDiff << " 秒 (" << timeDiff / (24 * 60 * 60) << " 天)" << std::endl;
        
        if (timeDiff > maxValidTime) {
            std::cout << "授权已过期!" << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cout << "验证授权码时出错: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "VoxelPose 授权码生成器" << std::endl;
    std::cout << "========================" << std::endl;
    
    if (argc > 1 && std::string(argv[1]) == "verify") {
        // 验证模式
        std::cout << "验证授权码..." << std::endl;
        
        std::ifstream file("license.dat");
        if (!file.is_open()) {
            std::cout << "无法找到授权码文件!" << std::endl;
            return 1;
        }
        
        std::string license;
        std::getline(file, license);
        file.close();
        
        if (verifyLicense(license)) {
            std::cout << "授权验证成功!" << std::endl;
            return 0;
        } else {
            std::cout << "授权验证失败!" << std::endl;
            return 1;
        }
    } else {
        // 生成模式
        std::cout << "生成授权码..." << std::endl;
        
        int authorizedDays;
        
        // 检查命令行参数
        if (argc >= 2) {
            try {
                authorizedDays = std::stoi(argv[1]);
                if (authorizedDays <= 0) {
                    std::cout << "错误: 授权天数必须大于0" << std::endl;
                    return 1;
                }
                std::cout << "使用命令行参数授权天数: " << authorizedDays << " 天" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "错误: 无效的授权天数参数 '" << argv[1] << "'" << std::endl;
                std::cout << "用法: " << argv[0] << " [授权天数]" << std::endl;
                std::cout << "      " << argv[0] << " verify" << std::endl;
                return 1;
            }
        } else {
            // 交互式输入
            std::cout << "请输入授权天数: ";
            if (!(std::cin >> authorizedDays) || authorizedDays <= 0) {
                std::cout << "错误: 请输入有效的授权天数（大于0的整数）" << std::endl;
                return 1;
            }
        }
        
        std::string license = generateLicense(authorizedDays);
        
        // 保存到文件
        std::ofstream file("license.dat");
        if (file.is_open()) {
            file << license;
            file.close();
            std::cout << "授权码已生成并保存到 license.dat" << std::endl;
            std::cout << "授权码: " << license << std::endl;
            std::cout << "有效期: " << authorizedDays << " 天" << std::endl;
        } else {
            std::cout << "无法保存授权码文件!" << std::endl;
            return 1;
        }
    }
    
    return 0;
} 