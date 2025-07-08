#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "gpu_common.h"

// GPU设备管理器全局实例
static gpu_device_manager_t g_device_manager = {0};
static bool g_manager_initialized = false;

// GPU设备信息结构
typedef struct {
    int device_id;
    char name[GPU_MAX_NAME_LEN];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    bool is_available;
    gpu_vendor_t vendor;
} gpu_device_info_t;

// 错误码到字符串的映射
static const char* gpu_error_strings[] = {
    [GPU_SUCCESS] = "Success",
    [GPU_ERROR_INVALID_PARAM] = "Invalid parameter",
    [GPU_ERROR_OUT_OF_MEMORY] = "Out of memory",
    [GPU_ERROR_DEVICE_NOT_FOUND] = "Device not found",
    [GPU_ERROR_INITIALIZATION_FAILED] = "Initialization failed",
    [GPU_ERROR_OPERATION_FAILED] = "Operation failed",
    [GPU_ERROR_PERMISSION_DENIED] = "Permission denied",
    [GPU_ERROR_TIMEOUT] = "Operation timeout",
    [GPU_ERROR_UNKNOWN] = "Unknown error"
};

// GPU厂商检测
static gpu_vendor_t detect_gpu_vendor(int device_id) {
    char vendor_path[256];
    char vendor_str[64] = {0};
    FILE *fp;
    
    // 尝试读取PCI设备信息
    snprintf(vendor_path, sizeof(vendor_path), 
             "/sys/class/drm/card%d/device/vendor", device_id);
    
    fp = fopen(vendor_path, "r");
    if (fp) {
        if (fgets(vendor_str, sizeof(vendor_str), fp)) {
            // NVIDIA: 0x10de, AMD: 0x1002, Intel: 0x8086
            if (strstr(vendor_str, "0x10de")) {
                fclose(fp);
                return GPU_VENDOR_NVIDIA;
            } else if (strstr(vendor_str, "0x1002")) {
                fclose(fp);
                return GPU_VENDOR_AMD;
            } else if (strstr(vendor_str, "0x8086")) {
                fclose(fp);
                return GPU_VENDOR_INTEL;
            }
        }
        fclose(fp);
    }
    
    return GPU_VENDOR_UNKNOWN;
}

// GPU设备枚举
static int enumerate_gpu_devices(gpu_device_info_t *devices, int max_devices) {
    int device_count = 0;
    char device_path[256];
    struct stat st;
    
    // 扫描/dev/dri/card*设备
    for (int i = 0; i < max_devices && device_count < max_devices; i++) {
        snprintf(device_path, sizeof(device_path), "/dev/dri/card%d", i);
        
        if (stat(device_path, &st) == 0) {
            devices[device_count].device_id = i;
            devices[device_count].vendor = detect_gpu_vendor(i);
            devices[device_count].is_available = true;
            
            // 设置默认名称
            switch (devices[device_count].vendor) {
                case GPU_VENDOR_NVIDIA:
                    snprintf(devices[device_count].name, GPU_MAX_NAME_LEN, 
                             "NVIDIA GPU %d", i);
                    break;
                case GPU_VENDOR_AMD:
                    snprintf(devices[device_count].name, GPU_MAX_NAME_LEN, 
                             "AMD GPU %d", i);
                    break;
                case GPU_VENDOR_INTEL:
                    snprintf(devices[device_count].name, GPU_MAX_NAME_LEN, 
                             "Intel GPU %d", i);
                    break;
                default:
                    snprintf(devices[device_count].name, GPU_MAX_NAME_LEN, 
                             "Unknown GPU %d", i);
                    break;
            }
            
            // 设置默认内存大小（实际应该从驱动获取）
            devices[device_count].total_memory = 8ULL * 1024 * 1024 * 1024; // 8GB
            devices[device_count].free_memory = devices[device_count].total_memory;
            devices[device_count].compute_capability_major = 7;
            devices[device_count].compute_capability_minor = 5;
            
            device_count++;
        }
    }
    
    return device_count;
}

// 初始化GPU设备管理器
gpu_result_t gpu_device_manager_init(void) {
    if (g_manager_initialized) {
        return GPU_SUCCESS;
    }
    
    memset(&g_device_manager, 0, sizeof(g_device_manager));
    
    // 枚举GPU设备
    gpu_device_info_t devices[GPU_MAX_DEVICES];
    int device_count = enumerate_gpu_devices(devices, GPU_MAX_DEVICES);
    
    if (device_count == 0) {
        fprintf(stderr, "No GPU devices found\n");
        return GPU_ERROR_DEVICE_NOT_FOUND;
    }
    
    g_device_manager.device_count = device_count;
    
    // 复制设备信息
    for (int i = 0; i < device_count; i++) {
        g_device_manager.devices[i].device_id = devices[i].device_id;
        strncpy(g_device_manager.devices[i].name, devices[i].name, GPU_MAX_NAME_LEN - 1);
        g_device_manager.devices[i].total_memory = devices[i].total_memory;
        g_device_manager.devices[i].free_memory = devices[i].free_memory;
        g_device_manager.devices[i].is_available = devices[i].is_available;
        g_device_manager.devices[i].vendor = devices[i].vendor;
    }
    
    g_manager_initialized = true;
    printf("GPU device manager initialized with %d devices\n", device_count);
    
    return GPU_SUCCESS;
}

// 清理GPU设备管理器
void gpu_device_manager_cleanup(void) {
    if (!g_manager_initialized) {
        return;
    }
    
    memset(&g_device_manager, 0, sizeof(g_device_manager));
    g_manager_initialized = false;
    
    printf("GPU device manager cleaned up\n");
}

// 获取GPU设备数量
int gpu_get_device_count(void) {
    if (!g_manager_initialized) {
        return 0;
    }
    
    return g_device_manager.device_count;
}

// 获取GPU设备信息
gpu_result_t gpu_get_device_info(int device_id, gpu_device_t *device_info) {
    if (!g_manager_initialized) {
        return GPU_ERROR_INITIALIZATION_FAILED;
    }
    
    if (device_info == NULL) {
        return GPU_ERROR_INVALID_PARAM;
    }
    
    if (device_id < 0 || device_id >= g_device_manager.device_count) {
        return GPU_ERROR_DEVICE_NOT_FOUND;
    }
    
    *device_info = g_device_manager.devices[device_id];
    return GPU_SUCCESS;
}

// 检查设备兼容性
bool gpu_is_device_compatible(int device_id, const gpu_requirements_t *requirements) {
    if (!g_manager_initialized || requirements == NULL) {
        return false;
    }
    
    if (device_id < 0 || device_id >= g_device_manager.device_count) {
        return false;
    }
    
    gpu_device_t *device = &g_device_manager.devices[device_id];
    
    // 检查内存要求
    if (requirements->min_memory > 0 && 
        device->total_memory < requirements->min_memory) {
        return false;
    }
    
    // 检查厂商要求
    if (requirements->vendor != GPU_VENDOR_ANY && 
        device->vendor != requirements->vendor) {
        return false;
    }
    
    // 检查计算能力要求
    if (requirements->min_compute_capability > 0) {
        int device_capability = device->compute_capability_major * 10 + 
                               device->compute_capability_minor;
        if (device_capability < requirements->min_compute_capability) {
            return false;
        }
    }
    
    return true;
}

// 获取错误字符串
const char* gpu_get_error_string(gpu_result_t error) {
    if (error >= 0 && error < sizeof(gpu_error_strings) / sizeof(gpu_error_strings[0])) {
        return gpu_error_strings[error];
    }
    return "Unknown error code";
}

// 设备状态查询
gpu_result_t gpu_query_device_status(int device_id, gpu_device_status_t *status) {
    if (!g_manager_initialized) {
        return GPU_ERROR_INITIALIZATION_FAILED;
    }
    
    if (status == NULL) {
        return GPU_ERROR_INVALID_PARAM;
    }
    
    if (device_id < 0 || device_id >= g_device_manager.device_count) {
        return GPU_ERROR_DEVICE_NOT_FOUND;
    }
    
    gpu_device_t *device = &g_device_manager.devices[device_id];
    
    // 模拟状态查询
    status->is_available = device->is_available;
    status->memory_used = device->total_memory - device->free_memory;
    status->memory_total = device->total_memory;
    status->utilization = 0; // 需要从驱动获取实际利用率
    status->temperature = 65; // 模拟温度
    status->power_usage = 150; // 模拟功耗
    
    return GPU_SUCCESS;
}

// 通用错误处理
void gpu_handle_error(gpu_result_t error, const char *context) {
    if (error != GPU_SUCCESS) {
        fprintf(stderr, "GPU Error in %s: %s\n", 
                context ? context : "unknown", 
                gpu_get_error_string(error));
    }
}

// 日志记录
void gpu_log(gpu_log_level_t level, const char *format, ...) {
    const char *level_str[] = {
        [GPU_LOG_DEBUG] = "DEBUG",
        [GPU_LOG_INFO] = "INFO",
        [GPU_LOG_WARNING] = "WARNING",
        [GPU_LOG_ERROR] = "ERROR"
    };
    
    va_list args;
    va_start(args, format);
    
    printf("[GPU_%s] ", level_str[level]);
    vprintf(format, args);
    printf("\n");
    
    va_end(args);
}

// 内存对齐工具函数
size_t gpu_align_memory(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// 时间戳获取
uint64_t gpu_get_timestamp_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}