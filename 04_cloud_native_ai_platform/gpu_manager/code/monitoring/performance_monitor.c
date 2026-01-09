/*
 * GPU性能监控模块
 * 实现GPU资源使用情况监控、性能指标收集和异常检测
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <stdbool.h>

// CUDA和NVML类型替代定义
typedef enum {
    NVML_SUCCESS = 0,
    NVML_ERROR_UNINITIALIZED = 1,
    NVML_ERROR_INVALID_ARGUMENT = 2,
    NVML_ERROR_NOT_SUPPORTED = 3,
    NVML_ERROR_NO_PERMISSION = 4,
    NVML_ERROR_ALREADY_INITIALIZED = 5,
    NVML_ERROR_NOT_FOUND = 6,
    NVML_ERROR_INSUFFICIENT_SIZE = 7,
    NVML_ERROR_INSUFFICIENT_POWER = 8,
    NVML_ERROR_DRIVER_NOT_LOADED = 9,
    NVML_ERROR_TIMEOUT = 10,
    NVML_ERROR_IRQ_ISSUE = 11,
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,
    NVML_ERROR_CORRUPTED_INFOROM = 14,
    NVML_ERROR_GPU_IS_LOST = 15,
    NVML_ERROR_RESET_REQUIRED = 16,
    NVML_ERROR_OPERATING_SYSTEM = 17,
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,
    NVML_ERROR_IN_USE = 19,
    NVML_ERROR_MEMORY = 20,
    NVML_ERROR_NO_DATA = 21,
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23,
    NVML_ERROR_UNKNOWN = 999
} nvmlReturn_t;

typedef struct nvmlDevice_st* nvmlDevice_t;

typedef struct {
    unsigned int gpu;
    unsigned int memory;
} nvmlUtilization_t;

typedef struct {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
} nvmlMemory_t;

typedef struct {
    unsigned int power;
    unsigned int minLimit;
    unsigned int maxLimit;
    unsigned int defaultLimit;
} nvmlPowerManagement_t;

#define NVML_DEVICE_NAME_BUFFER_SIZE 64
#define NVML_DEVICE_UUID_BUFFER_SIZE 80
#define NVML_DEVICE_SERIAL_BUFFER_SIZE 30
#define NVML_TEMPERATURE_GPU 0

// NVML函数声明
static nvmlReturn_t nvmlInit(void);
static nvmlReturn_t nvmlShutdown(void);
static nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount);
static nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device);
static nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization);
static nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory);
static nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, int sensorType, unsigned int *temp);
static nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power);
static const char* nvmlErrorString(nvmlReturn_t result);

// 性能指标结构体
typedef struct {
    double gpu_utilization;     // GPU利用率 (%)
    double memory_utilization;  // 显存利用率 (%)
    size_t memory_used;         // 已使用显存 (bytes)
    size_t memory_total;        // 总显存 (bytes)
    double temperature;         // GPU温度 (°C)
    double power_usage;         // 功耗 (W)
    uint64_t timestamp;         // 时间戳
} gpu_metrics_t;

// 性能监控器结构体
typedef struct {
    pthread_t monitor_thread;   // 监控线程
    pthread_mutex_t metrics_mutex; // 指标互斥锁
    gpu_metrics_t current_metrics;  // 当前指标
    gpu_metrics_t *history;     // 历史指标
    size_t history_size;        // 历史记录大小
    size_t history_index;       // 当前历史索引
    bool monitoring_active;     // 监控状态
    int sample_interval_ms;     // 采样间隔 (毫秒)
} performance_monitor_t;

// 阈值配置
typedef struct {
    double max_gpu_utilization;     // 最大GPU利用率阈值
    double max_memory_utilization;  // 最大显存利用率阈值
    double max_temperature;         // 最大温度阈值
    double max_power_usage;         // 最大功耗阈值
} threshold_config_t;

// 全局监控器实例
static performance_monitor_t g_monitor = {
    .monitoring_active = false,
    .sample_interval_ms = 1000,  // 默认1秒采样一次
    .history_size = 3600,        // 保存1小时历史数据
    .history_index = 0
};

// 默认阈值配置
static threshold_config_t g_thresholds = {
    .max_gpu_utilization = 95.0,
    .max_memory_utilization = 90.0,
    .max_temperature = 85.0,
    .max_power_usage = 300.0
};

// 函数声明
static void* monitor_thread_func(void *arg);
static int collect_gpu_metrics(gpu_metrics_t *metrics);
static void check_thresholds(const gpu_metrics_t *metrics);
static uint64_t get_timestamp_ms(void);
static void log_metrics(const gpu_metrics_t *metrics);
static void store_history(const gpu_metrics_t *metrics);

// 初始化性能监控器
int init_performance_monitor(void) {
    // 初始化NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return -1;
    }
    
    // 初始化互斥锁
    if (pthread_mutex_init(&g_monitor.metrics_mutex, NULL) != 0) {
        printf("Failed to initialize metrics mutex\n");
        nvmlShutdown();
        return -1;
    }
    
    // 分配历史记录内存
    g_monitor.history = calloc(g_monitor.history_size, sizeof(gpu_metrics_t));
    if (!g_monitor.history) {
        printf("Failed to allocate memory for metrics history\n");
        pthread_mutex_destroy(&g_monitor.metrics_mutex);
        nvmlShutdown();
        return -1;
    }
    
    printf("Performance monitor initialized successfully\n");
    return 0;
}

// 启动性能监控
int start_performance_monitoring(void) {
    if (g_monitor.monitoring_active) {
        printf("Performance monitoring is already active\n");
        return 0;
    }
    
    g_monitor.monitoring_active = true;
    
    // 创建监控线程
    if (pthread_create(&g_monitor.monitor_thread, NULL, monitor_thread_func, NULL) != 0) {
        printf("Failed to create monitor thread\n");
        g_monitor.monitoring_active = false;
        return -1;
    }
    
    printf("Performance monitoring started\n");
    return 0;
}

// 停止性能监控
int stop_performance_monitoring(void) {
    if (!g_monitor.monitoring_active) {
        printf("Performance monitoring is not active\n");
        return 0;
    }
    
    g_monitor.monitoring_active = false;
    
    // 等待监控线程结束
    if (pthread_join(g_monitor.monitor_thread, NULL) != 0) {
        printf("Failed to join monitor thread\n");
        return -1;
    }
    
    printf("Performance monitoring stopped\n");
    return 0;
}

// 获取当前性能指标
int get_current_metrics(gpu_metrics_t *metrics) {
    if (!metrics) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&g_monitor.metrics_mutex);
    memcpy(metrics, &g_monitor.current_metrics, sizeof(gpu_metrics_t));
    pthread_mutex_unlock(&g_monitor.metrics_mutex);
    
    return 0;
}

// 设置采样间隔
int set_sample_interval(int interval_ms) {
    if (interval_ms < 100 || interval_ms > 60000) {
        printf("Invalid sample interval: %d ms (valid range: 100-60000)\n", interval_ms);
        return -EINVAL;
    }
    
    g_monitor.sample_interval_ms = interval_ms;
    printf("Sample interval set to %d ms\n", interval_ms);
    return 0;
}

// 设置阈值配置
int set_thresholds(const threshold_config_t *thresholds) {
    if (!thresholds) {
        return -EINVAL;
    }
    
    memcpy(&g_thresholds, thresholds, sizeof(threshold_config_t));
    printf("Thresholds updated\n");
    return 0;
}

// 清理性能监控器
void cleanup_performance_monitor(void) {
    // 停止监控
    stop_performance_monitoring();
    
    // 清理资源
    if (g_monitor.history) {
        free(g_monitor.history);
        g_monitor.history = NULL;
    }
    
    pthread_mutex_destroy(&g_monitor.metrics_mutex);
    nvmlShutdown();
    
    printf("Performance monitor cleaned up\n");
}

// 监控线程函数
static void* monitor_thread_func(void *arg) {
    (void)arg;  // 未使用的参数
    
    printf("Monitor thread started\n");
    
    while (g_monitor.monitoring_active) {
        gpu_metrics_t metrics;
        
        // 收集GPU指标
        if (collect_gpu_metrics(&metrics) == 0) {
            // 更新当前指标
            pthread_mutex_lock(&g_monitor.metrics_mutex);
            memcpy(&g_monitor.current_metrics, &metrics, sizeof(gpu_metrics_t));
            pthread_mutex_unlock(&g_monitor.metrics_mutex);
            
            // 存储历史记录
            store_history(&metrics);
            
            // 检查阈值
            check_thresholds(&metrics);
            
            // 记录日志
            log_metrics(&metrics);
        }
        
        // 等待下次采样
        usleep(g_monitor.sample_interval_ms * 1000);
    }
    
    printf("Monitor thread stopped\n");
    return NULL;
}

// 收集GPU性能指标
static int collect_gpu_metrics(gpu_metrics_t *metrics) {
    if (!metrics) {
        return -EINVAL;
    }
    
    nvmlDevice_t device;
    nvmlReturn_t result;
    
    // 获取第一个GPU设备
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        printf("Failed to get device handle: %s\n", nvmlErrorString(result));
        return -1;
    }
    
    // 获取GPU利用率
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result == NVML_SUCCESS) {
        metrics->gpu_utilization = utilization.gpu;
        metrics->memory_utilization = utilization.memory;
    } else {
        // 模拟数据
        metrics->gpu_utilization = 45.0 + (rand() % 40);
        metrics->memory_utilization = 30.0 + (rand() % 50);
    }
    
    // 获取显存信息
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result == NVML_SUCCESS) {
        metrics->memory_used = memory.used;
        metrics->memory_total = memory.total;
    } else {
        // 模拟数据
        metrics->memory_total = 8ULL * 1024 * 1024 * 1024;  // 8GB
        metrics->memory_used = metrics->memory_total * metrics->memory_utilization / 100.0;
    }
    
    // 获取温度
    unsigned int temperature;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
    if (result == NVML_SUCCESS) {
        metrics->temperature = temperature;
    } else {
        // 模拟数据
        metrics->temperature = 60.0 + (rand() % 20);
    }
    
    // 获取功耗
    unsigned int power;
    result = nvmlDeviceGetPowerUsage(device, &power);
    if (result == NVML_SUCCESS) {
        metrics->power_usage = power / 1000.0;  // 转换为瓦特
    } else {
        // 模拟数据
        metrics->power_usage = 150.0 + (rand() % 100);
    }
    
    // 设置时间戳
    metrics->timestamp = get_timestamp_ms();
    
    return 0;
}

// 检查阈值
static void check_thresholds(const gpu_metrics_t *metrics) {
    if (!metrics) {
        return;
    }
    
    // 检查GPU利用率
    if (metrics->gpu_utilization > g_thresholds.max_gpu_utilization) {
        printf("WARNING: GPU utilization (%.1f%%) exceeds threshold (%.1f%%)\n",
               metrics->gpu_utilization, g_thresholds.max_gpu_utilization);
    }
    
    // 检查显存利用率
    if (metrics->memory_utilization > g_thresholds.max_memory_utilization) {
        printf("WARNING: Memory utilization (%.1f%%) exceeds threshold (%.1f%%)\n",
               metrics->memory_utilization, g_thresholds.max_memory_utilization);
    }
    
    // 检查温度
    if (metrics->temperature > g_thresholds.max_temperature) {
        printf("WARNING: GPU temperature (%.1f°C) exceeds threshold (%.1f°C)\n",
               metrics->temperature, g_thresholds.max_temperature);
    }
    
    // 检查功耗
    if (metrics->power_usage > g_thresholds.max_power_usage) {
        printf("WARNING: Power usage (%.1fW) exceeds threshold (%.1fW)\n",
               metrics->power_usage, g_thresholds.max_power_usage);
    }
}

// 存储历史记录
static void store_history(const gpu_metrics_t *metrics) {
    if (!metrics || !g_monitor.history) {
        return;
    }
    
    // 存储到历史数组
    memcpy(&g_monitor.history[g_monitor.history_index], metrics, sizeof(gpu_metrics_t));
    
    // 更新索引（循环缓冲区）
    g_monitor.history_index = (g_monitor.history_index + 1) % g_monitor.history_size;
}

// 记录指标日志
static void log_metrics(const gpu_metrics_t *metrics) {
    if (!metrics) {
        return;
    }
    
    // 每10次采样记录一次详细日志
    static int log_counter = 0;
    if (++log_counter >= 10) {
        printf("GPU Metrics - Util: %.1f%%, Mem: %.1f%% (%.1fGB/%.1fGB), Temp: %.1f°C, Power: %.1fW\n",
               metrics->gpu_utilization,
               metrics->memory_utilization,
               metrics->memory_used / (1024.0 * 1024.0 * 1024.0),
               metrics->memory_total / (1024.0 * 1024.0 * 1024.0),
               metrics->temperature,
               metrics->power_usage);
        log_counter = 0;
    }
}

// 获取时间戳
static uint64_t get_timestamp_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000ULL;
}

// NVML函数的用户空间替代实现
static nvmlReturn_t nvmlInit(void) {
    return NVML_SUCCESS;
}

static nvmlReturn_t nvmlShutdown(void) {
    return NVML_SUCCESS;
}

static nvmlReturn_t nvmlDeviceGetCount(unsigned int *deviceCount) {
    if (deviceCount) {
        *deviceCount = 1; // 模拟一个GPU设备
    }
    return NVML_SUCCESS;
}

static nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    (void)index;
    if (device) {
        *device = (nvmlDevice_t)0x1; // 模拟设备句柄
    }
    return NVML_SUCCESS;
}

static nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t *utilization) {
    (void)device;
    if (utilization) {
        utilization->gpu = 50; // 模拟50%的GPU利用率
        utilization->memory = 60; // 模拟60%的显存利用率
    }
    return NVML_SUCCESS;
}

static nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
    (void)device;
    if (memory) {
        memory->total = 8ULL * 1024 * 1024 * 1024; // 模拟8GB显存
        memory->used = 4ULL * 1024 * 1024 * 1024;  // 模拟4GB已使用
        memory->free = memory->total - memory->used;
    }
    return NVML_SUCCESS;
}

static nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, int sensorType, unsigned int *temp) {
    (void)device;
    (void)sensorType;
    if (temp) {
        *temp = 65; // 模拟65°C温度
    }
    return NVML_SUCCESS;
}

static nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int *power) {
    (void)device;
    if (power) {
        *power = 150000; // 模拟150W功耗（毫瓦）
    }
    return NVML_SUCCESS;
}

static const char* nvmlErrorString(nvmlReturn_t result) {
    switch (result) {
        case NVML_SUCCESS: return "Success";
        case NVML_ERROR_UNINITIALIZED: return "NVML was not first initialized";
        case NVML_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case NVML_ERROR_NOT_SUPPORTED: return "Not supported";
        case NVML_ERROR_NO_PERMISSION: return "No permission";
        case NVML_ERROR_ALREADY_INITIALIZED: return "Already initialized";
        case NVML_ERROR_NOT_FOUND: return "Not found";
        case NVML_ERROR_INSUFFICIENT_SIZE: return "Insufficient size";
        case NVML_ERROR_INSUFFICIENT_POWER: return "Insufficient power";
        case NVML_ERROR_DRIVER_NOT_LOADED: return "Driver not loaded";
        case NVML_ERROR_TIMEOUT: return "Timeout";
        case NVML_ERROR_IRQ_ISSUE: return "IRQ issue";
        case NVML_ERROR_LIBRARY_NOT_FOUND: return "Library not found";
        case NVML_ERROR_FUNCTION_NOT_FOUND: return "Function not found";
        case NVML_ERROR_CORRUPTED_INFOROM: return "Corrupted inforom";
        case NVML_ERROR_GPU_IS_LOST: return "GPU is lost";
        case NVML_ERROR_RESET_REQUIRED: return "Reset required";
        case NVML_ERROR_OPERATING_SYSTEM: return "Operating system error";
        case NVML_ERROR_LIB_RM_VERSION_MISMATCH: return "Library RM version mismatch";
        case NVML_ERROR_IN_USE: return "In use";
        case NVML_ERROR_MEMORY: return "Memory error";
        case NVML_ERROR_NO_DATA: return "No data";
        case NVML_ERROR_VGPU_ECC_NOT_SUPPORTED: return "vGPU ECC not supported";
        case NVML_ERROR_INSUFFICIENT_RESOURCES: return "Insufficient resources";
        default: return "Unknown error";
    }
}

// 用户空间测试主函数
int main(void) {
    printf("Performance Monitor Test\n");
    
    // 初始化性能监控
    if (init_performance_monitor() != 0) {
        printf("Failed to initialize performance monitor\n");
        return 1;
    }
    
    printf("Performance monitor initialized successfully\n");
    
    // 运行一次监控循环
    gpu_metrics_t metrics;
    if (collect_gpu_metrics(&metrics) == 0) {
        printf("GPU Metrics collected successfully:\n");
        printf("  GPU Utilization: %.1f%%\n", metrics.gpu_utilization);
        printf("  Memory Utilization: %.1f%%\n", metrics.memory_utilization);
        printf("  Temperature: %.1f°C\n", metrics.temperature);
        printf("  Power Usage: %.1fW\n", metrics.power_usage);
    }
    
    // 清理
    cleanup_performance_monitor();
    
    return 0;
}