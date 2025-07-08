#ifndef GPU_COMMON_H
#define GPU_COMMON_H

#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// 常量定义
#define GPU_MAX_DEVICES 16
#define MAX_GPU_DEVICES 16  // 兼容性别名
#define GPU_MAX_NAME_LEN 256
#define GPU_MAX_ERROR_LEN 512

// GPU厂商枚举
typedef enum {
    GPU_VENDOR_UNKNOWN = 0,
    GPU_VENDOR_NVIDIA,
    GPU_VENDOR_AMD,
    GPU_VENDOR_INTEL,
    GPU_VENDOR_ANY = 0xFF
} gpu_vendor_t;

// GPU错误码
typedef enum {
    GPU_SUCCESS = 0,
    GPU_ERROR_INVALID_PARAM,
    GPU_ERROR_OUT_OF_MEMORY,
    GPU_ERROR_DEVICE_NOT_FOUND,
    GPU_ERROR_INITIALIZATION_FAILED,
    GPU_ERROR_OPERATION_FAILED,
    GPU_ERROR_PERMISSION_DENIED,
    GPU_ERROR_TIMEOUT,
    GPU_ERROR_UNKNOWN
} gpu_result_t;

// 日志级别
typedef enum {
    GPU_LOG_DEBUG = 0,
    GPU_LOG_INFO,
    GPU_LOG_WARNING,
    GPU_LOG_ERROR
} gpu_log_level_t;

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
} gpu_device_t;

// GPU设备状态
typedef struct {
    bool is_available;
    size_t memory_used;
    size_t memory_total;
    float utilization;      // 0.0 - 1.0
    int temperature;        // 摄氏度
    int power_usage;        // 瓦特
} gpu_device_status_t;

// GPU设备要求
typedef struct {
    size_t min_memory;
    gpu_vendor_t vendor;
    int min_compute_capability;
} gpu_requirements_t;

// GPU设备管理器
typedef struct {
    gpu_device_t devices[GPU_MAX_DEVICES];
    int device_count;
    bool initialized;
} gpu_device_manager_t;

// 核心API函数

/**
 * 初始化GPU设备管理器
 * @return GPU_SUCCESS 成功，其他值表示错误
 */
gpu_result_t gpu_device_manager_init(void);

/**
 * 清理GPU设备管理器
 */
void gpu_device_manager_cleanup(void);

/**
 * 获取GPU设备数量
 * @return 设备数量，0表示无设备或未初始化
 */
int gpu_get_device_count(void);

/**
 * 获取GPU设备信息
 * @param device_id 设备ID
 * @param device_info 输出设备信息
 * @return GPU_SUCCESS 成功，其他值表示错误
 */
gpu_result_t gpu_get_device_info(int device_id, gpu_device_t *device_info);

/**
 * 检查设备兼容性
 * @param device_id 设备ID
 * @param requirements 设备要求
 * @return true 兼容，false 不兼容
 */
bool gpu_is_device_compatible(int device_id, const gpu_requirements_t *requirements);

/**
 * 查询设备状态
 * @param device_id 设备ID
 * @param status 输出设备状态
 * @return GPU_SUCCESS 成功，其他值表示错误
 */
gpu_result_t gpu_query_device_status(int device_id, gpu_device_status_t *status);

// 错误处理函数

/**
 * 获取错误字符串
 * @param error 错误码
 * @return 错误描述字符串
 */
const char* gpu_get_error_string(gpu_result_t error);

/**
 * 通用错误处理
 * @param error 错误码
 * @param context 错误上下文
 */
void gpu_handle_error(gpu_result_t error, const char *context);

// 日志函数

/**
 * 记录日志
 * @param level 日志级别
 * @param format 格式字符串
 * @param ... 可变参数
 */
void gpu_log(gpu_log_level_t level, const char *format, ...);

// 工具函数

/**
 * 内存对齐
 * @param size 原始大小
 * @param alignment 对齐字节数
 * @return 对齐后的大小
 */
size_t gpu_align_memory(size_t size, size_t alignment);

/**
 * 获取微秒级时间戳
 * @return 时间戳（微秒）
 */
uint64_t gpu_get_timestamp_us(void);

// 宏定义

#define GPU_CHECK(call) do { \
    gpu_result_t result = (call); \
    if (result != GPU_SUCCESS) { \
        gpu_handle_error(result, #call); \
        return result; \
    } \
} while(0)

#define GPU_CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        gpu_log(GPU_LOG_ERROR, "Null pointer: %s", #ptr); \
        return GPU_ERROR_INVALID_PARAM; \
    } \
} while(0)

#define GPU_ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))
#define GPU_ALIGN_DOWN(x, align) ((x) & ~((align) - 1))

#define GPU_MIN(a, b) ((a) < (b) ? (a) : (b))
#define GPU_MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef __cplusplus
}
#endif

#endif // GPU_COMMON_H