#ifndef LATENCY_OPTIMIZER_H
#define LATENCY_OPTIMIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// 常量定义
#define MAX_GPU_DEVICES 16
#define MAX_BATCH_SIZE 128
#define MAX_QUEUE_SIZE 1024

// 优化器结果枚举
typedef enum {
    OPTIMIZER_SUCCESS = 0,
    OPTIMIZER_ERROR_INVALID_PARAM,
    OPTIMIZER_ERROR_NOT_INITIALIZED,
    OPTIMIZER_ERROR_OUT_OF_MEMORY,
    OPTIMIZER_ERROR_INVALID_DEVICE,
    OPTIMIZER_ERROR_REQUEST_NOT_FOUND,
    OPTIMIZER_ERROR_INITIALIZATION_FAILED,
    OPTIMIZER_ERROR_THREAD_CREATE_FAILED
} optimizer_result_t;

// GPU操作类型枚举
typedef enum {
    GPU_OP_COMPUTE = 0,
    GPU_OP_MEMORY_COPY,
    GPU_OP_KERNEL_LAUNCH,
    GPU_OP_SYNCHRONIZE,
    GPU_OP_CUSTOM
} gpu_operation_type_t;

// 请求状态枚举
typedef enum {
    REQUEST_STATUS_PENDING = 0,
    REQUEST_STATUS_QUEUED,
    REQUEST_STATUS_EXECUTING,
    REQUEST_STATUS_COMPLETED,
    REQUEST_STATUS_FAILED,
    REQUEST_STATUS_CANCELLED
} request_status_t;

// 请求优先级枚举
typedef enum {
    REQUEST_PRIORITY_LOW = 0,
    REQUEST_PRIORITY_NORMAL = 1,
    REQUEST_PRIORITY_HIGH = 2,
    REQUEST_PRIORITY_CRITICAL = 3
} request_priority_t;

// 批处理策略枚举
typedef enum {
    BATCH_STRATEGY_SIZE_BASED = 0,    // 基于批次大小
    BATCH_STRATEGY_TIME_BASED,        // 基于时间超时
    BATCH_STRATEGY_ADAPTIVE,          // 自适应策略
    BATCH_STRATEGY_PRIORITY_BASED     // 基于优先级
} batch_strategy_t;

// 请求回调函数类型
typedef void (*request_callback_t)(uint64_t request_id, request_status_t status, void *user_data);

// 请求参数结构
typedef struct {
    int device_id;                    // 目标GPU设备ID
    gpu_operation_type_t operation_type; // 操作类型
    void *data;                       // 操作数据
    size_t data_size;                 // 数据大小
    request_priority_t priority;      // 请求优先级
    uint32_t timeout_ms;              // 超时时间（毫秒）
    void *kernel_params;              // 内核参数（可选）
    size_t kernel_params_size;        // 内核参数大小
} request_params_t;

// GPU请求结构
typedef struct gpu_request {
    uint64_t request_id;              // 请求唯一ID
    request_params_t params;          // 请求参数
    request_callback_t callback;      // 完成回调函数
    void *user_data;                  // 用户数据
    uint64_t submit_time;             // 提交时间戳
    uint64_t start_time;              // 开始执行时间戳
    uint64_t completion_time;         // 完成时间戳
    request_status_t status;          // 请求状态
    request_priority_t priority;      // 优先级
    uint64_t estimated_duration_us;   // 估算执行时间（微秒）
    struct gpu_request *next;         // 链表指针
} gpu_request_t;

// 请求队列结构
typedef struct {
    gpu_request_t *head;              // 队列头
    gpu_request_t *tail;              // 队列尾
    uint32_t count;                   // 队列中请求数量
    uint32_t max_size;                // 队列最大大小
} request_queue_t;

// 请求批次结构
typedef struct {
    uint64_t batch_id;                // 批次唯一ID
    int device_id;                    // 目标设备ID
    gpu_request_t *requests[MAX_BATCH_SIZE]; // 批次中的请求
    uint32_t request_count;           // 请求数量
    uint64_t created_time;            // 创建时间戳
    uint64_t start_time;              // 开始执行时间戳
    uint64_t completion_time;         // 完成时间戳
    uint64_t total_estimated_duration_us; // 总估算时间
    uint64_t actual_duration_us;      // 实际执行时间
} request_batch_t;

// 优化器配置结构
typedef struct {
    uint32_t max_batch_size;          // 最大批次大小
    uint64_t batch_timeout_us;        // 批次超时时间（微秒）
    uint64_t check_interval_us;       // 检查间隔（微秒）
    batch_strategy_t batch_strategy;  // 批处理策略
    bool enable_priority_scheduling;  // 是否启用优先级调度
    bool enable_load_balancing;       // 是否启用负载均衡
    bool enable_adaptive_batching;    // 是否启用自适应批处理
    double target_latency_ms;         // 目标延迟（毫秒）
} optimizer_config_t;

// 优化器统计信息
typedef struct {
    uint64_t total_requests_submitted; // 总提交请求数
    uint64_t total_requests_processed; // 总处理请求数
    uint64_t total_batches_executed;   // 总执行批次数
    uint64_t total_execution_time_us;  // 总执行时间
    double average_batch_size;         // 平均批次大小
    double average_execution_time_us;  // 平均执行时间
    double average_latency_ms;         // 平均延迟
    double throughput_requests_per_sec; // 吞吐量（请求/秒）
} optimizer_stats_t;

// 延迟优化器结构
typedef struct {
    optimizer_config_t config;        // 配置
    optimizer_stats_t stats;          // 统计信息
    bool is_running;                  // 运行状态
    pthread_t batch_processor_thread; // 批处理器线程
    pthread_mutex_t stats_mutex;      // 统计信息互斥锁
} latency_optimizer_t;

// 时间片调度结构
typedef struct {
    int device_id;                    // 设备ID
    uint64_t time_slice_us;           // 时间片长度（微秒）
    uint64_t remaining_time_us;       // 剩余时间
    request_priority_t current_priority; // 当前优先级
    bool is_preemptible;              // 是否可抢占
} time_slice_scheduler_t;

// 核心API函数

/**
 * 初始化延迟优化器
 * @param config 优化器配置
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t latency_optimizer_init(const optimizer_config_t *config);

/**
 * 清理延迟优化器
 */
void latency_optimizer_cleanup(void);

/**
 * 提交GPU请求
 * @param params 请求参数
 * @param callback 完成回调函数
 * @param user_data 用户数据
 * @param request_id 输出请求ID
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t latency_optimizer_submit_request(const request_params_t *params,
                                                   request_callback_t callback,
                                                   void *user_data,
                                                   uint64_t *request_id);

/**
 * 取消GPU请求
 * @param request_id 请求ID
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t latency_optimizer_cancel_request(uint64_t request_id);

/**
 * 获取请求状态
 * @param request_id 请求ID
 * @param status 输出请求状态
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t latency_optimizer_get_request_status(uint64_t request_id,
                                                       request_status_t *status);

/**
 * 获取优化器统计信息
 * @param stats 输出统计信息
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t latency_optimizer_get_stats(optimizer_stats_t *stats);

// 批处理管理函数

/**
 * 执行LRU淘汰策略
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t cache_evict_lru(void);

/**
 * 创建请求批次
 * @param device_id 设备ID
 * @param requests 请求数组
 * @param request_count 请求数量
 * @param batch 输出批次
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t create_request_batch(int device_id,
                                       gpu_request_t **requests,
                                       uint32_t request_count,
                                       request_batch_t **batch);

/**
 * 执行请求批次
 * @param batch 请求批次
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t execute_request_batch(request_batch_t *batch);

// 时间片调度函数

/**
 * 初始化时间片调度器
 * @param device_id 设备ID
 * @param time_slice_us 时间片长度（微秒）
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t schedule_time_slice_init(int device_id, uint64_t time_slice_us);

/**
 * 调度下一个时间片
 * @param device_id 设备ID
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t schedule_time_slice(int device_id);

/**
 * 抢占当前时间片
 * @param device_id 设备ID
 * @param priority 新的优先级
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t preempt_time_slice(int device_id, request_priority_t priority);

// 负载均衡函数

/**
 * 选择最优设备
 * @param requirements 设备要求
 * @param device_id 输出设备ID
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t select_optimal_device(const request_params_t *requirements,
                                        int *device_id);

/**
 * 获取设备负载
 * @param device_id 设备ID
 * @param load 输出负载值（0.0-1.0）
 * @return OPTIMIZER_SUCCESS 成功，其他值表示错误
 */
optimizer_result_t get_device_load(int device_id, double *load);

// 错误处理函数

/**
 * 获取错误字符串
 * @param error 错误码
 * @return 错误描述字符串
 */
const char* optimizer_get_error_string(optimizer_result_t error);

// 工具宏

#define OPTIMIZER_CHECK(call) do { \
    optimizer_result_t result = (call); \
    if (result != OPTIMIZER_SUCCESS) { \
        fprintf(stderr, "Optimizer error in %s: %s\n", #call, \
                optimizer_get_error_string(result)); \
        return result; \
    } \
} while(0)

#define OPTIMIZER_CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Null pointer: %s\n", #ptr); \
        return OPTIMIZER_ERROR_INVALID_PARAM; \
    } \
} while(0)

// 默认配置值
#define DEFAULT_MAX_BATCH_SIZE 32
#define DEFAULT_BATCH_TIMEOUT_US 10000
#define DEFAULT_CHECK_INTERVAL_US 1000
#define DEFAULT_TARGET_LATENCY_MS 10.0

// 创建默认优化器配置的辅助宏
#define OPTIMIZER_CONFIG_DEFAULT() { \
    .max_batch_size = DEFAULT_MAX_BATCH_SIZE, \
    .batch_timeout_us = DEFAULT_BATCH_TIMEOUT_US, \
    .check_interval_us = DEFAULT_CHECK_INTERVAL_US, \
    .batch_strategy = BATCH_STRATEGY_ADAPTIVE, \
    .enable_priority_scheduling = true, \
    .enable_load_balancing = true, \
    .enable_adaptive_batching = true, \
    .target_latency_ms = DEFAULT_TARGET_LATENCY_MS \
}

// 创建请求参数的辅助宏
#define REQUEST_PARAMS_INIT(dev_id, op_type, data_ptr, data_sz, prio) { \
    .device_id = dev_id, \
    .operation_type = op_type, \
    .data = data_ptr, \
    .data_size = data_sz, \
    .priority = prio, \
    .timeout_ms = 0, \
    .kernel_params = NULL, \
    .kernel_params_size = 0 \
}

#ifdef __cplusplus
}
#endif

#endif // LATENCY_OPTIMIZER_H