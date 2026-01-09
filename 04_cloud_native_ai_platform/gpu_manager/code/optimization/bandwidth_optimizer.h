#ifndef BANDWIDTH_OPTIMIZER_H
#define BANDWIDTH_OPTIMIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// 常量定义
#define MAX_TRANSFER_CHANNELS 16
#define MAX_CONCURRENT_TRANSFERS 32
#define MIN_COMPRESSION_SIZE 1024

// 带宽优化器结果枚举
typedef enum {
    BANDWIDTH_SUCCESS = 0,
    BANDWIDTH_ERROR_INVALID_PARAM,
    BANDWIDTH_ERROR_NOT_INITIALIZED,
    BANDWIDTH_ERROR_OUT_OF_MEMORY,
    BANDWIDTH_ERROR_NO_AVAILABLE_CHANNEL,
    BANDWIDTH_ERROR_MAX_CHANNELS_REACHED,
    BANDWIDTH_ERROR_TRANSFER_FAILED,
    BANDWIDTH_ERROR_INITIALIZATION_FAILED
} bandwidth_result_t;

// 压缩结果枚举
typedef enum {
    COMPRESSION_SUCCESS = 0,
    COMPRESSION_ERROR_INVALID_PARAM,
    COMPRESSION_ERROR_OUT_OF_MEMORY,
    COMPRESSION_ERROR_COMPRESSION_FAILED,
    COMPRESSION_ERROR_DECOMPRESSION_FAILED
} compression_result_t;

// 传输状态枚举
typedef enum {
    TRANSFER_STATUS_PENDING = 0,
    TRANSFER_STATUS_COMPRESSING,
    TRANSFER_STATUS_TRANSFERRING,
    TRANSFER_STATUS_COMPLETED,
    TRANSFER_STATUS_FAILED,
    TRANSFER_STATUS_CANCELLED
} transfer_status_t;

// 传输类型枚举
typedef enum {
    TRANSFER_TYPE_HOST_TO_DEVICE = 0,
    TRANSFER_TYPE_DEVICE_TO_HOST,
    TRANSFER_TYPE_DEVICE_TO_DEVICE,
    TRANSFER_TYPE_HOST_TO_HOST
} transfer_type_t;

// 压缩算法枚举
typedef enum {
    COMPRESSION_NONE = 0,
    COMPRESSION_ZLIB,
    COMPRESSION_LZ4,
    COMPRESSION_SNAPPY,
    COMPRESSION_ADAPTIVE
} compression_algorithm_t;

// 传输进度回调函数类型
typedef void (*transfer_progress_callback_t)(uint64_t transfer_id, double progress, void *user_data);

// 传输完成回调函数类型
typedef void (*transfer_completion_callback_t)(uint64_t transfer_id, bandwidth_result_t result, void *user_data);

// 带宽信息结构
typedef struct {
    double max_bandwidth_mbps;        // 最大带宽（MB/s）
    double current_bandwidth_mbps;    // 当前带宽（MB/s）
    double average_latency_ms;        // 平均延迟（毫秒）
    double packet_loss_rate;          // 丢包率
    uint64_t total_bytes_transferred; // 总传输字节数
    uint64_t total_transfer_time_us;  // 总传输时间（微秒）
} bandwidth_info_t;

// 传输参数结构
typedef struct {
    int source_device_id;             // 源设备ID（-1表示主机）
    int dest_device_id;               // 目标设备ID（-1表示主机）
    transfer_type_t transfer_type;    // 传输类型
    void *data;                       // 数据指针
    size_t data_size;                 // 数据大小
    bool enable_compression;          // 是否启用压缩
    compression_algorithm_t compression_algo; // 压缩算法
    uint32_t priority;                // 传输优先级
    transfer_progress_callback_t progress_callback;   // 进度回调
    transfer_completion_callback_t completion_callback; // 完成回调
    void *user_data;                  // 用户数据
} transfer_params_t;

// 数据传输结构
typedef struct {
    uint64_t transfer_id;             // 传输唯一ID
    transfer_params_t params;         // 传输参数
    transfer_status_t status;         // 传输状态
    uint64_t created_time;            // 创建时间戳
    uint64_t start_time;              // 开始时间戳
    uint64_t completion_time;         // 完成时间戳
    uint64_t actual_duration_us;      // 实际传输时间
    double progress;                  // 传输进度（0.0-1.0）
    bool is_compressed;               // 是否已压缩
    void *compressed_data;            // 压缩后的数据
    size_t compressed_size;           // 压缩后的大小
    double compression_ratio;         // 压缩比
} data_transfer_t;

// 传输通道结构
typedef struct {
    int channel_id;                   // 通道ID
    int source_device_id;             // 源设备ID
    int dest_device_id;               // 目标设备ID
    bandwidth_info_t bandwidth_info;  // 带宽信息
    double current_load;              // 当前负载（0.0-1.0）
    bool is_active;                   // 是否激活
    pthread_mutex_t mutex;            // 通道互斥锁
} transfer_channel_t;

// 带宽优化器配置结构
typedef struct {
    uint32_t max_concurrent_transfers;    // 最大并发传输数
    size_t compression_threshold;         // 压缩阈值（字节）
    compression_algorithm_t default_compression; // 默认压缩算法
    bool enable_adaptive_compression;     // 是否启用自适应压缩
    bool enable_load_balancing;          // 是否启用负载均衡
    double target_bandwidth_mbps;        // 目标带宽（MB/s）
    double compression_cpu_threshold;    // 压缩CPU阈值
    uint32_t max_chunk_size;             // 最大块大小
    uint32_t min_chunk_size;             // 最小块大小
} bandwidth_config_t;

// 带宽优化器统计信息
typedef struct {
    uint64_t total_transfers_submitted;   // 总提交传输数
    uint64_t total_transfers_completed;   // 总完成传输数
    uint64_t total_bytes_transferred;     // 总传输字节数
    uint64_t total_bytes_compressed;      // 总压缩字节数
    uint64_t total_compression_savings;   // 总压缩节省字节数
    uint64_t total_transfer_time_us;      // 总传输时间
    double average_transfer_time_us;      // 平均传输时间
    double average_bandwidth_mbps;        // 平均带宽
    double compression_ratio;             // 平均压缩比
    double cpu_usage_percent;             // CPU使用率
} bandwidth_stats_t;

// 带宽优化器结构
typedef struct {
    bandwidth_config_t config;        // 配置
    bandwidth_stats_t stats;          // 统计信息
    bool is_running;                  // 运行状态
    pthread_mutex_t stats_mutex;      // 统计信息互斥锁
} bandwidth_optimizer_t;

// 数据压缩实现结构
typedef struct {
    compression_algorithm_t algorithm; // 压缩算法
    uint32_t compression_level;       // 压缩级别
    size_t input_size;                // 输入大小
    size_t output_size;               // 输出大小
    uint64_t compression_time_us;     // 压缩时间
    double compression_ratio;         // 压缩比
    bool is_beneficial;               // 是否有益
} compression_info_t;

// 核心API函数

/**
 * 初始化带宽优化器
 * @param config 带宽优化器配置
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t bandwidth_optimizer_init(const bandwidth_config_t *config);

/**
 * 清理带宽优化器
 */
void bandwidth_optimizer_cleanup(void);

/**
 * 提交数据传输任务
 * @param params 传输参数
 * @param transfer_id 输出传输ID
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t bandwidth_optimizer_submit_transfer(const transfer_params_t *params,
                                                      uint64_t *transfer_id);

/**
 * 取消数据传输
 * @param transfer_id 传输ID
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t bandwidth_optimizer_cancel_transfer(uint64_t transfer_id);

/**
 * 获取传输状态
 * @param transfer_id 传输ID
 * @param status 输出传输状态
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t bandwidth_optimizer_get_transfer_status(uint64_t transfer_id,
                                                          transfer_status_t *status);

/**
 * 获取带宽优化器统计信息
 * @param stats 输出统计信息
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t bandwidth_optimizer_get_stats(bandwidth_stats_t *stats);

// 传输通道管理函数

/**
 * 注册传输通道
 * @param source_device_id 源设备ID
 * @param dest_device_id 目标设备ID
 * @param max_bandwidth_mbps 最大带宽
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t bandwidth_optimizer_register_channel(int source_device_id, 
                                                       int dest_device_id,
                                                       double max_bandwidth_mbps);

/**
 * 注销传输通道
 * @param channel_id 通道ID
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t bandwidth_optimizer_unregister_channel(int channel_id);

/**
 * 获取通道信息
 * @param channel_id 通道ID
 * @param channel_info 输出通道信息
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t bandwidth_optimizer_get_channel_info(int channel_id,
                                                       transfer_channel_t *channel_info);

// 数据压缩函数

/**
 * 压缩数据
 * @param algorithm 压缩算法
 * @param input_data 输入数据
 * @param input_size 输入大小
 * @param output_data 输出数据指针
 * @param output_size 输出大小指针
 * @return COMPRESSION_SUCCESS 成功，其他值表示错误
 */
compression_result_t compress_data(compression_algorithm_t algorithm,
                                  const void *input_data, size_t input_size,
                                  void **output_data, size_t *output_size);

/**
 * 解压缩数据
 * @param algorithm 压缩算法
 * @param input_data 输入数据
 * @param input_size 输入大小
 * @param output_data 输出数据指针
 * @param expected_size 期望输出大小
 * @return COMPRESSION_SUCCESS 成功，其他值表示错误
 */
compression_result_t decompress_data(compression_algorithm_t algorithm,
                                    const void *input_data, size_t input_size,
                                    void **output_data, size_t expected_size);

/**
 * 评估压缩效果
 * @param data 数据指针
 * @param size 数据大小
 * @param algorithm 压缩算法
 * @param info 输出压缩信息
 * @return COMPRESSION_SUCCESS 成功，其他值表示错误
 */
compression_result_t evaluate_compression(const void *data, size_t size,
                                         compression_algorithm_t algorithm,
                                         compression_info_t *info);

// 负载均衡函数

/**
 * 选择最优传输通道
 * @param params 传输参数
 * @param channel_id 输出通道ID
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t select_optimal_channel(const transfer_params_t *params, int *channel_id);

/**
 * 更新通道负载
 * @param channel_id 通道ID
 * @param load 负载值（0.0-1.0）
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t update_channel_load(int channel_id, double load);

/**
 * 获取系统总负载
 * @param total_load 输出总负载
 * @return BANDWIDTH_SUCCESS 成功，其他值表示错误
 */
bandwidth_result_t get_system_load(double *total_load);

// 错误处理函数

/**
 * 获取错误字符串
 * @param error 错误码
 * @return 错误描述字符串
 */
const char* bandwidth_get_error_string(bandwidth_result_t error);

/**
 * 获取压缩错误字符串
 * @param error 压缩错误码
 * @return 错误描述字符串
 */
const char* compression_get_error_string(compression_result_t error);

// 工具宏

#define BANDWIDTH_CHECK(call) do { \
    bandwidth_result_t result = (call); \
    if (result != BANDWIDTH_SUCCESS) { \
        fprintf(stderr, "Bandwidth error in %s: %s\n", #call, \
                bandwidth_get_error_string(result)); \
        return result; \
    } \
} while(0)

#define BANDWIDTH_CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Null pointer: %s\n", #ptr); \
        return BANDWIDTH_ERROR_INVALID_PARAM; \
    } \
} while(0)

// 默认配置值
#define DEFAULT_MAX_CONCURRENT_TRANSFERS 4
#define DEFAULT_COMPRESSION_THRESHOLD (1024 * 1024)  // 1MB
#define DEFAULT_TARGET_BANDWIDTH_MBPS 1000.0
#define DEFAULT_MAX_CHUNK_SIZE (16 * 1024 * 1024)    // 16MB
#define DEFAULT_MIN_CHUNK_SIZE (64 * 1024)           // 64KB

// 创建默认带宽配置的辅助宏
#define BANDWIDTH_CONFIG_DEFAULT() { \
    .max_concurrent_transfers = DEFAULT_MAX_CONCURRENT_TRANSFERS, \
    .compression_threshold = DEFAULT_COMPRESSION_THRESHOLD, \
    .default_compression = COMPRESSION_ZLIB, \
    .enable_adaptive_compression = true, \
    .enable_load_balancing = true, \
    .target_bandwidth_mbps = DEFAULT_TARGET_BANDWIDTH_MBPS, \
    .compression_cpu_threshold = 0.8, \
    .max_chunk_size = DEFAULT_MAX_CHUNK_SIZE, \
    .min_chunk_size = DEFAULT_MIN_CHUNK_SIZE \
}

// 创建传输参数的辅助宏
#define TRANSFER_PARAMS_INIT(src_dev, dst_dev, data_ptr, data_sz) { \
    .source_device_id = src_dev, \
    .dest_device_id = dst_dev, \
    .transfer_type = TRANSFER_TYPE_HOST_TO_DEVICE, \
    .data = data_ptr, \
    .data_size = data_sz, \
    .enable_compression = true, \
    .compression_algo = COMPRESSION_ADAPTIVE, \
    .priority = 1, \
    .progress_callback = NULL, \
    .completion_callback = NULL, \
    .user_data = NULL \
}

#ifdef __cplusplus
}
#endif

#endif // BANDWIDTH_OPTIMIZER_H