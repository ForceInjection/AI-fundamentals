#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
#include <zlib.h>

#include "bandwidth_optimizer.h"
#include "../common/gpu_common.h"

// 带宽优化器全局实例
static bandwidth_optimizer_t g_bandwidth_optimizer = {0};
static bool g_bandwidth_optimizer_initialized = false;
static pthread_mutex_t g_bandwidth_mutex = PTHREAD_MUTEX_INITIALIZER;

// 传输通道管理
static transfer_channel_t g_transfer_channels[MAX_TRANSFER_CHANNELS];
static pthread_mutex_t g_channel_mutexes[MAX_TRANSFER_CHANNELS];

// 数据压缩实现
static compression_result_t compress_data_zlib(const void *input_data, size_t input_size,
                                              void **output_data, size_t *output_size) {
    if (!input_data || input_size == 0 || !output_data || !output_size) {
        return COMPRESSION_ERROR_INVALID_PARAM;
    }
    
    // 估算压缩后的最大大小
    uLongf compressed_size = compressBound(input_size);
    *output_data = malloc(compressed_size);
    if (!*output_data) {
        return COMPRESSION_ERROR_OUT_OF_MEMORY;
    }
    
    // 执行压缩
    int result = compress2((Bytef*)*output_data, &compressed_size,
                          (const Bytef*)input_data, input_size,
                          Z_DEFAULT_COMPRESSION);
    
    if (result != Z_OK) {
        free(*output_data);
        *output_data = NULL;
        return COMPRESSION_ERROR_COMPRESSION_FAILED;
    }
    
    *output_size = compressed_size;
    
    gpu_log(GPU_LOG_DEBUG, "Data compressed: %zu -> %zu bytes (%.1f%% reduction)",
            input_size, compressed_size, 
            100.0 * (1.0 - (double)compressed_size / input_size));
    
    return COMPRESSION_SUCCESS;
}

// 数据解压缩实现
static compression_result_t decompress_data_zlib(const void *input_data, size_t input_size,
                                                void **output_data, size_t expected_size) {
    if (!input_data || input_size == 0 || !output_data || expected_size == 0) {
        return COMPRESSION_ERROR_INVALID_PARAM;
    }
    
    *output_data = malloc(expected_size);
    if (!*output_data) {
        return COMPRESSION_ERROR_OUT_OF_MEMORY;
    }
    
    uLongf decompressed_size = expected_size;
    int result = uncompress((Bytef*)*output_data, &decompressed_size,
                           (const Bytef*)input_data, input_size);
    
    if (result != Z_OK || decompressed_size != expected_size) {
        free(*output_data);
        *output_data = NULL;
        return COMPRESSION_ERROR_DECOMPRESSION_FAILED;
    }
    
    gpu_log(GPU_LOG_DEBUG, "Data decompressed: %zu -> %zu bytes",
            input_size, decompressed_size);
    
    return COMPRESSION_SUCCESS;
}

// 计算数据传输的最优块大小
static size_t calculate_optimal_chunk_size(size_t total_size, 
                                          bandwidth_info_t *bandwidth_info) {
    // 基于带宽和延迟计算最优块大小
    double bandwidth_mbps = bandwidth_info->current_bandwidth_mbps;
    double latency_ms = bandwidth_info->average_latency_ms;
    
    // 目标：使传输时间 >> 延迟时间，以提高效率
    // 块大小 = 带宽 * 目标传输时间
    double target_transfer_time_ms = latency_ms * 10; // 10倍延迟时间
    size_t optimal_size = (size_t)(bandwidth_mbps * target_transfer_time_ms * 1024 / 8);
    
    // 限制在合理范围内
    size_t min_chunk = 64 * 1024;    // 64KB
    size_t max_chunk = 16 * 1024 * 1024; // 16MB
    
    if (optimal_size < min_chunk) {
        optimal_size = min_chunk;
    } else if (optimal_size > max_chunk) {
        optimal_size = max_chunk;
    }
    
    // 确保不超过总大小
    if (optimal_size > total_size) {
        optimal_size = total_size;
    }
    
    return optimal_size;
}

// 创建数据传输任务
static data_transfer_t* create_transfer_task(const transfer_params_t *params) {
    data_transfer_t *transfer = malloc(sizeof(data_transfer_t));
    if (!transfer) {
        return NULL;
    }
    
    memset(transfer, 0, sizeof(data_transfer_t));
    
    // 生成唯一ID
    static uint64_t transfer_id_counter = 0;
    transfer->transfer_id = __sync_add_and_fetch(&transfer_id_counter, 1);
    
    transfer->params = *params;
    transfer->status = TRANSFER_STATUS_PENDING;
    transfer->created_time = gpu_get_timestamp_us();
    transfer->progress = 0.0;
    
    return transfer;
}

// 释放数据传输任务
static void free_transfer_task(data_transfer_t *transfer) {
    if (transfer) {
        if (transfer->compressed_data && transfer->compressed_data != transfer->params.data) {
            free(transfer->compressed_data);
        }
        free(transfer);
    }
}

// 执行数据压缩
static bandwidth_result_t perform_compression(data_transfer_t *transfer) {
    if (!g_bandwidth_optimizer.config.enable_compression) {
        return BANDWIDTH_SUCCESS; // 压缩未启用
    }
    
    // 检查数据大小是否值得压缩
    if (transfer->params.data_size < g_bandwidth_optimizer.config.compression_threshold) {
        return BANDWIDTH_SUCCESS; // 数据太小，不值得压缩
    }
    
    compression_result_t result = compress_data_zlib(
        transfer->params.data, transfer->params.data_size,
        &transfer->compressed_data, &transfer->compressed_size
    );
    
    if (result != COMPRESSION_SUCCESS) {
        gpu_log(GPU_LOG_WARNING, "Compression failed for transfer %lu", transfer->transfer_id);
        return BANDWIDTH_SUCCESS; // 压缩失败，使用原始数据
    }
    
    // 检查压缩效果
    double compression_ratio = (double)transfer->compressed_size / transfer->params.data_size;
    if (compression_ratio > 0.9) {
        // 压缩效果不好，使用原始数据
        free(transfer->compressed_data);
        transfer->compressed_data = NULL;
        transfer->compressed_size = 0;
        return BANDWIDTH_SUCCESS;
    }
    
    transfer->is_compressed = true;
    
    // 更新统计信息
    pthread_mutex_lock(&g_bandwidth_mutex);
    g_bandwidth_optimizer.stats.total_bytes_compressed += transfer->params.data_size;
    g_bandwidth_optimizer.stats.total_compression_savings += 
        (transfer->params.data_size - transfer->compressed_size);
    pthread_mutex_unlock(&g_bandwidth_mutex);
    
    gpu_log(GPU_LOG_DEBUG, "Transfer %lu compressed: %zu -> %zu bytes (%.1f%% reduction)",
            transfer->transfer_id, transfer->params.data_size, transfer->compressed_size,
            100.0 * (1.0 - compression_ratio));
    
    return BANDWIDTH_SUCCESS;
}

// 执行分块传输
static bandwidth_result_t perform_chunked_transfer(data_transfer_t *transfer, 
                                                  transfer_channel_t *channel) {
    const void *data_to_transfer = transfer->is_compressed ? 
                                  transfer->compressed_data : transfer->params.data;
    size_t total_size = transfer->is_compressed ? 
                       transfer->compressed_size : transfer->params.data_size;
    
    size_t chunk_size = calculate_optimal_chunk_size(total_size, &channel->bandwidth_info);
    size_t transferred = 0;
    
    transfer->status = TRANSFER_STATUS_TRANSFERRING;
    transfer->start_time = gpu_get_timestamp_us();
    
    gpu_log(GPU_LOG_DEBUG, "Starting chunked transfer %lu: %zu bytes in %zu-byte chunks",
            transfer->transfer_id, total_size, chunk_size);
    
    while (transferred < total_size) {
        size_t current_chunk_size = GPU_MIN(chunk_size, total_size - transferred);
        const char *chunk_data = (const char*)data_to_transfer + transferred;
        
        // 模拟数据传输
        uint64_t chunk_start = gpu_get_timestamp_us();
        
        // 在实际实现中，这里会调用GPU驱动API进行数据传输
        // 这里用sleep模拟传输时间
        double transfer_time_ms = (double)current_chunk_size / 
                                 (channel->bandwidth_info.current_bandwidth_mbps * 1024 / 8);
        usleep((useconds_t)(transfer_time_ms * 1000));
        
        uint64_t chunk_end = gpu_get_timestamp_us();
        uint64_t chunk_duration = chunk_end - chunk_start;
        
        transferred += current_chunk_size;
        transfer->progress = (double)transferred / total_size;
        
        // 更新带宽统计
        double actual_bandwidth = (double)current_chunk_size / (chunk_duration / 1000000.0) / (1024 * 1024);
        channel->bandwidth_info.current_bandwidth_mbps = 
            0.9 * channel->bandwidth_info.current_bandwidth_mbps + 0.1 * actual_bandwidth;
        
        // 调用进度回调
        if (transfer->params.progress_callback) {
            transfer->params.progress_callback(transfer->transfer_id, transfer->progress, 
                                             transfer->params.user_data);
        }
        
        gpu_log(GPU_LOG_DEBUG, "Transfer %lu: chunk %zu/%zu completed (%.1f%% total)",
                transfer->transfer_id, transferred / chunk_size, 
                (total_size + chunk_size - 1) / chunk_size, transfer->progress * 100.0);
    }
    
    transfer->completion_time = gpu_get_timestamp_us();
    transfer->actual_duration_us = transfer->completion_time - transfer->start_time;
    transfer->status = TRANSFER_STATUS_COMPLETED;
    
    // 更新统计信息
    pthread_mutex_lock(&g_bandwidth_mutex);
    g_bandwidth_optimizer.stats.total_transfers_completed++;
    g_bandwidth_optimizer.stats.total_bytes_transferred += total_size;
    g_bandwidth_optimizer.stats.total_transfer_time_us += transfer->actual_duration_us;
    pthread_mutex_unlock(&g_bandwidth_mutex);
    
    gpu_log(GPU_LOG_INFO, "Transfer %lu completed: %zu bytes in %lu us (%.2f MB/s)",
            transfer->transfer_id, total_size, transfer->actual_duration_us,
            (double)total_size / (transfer->actual_duration_us / 1000000.0) / (1024 * 1024));
    
    return BANDWIDTH_SUCCESS;
}

// 选择最优传输通道
static int select_optimal_channel(const transfer_params_t *params) {
    int best_channel = -1;
    double best_score = -1.0;
    
    for (int i = 0; i < MAX_TRANSFER_CHANNELS; i++) {
        transfer_channel_t *channel = &g_transfer_channels[i];
        
        if (!channel->is_active) {
            continue;
        }
        
        // 检查设备兼容性
        if (params->source_device_id >= 0 && 
            channel->source_device_id != params->source_device_id) {
            continue;
        }
        if (params->dest_device_id >= 0 && 
            channel->dest_device_id != params->dest_device_id) {
            continue;
        }
        
        // 计算通道评分（基于带宽、延迟和当前负载）
        double bandwidth_score = channel->bandwidth_info.current_bandwidth_mbps / 
                                channel->bandwidth_info.max_bandwidth_mbps;
        double latency_score = 1.0 / (1.0 + channel->bandwidth_info.average_latency_ms);
        double load_score = 1.0 - channel->current_load;
        
        double total_score = bandwidth_score * 0.4 + latency_score * 0.3 + load_score * 0.3;
        
        if (total_score > best_score) {
            best_score = total_score;
            best_channel = i;
        }
    }
    
    return best_channel;
}

// 初始化带宽优化器
bandwidth_result_t bandwidth_optimizer_init(const bandwidth_config_t *config) {
    if (!config) {
        return BANDWIDTH_ERROR_INVALID_PARAM;
    }
    
    if (g_bandwidth_optimizer_initialized) {
        gpu_log(GPU_LOG_WARNING, "Bandwidth optimizer already initialized");
        return BANDWIDTH_SUCCESS;
    }
    
    memset(&g_bandwidth_optimizer, 0, sizeof(g_bandwidth_optimizer));
    g_bandwidth_optimizer.config = *config;
    
    // 验证配置参数
    if (g_bandwidth_optimizer.config.max_concurrent_transfers == 0) {
        g_bandwidth_optimizer.config.max_concurrent_transfers = 4;
    }
    if (g_bandwidth_optimizer.config.compression_threshold == 0) {
        g_bandwidth_optimizer.config.compression_threshold = 1024 * 1024; // 1MB
    }
    if (g_bandwidth_optimizer.config.target_bandwidth_mbps == 0) {
        g_bandwidth_optimizer.config.target_bandwidth_mbps = 1000.0; // 1GB/s
    }
    
    // 初始化传输通道
    for (int i = 0; i < MAX_TRANSFER_CHANNELS; i++) {
        memset(&g_transfer_channels[i], 0, sizeof(transfer_channel_t));
        g_transfer_channels[i].channel_id = i;
        g_transfer_channels[i].is_active = false;
        
        if (pthread_mutex_init(&g_channel_mutexes[i], NULL) != 0) {
            gpu_log(GPU_LOG_ERROR, "Failed to initialize channel mutex %d", i);
            return BANDWIDTH_ERROR_INITIALIZATION_FAILED;
        }
    }
    
    g_bandwidth_optimizer_initialized = true;
    
    gpu_log(GPU_LOG_INFO, "Bandwidth optimizer initialized successfully");
    return BANDWIDTH_SUCCESS;
}

// 清理带宽优化器
void bandwidth_optimizer_cleanup(void) {
    if (!g_bandwidth_optimizer_initialized) {
        return;
    }
    
    gpu_log(GPU_LOG_INFO, "Stopping bandwidth optimizer...");
    
    // 清理传输通道
    for (int i = 0; i < MAX_TRANSFER_CHANNELS; i++) {
        pthread_mutex_destroy(&g_channel_mutexes[i]);
    }
    
    memset(&g_bandwidth_optimizer, 0, sizeof(g_bandwidth_optimizer));
    g_bandwidth_optimizer_initialized = false;
    
    gpu_log(GPU_LOG_INFO, "Bandwidth optimizer stopped");
}

// 提交数据传输任务
bandwidth_result_t bandwidth_optimizer_submit_transfer(const transfer_params_t *params,
                                                      uint64_t *transfer_id) {
    if (!g_bandwidth_optimizer_initialized || !params || !transfer_id) {
        return BANDWIDTH_ERROR_INVALID_PARAM;
    }
    
    data_transfer_t *transfer = create_transfer_task(params);
    if (!transfer) {
        return BANDWIDTH_ERROR_OUT_OF_MEMORY;
    }
    
    *transfer_id = transfer->transfer_id;
    
    // 选择最优传输通道
    int channel_id = select_optimal_channel(params);
    if (channel_id < 0) {
        free_transfer_task(transfer);
        return BANDWIDTH_ERROR_NO_AVAILABLE_CHANNEL;
    }
    
    transfer_channel_t *channel = &g_transfer_channels[channel_id];
    
    pthread_mutex_lock(&g_channel_mutexes[channel_id]);
    
    // 执行压缩（如果启用）
    bandwidth_result_t result = perform_compression(transfer);
    if (result != BANDWIDTH_SUCCESS) {
        pthread_mutex_unlock(&g_channel_mutexes[channel_id]);
        free_transfer_task(transfer);
        return result;
    }
    
    // 执行分块传输
    result = perform_chunked_transfer(transfer, channel);
    
    // 调用完成回调
    if (transfer->params.completion_callback) {
        transfer->params.completion_callback(transfer->transfer_id, 
                                           transfer->status == TRANSFER_STATUS_COMPLETED ? 
                                           BANDWIDTH_SUCCESS : BANDWIDTH_ERROR_TRANSFER_FAILED,
                                           transfer->params.user_data);
    }
    
    pthread_mutex_unlock(&g_channel_mutexes[channel_id]);
    
    free_transfer_task(transfer);
    
    gpu_log(GPU_LOG_DEBUG, "Transfer %lu submitted and completed", *transfer_id);
    
    return result;
}

// 注册传输通道
bandwidth_result_t bandwidth_optimizer_register_channel(int source_device_id, 
                                                       int dest_device_id,
                                                       double max_bandwidth_mbps) {
    if (!g_bandwidth_optimizer_initialized) {
        return BANDWIDTH_ERROR_NOT_INITIALIZED;
    }
    
    // 查找空闲通道
    int channel_id = -1;
    for (int i = 0; i < MAX_TRANSFER_CHANNELS; i++) {
        if (!g_transfer_channels[i].is_active) {
            channel_id = i;
            break;
        }
    }
    
    if (channel_id < 0) {
        return BANDWIDTH_ERROR_MAX_CHANNELS_REACHED;
    }
    
    transfer_channel_t *channel = &g_transfer_channels[channel_id];
    
    pthread_mutex_lock(&g_channel_mutexes[channel_id]);
    
    channel->source_device_id = source_device_id;
    channel->dest_device_id = dest_device_id;
    channel->bandwidth_info.max_bandwidth_mbps = max_bandwidth_mbps;
    channel->bandwidth_info.current_bandwidth_mbps = max_bandwidth_mbps * 0.8; // 初始估算
    channel->bandwidth_info.average_latency_ms = 1.0; // 初始估算
    channel->current_load = 0.0;
    channel->is_active = true;
    
    pthread_mutex_unlock(&g_channel_mutexes[channel_id]);
    
    gpu_log(GPU_LOG_INFO, "Registered transfer channel %d: device %d -> %d (%.2f MB/s)",
            channel_id, source_device_id, dest_device_id, max_bandwidth_mbps);
    
    return BANDWIDTH_SUCCESS;
}

// 获取带宽优化器统计信息
bandwidth_result_t bandwidth_optimizer_get_stats(bandwidth_stats_t *stats) {
    if (!g_bandwidth_optimizer_initialized || !stats) {
        return BANDWIDTH_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_bandwidth_mutex);
    
    *stats = g_bandwidth_optimizer.stats;
    
    // 计算平均值
    if (stats->total_transfers_completed > 0) {
        stats->average_transfer_time_us = stats->total_transfer_time_us / 
                                         stats->total_transfers_completed;
        stats->average_bandwidth_mbps = (double)stats->total_bytes_transferred / 
                                       (stats->total_transfer_time_us / 1000000.0) / 
                                       (1024 * 1024);
    }
    
    if (stats->total_bytes_compressed > 0) {
        stats->compression_ratio = (double)stats->total_compression_savings / 
                                  stats->total_bytes_compressed;
    }
    
    pthread_mutex_unlock(&g_bandwidth_mutex);
    
    return BANDWIDTH_SUCCESS;
}

// 获取错误字符串
const char* bandwidth_get_error_string(bandwidth_result_t error) {
    static const char* error_strings[] = {
        [BANDWIDTH_SUCCESS] = "Success",
        [BANDWIDTH_ERROR_INVALID_PARAM] = "Invalid parameter",
        [BANDWIDTH_ERROR_NOT_INITIALIZED] = "Bandwidth optimizer not initialized",
        [BANDWIDTH_ERROR_OUT_OF_MEMORY] = "Out of memory",
        [BANDWIDTH_ERROR_NO_AVAILABLE_CHANNEL] = "No available transfer channel",
        [BANDWIDTH_ERROR_MAX_CHANNELS_REACHED] = "Maximum channels reached",
        [BANDWIDTH_ERROR_TRANSFER_FAILED] = "Transfer failed",
        [BANDWIDTH_ERROR_INITIALIZATION_FAILED] = "Initialization failed"
    };
    
    if (error >= 0 && error < sizeof(error_strings) / sizeof(error_strings[0])) {
        return error_strings[error];
    }
    return "Unknown error";
}