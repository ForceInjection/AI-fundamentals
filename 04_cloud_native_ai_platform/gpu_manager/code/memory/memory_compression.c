/**
 * GPU内存压缩算法实现
 * 功能：实现多种内存压缩算法，提高内存利用率
 * 支持：LZ4、ZSTD、Snappy等压缩算法
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>

// 压缩算法类型
typedef enum {
    COMPRESS_NONE = 0,
    COMPRESS_LZ4,
    COMPRESS_ZSTD,
    COMPRESS_SNAPPY,
    COMPRESS_ADAPTIVE
} compression_algorithm_t;

// 压缩质量等级
typedef enum {
    QUALITY_FAST = 1,      // 快速压缩
    QUALITY_BALANCED = 5,  // 平衡模式
    QUALITY_BEST = 9       // 最佳压缩
} compression_quality_t;

// 压缩统计信息
typedef struct {
    uint64_t total_compressions;    // 总压缩次数
    uint64_t total_decompressions;  // 总解压次数
    uint64_t total_input_bytes;     // 总输入字节数
    uint64_t total_output_bytes;    // 总输出字节数
    uint64_t compression_time_us;   // 压缩总时间（微秒）
    uint64_t decompression_time_us; // 解压总时间（微秒）
    double avg_compression_ratio;   // 平均压缩比
    uint32_t compression_failures;  // 压缩失败次数
} compression_stats_t;

// 压缩上下文
typedef struct {
    compression_algorithm_t algorithm;
    compression_quality_t quality;
    size_t block_size;              // 压缩块大小
    bool parallel_enabled;          // 是否启用并行压缩
    uint32_t thread_count;          // 压缩线程数
    compression_stats_t stats;      // 统计信息
    pthread_mutex_t stats_lock;     // 统计锁
} compression_context_t;

// 压缩块描述符
typedef struct {
    void *input_data;               // 输入数据
    size_t input_size;              // 输入大小
    void *output_data;              // 输出数据
    size_t output_size;             // 输出大小
    size_t max_output_size;         // 最大输出大小
    compression_algorithm_t algorithm; // 压缩算法
    compression_quality_t quality;  // 压缩质量
    uint64_t start_time;            // 开始时间
    uint64_t end_time;              // 结束时间
    int result;                     // 压缩结果
} compression_block_t;

// 全局压缩上下文
static compression_context_t g_compression_ctx = {
    .algorithm = COMPRESS_LZ4,
    .quality = QUALITY_BALANCED,
    .block_size = 64 * 1024,  // 64KB块
    .parallel_enabled = true,
    .thread_count = 4,
    .stats = {0}
};

// 函数声明
static uint64_t get_current_time_us(void);
static int lz4_compress_block(compression_block_t *block);
static int lz4_decompress_block(compression_block_t *block);
static int zstd_compress_block(compression_block_t *block);
static int zstd_decompress_block(compression_block_t *block);
static int snappy_compress_block(compression_block_t *block);
static int snappy_decompress_block(compression_block_t *block);
static compression_algorithm_t select_best_algorithm(const void *data, size_t size);
static void update_compression_stats(compression_block_t *block, bool is_compression);
static void* compression_worker_thread(void *arg);

// 初始化压缩系统
int init_compression_system(compression_algorithm_t algorithm, 
                           compression_quality_t quality,
                           bool parallel_enabled) {
    g_compression_ctx.algorithm = algorithm;
    g_compression_ctx.quality = quality;
    g_compression_ctx.parallel_enabled = parallel_enabled;
    
    if (parallel_enabled) {
        g_compression_ctx.thread_count = 4; // 默认4个线程
    } else {
        g_compression_ctx.thread_count = 1;
    }
    
    pthread_mutex_init(&g_compression_ctx.stats_lock, NULL);
    
    printf("Compression system initialized:\n");
    printf("  Algorithm: %d\n", algorithm);
    printf("  Quality: %d\n", quality);
    printf("  Parallel: %s\n", parallel_enabled ? "enabled" : "disabled");
    printf("  Threads: %u\n", g_compression_ctx.thread_count);
    printf("  Block size: %zu KB\n", g_compression_ctx.block_size / 1024);
    
    return 0;
}

// 内存压缩主函数
int compress_memory(const void *input, size_t input_size, 
                   void **output, size_t *output_size,
                   compression_algorithm_t algorithm) {
    if (!input || input_size == 0 || !output || !output_size) {
        return -1;
    }
    
    // 选择压缩算法
    compression_algorithm_t selected_algo = algorithm;
    if (algorithm == COMPRESS_ADAPTIVE) {
        selected_algo = select_best_algorithm(input, input_size);
    }
    
    // 分配输出缓冲区（预留额外空间）
    size_t max_output_size = input_size + (input_size / 8) + 1024;
    *output = malloc(max_output_size);
    if (!*output) {
        return -1;
    }
    
    // 创建压缩块
    compression_block_t block = {
        .input_data = (void*)input,
        .input_size = input_size,
        .output_data = *output,
        .output_size = 0,
        .max_output_size = max_output_size,
        .algorithm = selected_algo,
        .quality = g_compression_ctx.quality,
        .start_time = get_current_time_us(),
        .result = 0
    };
    
    // 执行压缩
    int result = 0;
    switch (selected_algo) {
    case COMPRESS_LZ4:
        result = lz4_compress_block(&block);
        break;
    case COMPRESS_ZSTD:
        result = zstd_compress_block(&block);
        break;
    case COMPRESS_SNAPPY:
        result = snappy_compress_block(&block);
        break;
    default:
        result = -1;
        break;
    }
    
    block.end_time = get_current_time_us();
    
    if (result == 0) {
        *output_size = block.output_size;
        update_compression_stats(&block, true);
        
        printf("Compressed %zu bytes to %zu bytes (ratio: %.2f%%, algorithm: %d)\n",
               input_size, *output_size, 
               (1.0 - (double)*output_size / input_size) * 100, selected_algo);
    } else {
        free(*output);
        *output = NULL;
        *output_size = 0;
        
        pthread_mutex_lock(&g_compression_ctx.stats_lock);
        g_compression_ctx.stats.compression_failures++;
        pthread_mutex_unlock(&g_compression_ctx.stats_lock);
    }
    
    return result;
}

// 内存解压缩主函数
int decompress_memory(const void *input, size_t input_size,
                     void **output, size_t *output_size,
                     compression_algorithm_t algorithm) {
    if (!input || input_size == 0 || !output || !output_size) {
        return -1;
    }
    
    // 预估解压后大小（通常是压缩前的2-4倍）
    size_t estimated_size = input_size * 4;
    *output = malloc(estimated_size);
    if (!*output) {
        return -1;
    }
    
    // 创建解压块
    compression_block_t block = {
        .input_data = (void*)input,
        .input_size = input_size,
        .output_data = *output,
        .output_size = 0,
        .max_output_size = estimated_size,
        .algorithm = algorithm,
        .quality = g_compression_ctx.quality,
        .start_time = get_current_time_us(),
        .result = 0
    };
    
    // 执行解压缩
    int result = 0;
    switch (algorithm) {
    case COMPRESS_LZ4:
        result = lz4_decompress_block(&block);
        break;
    case COMPRESS_ZSTD:
        result = zstd_decompress_block(&block);
        break;
    case COMPRESS_SNAPPY:
        result = snappy_decompress_block(&block);
        break;
    default:
        result = -1;
        break;
    }
    
    block.end_time = get_current_time_us();
    
    if (result == 0) {
        *output_size = block.output_size;
        update_compression_stats(&block, false);
        
        printf("Decompressed %zu bytes to %zu bytes (algorithm: %d)\n",
               input_size, *output_size, algorithm);
    } else {
        free(*output);
        *output = NULL;
        *output_size = 0;
    }
    
    return result;
}

// LZ4压缩实现（简化版）
static int lz4_compress_block(compression_block_t *block) {
    // 简化的LZ4压缩实现
    // 实际应用中应使用真正的LZ4库
    
    const uint8_t *src = (const uint8_t*)block->input_data;
    uint8_t *dst = (uint8_t*)block->output_data;
    size_t src_size = block->input_size;
    size_t dst_capacity = block->max_output_size;
    
    // 简单的RLE压缩作为示例
    size_t dst_pos = 0;
    size_t src_pos = 0;
    
    while (src_pos < src_size && dst_pos < dst_capacity - 2) {
        uint8_t current = src[src_pos];
        size_t count = 1;
        
        // 计算连续相同字节的数量
        while (src_pos + count < src_size && 
               src[src_pos + count] == current && 
               count < 255) {
            count++;
        }
        
        if (count >= 3) {
            // 压缩连续字节
            dst[dst_pos++] = 0xFF; // 压缩标记
            dst[dst_pos++] = (uint8_t)count;
            dst[dst_pos++] = current;
        } else {
            // 直接复制
            for (size_t i = 0; i < count && dst_pos < dst_capacity; i++) {
                dst[dst_pos++] = src[src_pos + i];
            }
        }
        
        src_pos += count;
    }
    
    // 复制剩余数据
    while (src_pos < src_size && dst_pos < dst_capacity) {
        dst[dst_pos++] = src[src_pos++];
    }
    
    block->output_size = dst_pos;
    return (src_pos == src_size) ? 0 : -1;
}

// LZ4解压缩实现（简化版）
static int lz4_decompress_block(compression_block_t *block) {
    const uint8_t *src = (const uint8_t*)block->input_data;
    uint8_t *dst = (uint8_t*)block->output_data;
    size_t src_size = block->input_size;
    size_t dst_capacity = block->max_output_size;
    
    size_t src_pos = 0;
    size_t dst_pos = 0;
    
    while (src_pos < src_size && dst_pos < dst_capacity) {
        if (src[src_pos] == 0xFF && src_pos + 2 < src_size) {
            // 解压缩标记
            uint8_t count = src[src_pos + 1];
            uint8_t value = src[src_pos + 2];
            
            for (uint8_t i = 0; i < count && dst_pos < dst_capacity; i++) {
                dst[dst_pos++] = value;
            }
            
            src_pos += 3;
        } else {
            // 直接复制
            dst[dst_pos++] = src[src_pos++];
        }
    }
    
    block->output_size = dst_pos;
    return 0;
}

// ZSTD压缩实现（简化版）
static int zstd_compress_block(compression_block_t *block) {
    // 简化实现，实际应使用ZSTD库
    // 这里使用更复杂的压缩策略
    
    const uint8_t *src = (const uint8_t*)block->input_data;
    uint8_t *dst = (uint8_t*)block->output_data;
    size_t src_size = block->input_size;
    
    // 字典压缩示例
    uint8_t dict[256] = {0};
    size_t dict_size = 0;
    size_t dst_pos = 0;
    
    // 构建简单字典
    for (size_t i = 0; i < src_size && dict_size < 256; i++) {
        bool found = false;
        for (size_t j = 0; j < dict_size; j++) {
            if (dict[j] == src[i]) {
                found = true;
                break;
            }
        }
        if (!found) {
            dict[dict_size++] = src[i];
        }
    }
    
    // 写入字典大小
    dst[dst_pos++] = (uint8_t)dict_size;
    
    // 写入字典
    memcpy(dst + dst_pos, dict, dict_size);
    dst_pos += dict_size;
    
    // 压缩数据
    for (size_t i = 0; i < src_size; i++) {
        for (size_t j = 0; j < dict_size; j++) {
            if (dict[j] == src[i]) {
                dst[dst_pos++] = (uint8_t)j;
                break;
            }
        }
    }
    
    block->output_size = dst_pos;
    return 0;
}

// ZSTD解压缩实现（简化版）
static int zstd_decompress_block(compression_block_t *block) {
    const uint8_t *src = (const uint8_t*)block->input_data;
    uint8_t *dst = (uint8_t*)block->output_data;
    size_t src_size = block->input_size;
    
    if (src_size < 1) return -1;
    
    // 读取字典大小
    uint8_t dict_size = src[0];
    if (src_size < 1 + dict_size) return -1;
    
    // 读取字典
    const uint8_t *dict = src + 1;
    
    // 解压缩数据
    size_t src_pos = 1 + dict_size;
    size_t dst_pos = 0;
    
    while (src_pos < src_size) {
        uint8_t dict_index = src[src_pos++];
        if (dict_index < dict_size) {
            dst[dst_pos++] = dict[dict_index];
        }
    }
    
    block->output_size = dst_pos;
    return 0;
}

// Snappy压缩实现（简化版）
static int snappy_compress_block(compression_block_t *block) {
    // 简化的Snappy实现
    // 重点是速度而非压缩比
    
    const uint8_t *src = (const uint8_t*)block->input_data;
    uint8_t *dst = (uint8_t*)block->output_data;
    size_t src_size = block->input_size;
    
    // 简单的字节替换压缩
    size_t dst_pos = 0;
    
    for (size_t i = 0; i < src_size; i++) {
        if (src[i] == 0x00) {
            // 压缩零字节
            dst[dst_pos++] = 0xFE; // 特殊标记
        } else if (src[i] == 0xFE) {
            // 转义特殊标记
            dst[dst_pos++] = 0xFE;
            dst[dst_pos++] = 0xFF;
        } else {
            dst[dst_pos++] = src[i];
        }
    }
    
    block->output_size = dst_pos;
    return 0;
}

// Snappy解压缩实现（简化版）
static int snappy_decompress_block(compression_block_t *block) {
    const uint8_t *src = (const uint8_t*)block->input_data;
    uint8_t *dst = (uint8_t*)block->output_data;
    size_t src_size = block->input_size;
    
    size_t src_pos = 0;
    size_t dst_pos = 0;
    
    while (src_pos < src_size) {
        if (src[src_pos] == 0xFE) {
            if (src_pos + 1 < src_size && src[src_pos + 1] == 0xFF) {
                // 转义的特殊标记
                dst[dst_pos++] = 0xFE;
                src_pos += 2;
            } else {
                // 压缩的零字节
                dst[dst_pos++] = 0x00;
                src_pos++;
            }
        } else {
            dst[dst_pos++] = src[src_pos++];
        }
    }
    
    block->output_size = dst_pos;
    return 0;
}

// 自适应算法选择
static compression_algorithm_t select_best_algorithm(const void *data, size_t size) {
    const uint8_t *bytes = (const uint8_t*)data;
    
    // 分析数据特征
    uint32_t zero_count = 0;
    uint32_t repeat_count = 0;
    uint8_t prev_byte = 0;
    
    for (size_t i = 0; i < size && i < 1024; i++) { // 只分析前1KB
        if (bytes[i] == 0) {
            zero_count++;
        }
        if (i > 0 && bytes[i] == prev_byte) {
            repeat_count++;
        }
        prev_byte = bytes[i];
    }
    
    double zero_ratio = (double)zero_count / (size < 1024 ? size : 1024);
    double repeat_ratio = (double)repeat_count / (size < 1024 ? size : 1024);
    
    // 根据数据特征选择算法
    if (zero_ratio > 0.3) {
        return COMPRESS_SNAPPY; // 大量零字节，使用快速算法
    } else if (repeat_ratio > 0.5) {
        return COMPRESS_LZ4; // 大量重复，使用LZ4
    } else {
        return COMPRESS_ZSTD; // 复杂数据，使用ZSTD
    }
}

// 更新压缩统计信息
static void update_compression_stats(compression_block_t *block, bool is_compression) {
    pthread_mutex_lock(&g_compression_ctx.stats_lock);
    
    uint64_t duration = block->end_time - block->start_time;
    
    if (is_compression) {
        g_compression_ctx.stats.total_compressions++;
        g_compression_ctx.stats.compression_time_us += duration;
        g_compression_ctx.stats.total_input_bytes += block->input_size;
        g_compression_ctx.stats.total_output_bytes += block->output_size;
        
        // 更新平均压缩比
        if (g_compression_ctx.stats.total_input_bytes > 0) {
            g_compression_ctx.stats.avg_compression_ratio = 
                (double)g_compression_ctx.stats.total_output_bytes / 
                g_compression_ctx.stats.total_input_bytes;
        }
    } else {
        g_compression_ctx.stats.total_decompressions++;
        g_compression_ctx.stats.decompression_time_us += duration;
    }
    
    pthread_mutex_unlock(&g_compression_ctx.stats_lock);
}

// 获取当前时间（微秒）
static uint64_t get_current_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

// 打印压缩统计信息
void print_compression_stats(void) {
    pthread_mutex_lock(&g_compression_ctx.stats_lock);
    
    printf("\n=== Compression Statistics ===\n");
    printf("Total Compressions: %llu\n", g_compression_ctx.stats.total_compressions);
    printf("Total Decompressions: %llu\n", g_compression_ctx.stats.total_decompressions);
    printf("Total Input Bytes: %llu MB\n", 
           g_compression_ctx.stats.total_input_bytes / (1024*1024));
    printf("Total Output Bytes: %llu MB\n", 
           g_compression_ctx.stats.total_output_bytes / (1024*1024));
    printf("Average Compression Ratio: %.2f%%\n", 
           (1.0 - g_compression_ctx.stats.avg_compression_ratio) * 100);
    printf("Compression Time: %llu ms\n", 
           g_compression_ctx.stats.compression_time_us / 1000);
    printf("Decompression Time: %llu ms\n", 
           g_compression_ctx.stats.decompression_time_us / 1000);
    printf("Compression Failures: %u\n", g_compression_ctx.stats.compression_failures);
    
    if (g_compression_ctx.stats.total_compressions > 0) {
        printf("Avg Compression Speed: %.2f MB/s\n",
               (double)g_compression_ctx.stats.total_input_bytes / 
               (g_compression_ctx.stats.compression_time_us / 1000000.0) / (1024*1024));
    }
    
    if (g_compression_ctx.stats.total_decompressions > 0) {
        printf("Avg Decompression Speed: %.2f MB/s\n",
               (double)g_compression_ctx.stats.total_output_bytes / 
               (g_compression_ctx.stats.decompression_time_us / 1000000.0) / (1024*1024));
    }
    
    printf("==============================\n\n");
    
    pthread_mutex_unlock(&g_compression_ctx.stats_lock);
}

// 清理压缩系统
void cleanup_compression_system(void) {
    pthread_mutex_destroy(&g_compression_ctx.stats_lock);
    printf("Compression system cleaned up\n");
}