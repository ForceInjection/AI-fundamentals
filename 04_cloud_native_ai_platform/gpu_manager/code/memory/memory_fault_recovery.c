/**
 * GPU内存故障恢复实现
 * 功能：检测和恢复GPU内存故障，保障系统可靠性
 * 特性：ECC错误处理、内存检查点、自动故障转移
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <stdbool.h>
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif
#include <sys/time.h>

#define MAX_FAULT_RECORDS 1000
#define MAX_CHECKPOINTS 100
#define MEMORY_CHECK_INTERVAL 5  // 5秒
#define MAX_RECOVERY_ATTEMPTS 3
#define FAULT_THRESHOLD 10  // 故障阈值

// 故障类型
typedef enum {
    FAULT_TYPE_ECC_SINGLE = 0,      // 单比特ECC错误
    FAULT_TYPE_ECC_DOUBLE,          // 双比特ECC错误
    FAULT_TYPE_MEMORY_CORRUPTION,   // 内存损坏
    FAULT_TYPE_ACCESS_VIOLATION,    // 访问违规
    FAULT_TYPE_OUT_OF_MEMORY,       // 内存不足
    FAULT_TYPE_DEVICE_LOST,         // 设备丢失
    FAULT_TYPE_UNKNOWN              // 未知错误
} fault_type_t;

// 故障严重程度
typedef enum {
    FAULT_SEVERITY_LOW = 0,     // 低严重程度
    FAULT_SEVERITY_MEDIUM,      // 中等严重程度
    FAULT_SEVERITY_HIGH,        // 高严重程度
    FAULT_SEVERITY_CRITICAL     // 关键严重程度
} fault_severity_t;

// 恢复策略
typedef enum {
    RECOVERY_STRATEGY_RETRY = 0,    // 重试
    RECOVERY_STRATEGY_RELOCATE,     // 重新分配
    RECOVERY_STRATEGY_CHECKPOINT,   // 检查点恢复
    RECOVERY_STRATEGY_FAILOVER,     // 故障转移
    RECOVERY_STRATEGY_RESTART       // 重启
} recovery_strategy_t;

// 故障记录
typedef struct {
    uint64_t fault_id;              // 故障ID
    fault_type_t type;              // 故障类型
    fault_severity_t severity;      // 严重程度
    int gpu_id;                     // GPU ID
    void *memory_addr;              // 内存地址
    size_t memory_size;             // 内存大小
    time_t occurrence_time;         // 发生时间
    char description[256];          // 故障描述
    recovery_strategy_t strategy;   // 恢复策略
    int recovery_attempts;          // 恢复尝试次数
    bool is_recovered;              // 是否已恢复
} fault_record_t;

// 内存检查点
typedef struct {
    uint64_t checkpoint_id;         // 检查点ID
    int gpu_id;                     // GPU ID
    void *memory_addr;              // 内存地址
    size_t memory_size;             // 内存大小
    void *backup_data;              // 备份数据
    uint32_t checksum;              // 校验和
    time_t creation_time;           // 创建时间
    bool is_valid;                  // 是否有效
} memory_checkpoint_t;

// 故障恢复统计
typedef struct {
    uint64_t total_faults;          // 总故障数
    uint64_t recovered_faults;      // 已恢复故障数
    uint64_t critical_faults;       // 关键故障数
    uint64_t checkpoints_created;   // 创建的检查点数
    uint64_t checkpoints_restored;  // 恢复的检查点数
    uint64_t failovers_performed;   // 执行的故障转移数
    double recovery_success_rate;   // 恢复成功率
    uint64_t total_downtime_ms;     // 总停机时间
} fault_recovery_stats_t;

// 故障恢复管理器
typedef struct {
    fault_record_t fault_records[MAX_FAULT_RECORDS];
    memory_checkpoint_t checkpoints[MAX_CHECKPOINTS];
    uint32_t fault_count;           // 故障数量
    uint32_t checkpoint_count;      // 检查点数量
    uint64_t next_fault_id;         // 下一个故障ID
    uint64_t next_checkpoint_id;    // 下一个检查点ID
    
    pthread_mutex_t recovery_lock;  // 恢复锁
    pthread_t monitor_thread;       // 监控线程
    bool monitoring_enabled;        // 是否启用监控
    bool is_initialized;            // 是否已初始化
    
    fault_recovery_stats_t stats;   // 统计信息
    
    // 回调函数
    void (*fault_callback)(const fault_record_t *fault);
    void (*recovery_callback)(uint64_t fault_id, bool success);
} fault_recovery_manager_t;

// 全局故障恢复管理器
static fault_recovery_manager_t g_recovery_manager = {0};

// 计算校验和
static uint32_t calculate_checksum(const void *data, size_t size) {
    uint32_t checksum = 0;
    const uint8_t *bytes = (const uint8_t*)data;
    
    for (size_t i = 0; i < size; i++) {
        checksum = (checksum << 1) ^ bytes[i];
    }
    
    return checksum;
}

// 检测ECC错误
static int detect_ecc_errors(int gpu_id) {
    // 简化实现：实际需要使用NVML API获取详细ECC信息
    // 这里模拟ECC错误检测
    static unsigned long long prev_single = 0, prev_double = 0;
    unsigned long long single_bit_errors = 0, double_bit_errors = 0;
    
#ifdef CUDA_ENABLED
    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        return -1;
    }
    
    // 获取ECC错误计数
    err = cudaDeviceGetAttribute((int*)&single_bit_errors, 
                                cudaDevAttrEccEnabled, gpu_id);
    if (err != cudaSuccess) {
        return 0;  // ECC不支持
    }
#else
    // CUDA未启用时的模拟实现
    (void)gpu_id;  // 避免未使用参数警告
    // 使用默认值0
#endif
    
    if (single_bit_errors > prev_single) {
        prev_single = single_bit_errors;
        return 1;  // 检测到单比特错误
    }
    
    if (double_bit_errors > prev_double) {
        prev_double = double_bit_errors;
        return 2;  // 检测到双比特错误
    }
    
    return 0;  // 无错误
}

// 记录故障
static uint64_t record_fault(fault_type_t type, fault_severity_t severity,
                            int gpu_id, void *addr, size_t size,
                            const char *description) {
    pthread_mutex_lock(&g_recovery_manager.recovery_lock);
    
    if (g_recovery_manager.fault_count >= MAX_FAULT_RECORDS) {
        // 移除最旧的故障记录
        memmove(&g_recovery_manager.fault_records[0],
                &g_recovery_manager.fault_records[1],
                (MAX_FAULT_RECORDS - 1) * sizeof(fault_record_t));
        g_recovery_manager.fault_count--;
    }
    
    fault_record_t *fault = &g_recovery_manager.fault_records[g_recovery_manager.fault_count];
    fault->fault_id = g_recovery_manager.next_fault_id++;
    fault->type = type;
    fault->severity = severity;
    fault->gpu_id = gpu_id;
    fault->memory_addr = addr;
    fault->memory_size = size;
    fault->occurrence_time = time(NULL);
    strncpy(fault->description, description, sizeof(fault->description) - 1);
    fault->recovery_attempts = 0;
    fault->is_recovered = false;
    
    // 选择恢复策略
    switch (severity) {
        case FAULT_SEVERITY_LOW:
            fault->strategy = RECOVERY_STRATEGY_RETRY;
            break;
        case FAULT_SEVERITY_MEDIUM:
            fault->strategy = RECOVERY_STRATEGY_RELOCATE;
            break;
        case FAULT_SEVERITY_HIGH:
            fault->strategy = RECOVERY_STRATEGY_CHECKPOINT;
            break;
        case FAULT_SEVERITY_CRITICAL:
            fault->strategy = RECOVERY_STRATEGY_FAILOVER;
            break;
    }
    
    g_recovery_manager.fault_count++;
    g_recovery_manager.stats.total_faults++;
    
    if (severity == FAULT_SEVERITY_CRITICAL) {
        g_recovery_manager.stats.critical_faults++;
    }
    
    uint64_t fault_id = fault->fault_id;
    
    pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
    
    printf("Fault recorded: ID=%llu, Type=%d, Severity=%d, GPU=%d\n", 
           fault_id, type, severity, gpu_id);
    
    // 调用故障回调
    if (g_recovery_manager.fault_callback) {
        g_recovery_manager.fault_callback(fault);
    }
    
    return fault_id;
}

// 创建内存检查点
uint64_t create_memory_checkpoint(int gpu_id, void *memory_addr, size_t size) {
    if (!memory_addr || size == 0) {
        return 0;
    }
    
    pthread_mutex_lock(&g_recovery_manager.recovery_lock);
    
    if (g_recovery_manager.checkpoint_count >= MAX_CHECKPOINTS) {
        // 移除最旧的检查点
        memory_checkpoint_t *oldest = &g_recovery_manager.checkpoints[0];
        if (oldest->backup_data) {
            free(oldest->backup_data);
        }
        
        memmove(&g_recovery_manager.checkpoints[0],
                &g_recovery_manager.checkpoints[1],
                (MAX_CHECKPOINTS - 1) * sizeof(memory_checkpoint_t));
        g_recovery_manager.checkpoint_count--;
    }
    
    memory_checkpoint_t *checkpoint = 
        &g_recovery_manager.checkpoints[g_recovery_manager.checkpoint_count];
    
    checkpoint->checkpoint_id = g_recovery_manager.next_checkpoint_id++;
    checkpoint->gpu_id = gpu_id;
    checkpoint->memory_addr = memory_addr;
    checkpoint->memory_size = size;
    checkpoint->creation_time = time(NULL);
    
    // 分配备份内存
    checkpoint->backup_data = malloc(size);
    if (!checkpoint->backup_data) {
        pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
        return 0;
    }
    
    // 复制GPU内存到主机内存
#ifdef CUDA_ENABLED
    cudaSetDevice(gpu_id);
    cudaError_t err = cudaMemcpy(checkpoint->backup_data, memory_addr, 
                                size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(checkpoint->backup_data);
        pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
        return 0;
    }
#else
    // CUDA未启用时的模拟实现
    (void)gpu_id;  // 避免未使用参数警告
    memcpy(checkpoint->backup_data, memory_addr, size);
#endif
    
    // 计算校验和
    checkpoint->checksum = calculate_checksum(checkpoint->backup_data, size);
    checkpoint->is_valid = true;
    
    g_recovery_manager.checkpoint_count++;
    g_recovery_manager.stats.checkpoints_created++;
    
    uint64_t checkpoint_id = checkpoint->checkpoint_id;
    
    pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
    
    printf("Memory checkpoint created: ID=%llu, GPU=%d, Size=%zu\n", 
           checkpoint_id, gpu_id, size);
    
    return checkpoint_id;
}

// 恢复内存检查点
int restore_memory_checkpoint(uint64_t checkpoint_id) {
    pthread_mutex_lock(&g_recovery_manager.recovery_lock);
    
    memory_checkpoint_t *checkpoint = NULL;
    for (uint32_t i = 0; i < g_recovery_manager.checkpoint_count; i++) {
        if (g_recovery_manager.checkpoints[i].checkpoint_id == checkpoint_id) {
            checkpoint = &g_recovery_manager.checkpoints[i];
            break;
        }
    }
    
    if (!checkpoint || !checkpoint->is_valid) {
        pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
        return -1;
    }
    
    // 验证校验和
    uint32_t current_checksum = calculate_checksum(checkpoint->backup_data, 
                                                  checkpoint->memory_size);
    if (current_checksum != checkpoint->checksum) {
        printf("Checkpoint %llu corrupted, checksum mismatch\n", checkpoint_id);
        checkpoint->is_valid = false;
        pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
        return -1;
    }
    
    // 恢复内存
#ifdef CUDA_ENABLED
    cudaSetDevice(checkpoint->gpu_id);
    cudaError_t err = cudaMemcpy(checkpoint->memory_addr, checkpoint->backup_data,
                                checkpoint->memory_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
        return -1;
    }
#else
    // CUDA未启用时的模拟实现
    memcpy(checkpoint->memory_addr, checkpoint->backup_data, checkpoint->memory_size);
#endif
    
    g_recovery_manager.stats.checkpoints_restored++;
    
    pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
    
    printf("Memory checkpoint restored: ID=%llu\n", checkpoint_id);
    return 0;
}

// 执行故障恢复
static int perform_fault_recovery(fault_record_t *fault) {
    int result = -1;
    uint64_t start_time = time(NULL);
    
    printf("Attempting recovery for fault %llu using strategy %d\n", 
           fault->fault_id, fault->strategy);
    
    switch (fault->strategy) {
        case RECOVERY_STRATEGY_RETRY:
            // 简单重试
            usleep(100000);  // 100ms延迟
            result = 0;
            break;
            
        case RECOVERY_STRATEGY_RELOCATE:
            // 重新分配内存
            if (fault->memory_addr && fault->memory_size > 0) {
                void *new_ptr;
#ifdef CUDA_ENABLED
                cudaSetDevice(fault->gpu_id);
                cudaError_t err = cudaMalloc(&new_ptr, fault->memory_size);
                if (err == cudaSuccess) {
                    // 更新内存地址（实际应用中需要更新所有引用）
                    fault->memory_addr = new_ptr;
                    result = 0;
                }
#else
                // CUDA未启用时的模拟实现
                new_ptr = malloc(fault->memory_size);
                if (new_ptr) {
                    fault->memory_addr = new_ptr;
                    result = 0;
                }
#endif
            }
            break;
            
        case RECOVERY_STRATEGY_CHECKPOINT:
            // 查找相关检查点并恢复
            for (uint32_t i = 0; i < g_recovery_manager.checkpoint_count; i++) {
                memory_checkpoint_t *cp = &g_recovery_manager.checkpoints[i];
                if (cp->gpu_id == fault->gpu_id && 
                    cp->memory_addr == fault->memory_addr &&
                    cp->is_valid) {
                    result = restore_memory_checkpoint(cp->checkpoint_id);
                    break;
                }
            }
            break;
            
        case RECOVERY_STRATEGY_FAILOVER:
            // 故障转移到其他GPU
            printf("Performing failover for GPU %d\n", fault->gpu_id);
            g_recovery_manager.stats.failovers_performed++;
            result = 0;  // 简化实现
            break;
            
        case RECOVERY_STRATEGY_RESTART:
            // 重启GPU上下文
#ifdef CUDA_ENABLED
            cudaSetDevice(fault->gpu_id);
            cudaDeviceReset();
#else
            // CUDA未启用时的模拟实现
            printf("Simulating GPU %d context restart\n", fault->gpu_id);
#endif
            result = 0;
            break;
    }
    
    uint64_t end_time = time(NULL);
    g_recovery_manager.stats.total_downtime_ms += (end_time - start_time) * 1000;
    
    return result;
}

// 处理故障
int handle_fault(uint64_t fault_id) {
    pthread_mutex_lock(&g_recovery_manager.recovery_lock);
    
    fault_record_t *fault = NULL;
    for (uint32_t i = 0; i < g_recovery_manager.fault_count; i++) {
        if (g_recovery_manager.fault_records[i].fault_id == fault_id) {
            fault = &g_recovery_manager.fault_records[i];
            break;
        }
    }
    
    if (!fault || fault->is_recovered) {
        pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
        return -1;
    }
    
    fault->recovery_attempts++;
    
    pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
    
    // 执行恢复
    int result = perform_fault_recovery(fault);
    
    pthread_mutex_lock(&g_recovery_manager.recovery_lock);
    
    if (result == 0) {
        fault->is_recovered = true;
        g_recovery_manager.stats.recovered_faults++;
        printf("Fault %llu recovered successfully\n", fault_id);
    } else if (fault->recovery_attempts >= MAX_RECOVERY_ATTEMPTS) {
        printf("Fault %llu recovery failed after %d attempts\n", 
               fault_id, fault->recovery_attempts);
    }
    
    // 更新成功率
    if (g_recovery_manager.stats.total_faults > 0) {
        g_recovery_manager.stats.recovery_success_rate = 
            (double)g_recovery_manager.stats.recovered_faults / 
            g_recovery_manager.stats.total_faults;
    }
    
    pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
    
    // 调用恢复回调
    if (g_recovery_manager.recovery_callback) {
        g_recovery_manager.recovery_callback(fault_id, result == 0);
    }
    
    return result;
}

// 内存监控线程
static void* memory_monitor_thread(void *arg) {
    while (g_recovery_manager.monitoring_enabled) {
        // 检查所有GPU的ECC错误
        int gpu_count;
#ifdef CUDA_ENABLED
        cudaGetDeviceCount(&gpu_count);
#else
        // CUDA未启用时的模拟实现
        gpu_count = 1;  // 模拟1个GPU
#endif
        
        for (int i = 0; i < gpu_count; i++) {
            int ecc_result = detect_ecc_errors(i);
            
            if (ecc_result == 1) {
                // 单比特ECC错误
                uint64_t fault_id = record_fault(FAULT_TYPE_ECC_SINGLE, 
                                                FAULT_SEVERITY_LOW, i, NULL, 0,
                                                "Single-bit ECC error detected");
                handle_fault(fault_id);
            } else if (ecc_result == 2) {
                // 双比特ECC错误
                uint64_t fault_id = record_fault(FAULT_TYPE_ECC_DOUBLE, 
                                                FAULT_SEVERITY_HIGH, i, NULL, 0,
                                                "Double-bit ECC error detected");
                handle_fault(fault_id);
            }
        }
        
        // 检查内存完整性
        // 这里可以添加更多的内存检查逻辑
        
        sleep(MEMORY_CHECK_INTERVAL);
    }
    
    return NULL;
}

// 初始化故障恢复管理器
int init_fault_recovery_manager(void (*fault_cb)(const fault_record_t*),
                               void (*recovery_cb)(uint64_t, bool)) {
    if (g_recovery_manager.is_initialized) {
        return 0;
    }
    
    memset(&g_recovery_manager, 0, sizeof(fault_recovery_manager_t));
    
    if (pthread_mutex_init(&g_recovery_manager.recovery_lock, NULL) != 0) {
        return -1;
    }
    
    g_recovery_manager.next_fault_id = 1;
    g_recovery_manager.next_checkpoint_id = 1;
    g_recovery_manager.fault_callback = fault_cb;
    g_recovery_manager.recovery_callback = recovery_cb;
    g_recovery_manager.monitoring_enabled = true;
    g_recovery_manager.is_initialized = true;
    
    // 启动监控线程
    if (pthread_create(&g_recovery_manager.monitor_thread, NULL,
                      memory_monitor_thread, NULL) != 0) {
        pthread_mutex_destroy(&g_recovery_manager.recovery_lock);
        g_recovery_manager.is_initialized = false;
        return -1;
    }
    
    printf("Fault recovery manager initialized\n");
    return 0;
}

// 报告内存故障
uint64_t report_memory_fault(fault_type_t type, int gpu_id, 
                            void *addr, size_t size, const char *description) {
    fault_severity_t severity;
    
    // 根据故障类型确定严重程度
    switch (type) {
        case FAULT_TYPE_ECC_SINGLE:
            severity = FAULT_SEVERITY_LOW;
            break;
        case FAULT_TYPE_ECC_DOUBLE:
        case FAULT_TYPE_MEMORY_CORRUPTION:
            severity = FAULT_SEVERITY_HIGH;
            break;
        case FAULT_TYPE_DEVICE_LOST:
            severity = FAULT_SEVERITY_CRITICAL;
            break;
        default:
            severity = FAULT_SEVERITY_MEDIUM;
            break;
    }
    
    uint64_t fault_id = record_fault(type, severity, gpu_id, addr, size, description);
    
    // 自动处理故障
    handle_fault(fault_id);
    
    return fault_id;
}

// 获取故障恢复统计信息
fault_recovery_stats_t get_fault_recovery_stats(void) {
    pthread_mutex_lock(&g_recovery_manager.recovery_lock);
    fault_recovery_stats_t stats = g_recovery_manager.stats;
    pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
    
    return stats;
}

// 打印故障恢复统计信息
void print_fault_recovery_stats(void) {
    fault_recovery_stats_t stats = get_fault_recovery_stats();
    double recovery_rate = stats.total_faults > 0 ? 
        (double)stats.recovered_faults / stats.total_faults * 100.0 : 0.0;
    
    printf("\n=== Fault Recovery Statistics ===\n");
    printf("Total Faults: %llu\n", stats.total_faults);
    printf("Recovered Faults: %llu\n", stats.recovered_faults);
    printf("Critical Faults: %llu\n", stats.critical_faults);
    printf("Recovery Rate: %.2f%%\n", recovery_rate);
    printf("Checkpoints Created: %llu\n", stats.checkpoints_created);
    printf("Checkpoints Restored: %llu\n", stats.checkpoints_restored);
    printf("Failovers Performed: %llu\n", stats.failovers_performed);
    printf("Total Downtime: %llu ms\n", stats.total_downtime_ms);
    printf("=================================\n\n");
}

// 启用/禁用内存监控
void enable_memory_monitoring(bool enable) {
    g_recovery_manager.monitoring_enabled = enable;
}

// 清理故障恢复管理器
void cleanup_fault_recovery_manager(void) {
    if (!g_recovery_manager.is_initialized) {
        return;
    }
    
    g_recovery_manager.monitoring_enabled = false;
    
    // 等待监控线程结束
    pthread_join(g_recovery_manager.monitor_thread, NULL);
    
    pthread_mutex_lock(&g_recovery_manager.recovery_lock);
    
    // 清理检查点数据
    for (uint32_t i = 0; i < g_recovery_manager.checkpoint_count; i++) {
        if (g_recovery_manager.checkpoints[i].backup_data) {
            free(g_recovery_manager.checkpoints[i].backup_data);
        }
    }
    
    pthread_mutex_unlock(&g_recovery_manager.recovery_lock);
    pthread_mutex_destroy(&g_recovery_manager.recovery_lock);
    
    g_recovery_manager.is_initialized = false;
    
    printf("Fault recovery manager cleaned up\n");
}