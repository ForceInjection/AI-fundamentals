/**
 * 高级GPU内存虚拟化技术综合演示程序
 * 功能：展示内存碎片整理、NUMA感知、热迁移、故障恢复等高级特性
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <stdbool.h>

// 包含所有内存虚拟化模块的头文件声明
// 注意：实际使用时需要包含对应的头文件

// 内存碎片整理相关声明
typedef struct {
    uint64_t total_memory;
    uint64_t allocated_memory;
    uint64_t free_memory;
    uint32_t total_blocks;
    uint32_t free_blocks;
    uint32_t largest_free_block;
    double fragmentation_ratio;
    uint32_t defrag_operations;
    uint64_t bytes_moved;
    uint64_t defrag_time_ms;
} defrag_stats_t;

extern int init_defragmenter(double threshold);
extern int trigger_defragmentation(void);
extern defrag_stats_t get_defrag_stats(void);
extern void print_defrag_stats(void);
extern void cleanup_defragmenter(void);

// NUMA感知内存管理相关声明
extern int init_numa_memory_manager(void);
extern void* numa_aware_malloc(size_t size, int preferred_node);
extern void numa_aware_free(void *ptr);
extern void* numa_memory_access(void *ptr, size_t offset);
extern void print_numa_stats(void);
extern void cleanup_numa_memory_manager(void);

// 内存热迁移相关声明
typedef enum {
    MIGRATION_STRATEGY_STOP_AND_COPY = 0,
    MIGRATION_STRATEGY_PRE_COPY,
    MIGRATION_STRATEGY_POST_COPY,
    MIGRATION_STRATEGY_HYBRID
} migration_strategy_t;

typedef enum {
    MIGRATION_IDLE = 0,
    MIGRATION_PREPARING,
    MIGRATION_COPYING,
    MIGRATION_SYNCING,
    MIGRATION_FINALIZING,
    MIGRATION_COMPLETED,
    MIGRATION_FAILED,
    MIGRATION_CANCELLED
} migration_state_t;

typedef struct {
    uint64_t total_bytes;
    uint64_t transferred_bytes;
    uint64_t dirty_bytes;
    uint32_t total_pages;
    uint32_t transferred_pages;
    uint32_t dirty_pages;
    uint32_t iterations;
    double transfer_rate_mbps;
    uint64_t downtime_us;
    time_t start_time;
    time_t end_time;
} migration_stats_t;

extern int init_hot_migration_manager(void);
extern uint32_t start_memory_migration(void *source_ptr, size_t size, 
                                      int source_gpu, int target_gpu,
                                      migration_strategy_t strategy,
                                      void (*progress_cb)(uint32_t, double),
                                      void (*completion_cb)(uint32_t, int));
extern migration_state_t get_migration_state(uint32_t session_id);
extern migration_stats_t get_migration_stats(uint32_t session_id);
extern int wait_for_migration(uint32_t session_id, int timeout_sec);
extern void cleanup_hot_migration_manager(void);

// 内存故障恢复相关声明
typedef enum {
    FAULT_TYPE_ECC_SINGLE = 0,
    FAULT_TYPE_ECC_DOUBLE,
    FAULT_TYPE_MEMORY_CORRUPTION,
    FAULT_TYPE_ACCESS_VIOLATION,
    FAULT_TYPE_OUT_OF_MEMORY,
    FAULT_TYPE_DEVICE_LOST,
    FAULT_TYPE_UNKNOWN
} fault_type_t;

typedef struct {
    uint64_t total_faults;
    uint64_t recovered_faults;
    uint64_t critical_faults;
    uint64_t checkpoints_created;
    uint64_t checkpoints_restored;
    uint64_t failovers_performed;
    double recovery_success_rate;
    uint64_t total_downtime_ms;
} fault_recovery_stats_t;

typedef struct {
    uint64_t fault_id;
    fault_type_t type;
    int gpu_id;
    void *memory_addr;
    size_t memory_size;
    time_t occurrence_time;
    char description[256];
    int recovery_attempts;
    bool is_recovered;
} fault_record_t;

extern int init_fault_recovery_manager(void (*fault_cb)(const fault_record_t*),
                                      void (*recovery_cb)(uint64_t, bool));
extern uint64_t create_memory_checkpoint(int gpu_id, void *memory_addr, size_t size);
extern uint64_t report_memory_fault(fault_type_t type, int gpu_id, 
                                   void *addr, size_t size, const char *description);
extern fault_recovery_stats_t get_fault_recovery_stats(void);
extern void print_fault_recovery_stats(void);
extern void cleanup_fault_recovery_manager(void);

// 全局变量
static bool demo_running = true;
static pthread_mutex_t demo_lock = PTHREAD_MUTEX_INITIALIZER;

// 迁移进度回调
void migration_progress_callback(uint32_t session_id, double progress) {
    printf("[Migration %u] Progress: %.1f%%\n", session_id, progress * 100);
}

// 迁移完成回调
void migration_completion_callback(uint32_t session_id, int result) {
    printf("[Migration %u] %s\n", session_id, 
           result == 0 ? "Completed successfully" : "Failed");
}

// 故障检测回调
void fault_detection_callback(const fault_record_t *fault) {
    printf("[Fault Detection] Fault %llu detected: Type=%d, GPU=%d, Addr=%p\n", 
           fault->fault_id, fault->type, fault->gpu_id, fault->memory_addr);
}

// 故障恢复回调
void fault_recovery_callback(uint64_t fault_id, bool success) {
    printf("[Fault Recovery] Fault %llu recovery %s\n", 
           fault_id, success ? "succeeded" : "failed");
}

// 演示内存碎片整理
void demo_memory_defragmentation(void) {
    printf("\n=== Memory Defragmentation Demo ===\n");
    
    // 初始化碎片整理器
    if (init_defragmenter(0.3) != 0) {  // 30%碎片率阈值
        printf("Failed to initialize defragmenter\n");
        return;
    }
    
    printf("Defragmenter initialized with 30%% fragmentation threshold\n");
    
    // 模拟内存分配和释放，产生碎片
    printf("Simulating memory fragmentation...\n");
    sleep(2);
    
    // 获取初始统计信息
    defrag_stats_t initial_stats = get_defrag_stats();
    printf("Initial fragmentation ratio: %.2f%%\n", 
           initial_stats.fragmentation_ratio * 100);
    
    // 手动触发碎片整理
    printf("Triggering manual defragmentation...\n");
    if (trigger_defragmentation() == 0) {
        printf("Defragmentation completed\n");
    } else {
        printf("Defragmentation failed\n");
    }
    
    // 获取最终统计信息
    defrag_stats_t final_stats = get_defrag_stats();
    printf("Final fragmentation ratio: %.2f%%\n", 
           final_stats.fragmentation_ratio * 100);
    
    // 打印详细统计
    print_defrag_stats();
    
    // 清理
    cleanup_defragmenter();
    printf("Defragmentation demo completed\n");
}

// 演示NUMA感知内存管理
void demo_numa_aware_memory(void) {
    printf("\n=== NUMA-Aware Memory Demo ===\n");
    
    // 初始化NUMA感知内存管理器
    if (init_numa_memory_manager() != 0) {
        printf("Failed to initialize NUMA memory manager\n");
        return;
    }
    
    printf("NUMA-aware memory manager initialized\n");
    
    // 在不同NUMA节点分配内存
    void *ptr1 = numa_aware_malloc(1024 * 1024, 0);  // 首选节点0
    void *ptr2 = numa_aware_malloc(2 * 1024 * 1024, 1);  // 首选节点1
    void *ptr3 = numa_aware_malloc(512 * 1024, -1);  // 自动选择
    
    if (ptr1 && ptr2 && ptr3) {
        printf("Successfully allocated memory on different NUMA nodes\n");
        
        // 模拟内存访问
        printf("Simulating memory access patterns...\n");
        for (int i = 0; i < 100; i++) {
            numa_memory_access(ptr1, i * 1024);
            numa_memory_access(ptr2, i * 2048);
            numa_memory_access(ptr3, i * 512);
            usleep(1000);  // 1ms延迟
        }
        
        printf("Memory access simulation completed\n");
        
        // 释放内存
        numa_aware_free(ptr1);
        numa_aware_free(ptr2);
        numa_aware_free(ptr3);
        
        printf("Memory freed\n");
    } else {
        printf("Failed to allocate NUMA-aware memory\n");
    }
    
    // 打印NUMA统计信息
    print_numa_stats();
    
    // 清理
    cleanup_numa_memory_manager();
    printf("NUMA-aware memory demo completed\n");
}

// 演示内存热迁移
void demo_memory_hot_migration(void) {
    printf("\n=== Memory Hot Migration Demo ===\n");
    
    // 初始化热迁移管理器
    if (init_hot_migration_manager() != 0) {
        printf("Failed to initialize hot migration manager\n");
        return;
    }
    
    printf("Hot migration manager initialized\n");
    
    // 模拟GPU内存分配
    void *source_memory = malloc(4 * 1024 * 1024);  // 4MB模拟GPU内存
    if (!source_memory) {
        printf("Failed to allocate source memory\n");
        cleanup_hot_migration_manager();
        return;
    }
    
    // 填充测试数据
    memset(source_memory, 0xAA, 4 * 1024 * 1024);
    printf("Source memory allocated and initialized\n");
    
    // 开始内存迁移（模拟从GPU 0到GPU 1）
    printf("Starting memory migration from GPU 0 to GPU 1...\n");
    uint32_t session_id = start_memory_migration(
        source_memory, 4 * 1024 * 1024,
        0, 1,  // 从GPU 0到GPU 1
        MIGRATION_STRATEGY_PRE_COPY,
        migration_progress_callback,
        migration_completion_callback
    );
    
    if (session_id == 0) {
        printf("Failed to start memory migration\n");
        free(source_memory);
        cleanup_hot_migration_manager();
        return;
    }
    
    printf("Migration session %u started\n", session_id);
    
    // 监控迁移状态
    migration_state_t state;
    do {
        state = get_migration_state(session_id);
        migration_stats_t stats = get_migration_stats(session_id);
        
        printf("Migration state: %d, Progress: %.1f%%\n", 
               state, (double)stats.transferred_bytes / stats.total_bytes * 100);
        
        sleep(1);
    } while (state != MIGRATION_COMPLETED && state != MIGRATION_FAILED && 
             state != MIGRATION_CANCELLED);
    
    // 等待迁移完成
    if (wait_for_migration(session_id, 30) == 0) {
        printf("Migration completed successfully\n");
        
        // 获取最终统计信息
        migration_stats_t final_stats = get_migration_stats(session_id);
        printf("Migration statistics:\n");
        printf("  Total bytes: %llu\n", final_stats.total_bytes);
        printf("  Transfer rate: %.2f MB/s\n", final_stats.transfer_rate_mbps);
        printf("  Downtime: %llu us\n", final_stats.downtime_us);
        printf("  Iterations: %u\n", final_stats.iterations);
    } else {
        printf("Migration failed or timed out\n");
    }
    
    // 清理
    free(source_memory);
    cleanup_hot_migration_manager();
    printf("Memory hot migration demo completed\n");
}

// 演示内存故障恢复
void demo_memory_fault_recovery(void) {
    printf("\n=== Memory Fault Recovery Demo ===\n");
    
    // 初始化故障恢复管理器
    if (init_fault_recovery_manager(fault_detection_callback, 
                                   fault_recovery_callback) != 0) {
        printf("Failed to initialize fault recovery manager\n");
        return;
    }
    
    printf("Fault recovery manager initialized\n");
    
    // 模拟GPU内存分配
    void *gpu_memory = malloc(2 * 1024 * 1024);  // 2MB模拟GPU内存
    if (!gpu_memory) {
        printf("Failed to allocate GPU memory\n");
        cleanup_fault_recovery_manager();
        return;
    }
    
    // 填充测试数据
    memset(gpu_memory, 0x55, 2 * 1024 * 1024);
    printf("GPU memory allocated and initialized\n");
    
    // 创建内存检查点
    printf("Creating memory checkpoint...\n");
    uint64_t checkpoint_id = create_memory_checkpoint(0, gpu_memory, 2 * 1024 * 1024);
    if (checkpoint_id > 0) {
        printf("Memory checkpoint %llu created\n", checkpoint_id);
    } else {
        printf("Failed to create memory checkpoint\n");
    }
    
    // 模拟不同类型的内存故障
    printf("\nSimulating memory faults...\n");
    
    // 1. 单比特ECC错误
    uint64_t fault1 = report_memory_fault(
        FAULT_TYPE_ECC_SINGLE, 0, gpu_memory, 4096,
        "Single-bit ECC error in page 0"
    );
    printf("Reported single-bit ECC error: fault %llu\n", fault1);
    sleep(1);
    
    // 2. 内存损坏
    uint64_t fault2 = report_memory_fault(
        FAULT_TYPE_MEMORY_CORRUPTION, 0, 
        (char*)gpu_memory + 1024*1024, 4096,
        "Memory corruption detected"
    );
    printf("Reported memory corruption: fault %llu\n", fault2);
    sleep(1);
    
    // 3. 双比特ECC错误（关键故障）
    uint64_t fault3 = report_memory_fault(
        FAULT_TYPE_ECC_DOUBLE, 0, 
        (char*)gpu_memory + 512*1024, 4096,
        "Double-bit ECC error - critical"
    );
    printf("Reported double-bit ECC error: fault %llu\n", fault3);
    sleep(2);
    
    // 等待故障恢复完成
    printf("\nWaiting for fault recovery to complete...\n");
    sleep(3);
    
    // 打印故障恢复统计信息
    print_fault_recovery_stats();
    
    // 清理
    free(gpu_memory);
    cleanup_fault_recovery_manager();
    printf("Memory fault recovery demo completed\n");
}

// 综合性能测试
void demo_comprehensive_performance(void) {
    printf("\n=== Comprehensive Performance Test ===\n");
    
    clock_t start_time = clock();
    
    // 初始化所有模块
    printf("Initializing all memory virtualization modules...\n");
    
    if (init_defragmenter(0.25) != 0 ||
        init_numa_memory_manager() != 0 ||
        init_hot_migration_manager() != 0 ||
        init_fault_recovery_manager(fault_detection_callback, 
                                   fault_recovery_callback) != 0) {
        printf("Failed to initialize some modules\n");
        return;
    }
    
    printf("All modules initialized successfully\n");
    
    // 执行综合测试
    printf("\nExecuting comprehensive memory operations...\n");
    
    // 分配多个内存块
    void *ptrs[10];
    for (int i = 0; i < 10; i++) {
        ptrs[i] = numa_aware_malloc((i + 1) * 1024 * 1024, i % 2);
        if (ptrs[i]) {
            // 创建检查点
            create_memory_checkpoint(0, ptrs[i], (i + 1) * 1024 * 1024);
            
            // 模拟内存访问
            numa_memory_access(ptrs[i], 0);
        }
    }
    
    printf("Memory allocation and checkpoint creation completed\n");
    
    // 触发碎片整理
    printf("Triggering defragmentation...\n");
    trigger_defragmentation();
    
    // 模拟一些故障
    printf("Simulating faults...\n");
    for (int i = 0; i < 3; i++) {
        if (ptrs[i]) {
            report_memory_fault(FAULT_TYPE_ECC_SINGLE, 0, ptrs[i], 4096,
                              "Simulated ECC error");
        }
    }
    
    // 等待处理完成
    sleep(2);
    
    // 清理内存
    printf("Cleaning up memory...\n");
    for (int i = 0; i < 10; i++) {
        if (ptrs[i]) {
            numa_aware_free(ptrs[i]);
        }
    }
    
    // 打印所有统计信息
    printf("\n=== Final Statistics ===\n");
    print_defrag_stats();
    print_numa_stats();
    print_fault_recovery_stats();
    
    // 清理所有模块
    cleanup_defragmenter();
    cleanup_numa_memory_manager();
    cleanup_hot_migration_manager();
    cleanup_fault_recovery_manager();
    
    clock_t end_time = clock();
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("\nComprehensive performance test completed in %.2f seconds\n", total_time);
}

// 主函数
int main(int argc, char *argv[]) {
    printf("=== Advanced GPU Memory Virtualization Demo ===\n");
    printf("This demo showcases advanced memory virtualization features:\n");
    printf("- Memory Defragmentation\n");
    printf("- NUMA-Aware Memory Management\n");
    printf("- Memory Hot Migration\n");
    printf("- Memory Fault Recovery\n\n");
    
    // 检查命令行参数
    if (argc > 1) {
        if (strcmp(argv[1], "defrag") == 0) {
            demo_memory_defragmentation();
            return 0;
        } else if (strcmp(argv[1], "numa") == 0) {
            demo_numa_aware_memory();
            return 0;
        } else if (strcmp(argv[1], "migration") == 0) {
            demo_memory_hot_migration();
            return 0;
        } else if (strcmp(argv[1], "fault") == 0) {
            demo_memory_fault_recovery();
            return 0;
        } else if (strcmp(argv[1], "performance") == 0) {
            demo_comprehensive_performance();
            return 0;
        } else {
            printf("Usage: %s [defrag|numa|migration|fault|performance]\n", argv[0]);
            return 1;
        }
    }
    
    // 运行所有演示
    printf("Running all demonstrations...\n\n");
    
    demo_memory_defragmentation();
    sleep(1);
    
    demo_numa_aware_memory();
    sleep(1);
    
    demo_memory_hot_migration();
    sleep(1);
    
    demo_memory_fault_recovery();
    sleep(1);
    
    demo_comprehensive_performance();
    
    printf("\n=== All Demonstrations Completed ===\n");
    printf("Advanced GPU memory virtualization features have been successfully demonstrated.\n");
    printf("These technologies provide:\n");
    printf("- Improved memory utilization through defragmentation\n");
    printf("- Optimized performance on NUMA architectures\n");
    printf("- Zero-downtime memory migration capabilities\n");
    printf("- Robust fault detection and recovery mechanisms\n\n");
    
    return 0;
}