/**
 * GPU内存虚拟化技术演示程序
 * 功能：演示内存压缩、交换、统一地址空间和QoS等功能
 * 用途：测试和验证内存虚拟化技术的完整性
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>

// 包含其他模块的头文件声明
// 在实际项目中，这些应该在独立的头文件中定义

// 内存压缩相关
typedef enum {
    COMPRESS_NONE = 0,
    COMPRESS_LZ4,
    COMPRESS_ZSTD,
    COMPRESS_SNAPPY,
    COMPRESS_ADAPTIVE
} compression_algorithm_t;

typedef enum {
    QUALITY_FAST = 1,
    QUALITY_BALANCED = 5,
    QUALITY_BEST = 9
} compression_quality_t;

// 内存交换相关
typedef enum {
    SWAP_SYSTEM_MEMORY = 0,
    SWAP_SSD_STORAGE,
    SWAP_HDD_STORAGE,
    SWAP_NVME_STORAGE,
    SWAP_REMOTE_STORAGE
} swap_storage_type_t;

// 统一地址空间相关
typedef enum {
    MEMORY_TYPE_HOST = 0,
    MEMORY_TYPE_DEVICE,
    MEMORY_TYPE_MANAGED,
    MEMORY_TYPE_PINNED
} memory_type_t;

typedef enum {
    ACCESS_NONE = 0,
    ACCESS_READ = 1,
    ACCESS_WRITE = 2,
    ACCESS_EXECUTE = 4,
    ACCESS_RW = ACCESS_READ | ACCESS_WRITE,
    ACCESS_ALL = ACCESS_READ | ACCESS_WRITE | ACCESS_EXECUTE
} access_permission_t;

// QoS相关
typedef enum {
    QOS_LEVEL_CRITICAL = 0,
    QOS_LEVEL_HIGH,
    QOS_LEVEL_NORMAL,
    QOS_LEVEL_LOW,
    QOS_LEVEL_BACKGROUND
} qos_level_t;

typedef enum {
    ACCESS_TYPE_READ = 0,
    ACCESS_TYPE_WRITE,
    ACCESS_TYPE_ATOMIC,
    ACCESS_TYPE_PREFETCH
} memory_access_type_t;

typedef enum {
    BANDWIDTH_POLICY_FAIR = 0,
    BANDWIDTH_POLICY_PRIORITY,
    BANDWIDTH_POLICY_WEIGHTED,
    BANDWIDTH_POLICY_ADAPTIVE
} bandwidth_policy_t;

// 外部函数声明（来自其他模块）
extern int init_compression_system(compression_algorithm_t algorithm, 
                                  compression_quality_t quality, bool parallel_enabled);
extern int compress_memory(const void *input, size_t input_size, 
                          void **output, size_t *output_size, compression_algorithm_t algorithm);
extern int decompress_memory(const void *input, size_t input_size,
                            void **output, size_t *output_size, compression_algorithm_t algorithm);
extern void print_compression_stats(void);
extern void cleanup_compression_system(void);

extern int init_swap_system(size_t max_gpu_memory, size_t page_size, double swap_threshold);
extern void* allocate_gpu_memory(size_t size);
extern void free_gpu_memory(void *ptr);
extern void* access_gpu_memory(void *ptr);
extern void print_swap_stats(void);
extern void cleanup_swap_system(void);

extern int init_unified_address_space(size_t total_size, size_t page_size);
extern void* allocate_unified_memory(size_t size, memory_type_t type, access_permission_t permissions);
extern void free_unified_memory(void *ptr);
extern void* access_unified_memory(void *ptr, access_permission_t required_permissions);
extern int sync_memory_region(void *ptr, size_t size);
extern int set_memory_permissions(void *ptr, size_t size, access_permission_t permissions);
extern void print_address_space_stats(void);
extern void cleanup_unified_address_space(void);

extern int init_memory_qos(uint32_t total_bandwidth_mbps, bandwidth_policy_t policy);
extern int submit_memory_request(void *address, size_t size, memory_access_type_t type, 
                                qos_level_t qos_level, uint32_t client_id);
extern void get_qos_stats(void);
extern void cleanup_memory_qos(void);

// 演示配置
typedef struct {
    bool enable_compression;     // 启用压缩
    bool enable_swap;           // 启用交换
    bool enable_unified_addr;   // 启用统一地址空间
    bool enable_qos;            // 启用QoS
    
    size_t test_data_size;      // 测试数据大小
    uint32_t num_threads;       // 线程数量
    uint32_t test_duration_sec; // 测试持续时间
    
    compression_algorithm_t compression_algo; // 压缩算法
    bandwidth_policy_t qos_policy;           // QoS策略
} demo_config_t;

// 测试统计信息
typedef struct {
    uint64_t total_operations;   // 总操作数
    uint64_t successful_ops;     // 成功操作数
    uint64_t failed_ops;         // 失败操作数
    uint64_t total_bytes;        // 总字节数
    uint64_t start_time;         // 开始时间
    uint64_t end_time;           // 结束时间
    pthread_mutex_t stats_lock;  // 统计锁
} test_stats_t;

// 工作线程参数
typedef struct {
    uint32_t thread_id;         // 线程ID
    demo_config_t *config;      // 配置
    test_stats_t *stats;        // 统计信息
    bool *shutdown_flag;        // 关闭标志
} worker_thread_args_t;

// 全局变量
static demo_config_t g_demo_config = {
    .enable_compression = true,
    .enable_swap = true,
    .enable_unified_addr = true,
    .enable_qos = true,
    .test_data_size = 1024 * 1024,  // 1MB
    .num_threads = 4,
    .test_duration_sec = 30,
    .compression_algo = COMPRESS_LZ4,
    .qos_policy = BANDWIDTH_POLICY_ADAPTIVE
};

static test_stats_t g_test_stats = {0};
static bool g_shutdown_flag = false;
static pthread_t *g_worker_threads = NULL;
static worker_thread_args_t *g_thread_args = NULL;

// 函数声明
static uint64_t get_current_time_us(void);
static void* generate_test_data(size_t size);
static void signal_handler(int sig);
static void print_demo_banner(void);
static void print_config(const demo_config_t *config);
static int init_all_systems(void);
static void cleanup_all_systems(void);
static void* worker_thread(void *arg);
static void test_compression_module(uint32_t thread_id);
static void test_swap_module(uint32_t thread_id);
static void test_unified_address_module(uint32_t thread_id);
static void test_qos_module(uint32_t thread_id);
static void update_stats(bool success, size_t bytes);
static void print_final_results(void);
static void run_performance_benchmark(void);
static void run_stress_test(void);
static void run_integration_test(void);

int main(int argc, char *argv[]) {
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    print_demo_banner();
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-compression") == 0) {
            g_demo_config.enable_compression = false;
        } else if (strcmp(argv[i], "--no-swap") == 0) {
            g_demo_config.enable_swap = false;
        } else if (strcmp(argv[i], "--no-unified") == 0) {
            g_demo_config.enable_unified_addr = false;
        } else if (strcmp(argv[i], "--no-qos") == 0) {
            g_demo_config.enable_qos = false;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            g_demo_config.num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            g_demo_config.test_duration_sec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            g_demo_config.test_data_size = atoi(argv[++i]) * 1024; // KB
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --no-compression  Disable compression module\n");
            printf("  --no-swap         Disable swap module\n");
            printf("  --no-unified      Disable unified address space\n");
            printf("  --no-qos          Disable QoS module\n");
            printf("  --threads N       Number of worker threads (default: 4)\n");
            printf("  --duration N      Test duration in seconds (default: 30)\n");
            printf("  --size N          Test data size in KB (default: 1024)\n");
            printf("  --help            Show this help message\n");
            return 0;
        }
    }
    
    print_config(&g_demo_config);
    
    // 初始化统计信息
    pthread_mutex_init(&g_test_stats.stats_lock, NULL);
    g_test_stats.start_time = get_current_time_us();
    
    // 初始化所有系统
    if (init_all_systems() != 0) {
        printf("Failed to initialize systems\n");
        return 1;
    }
    
    printf("\n=== Starting Memory Virtualization Demo ===\n\n");
    
    // 运行不同类型的测试
    printf("1. Running Performance Benchmark...\n");
    run_performance_benchmark();
    
    printf("\n2. Running Stress Test...\n");
    run_stress_test();
    
    printf("\n3. Running Integration Test...\n");
    run_integration_test();
    
    // 等待测试完成或用户中断
    printf("\nDemo running for %u seconds. Press Ctrl+C to stop early.\n", 
           g_demo_config.test_duration_sec);
    
    sleep(g_demo_config.test_duration_sec);
    
    // 停止测试
    g_shutdown_flag = true;
    
    // 等待所有工作线程结束
    if (g_worker_threads) {
        for (uint32_t i = 0; i < g_demo_config.num_threads; i++) {
            pthread_join(g_worker_threads[i], NULL);
        }
    }
    
    g_test_stats.end_time = get_current_time_us();
    
    // 打印最终结果
    print_final_results();
    
    // 清理系统
    cleanup_all_systems();
    
    // 清理资源
    if (g_worker_threads) {
        free(g_worker_threads);
    }
    if (g_thread_args) {
        free(g_thread_args);
    }
    
    pthread_mutex_destroy(&g_test_stats.stats_lock);
    
    printf("\n=== Demo Completed Successfully ===\n");
    return 0;
}

// 性能基准测试
static void run_performance_benchmark(void) {
    printf("  Testing individual module performance...\n");
    
    // 测试压缩性能
    if (g_demo_config.enable_compression) {
        void *test_data = generate_test_data(g_demo_config.test_data_size);
        void *compressed_data = NULL;
        size_t compressed_size = 0;
        
        uint64_t start = get_current_time_us();
        int result = compress_memory(test_data, g_demo_config.test_data_size,
                                   &compressed_data, &compressed_size, 
                                   g_demo_config.compression_algo);
        uint64_t end = get_current_time_us();
        
        if (result == 0) {
            double compression_ratio = (1.0 - (double)compressed_size / g_demo_config.test_data_size) * 100;
            double speed_mbps = (double)g_demo_config.test_data_size / (end - start) * 1000000 / (1024*1024);
            printf("    Compression: %.1f%% ratio, %.2f MB/s\n", compression_ratio, speed_mbps);
            free(compressed_data);
        }
        
        free(test_data);
    }
    
    // 测试内存分配性能
    if (g_demo_config.enable_swap) {
        uint64_t start = get_current_time_us();
        void *gpu_mem = allocate_gpu_memory(g_demo_config.test_data_size);
        uint64_t end = get_current_time_us();
        
        if (gpu_mem) {
            double alloc_speed = (double)g_demo_config.test_data_size / (end - start) * 1000000 / (1024*1024);
            printf("    GPU Memory Allocation: %.2f MB/s\n", alloc_speed);
            free_gpu_memory(gpu_mem);
        }
    }
    
    // 测试统一地址空间性能
    if (g_demo_config.enable_unified_addr) {
        uint64_t start = get_current_time_us();
        void *unified_mem = allocate_unified_memory(g_demo_config.test_data_size, 
                                                   MEMORY_TYPE_MANAGED, ACCESS_RW);
        uint64_t end = get_current_time_us();
        
        if (unified_mem) {
            double alloc_speed = (double)g_demo_config.test_data_size / (end - start) * 1000000 / (1024*1024);
            printf("    Unified Memory Allocation: %.2f MB/s\n", alloc_speed);
            free_unified_memory(unified_mem);
        }
    }
}

// 压力测试
static void run_stress_test(void) {
    printf("  Running stress test with %u threads...\n", g_demo_config.num_threads);
    
    // 分配线程资源
    g_worker_threads = malloc(g_demo_config.num_threads * sizeof(pthread_t));
    g_thread_args = malloc(g_demo_config.num_threads * sizeof(worker_thread_args_t));
    
    if (!g_worker_threads || !g_thread_args) {
        printf("Failed to allocate thread resources\n");
        return;
    }
    
    // 启动工作线程
    for (uint32_t i = 0; i < g_demo_config.num_threads; i++) {
        g_thread_args[i].thread_id = i;
        g_thread_args[i].config = &g_demo_config;
        g_thread_args[i].stats = &g_test_stats;
        g_thread_args[i].shutdown_flag = &g_shutdown_flag;
        
        if (pthread_create(&g_worker_threads[i], NULL, worker_thread, &g_thread_args[i]) != 0) {
            printf("Failed to create worker thread %u\n", i);
        }
    }
    
    printf("    Stress test started with %u threads\n", g_demo_config.num_threads);
}

// 集成测试
static void run_integration_test(void) {
    printf("  Testing module integration...\n");
    
    // 测试压缩+交换集成
    if (g_demo_config.enable_compression && g_demo_config.enable_swap) {
        void *test_data = generate_test_data(g_demo_config.test_data_size);
        void *compressed_data = NULL;
        size_t compressed_size = 0;
        
        // 压缩数据
        if (compress_memory(test_data, g_demo_config.test_data_size,
                           &compressed_data, &compressed_size, 
                           g_demo_config.compression_algo) == 0) {
            
            // 分配GPU内存并存储压缩数据
            void *gpu_mem = allocate_gpu_memory(compressed_size);
            if (gpu_mem) {
                memcpy(gpu_mem, compressed_data, compressed_size);
                printf("    Compression + Swap integration: SUCCESS\n");
                free_gpu_memory(gpu_mem);
            }
            
            free(compressed_data);
        }
        
        free(test_data);
    }
    
    // 测试统一地址空间+QoS集成
    if (g_demo_config.enable_unified_addr && g_demo_config.enable_qos) {
        void *unified_mem = allocate_unified_memory(g_demo_config.test_data_size,
                                                   MEMORY_TYPE_MANAGED, ACCESS_RW);
        if (unified_mem) {
            // 提交QoS请求
            submit_memory_request(unified_mem, g_demo_config.test_data_size,
                                ACCESS_TYPE_WRITE, QOS_LEVEL_HIGH, 1);
            
            printf("    Unified Address + QoS integration: SUCCESS\n");
            free_unified_memory(unified_mem);
        }
    }
}

// 工作线程函数
static void* worker_thread(void *arg) {
    worker_thread_args_t *args = (worker_thread_args_t*)arg;
    
    printf("    Worker thread %u started\n", args->thread_id);
    
    while (!*args->shutdown_flag) {
        // 随机选择测试模块
        int module = rand() % 4;
        
        switch (module) {
        case 0:
            if (args->config->enable_compression) {
                test_compression_module(args->thread_id);
            }
            break;
        case 1:
            if (args->config->enable_swap) {
                test_swap_module(args->thread_id);
            }
            break;
        case 2:
            if (args->config->enable_unified_addr) {
                test_unified_address_module(args->thread_id);
            }
            break;
        case 3:
            if (args->config->enable_qos) {
                test_qos_module(args->thread_id);
            }
            break;
        }
        
        // 短暂休息
        usleep(1000 + rand() % 9000); // 1-10ms
    }
    
    printf("    Worker thread %u finished\n", args->thread_id);
    return NULL;
}

// 测试压缩模块
static void test_compression_module(uint32_t thread_id) {
    size_t data_size = 1024 + rand() % (g_demo_config.test_data_size - 1024);
    void *test_data = generate_test_data(data_size);
    void *compressed_data = NULL;
    size_t compressed_size = 0;
    
    bool success = false;
    if (compress_memory(test_data, data_size, &compressed_data, &compressed_size,
                       g_demo_config.compression_algo) == 0) {
        // 测试解压缩
        void *decompressed_data = NULL;
        size_t decompressed_size = 0;
        
        if (decompress_memory(compressed_data, compressed_size,
                             &decompressed_data, &decompressed_size,
                             g_demo_config.compression_algo) == 0) {
            success = (decompressed_size == data_size && 
                      memcmp(test_data, decompressed_data, data_size) == 0);
            free(decompressed_data);
        }
        
        free(compressed_data);
    }
    
    update_stats(success, data_size);
    free(test_data);
}

// 测试交换模块
static void test_swap_module(uint32_t thread_id) {
    size_t alloc_size = 1024 + rand() % (g_demo_config.test_data_size - 1024);
    
    void *gpu_mem = allocate_gpu_memory(alloc_size);
    bool success = false;
    
    if (gpu_mem) {
        // 写入测试数据
        memset(gpu_mem, 0xAA + thread_id, alloc_size);
        
        // 访问内存
        void *accessed_mem = access_gpu_memory(gpu_mem);
        if (accessed_mem) {
            // 验证数据
            success = (((uint8_t*)accessed_mem)[0] == (0xAA + thread_id));
        }
        
        free_gpu_memory(gpu_mem);
    }
    
    update_stats(success, alloc_size);
}

// 测试统一地址空间模块
static void test_unified_address_module(uint32_t thread_id) {
    size_t alloc_size = 1024 + rand() % (g_demo_config.test_data_size - 1024);
    memory_type_t mem_type = (memory_type_t)(rand() % 4);
    
    void *unified_mem = allocate_unified_memory(alloc_size, mem_type, ACCESS_RW);
    bool success = false;
    
    if (unified_mem) {
        // 访问内存
        void *accessed_mem = access_unified_memory(unified_mem, ACCESS_WRITE);
        if (accessed_mem) {
            // 写入和读取测试
            memset(accessed_mem, 0xBB + thread_id, alloc_size);
            
            accessed_mem = access_unified_memory(unified_mem, ACCESS_READ);
            if (accessed_mem) {
                success = (((uint8_t*)accessed_mem)[0] == (0xBB + thread_id));
            }
        }
        
        free_unified_memory(unified_mem);
    }
    
    update_stats(success, alloc_size);
}

// 测试QoS模块
static void test_qos_module(uint32_t thread_id) {
    size_t request_size = 1024 + rand() % (g_demo_config.test_data_size - 1024);
    qos_level_t qos_level = (qos_level_t)(rand() % 5);
    memory_access_type_t access_type = (memory_access_type_t)(rand() % 4);
    
    // 模拟内存地址
    void *fake_addr = (void*)(0x10000000 + thread_id * 0x1000000);
    
    int result = submit_memory_request(fake_addr, request_size, access_type, 
                                      qos_level, thread_id);
    
    update_stats(result == 0, request_size);
}

// 更新统计信息
static void update_stats(bool success, size_t bytes) {
    pthread_mutex_lock(&g_test_stats.stats_lock);
    
    g_test_stats.total_operations++;
    g_test_stats.total_bytes += bytes;
    
    if (success) {
        g_test_stats.successful_ops++;
    } else {
        g_test_stats.failed_ops++;
    }
    
    pthread_mutex_unlock(&g_test_stats.stats_lock);
}

// 生成测试数据
static void* generate_test_data(size_t size) {
    void *data = malloc(size);
    if (!data) {
        return NULL;
    }
    
    // 生成半随机数据（有一定的重复性，便于压缩测试）
    uint8_t *bytes = (uint8_t*)data;
    for (size_t i = 0; i < size; i++) {
        if (i % 16 < 8) {
            bytes[i] = 0xAA; // 重复模式
        } else {
            bytes[i] = rand() % 256; // 随机数据
        }
    }
    
    return data;
}

// 初始化所有系统
static int init_all_systems(void) {
    printf("Initializing memory virtualization systems...\n");
    
    // 初始化压缩系统
    if (g_demo_config.enable_compression) {
        if (init_compression_system(g_demo_config.compression_algo, 
                                   QUALITY_BALANCED, true) != 0) {
            printf("Failed to initialize compression system\n");
            return -1;
        }
        printf("  ✓ Compression system initialized\n");
    }
    
    // 初始化交换系统
    if (g_demo_config.enable_swap) {
        size_t max_gpu_memory = 512 * 1024 * 1024; // 512MB
        size_t page_size = 4 * 1024; // 4KB
        double swap_threshold = 0.8; // 80%
        
        if (init_swap_system(max_gpu_memory, page_size, swap_threshold) != 0) {
            printf("Failed to initialize swap system\n");
            return -1;
        }
        printf("  ✓ Swap system initialized\n");
    }
    
    // 初始化统一地址空间
    if (g_demo_config.enable_unified_addr) {
        size_t total_size = 1024 * 1024 * 1024; // 1GB
        size_t page_size = 4 * 1024; // 4KB
        
        if (init_unified_address_space(total_size, page_size) != 0) {
            printf("Failed to initialize unified address space\n");
            return -1;
        }
        printf("  ✓ Unified address space initialized\n");
    }
    
    // 初始化QoS系统
    if (g_demo_config.enable_qos) {
        uint32_t total_bandwidth = 10000; // 10 GB/s
        
        if (init_memory_qos(total_bandwidth, g_demo_config.qos_policy) != 0) {
            printf("Failed to initialize QoS system\n");
            return -1;
        }
        printf("  ✓ QoS system initialized\n");
    }
    
    return 0;
}

// 清理所有系统
static void cleanup_all_systems(void) {
    printf("\nCleaning up systems...\n");
    
    if (g_demo_config.enable_qos) {
        cleanup_memory_qos();
        printf("  ✓ QoS system cleaned up\n");
    }
    
    if (g_demo_config.enable_unified_addr) {
        cleanup_unified_address_space();
        printf("  ✓ Unified address space cleaned up\n");
    }
    
    if (g_demo_config.enable_swap) {
        cleanup_swap_system();
        printf("  ✓ Swap system cleaned up\n");
    }
    
    if (g_demo_config.enable_compression) {
        cleanup_compression_system();
        printf("  ✓ Compression system cleaned up\n");
    }
}

// 打印最终结果
static void print_final_results(void) {
    printf("\n=== Final Test Results ===\n");
    
    uint64_t duration_us = g_test_stats.end_time - g_test_stats.start_time;
    double duration_sec = duration_us / 1000000.0;
    
    printf("Test Duration: %.2f seconds\n", duration_sec);
    printf("Total Operations: %llu\n", g_test_stats.total_operations);
    printf("Successful Operations: %llu\n", g_test_stats.successful_ops);
    printf("Failed Operations: %llu\n", g_test_stats.failed_ops);
    
    if (g_test_stats.total_operations > 0) {
        double success_rate = (double)g_test_stats.successful_ops / g_test_stats.total_operations * 100;
        printf("Success Rate: %.2f%%\n", success_rate);
        
        double ops_per_sec = g_test_stats.total_operations / duration_sec;
        printf("Operations per Second: %.2f\n", ops_per_sec);
        
        double throughput_mbps = (double)g_test_stats.total_bytes / duration_sec / (1024*1024);
        printf("Throughput: %.2f MB/s\n", throughput_mbps);
    }
    
    printf("Total Data Processed: %.2f MB\n", 
           (double)g_test_stats.total_bytes / (1024*1024));
    
    // 打印各模块统计信息
    printf("\n=== Module Statistics ===\n");
    
    if (g_demo_config.enable_compression) {
        print_compression_stats();
    }
    
    if (g_demo_config.enable_swap) {
        print_swap_stats();
    }
    
    if (g_demo_config.enable_unified_addr) {
        print_address_space_stats();
    }
    
    if (g_demo_config.enable_qos) {
        get_qos_stats();
    }
}

// 打印演示横幅
static void print_demo_banner(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              GPU Memory Virtualization Demo                 ║\n");
    printf("║                                                              ║\n");
    printf("║  Features:                                                   ║\n");
    printf("║  • Memory Compression (LZ4, ZSTD, Snappy)                   ║\n");
    printf("║  • Memory Swapping (Multi-tier Storage)                     ║\n");
    printf("║  • Unified Address Space Management                         ║\n");
    printf("║  • Memory QoS (Bandwidth Control & Latency Guarantee)       ║\n");
    printf("║                                                              ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

// 打印配置信息
static void print_config(const demo_config_t *config) {
    printf("Demo Configuration:\n");
    printf("  Compression: %s\n", config->enable_compression ? "Enabled" : "Disabled");
    printf("  Swap: %s\n", config->enable_swap ? "Enabled" : "Disabled");
    printf("  Unified Address Space: %s\n", config->enable_unified_addr ? "Enabled" : "Disabled");
    printf("  QoS: %s\n", config->enable_qos ? "Enabled" : "Disabled");
    printf("  Worker Threads: %u\n", config->num_threads);
    printf("  Test Duration: %u seconds\n", config->test_duration_sec);
    printf("  Test Data Size: %zu KB\n", config->test_data_size / 1024);
    printf("\n");
}

// 信号处理函数
static void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down gracefully...\n", sig);
    g_shutdown_flag = true;
}

// 获取当前时间（微秒）
static uint64_t get_current_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}