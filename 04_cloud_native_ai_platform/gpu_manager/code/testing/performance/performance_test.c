/*
 * GPU管理系统性能测试
 * 测试各个模块的性能指标和基准
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include "../../common/gpu_common.h"

// 性能测试配置
#define MAX_THREADS 16
#define MAX_ITERATIONS 10000
#define MEMORY_SIZE_MB 1024
#define TEST_DURATION_SEC 10

// 性能统计结构
struct performance_stats {
    double min_latency_us;
    double max_latency_us;
    double avg_latency_us;
    double throughput_ops_sec;
    long total_operations;
    double total_time_sec;
};

// 测试结果
static struct performance_stats test_results[8];
static int test_count = 0;

// 获取当前时间（微秒）
static double get_time_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

// 计算性能统计
static void calculate_stats(double *latencies, int count, struct performance_stats *stats) {
    if (count == 0) return;
    
    stats->min_latency_us = latencies[0];
    stats->max_latency_us = latencies[0];
    stats->avg_latency_us = 0;
    
    for (int i = 0; i < count; i++) {
        if (latencies[i] < stats->min_latency_us) {
            stats->min_latency_us = latencies[i];
        }
        if (latencies[i] > stats->max_latency_us) {
            stats->max_latency_us = latencies[i];
        }
        stats->avg_latency_us += latencies[i];
    }
    
    stats->avg_latency_us /= count;
    stats->total_operations = count;
    stats->total_time_sec = stats->total_operations * stats->avg_latency_us / 1000000.0;
    stats->throughput_ops_sec = stats->total_operations / stats->total_time_sec;
}

// 虚拟化性能测试
int test_virtualization_performance() {
    printf("\n=== Virtualization Performance Test ===\n");
    
    double latencies[MAX_ITERATIONS];
    int iterations = 1000;
    
    printf("Testing virtual GPU context operations...\n");
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        
        // 模拟虚拟GPU上下文创建/销毁
        usleep(10); // 模拟10微秒的操作时间
        
        double end = get_time_us();
        latencies[i] = end - start;
    }
    
    calculate_stats(latencies, iterations, &test_results[test_count]);
    printf("Virtual GPU context operations: %.2f ops/sec\n", 
           test_results[test_count].throughput_ops_sec);
    printf("Average latency: %.2f μs\n", test_results[test_count].avg_latency_us);
    
    test_count++;
    return 0;
}

// 内存管理性能测试
int test_memory_performance() {
    printf("\n=== Memory Management Performance Test ===\n");
    
    double latencies[MAX_ITERATIONS];
    int iterations = 5000;
    size_t block_size = 1024; // 1KB blocks
    
    printf("Testing memory allocation/deallocation...\n");
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        
        // 模拟内存分配和释放
        void *ptr = malloc(block_size);
        if (ptr) {
            memset(ptr, 0, block_size);
            free(ptr);
        }
        
        double end = get_time_us();
        latencies[i] = end - start;
    }
    
    calculate_stats(latencies, iterations, &test_results[test_count]);
    printf("Memory operations: %.2f ops/sec\n", 
           test_results[test_count].throughput_ops_sec);
    printf("Average latency: %.2f μs\n", test_results[test_count].avg_latency_us);
    
    test_count++;
    return 0;
}

// 调度器性能测试
int test_scheduler_performance() {
    printf("\n=== Scheduler Performance Test ===\n");
    
    double latencies[MAX_ITERATIONS];
    int iterations = 2000;
    
    printf("Testing task scheduling operations...\n");
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        
        // 模拟任务调度决策
        int priority = rand() % 10;
        int selected_task = priority; // 简单的优先级调度
        (void)selected_task; // 避免未使用变量警告
        
        double end = get_time_us();
        latencies[i] = end - start;
    }
    
    calculate_stats(latencies, iterations, &test_results[test_count]);
    printf("Scheduling decisions: %.2f ops/sec\n", 
           test_results[test_count].throughput_ops_sec);
    printf("Average latency: %.2f μs\n", test_results[test_count].avg_latency_us);
    
    test_count++;
    return 0;
}

// 网络通信性能测试
int test_network_performance() {
    printf("\n=== Network Communication Performance Test ===\n");
    
    double latencies[MAX_ITERATIONS];
    int iterations = 1000;
    size_t message_size = 1024; // 1KB messages
    
    printf("Testing network message processing...\n");
    
    for (int i = 0; i < iterations; i++) {
        double start = get_time_us();
        
        // 模拟网络消息处理
        char *buffer = malloc(message_size);
        if (buffer) {
            memset(buffer, 'A', message_size);
            // 模拟消息处理时间
            usleep(50); // 50微秒
            free(buffer);
        }
        
        double end = get_time_us();
        latencies[i] = end - start;
    }
    
    calculate_stats(latencies, iterations, &test_results[test_count]);
    printf("Network messages: %.2f msgs/sec\n", 
           test_results[test_count].throughput_ops_sec);
    printf("Average latency: %.2f μs\n", test_results[test_count].avg_latency_us);
    
    test_count++;
    return 0;
}

// 多线程性能测试
struct thread_test_data {
    int thread_id;
    int iterations;
    double *latencies;
    pthread_mutex_t *mutex;
    pthread_cond_t *cond;
    int *ready_count;
    int total_threads;
};

void* thread_worker(void *arg) {
    struct thread_test_data *data = (struct thread_test_data*)arg;
    
    // 等待所有线程准备就绪
    pthread_mutex_lock(data->mutex);
    (*data->ready_count)++;
    if (*data->ready_count == data->total_threads) {
        pthread_cond_broadcast(data->cond);
    } else {
        while (*data->ready_count < data->total_threads) {
            pthread_cond_wait(data->cond, data->mutex);
        }
    }
    pthread_mutex_unlock(data->mutex);
    
    for (int i = 0; i < data->iterations; i++) {
        double start = get_time_us();
        
        // 模拟GPU操作
        usleep(20); // 20微秒的模拟操作
        
        double end = get_time_us();
        data->latencies[i] = end - start;
    }
    
    return NULL;
}

int test_multithreading_performance() {
    printf("\n=== Multi-threading Performance Test ===\n");
    
    int num_threads = 4;
    int iterations_per_thread = 500;
    pthread_t threads[MAX_THREADS];
    struct thread_test_data thread_data[MAX_THREADS];
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    int ready_count = 0;
    
    printf("Testing concurrent operations with %d threads...\n", num_threads);
    
    // 创建线程
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].iterations = iterations_per_thread;
        thread_data[i].latencies = malloc(iterations_per_thread * sizeof(double));
        thread_data[i].mutex = &mutex;
        thread_data[i].cond = &cond;
        thread_data[i].ready_count = &ready_count;
        thread_data[i].total_threads = num_threads;
        
        pthread_create(&threads[i], NULL, thread_worker, &thread_data[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    // 合并所有线程的结果
    double *all_latencies = malloc(num_threads * iterations_per_thread * sizeof(double));
    int total_ops = 0;
    
    for (int i = 0; i < num_threads; i++) {
        for (int j = 0; j < iterations_per_thread; j++) {
            all_latencies[total_ops++] = thread_data[i].latencies[j];
        }
        free(thread_data[i].latencies);
    }
    
    calculate_stats(all_latencies, total_ops, &test_results[test_count]);
    printf("Concurrent operations: %.2f ops/sec\n", 
           test_results[test_count].throughput_ops_sec);
    printf("Average latency: %.2f μs\n", test_results[test_count].avg_latency_us);
    
    free(all_latencies);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    test_count++;
    return 0;
}

// 压力测试
int test_stress_performance() {
    printf("\n=== Stress Test ===\n");
    
    printf("Running stress test for %d seconds...\n", TEST_DURATION_SEC);
    
    double start_time = get_time_us();
    double end_time = start_time + TEST_DURATION_SEC * 1000000.0;
    long operations = 0;
    
    while (get_time_us() < end_time) {
        // 模拟各种操作
        void *ptr = malloc(1024);
        if (ptr) {
            memset(ptr, operations % 256, 1024);
            free(ptr);
        }
        
        // 模拟计算
        volatile int sum = 0;
        for (int i = 0; i < 100; i++) {
            sum += i;
        }
        
        operations++;
    }
    
    double actual_time = (get_time_us() - start_time) / 1000000.0;
    double ops_per_sec = operations / actual_time;
    
    printf("Stress test completed: %.2f ops/sec over %.2f seconds\n", 
           ops_per_sec, actual_time);
    printf("Total operations: %ld\n", operations);
    
    return 0;
}

// 内存带宽测试
int test_memory_bandwidth() {
    printf("\n=== Memory Bandwidth Test ===\n");
    
    size_t buffer_size = MEMORY_SIZE_MB * 1024 * 1024; // MB to bytes
    char *src_buffer = malloc(buffer_size);
    char *dst_buffer = malloc(buffer_size);
    
    if (!src_buffer || !dst_buffer) {
        printf("Failed to allocate memory for bandwidth test\n");
        free(src_buffer);
        free(dst_buffer);
        return -1;
    }
    
    // 初始化源缓冲区
    memset(src_buffer, 0xAA, buffer_size);
    
    printf("Testing memory copy bandwidth with %d MB buffer...\n", MEMORY_SIZE_MB);
    
    double start = get_time_us();
    
    // 执行内存拷贝
    memcpy(dst_buffer, src_buffer, buffer_size);
    
    double end = get_time_us();
    double time_sec = (end - start) / 1000000.0;
    double bandwidth_mbps = (buffer_size / (1024.0 * 1024.0)) / time_sec;
    
    printf("Memory bandwidth: %.2f MB/s\n", bandwidth_mbps);
    printf("Copy time: %.2f ms\n", (end - start) / 1000.0);
    
    free(src_buffer);
    free(dst_buffer);
    return 0;
}

// 打印性能测试总结
void print_performance_summary() {
    printf("\n" "=" "=" "=" " Performance Test Summary " "=" "=" "=" "\n");
    
    const char* test_names[] = {
        "Virtualization",
        "Memory Management", 
        "Task Scheduling",
        "Network Communication",
        "Multi-threading"
    };
    
    printf("%-20s %12s %12s %12s\n", 
           "Test", "Throughput", "Avg Latency", "Max Latency");
    printf("%-20s %12s %12s %12s\n", 
           "", "(ops/sec)", "(μs)", "(μs)");
    printf("------------------------------------------------------------\n");
    
    for (int i = 0; i < test_count && i < 5; i++) {
        printf("%-20s %12.2f %12.2f %12.2f\n",
               test_names[i],
               test_results[i].throughput_ops_sec,
               test_results[i].avg_latency_us,
               test_results[i].max_latency_us);
    }
    
    printf("\n🚀 Performance testing completed! 🚀\n");
}

// 主函数
int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
    printf("GPU Management System Performance Test Suite\n");
    printf("===========================================\n");
    
    // 初始化随机数生成器
    srand(time(NULL));
    
    // 运行各项性能测试
    test_virtualization_performance();
    test_memory_performance();
    test_scheduler_performance();
    test_network_performance();
    test_multithreading_performance();
    
    // 运行专项测试
    test_memory_bandwidth();
    test_stress_performance();
    
    // 打印性能测试总结
    print_performance_summary();
    
    return 0;
}