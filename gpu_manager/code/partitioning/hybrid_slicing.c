/**
 * 混合GPU切分策略实现
 * 来源：GPU 管理相关技术深度解析 - 混合切分策略
 * 功能：实现时间切分和空间切分的混合调度策略
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>

// 混合调度条目结构
typedef struct {
    int partition_id;        // 空间分区ID
    int time_slice_id;       // 时间片ID
    int priority;            // 优先级
    uint64_t start_time;     // 开始时间(微秒)
    uint64_t duration;       // 持续时间(微秒)
} hybrid_schedule_entry_t;

// GPU任务结构
typedef struct {
    int task_id;
    size_t memory_requirement;  // 内存需求(字节)
    uint64_t estimated_duration; // 预估执行时间(微秒)
    int priority;               // 任务优先级
    void *task_data;           // 任务数据
} gpu_task_t;

// 时间切分配置
typedef struct {
    uint64_t time_slice_duration_us; // 时间片长度(微秒)
    int max_time_slices;            // 最大时间片数
    double gpu_utilization;         // GPU利用率
    int context_switch_overhead_us; // 上下文切换开销(微秒)
} time_slicing_config_t;

// 空间切分配置
typedef struct {
    int num_partitions;             // 分区数量
    size_t partition_memory_size;   // 每个分区内存大小
    double isolation_efficiency;    // 隔离效率
    int memory_overhead_percent;    // 内存开销百分比
} spatial_slicing_config_t;

// 混合切分管理器
typedef struct {
    time_slicing_config_t time_config;
    spatial_slicing_config_t spatial_config;
    hybrid_schedule_entry_t *schedule_table;
    int schedule_capacity;
    int active_schedules;
    bool *partition_usage;          // 分区使用状态
    bool *time_slot_usage;         // 时间片使用状态
} hybrid_slicing_manager_t;

// 全局配置
static time_slicing_config_t time_slicing = {
    .time_slice_duration_us = 10000,  // 10ms时间片
    .max_time_slices = 100,
    .gpu_utilization = 0.85,
    .context_switch_overhead_us = 50
};

static spatial_slicing_config_t spatial_slicing = {
    .num_partitions = 8,
    .partition_memory_size = 1024 * 1024 * 1024, // 1GB per partition
    .isolation_efficiency = 0.92,
    .memory_overhead_percent = 5
};

static hybrid_slicing_manager_t hybrid_manager;

// 函数声明
static uint64_t get_current_time_us(void);
static int allocate_spatial_partition(size_t memory_requirement);
static int allocate_time_slot(int partition, uint64_t duration);
static void release_spatial_partition(int partition, size_t memory_requirement);
static int execute_hybrid_task(gpu_task_t *task, hybrid_schedule_entry_t *entry);
static double calculate_spatial_efficiency(void);
static double calculate_temporal_efficiency(void);
static int increase_spatial_partitions(void);
static int decrease_time_slice_duration(void);
static void evaluate_optimization_effectiveness(void);

// 混合切分任务调度算法
// 功能：结合空间分区和时间切分的二维调度策略
// 核心思想：先进行空间资源分配，再在分配的分区内进行时间调度
// 算法优势：同时实现资源隔离和时间复用，最大化GPU利用率
int schedule_hybrid_task(gpu_task_t *task) {
    // 参数有效性检查
    if (!task) {
        return -1;
    }
    
    printf("Scheduling hybrid task %d (memory: %zu bytes, duration: %llu us)\n",
           task->task_id, task->memory_requirement, (unsigned long long)task->estimated_duration);
    
    // === 第一阶段：空间分区分配 ===
    // 根据任务的内存需求分配独立的GPU分区
    // 空间分区提供硬件级别的资源隔离，防止任务间相互干扰
    int partition = allocate_spatial_partition(task->memory_requirement);
    if (partition < 0) {
        printf("Failed to allocate spatial partition\n");
        return -1;  // 空间资源不足，调度失败
    }
    
    // === 第二阶段：时间片分配 ===
    // 在已分配的分区内进行时间调度，实现分区内的时间复用
    // 时间调度考虑任务的预估执行时间和优先级
    int time_slot = allocate_time_slot(partition, task->estimated_duration);
    if (time_slot < 0) {
        // 时间调度失败，需要回滚空间分区分配
        release_spatial_partition(partition, task->memory_requirement);
        printf("Failed to allocate time slot\n");
        return -1;  // 时间资源不足，调度失败
    }
    
    // === 第三阶段：创建混合调度条目 ===
    // 将空间和时间维度的调度信息封装为统一的调度条目
    hybrid_schedule_entry_t entry = {
        .partition_id = partition,              // 空间分区标识
        .time_slice_id = time_slot,            // 时间片标识
        .priority = task->priority,            // 任务优先级
        .start_time = get_current_time_us(),   // 调度开始时间
        .duration = task->estimated_duration   // 预估执行时间
    };
    
    printf("Task %d scheduled: partition=%d, time_slot=%d\n",
           task->task_id, partition, time_slot);
    
    // === 第四阶段：执行混合调度任务 ===
    // 将任务提交到指定的分区和时间片中执行
    return execute_hybrid_task(task, &entry);
}

// 混合切分性能优化算法
// 功能：基于实时性能指标动态调整空间和时间切分策略
// 优化目标：最大化GPU资源利用率，最小化调度开销
// 核心机制：双维度自适应调优 + 阈值触发优化
int optimize_hybrid_slicing() {
    // === 第一步：性能指标采集和分析 ===
    // 计算空间维度的资源利用效率
    double spatial_efficiency = calculate_spatial_efficiency();
    // 计算时间维度的资源利用效率
    double temporal_efficiency = calculate_temporal_efficiency();
    
    printf("Current Efficiency - Spatial: %.1f%%, Temporal: %.1f%%\n", 
           spatial_efficiency * 100, temporal_efficiency * 100);
    
    // === 第二步：空间维度优化策略 ===
    // 当空间利用率低于70%时，说明分区粒度过粗，存在资源浪费
    if (spatial_efficiency < 0.7) {
        // 增加分区数量，提高空间资源的细粒度分配能力
        // 更多分区 -> 更好的资源匹配 -> 减少内部碎片
        if (increase_spatial_partitions() == 0) {
            printf("Increased spatial partitions for better efficiency\n");
        }
    }
    
    // === 第三步：时间维度优化策略 ===
    // 当时间利用率低于80%时，说明时间片过长，存在时间浪费
    if (temporal_efficiency < 0.8) {
        // 减少时间片长度，提高时间复用的精细度
        // 更短时间片 -> 更快响应 -> 更好的并发性
        if (decrease_time_slice_duration() == 0) {
            printf("Decreased time slice duration for better efficiency\n");
        }
    }
    
    // === 第四步：全局优化效果评估 ===
    // 量化评估优化效果和历史趋势分析
    evaluate_optimization_effectiveness();
    // 避免频繁调整导致的系统震荡
    
    return 0;  // 优化完成
}

// 切分策略对比分析
void compare_slicing_strategies() {
    printf("\nSlicing Strategy Comparison\n");
    printf("===========================\n");
    printf("Strategy        Isolation    Overhead    Flexibility    Complexity\n");
    printf("Time Slicing    Medium       Low         High          Low\n");
    printf("Spatial Slicing High         Medium      Medium        Medium\n");
    printf("Hybrid Slicing  High         Medium      High          High\n");
    
    printf("\nPerformance Metrics:\n");
    printf("Time Slicing:    %.1f%% GPU utilization\n", time_slicing.gpu_utilization * 100);
    printf("Spatial Slicing: %.1f%% isolation efficiency\n", spatial_slicing.isolation_efficiency * 100);
    printf("Hybrid Slicing:  %.1f%% overall efficiency\n", 
           (calculate_spatial_efficiency() + calculate_temporal_efficiency()) / 2 * 100);
}

// 混合切分管理器初始化算法
// 功能：初始化混合切分系统的核心数据结构和资源管理组件
// 初始化内容：调度表、分区状态、时间片状态、配置参数
// 设计原则：资源预分配 + 状态跟踪 + 错误回滚
int init_hybrid_slicing_manager() {
    // === 第一步：配置参数初始化 ===
    // 复制全局时间切分配置到管理器实例
    hybrid_manager.time_config = time_slicing;
    // 复制全局空间切分配置到管理器实例
    hybrid_manager.spatial_config = spatial_slicing;
    // 设置调度表容量（支持1000个并发调度条目）
    hybrid_manager.schedule_capacity = 1000;
    // 初始化活跃调度计数
    hybrid_manager.active_schedules = 0;
    
    // === 第二步：调度表内存分配 ===
    // 为混合调度条目分配连续内存空间
    // 调度表是系统的核心数据结构，存储所有活跃的调度信息
    hybrid_manager.schedule_table = malloc(hybrid_manager.schedule_capacity * sizeof(hybrid_schedule_entry_t));
    if (!hybrid_manager.schedule_table) {
        return -1;  // 内存分配失败
    }
    
    // === 第三步：空间分区状态跟踪数组分配 ===
    // 使用calloc确保初始状态为false（未使用）
    // 分区使用状态用于快速查找可用的空间分区
    hybrid_manager.partition_usage = calloc(spatial_slicing.num_partitions, sizeof(bool));
    if (!hybrid_manager.partition_usage) {
        free(hybrid_manager.schedule_table);  // 回滚已分配的内存
        return -1;
    }
    
    // === 第四步：时间片状态跟踪数组分配 ===
    // 时间片使用状态用于时间调度算法的快速决策
    hybrid_manager.time_slot_usage = calloc(time_slicing.max_time_slices, sizeof(bool));
    if (!hybrid_manager.time_slot_usage) {
        // 内存分配失败，回滚所有已分配的资源
        free(hybrid_manager.schedule_table);
        free(hybrid_manager.partition_usage);
        return -1;
    }
    
    // === 第五步：初始化完成确认 ===
    printf("Hybrid slicing manager initialized with %d partitions and %d time slices\n",
           spatial_slicing.num_partitions, time_slicing.max_time_slices);
    
    return 0;
}

// 清理混合切分管理器
void cleanup_hybrid_slicing_manager() {
    if (hybrid_manager.schedule_table) {
        free(hybrid_manager.schedule_table);
    }
    if (hybrid_manager.partition_usage) {
        free(hybrid_manager.partition_usage);
    }
    if (hybrid_manager.time_slot_usage) {
        free(hybrid_manager.time_slot_usage);
    }
    
    memset(&hybrid_manager, 0, sizeof(hybrid_manager));
}

// 获取混合切分统计信息
void get_hybrid_slicing_stats() {
    int used_partitions = 0;
    int used_time_slots = 0;
    
    for (int i = 0; i < spatial_slicing.num_partitions; i++) {
        if (hybrid_manager.partition_usage[i]) {
            used_partitions++;
        }
    }
    
    for (int i = 0; i < time_slicing.max_time_slices; i++) {
        if (hybrid_manager.time_slot_usage[i]) {
            used_time_slots++;
        }
    }
    
    printf("\n=== Hybrid Slicing Statistics ===\n");
    printf("Active Schedules: %d/%d\n", hybrid_manager.active_schedules, hybrid_manager.schedule_capacity);
    printf("Used Partitions: %d/%d (%.1f%%)\n", 
           used_partitions, spatial_slicing.num_partitions,
           (double)used_partitions / spatial_slicing.num_partitions * 100);
    printf("Used Time Slots: %d/%d (%.1f%%)\n", 
           used_time_slots, time_slicing.max_time_slices,
           (double)used_time_slots / time_slicing.max_time_slices * 100);
    printf("Spatial Efficiency: %.1f%%\n", calculate_spatial_efficiency() * 100);
    printf("Temporal Efficiency: %.1f%%\n", calculate_temporal_efficiency() * 100);
}

// 辅助函数实现
static uint64_t get_current_time_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000UL + ts.tv_nsec / 1000;
}

static int allocate_spatial_partition(size_t memory_requirement) {
    // 查找可用的空间分区
    for (int i = 0; i < spatial_slicing.num_partitions; i++) {
        if (!hybrid_manager.partition_usage[i] && 
            spatial_slicing.partition_memory_size >= memory_requirement) {
            hybrid_manager.partition_usage[i] = true;
            return i;
        }
    }
    return -1; // 没有可用分区
}

static int allocate_time_slot(int partition, uint64_t duration) {
    (void)partition; // 避免未使用参数警告
    (void)duration; // 避免未使用参数警告
    // 简化实现：查找可用时间片
    for (int i = 0; i < time_slicing.max_time_slices; i++) {
        if (!hybrid_manager.time_slot_usage[i]) {
            hybrid_manager.time_slot_usage[i] = true;
            return i;
        }
    }
    return -1; // 没有可用时间片
}

static void release_spatial_partition(int partition, size_t memory_requirement) {
    (void)memory_requirement; // 避免未使用参数警告
    // 释放空间分区
    if (partition >= 0 && partition < spatial_slicing.num_partitions) {
        hybrid_manager.partition_usage[partition] = false;
    }
}

static int execute_hybrid_task(gpu_task_t *task, hybrid_schedule_entry_t *entry) {
    // 模拟任务执行
    printf("Executing task %d in partition %d, time slot %d\n",
           task->task_id, entry->partition_id, entry->time_slice_id);
    
    // 添加到调度表
    if (hybrid_manager.active_schedules < hybrid_manager.schedule_capacity) {
        hybrid_manager.schedule_table[hybrid_manager.active_schedules++] = *entry;
    }
    
    return 0;
}

static double calculate_spatial_efficiency() {
    int used_partitions = 0;
    for (int i = 0; i < spatial_slicing.num_partitions; i++) {
        if (hybrid_manager.partition_usage[i]) {
            used_partitions++;
        }
    }
    return (double)used_partitions / spatial_slicing.num_partitions;
}

static double calculate_temporal_efficiency() {
    int used_time_slots = 0;
    for (int i = 0; i < time_slicing.max_time_slices; i++) {
        if (hybrid_manager.time_slot_usage[i]) {
            used_time_slots++;
        }
    }
    return (double)used_time_slots / time_slicing.max_time_slices;
}

static int increase_spatial_partitions() {
    // 简化实现：增加分区数量
    if (spatial_slicing.num_partitions < 16) {
        spatial_slicing.num_partitions += 2;
        return 0;
    }
    return -1;
}

static int decrease_time_slice_duration() {
    // 简化实现：减少时间片长度
    if (time_slicing.time_slice_duration_us > 1000) {
        time_slicing.time_slice_duration_us -= 1000;
        return 0;
    }
    return -1;
}

// 优化效果历史记录结构
typedef struct {
    uint64_t timestamp;
    double spatial_efficiency;
    double temporal_efficiency;
    double overall_utilization;
    int active_partitions;
    int active_time_slots;
} optimization_history_t;

// 优化效果统计
typedef struct {
    optimization_history_t history[100];  // 保存最近100次记录
    int history_count;
    int history_index;
    double avg_spatial_efficiency;
    double avg_temporal_efficiency;
    double trend_spatial;  // 空间效率趋势
    double trend_temporal; // 时间效率趋势
    uint64_t last_optimization_time;
    int optimization_count;
} optimization_stats_t;

static optimization_stats_t g_opt_stats = {0};

// 优化效果量化评估和历史趋势分析
static void evaluate_optimization_effectiveness() {
    uint64_t current_time = get_current_time_us();
    
    // 防止频繁调整（至少间隔1秒）
    if (current_time - g_opt_stats.last_optimization_time < 1000000) {
        return;
    }
    
    // 计算当前效率指标
    double spatial_eff = calculate_spatial_efficiency();
    double temporal_eff = calculate_temporal_efficiency();
    double overall_util = (spatial_eff + temporal_eff) / 2.0;
    
    // 记录历史数据
    optimization_history_t *current = &g_opt_stats.history[g_opt_stats.history_index];
    current->timestamp = current_time;
    current->spatial_efficiency = spatial_eff;
    current->temporal_efficiency = temporal_eff;
    current->overall_utilization = overall_util;
    current->active_partitions = spatial_slicing.num_partitions;
    current->active_time_slots = time_slicing.max_time_slices;
    
    // 更新历史索引
    g_opt_stats.history_index = (g_opt_stats.history_index + 1) % 100;
    if (g_opt_stats.history_count < 100) {
        g_opt_stats.history_count++;
    }
    
    // 计算平均效率
    double sum_spatial = 0.0, sum_temporal = 0.0;
    for (int i = 0; i < g_opt_stats.history_count; i++) {
        sum_spatial += g_opt_stats.history[i].spatial_efficiency;
        sum_temporal += g_opt_stats.history[i].temporal_efficiency;
    }
    g_opt_stats.avg_spatial_efficiency = sum_spatial / g_opt_stats.history_count;
    g_opt_stats.avg_temporal_efficiency = sum_temporal / g_opt_stats.history_count;
    
    // 计算趋势（最近10次记录的线性趋势）
    if (g_opt_stats.history_count >= 10) {
        int start_idx = (g_opt_stats.history_index - 10 + 100) % 100;
        double spatial_trend = 0.0, temporal_trend = 0.0;
        
        for (int i = 0; i < 9; i++) {
            int curr_idx = (start_idx + i + 1) % 100;
            int prev_idx = (start_idx + i) % 100;
            spatial_trend += g_opt_stats.history[curr_idx].spatial_efficiency - 
                           g_opt_stats.history[prev_idx].spatial_efficiency;
            temporal_trend += g_opt_stats.history[curr_idx].temporal_efficiency - 
                            g_opt_stats.history[prev_idx].temporal_efficiency;
        }
        
        g_opt_stats.trend_spatial = spatial_trend / 9.0;
        g_opt_stats.trend_temporal = temporal_trend / 9.0;
    }
    
    // 输出优化效果评估报告
    printf("\n=== Optimization Effectiveness Report ===\n");
    printf("Current Spatial Efficiency: %.2f%%\n", spatial_eff * 100);
    printf("Current Temporal Efficiency: %.2f%%\n", temporal_eff * 100);
    printf("Overall Utilization: %.2f%%\n", overall_util * 100);
    printf("Average Spatial Efficiency: %.2f%%\n", g_opt_stats.avg_spatial_efficiency * 100);
    printf("Average Temporal Efficiency: %.2f%%\n", g_opt_stats.avg_temporal_efficiency * 100);
    
    if (g_opt_stats.history_count >= 10) {
        printf("Spatial Efficiency Trend: %+.3f%%\n", g_opt_stats.trend_spatial * 100);
        printf("Temporal Efficiency Trend: %+.3f%%\n", g_opt_stats.trend_temporal * 100);
        
        // 趋势分析和建议
        if (g_opt_stats.trend_spatial < -0.05) {
            printf("Warning: Spatial efficiency declining, consider partition rebalancing\n");
        }
        if (g_opt_stats.trend_temporal < -0.05) {
            printf("Warning: Temporal efficiency declining, consider time slice adjustment\n");
        }
        if (g_opt_stats.trend_spatial > 0.05 && g_opt_stats.trend_temporal > 0.05) {
            printf("Good: Both efficiencies improving, optimization working well\n");
        }
    }
    
    g_opt_stats.last_optimization_time = current_time;
    g_opt_stats.optimization_count++;
    printf("Optimization Count: %d\n", g_opt_stats.optimization_count);
    printf("==========================================\n\n");
}

int main() {
    printf("Hybrid GPU Slicing Module\n");
    return 0;
}