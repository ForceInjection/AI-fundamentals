// 自适应时间片分配算法实现
// 根据任务类型、GPU利用率和历史性能动态调整时间片

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DEFAULT_TIMESLICE 10  // 默认时间片（毫秒）
#define MIN_TIMESLICE 1       // 最小时间片
#define MAX_TIMESLICE 100     // 最大时间片

// 任务类型枚举
typedef enum {
    TASK_TYPE_TRAINING,     // 训练任务
    TASK_TYPE_INFERENCE,    // 推理任务
    TASK_TYPE_COMPUTE,      // 通用计算
    TASK_TYPE_MEMORY_BOUND, // 内存密集型
    TASK_TYPE_IO_BOUND      // IO密集型
} task_type_t;

// 任务结构
typedef struct {
    int task_id;
    task_type_t type;
    float priority;         // 任务优先级
    int execution_count;    // 执行次数
    float avg_execution_time; // 平均执行时间
    float last_performance; // 最近性能指标
    time_t creation_time;   // 创建时间
} task_t;

// GPU上下文
typedef struct {
    int gpu_id;
    float utilization;      // GPU利用率 (0.0-1.0)
    float memory_usage;     // 内存使用率 (0.0-1.0)
    float temperature;      // 温度
    int active_tasks;       // 活跃任务数
    float avg_response_time; // 平均响应时间
} gpu_context_t;

// 获取任务类型因子
float get_task_type_factor(task_type_t type) {
    switch (type) {
    case TASK_TYPE_TRAINING:
        return 1.5;  // 训练任务需要更长时间片
    case TASK_TYPE_INFERENCE:
        return 0.8;  // 推理任务需要快速响应
    case TASK_TYPE_COMPUTE:
        return 1.2;  // 计算任务中等时间片
    case TASK_TYPE_MEMORY_BOUND:
        return 1.3;  // 内存密集型任务
    case TASK_TYPE_IO_BOUND:
        return 0.6;  // IO密集型任务时间片较短
    default:
        return 1.0;
    }
}

// 获取GPU利用率
float get_gpu_utilization(gpu_context_t *context) {
    // 模拟GPU利用率获取
    // 实际实现中需要调用NVIDIA-ML API
    return context->utilization;
}

// 计算性能因子
float calculate_performance_factor(task_t *task) {
    if (task->execution_count == 0) {
        return 1.0;  // 新任务使用默认因子
    }
    
    // 基于历史性能调整
    float perf_ratio = task->last_performance / task->avg_execution_time;
    
    if (perf_ratio > 1.2) {
        return 1.3;  // 性能良好，给予更多时间片
    } else if (perf_ratio < 0.8) {
        return 0.7;  // 性能较差，减少时间片
    } else {
        return 1.0;  // 性能正常
    }
}

// 限制值在指定范围内
int clamp(int value, int min_val, int max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// 自适应时间片计算
int calculate_adaptive_timeslice(task_t *task, gpu_context_t *context) {
    // 基础时间片
    int base_timeslice = DEFAULT_TIMESLICE;
    
    // 根据任务类型调整
    float type_factor = get_task_type_factor(task->type);
    
    // 根据GPU利用率调整
    float utilization = get_gpu_utilization(context);
    float util_factor = (utilization < 0.8) ? 1.2 : 0.8;
    
    // 根据任务历史性能调整
    float perf_factor = calculate_performance_factor(task);
    
    // 根据任务优先级调整
    float priority_factor = 0.5 + task->priority;  // 优先级范围0.5-1.5
    
    // 根据系统负载调整
    float load_factor = 1.0;
    if (context->active_tasks > 10) {
        load_factor = 0.8;  // 高负载时减少时间片
    } else if (context->active_tasks < 3) {
        load_factor = 1.3;  // 低负载时增加时间片
    }
    
    // 计算最终时间片
    int adaptive_timeslice = (int)(base_timeslice * type_factor * 
                                  util_factor * perf_factor * 
                                  priority_factor * load_factor);
    
    // 限制在合理范围内
    return clamp(adaptive_timeslice, MIN_TIMESLICE, MAX_TIMESLICE);
}

// 更新任务性能统计
void update_task_performance(task_t *task, float execution_time) {
    task->execution_count++;
    
    if (task->execution_count == 1) {
        task->avg_execution_time = execution_time;
    } else {
        // 指数移动平均
        float alpha = 0.3;  // 平滑因子
        task->avg_execution_time = alpha * execution_time + 
                                  (1 - alpha) * task->avg_execution_time;
    }
    
    task->last_performance = execution_time;
}

// 创建任务
task_t* create_task(int task_id, task_type_t type, float priority) {
    task_t *task = malloc(sizeof(task_t));
    if (!task) return NULL;
    
    task->task_id = task_id;
    task->type = type;
    task->priority = priority;
    task->execution_count = 0;
    task->avg_execution_time = 0.0;
    task->last_performance = 0.0;
    task->creation_time = time(NULL);
    
    return task;
}

// 创建GPU上下文
gpu_context_t* create_gpu_context(int gpu_id) {
    gpu_context_t *context = malloc(sizeof(gpu_context_t));
    if (!context) return NULL;
    
    context->gpu_id = gpu_id;
    context->utilization = 0.5;  // 初始利用率50%
    context->memory_usage = 0.3;  // 初始内存使用30%
    context->temperature = 65.0;  // 初始温度65°C
    context->active_tasks = 5;    // 初始活跃任务数
    context->avg_response_time = 10.0;  // 初始平均响应时间
    
    return context;
}

// 模拟GPU利用率变化
void simulate_gpu_load_change(gpu_context_t *context) {
    // 模拟利用率在0.2-0.95之间变化
    context->utilization += (rand() % 21 - 10) / 100.0;  // ±10%变化
    if (context->utilization < 0.2) context->utilization = 0.2;
    if (context->utilization > 0.95) context->utilization = 0.95;
    
    // 活跃任务数变化
    context->active_tasks += rand() % 5 - 2;  // ±2变化
    if (context->active_tasks < 1) context->active_tasks = 1;
    if (context->active_tasks > 20) context->active_tasks = 20;
}

// 时间片调度器
typedef struct {
    task_t **tasks;
    int task_count;
    gpu_context_t *context;
    int current_task_index;
} timeslice_scheduler_t;

// 执行时间片调度
void execute_timeslice_scheduling(timeslice_scheduler_t *scheduler) {
    if (scheduler->task_count == 0) return;
    
    task_t *current_task = scheduler->tasks[scheduler->current_task_index];
    
    // 计算自适应时间片
    int timeslice = calculate_adaptive_timeslice(current_task, scheduler->context);
    
    printf("Task %d (type=%d, priority=%.2f): timeslice=%dms, GPU util=%.2f\n",
           current_task->task_id, current_task->type, current_task->priority,
           timeslice, scheduler->context->utilization);
    
    // 模拟任务执行
    float execution_time = timeslice + (rand() % 10 - 5);  // 模拟执行时间变化
    update_task_performance(current_task, execution_time);
    
    // 切换到下一个任务
    scheduler->current_task_index = (scheduler->current_task_index + 1) % scheduler->task_count;
    
    // 模拟GPU负载变化
    simulate_gpu_load_change(scheduler->context);
}

// 示例使用
int main() {
    srand(time(NULL));
    
    // 创建GPU上下文
    gpu_context_t *context = create_gpu_context(0);
    
    // 创建不同类型的任务
    task_t *tasks[5];
    tasks[0] = create_task(1, TASK_TYPE_TRAINING, 0.8);
    tasks[1] = create_task(2, TASK_TYPE_INFERENCE, 0.9);
    tasks[2] = create_task(3, TASK_TYPE_COMPUTE, 0.6);
    tasks[3] = create_task(4, TASK_TYPE_MEMORY_BOUND, 0.7);
    tasks[4] = create_task(5, TASK_TYPE_IO_BOUND, 0.5);
    
    // 创建调度器
    timeslice_scheduler_t scheduler = {
        .tasks = tasks,
        .task_count = 5,
        .context = context,
        .current_task_index = 0
    };
    
    // 模拟调度过程
    printf("=== Adaptive Timeslice Scheduling Simulation ===\n");
    for (int i = 0; i < 20; i++) {
        printf("\nRound %d:\n", i + 1);
        execute_timeslice_scheduling(&scheduler);
    }
    
    // 清理资源
    for (int i = 0; i < 5; i++) {
        free(tasks[i]);
    }
    free(context);
    
    return 0;
}