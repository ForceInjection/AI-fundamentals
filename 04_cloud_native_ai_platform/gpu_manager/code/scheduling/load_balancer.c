// 多GPU负载均衡算法实现
// 支持最小负载优先、轮询和加权轮询等多种负载均衡策略

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>

#define MAX_GPUS 8
#define MAX_TASKS_PER_GPU 100

// GPU信息结构
typedef struct {
    int gpu_id;
    float compute_capability;   // 计算能力
    size_t total_memory;        // 总内存
    size_t available_memory;    // 可用内存
    float utilization;          // 当前利用率
    float temperature;          // 温度
    int active_tasks;           // 活跃任务数
    float power_consumption;    // 功耗
    int is_available;           // 是否可用
} gpu_info_t;

// 任务信息结构
typedef struct {
    int task_id;
    size_t memory_required;     // 所需内存
    float compute_required;     // 所需计算资源
    int priority;               // 优先级
    float estimated_duration;   // 预估执行时间
    int gpu_compatibility;      // GPU兼容性要求
} task_t;

// 负载均衡算法类型
typedef enum {
    LB_ROUND_ROBIN,            // 轮询
    LB_LEAST_LOADED,           // 最小负载
    LB_WEIGHTED_ROUND_ROBIN,   // 加权轮询
    LB_LEAST_CONNECTIONS,      // 最少连接
    LB_RESOURCE_AWARE          // 资源感知
} load_balance_algorithm_t;

// 负载均衡器结构
typedef struct {
    gpu_info_t *gpu_list;
    int gpu_count;
    float *load_factors;        // 负载因子
    int balancing_algorithm;
    int round_robin_index;      // 轮询索引
    float *weights;             // 权重（用于加权轮询）
} load_balancer_t;

// 检查GPU兼容性
int is_gpu_compatible(gpu_info_t gpu, task_t *task) {
    // 检查内存需求
    if (task->memory_required > gpu.available_memory) {
        return 0;
    }
    
    // 检查GPU是否可用
    if (!gpu.is_available) {
        return 0;
    }
    
    // 检查温度限制（防止过热）
    if (gpu.temperature > 85.0) {
        return 0;
    }
    
    // 检查计算能力兼容性
    if (task->gpu_compatibility > 0 && 
        gpu.compute_capability < task->gpu_compatibility) {
        return 0;
    }
    
    return 1;
}

// 计算GPU负载
float calculate_gpu_load(gpu_info_t gpu) {
    // 综合考虑多个因素的负载计算
    float util_factor = gpu.utilization;
    float memory_factor = (float)(gpu.total_memory - gpu.available_memory) / gpu.total_memory;
    float task_factor = (float)gpu.active_tasks / MAX_TASKS_PER_GPU;
    float temp_factor = (gpu.temperature - 30.0) / 55.0;  // 30-85°C范围
    
    // 加权平均
    float load = 0.4 * util_factor + 0.3 * memory_factor + 
                 0.2 * task_factor + 0.1 * temp_factor;
    
    return fmaxf(0.0, fminf(1.0, load));  // 限制在0-1范围
}

// 预测任务负载
float predict_task_load(task_t *task, gpu_info_t gpu) {
    // 基于任务需求预测增加的负载
    float memory_load = (float)task->memory_required / gpu.total_memory;
    float compute_load = task->compute_required / gpu.compute_capability;
    float duration_factor = task->estimated_duration / 100.0;  // 归一化
    
    return 0.5 * memory_load + 0.3 * compute_load + 0.2 * duration_factor;
}

// 最小负载优先算法
int select_optimal_gpu(load_balancer_t *balancer, task_t *task) {
    int selected_gpu = -1;
    float min_load = FLOAT_MAX;
    
    for (int i = 0; i < balancer->gpu_count; i++) {
        // 检查GPU兼容性
        if (!is_gpu_compatible(balancer->gpu_list[i], task)) {
            continue;
        }
        
        // 计算综合负载指标
        float current_load = calculate_gpu_load(balancer->gpu_list[i]);
        float predicted_load = predict_task_load(task, balancer->gpu_list[i]);
        float total_load = current_load + predicted_load;
        
        if (total_load < min_load) {
            min_load = total_load;
            selected_gpu = i;
        }
    }
    
    return selected_gpu;
}

// 轮询算法
int select_round_robin_gpu(load_balancer_t *balancer, task_t *task) {
    int start_index = balancer->round_robin_index;
    
    for (int i = 0; i < balancer->gpu_count; i++) {
        int gpu_index = (start_index + i) % balancer->gpu_count;
        
        if (is_gpu_compatible(balancer->gpu_list[gpu_index], task)) {
            balancer->round_robin_index = (gpu_index + 1) % balancer->gpu_count;
            return gpu_index;
        }
    }
    
    return -1;  // 没有可用GPU
}

// 加权轮询算法
int select_weighted_round_robin_gpu(load_balancer_t *balancer, task_t *task) {
    static int *weight_counters = NULL;
    
    if (!weight_counters) {
        weight_counters = calloc(balancer->gpu_count, sizeof(int));
    }
    
    // 找到权重计数器最小的可用GPU
    int selected_gpu = -1;
    int min_counter = INT_MAX;
    
    for (int i = 0; i < balancer->gpu_count; i++) {
        if (is_gpu_compatible(balancer->gpu_list[i], task) &&
            weight_counters[i] < min_counter) {
            min_counter = weight_counters[i];
            selected_gpu = i;
        }
    }
    
    if (selected_gpu >= 0) {
        weight_counters[selected_gpu] += (int)(1.0 / balancer->weights[selected_gpu]);
    }
    
    return selected_gpu;
}

// 最少连接算法
int select_least_connections_gpu(load_balancer_t *balancer, task_t *task) {
    int selected_gpu = -1;
    int min_connections = INT_MAX;
    
    for (int i = 0; i < balancer->gpu_count; i++) {
        if (is_gpu_compatible(balancer->gpu_list[i], task) &&
            balancer->gpu_list[i].active_tasks < min_connections) {
            min_connections = balancer->gpu_list[i].active_tasks;
            selected_gpu = i;
        }
    }
    
    return selected_gpu;
}

// 资源感知算法
int select_resource_aware_gpu(load_balancer_t *balancer, task_t *task) {
    int selected_gpu = -1;
    float best_score = -1.0;
    
    for (int i = 0; i < balancer->gpu_count; i++) {
        if (!is_gpu_compatible(balancer->gpu_list[i], task)) {
            continue;
        }
        
        gpu_info_t *gpu = &balancer->gpu_list[i];
        
        // 计算资源匹配度评分
        float memory_score = (float)gpu->available_memory / task->memory_required;
        float compute_score = gpu->compute_capability / task->compute_required;
        float load_score = 1.0 - calculate_gpu_load(*gpu);
        float priority_score = (float)task->priority / 10.0;
        
        // 综合评分
        float total_score = 0.3 * memory_score + 0.3 * compute_score + 
                           0.3 * load_score + 0.1 * priority_score;
        
        if (total_score > best_score) {
            best_score = total_score;
            selected_gpu = i;
        }
    }
    
    return selected_gpu;
}

// 选择GPU（根据算法）
int select_gpu_for_task(load_balancer_t *balancer, task_t *task) {
    switch (balancer->balancing_algorithm) {
    case LB_ROUND_ROBIN:
        return select_round_robin_gpu(balancer, task);
    case LB_LEAST_LOADED:
        return select_optimal_gpu(balancer, task);
    case LB_WEIGHTED_ROUND_ROBIN:
        return select_weighted_round_robin_gpu(balancer, task);
    case LB_LEAST_CONNECTIONS:
        return select_least_connections_gpu(balancer, task);
    case LB_RESOURCE_AWARE:
        return select_resource_aware_gpu(balancer, task);
    default:
        return select_optimal_gpu(balancer, task);
    }
}

// 分配任务到GPU
int assign_task_to_gpu(load_balancer_t *balancer, task_t *task) {
    int gpu_index = select_gpu_for_task(balancer, task);
    
    if (gpu_index < 0) {
        printf("No suitable GPU found for task %d\n", task->task_id);
        return -1;
    }
    
    // 更新GPU状态
    gpu_info_t *gpu = &balancer->gpu_list[gpu_index];
    gpu->available_memory -= task->memory_required;
    gpu->active_tasks++;
    gpu->utilization += predict_task_load(task, *gpu);
    
    printf("Task %d assigned to GPU %d (load: %.2f, memory: %zu MB)\n",
           task->task_id, gpu->gpu_id, calculate_gpu_load(*gpu),
           gpu->available_memory / (1024 * 1024));
    
    return gpu_index;
}

// 初始化负载均衡器
load_balancer_t* init_load_balancer(int gpu_count, load_balance_algorithm_t algorithm) {
    load_balancer_t *balancer = malloc(sizeof(load_balancer_t));
    if (!balancer) return NULL;
    
    balancer->gpu_list = malloc(gpu_count * sizeof(gpu_info_t));
    balancer->load_factors = calloc(gpu_count, sizeof(float));
    balancer->weights = malloc(gpu_count * sizeof(float));
    balancer->gpu_count = gpu_count;
    balancer->balancing_algorithm = algorithm;
    balancer->round_robin_index = 0;
    
    // 初始化GPU信息（模拟数据）
    for (int i = 0; i < gpu_count; i++) {
        gpu_info_t *gpu = &balancer->gpu_list[i];
        gpu->gpu_id = i;
        gpu->compute_capability = 7.0 + (rand() % 20) / 10.0;  // 7.0-9.0
        gpu->total_memory = (8 + rand() % 25) * 1024 * 1024 * 1024UL;  // 8-32GB
        gpu->available_memory = gpu->total_memory * (0.7 + (rand() % 30) / 100.0);  // 70-100%
        gpu->utilization = (rand() % 50) / 100.0;  // 0-50%
        gpu->temperature = 40.0 + rand() % 30;  // 40-70°C
        gpu->active_tasks = rand() % 10;  // 0-9个任务
        gpu->power_consumption = 150.0 + rand() % 200;  // 150-350W
        gpu->is_available = 1;
        
        // 设置权重（基于计算能力）
        balancer->weights[i] = gpu->compute_capability / 10.0;
    }
    
    return balancer;
}

// 清理负载均衡器
void cleanup_load_balancer(load_balancer_t *balancer) {
    if (balancer) {
        free(balancer->gpu_list);
        free(balancer->load_factors);
        free(balancer->weights);
        free(balancer);
    }
}

// 创建测试任务
task_t* create_test_task(int task_id) {
    task_t *task = malloc(sizeof(task_t));
    if (!task) return NULL;
    
    task->task_id = task_id;
    task->memory_required = (1 + rand() % 8) * 1024 * 1024 * 1024UL;  // 1-8GB
    task->compute_required = 1.0 + (rand() % 50) / 10.0;  // 1.0-6.0
    task->priority = 1 + rand() % 10;  // 1-10
    task->estimated_duration = 10.0 + rand() % 100;  // 10-110秒
    task->gpu_compatibility = 0;  // 无特殊要求
    
    return task;
}

// 示例使用
int main() {
    srand(time(NULL));
    
    // 测试不同的负载均衡算法
    const char* algorithm_names[] = {
        "Round Robin", "Least Loaded", "Weighted Round Robin",
        "Least Connections", "Resource Aware"
    };
    
    for (int alg = 0; alg < 5; alg++) {
        printf("\n=== Testing %s Algorithm ===\n", algorithm_names[alg]);
        
        // 初始化负载均衡器
        load_balancer_t *balancer = init_load_balancer(4, alg);
        
        // 创建并分配任务
        for (int i = 0; i < 10; i++) {
            task_t *task = create_test_task(i + 1);
            assign_task_to_gpu(balancer, task);
            free(task);
        }
        
        // 显示最终GPU状态
        printf("\nFinal GPU Status:\n");
        for (int i = 0; i < balancer->gpu_count; i++) {
            gpu_info_t *gpu = &balancer->gpu_list[i];
            printf("GPU %d: Load=%.2f, Tasks=%d, Memory=%zu MB\n",
                   gpu->gpu_id, calculate_gpu_load(*gpu), gpu->active_tasks,
                   gpu->available_memory / (1024 * 1024));
        }
        
        cleanup_load_balancer(balancer);
    }
    
    return 0;
}