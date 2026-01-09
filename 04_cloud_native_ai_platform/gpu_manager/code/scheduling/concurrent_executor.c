/*
 * GPU并发执行器实现
 * 负责管理GPU任务的并发执行、负载均衡和资源分配
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/time.h>
#include <stdatomic.h>
#include "../common/gpu_common.h"

// GPU任务结构
typedef struct gpu_task {
    int task_id;
    int priority;
    int device_id;
    size_t memory_required;
    unsigned long start_time;
    unsigned long deadline;
    int state;
    struct list_head list;
    struct list_head device_list;
} gpu_task_t;

// 任务状态定义
enum task_state {
    TASK_PENDING = 0,
    TASK_READY,
    TASK_RUNNING,
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_ISOLATED
};

// 并发执行器结构
typedef struct {
    pthread_t *worker_threads;
    int num_workers;
    struct list_head task_queue;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_cond;
    atomic_t active_tasks;
    atomic_t completed_tasks;
    int shutdown;
} concurrent_executor_t;

// 负载均衡器结构
typedef struct {
    int *gpu_loads;           // 每个GPU的负载
    int *gpu_memory_usage;    // 每个GPU的内存使用
    int num_gpus;
    pthread_mutex_t balance_mutex;
    struct timeval last_balance_time;
} load_balancer_t;

// 全局变量
static concurrent_executor_t *global_executor = NULL;
static load_balancer_t *global_balancer = NULL;

// 函数声明
int init_concurrent_executor(int num_workers);
void cleanup_concurrent_executor(void);
int submit_concurrent_task(gpu_task_t *task);
int select_gpu_for_task(gpu_task_t *task);
void* worker_thread_func(void *arg);
int init_load_balancer(int num_gpus);
void update_gpu_load(int gpu_id, int load_delta);
int get_least_loaded_gpu(void);
void balance_gpu_loads(void);

// 初始化并发执行器
int init_concurrent_executor(int num_workers) {
    global_executor = malloc(sizeof(concurrent_executor_t));
    if (!global_executor) {
        return -ENOMEM;
    }
    
    memset(global_executor, 0, sizeof(concurrent_executor_t));
    global_executor->num_workers = num_workers;
    
    // 初始化任务队列
    INIT_LIST_HEAD(&global_executor->task_queue);
    pthread_mutex_init(&global_executor->queue_mutex, NULL);
    pthread_cond_init(&global_executor->queue_cond, NULL);
    
    // 初始化原子计数器
    atomic_set(&global_executor->active_tasks, 0);
    atomic_set(&global_executor->completed_tasks, 0);
    
    // 创建工作线程
    global_executor->worker_threads = malloc(num_workers * sizeof(pthread_t));
    if (!global_executor->worker_threads) {
        free(global_executor);
        return -ENOMEM;
    }
    
    for (int i = 0; i < num_workers; i++) {
        if (pthread_create(&global_executor->worker_threads[i], NULL, 
                          worker_thread_func, NULL) != 0) {
            // 清理已创建的线程
            global_executor->shutdown = 1;
            for (int j = 0; j < i; j++) {
                pthread_cancel(global_executor->worker_threads[j]);
            }
            free(global_executor->worker_threads);
            free(global_executor);
            return -1;
        }
    }
    
    return 0;
}

// 清理并发执行器
void cleanup_concurrent_executor(void) {
    if (!global_executor) return;
    
    // 设置关闭标志
    global_executor->shutdown = 1;
    
    // 唤醒所有工作线程
    pthread_cond_broadcast(&global_executor->queue_cond);
    
    // 等待所有工作线程结束
    for (int i = 0; i < global_executor->num_workers; i++) {
        pthread_join(global_executor->worker_threads[i], NULL);
    }
    
    // 清理资源
    pthread_mutex_destroy(&global_executor->queue_mutex);
    pthread_cond_destroy(&global_executor->queue_cond);
    free(global_executor->worker_threads);
    free(global_executor);
    global_executor = NULL;
}

// 提交并发任务
int submit_concurrent_task(gpu_task_t *task) {
    if (!global_executor || !task) {
        return -EINVAL;
    }
    
    // 选择最适合的GPU
    int selected_gpu = select_gpu_for_task(task);
    if (selected_gpu < 0) {
        return -ENODEV;
    }
    
    task->device_id = selected_gpu;
    task->state = TASK_READY;
    
    // 加入任务队列
    pthread_mutex_lock(&global_executor->queue_mutex);
    list_add_tail(&task->list, &global_executor->task_queue);
    atomic_inc(&global_executor->active_tasks);
    pthread_mutex_unlock(&global_executor->queue_mutex);
    
    // 通知工作线程
    pthread_cond_signal(&global_executor->queue_cond);
    
    return 0;
}

// 为任务选择GPU
int select_gpu_for_task(gpu_task_t *task) {
    if (!global_balancer || !task) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&global_balancer->balance_mutex);
    
    int best_gpu = -1;
    int min_load = INT_MAX;
    
    // 遍历所有GPU，选择负载最小且满足内存要求的
    for (int i = 0; i < global_balancer->num_gpus; i++) {
        // 检查内存是否足够
        if (global_balancer->gpu_memory_usage[i] + task->memory_required > 
            get_gpu_memory_limit(i)) {
            continue;
        }
        
        // 选择负载最小的GPU
        if (global_balancer->gpu_loads[i] < min_load) {
            min_load = global_balancer->gpu_loads[i];
            best_gpu = i;
        }
    }
    
    // 更新选中GPU的负载
    if (best_gpu >= 0) {
        update_gpu_load(best_gpu, 1);
        global_balancer->gpu_memory_usage[best_gpu] += task->memory_required;
    }
    
    pthread_mutex_unlock(&global_balancer->balance_mutex);
    
    return best_gpu;
}

// 工作线程函数
void* worker_thread_func(void *arg) {
    gpu_task_t *task;
    
    while (!global_executor->shutdown) {
        pthread_mutex_lock(&global_executor->queue_mutex);
        
        // 等待任务
        while (list_empty(&global_executor->task_queue) && 
               !global_executor->shutdown) {
            pthread_cond_wait(&global_executor->queue_cond, 
                            &global_executor->queue_mutex);
        }
        
        if (global_executor->shutdown) {
            pthread_mutex_unlock(&global_executor->queue_mutex);
            break;
        }
        
        // 获取任务
        task = list_first_entry(&global_executor->task_queue, 
                               gpu_task_t, list);
        list_del(&task->list);
        pthread_mutex_unlock(&global_executor->queue_mutex);
        
        // 执行任务
        task->state = TASK_RUNNING;
        task->start_time = get_timestamp_us();
        
        int result = execute_gpu_task(task);
        
        // 更新任务状态
        if (result == 0) {
            task->state = TASK_COMPLETED;
            atomic_inc(&global_executor->completed_tasks);
        } else {
            task->state = TASK_FAILED;
        }
        
        // 更新GPU负载
        update_gpu_load(task->device_id, -1);
        global_balancer->gpu_memory_usage[task->device_id] -= task->memory_required;
        
        atomic_dec(&global_executor->active_tasks);
    }
    
    return NULL;
}

// 初始化负载均衡器
int init_load_balancer(int num_gpus) {
    global_balancer = malloc(sizeof(load_balancer_t));
    if (!global_balancer) {
        return -ENOMEM;
    }
    
    memset(global_balancer, 0, sizeof(load_balancer_t));
    global_balancer->num_gpus = num_gpus;
    
    // 分配GPU负载数组
    global_balancer->gpu_loads = calloc(num_gpus, sizeof(int));
    global_balancer->gpu_memory_usage = calloc(num_gpus, sizeof(int));
    
    if (!global_balancer->gpu_loads || !global_balancer->gpu_memory_usage) {
        free(global_balancer->gpu_loads);
        free(global_balancer->gpu_memory_usage);
        free(global_balancer);
        return -ENOMEM;
    }
    
    pthread_mutex_init(&global_balancer->balance_mutex, NULL);
    gettimeofday(&global_balancer->last_balance_time, NULL);
    
    return 0;
}

// 更新GPU负载
void update_gpu_load(int gpu_id, int load_delta) {
    if (!global_balancer || gpu_id < 0 || gpu_id >= global_balancer->num_gpus) {
        return;
    }
    
    global_balancer->gpu_loads[gpu_id] += load_delta;
    
    // 确保负载不为负数
    if (global_balancer->gpu_loads[gpu_id] < 0) {
        global_balancer->gpu_loads[gpu_id] = 0;
    }
}

// 获取负载最小的GPU
int get_least_loaded_gpu(void) {
    if (!global_balancer) {
        return -1;
    }
    
    int min_load = INT_MAX;
    int best_gpu = -1;
    
    for (int i = 0; i < global_balancer->num_gpus; i++) {
        if (global_balancer->gpu_loads[i] < min_load) {
            min_load = global_balancer->gpu_loads[i];
            best_gpu = i;
        }
    }
    
    return best_gpu;
}

// 负载均衡
void balance_gpu_loads(void) {
    if (!global_balancer) return;
    
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    
    // 检查是否需要重新平衡（每秒最多一次）
    if (current_time.tv_sec - global_balancer->last_balance_time.tv_sec < 1) {
        return;
    }
    
    pthread_mutex_lock(&global_balancer->balance_mutex);
    
    // 计算平均负载
    int total_load = 0;
    for (int i = 0; i < global_balancer->num_gpus; i++) {
        total_load += global_balancer->gpu_loads[i];
    }
    
    int avg_load = total_load / global_balancer->num_gpus;
    
    // 记录重新平衡时间
    global_balancer->last_balance_time = current_time;
    
    pthread_mutex_unlock(&global_balancer->balance_mutex);
    
    // 这里可以实现更复杂的负载均衡策略
    // 比如任务迁移、优先级调整等
}

// 获取执行器统计信息
void get_executor_stats(int *active_tasks, int *completed_tasks) {
    if (!global_executor) {
        *active_tasks = 0;
        *completed_tasks = 0;
        return;
    }
    
    *active_tasks = atomic_read(&global_executor->active_tasks);
    *completed_tasks = atomic_read(&global_executor->completed_tasks);
}

// 获取负载均衡器统计信息
void get_balancer_stats(int gpu_id, int *load, int *memory_usage) {
    if (!global_balancer || gpu_id < 0 || gpu_id >= global_balancer->num_gpus) {
        *load = 0;
        *memory_usage = 0;
        return;
    }
    
    pthread_mutex_lock(&global_balancer->balance_mutex);
    *load = global_balancer->gpu_loads[gpu_id];
    *memory_usage = global_balancer->gpu_memory_usage[gpu_id];
    pthread_mutex_unlock(&global_balancer->balance_mutex);
}

// 辅助函数（需要在其他地方实现）
extern unsigned long get_timestamp_us(void);
extern int execute_gpu_task(gpu_task_t *task);
extern size_t get_gpu_memory_limit(int gpu_id);