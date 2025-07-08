/**
 * 多级队列调度器
 * 演示GPU任务的多级队列调度、优先级管理和负载均衡
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include "../common/gpu_common.h"

#define MAX_PRIORITY_LEVELS 4
#define MAX_QUEUE_SIZE 100
#define MAX_DEVICES 8

// 任务优先级定义
typedef enum
{
    PRIORITY_CRITICAL = 0, // 关键任务
    PRIORITY_HIGH = 1,     // 高优先级
    PRIORITY_NORMAL = 2,   // 普通优先级
    PRIORITY_LOW = 3       // 低优先级
} task_priority_t;

// 任务状态
typedef enum
{
    TASK_PENDING,
    TASK_RUNNING,
    TASK_COMPLETED,
    TASK_FAILED
} task_status_t;

// GPU任务结构
typedef struct
{
    int task_id;
    task_priority_t priority;
    task_status_t status;
    int device_id;
    size_t memory_required;
    int compute_units_required;
    uint64_t submit_time;
    uint64_t start_time;
    uint64_t end_time;
    void (*task_func)(void *args);
    void *args;
} gpu_task_t;

// 任务队列
typedef struct
{
    gpu_task_t *tasks[MAX_QUEUE_SIZE];
    int front;
    int rear;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} task_queue_t;

// 设备状态
typedef struct
{
    int device_id;
    int is_available;
    size_t total_memory;
    size_t available_memory;
    int total_compute_units;
    int available_compute_units;
    gpu_task_t *current_task;
    pthread_mutex_t mutex;
} device_state_t;

// 调度器结构
typedef struct
{
    task_queue_t priority_queues[MAX_PRIORITY_LEVELS];
    device_state_t devices[MAX_DEVICES];
    int device_count;
    pthread_t scheduler_thread;
    int running;
    pthread_mutex_t scheduler_mutex;
} multi_level_scheduler_t;

// 全局调度器实例
static multi_level_scheduler_t g_scheduler;

/**
 * 获取当前时间戳（微秒）
 */
static uint64_t get_timestamp_us()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

/**
 * 初始化任务队列
 */
static int init_task_queue(task_queue_t *queue)
{
    queue->front = 0;
    queue->rear = 0;
    queue->count = 0;

    if (pthread_mutex_init(&queue->mutex, NULL) != 0)
    {
        return -1;
    }

    if (pthread_cond_init(&queue->not_empty, NULL) != 0)
    {
        pthread_mutex_destroy(&queue->mutex);
        return -1;
    }

    if (pthread_cond_init(&queue->not_full, NULL) != 0)
    {
        pthread_mutex_destroy(&queue->mutex);
        pthread_cond_destroy(&queue->not_empty);
        return -1;
    }

    return 0;
}

/**
 * 向队列添加任务
 */
static int enqueue_task(task_queue_t *queue, gpu_task_t *task)
{
    pthread_mutex_lock(&queue->mutex);

    while (queue->count >= MAX_QUEUE_SIZE)
    {
        pthread_cond_wait(&queue->not_full, &queue->mutex);
    }

    queue->tasks[queue->rear] = task;
    queue->rear = (queue->rear + 1) % MAX_QUEUE_SIZE;
    queue->count++;

    pthread_cond_signal(&queue->not_empty);
    pthread_mutex_unlock(&queue->mutex);

    return 0;
}

/**
 * 从队列取出任务
 */
static gpu_task_t *dequeue_task(task_queue_t *queue)
{
    pthread_mutex_lock(&queue->mutex);

    while (queue->count == 0)
    {
        pthread_cond_wait(&queue->not_empty, &queue->mutex);
    }

    gpu_task_t *task = queue->tasks[queue->front];
    queue->front = (queue->front + 1) % MAX_QUEUE_SIZE;
    queue->count--;

    pthread_cond_signal(&queue->not_full);
    pthread_mutex_unlock(&queue->mutex);

    return task;
}

/**
 * 初始化设备状态
 */
static int init_device_state(device_state_t *device, int device_id)
{
    device->device_id = device_id;
    device->is_available = 1;
    device->current_task = NULL;

    // 查询设备属性
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess)
    {
        printf("Failed to get device %d properties: %s\n",
               device_id, cudaGetErrorString(err));
        return -1;
    }

    device->total_memory = prop.totalGlobalMem;
    device->available_memory = prop.totalGlobalMem;
    device->total_compute_units = prop.multiProcessorCount;
    device->available_compute_units = prop.multiProcessorCount;

    if (pthread_mutex_init(&device->mutex, NULL) != 0)
    {
        return -1;
    }

    printf("Device %d initialized: %s, Memory: %zu MB, SMs: %d\n",
           device_id, prop.name, device->total_memory / (1024 * 1024),
           device->total_compute_units);

    return 0;
}

/**
 * 选择最佳设备
 */
static int select_best_device(gpu_task_t *task)
{
    int best_device = -1;
    double best_score = -1.0;

    for (int i = 0; i < g_scheduler.device_count; i++)
    {
        device_state_t *device = &g_scheduler.devices[i];

        pthread_mutex_lock(&device->mutex);

        // 检查设备是否可用且资源充足
        if (device->is_available &&
            device->available_memory >= task->memory_required &&
            device->available_compute_units >= task->compute_units_required)
        {

            // 计算设备适合度分数
            double memory_ratio = (double)device->available_memory / device->total_memory;
            double compute_ratio = (double)device->available_compute_units / device->total_compute_units;
            double score = memory_ratio * 0.6 + compute_ratio * 0.4;

            if (score > best_score)
            {
                best_score = score;
                best_device = i;
            }
        }

        pthread_mutex_unlock(&device->mutex);
    }

    return best_device;
}

/**
 * 执行任务
 */
static void execute_task(gpu_task_t *task, int device_id)
{
    device_state_t *device = &g_scheduler.devices[device_id];

    pthread_mutex_lock(&device->mutex);

    // 分配资源
    device->available_memory -= task->memory_required;
    device->available_compute_units -= task->compute_units_required;
    device->current_task = task;

    pthread_mutex_unlock(&device->mutex);

    // 设置CUDA设备
    cudaSetDevice(device_id);

    // 记录开始时间
    task->start_time = get_timestamp_us();
    task->status = TASK_RUNNING;

    printf("Task %d started on device %d (Priority: %d)\n",
           task->task_id, device_id, task->priority);

    // 执行任务函数
    if (task->task_func)
    {
        task->task_func(task->args);
    }

    // 记录结束时间
    task->end_time = get_timestamp_us();
    task->status = TASK_COMPLETED;

    pthread_mutex_lock(&device->mutex);

    // 释放资源
    device->available_memory += task->memory_required;
    device->available_compute_units += task->compute_units_required;
    device->current_task = NULL;

    pthread_mutex_unlock(&device->mutex);

    printf("Task %d completed on device %d (Duration: %lu us)\n",
           task->task_id, device_id, task->end_time - task->start_time);
}

/**
 * 调度器主循环
 */
static void *scheduler_main_loop(void *arg)
{
    while (g_scheduler.running)
    {
        gpu_task_t *task = NULL;

        // 按优先级顺序检查队列
        for (int priority = 0; priority < MAX_PRIORITY_LEVELS; priority++)
        {
            task_queue_t *queue = &g_scheduler.priority_queues[priority];

            pthread_mutex_lock(&queue->mutex);
            if (queue->count > 0)
            {
                task = queue->tasks[queue->front];
                queue->front = (queue->front + 1) % MAX_QUEUE_SIZE;
                queue->count--;
                pthread_cond_signal(&queue->not_full);
            }
            pthread_mutex_unlock(&queue->mutex);

            if (task)
                break;
        }

        if (task)
        {
            // 选择最佳设备
            int device_id = select_best_device(task);

            if (device_id >= 0)
            {
                // 执行任务
                execute_task(task, device_id);
            }
            else
            {
                // 没有可用设备，重新入队
                enqueue_task(&g_scheduler.priority_queues[task->priority], task);
                usleep(1000); // 等待1ms
            }
        }
        else
        {
            usleep(10000); // 等待10ms
        }
    }

    return NULL;
}

/**
 * 初始化调度器
 */
int init_scheduler()
{
    memset(&g_scheduler, 0, sizeof(g_scheduler));

    // 初始化优先级队列
    for (int i = 0; i < MAX_PRIORITY_LEVELS; i++)
    {
        if (init_task_queue(&g_scheduler.priority_queues[i]) != 0)
        {
            printf("Failed to initialize priority queue %d\n", i);
            return -1;
        }
    }

    // 获取设备数量
    cudaGetDeviceCount(&g_scheduler.device_count);
    if (g_scheduler.device_count > MAX_DEVICES)
    {
        g_scheduler.device_count = MAX_DEVICES;
    }

    // 初始化设备状态
    for (int i = 0; i < g_scheduler.device_count; i++)
    {
        if (init_device_state(&g_scheduler.devices[i], i) != 0)
        {
            printf("Failed to initialize device %d\n", i);
            return -1;
        }
    }

    // 初始化调度器互斥锁
    if (pthread_mutex_init(&g_scheduler.scheduler_mutex, NULL) != 0)
    {
        return -1;
    }

    // 启动调度器线程
    g_scheduler.running = 1;
    if (pthread_create(&g_scheduler.scheduler_thread, NULL, scheduler_main_loop, NULL) != 0)
    {
        printf("Failed to create scheduler thread\n");
        return -1;
    }

    printf("Multi-level scheduler initialized with %d devices\n", g_scheduler.device_count);
    return 0;
}

/**
 * 提交任务
 */
int submit_task(int task_id, task_priority_t priority, size_t memory_required,
                int compute_units_required, void (*task_func)(void *), void *args)
{

    gpu_task_t *task = malloc(sizeof(gpu_task_t));
    if (!task)
    {
        return -1;
    }

    task->task_id = task_id;
    task->priority = priority;
    task->status = TASK_PENDING;
    task->device_id = -1;
    task->memory_required = memory_required;
    task->compute_units_required = compute_units_required;
    task->submit_time = get_timestamp_us();
    task->start_time = 0;
    task->end_time = 0;
    task->task_func = task_func;
    task->args = args;

    // 添加到对应优先级队列
    enqueue_task(&g_scheduler.priority_queues[priority], task);

    printf("Task %d submitted with priority %d\n", task_id, priority);
    return 0;
}

/**
 * 获取调度器统计信息
 */
void get_scheduler_stats()
{
    printf("\n=== Scheduler Statistics ===\n");

    // 队列统计
    for (int i = 0; i < MAX_PRIORITY_LEVELS; i++)
    {
        task_queue_t *queue = &g_scheduler.priority_queues[i];
        pthread_mutex_lock(&queue->mutex);
        printf("Priority %d queue: %d tasks\n", i, queue->count);
        pthread_mutex_unlock(&queue->mutex);
    }

    // 设备统计
    for (int i = 0; i < g_scheduler.device_count; i++)
    {
        device_state_t *device = &g_scheduler.devices[i];
        pthread_mutex_lock(&device->mutex);
        printf("Device %d: Available Memory: %zu MB, Available SMs: %d\n",
               i, device->available_memory / (1024 * 1024), device->available_compute_units);
        if (device->current_task)
        {
            printf("  Running task: %d\n", device->current_task->task_id);
        }
        pthread_mutex_unlock(&device->mutex);
    }

    printf("============================\n\n");
}

/**
 * 示例任务函数
 */
void sample_task_function(void *args)
{
    int *duration = (int *)args;

    // 模拟GPU计算
    usleep(*duration * 1000); // 转换为微秒

    // 这里可以添加实际的CUDA kernel调用
    // 例如：sample_kernel<<<blocks, threads>>>();
    // cudaDeviceSynchronize();
}

/**
 * 清理调度器
 */
void cleanup_scheduler()
{
    g_scheduler.running = 0;

    // 等待调度器线程结束
    pthread_join(g_scheduler.scheduler_thread, NULL);

    // 清理队列
    for (int i = 0; i < MAX_PRIORITY_LEVELS; i++)
    {
        task_queue_t *queue = &g_scheduler.priority_queues[i];
        pthread_mutex_destroy(&queue->mutex);
        pthread_cond_destroy(&queue->not_empty);
        pthread_cond_destroy(&queue->not_full);
    }

    // 清理设备
    for (int i = 0; i < g_scheduler.device_count; i++)
    {
        pthread_mutex_destroy(&g_scheduler.devices[i].mutex);
    }

    pthread_mutex_destroy(&g_scheduler.scheduler_mutex);

    printf("Scheduler cleaned up\n");
}

/**
 * 示例主函数
 */
int main()
{
    // 初始化调度器
    if (init_scheduler() != 0)
    {
        printf("Failed to initialize scheduler\n");
        return -1;
    }

    // 提交不同优先级的任务
    int duration1 = 100, duration2 = 200, duration3 = 50, duration4 = 300;

    submit_task(1, PRIORITY_CRITICAL, 100 * 1024 * 1024, 1, sample_task_function, &duration1);
    submit_task(2, PRIORITY_HIGH, 200 * 1024 * 1024, 2, sample_task_function, &duration2);
    submit_task(3, PRIORITY_NORMAL, 150 * 1024 * 1024, 1, sample_task_function, &duration3);
    submit_task(4, PRIORITY_LOW, 50 * 1024 * 1024, 1, sample_task_function, &duration4);
    submit_task(5, PRIORITY_CRITICAL, 80 * 1024 * 1024, 1, sample_task_function, &duration1);

    // 运行一段时间
    for (int i = 0; i < 10; i++)
    {
        sleep(1);
        get_scheduler_stats();
    }

    // 清理
    cleanup_scheduler();

    return 0;
}