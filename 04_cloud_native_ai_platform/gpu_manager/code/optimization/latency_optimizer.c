#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>

#include "latency_optimizer.h"
#include "../common/gpu_common.h"

// 延迟优化器全局实例
static latency_optimizer_t g_optimizer = {0};
static bool g_optimizer_initialized = false;
static pthread_mutex_t g_optimizer_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_batch_ready_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_batch_processor_thread;
static bool g_processor_running = false;

// 请求队列管理
static request_queue_t g_request_queues[MAX_GPU_DEVICES];
static pthread_mutex_t g_queue_mutexes[MAX_GPU_DEVICES];

// 创建GPU请求
static gpu_request_t* create_gpu_request(const request_params_t *params, 
                                        request_callback_t callback, 
                                        void *user_data) {
    gpu_request_t *request = malloc(sizeof(gpu_request_t));
    if (!request) {
        return NULL;
    }
    
    memset(request, 0, sizeof(gpu_request_t));
    
    // 生成唯一ID
    static uint64_t request_id_counter = 0;
    request->request_id = __sync_add_and_fetch(&request_id_counter, 1);
    
    request->params = *params;
    request->callback = callback;
    request->user_data = user_data;
    request->submit_time = gpu_get_timestamp_us();
    request->status = REQUEST_STATUS_PENDING;
    request->priority = params->priority;
    request->estimated_duration_us = 0; // 将由预测器估算
    request->next = NULL;
    
    return request;
}

// 释放GPU请求
static void free_gpu_request(gpu_request_t *request) {
    if (request) {
        free(request);
    }
}

// 估算请求执行时间
static uint64_t estimate_request_duration(const request_params_t *params) {
    // 基于操作类型和数据大小的简单估算模型
    uint64_t base_duration = 1000; // 1ms基础时间
    
    // 根据操作类型调整
    switch (params->operation_type) {
        case GPU_OP_COMPUTE:
            base_duration *= 10;
            break;
        case GPU_OP_MEMORY_COPY:
            base_duration *= 2;
            break;
        case GPU_OP_KERNEL_LAUNCH:
            base_duration *= 5;
            break;
        case GPU_OP_SYNCHRONIZE:
            base_duration *= 1;
            break;
        default:
            break;
    }
    
    // 根据数据大小调整
    if (params->data_size > 0) {
        // 假设每MB数据增加100us
        uint64_t size_factor = (params->data_size / (1024 * 1024)) * 100;
        base_duration += size_factor;
    }
    
    return base_duration;
}

// 请求优先级比较函数
static int compare_request_priority(const gpu_request_t *a, const gpu_request_t *b) {
    // 优先级高的排在前面
    if (a->priority != b->priority) {
        return b->priority - a->priority;
    }
    
    // 优先级相同时，按提交时间排序
    if (a->submit_time < b->submit_time) {
        return -1;
    } else if (a->submit_time > b->submit_time) {
        return 1;
    }
    
    return 0;
}

// 向队列中插入请求（按优先级排序）
static void insert_request_sorted(request_queue_t *queue, gpu_request_t *request) {
    if (!queue->head || compare_request_priority(request, queue->head) < 0) {
        // 插入到队列头部
        request->next = queue->head;
        queue->head = request;
        if (!queue->tail) {
            queue->tail = request;
        }
    } else {
        // 找到合适的插入位置
        gpu_request_t *current = queue->head;
        while (current->next && compare_request_priority(request, current->next) >= 0) {
            current = current->next;
        }
        
        request->next = current->next;
        current->next = request;
        
        if (!request->next) {
            queue->tail = request;
        }
    }
    
    queue->count++;
}

// 从队列中移除请求
static gpu_request_t* dequeue_request(request_queue_t *queue) {
    if (!queue->head) {
        return NULL;
    }
    
    gpu_request_t *request = queue->head;
    queue->head = request->next;
    
    if (!queue->head) {
        queue->tail = NULL;
    }
    
    queue->count--;
    request->next = NULL;
    
    return request;
}

// 创建请求批次
optimizer_result_t create_request_batch(int device_id,
                                       gpu_request_t **requests,
                                       uint32_t request_count,
                                       request_batch_t **batch) {
    if (!batch || !requests) {
        return OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    request_batch_t *new_batch = malloc(sizeof(request_batch_t));
    if (!new_batch) {
        return OPTIMIZER_ERROR_OUT_OF_MEMORY;
    }
    
    memset(new_batch, 0, sizeof(request_batch_t));
    
    static uint64_t batch_id_counter = 0;
    new_batch->batch_id = __sync_add_and_fetch(&batch_id_counter, 1);
    new_batch->device_id = device_id;
    new_batch->created_time = gpu_get_timestamp_us();
    
    // 复制请求到批次中
    if (request_count > 0 && request_count <= MAX_BATCH_SIZE) {
        for (uint32_t i = 0; i < request_count; i++) {
            if (requests[i]) {
                new_batch->requests[new_batch->request_count++] = requests[i];
                new_batch->total_estimated_duration_us += requests[i]->estimated_duration_us;
            }
        }
    }
    
    *batch = new_batch;
    return OPTIMIZER_SUCCESS;
}

// 内部辅助函数：创建空批次
static request_batch_t* create_empty_batch(int device_id) {
    request_batch_t *batch = malloc(sizeof(request_batch_t));
    if (!batch) {
        return NULL;
    }
    
    memset(batch, 0, sizeof(request_batch_t));
    
    static uint64_t batch_id_counter = 0;
    batch->batch_id = __sync_add_and_fetch(&batch_id_counter, 1);
    batch->device_id = device_id;
    batch->created_time = gpu_get_timestamp_us();
    
    return batch;
}

// 释放请求批次
static void free_request_batch(request_batch_t *batch) {
    if (batch) {
        for (int i = 0; i < batch->request_count; i++) {
            free_gpu_request(batch->requests[i]);
        }
        free(batch);
    }
}

// 检查是否可以创建批次
static bool should_create_batch(int device_id) {
    request_queue_t *queue = &g_request_queues[device_id];
    
    if (queue->count == 0) {
        return false;
    }
    
    // 检查批次大小条件
    if (queue->count >= g_optimizer.config.max_batch_size) {
        return true;
    }
    
    // 检查超时条件
    if (queue->head) {
        uint64_t now = gpu_get_timestamp_us();
        uint64_t wait_time = now - queue->head->submit_time;
        if (wait_time >= g_optimizer.config.batch_timeout_us) {
            return true;
        }
    }
    
    return false;
}

// 创建并填充批次
static request_batch_t* build_request_batch(int device_id) {
    request_queue_t *queue = &g_request_queues[device_id];
    
    if (queue->count == 0) {
        return NULL;
    }
    
    request_batch_t *batch = create_empty_batch(device_id);
    if (!batch) {
        return NULL;
    }
    
    // 从队列中取出请求填充批次
    while (batch->request_count < MAX_BATCH_SIZE && queue->count > 0) {
        gpu_request_t *request = dequeue_request(queue);
        if (request) {
            batch->requests[batch->request_count++] = request;
            batch->total_estimated_duration_us += request->estimated_duration_us;
        }
    }
    
    return batch;
}

// 执行请求批次
optimizer_result_t execute_request_batch(request_batch_t *batch) {
    if (!batch || batch->request_count == 0) {
        return OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    gpu_log(GPU_LOG_DEBUG, "Executing batch %lu with %d requests on device %d", 
            batch->batch_id, batch->request_count, batch->device_id);
    
    batch->start_time = gpu_get_timestamp_us();
    
    // 模拟批次执行
    // 在实际实现中，这里会调用GPU驱动API执行批次
    for (int i = 0; i < batch->request_count; i++) {
        gpu_request_t *request = batch->requests[i];
        request->status = REQUEST_STATUS_EXECUTING;
        
        // 模拟执行时间
        usleep(request->estimated_duration_us);
        
        request->status = REQUEST_STATUS_COMPLETED;
        request->completion_time = gpu_get_timestamp_us();
        
        // 调用回调函数
        if (request->callback) {
            request->callback(request->request_id, REQUEST_STATUS_COMPLETED, 
                            request->user_data);
        }
    }
    
    batch->completion_time = gpu_get_timestamp_us();
    batch->actual_duration_us = batch->completion_time - batch->start_time;
    
    // 更新统计信息
    pthread_mutex_lock(&g_optimizer_mutex);
    g_optimizer.stats.total_batches_executed++;
    g_optimizer.stats.total_requests_processed += batch->request_count;
    g_optimizer.stats.total_execution_time_us += batch->actual_duration_us;
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    gpu_log(GPU_LOG_DEBUG, "Batch %lu completed in %lu us", 
            batch->batch_id, batch->actual_duration_us);
    
    return OPTIMIZER_SUCCESS;
}

// 批处理器线程主函数
static void* batch_processor_thread_func(void *arg) {
    gpu_log(GPU_LOG_INFO, "Batch processor thread started");
    
    while (g_processor_running) {
        bool batch_processed = false;
        
        // 检查所有设备的请求队列
        for (int device_id = 0; device_id < MAX_GPU_DEVICES; device_id++) {
            pthread_mutex_lock(&g_queue_mutexes[device_id]);
            
            if (should_create_batch(device_id)) {
                request_batch_t *batch = build_request_batch(device_id);
                
                pthread_mutex_unlock(&g_queue_mutexes[device_id]);
                
                if (batch) {
                    execute_request_batch(batch);
                    free_request_batch(batch);
                    batch_processed = true;
                }
            } else {
                pthread_mutex_unlock(&g_queue_mutexes[device_id]);
            }
        }
        
        if (!batch_processed) {
            // 没有批次需要处理，等待一段时间
            usleep(g_optimizer.config.check_interval_us);
        }
    }
    
    gpu_log(GPU_LOG_INFO, "Batch processor thread stopped");
    return NULL;
}

// 初始化延迟优化器
optimizer_result_t latency_optimizer_init(const optimizer_config_t *config) {
    if (!config) {
        return OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    if (g_optimizer_initialized) {
        gpu_log(GPU_LOG_WARNING, "Latency optimizer already initialized");
        return OPTIMIZER_SUCCESS;
    }
    
    memset(&g_optimizer, 0, sizeof(g_optimizer));
    g_optimizer.config = *config;
    
    // 验证配置参数
    if (g_optimizer.config.max_batch_size == 0) {
        g_optimizer.config.max_batch_size = 32; // 默认批次大小
    }
    if (g_optimizer.config.batch_timeout_us == 0) {
        g_optimizer.config.batch_timeout_us = 10000; // 默认10ms超时
    }
    if (g_optimizer.config.check_interval_us == 0) {
        g_optimizer.config.check_interval_us = 1000; // 默认1ms检查间隔
    }
    
    // 初始化请求队列
    for (int i = 0; i < MAX_GPU_DEVICES; i++) {
        memset(&g_request_queues[i], 0, sizeof(request_queue_t));
        if (pthread_mutex_init(&g_queue_mutexes[i], NULL) != 0) {
            gpu_log(GPU_LOG_ERROR, "Failed to initialize queue mutex for device %d", i);
            return OPTIMIZER_ERROR_INITIALIZATION_FAILED;
        }
    }
    
    // 启动批处理器线程
    g_processor_running = true;
    if (pthread_create(&g_batch_processor_thread, NULL, batch_processor_thread_func, NULL) != 0) {
        gpu_log(GPU_LOG_ERROR, "Failed to create batch processor thread: %s", strerror(errno));
        g_processor_running = false;
        return OPTIMIZER_ERROR_THREAD_CREATE_FAILED;
    }
    
    g_optimizer_initialized = true;
    
    gpu_log(GPU_LOG_INFO, "Latency optimizer initialized successfully");
    return OPTIMIZER_SUCCESS;
}

// 清理延迟优化器
void latency_optimizer_cleanup(void) {
    if (!g_optimizer_initialized) {
        return;
    }
    
    gpu_log(GPU_LOG_INFO, "Stopping latency optimizer...");
    
    // 停止批处理器线程
    g_processor_running = false;
    if (pthread_join(g_batch_processor_thread, NULL) != 0) {
        gpu_log(GPU_LOG_WARNING, "Failed to join batch processor thread");
    }
    
    // 清理请求队列
    for (int i = 0; i < MAX_GPU_DEVICES; i++) {
        pthread_mutex_lock(&g_queue_mutexes[i]);
        
        // 清理队列中的请求
        while (g_request_queues[i].head) {
            gpu_request_t *request = dequeue_request(&g_request_queues[i]);
            if (request && request->callback) {
                request->callback(request->request_id, REQUEST_STATUS_CANCELLED, 
                                request->user_data);
            }
            free_gpu_request(request);
        }
        
        pthread_mutex_unlock(&g_queue_mutexes[i]);
        pthread_mutex_destroy(&g_queue_mutexes[i]);
    }
    
    memset(&g_optimizer, 0, sizeof(g_optimizer));
    g_optimizer_initialized = false;
    
    gpu_log(GPU_LOG_INFO, "Latency optimizer stopped");
}

// 提交GPU请求
optimizer_result_t latency_optimizer_submit_request(const request_params_t *params,
                                                   request_callback_t callback,
                                                   void *user_data,
                                                   uint64_t *request_id) {
    if (!g_optimizer_initialized || !params || !request_id) {
        return OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    if (params->device_id < 0 || params->device_id >= MAX_GPU_DEVICES) {
        return OPTIMIZER_ERROR_INVALID_DEVICE;
    }
    
    gpu_request_t *request = create_gpu_request(params, callback, user_data);
    if (!request) {
        return OPTIMIZER_ERROR_OUT_OF_MEMORY;
    }
    
    // 估算执行时间
    request->estimated_duration_us = estimate_request_duration(params);
    
    *request_id = request->request_id;
    
    // 将请求添加到对应设备的队列
    pthread_mutex_lock(&g_queue_mutexes[params->device_id]);
    insert_request_sorted(&g_request_queues[params->device_id], request);
    pthread_mutex_unlock(&g_queue_mutexes[params->device_id]);
    
    // 更新统计信息
    pthread_mutex_lock(&g_optimizer_mutex);
    g_optimizer.stats.total_requests_submitted++;
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    gpu_log(GPU_LOG_DEBUG, "Request %lu submitted to device %d", 
            request->request_id, params->device_id);
    
    return OPTIMIZER_SUCCESS;
}

// 取消GPU请求
optimizer_result_t latency_optimizer_cancel_request(uint64_t request_id) {
    if (!g_optimizer_initialized) {
        return OPTIMIZER_ERROR_NOT_INITIALIZED;
    }
    
    // 在所有设备队列中查找并移除请求
    for (int device_id = 0; device_id < MAX_GPU_DEVICES; device_id++) {
        pthread_mutex_lock(&g_queue_mutexes[device_id]);
        
        request_queue_t *queue = &g_request_queues[device_id];
        gpu_request_t *prev = NULL;
        gpu_request_t *current = queue->head;
        
        while (current) {
            if (current->request_id == request_id) {
                // 找到请求，移除它
                if (prev) {
                    prev->next = current->next;
                } else {
                    queue->head = current->next;
                }
                
                if (current == queue->tail) {
                    queue->tail = prev;
                }
                
                queue->count--;
                
                // 调用回调通知取消
                if (current->callback) {
                    current->callback(request_id, REQUEST_STATUS_CANCELLED, 
                                    current->user_data);
                }
                
                free_gpu_request(current);
                
                pthread_mutex_unlock(&g_queue_mutexes[device_id]);
                
                gpu_log(GPU_LOG_DEBUG, "Request %lu cancelled", request_id);
                return OPTIMIZER_SUCCESS;
            }
            
            prev = current;
            current = current->next;
        }
        
        pthread_mutex_unlock(&g_queue_mutexes[device_id]);
    }
    
    return OPTIMIZER_ERROR_REQUEST_NOT_FOUND;
}

// 获取优化器统计信息
optimizer_result_t latency_optimizer_get_stats(optimizer_stats_t *stats) {
    if (!g_optimizer_initialized || !stats) {
        return OPTIMIZER_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_optimizer_mutex);
    *stats = g_optimizer.stats;
    
    // 计算平均值
    if (stats->total_batches_executed > 0) {
        stats->average_batch_size = (double)stats->total_requests_processed / 
                                   stats->total_batches_executed;
        stats->average_execution_time_us = stats->total_execution_time_us / 
                                          stats->total_batches_executed;
    }
    
    pthread_mutex_unlock(&g_optimizer_mutex);
    
    return OPTIMIZER_SUCCESS;
}

// 获取错误字符串
const char* optimizer_get_error_string(optimizer_result_t error) {
    static const char* error_strings[] = {
        [OPTIMIZER_SUCCESS] = "Success",
        [OPTIMIZER_ERROR_INVALID_PARAM] = "Invalid parameter",
        [OPTIMIZER_ERROR_NOT_INITIALIZED] = "Optimizer not initialized",
        [OPTIMIZER_ERROR_OUT_OF_MEMORY] = "Out of memory",
        [OPTIMIZER_ERROR_INVALID_DEVICE] = "Invalid device",
        [OPTIMIZER_ERROR_REQUEST_NOT_FOUND] = "Request not found",
        [OPTIMIZER_ERROR_INITIALIZATION_FAILED] = "Initialization failed",
        [OPTIMIZER_ERROR_THREAD_CREATE_FAILED] = "Thread creation failed"
    };
    
    if (error >= 0 && error < sizeof(error_strings) / sizeof(error_strings[0])) {
        return error_strings[error];
    }
    return "Unknown error";
}