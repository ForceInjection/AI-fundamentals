/**
 * GPU内存QoS保障机制实现
 * 功能：实现内存访问的服务质量保障
 * 支持：带宽控制、延迟保障、优先级调度
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <errno.h>

// 前向声明
void cleanup_memory_qos(void);

// QoS等级定义
typedef enum {
    QOS_LEVEL_CRITICAL = 0,  // 关键级别
    QOS_LEVEL_HIGH,          // 高优先级
    QOS_LEVEL_NORMAL,        // 普通优先级
    QOS_LEVEL_LOW,           // 低优先级
    QOS_LEVEL_BACKGROUND     // 后台级别
} qos_level_t;

// 内存访问类型
typedef enum {
    ACCESS_TYPE_READ = 0,    // 读访问
    ACCESS_TYPE_WRITE,       // 写访问
    ACCESS_TYPE_ATOMIC,      // 原子操作
    ACCESS_TYPE_PREFETCH     // 预取访问
} memory_access_type_t;

// 带宽控制策略
typedef enum {
    BANDWIDTH_POLICY_FAIR = 0,     // 公平分配
    BANDWIDTH_POLICY_PRIORITY,     // 优先级分配
    BANDWIDTH_POLICY_WEIGHTED,     // 加权分配
    BANDWIDTH_POLICY_ADAPTIVE      // 自适应分配
} bandwidth_policy_t;

// QoS配置
typedef struct {
    qos_level_t level;              // QoS等级
    uint32_t max_bandwidth_mbps;    // 最大带宽（MB/s）
    uint32_t min_bandwidth_mbps;    // 最小带宽（MB/s）
    uint32_t max_latency_us;        // 最大延迟（微秒）
    uint32_t priority;              // 优先级（0-255）
    double weight;                  // 权重（0.0-1.0）
    bool preemptible;               // 是否可抢占
    bool burst_allowed;             // 是否允许突发
} qos_config_t;

// 内存访问请求
typedef struct memory_request {
    uint64_t request_id;            // 请求ID
    void *address;                  // 内存地址
    size_t size;                    // 访问大小
    memory_access_type_t type;      // 访问类型
    qos_level_t qos_level;          // QoS等级
    uint32_t client_id;             // 客户端ID
    uint64_t submit_time;           // 提交时间
    uint64_t start_time;            // 开始时间
    uint64_t complete_time;         // 完成时间
    uint32_t estimated_duration_us; // 预估持续时间
    bool is_urgent;                 // 是否紧急
    struct memory_request *next;    // 链表指针
} memory_request_t;

// 带宽令牌桶
typedef struct {
    uint32_t capacity;              // 桶容量
    uint32_t tokens;                // 当前令牌数
    uint32_t refill_rate;           // 填充速率（令牌/秒）
    uint64_t last_refill_time;      // 上次填充时间
    pthread_mutex_t bucket_lock;    // 桶锁
} token_bucket_t;

// QoS调度器
typedef struct {
    memory_request_t *queues[5];    // 按QoS等级分类的队列
    uint32_t queue_lengths[5];      // 队列长度
    pthread_mutex_t queue_locks[5]; // 队列锁
    
    token_bucket_t bandwidth_buckets[5]; // 带宽令牌桶
    qos_config_t qos_configs[5];    // QoS配置
    
    bandwidth_policy_t policy;      // 带宽策略
    uint32_t total_bandwidth_mbps;  // 总带宽
    uint32_t available_bandwidth_mbps; // 可用带宽
    
    pthread_t scheduler_thread;     // 调度线程
    pthread_cond_t scheduler_cond;  // 调度条件变量
    pthread_mutex_t scheduler_lock; // 调度锁
    
    bool shutdown;                  // 关闭标志
    
    // 统计信息
    struct {
        uint64_t total_requests;     // 总请求数
        uint64_t completed_requests; // 完成请求数
        uint64_t dropped_requests;   // 丢弃请求数
        uint64_t total_latency_us;   // 总延迟
        uint64_t total_bandwidth_used; // 总带宽使用
        uint32_t qos_violations;     // QoS违规次数
        uint32_t preemptions;        // 抢占次数
    } stats;
} qos_scheduler_t;

// 全局QoS调度器
static qos_scheduler_t g_qos_scheduler = {0};

// 函数声明
static uint64_t get_current_time_us(void);
static void init_token_bucket(token_bucket_t *bucket, uint32_t capacity, uint32_t refill_rate);
static bool consume_tokens(token_bucket_t *bucket, uint32_t tokens);
static void refill_tokens(token_bucket_t *bucket);
static memory_request_t* create_memory_request(void *address, size_t size, 
                                              memory_access_type_t type, qos_level_t qos_level);
static void free_memory_request(memory_request_t *request);
static void enqueue_request(memory_request_t *request);
static memory_request_t* dequeue_request(void);
static memory_request_t* select_next_request(void);
static void execute_memory_request(memory_request_t *request);
static void update_bandwidth_allocation(void);
static bool check_qos_violation(memory_request_t *request);
static void handle_qos_violation(memory_request_t *request);
static void* qos_scheduler_thread(void *arg);
static double calculate_priority_score(memory_request_t *request);
static void adaptive_bandwidth_adjustment(void);

// 初始化QoS系统
int init_memory_qos(uint32_t total_bandwidth_mbps, bandwidth_policy_t policy) {
    memset(&g_qos_scheduler, 0, sizeof(qos_scheduler_t));
    
    g_qos_scheduler.total_bandwidth_mbps = total_bandwidth_mbps;
    g_qos_scheduler.available_bandwidth_mbps = total_bandwidth_mbps;
    g_qos_scheduler.policy = policy;
    g_qos_scheduler.shutdown = false;
    
    // 初始化队列锁
    for (int i = 0; i < 5; i++) {
        pthread_mutex_init(&g_qos_scheduler.queue_locks[i], NULL);
    }
    
    pthread_mutex_init(&g_qos_scheduler.scheduler_lock, NULL);
    pthread_cond_init(&g_qos_scheduler.scheduler_cond, NULL);
    
    // 配置默认QoS等级
    qos_config_t default_configs[5] = {
        // CRITICAL
        {QOS_LEVEL_CRITICAL, total_bandwidth_mbps * 50 / 100, total_bandwidth_mbps * 20 / 100, 
         100, 255, 0.5, false, true},
        // HIGH
        {QOS_LEVEL_HIGH, total_bandwidth_mbps * 30 / 100, total_bandwidth_mbps * 15 / 100, 
         500, 200, 0.3, true, true},
        // NORMAL
        {QOS_LEVEL_NORMAL, total_bandwidth_mbps * 20 / 100, total_bandwidth_mbps * 10 / 100, 
         1000, 128, 0.15, true, false},
        // LOW
        {QOS_LEVEL_LOW, total_bandwidth_mbps * 10 / 100, total_bandwidth_mbps * 3 / 100, 
         5000, 64, 0.04, true, false},
        // BACKGROUND
        {QOS_LEVEL_BACKGROUND, total_bandwidth_mbps * 5 / 100, total_bandwidth_mbps * 1 / 100, 
         10000, 32, 0.01, true, false}
    };
    
    memcpy(g_qos_scheduler.qos_configs, default_configs, sizeof(default_configs));
    
    // 初始化令牌桶
    for (int i = 0; i < 5; i++) {
        qos_config_t *config = &g_qos_scheduler.qos_configs[i];
        init_token_bucket(&g_qos_scheduler.bandwidth_buckets[i], 
                         config->max_bandwidth_mbps * 2, // 桶容量为最大带宽的2倍
                         config->max_bandwidth_mbps);    // 填充速率等于最大带宽
    }
    
    // 启动调度线程
    if (pthread_create(&g_qos_scheduler.scheduler_thread, NULL, qos_scheduler_thread, NULL) != 0) {
        cleanup_memory_qos();
        return -1;
    }
    
    printf("Memory QoS system initialized:\n");
    printf("  Total Bandwidth: %u MB/s\n", total_bandwidth_mbps);
    printf("  Policy: %s\n", 
           policy == BANDWIDTH_POLICY_FAIR ? "Fair" :
           policy == BANDWIDTH_POLICY_PRIORITY ? "Priority" :
           policy == BANDWIDTH_POLICY_WEIGHTED ? "Weighted" : "Adaptive");
    
    for (int i = 0; i < 5; i++) {
        qos_config_t *config = &g_qos_scheduler.qos_configs[i];
        printf("  QoS Level %d: %u-%u MB/s, max latency %u us, priority %u\n",
               i, config->min_bandwidth_mbps, config->max_bandwidth_mbps,
               config->max_latency_us, config->priority);
    }
    
    return 0;
}

// 提交内存访问请求
int submit_memory_request(void *address, size_t size, memory_access_type_t type, 
                         qos_level_t qos_level, uint32_t client_id) {
    if (!address || size == 0 || qos_level >= 5) {
        return -1;
    }
    
    // 创建内存请求
    memory_request_t *request = create_memory_request(address, size, type, qos_level);
    if (!request) {
        return -1;
    }
    
    request->client_id = client_id;
    request->submit_time = get_current_time_us();
    
    // 检查是否为紧急请求
    if (qos_level == QOS_LEVEL_CRITICAL) {
        request->is_urgent = true;
    }
    
    // 估算执行时间
    qos_config_t *config = &g_qos_scheduler.qos_configs[qos_level];
    request->estimated_duration_us = (size * 1000000ULL) / (config->max_bandwidth_mbps * 1024 * 1024);
    
    // 入队
    enqueue_request(request);
    
    // 通知调度器
    pthread_cond_signal(&g_qos_scheduler.scheduler_cond);
    
    g_qos_scheduler.stats.total_requests++;
    
    printf("Submitted memory request: %p (%zu bytes, QoS: %d, client: %u)\n",
           address, size, qos_level, client_id);
    
    return 0;
}

// 设置QoS配置
int set_qos_config(qos_level_t level, const qos_config_t *config) {
    if (level >= 5 || !config) {
        return -1;
    }
    
    pthread_mutex_lock(&g_qos_scheduler.scheduler_lock);
    
    // 更新配置
    g_qos_scheduler.qos_configs[level] = *config;
    
    // 重新初始化令牌桶
    init_token_bucket(&g_qos_scheduler.bandwidth_buckets[level],
                     config->max_bandwidth_mbps * 2,
                     config->max_bandwidth_mbps);
    
    // 更新带宽分配
    update_bandwidth_allocation();
    
    pthread_mutex_unlock(&g_qos_scheduler.scheduler_lock);
    
    printf("Updated QoS config for level %d: %u-%u MB/s, max latency %u us\n",
           level, config->min_bandwidth_mbps, config->max_bandwidth_mbps, config->max_latency_us);
    
    return 0;
}

// 获取QoS统计信息
void get_qos_stats(void) {
    pthread_mutex_lock(&g_qos_scheduler.scheduler_lock);
    
    printf("\n=== Memory QoS Statistics ===\n");
    printf("Total Requests: %llu\n", g_qos_scheduler.stats.total_requests);
    printf("Completed Requests: %llu\n", g_qos_scheduler.stats.completed_requests);
    printf("Dropped Requests: %llu\n", g_qos_scheduler.stats.dropped_requests);
    printf("QoS Violations: %u\n", g_qos_scheduler.stats.qos_violations);
    printf("Preemptions: %u\n", g_qos_scheduler.stats.preemptions);
    
    if (g_qos_scheduler.stats.completed_requests > 0) {
        double avg_latency = (double)g_qos_scheduler.stats.total_latency_us / 
                            g_qos_scheduler.stats.completed_requests;
        printf("Average Latency: %.2f us\n", avg_latency);
        
        double avg_bandwidth = (double)g_qos_scheduler.stats.total_bandwidth_used / 
                              g_qos_scheduler.stats.completed_requests;
        printf("Average Bandwidth Used: %.2f MB/s\n", avg_bandwidth);
    }
    
    printf("Current Queue Lengths:\n");
    for (int i = 0; i < 5; i++) {
        printf("  QoS Level %d: %u requests\n", i, g_qos_scheduler.queue_lengths[i]);
    }
    
    printf("Available Bandwidth: %u MB/s\n", g_qos_scheduler.available_bandwidth_mbps);
    printf("=============================\n\n");
    
    pthread_mutex_unlock(&g_qos_scheduler.scheduler_lock);
}

// 创建内存请求
static memory_request_t* create_memory_request(void *address, size_t size, 
                                              memory_access_type_t type, qos_level_t qos_level) {
    memory_request_t *request = malloc(sizeof(memory_request_t));
    if (!request) {
        return NULL;
    }
    
    memset(request, 0, sizeof(memory_request_t));
    
    static uint64_t next_request_id = 1;
    request->request_id = next_request_id++;
    request->address = address;
    request->size = size;
    request->type = type;
    request->qos_level = qos_level;
    request->is_urgent = false;
    request->next = NULL;
    
    return request;
}

// 释放内存请求
static void free_memory_request(memory_request_t *request) {
    if (request) {
        free(request);
    }
}

// 请求入队
static void enqueue_request(memory_request_t *request) {
    if (!request || request->qos_level >= 5) {
        return;
    }
    
    int queue_index = request->qos_level;
    
    pthread_mutex_lock(&g_qos_scheduler.queue_locks[queue_index]);
    
    // 添加到队列尾部
    if (!g_qos_scheduler.queues[queue_index]) {
        g_qos_scheduler.queues[queue_index] = request;
    } else {
        memory_request_t *current = g_qos_scheduler.queues[queue_index];
        while (current->next) {
            current = current->next;
        }
        current->next = request;
    }
    
    g_qos_scheduler.queue_lengths[queue_index]++;
    
    pthread_mutex_unlock(&g_qos_scheduler.queue_locks[queue_index]);
}

// 请求出队
static memory_request_t* dequeue_request(void) {
    memory_request_t *selected = select_next_request();
    if (!selected) {
        return NULL;
    }
    
    int queue_index = selected->qos_level;
    
    pthread_mutex_lock(&g_qos_scheduler.queue_locks[queue_index]);
    
    // 从队列中移除
    if (g_qos_scheduler.queues[queue_index] == selected) {
        g_qos_scheduler.queues[queue_index] = selected->next;
    } else {
        memory_request_t *current = g_qos_scheduler.queues[queue_index];
        while (current && current->next != selected) {
            current = current->next;
        }
        if (current) {
            current->next = selected->next;
        }
    }
    
    g_qos_scheduler.queue_lengths[queue_index]--;
    selected->next = NULL;
    
    pthread_mutex_unlock(&g_qos_scheduler.queue_locks[queue_index]);
    
    return selected;
}

// 选择下一个请求
static memory_request_t* select_next_request(void) {
    memory_request_t *best_request = NULL;
    double best_score = -1.0;
    
    // 遍历所有队列
    for (int i = 0; i < 5; i++) {
        pthread_mutex_lock(&g_qos_scheduler.queue_locks[i]);
        
        memory_request_t *current = g_qos_scheduler.queues[i];
        while (current) {
            // 检查带宽限制
            uint32_t required_tokens = (current->size * 1000) / (1024 * 1024); // MB
            if (consume_tokens(&g_qos_scheduler.bandwidth_buckets[i], required_tokens)) {
                double score = calculate_priority_score(current);
                if (score > best_score) {
                    best_score = score;
                    best_request = current;
                }
            }
            current = current->next;
        }
        
        pthread_mutex_unlock(&g_qos_scheduler.queue_locks[i]);
    }
    
    return best_request;
}

// 计算优先级分数
static double calculate_priority_score(memory_request_t *request) {
    qos_config_t *config = &g_qos_scheduler.qos_configs[request->qos_level];
    uint64_t current_time = get_current_time_us();
    uint64_t wait_time = current_time - request->submit_time;
    
    double score = 0.0;
    
    // 基础优先级分数
    score += config->priority;
    
    // 等待时间惩罚
    if (wait_time > config->max_latency_us) {
        score += (wait_time - config->max_latency_us) * 0.001; // 延迟惩罚
    }
    
    // 紧急请求加分
    if (request->is_urgent) {
        score += 1000.0;
    }
    
    // QoS等级权重
    score *= config->weight * 1000;
    
    return score;
}

// 执行内存请求
static void execute_memory_request(memory_request_t *request) {
    if (!request) {
        return;
    }
    
    request->start_time = get_current_time_us();
    
    // 模拟内存访问
    qos_config_t *config = &g_qos_scheduler.qos_configs[request->qos_level];
    uint32_t bandwidth_mbps = config->max_bandwidth_mbps;
    
    // 计算执行时间
    uint64_t execution_time_us = (request->size * 1000000ULL) / (bandwidth_mbps * 1024 * 1024);
    
    // 模拟执行延迟
    usleep(execution_time_us);
    
    request->complete_time = get_current_time_us();
    
    // 检查QoS违规
    if (check_qos_violation(request)) {
        handle_qos_violation(request);
    }
    
    // 更新统计信息
    uint64_t latency = request->complete_time - request->submit_time;
    g_qos_scheduler.stats.completed_requests++;
    g_qos_scheduler.stats.total_latency_us += latency;
    g_qos_scheduler.stats.total_bandwidth_used += bandwidth_mbps;
    
    printf("Completed memory request %llu: %zu bytes in %llu us (QoS: %d)\n",
           request->request_id, request->size, latency, request->qos_level);
}

// 检查QoS违规
static bool check_qos_violation(memory_request_t *request) {
    qos_config_t *config = &g_qos_scheduler.qos_configs[request->qos_level];
    uint64_t latency = request->complete_time - request->submit_time;
    
    return latency > config->max_latency_us;
}

// 处理QoS违规
static void handle_qos_violation(memory_request_t *request) {
    g_qos_scheduler.stats.qos_violations++;
    
    printf("QoS violation detected for request %llu (level %d)\n",
           request->request_id, request->qos_level);
    
    // 触发自适应调整
    if (g_qos_scheduler.policy == BANDWIDTH_POLICY_ADAPTIVE) {
        adaptive_bandwidth_adjustment();
    }
}

// 自适应带宽调整
static void adaptive_bandwidth_adjustment(void) {
    // 分析当前负载和性能
    uint32_t total_queue_length = 0;
    for (int i = 0; i < 5; i++) {
        total_queue_length += g_qos_scheduler.queue_lengths[i];
    }
    
    if (total_queue_length > 10) {
        // 高负载，增加高优先级带宽
        for (int i = 0; i < 2; i++) { // CRITICAL和HIGH
            qos_config_t *config = &g_qos_scheduler.qos_configs[i];
            if (config->max_bandwidth_mbps < g_qos_scheduler.total_bandwidth_mbps * 0.6) {
                config->max_bandwidth_mbps += g_qos_scheduler.total_bandwidth_mbps * 0.05;
                printf("Increased bandwidth for QoS level %d to %u MB/s\n", 
                       i, config->max_bandwidth_mbps);
            }
        }
    }
}

// 更新带宽分配
static void update_bandwidth_allocation(void) {
    uint32_t allocated_bandwidth = 0;
    
    // 计算已分配的最小带宽
    for (int i = 0; i < 5; i++) {
        allocated_bandwidth += g_qos_scheduler.qos_configs[i].min_bandwidth_mbps;
    }
    
    g_qos_scheduler.available_bandwidth_mbps = 
        g_qos_scheduler.total_bandwidth_mbps - allocated_bandwidth;
    
    printf("Updated bandwidth allocation: %u MB/s available\n", 
           g_qos_scheduler.available_bandwidth_mbps);
}

// 初始化令牌桶
static void init_token_bucket(token_bucket_t *bucket, uint32_t capacity, uint32_t refill_rate) {
    bucket->capacity = capacity;
    bucket->tokens = capacity;
    bucket->refill_rate = refill_rate;
    bucket->last_refill_time = get_current_time_us();
    pthread_mutex_init(&bucket->bucket_lock, NULL);
}

// 消费令牌
static bool consume_tokens(token_bucket_t *bucket, uint32_t tokens) {
    pthread_mutex_lock(&bucket->bucket_lock);
    
    // 先填充令牌
    refill_tokens(bucket);
    
    bool success = false;
    if (bucket->tokens >= tokens) {
        bucket->tokens -= tokens;
        success = true;
    }
    
    pthread_mutex_unlock(&bucket->bucket_lock);
    return success;
}

// 填充令牌
static void refill_tokens(token_bucket_t *bucket) {
    uint64_t current_time = get_current_time_us();
    uint64_t time_elapsed = current_time - bucket->last_refill_time;
    
    if (time_elapsed > 0) {
        uint32_t tokens_to_add = (bucket->refill_rate * time_elapsed) / 1000000; // 每秒填充
        bucket->tokens = (bucket->tokens + tokens_to_add > bucket->capacity) ? 
                        bucket->capacity : bucket->tokens + tokens_to_add;
        bucket->last_refill_time = current_time;
    }
}

// QoS调度线程
static void* qos_scheduler_thread(void *arg) {
    (void)arg;
    
    while (!g_qos_scheduler.shutdown) {
        pthread_mutex_lock(&g_qos_scheduler.scheduler_lock);
        
        // 等待请求或超时
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 10000000; // 10ms超时
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        
        pthread_cond_timedwait(&g_qos_scheduler.scheduler_cond, 
                              &g_qos_scheduler.scheduler_lock, &timeout);
        
        // 处理请求
        memory_request_t *request = dequeue_request();
        if (request) {
            pthread_mutex_unlock(&g_qos_scheduler.scheduler_lock);
            execute_memory_request(request);
            free_memory_request(request);
        } else {
            pthread_mutex_unlock(&g_qos_scheduler.scheduler_lock);
        }
        
        // 定期填充令牌桶
        for (int i = 0; i < 5; i++) {
            refill_tokens(&g_qos_scheduler.bandwidth_buckets[i]);
        }
    }
    
    return NULL;
}

// 获取当前时间（微秒）
static uint64_t get_current_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

// 清理QoS系统
void cleanup_memory_qos(void) {
    g_qos_scheduler.shutdown = true;
    
    // 通知调度线程退出
    pthread_cond_signal(&g_qos_scheduler.scheduler_cond);
    
    // 等待调度线程结束
    pthread_join(g_qos_scheduler.scheduler_thread, NULL);
    
    // 清理队列
    for (int i = 0; i < 5; i++) {
        pthread_mutex_lock(&g_qos_scheduler.queue_locks[i]);
        
        memory_request_t *current = g_qos_scheduler.queues[i];
        while (current) {
            memory_request_t *next = current->next;
            free_memory_request(current);
            current = next;
        }
        
        pthread_mutex_unlock(&g_qos_scheduler.queue_locks[i]);
        pthread_mutex_destroy(&g_qos_scheduler.queue_locks[i]);
        
        // 清理令牌桶
        pthread_mutex_destroy(&g_qos_scheduler.bandwidth_buckets[i].bucket_lock);
    }
    
    // 清理锁
    pthread_mutex_destroy(&g_qos_scheduler.scheduler_lock);
    pthread_cond_destroy(&g_qos_scheduler.scheduler_cond);
    
    printf("Memory QoS system cleaned up\n");
}