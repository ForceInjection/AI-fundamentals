// QoS带宽控制器实现
// 基于令牌桶算法的GPU带宽限制和QoS保障

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>

#define MAX_TENANTS 16
#define NANOSECONDS_PER_SECOND 1000000000UL

// 令牌桶结构
typedef struct {
    double tokens;              // 当前令牌数
    double capacity;            // 桶容量
    double refill_rate;         // 令牌填充速率（令牌/秒）
    struct timespec last_refill; // 最后填充时间
    int is_active;              // 是否激活
} token_bucket_t;

// QoS带宽控制器
typedef struct {
    float total_bandwidth;      // 总带宽（GB/s）
    float *allocated_bandwidth; // 已分配带宽
    int *tenant_priorities;     // 租户优先级
    int tenant_count;
    token_bucket_t *rate_limiters; // 令牌桶限流器
    float *guaranteed_bandwidth; // 保证带宽
    float *max_bandwidth;       // 最大带宽
    int *violation_count;       // 违规计数
} qos_bandwidth_controller_t;

// 租户信息
typedef struct {
    int tenant_id;
    int priority;               // 优先级（1-10）
    float sla_bandwidth;        // SLA保证带宽
    float current_usage;        // 当前使用量
    int request_count;          // 请求计数
    struct timespec last_request; // 最后请求时间
} tenant_info_t;

// 初始化令牌桶
token_bucket_t* init_token_bucket(double capacity, double refill_rate) {
    token_bucket_t *bucket = malloc(sizeof(token_bucket_t));
    if (!bucket) return NULL;
    
    bucket->tokens = capacity;  // 初始时桶满
    bucket->capacity = capacity;
    bucket->refill_rate = refill_rate;
    bucket->is_active = 1;
    
    clock_gettime(CLOCK_MONOTONIC, &bucket->last_refill);
    
    return bucket;
}

// 令牌桶算法实现
int token_bucket_allow_request(token_bucket_t *bucket, size_t request_size) {
    if (!bucket || !bucket->is_active) {
        return 0;
    }
    
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);
    
    // 计算时间差（秒）
    double time_diff = (current_time.tv_sec - bucket->last_refill.tv_sec) +
                      (current_time.tv_nsec - bucket->last_refill.tv_nsec) / 1e9;
    
    // 添加令牌
    bucket->tokens += time_diff * bucket->refill_rate;
    if (bucket->tokens > bucket->capacity) {
        bucket->tokens = bucket->capacity;
    }
    
    bucket->last_refill = current_time;
    
    // 检查是否有足够令牌
    double tokens_needed = (double)request_size / (1024 * 1024);  // 转换为MB
    if (bucket->tokens >= tokens_needed) {
        bucket->tokens -= tokens_needed;
        return 1; // 允许请求
    }
    
    return 0; // 拒绝请求
}

// 动态调整令牌桶参数
void adjust_token_bucket(token_bucket_t *bucket, double new_rate, double new_capacity) {
    bucket->refill_rate = new_rate;
    bucket->capacity = new_capacity;
    
    // 确保当前令牌数不超过新容量
    if (bucket->tokens > bucket->capacity) {
        bucket->tokens = bucket->capacity;
    }
}

// 初始化QoS带宽控制器
qos_bandwidth_controller_t* init_qos_controller(float total_bandwidth, int tenant_count) {
    qos_bandwidth_controller_t *controller = malloc(sizeof(qos_bandwidth_controller_t));
    if (!controller) return NULL;
    
    controller->total_bandwidth = total_bandwidth;
    controller->tenant_count = tenant_count;
    
    // 分配内存
    controller->allocated_bandwidth = calloc(tenant_count, sizeof(float));
    controller->tenant_priorities = malloc(tenant_count * sizeof(int));
    controller->rate_limiters = malloc(tenant_count * sizeof(token_bucket_t));
    controller->guaranteed_bandwidth = malloc(tenant_count * sizeof(float));
    controller->max_bandwidth = malloc(tenant_count * sizeof(float));
    controller->violation_count = calloc(tenant_count, sizeof(int));
    
    // 初始化租户配置
    for (int i = 0; i < tenant_count; i++) {
        controller->tenant_priorities[i] = 5;  // 默认优先级
        controller->guaranteed_bandwidth[i] = total_bandwidth / tenant_count * 0.5;  // 50%保证
        controller->max_bandwidth[i] = total_bandwidth / tenant_count * 1.5;  // 150%最大
        
        // 初始化令牌桶
        double bucket_capacity = controller->max_bandwidth[i] * 2;  // 2秒的突发容量
        double refill_rate = controller->guaranteed_bandwidth[i];
        
        token_bucket_t *bucket = init_token_bucket(bucket_capacity, refill_rate);
        if (bucket) {
            controller->rate_limiters[i] = *bucket;
            free(bucket);
        }
    }
    
    return controller;
}

// 计算动态带宽分配
void calculate_dynamic_bandwidth_allocation(qos_bandwidth_controller_t *controller) {
    float total_guaranteed = 0;
    float total_requested = 0;
    
    // 计算总保证带宽和总请求带宽
    for (int i = 0; i < controller->tenant_count; i++) {
        total_guaranteed += controller->guaranteed_bandwidth[i];
        total_requested += controller->allocated_bandwidth[i];
    }
    
    // 如果总请求超过总带宽，按优先级分配
    if (total_requested > controller->total_bandwidth) {
        float available_bandwidth = controller->total_bandwidth;
        
        // 首先保证高优先级租户的基本需求
        for (int priority = 10; priority >= 1; priority--) {
            for (int i = 0; i < controller->tenant_count; i++) {
                if (controller->tenant_priorities[i] == priority) {
                    float guaranteed = controller->guaranteed_bandwidth[i];
                    if (available_bandwidth >= guaranteed) {
                        controller->allocated_bandwidth[i] = guaranteed;
                        available_bandwidth -= guaranteed;
                    } else {
                        controller->allocated_bandwidth[i] = available_bandwidth;
                        available_bandwidth = 0;
                    }
                }
            }
        }
        
        // 分配剩余带宽
        if (available_bandwidth > 0) {
            for (int priority = 10; priority >= 1; priority--) {
                for (int i = 0; i < controller->tenant_count; i++) {
                    if (controller->tenant_priorities[i] == priority && available_bandwidth > 0) {
                        float additional = fminf(available_bandwidth, 
                                               controller->max_bandwidth[i] - controller->allocated_bandwidth[i]);
                        controller->allocated_bandwidth[i] += additional;
                        available_bandwidth -= additional;
                    }
                }
            }
        }
    }
    
    // 更新令牌桶参数
    for (int i = 0; i < controller->tenant_count; i++) {
        adjust_token_bucket(&controller->rate_limiters[i], 
                           controller->allocated_bandwidth[i],
                           controller->allocated_bandwidth[i] * 2);
    }
}

// 处理带宽请求
int handle_bandwidth_request(qos_bandwidth_controller_t *controller, 
                           int tenant_id, size_t request_size) {
    if (tenant_id < 0 || tenant_id >= controller->tenant_count) {
        return -1;  // 无效租户ID
    }
    
    token_bucket_t *bucket = &controller->rate_limiters[tenant_id];
    
    // 检查令牌桶是否允许请求
    if (token_bucket_allow_request(bucket, request_size)) {
        printf("Tenant %d: Request %zu bytes ALLOWED (tokens: %.2f)\n", 
               tenant_id, request_size, bucket->tokens);
        return 1;  // 允许
    } else {
        controller->violation_count[tenant_id]++;
        printf("Tenant %d: Request %zu bytes DENIED (tokens: %.2f, violations: %d)\n", 
               tenant_id, request_size, bucket->tokens, controller->violation_count[tenant_id]);
        return 0;  // 拒绝
    }
}

// 设置租户QoS参数
void set_tenant_qos(qos_bandwidth_controller_t *controller, int tenant_id,
                   int priority, float guaranteed_bw, float max_bw) {
    if (tenant_id < 0 || tenant_id >= controller->tenant_count) {
        return;
    }
    
    controller->tenant_priorities[tenant_id] = priority;
    controller->guaranteed_bandwidth[tenant_id] = guaranteed_bw;
    controller->max_bandwidth[tenant_id] = max_bw;
    
    // 更新令牌桶
    adjust_token_bucket(&controller->rate_limiters[tenant_id], guaranteed_bw, max_bw * 2);
    
    printf("Tenant %d QoS updated: Priority=%d, Guaranteed=%.2f GB/s, Max=%.2f GB/s\n",
           tenant_id, priority, guaranteed_bw, max_bw);
}

// 监控和报告
void report_qos_status(qos_bandwidth_controller_t *controller) {
    printf("\n=== QoS Bandwidth Controller Status ===\n");
    printf("Total Bandwidth: %.2f GB/s\n", controller->total_bandwidth);
    printf("\nTenant Status:\n");
    printf("ID\tPriority\tGuaranteed\tMax\t\tAllocated\tViolations\tTokens\n");
    printf("--\t--------\t----------\t---\t\t---------\t----------\t------\n");
    
    for (int i = 0; i < controller->tenant_count; i++) {
        printf("%d\t%d\t\t%.2f\t\t%.2f\t\t%.2f\t\t%d\t\t%.2f\n",
               i, controller->tenant_priorities[i],
               controller->guaranteed_bandwidth[i],
               controller->max_bandwidth[i],
               controller->allocated_bandwidth[i],
               controller->violation_count[i],
               controller->rate_limiters[i].tokens);
    }
}

// 自适应QoS调整
void adaptive_qos_adjustment(qos_bandwidth_controller_t *controller) {
    for (int i = 0; i < controller->tenant_count; i++) {
        // 如果违规次数过多，降低优先级
        if (controller->violation_count[i] > 10) {
            if (controller->tenant_priorities[i] > 1) {
                controller->tenant_priorities[i]--;
                printf("Tenant %d priority decreased to %d due to violations\n",
                       i, controller->tenant_priorities[i]);
            }
            controller->violation_count[i] = 0;  // 重置计数
        }
        
        // 如果长时间无违规，可以提升优先级
        if (controller->violation_count[i] == 0 && 
            controller->tenant_priorities[i] < 10) {
            // 这里可以添加更复杂的逻辑
        }
    }
    
    // 重新计算带宽分配
    calculate_dynamic_bandwidth_allocation(controller);
}

// 清理QoS控制器
void cleanup_qos_controller(qos_bandwidth_controller_t *controller) {
    if (controller) {
        free(controller->allocated_bandwidth);
        free(controller->tenant_priorities);
        free(controller->rate_limiters);
        free(controller->guaranteed_bandwidth);
        free(controller->max_bandwidth);
        free(controller->violation_count);
        free(controller);
    }
}

// 模拟带宽请求
void simulate_bandwidth_requests(qos_bandwidth_controller_t *controller) {
    printf("\n=== Simulating Bandwidth Requests ===\n");
    
    // 模拟不同大小的请求
    size_t request_sizes[] = {1024*1024, 5*1024*1024, 10*1024*1024, 50*1024*1024};
    int num_sizes = sizeof(request_sizes) / sizeof(request_sizes[0]);
    
    for (int round = 0; round < 5; round++) {
        printf("\nRound %d:\n", round + 1);
        
        for (int tenant = 0; tenant < controller->tenant_count; tenant++) {
            size_t request_size = request_sizes[rand() % num_sizes];
            handle_bandwidth_request(controller, tenant, request_size);
        }
        
        // 等待一段时间让令牌桶恢复
        usleep(100000);  // 100ms
        
        // 每隔几轮进行自适应调整
        if (round % 2 == 1) {
            adaptive_qos_adjustment(controller);
        }
    }
}

// 示例使用
int main() {
    srand(time(NULL));
    
    // 初始化QoS控制器（总带宽10 GB/s，4个租户）
    qos_bandwidth_controller_t *controller = init_qos_controller(10.0, 4);
    if (!controller) {
        printf("Failed to initialize QoS controller\n");
        return -1;
    }
    
    // 设置不同租户的QoS参数
    set_tenant_qos(controller, 0, 9, 3.0, 5.0);  // 高优先级租户
    set_tenant_qos(controller, 1, 7, 2.0, 4.0);  // 中高优先级
    set_tenant_qos(controller, 2, 5, 1.5, 3.0);  // 中等优先级
    set_tenant_qos(controller, 3, 3, 1.0, 2.0);  // 低优先级
    
    // 计算初始带宽分配
    calculate_dynamic_bandwidth_allocation(controller);
    
    // 显示初始状态
    report_qos_status(controller);
    
    // 模拟带宽请求
    simulate_bandwidth_requests(controller);
    
    // 显示最终状态
    report_qos_status(controller);
    
    // 清理资源
    cleanup_qos_controller(controller);
    
    return 0;
}