// 故障检测与隔离机制实现
// 支持多租户环境下的错误检测、隔离和恢复

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#define MAX_TENANTS 32
#define MAX_ERROR_TYPES 16
#define ERROR_WINDOW_SIZE 60  // 错误窗口大小（秒）
#define MAX_LOG_ENTRIES 1000

// 错误类型枚举
typedef enum {
    ERROR_MEMORY_FAULT,      // 内存错误
    ERROR_COMPUTE_TIMEOUT,   // 计算超时
    ERROR_KERNEL_CRASH,      // 内核崩溃
    ERROR_RESOURCE_LEAK,     // 资源泄漏
    ERROR_PERMISSION_DENIED, // 权限拒绝
    ERROR_DEVICE_BUSY,       // 设备忙
    ERROR_INVALID_OPERATION, // 无效操作
    ERROR_NETWORK_FAILURE,   // 网络故障
    ERROR_THERMAL_THROTTLE,  // 温度限制
    ERROR_POWER_LIMIT,       // 功耗限制
    ERROR_UNKNOWN           // 未知错误
} error_type_t;

// 错误严重程度
typedef enum {
    SEVERITY_LOW = 1,
    SEVERITY_MEDIUM = 2,
    SEVERITY_HIGH = 3,
    SEVERITY_CRITICAL = 4
} error_severity_t;

// 错误信息结构
typedef struct {
    error_type_t type;
    error_severity_t severity;
    time_t timestamp;
    int tenant_id;
    char description[256];
    char context[512];
    int error_code;
} error_info_t;

// 隔离状态
typedef enum {
    ISOLATION_NONE,          // 无隔离
    ISOLATION_WARNING,       // 警告状态
    ISOLATION_THROTTLED,     // 限制状态
    ISOLATION_SUSPENDED,     // 暂停状态
    ISOLATION_TERMINATED     // 终止状态
} isolation_status_t;

// 租户状态
typedef struct {
    int tenant_id;
    isolation_status_t isolation_status;
    time_t isolation_start_time;
    int total_errors;
    int error_counts[MAX_ERROR_TYPES];
    time_t last_error_time;
    float error_rate;        // 错误率（错误/分钟）
    int consecutive_errors;  // 连续错误数
    int recovery_attempts;   // 恢复尝试次数
} tenant_status_t;

// 故障检测器
typedef struct {
    int error_threshold;        // 错误阈值
    int *error_counters;        // 错误计数器
    time_t *last_error_time;    // 最后错误时间
    isolation_status_t *isolation_status; // 隔离状态
    int tenant_count;
    tenant_status_t *tenant_status; // 租户状态
    error_info_t *error_log;    // 错误日志
    int log_index;              // 日志索引
    int log_count;              // 日志计数
    time_t start_time;          // 启动时间
} fault_detector_t;

// 错误类型名称
const char* error_type_names[] = {
    "Memory Fault", "Compute Timeout", "Kernel Crash", "Resource Leak",
    "Permission Denied", "Device Busy", "Invalid Operation", "Network Failure",
    "Thermal Throttle", "Power Limit", "Unknown"
};

// 严重程度名称
const char* severity_names[] = {
    "", "Low", "Medium", "High", "Critical"
};

// 隔离状态名称
const char* isolation_status_names[] = {
    "None", "Warning", "Throttled", "Suspended", "Terminated"
};

// 初始化故障检测器
fault_detector_t* init_fault_detector(int tenant_count, int error_threshold) {
    fault_detector_t *detector = malloc(sizeof(fault_detector_t));
    if (!detector) return NULL;
    
    detector->tenant_count = tenant_count;
    detector->error_threshold = error_threshold;
    detector->log_index = 0;
    detector->log_count = 0;
    detector->start_time = time(NULL);
    
    // 分配内存
    detector->error_counters = calloc(tenant_count, sizeof(int));
    detector->last_error_time = calloc(tenant_count, sizeof(time_t));
    detector->isolation_status = calloc(tenant_count, sizeof(isolation_status_t));
    detector->tenant_status = calloc(tenant_count, sizeof(tenant_status_t));
    detector->error_log = calloc(MAX_LOG_ENTRIES, sizeof(error_info_t));
    
    // 初始化租户状态
    for (int i = 0; i < tenant_count; i++) {
        detector->tenant_status[i].tenant_id = i;
        detector->tenant_status[i].isolation_status = ISOLATION_NONE;
    }
    
    return detector;
}

// 记录错误日志
void log_error(fault_detector_t *detector, error_info_t *error) {
    if (detector->log_count < MAX_LOG_ENTRIES) {
        detector->error_log[detector->log_index] = *error;
        detector->log_index = (detector->log_index + 1) % MAX_LOG_ENTRIES;
        detector->log_count++;
    } else {
        // 覆盖最旧的日志
        detector->error_log[detector->log_index] = *error;
        detector->log_index = (detector->log_index + 1) % MAX_LOG_ENTRIES;
    }
    
    printf("[LOG] Tenant %d: %s (%s) - %s\n",
           error->tenant_id, error_type_names[error->type],
           severity_names[error->severity], error->description);
}

// 清理租户资源
int cleanup_tenant_resources(int tenant_id) {
    printf("[CLEANUP] Cleaning up resources for tenant %d\n", tenant_id);
    
    // 模拟资源清理过程
    // 实际实现中需要：
    // 1. 停止租户的所有GPU任务
    // 2. 释放分配的GPU内存
    // 3. 清理临时文件
    // 4. 重置GPU上下文
    
    usleep(100000);  // 模拟清理时间
    printf("[CLEANUP] Resources cleaned for tenant %d\n", tenant_id);
    
    return 0;
}

// 记录隔离事件
void log_isolation_event(int tenant_id, error_info_t *error) {
    printf("[ISOLATION] Tenant %d isolated due to %s (error code: %d)\n",
           tenant_id, error_type_names[error->type], error->error_code);
    
    // 可以发送告警通知
    // send_alert_notification(tenant_id, error);
}

// 计算错误率
float calculate_error_rate(fault_detector_t *detector, int tenant_id) {
    time_t current_time = time(NULL);
    time_t window_start = current_time - ERROR_WINDOW_SIZE;
    
    int errors_in_window = 0;
    
    // 统计时间窗口内的错误数
    for (int i = 0; i < detector->log_count; i++) {
        error_info_t *error = &detector->error_log[i];
        if (error->tenant_id == tenant_id && error->timestamp >= window_start) {
            errors_in_window++;
        }
    }
    
    // 计算每分钟错误率
    float rate = (float)errors_in_window / (ERROR_WINDOW_SIZE / 60.0);
    detector->tenant_status[tenant_id].error_rate = rate;
    
    return rate;
}

// 确定隔离级别
isolation_status_t determine_isolation_level(fault_detector_t *detector, 
                                            int tenant_id, error_info_t *error) {
    tenant_status_t *status = &detector->tenant_status[tenant_id];
    
    // 基于错误严重程度
    if (error->severity == SEVERITY_CRITICAL) {
        return ISOLATION_TERMINATED;
    }
    
    // 基于错误计数
    if (status->total_errors > detector->error_threshold * 2) {
        return ISOLATION_SUSPENDED;
    } else if (status->total_errors > detector->error_threshold) {
        return ISOLATION_THROTTLED;
    }
    
    // 基于错误率
    float error_rate = calculate_error_rate(detector, tenant_id);
    if (error_rate > 10.0) {  // 每分钟超过10个错误
        return ISOLATION_SUSPENDED;
    } else if (error_rate > 5.0) {
        return ISOLATION_THROTTLED;
    } else if (error_rate > 2.0) {
        return ISOLATION_WARNING;
    }
    
    // 基于连续错误
    if (status->consecutive_errors > 5) {
        return ISOLATION_THROTTLED;
    } else if (status->consecutive_errors > 3) {
        return ISOLATION_WARNING;
    }
    
    return ISOLATION_NONE;
}

// 执行隔离操作
int execute_isolation(fault_detector_t *detector, int tenant_id, 
                     isolation_status_t new_status) {
    tenant_status_t *status = &detector->tenant_status[tenant_id];
    isolation_status_t old_status = status->isolation_status;
    
    if (new_status == old_status) {
        return 0;  // 状态未变化
    }
    
    status->isolation_status = new_status;
    status->isolation_start_time = time(NULL);
    
    printf("[ISOLATION] Tenant %d: %s -> %s\n",
           tenant_id, isolation_status_names[old_status], 
           isolation_status_names[new_status]);
    
    switch (new_status) {
    case ISOLATION_WARNING:
        printf("[ACTION] Tenant %d: Sending warning notification\n", tenant_id);
        break;
        
    case ISOLATION_THROTTLED:
        printf("[ACTION] Tenant %d: Reducing resource allocation by 50%%\n", tenant_id);
        // 实际实现中需要调整资源分配
        break;
        
    case ISOLATION_SUSPENDED:
        printf("[ACTION] Tenant %d: Suspending all operations\n", tenant_id);
        cleanup_tenant_resources(tenant_id);
        break;
        
    case ISOLATION_TERMINATED:
        printf("[ACTION] Tenant %d: Terminating tenant access\n", tenant_id);
        cleanup_tenant_resources(tenant_id);
        break;
        
    default:
        break;
    }
    
    return 1;
}

// 故障检测算法
int detect_and_isolate_faults(fault_detector_t *detector, int tenant_id, 
                             error_info_t *error) {
    if (tenant_id < 0 || tenant_id >= detector->tenant_count) {
        return -1;
    }
    
    time_t current_time = time(NULL);
    tenant_status_t *status = &detector->tenant_status[tenant_id];
    
    // 记录错误日志
    log_error(detector, error);
    
    // 更新错误计数
    detector->error_counters[tenant_id]++;
    detector->last_error_time[tenant_id] = current_time;
    status->total_errors++;
    status->last_error_time = current_time;
    
    if (error->type < MAX_ERROR_TYPES) {
        status->error_counts[error->type]++;
    }
    
    // 检查是否为连续错误
    if (current_time - status->last_error_time < 5) {  // 5秒内的错误视为连续
        status->consecutive_errors++;
    } else {
        status->consecutive_errors = 1;
    }
    
    // 确定隔离级别
    isolation_status_t new_isolation = determine_isolation_level(detector, tenant_id, error);
    
    // 执行隔离
    int isolation_changed = execute_isolation(detector, tenant_id, new_isolation);
    
    // 检查是否需要立即隔离
    if (detector->error_counters[tenant_id] > detector->error_threshold) {
        detector->isolation_status[tenant_id] = new_isolation;
        return 1; // 已隔离
    }
    
    return 0; // 未隔离
}

// 尝试恢复租户
int attempt_tenant_recovery(fault_detector_t *detector, int tenant_id) {
    tenant_status_t *status = &detector->tenant_status[tenant_id];
    
    if (status->isolation_status == ISOLATION_NONE) {
        return 0;  // 无需恢复
    }
    
    status->recovery_attempts++;
    
    printf("[RECOVERY] Attempting recovery for tenant %d (attempt %d)\n",
           tenant_id, status->recovery_attempts);
    
    // 模拟恢复过程
    usleep(200000);  // 200ms恢复时间
    
    // 检查恢复条件
    time_t current_time = time(NULL);
    time_t isolation_duration = current_time - status->isolation_start_time;
    
    // 如果隔离时间足够长且错误率下降，尝试恢复
    if (isolation_duration > 30) {  // 隔离30秒后
        float current_error_rate = calculate_error_rate(detector, tenant_id);
        
        if (current_error_rate < 1.0) {  // 错误率低于1/分钟
            status->isolation_status = ISOLATION_NONE;
            status->consecutive_errors = 0;
            printf("[RECOVERY] Tenant %d successfully recovered\n", tenant_id);
            return 1;
        }
    }
    
    printf("[RECOVERY] Tenant %d recovery failed\n", tenant_id);
    return 0;
}

// 生成状态报告
void generate_status_report(fault_detector_t *detector) {
    printf("\n=== Fault Detection Status Report ===\n");
    printf("Uptime: %ld seconds\n", time(NULL) - detector->start_time);
    printf("Total log entries: %d\n", detector->log_count);
    printf("\nTenant Status:\n");
    printf("ID\tStatus\t\tErrors\tRate\tConsecutive\tRecovery\n");
    printf("--\t------\t\t------\t----\t-----------\t--------\n");
    
    for (int i = 0; i < detector->tenant_count; i++) {
        tenant_status_t *status = &detector->tenant_status[i];
        float error_rate = calculate_error_rate(detector, i);
        
        printf("%d\t%s\t%d\t%.1f\t%d\t\t%d\n",
               i, isolation_status_names[status->isolation_status],
               status->total_errors, error_rate, status->consecutive_errors,
               status->recovery_attempts);
    }
}

// 清理故障检测器
void cleanup_fault_detector(fault_detector_t *detector) {
    if (detector) {
        free(detector->error_counters);
        free(detector->last_error_time);
        free(detector->isolation_status);
        free(detector->tenant_status);
        free(detector->error_log);
        free(detector);
    }
}

// 创建测试错误
error_info_t create_test_error(int tenant_id, error_type_t type, error_severity_t severity) {
    error_info_t error = {0};
    error.type = type;
    error.severity = severity;
    error.timestamp = time(NULL);
    error.tenant_id = tenant_id;
    error.error_code = rand() % 1000;
    
    snprintf(error.description, sizeof(error.description),
             "Test error %d for tenant %d", error.error_code, tenant_id);
    snprintf(error.context, sizeof(error.context),
             "Context: GPU operation failed with code %d", error.error_code);
    
    return error;
}

// 模拟故障场景
void simulate_fault_scenarios(fault_detector_t *detector) {
    printf("\n=== Simulating Fault Scenarios ===\n");
    
    // 场景1：正常错误
    for (int i = 0; i < 3; i++) {
        error_info_t error = create_test_error(0, ERROR_MEMORY_FAULT, SEVERITY_LOW);
        detect_and_isolate_faults(detector, 0, &error);
        sleep(1);
    }
    
    // 场景2：高频错误
    for (int i = 0; i < 8; i++) {
        error_info_t error = create_test_error(1, ERROR_COMPUTE_TIMEOUT, SEVERITY_MEDIUM);
        detect_and_isolate_faults(detector, 1, &error);
        usleep(500000);  // 0.5秒间隔
    }
    
    // 场景3：严重错误
    error_info_t critical_error = create_test_error(2, ERROR_KERNEL_CRASH, SEVERITY_CRITICAL);
    detect_and_isolate_faults(detector, 2, &critical_error);
    
    // 场景4：尝试恢复
    sleep(2);
    attempt_tenant_recovery(detector, 1);
    
    // 场景5：连续错误
    for (int i = 0; i < 6; i++) {
        error_info_t error = create_test_error(3, ERROR_RESOURCE_LEAK, SEVERITY_HIGH);
        detect_and_isolate_faults(detector, 3, &error);
        usleep(100000);  // 0.1秒间隔
    }
}

// 示例使用
int main() {
    srand(time(NULL));
    
    // 初始化故障检测器（4个租户，错误阈值5）
    fault_detector_t *detector = init_fault_detector(4, 5);
    if (!detector) {
        printf("Failed to initialize fault detector\n");
        return -1;
    }
    
    // 模拟故障场景
    simulate_fault_scenarios(detector);
    
    // 生成状态报告
    generate_status_report(detector);
    
    // 清理资源
    cleanup_fault_detector(detector);
    
    return 0;
}