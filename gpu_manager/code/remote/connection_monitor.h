#ifndef CONNECTION_MONITOR_H
#define CONNECTION_MONITOR_H

#include <stdint.h>
#include <stdbool.h>
#include <sys/socket.h>

#ifdef __cplusplus
extern "C" {
#endif

// 常量定义
#define MAX_MONITORED_CONNECTIONS 64
#define MAX_ADDRESS_LEN 256
#define HEARTBEAT_MAX_PAYLOAD 256

// 监控结果枚举
typedef enum {
    MONITOR_SUCCESS = 0,
    MONITOR_ERROR_INVALID_PARAM,
    MONITOR_ERROR_INVALID_CONNECTION,
    MONITOR_ERROR_CONNECTION_EXISTS,
    MONITOR_ERROR_CONNECTION_NOT_FOUND,
    MONITOR_ERROR_MAX_CONNECTIONS,
    MONITOR_ERROR_SEND_FAILED,
    MONITOR_ERROR_RECEIVE_FAILED,
    MONITOR_ERROR_TIMEOUT,
    MONITOR_ERROR_CONNECTION_CLOSED,
    MONITOR_ERROR_INVALID_RESPONSE,
    MONITOR_ERROR_THREAD_CREATE_FAILED
} monitor_result_t;

// 连接状态枚举
typedef enum {
    CONNECTION_STATUS_UNKNOWN = 0,
    CONNECTION_STATUS_CONNECTING,
    CONNECTION_STATUS_CONNECTED,
    CONNECTION_STATUS_DISCONNECTED,
    CONNECTION_STATUS_FAILED
} connection_status_t;

// 故障类型枚举
typedef enum {
    FAILURE_TYPE_HEARTBEAT_TIMEOUT = 0,
    FAILURE_TYPE_CONNECTION_LOST,
    FAILURE_TYPE_PROTOCOL_ERROR,
    FAILURE_TYPE_AUTHENTICATION_FAILED
} failure_type_t;

// 故障回调函数类型
typedef void (*failure_callback_t)(int connection_id, failure_type_t failure_type);

// 恢复回调函数类型
typedef void (*recovery_callback_t)(int connection_id);

// 心跳配置结构
typedef struct {
    uint32_t heartbeat_interval_ms;    // 心跳间隔（毫秒）
    uint32_t heartbeat_timeout_ms;     // 心跳超时（毫秒）
    uint32_t max_failures;             // 最大连续失败次数
    uint32_t check_interval_ms;        // 检查间隔（毫秒）
    failure_callback_t failure_callback;   // 故障回调函数
    recovery_callback_t recovery_callback; // 恢复回调函数
} heartbeat_config_t;

// 连接信息结构
typedef struct {
    int connection_id;                 // 连接ID
    int socket_fd;                     // 套接字文件描述符
    char remote_address[MAX_ADDRESS_LEN]; // 远程地址
    connection_status_t status;        // 连接状态
    uint64_t last_heartbeat_sent;      // 最后发送心跳时间戳
    uint64_t last_heartbeat_received;  // 最后接收心跳时间戳
    uint32_t heartbeat_sequence;       // 心跳序列号
    uint32_t consecutive_failures;     // 连续失败次数
    uint64_t last_rtt_us;             // 最后一次往返时间（微秒）
} connection_info_t;

// 连接统计信息
typedef struct {
    uint64_t last_heartbeat_sent;
    uint64_t last_heartbeat_received;
    uint32_t consecutive_failures;
    uint64_t last_rtt_us;
} connection_stats_t;

// 连接监控器结构
typedef struct {
    connection_info_t connections[MAX_MONITORED_CONNECTIONS];
    int connection_count;
    heartbeat_config_t config;
    bool is_running;
} connection_monitor_t;

// 核心API函数

/**
 * 初始化连接监控器
 * @param config 心跳配置
 * @return MONITOR_SUCCESS 成功，其他值表示错误
 */
monitor_result_t connection_monitor_init(const heartbeat_config_t *config);

/**
 * 清理连接监控器
 */
void connection_monitor_cleanup(void);

/**
 * 添加连接到监控列表
 * @param connection_id 连接ID
 * @param socket_fd 套接字文件描述符
 * @param remote_address 远程地址
 * @return MONITOR_SUCCESS 成功，其他值表示错误
 */
monitor_result_t connection_monitor_add(int connection_id, int socket_fd, 
                                       const char *remote_address);

/**
 * 从监控列表中移除连接
 * @param connection_id 连接ID
 * @return MONITOR_SUCCESS 成功，其他值表示错误
 */
monitor_result_t connection_monitor_remove(int connection_id);

/**
 * 获取连接状态
 * @param connection_id 连接ID
 * @param status 输出连接状态
 * @return MONITOR_SUCCESS 成功，其他值表示错误
 */
monitor_result_t connection_monitor_get_status(int connection_id, 
                                              connection_status_t *status);

/**
 * 获取连接统计信息
 * @param connection_id 连接ID
 * @param stats 输出统计信息
 * @return MONITOR_SUCCESS 成功，其他值表示错误
 */
monitor_result_t connection_monitor_get_stats(int connection_id, 
                                             connection_stats_t *stats);

// 错误处理函数

/**
 * 获取错误字符串
 * @param error 错误码
 * @return 错误描述字符串
 */
const char* monitor_get_error_string(monitor_result_t error);

// 工具宏

#define MONITOR_CHECK(call) do { \
    monitor_result_t result = (call); \
    if (result != MONITOR_SUCCESS) { \
        fprintf(stderr, "Monitor error in %s: %s\n", #call, \
                monitor_get_error_string(result)); \
        return result; \
    } \
} while(0)

#define MONITOR_CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Null pointer: %s\n", #ptr); \
        return MONITOR_ERROR_INVALID_PARAM; \
    } \
} while(0)

// 默认配置值
#define DEFAULT_HEARTBEAT_INTERVAL_MS 5000
#define DEFAULT_HEARTBEAT_TIMEOUT_MS  2000
#define DEFAULT_MAX_FAILURES          3
#define DEFAULT_CHECK_INTERVAL_MS     1000

// 创建默认心跳配置的辅助宏
#define HEARTBEAT_CONFIG_DEFAULT() { \
    .heartbeat_interval_ms = DEFAULT_HEARTBEAT_INTERVAL_MS, \
    .heartbeat_timeout_ms = DEFAULT_HEARTBEAT_TIMEOUT_MS, \
    .max_failures = DEFAULT_MAX_FAILURES, \
    .check_interval_ms = DEFAULT_CHECK_INTERVAL_MS, \
    .failure_callback = NULL, \
    .recovery_callback = NULL \
}

#ifdef __cplusplus
}
#endif

#endif // CONNECTION_MONITOR_H