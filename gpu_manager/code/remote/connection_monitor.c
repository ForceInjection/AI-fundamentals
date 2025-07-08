#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <signal.h>

#include "connection_monitor.h"
#include "../common/gpu_common.h"

// 连接监控器全局实例
static connection_monitor_t g_monitor = {0};
static bool g_monitor_running = false;
static pthread_t g_monitor_thread;
static pthread_mutex_t g_monitor_mutex = PTHREAD_MUTEX_INITIALIZER;

// 心跳包结构
typedef struct {
    uint32_t magic;           // 魔数标识
    uint32_t sequence;        // 序列号
    uint64_t timestamp;       // 时间戳
    uint32_t payload_size;    // 负载大小
    char payload[HEARTBEAT_MAX_PAYLOAD];
} heartbeat_packet_t;

#define HEARTBEAT_MAGIC 0x48425447  // "HBTG" - HeartBeat To GPU

// 发送心跳包
static monitor_result_t send_heartbeat(connection_info_t *conn) {
    if (!conn || conn->socket_fd < 0) {
        return MONITOR_ERROR_INVALID_CONNECTION;
    }
    
    heartbeat_packet_t packet = {0};
    packet.magic = HEARTBEAT_MAGIC;
    packet.sequence = conn->heartbeat_sequence++;
    packet.timestamp = gpu_get_timestamp_us();
    
    // 添加简单的负载信息
    snprintf(packet.payload, sizeof(packet.payload), 
             "heartbeat from client %d", conn->connection_id);
    packet.payload_size = strlen(packet.payload);
    
    ssize_t sent = send(conn->socket_fd, &packet, sizeof(packet), MSG_NOSIGNAL);
    if (sent != sizeof(packet)) {
        gpu_log(GPU_LOG_ERROR, "Failed to send heartbeat to connection %d: %s", 
                conn->connection_id, strerror(errno));
        return MONITOR_ERROR_SEND_FAILED;
    }
    
    conn->last_heartbeat_sent = packet.timestamp;
    gpu_log(GPU_LOG_DEBUG, "Heartbeat sent to connection %d, seq=%u", 
            conn->connection_id, packet.sequence);
    
    return MONITOR_SUCCESS;
}

// 接收心跳响应
static monitor_result_t receive_heartbeat_response(connection_info_t *conn, int timeout_ms) {
    if (!conn || conn->socket_fd < 0) {
        return MONITOR_ERROR_INVALID_CONNECTION;
    }
    
    // 设置接收超时
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    
    if (setsockopt(conn->socket_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        gpu_log(GPU_LOG_WARNING, "Failed to set receive timeout: %s", strerror(errno));
    }
    
    heartbeat_packet_t response;
    ssize_t received = recv(conn->socket_fd, &response, sizeof(response), 0);
    
    if (received != sizeof(response)) {
        if (received == 0) {
            gpu_log(GPU_LOG_WARNING, "Connection %d closed by remote", conn->connection_id);
            return MONITOR_ERROR_CONNECTION_CLOSED;
        } else if (received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                gpu_log(GPU_LOG_WARNING, "Heartbeat response timeout for connection %d", 
                        conn->connection_id);
                return MONITOR_ERROR_TIMEOUT;
            } else {
                gpu_log(GPU_LOG_ERROR, "Failed to receive heartbeat response: %s", 
                        strerror(errno));
                return MONITOR_ERROR_RECEIVE_FAILED;
            }
        }
    }
    
    // 验证响应
    if (response.magic != HEARTBEAT_MAGIC) {
        gpu_log(GPU_LOG_ERROR, "Invalid heartbeat response magic: 0x%x", response.magic);
        return MONITOR_ERROR_INVALID_RESPONSE;
    }
    
    uint64_t now = gpu_get_timestamp_us();
    uint64_t rtt = now - conn->last_heartbeat_sent;
    
    conn->last_heartbeat_received = now;
    conn->last_rtt_us = rtt;
    conn->consecutive_failures = 0;
    
    gpu_log(GPU_LOG_DEBUG, "Heartbeat response received from connection %d, RTT=%lu us", 
            conn->connection_id, rtt);
    
    return MONITOR_SUCCESS;
}

// 检查单个连接
static void check_connection(connection_info_t *conn) {
    if (!conn || conn->status == CONNECTION_STATUS_DISCONNECTED) {
        return;
    }
    
    uint64_t now = gpu_get_timestamp_us();
    uint64_t time_since_last = now - conn->last_heartbeat_sent;
    
    // 检查是否需要发送心跳
    if (time_since_last >= g_monitor.config.heartbeat_interval_ms * 1000) {
        monitor_result_t result = send_heartbeat(conn);
        
        if (result == MONITOR_SUCCESS) {
            // 等待响应
            result = receive_heartbeat_response(conn, g_monitor.config.heartbeat_timeout_ms);
        }
        
        if (result != MONITOR_SUCCESS) {
            conn->consecutive_failures++;
            gpu_log(GPU_LOG_WARNING, "Connection %d heartbeat failed (attempt %d/%d): %s", 
                    conn->connection_id, conn->consecutive_failures, 
                    g_monitor.config.max_failures, monitor_get_error_string(result));
            
            // 检查是否超过最大失败次数
            if (conn->consecutive_failures >= g_monitor.config.max_failures) {
                gpu_log(GPU_LOG_ERROR, "Connection %d marked as failed after %d consecutive failures", 
                        conn->connection_id, conn->consecutive_failures);
                conn->status = CONNECTION_STATUS_FAILED;
                
                // 调用故障回调
                if (g_monitor.config.failure_callback) {
                    g_monitor.config.failure_callback(conn->connection_id, 
                                                     FAILURE_TYPE_HEARTBEAT_TIMEOUT);
                }
            }
        } else {
            // 心跳成功，更新状态
            if (conn->status != CONNECTION_STATUS_CONNECTED) {
                gpu_log(GPU_LOG_INFO, "Connection %d recovered", conn->connection_id);
                conn->status = CONNECTION_STATUS_CONNECTED;
                
                // 调用恢复回调
                if (g_monitor.config.recovery_callback) {
                    g_monitor.config.recovery_callback(conn->connection_id);
                }
            }
        }
    }
}

// 监控线程主函数
static void* monitor_thread_func(void *arg) {
    gpu_log(GPU_LOG_INFO, "Connection monitor thread started");
    
    while (g_monitor_running) {
        pthread_mutex_lock(&g_monitor_mutex);
        
        // 检查所有连接
        for (int i = 0; i < g_monitor.connection_count; i++) {
            check_connection(&g_monitor.connections[i]);
        }
        
        pthread_mutex_unlock(&g_monitor_mutex);
        
        // 休眠一段时间
        usleep(g_monitor.config.check_interval_ms * 1000);
    }
    
    gpu_log(GPU_LOG_INFO, "Connection monitor thread stopped");
    return NULL;
}

// 初始化连接监控器
monitor_result_t connection_monitor_init(const heartbeat_config_t *config) {
    if (!config) {
        return MONITOR_ERROR_INVALID_PARAM;
    }
    
    if (g_monitor_running) {
        gpu_log(GPU_LOG_WARNING, "Connection monitor already running");
        return MONITOR_SUCCESS;
    }
    
    memset(&g_monitor, 0, sizeof(g_monitor));
    g_monitor.config = *config;
    
    // 验证配置参数
    if (g_monitor.config.heartbeat_interval_ms == 0) {
        g_monitor.config.heartbeat_interval_ms = 5000; // 默认5秒
    }
    if (g_monitor.config.heartbeat_timeout_ms == 0) {
        g_monitor.config.heartbeat_timeout_ms = 2000; // 默认2秒
    }
    if (g_monitor.config.max_failures == 0) {
        g_monitor.config.max_failures = 3; // 默认3次
    }
    if (g_monitor.config.check_interval_ms == 0) {
        g_monitor.config.check_interval_ms = 1000; // 默认1秒
    }
    
    // 启动监控线程
    g_monitor_running = true;
    if (pthread_create(&g_monitor_thread, NULL, monitor_thread_func, NULL) != 0) {
        gpu_log(GPU_LOG_ERROR, "Failed to create monitor thread: %s", strerror(errno));
        g_monitor_running = false;
        return MONITOR_ERROR_THREAD_CREATE_FAILED;
    }
    
    gpu_log(GPU_LOG_INFO, "Connection monitor initialized successfully");
    return MONITOR_SUCCESS;
}

// 清理连接监控器
void connection_monitor_cleanup(void) {
    if (!g_monitor_running) {
        return;
    }
    
    gpu_log(GPU_LOG_INFO, "Stopping connection monitor...");
    
    g_monitor_running = false;
    
    // 等待监控线程结束
    if (pthread_join(g_monitor_thread, NULL) != 0) {
        gpu_log(GPU_LOG_WARNING, "Failed to join monitor thread");
    }
    
    pthread_mutex_lock(&g_monitor_mutex);
    memset(&g_monitor, 0, sizeof(g_monitor));
    pthread_mutex_unlock(&g_monitor_mutex);
    
    gpu_log(GPU_LOG_INFO, "Connection monitor stopped");
}

// 添加连接监控
monitor_result_t connection_monitor_add(int connection_id, int socket_fd, 
                                       const char *remote_address) {
    if (connection_id < 0 || socket_fd < 0 || !remote_address) {
        return MONITOR_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_monitor_mutex);
    
    // 检查是否已存在
    for (int i = 0; i < g_monitor.connection_count; i++) {
        if (g_monitor.connections[i].connection_id == connection_id) {
            pthread_mutex_unlock(&g_monitor_mutex);
            gpu_log(GPU_LOG_WARNING, "Connection %d already exists", connection_id);
            return MONITOR_ERROR_CONNECTION_EXISTS;
        }
    }
    
    // 检查容量
    if (g_monitor.connection_count >= MAX_MONITORED_CONNECTIONS) {
        pthread_mutex_unlock(&g_monitor_mutex);
        gpu_log(GPU_LOG_ERROR, "Maximum monitored connections reached");
        return MONITOR_ERROR_MAX_CONNECTIONS;
    }
    
    // 添加新连接
    connection_info_t *conn = &g_monitor.connections[g_monitor.connection_count];
    conn->connection_id = connection_id;
    conn->socket_fd = socket_fd;
    strncpy(conn->remote_address, remote_address, sizeof(conn->remote_address) - 1);
    conn->status = CONNECTION_STATUS_CONNECTED;
    conn->last_heartbeat_sent = 0;
    conn->last_heartbeat_received = 0;
    conn->heartbeat_sequence = 0;
    conn->consecutive_failures = 0;
    conn->last_rtt_us = 0;
    
    g_monitor.connection_count++;
    
    pthread_mutex_unlock(&g_monitor_mutex);
    
    gpu_log(GPU_LOG_INFO, "Added connection %d (%s) to monitor", 
            connection_id, remote_address);
    
    return MONITOR_SUCCESS;
}

// 移除连接监控
monitor_result_t connection_monitor_remove(int connection_id) {
    pthread_mutex_lock(&g_monitor_mutex);
    
    // 查找连接
    int found_index = -1;
    for (int i = 0; i < g_monitor.connection_count; i++) {
        if (g_monitor.connections[i].connection_id == connection_id) {
            found_index = i;
            break;
        }
    }
    
    if (found_index == -1) {
        pthread_mutex_unlock(&g_monitor_mutex);
        return MONITOR_ERROR_CONNECTION_NOT_FOUND;
    }
    
    // 移除连接（移动后续元素）
    for (int i = found_index; i < g_monitor.connection_count - 1; i++) {
        g_monitor.connections[i] = g_monitor.connections[i + 1];
    }
    g_monitor.connection_count--;
    
    pthread_mutex_unlock(&g_monitor_mutex);
    
    gpu_log(GPU_LOG_INFO, "Removed connection %d from monitor", connection_id);
    
    return MONITOR_SUCCESS;
}

// 获取连接状态
monitor_result_t connection_monitor_get_status(int connection_id, 
                                              connection_status_t *status) {
    if (!status) {
        return MONITOR_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_monitor_mutex);
    
    // 查找连接
    connection_info_t *conn = NULL;
    for (int i = 0; i < g_monitor.connection_count; i++) {
        if (g_monitor.connections[i].connection_id == connection_id) {
            conn = &g_monitor.connections[i];
            break;
        }
    }
    
    if (!conn) {
        pthread_mutex_unlock(&g_monitor_mutex);
        return MONITOR_ERROR_CONNECTION_NOT_FOUND;
    }
    
    *status = conn->status;
    
    pthread_mutex_unlock(&g_monitor_mutex);
    
    return MONITOR_SUCCESS;
}

// 获取连接统计信息
monitor_result_t connection_monitor_get_stats(int connection_id, 
                                             connection_stats_t *stats) {
    if (!stats) {
        return MONITOR_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_monitor_mutex);
    
    // 查找连接
    connection_info_t *conn = NULL;
    for (int i = 0; i < g_monitor.connection_count; i++) {
        if (g_monitor.connections[i].connection_id == connection_id) {
            conn = &g_monitor.connections[i];
            break;
        }
    }
    
    if (!conn) {
        pthread_mutex_unlock(&g_monitor_mutex);
        return MONITOR_ERROR_CONNECTION_NOT_FOUND;
    }
    
    stats->last_heartbeat_sent = conn->last_heartbeat_sent;
    stats->last_heartbeat_received = conn->last_heartbeat_received;
    stats->consecutive_failures = conn->consecutive_failures;
    stats->last_rtt_us = conn->last_rtt_us;
    
    pthread_mutex_unlock(&g_monitor_mutex);
    
    return MONITOR_SUCCESS;
}

// 获取错误字符串
const char* monitor_get_error_string(monitor_result_t error) {
    static const char* error_strings[] = {
        [MONITOR_SUCCESS] = "Success",
        [MONITOR_ERROR_INVALID_PARAM] = "Invalid parameter",
        [MONITOR_ERROR_INVALID_CONNECTION] = "Invalid connection",
        [MONITOR_ERROR_CONNECTION_EXISTS] = "Connection already exists",
        [MONITOR_ERROR_CONNECTION_NOT_FOUND] = "Connection not found",
        [MONITOR_ERROR_MAX_CONNECTIONS] = "Maximum connections reached",
        [MONITOR_ERROR_SEND_FAILED] = "Send failed",
        [MONITOR_ERROR_RECEIVE_FAILED] = "Receive failed",
        [MONITOR_ERROR_TIMEOUT] = "Timeout",
        [MONITOR_ERROR_CONNECTION_CLOSED] = "Connection closed",
        [MONITOR_ERROR_INVALID_RESPONSE] = "Invalid response",
        [MONITOR_ERROR_THREAD_CREATE_FAILED] = "Thread creation failed"
    };
    
    if (error >= 0 && error < sizeof(error_strings) / sizeof(error_strings[0])) {
        return error_strings[error];
    }
    return "Unknown error";
}