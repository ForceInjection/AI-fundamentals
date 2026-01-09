/**
 * 远程GPU调用完整示例
 * 演示客户端-服务器架构的远程GPU调用实现
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include "../common/gpu_common.h"
#include "../remote/remote_gpu_protocol.c"

#define SERVER_PORT 8888
#define MAX_CLIENTS 10
#define BUFFER_SIZE 4096

// 远程调用上下文
typedef struct {
    int socket_fd;
    struct sockaddr_in server_addr;
    int connected;
    pthread_mutex_t send_mutex;
    uint32_t message_id_counter;
} remote_context_t;

// 全局远程上下文
static remote_context_t g_remote_ctx;
static int g_server_mode = 0;

/**
 * 初始化远程上下文
 */
int init_remote_context(const char *server_ip, int port) {
    memset(&g_remote_ctx, 0, sizeof(g_remote_ctx));
    
    g_remote_ctx.socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_remote_ctx.socket_fd < 0) {
        perror("socket creation failed");
        return -1;
    }
    
    g_remote_ctx.server_addr.sin_family = AF_INET;
    g_remote_ctx.server_addr.sin_port = htons(port);
    
    if (server_ip) {
        inet_pton(AF_INET, server_ip, &g_remote_ctx.server_addr.sin_addr);
    } else {
        g_remote_ctx.server_addr.sin_addr.s_addr = INADDR_ANY;
    }
    
    if (pthread_mutex_init(&g_remote_ctx.send_mutex, NULL) != 0) {
        close(g_remote_ctx.socket_fd);
        return -1;
    }
    
    g_remote_ctx.message_id_counter = 1;
    
    return 0;
}

/**
 * 连接到远程GPU服务器
 */
int connect_to_server() {
    if (connect(g_remote_ctx.socket_fd, 
                (struct sockaddr*)&g_remote_ctx.server_addr, 
                sizeof(g_remote_ctx.server_addr)) < 0) {
        perror("connection failed");
        return -1;
    }
    
    g_remote_ctx.connected = 1;
    printf("Connected to remote GPU server\n");
    
    return 0;
}

/**
 * 发送远程请求
 */
int send_remote_request(enum message_type msg_type, const void *data, size_t data_size) {
    if (!g_remote_ctx.connected) {
        printf("Not connected to server\n");
        return -1;
    }
    
    pthread_mutex_lock(&g_remote_ctx.send_mutex);
    
    // 构建消息头
    message_header_t header;
    header.magic = PROTOCOL_MAGIC;
    header.version = PROTOCOL_VERSION;
    header.message_type = msg_type;
    header.message_id = g_remote_ctx.message_id_counter++;
    header.timestamp = gpu_get_timestamp_us();
    header.payload_size = data_size;
    header.checksum = 0; // 简化实现，实际应计算校验和
    
    // 发送消息头
    if (send(g_remote_ctx.socket_fd, &header, sizeof(header), 0) < 0) {
        perror("send header failed");
        pthread_mutex_unlock(&g_remote_ctx.send_mutex);
        return -1;
    }
    
    // 发送数据
    if (data_size > 0 && data) {
        if (send(g_remote_ctx.socket_fd, data, data_size, 0) < 0) {
            perror("send data failed");
            pthread_mutex_unlock(&g_remote_ctx.send_mutex);
            return -1;
        }
    }
    
    pthread_mutex_unlock(&g_remote_ctx.send_mutex);
    
    printf("Sent request: type=%d, id=%d, size=%zu\n", 
           msg_type, header.message_id, data_size);
    
    return header.message_id;
}

/**
 * 接收远程响应
 */
int receive_remote_response(void *buffer, size_t buffer_size) {
    if (!g_remote_ctx.connected) {
        return -1;
    }
    
    // 接收消息头
    message_header_t header;
    ssize_t received = recv(g_remote_ctx.socket_fd, &header, sizeof(header), 0);
    if (received != sizeof(header)) {
        printf("Failed to receive response header\n");
        return -1;
    }
    
    // 验证消息
    if (header.magic != PROTOCOL_MAGIC || header.version != PROTOCOL_VERSION) {
        printf("Invalid response header\n");
        return -1;
    }
    
    // 接收数据
    if (header.payload_size > 0) {
        if (header.payload_size > buffer_size) {
            printf("Response too large for buffer\n");
            return -1;
        }
        
        received = recv(g_remote_ctx.socket_fd, buffer, header.payload_size, 0);
        if (received != header.payload_size) {
            printf("Failed to receive response data\n");
            return -1;
        }
    }
    
    printf("Received response: type=%d, id=%d, size=%d\n", 
           header.message_type, header.message_id, header.payload_size);
    
    return header.payload_size;
}

/**
 * 远程cudaMalloc实现
 */
void* remote_cuda_malloc(size_t size) {
    // 构建请求数据
    struct {
        size_t size;
    } request = { size };
    
    // 发送请求
    int msg_id = send_remote_request(MSG_CUDA_MALLOC, &request, sizeof(request));
    if (msg_id < 0) {
        return NULL;
    }
    
    // 接收响应
    struct {
        void *ptr;
        int error_code;
    } response;
    
    if (receive_remote_response(&response, sizeof(response)) < 0) {
        return NULL;
    }
    
    if (response.error_code != 0) {
        printf("Remote cudaMalloc failed with error %d\n", response.error_code);
        return NULL;
    }
    
    printf("Remote allocated %zu bytes at %p\n", size, response.ptr);
    return response.ptr;
}

/**
 * 远程cudaFree实现
 */
int remote_cuda_free(void *ptr) {
    // 构建请求数据
    struct {
        void *ptr;
    } request = { ptr };
    
    // 发送请求
    int msg_id = send_remote_request(MSG_CUDA_FREE, &request, sizeof(request));
    if (msg_id < 0) {
        return -1;
    }
    
    // 接收响应
    struct {
        int error_code;
    } response;
    
    if (receive_remote_response(&response, sizeof(response)) < 0) {
        return -1;
    }
    
    if (response.error_code != 0) {
        printf("Remote cudaFree failed with error %d\n", response.error_code);
        return -1;
    }
    
    printf("Remote freed memory at %p\n", ptr);
    return 0;
}

/**
 * 远程cudaMemcpy实现
 */
int remote_cuda_memcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    // 构建请求数据
    struct {
        void *dst;
        void *src;
        size_t count;
        enum cudaMemcpyKind kind;
        char data[]; // 如果是Host to Device，数据跟在后面
    } *request;
    
    size_t request_size = sizeof(*request);
    if (kind == cudaMemcpyHostToDevice) {
        request_size += count;
    }
    
    request = malloc(request_size);
    if (!request) {
        return -1;
    }
    
    request->dst = dst;
    request->src = (void*)src;
    request->count = count;
    request->kind = kind;
    
    // 如果是Host to Device，复制数据
    if (kind == cudaMemcpyHostToDevice) {
        memcpy(request->data, src, count);
    }
    
    // 发送请求
    int msg_id = send_remote_request(MSG_CUDA_MEMCPY, request, request_size);
    free(request);
    
    if (msg_id < 0) {
        return -1;
    }
    
    // 接收响应
    char response_buffer[BUFFER_SIZE];
    int response_size = receive_remote_response(response_buffer, sizeof(response_buffer));
    if (response_size < 0) {
        return -1;
    }
    
    struct {
        int error_code;
        char data[];
    } *response = (void*)response_buffer;
    
    if (response->error_code != 0) {
        printf("Remote cudaMemcpy failed with error %d\n", response->error_code);
        return -1;
    }
    
    // 如果是Device to Host，复制返回的数据
    if (kind == cudaMemcpyDeviceToHost && response_size > sizeof(int)) {
        memcpy((void*)dst, response->data, count);
    }
    
    printf("Remote memcpy: %zu bytes, kind=%d\n", count, kind);
    return 0;
}

/**
 * 处理客户端请求（服务器端）
 */
void* handle_client(void *arg) {
    int client_fd = *(int*)arg;
    free(arg);
    
    printf("Handling client connection: %d\n", client_fd);
    
    while (1) {
        // 接收消息头
        message_header_t header;
        ssize_t received = recv(client_fd, &header, sizeof(header), 0);
        if (received != sizeof(header)) {
            printf("Client disconnected or error\n");
            break;
        }
        
        // 验证消息
        if (header.magic != PROTOCOL_MAGIC || header.version != PROTOCOL_VERSION) {
            printf("Invalid message header\n");
            break;
        }
        
        // 接收数据
        char *data = NULL;
        if (header.payload_size > 0) {
            data = malloc(header.payload_size);
            if (!data) {
                break;
            }
            
            received = recv(client_fd, data, header.payload_size, 0);
            if (received != header.payload_size) {
                free(data);
                break;
            }
        }
        
        printf("Received request: type=%d, id=%d, size=%d\n", 
               header.message_type, header.message_id, header.payload_size);
        
        // 处理请求
        switch (header.message_type) {
            case MSG_CUDA_MALLOC: {
                struct {
                    size_t size;
                } *request = (void*)data;
                
                void *ptr;
                cudaError_t err = cudaMalloc(&ptr, request->size);
                
                struct {
                    void *ptr;
                    int error_code;
                } response = { ptr, err };
                
                // 发送响应
                message_header_t resp_header = {
                    .magic = PROTOCOL_MAGIC,
                    .version = PROTOCOL_VERSION,
                    .message_type = MSG_RESPONSE,
                    .message_id = header.message_id,
                    .timestamp = gpu_get_timestamp_us(),
                    .payload_size = sizeof(response),
                    .checksum = 0
                };
                
                send(client_fd, &resp_header, sizeof(resp_header), 0);
                send(client_fd, &response, sizeof(response), 0);
                
                printf("Allocated %zu bytes at %p (error=%d)\n", 
                       request->size, ptr, err);
                break;
            }
            
            case MSG_CUDA_FREE: {
                struct {
                    void *ptr;
                } *request = (void*)data;
                
                cudaError_t err = cudaFree(request->ptr);
                
                struct {
                    int error_code;
                } response = { err };
                
                // 发送响应
                message_header_t resp_header = {
                    .magic = PROTOCOL_MAGIC,
                    .version = PROTOCOL_VERSION,
                    .message_type = MSG_RESPONSE,
                    .message_id = header.message_id,
                    .timestamp = gpu_get_timestamp_us(),
                    .payload_size = sizeof(response),
                    .checksum = 0
                };
                
                send(client_fd, &resp_header, sizeof(resp_header), 0);
                send(client_fd, &response, sizeof(response), 0);
                
                printf("Freed memory at %p (error=%d)\n", request->ptr, err);
                break;
            }
            
            case MSG_CUDA_MEMCPY: {
                struct {
                    void *dst;
                    void *src;
                    size_t count;
                    enum cudaMemcpyKind kind;
                    char data[];
                } *request = (void*)data;
                
                cudaError_t err = cudaSuccess;
                
                if (request->kind == cudaMemcpyHostToDevice) {
                    err = cudaMemcpy(request->dst, request->data, request->count, request->kind);
                } else if (request->kind == cudaMemcpyDeviceToHost) {
                    // 需要分配临时缓冲区
                    char *temp_buffer = malloc(request->count);
                    if (temp_buffer) {
                        err = cudaMemcpy(temp_buffer, request->src, request->count, request->kind);
                    } else {
                        err = cudaErrorMemoryAllocation;
                    }
                    
                    // 发送响应（包含数据）
                    size_t response_size = sizeof(int) + (err == cudaSuccess ? request->count : 0);
                    char *response_buffer = malloc(response_size);
                    *(int*)response_buffer = err;
                    
                    if (err == cudaSuccess && temp_buffer) {
                        memcpy(response_buffer + sizeof(int), temp_buffer, request->count);
                    }
                    
                    message_header_t resp_header = {
                        .magic = PROTOCOL_MAGIC,
                        .version = PROTOCOL_VERSION,
                        .message_type = MSG_RESPONSE,
                        .message_id = header.message_id,
                        .timestamp = gpu_get_timestamp_us(),
                        .payload_size = response_size,
                        .checksum = 0
                    };
                    
                    send(client_fd, &resp_header, sizeof(resp_header), 0);
                    send(client_fd, response_buffer, response_size, 0);
                    
                    free(temp_buffer);
                    free(response_buffer);
                    
                    printf("Memcpy: %zu bytes, kind=%d (error=%d)\n", 
                           request->count, request->kind, err);
                    
                    if (data) free(data);
                    continue;
                } else {
                    err = cudaMemcpy(request->dst, request->src, request->count, request->kind);
                }
                
                struct {
                    int error_code;
                } response = { err };
                
                // 发送响应
                message_header_t resp_header = {
                    .magic = PROTOCOL_MAGIC,
                    .version = PROTOCOL_VERSION,
                    .message_type = MSG_RESPONSE,
                    .message_id = header.message_id,
                    .timestamp = gpu_get_timestamp_us(),
                    .payload_size = sizeof(response),
                    .checksum = 0
                };
                
                send(client_fd, &resp_header, sizeof(resp_header), 0);
                send(client_fd, &response, sizeof(response), 0);
                
                printf("Memcpy: %zu bytes, kind=%d (error=%d)\n", 
                       request->count, request->kind, err);
                break;
            }
            
            default:
                printf("Unknown message type: %d\n", header.message_type);
                break;
        }
        
        if (data) {
            free(data);
        }
    }
    
    close(client_fd);
    printf("Client connection closed\n");
    return NULL;
}

/**
 * 启动GPU服务器
 */
int start_gpu_server(int port) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket creation failed");
        return -1;
    }
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        close(server_fd);
        return -1;
    }
    
    if (listen(server_fd, MAX_CLIENTS) < 0) {
        perror("listen failed");
        close(server_fd);
        return -1;
    }
    
    printf("GPU server listening on port %d\n", port);
    
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("accept failed");
            continue;
        }
        
        printf("New client connected: %s:%d\n", 
               inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
        
        // 创建线程处理客户端
        pthread_t client_thread;
        int *client_fd_ptr = malloc(sizeof(int));
        *client_fd_ptr = client_fd;
        
        if (pthread_create(&client_thread, NULL, handle_client, client_fd_ptr) != 0) {
            perror("pthread_create failed");
            close(client_fd);
            free(client_fd_ptr);
        } else {
            pthread_detach(client_thread);
        }
    }
    
    close(server_fd);
    return 0;
}



/**
 * 主函数
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <server|client> [server_ip]\n", argv[0]);
        return -1;
    }
    
    if (strcmp(argv[1], "server") == 0) {
        // 服务器模式
        g_server_mode = 1;
        printf("Starting GPU server...\n");
        return start_gpu_server(SERVER_PORT);
    } else if (strcmp(argv[1], "client") == 0) {
        // 客户端模式
        const char *server_ip = (argc > 2) ? argv[2] : "127.0.0.1";
        
        if (init_remote_context(server_ip, SERVER_PORT) != 0) {
            printf("Failed to initialize remote context\n");
            return -1;
        }
        
        printf("Client mode initialized\n");
        
        // 清理
        if (g_remote_ctx.connected) {
            close(g_remote_ctx.socket_fd);
        }
        pthread_mutex_destroy(&g_remote_ctx.send_mutex);
        
        return 0;
    } else {
        printf("Invalid mode. Use 'server' or 'client'\n");
        return -1;
    }
}