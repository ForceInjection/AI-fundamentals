/*
 * 远程GPU调用客户端实现
 * 负责与远程GPU服务器通信，实现透明的远程GPU访问
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <dlfcn.h>
#include <cuda_runtime.h>

// 远程GPU协议定义
#define REMOTE_GPU_MAGIC 0x52474055  // "RGPU"
#define REMOTE_GPU_VERSION 1
#define MAX_MESSAGE_SIZE 1048576     // 1MB

// 消息类型
enum message_type {
    MSG_CUDA_MALLOC = 1,
    MSG_CUDA_FREE,
    MSG_CUDA_MEMCPY,
    MSG_CUDA_KERNEL_LAUNCH,
    MSG_CUDA_DEVICE_SYNC,
    MSG_CUDA_GET_DEVICE_PROPS
};

// 函数ID
enum function_id {
    FUNC_CUDA_MALLOC = 1,
    FUNC_CUDA_FREE,
    FUNC_CUDA_MEMCPY_H2D,
    FUNC_CUDA_MEMCPY_D2H,
    FUNC_CUDA_MEMCPY_D2D,
    FUNC_CUDA_LAUNCH_KERNEL,
    FUNC_CUDA_DEVICE_SYNC
};

// 消息头结构
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t message_type;
    uint32_t message_id;
    uint64_t timestamp;
    uint32_t payload_size;
    uint32_t checksum;
} message_header_t;

// 远程请求结构
typedef struct {
    message_header_t header;
    uint32_t function_id;
    uint32_t param_count;
    uint8_t payload[MAX_MESSAGE_SIZE];
} remote_request_t;

// 远程响应结构
typedef struct {
    message_header_t header;
    int32_t status;
    uint32_t result_size;
    void *result_data;
    uint8_t payload[MAX_MESSAGE_SIZE];
} remote_response_t;

// 远程客户端结构
typedef struct {
    int socket_fd;
    struct sockaddr_in server_addr;
    pthread_mutex_t send_mutex;
    pthread_mutex_t recv_mutex;
    uint32_t next_message_id;
    int connected;
    char server_host[256];
    int server_port;
} remote_client_t;

// API拦截表项
typedef struct {
    const char *function_name;
    void *original_function;
    void *intercept_function;
    bool is_intercepted;
} api_intercept_entry_t;

// API拦截器结构
typedef struct {
    void *original_lib_handle;
    remote_client_t *remote_client;
    bool intercept_enabled;
    pthread_mutex_t intercept_mutex;
} api_interceptor_t;

// 全局变量
static api_interceptor_t *global_interceptor = NULL;

// CUDA API拦截表
static api_intercept_entry_t cuda_intercept_table[] = {
    {"cudaMalloc", NULL, (void*)intercept_cudaMalloc, false},
    {"cudaFree", NULL, (void*)intercept_cudaFree, false},
    {"cudaMemcpy", NULL, (void*)intercept_cudaMemcpy, false},
    {"cudaDeviceSynchronize", NULL, (void*)intercept_cudaDeviceSynchronize, false},
    {NULL, NULL, NULL, false}
};

// 函数声明
remote_client_t* create_remote_client(const char *server_host, int server_port);
void destroy_remote_client(remote_client_t *client);
int connect_to_server(remote_client_t *client);
void disconnect_from_server(remote_client_t *client);
remote_response_t* send_remote_request(remote_client_t *client, remote_request_t *request);
int serialize_param(remote_request_t *request, int param_index, void *data, size_t size);
void* deserialize_result(remote_response_t *response, size_t *size);
uint32_t generate_message_id(void);
uint64_t get_timestamp_us(void);
uint32_t calculate_checksum(void *data, size_t size);
int init_api_interceptor(const char *server_host, int server_port);
void cleanup_api_interceptor(void);
void* get_original_function(const char *function_name);

// CUDA API拦截函数声明
cudaError_t intercept_cudaMalloc(void **devPtr, size_t size);
cudaError_t intercept_cudaFree(void *devPtr);
cudaError_t intercept_cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
cudaError_t intercept_cudaDeviceSynchronize(void);

// 远程内存操作函数声明
cudaError_t remote_memcpy_h2d(remote_client_t *client, void *dst, const void *src, size_t count);
cudaError_t remote_memcpy_d2h(remote_client_t *client, void *dst, const void *src, size_t count);
cudaError_t remote_memcpy_d2d(remote_client_t *client, void *dst, const void *src, size_t count);

// 创建远程客户端
remote_client_t* create_remote_client(const char *server_host, int server_port) {
    remote_client_t *client = malloc(sizeof(remote_client_t));
    if (!client) {
        return NULL;
    }
    
    memset(client, 0, sizeof(remote_client_t));
    strncpy(client->server_host, server_host, sizeof(client->server_host) - 1);
    client->server_port = server_port;
    client->socket_fd = -1;
    client->next_message_id = 1;
    
    pthread_mutex_init(&client->send_mutex, NULL);
    pthread_mutex_init(&client->recv_mutex, NULL);
    
    // 连接到服务器
    if (connect_to_server(client) != 0) {
        destroy_remote_client(client);
        return NULL;
    }
    
    return client;
}

// 销毁远程客户端
void destroy_remote_client(remote_client_t *client) {
    if (!client) return;
    
    disconnect_from_server(client);
    pthread_mutex_destroy(&client->send_mutex);
    pthread_mutex_destroy(&client->recv_mutex);
    free(client);
}

// 连接到服务器
int connect_to_server(remote_client_t *client) {
    if (!client) return -EINVAL;
    
    // 创建socket
    client->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client->socket_fd < 0) {
        return -errno;
    }
    
    // 设置服务器地址
    memset(&client->server_addr, 0, sizeof(client->server_addr));
    client->server_addr.sin_family = AF_INET;
    client->server_addr.sin_port = htons(client->server_port);
    
    if (inet_pton(AF_INET, client->server_host, &client->server_addr.sin_addr) <= 0) {
        close(client->socket_fd);
        client->socket_fd = -1;
        return -EINVAL;
    }
    
    // 连接到服务器
    if (connect(client->socket_fd, (struct sockaddr*)&client->server_addr, 
                sizeof(client->server_addr)) < 0) {
        close(client->socket_fd);
        client->socket_fd = -1;
        return -errno;
    }
    
    client->connected = 1;
    return 0;
}

// 断开服务器连接
void disconnect_from_server(remote_client_t *client) {
    if (!client || client->socket_fd < 0) return;
    
    close(client->socket_fd);
    client->socket_fd = -1;
    client->connected = 0;
}

// 发送远程请求
remote_response_t* send_remote_request(remote_client_t *client, remote_request_t *request) {
    if (!client || !request || !client->connected) {
        return NULL;
    }
    
    // 计算校验和
    request->header.checksum = calculate_checksum(request->payload, request->header.payload_size);
    
    pthread_mutex_lock(&client->send_mutex);
    
    // 发送请求
    size_t total_size = sizeof(message_header_t) + sizeof(uint32_t) * 2 + request->header.payload_size;
    ssize_t sent = send(client->socket_fd, request, total_size, 0);
    
    pthread_mutex_unlock(&client->send_mutex);
    
    if (sent != total_size) {
        return NULL;
    }
    
    // 接收响应
    remote_response_t *response = malloc(sizeof(remote_response_t));
    if (!response) {
        return NULL;
    }
    
    pthread_mutex_lock(&client->recv_mutex);
    
    // 接收响应头
    ssize_t received = recv(client->socket_fd, response, sizeof(message_header_t) + sizeof(int32_t) + sizeof(uint32_t), 0);
    if (received < sizeof(message_header_t) + sizeof(int32_t) + sizeof(uint32_t)) {
        pthread_mutex_unlock(&client->recv_mutex);
        free(response);
        return NULL;
    }
    
    // 接收响应数据
    if (response->result_size > 0) {
        received = recv(client->socket_fd, response->payload, response->result_size, 0);
        if (received != response->result_size) {
            pthread_mutex_unlock(&client->recv_mutex);
            free(response);
            return NULL;
        }
        response->result_data = response->payload;
    }
    
    pthread_mutex_unlock(&client->recv_mutex);
    
    return response;
}

// 序列化参数
int serialize_param(remote_request_t *request, int param_index, void *data, size_t size) {
    if (!request || !data || size == 0) {
        return -EINVAL;
    }
    
    if (request->header.payload_size + size > MAX_MESSAGE_SIZE) {
        return -ENOSPC;
    }
    
    memcpy(request->payload + request->header.payload_size, data, size);
    request->header.payload_size += size;
    
    return 0;
}

// 反序列化结果
void* deserialize_result(remote_response_t *response, size_t *size) {
    if (!response || !size) {
        return NULL;
    }
    
    *size = response->result_size;
    return response->result_data;
}

// 生成消息ID
uint32_t generate_message_id(void) {
    static uint32_t message_id = 0;
    return __sync_add_and_fetch(&message_id, 1);
}

// 获取时间戳（微秒）
uint64_t get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

// 计算校验和
uint32_t calculate_checksum(void *data, size_t size) {
    uint32_t checksum = 0;
    uint8_t *bytes = (uint8_t*)data;
    
    for (size_t i = 0; i < size; i++) {
        checksum += bytes[i];
    }
    
    return checksum;
}

// 初始化API拦截器
int init_api_interceptor(const char *server_host, int server_port) {
    global_interceptor = malloc(sizeof(api_interceptor_t));
    if (!global_interceptor) {
        return -ENOMEM;
    }
    
    memset(global_interceptor, 0, sizeof(api_interceptor_t));
    
    // 加载原始CUDA库
    global_interceptor->original_lib_handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!global_interceptor->original_lib_handle) {
        free(global_interceptor);
        return -1;
    }
    
    // 初始化远程客户端
    global_interceptor->remote_client = create_remote_client(server_host, server_port);
    if (!global_interceptor->remote_client) {
        dlclose(global_interceptor->original_lib_handle);
        free(global_interceptor);
        return -1;
    }
    
    // 设置函数拦截
    for (int i = 0; cuda_intercept_table[i].function_name; i++) {
        cuda_intercept_table[i].original_function = 
            dlsym(global_interceptor->original_lib_handle, cuda_intercept_table[i].function_name);
        
        if (cuda_intercept_table[i].original_function) {
            cuda_intercept_table[i].is_intercepted = true;
        }
    }
    
    global_interceptor->intercept_enabled = true;
    pthread_mutex_init(&global_interceptor->intercept_mutex, NULL);
    
    return 0;
}

// 清理API拦截器
void cleanup_api_interceptor(void) {
    if (!global_interceptor) return;
    
    global_interceptor->intercept_enabled = false;
    
    if (global_interceptor->remote_client) {
        destroy_remote_client(global_interceptor->remote_client);
    }
    
    if (global_interceptor->original_lib_handle) {
        dlclose(global_interceptor->original_lib_handle);
    }
    
    pthread_mutex_destroy(&global_interceptor->intercept_mutex);
    free(global_interceptor);
    global_interceptor = NULL;
}

// 获取原始函数
void* get_original_function(const char *function_name) {
    if (!global_interceptor || !function_name) {
        return NULL;
    }
    
    for (int i = 0; cuda_intercept_table[i].function_name; i++) {
        if (strcmp(cuda_intercept_table[i].function_name, function_name) == 0) {
            return cuda_intercept_table[i].original_function;
        }
    }
    
    return NULL;
}

// cudaMalloc拦截实现
cudaError_t intercept_cudaMalloc(void **devPtr, size_t size) {
    if (!global_interceptor || !global_interceptor->intercept_enabled) {
        // 调用原始函数
        cudaError_t (*original_cudaMalloc)(void**, size_t) = 
            (cudaError_t (*)(void**, size_t))get_original_function("cudaMalloc");
        return original_cudaMalloc(devPtr, size);
    }
    
    // 创建远程调用请求
    remote_request_t request = {
        .header = {
            .magic = REMOTE_GPU_MAGIC,
            .version = REMOTE_GPU_VERSION,
            .message_type = MSG_CUDA_MALLOC,
            .message_id = generate_message_id(),
            .timestamp = get_timestamp_us(),
            .payload_size = sizeof(size_t)
        },
        .function_id = FUNC_CUDA_MALLOC,
        .param_count = 1
    };
    
    // 序列化参数
    serialize_param(&request, 0, &size, sizeof(size_t));
    
    // 发送远程请求
    remote_response_t *response = send_remote_request(global_interceptor->remote_client, &request);
    if (!response) {
        return cudaErrorMemoryAllocation;
    }
    
    // 解析响应
    if (response->status == 0) {
        *devPtr = response->result_data;
        free(response);
        return cudaSuccess;
    }
    
    cudaError_t error = (cudaError_t)response->status;
    free(response);
    return error;
}

// cudaFree拦截实现
cudaError_t intercept_cudaFree(void *devPtr) {
    if (!global_interceptor || !global_interceptor->intercept_enabled) {
        cudaError_t (*original_cudaFree)(void*) = 
            (cudaError_t (*)(void*))get_original_function("cudaFree");
        return original_cudaFree(devPtr);
    }
    
    remote_request_t request = {
        .header = {
            .magic = REMOTE_GPU_MAGIC,
            .version = REMOTE_GPU_VERSION,
            .message_type = MSG_CUDA_FREE,
            .message_id = generate_message_id(),
            .timestamp = get_timestamp_us(),
            .payload_size = sizeof(void*)
        },
        .function_id = FUNC_CUDA_FREE,
        .param_count = 1
    };
    
    serialize_param(&request, 0, &devPtr, sizeof(void*));
    
    remote_response_t *response = send_remote_request(global_interceptor->remote_client, &request);
    if (!response) {
        return cudaErrorInvalidValue;
    }
    
    cudaError_t error = (cudaError_t)response->status;
    free(response);
    return error;
}

// cudaMemcpy拦截实现
cudaError_t intercept_cudaMemcpy(void *dst, const void *src, size_t count, 
                                cudaMemcpyKind kind) {
    if (!global_interceptor || !global_interceptor->intercept_enabled) {
        cudaError_t (*original_cudaMemcpy)(void*, const void*, size_t, cudaMemcpyKind) = 
            (cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind))get_original_function("cudaMemcpy");
        return original_cudaMemcpy(dst, src, count, kind);
    }
    
    // 处理不同的内存拷贝类型
    switch (kind) {
        case cudaMemcpyHostToDevice:
            return remote_memcpy_h2d(global_interceptor->remote_client, dst, src, count);
        case cudaMemcpyDeviceToHost:
            return remote_memcpy_d2h(global_interceptor->remote_client, dst, src, count);
        case cudaMemcpyDeviceToDevice:
            return remote_memcpy_d2d(global_interceptor->remote_client, dst, src, count);
        default:
            return cudaErrorInvalidMemcpyDirection;
    }
}

// cudaDeviceSynchronize拦截实现
cudaError_t intercept_cudaDeviceSynchronize(void) {
    if (!global_interceptor || !global_interceptor->intercept_enabled) {
        cudaError_t (*original_cudaDeviceSynchronize)(void) = 
            (cudaError_t (*)(void))get_original_function("cudaDeviceSynchronize");
        return original_cudaDeviceSynchronize();
    }
    
    remote_request_t request = {
        .header = {
            .magic = REMOTE_GPU_MAGIC,
            .version = REMOTE_GPU_VERSION,
            .message_type = MSG_CUDA_DEVICE_SYNC,
            .message_id = generate_message_id(),
            .timestamp = get_timestamp_us(),
            .payload_size = 0
        },
        .function_id = FUNC_CUDA_DEVICE_SYNC,
        .param_count = 0
    };
    
    remote_response_t *response = send_remote_request(global_interceptor->remote_client, &request);
    if (!response) {
        return cudaErrorUnknown;
    }
    
    cudaError_t error = (cudaError_t)response->status;
    free(response);
    return error;
}

// 远程内存拷贝实现
cudaError_t remote_memcpy_h2d(remote_client_t *client, void *dst, const void *src, size_t count) {
    remote_request_t request = {
        .header = {
            .magic = REMOTE_GPU_MAGIC,
            .version = REMOTE_GPU_VERSION,
            .message_type = MSG_CUDA_MEMCPY,
            .message_id = generate_message_id(),
            .timestamp = get_timestamp_us(),
            .payload_size = sizeof(void*) * 2 + sizeof(size_t) + count
        },
        .function_id = FUNC_CUDA_MEMCPY_H2D,
        .param_count = 3
    };
    
    // 序列化参数
    serialize_param(&request, 0, &dst, sizeof(void*));
    serialize_param(&request, 1, &count, sizeof(size_t));
    serialize_param(&request, 2, (void*)src, count);
    
    remote_response_t *response = send_remote_request(client, &request);
    if (!response) {
        return cudaErrorMemoryAllocation;
    }
    
    cudaError_t error = (cudaError_t)response->status;
    free(response);
    return error;
}

cudaError_t remote_memcpy_d2h(remote_client_t *client, void *dst, const void *src, size_t count) {
    remote_request_t request = {
        .header = {
            .magic = REMOTE_GPU_MAGIC,
            .version = REMOTE_GPU_VERSION,
            .message_type = MSG_CUDA_MEMCPY,
            .message_id = generate_message_id(),
            .timestamp = get_timestamp_us(),
            .payload_size = sizeof(void*) * 2 + sizeof(size_t)
        },
        .function_id = FUNC_CUDA_MEMCPY_D2H,
        .param_count = 3
    };
    
    serialize_param(&request, 0, (void*)&src, sizeof(void*));
    serialize_param(&request, 1, &count, sizeof(size_t));
    
    remote_response_t *response = send_remote_request(client, &request);
    if (!response) {
        return cudaErrorMemoryAllocation;
    }
    
    if (response->status == 0 && response->result_size == count) {
        memcpy(dst, response->result_data, count);
    }
    
    cudaError_t error = (cudaError_t)response->status;
    free(response);
    return error;
}

cudaError_t remote_memcpy_d2d(remote_client_t *client, void *dst, const void *src, size_t count) {
    remote_request_t request = {
        .header = {
            .magic = REMOTE_GPU_MAGIC,
            .version = REMOTE_GPU_VERSION,
            .message_type = MSG_CUDA_MEMCPY,
            .message_id = generate_message_id(),
            .timestamp = get_timestamp_us(),
            .payload_size = sizeof(void*) * 2 + sizeof(size_t)
        },
        .function_id = FUNC_CUDA_MEMCPY_D2D,
        .param_count = 3
    };
    
    serialize_param(&request, 0, &dst, sizeof(void*));
    serialize_param(&request, 1, (void*)&src, sizeof(void*));
    serialize_param(&request, 2, &count, sizeof(size_t));
    
    remote_response_t *response = send_remote_request(client, &request);
    if (!response) {
        return cudaErrorMemoryAllocation;
    }
    
    cudaError_t error = (cudaError_t)response->status;
    free(response);
    return error;
}