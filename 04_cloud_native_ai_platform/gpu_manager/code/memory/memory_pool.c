/**
 * 显存池化管理示例
 * 演示显存池的创建、分配、释放和管理机制
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "memory_pool.h"
#include "../common/gpu_common.h"

// CUDA运行时模拟（用于演示）
#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#else
// CUDA函数模拟
typedef enum {
    cudaSuccess = 0
} cudaError_t;

static cudaError_t cudaMalloc(void **devPtr, size_t size) {
    *devPtr = malloc(size);
    printf("[模拟] 分配GPU内存: %zu bytes\n", size);
    return cudaSuccess;
}

static cudaError_t cudaFree(void *devPtr) {
    free(devPtr);
    printf("[模拟] 释放GPU内存\n");
    return cudaSuccess;
}

static cudaError_t cudaSetDevice(int device) {
    printf("[模拟] 设置CUDA设备: %d\n", device);
    return cudaSuccess;
}

static cudaError_t cudaGetDeviceCount(int *count) {
    *count = 1; // 模拟1个设备
    printf("[模拟] 获取设备数量: %d\n", *count);
    return cudaSuccess;
}

static const char* cudaGetErrorString(cudaError_t error) {
    return (error == cudaSuccess) ? "success" : "error";
}
#endif

// 使用头文件中的结构体定义

// 全局内存池
static memory_pool_t g_memory_pools[MAX_GPU_DEVICES];
static int g_pool_initialized[MAX_GPU_DEVICES] = {0};

// 函数前向声明
static void merge_free_blocks(memory_pool_t *pool);
static memory_block_t* find_free_block(memory_pool_t *pool, size_t size);
static void split_block(memory_block_t *block, size_t size);

/**
 * 初始化显存池
 */
gpu_result_t init_memory_pool(memory_pool_t *pool, int device_id, size_t pool_size) {
    if (!pool || device_id < 0) {
        return GPU_ERROR_INVALID_PARAM;
    }
    
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        printf("Failed to set device %d: %s\n", device_id, cudaGetErrorString(err));
        return GPU_ERROR_DEVICE_NOT_FOUND;
    }
    
    // 分配显存池
    err = cudaMalloc(&pool->base_ptr, pool_size);
    if (err != cudaSuccess) {
        printf("Failed to allocate memory pool: %s\n", cudaGetErrorString(err));
        return GPU_ERROR_OUT_OF_MEMORY;
    }
    
    pool->total_size = pool_size;
    pool->used_size = 0;
    pool->device_id = device_id;
    pool->initialized = true;
    
    // 创建初始空闲块
    pool->free_blocks = malloc(sizeof(memory_block_t));
    if (!pool->free_blocks) {
        cudaFree(pool->base_ptr);
        return GPU_ERROR_OUT_OF_MEMORY;
    }
    
    pool->free_blocks->ptr = pool->base_ptr;
    pool->free_blocks->size = pool_size;
    pool->free_blocks->is_free = true;
    pool->free_blocks->next = NULL;
    pool->free_blocks->prev = NULL;
    
    pool->used_blocks = NULL;
    
    printf("Memory pool initialized on device %d, size: %zu MB\n", 
           device_id, pool_size / (1024 * 1024));
    
    return GPU_SUCCESS;
}

/**
 * 从显存池分配内存
 */
void* pool_malloc(int device_id, size_t size) {
    if (device_id >= MAX_GPU_DEVICES || !g_pool_initialized[device_id]) {
        return NULL;
    }
    
    memory_pool_t *pool = &g_memory_pools[device_id];
    memory_block_t *current = pool->free_blocks;
    memory_block_t *prev = NULL;
    
    // 对齐到256字节边界
    size_t aligned_size = (size + 255) & ~255;
    
    // 查找合适的空闲块
    while (current) {
        if (current->is_free && current->size >= aligned_size) {
            // 找到合适的块
            if (current->size > aligned_size + sizeof(memory_block_t)) {
                // 分割块
                memory_block_t *new_block = malloc(sizeof(memory_block_t));
                new_block->ptr = (char*)current->ptr + aligned_size;
                new_block->size = current->size - aligned_size;
                new_block->is_free = 1;
                new_block->next = current->next;
                
                current->size = aligned_size;
                current->next = new_block;
            }
            
            current->is_free = 0;
            pool->used_size += current->size;
            
            printf("Allocated %zu bytes from pool (device %d)\n", aligned_size, device_id);
            return current->ptr;
        }
        
        prev = current;
        current = current->next;
    }
    
    printf("Failed to allocate %zu bytes from pool (device %d)\n", size, device_id);
    return NULL;
}

/**
 * 释放显存池中的内存
 */
int pool_free(int device_id, void *ptr) {
    if (device_id >= MAX_GPU_DEVICES || !g_pool_initialized[device_id] || !ptr) {
        return -1;
    }
    
    memory_pool_t *pool = &g_memory_pools[device_id];
    memory_block_t *current = pool->free_blocks;
    
    // 查找对应的块
    while (current) {
        if (current->ptr == ptr && !current->is_free) {
            current->is_free = 1;
            pool->used_size -= current->size;
            
            // 尝试合并相邻的空闲块
            merge_free_blocks(pool);
            
            printf("Freed %zu bytes to pool (device %d)\n", current->size, device_id);
            return 0;
        }
        current = current->next;
    }
    
    printf("Failed to free memory: pointer not found in pool\n");
    return -1;
}

/**
 * 合并相邻的空闲块
 */
static void merge_free_blocks(memory_pool_t *pool) {
    memory_block_t *current = pool->free_blocks;
    
    while (current && current->next) {
        if (current->is_free && current->next->is_free &&
            (char*)current->ptr + current->size == current->next->ptr) {
            // 合并块
            memory_block_t *next_block = current->next;
            current->size += next_block->size;
            current->next = next_block->next;
            free(next_block);
        } else {
            current = current->next;
        }
    }
}

/**
 * 获取显存池统计信息
 */
void get_pool_stats(int device_id, size_t *total, size_t *used, size_t *free) {
    if (device_id >= MAX_GPU_DEVICES || !g_pool_initialized[device_id]) {
        *total = *used = *free = 0;
        return;
    }
    
    memory_pool_t *pool = &g_memory_pools[device_id];
    *total = pool->total_size;
    *used = pool->used_size;
    *free = pool->total_size - pool->used_size;
}

// 实现头文件中声明的函数
void* allocate_memory(memory_pool_t *pool, size_t size) {
    if (!pool || !pool->initialized) {
        return NULL;
    }
    
    // 查找合适的空闲块
    memory_block_t *current = pool->free_blocks;
    while (current) {
        if (current->is_free && current->size >= size) {
            current->is_free = 0;
            pool->used_size += current->size;
            return current->ptr;
        }
        current = current->next;
    }
    
    return NULL;
}

gpu_result_t free_memory(memory_pool_t *pool, void *ptr) {
    if (!pool || !pool->initialized || !ptr) {
        return GPU_ERROR_INVALID_PARAM;
    }
    
    // 查找对应的内存块
    memory_block_t *current = pool->free_blocks;
    while (current) {
        if (current->ptr == ptr) {
            current->is_free = 1;
            pool->used_size -= current->size;
            return GPU_SUCCESS;
        }
        current = current->next;
    }
    
    return GPU_ERROR_INVALID_PARAM;
}

gpu_result_t defragment_memory_pool(memory_pool_t *pool) {
    if (!pool || !pool->initialized) {
        return GPU_ERROR_INVALID_PARAM;
    }
    
    // 简单的碎片整理实现
    merge_free_blocks(pool);
    
    return GPU_SUCCESS;
}

gpu_result_t get_memory_stats(memory_pool_t *pool, memory_stats_t *stats) {
    if (!pool || !stats) {
        return GPU_ERROR_INVALID_PARAM;
    }
    
    stats->total_size = pool->total_size;
    stats->used_size = pool->used_size;
    stats->free_size = pool->total_size - pool->used_size;
    
    return GPU_SUCCESS;
}

bool validate_memory_pool(memory_pool_t *pool) {
    if (!pool || !pool->initialized) {
        return false;
    }
    
    // 简单的验证逻辑
    return pool->used_size <= pool->total_size;
}

gpu_result_t configure_memory_pool(memory_pool_t *pool, const void *config) {
    if (!pool || !config) {
        return GPU_ERROR_INVALID_PARAM;
    }
    
    // 简化实现，暂时不做配置
    return GPU_SUCCESS;
}

/**
 * 清理显存池
 */
gpu_result_t cleanup_memory_pool(memory_pool_t *pool) {
    if (!pool || !pool->initialized) {
        return GPU_ERROR_INVALID_PARAM;
    }
    
    // 释放显存
    cudaSetDevice(pool->device_id);
    cudaFree(pool->base_ptr);
    
    // 释放块链表
    memory_block_t *current = pool->free_blocks;
    while (current) {
        memory_block_t *next = current->next;
        free(current);
        current = next;
    }
    
    current = pool->used_blocks;
    while (current) {
        memory_block_t *next = current->next;
        free(current);
        current = next;
    }
    
    pool->initialized = false;
    printf("Memory pool cleaned up on device %d\n", pool->device_id);
    return GPU_SUCCESS;
}