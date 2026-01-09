#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <stddef.h>
#include <stdbool.h>
#include "../common/gpu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// 内存块结构
typedef struct memory_block {
    void *ptr;                    // 内存块指针
    size_t size;                  // 块大小
    bool is_free;                 // 是否空闲
    struct memory_block *next;    // 下一个块
    struct memory_block *prev;    // 上一个块
} memory_block_t;

// 显存池结构
typedef struct {
    void *base_ptr;               // 基地址
    size_t total_size;            // 总大小
    size_t used_size;             // 已使用大小
    int device_id;                // 设备ID
    memory_block_t *free_blocks;  // 空闲块链表
    memory_block_t *used_blocks;  // 已用块链表
    bool initialized;             // 是否已初始化
} memory_pool_t;

// 内存统计信息
typedef struct {
    size_t total_size;            // 总内存大小
    size_t used_size;             // 已使用内存
    size_t free_size;             // 空闲内存
    int fragment_count;           // 碎片数量
    size_t largest_free_block;    // 最大空闲块
} memory_stats_t;

// 函数声明

/**
 * 初始化显存池
 * @param pool 显存池指针
 * @param device_id GPU设备ID
 * @param size 池大小
 * @return 操作结果
 */
gpu_result_t init_memory_pool(memory_pool_t *pool, int device_id, size_t size);

/**
 * 从显存池分配内存
 * @param pool 显存池指针
 * @param size 请求大小
 * @return 分配的内存指针，失败返回NULL
 */
void* allocate_memory(memory_pool_t *pool, size_t size);

/**
 * 释放内存到显存池
 * @param pool 显存池指针
 * @param ptr 要释放的内存指针
 * @return 操作结果
 */
gpu_result_t free_memory(memory_pool_t *pool, void *ptr);

/**
 * 清理显存池
 * @param pool 显存池指针
 * @return 操作结果
 */
gpu_result_t cleanup_memory_pool(memory_pool_t *pool);

/**
 * 内存碎片整理
 * @param pool 显存池指针
 * @return 操作结果
 */
gpu_result_t defragment_memory_pool(memory_pool_t *pool);

/**
 * 获取内存统计信息
 * @param pool 显存池指针
 * @param stats 统计信息输出
 * @return 操作结果
 */
gpu_result_t get_memory_stats(memory_pool_t *pool, memory_stats_t *stats);

/**
 * 检查内存池完整性
 * @param pool 显存池指针
 * @return 是否完整
 */
bool validate_memory_pool(memory_pool_t *pool);

/**
 * 设置内存池配置
 * @param pool 显存池指针
 * @param config 配置参数
 * @return 操作结果
 */
gpu_result_t configure_memory_pool(memory_pool_t *pool, const void *config);

#ifdef __cplusplus
}
#endif

#endif // MEMORY_POOL_H