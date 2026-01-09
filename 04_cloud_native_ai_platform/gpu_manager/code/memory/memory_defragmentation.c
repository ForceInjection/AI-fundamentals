/**
 * GPU内存碎片整理实现
 * 功能：减少内存碎片，提高内存利用率
 * 特性：在线碎片整理、智能压缩、内存重排
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>

#define MAX_MEMORY_BLOCKS 10000
#define MIN_BLOCK_SIZE 1024
#define DEFRAG_THRESHOLD 0.3  // 碎片率阈值30%
#define COMPACTION_RATIO 0.8  // 压缩比阈值80%

// 内存块状态
typedef enum {
    BLOCK_FREE = 0,
    BLOCK_ALLOCATED,
    BLOCK_MOVING,
    BLOCK_COMPACTING
} block_state_t;

// 内存块描述符
typedef struct memory_block {
    void *address;              // 内存地址
    size_t size;               // 块大小
    block_state_t state;       // 块状态
    uint32_t ref_count;        // 引用计数
    uint64_t last_access;      // 最后访问时间
    bool is_pinned;            // 是否锁定
    uint32_t tenant_id;        // 租户ID
    struct memory_block *next; // 下一个块
    struct memory_block *prev; // 前一个块
} memory_block_t;

// 碎片整理统计信息
typedef struct {
    uint64_t total_memory;        // 总内存
    uint64_t allocated_memory;    // 已分配内存
    uint64_t free_memory;         // 空闲内存
    uint32_t total_blocks;        // 总块数
    uint32_t free_blocks;         // 空闲块数
    uint32_t largest_free_block;  // 最大空闲块
    double fragmentation_ratio;   // 碎片率
    uint32_t defrag_operations;   // 碎片整理操作数
    uint64_t bytes_moved;         // 移动的字节数
    uint64_t defrag_time_ms;      // 碎片整理时间
} defrag_stats_t;

// 碎片整理器
typedef struct {
    memory_block_t *block_list;     // 内存块链表
    memory_block_t *free_list;      // 空闲块链表
    defrag_stats_t stats;           // 统计信息
    pthread_mutex_t defrag_lock;    // 碎片整理锁
    pthread_t defrag_thread;        // 碎片整理线程
    bool defrag_enabled;            // 是否启用碎片整理
    bool defrag_running;            // 碎片整理是否运行中
    double threshold;               // 碎片率阈值
} defragmenter_t;

// 全局碎片整理器
static defragmenter_t g_defragmenter = {0};

// 计算碎片率
static double calculate_fragmentation_ratio(void) {
    if (g_defragmenter.stats.free_memory == 0) {
        return 0.0;
    }
    
    uint32_t free_blocks = g_defragmenter.stats.free_blocks;
    uint64_t free_memory = g_defragmenter.stats.free_memory;
    uint32_t largest_free = g_defragmenter.stats.largest_free_block;
    
    // 碎片率 = (空闲块数 - 1) / 总空闲内存 * 最大空闲块大小
    if (largest_free == 0) {
        return 1.0;  // 完全碎片化
    }
    
    return 1.0 - ((double)largest_free / free_memory);
}

// 更新统计信息
static void update_defrag_stats(void) {
    g_defragmenter.stats.total_blocks = 0;
    g_defragmenter.stats.free_blocks = 0;
    g_defragmenter.stats.allocated_memory = 0;
    g_defragmenter.stats.free_memory = 0;
    g_defragmenter.stats.largest_free_block = 0;
    
    memory_block_t *block = g_defragmenter.block_list;
    while (block) {
        g_defragmenter.stats.total_blocks++;
        
        if (block->state == BLOCK_FREE) {
            g_defragmenter.stats.free_blocks++;
            g_defragmenter.stats.free_memory += block->size;
            if (block->size > g_defragmenter.stats.largest_free_block) {
                g_defragmenter.stats.largest_free_block = block->size;
            }
        } else {
            g_defragmenter.stats.allocated_memory += block->size;
        }
        
        block = block->next;
    }
    
    g_defragmenter.stats.fragmentation_ratio = calculate_fragmentation_ratio();
}

// 合并相邻空闲块
static int merge_free_blocks(void) {
    int merged_count = 0;
    memory_block_t *block = g_defragmenter.block_list;
    
    while (block && block->next) {
        if (block->state == BLOCK_FREE && block->next->state == BLOCK_FREE) {
            // 检查是否相邻
            if ((char*)block->address + block->size == block->next->address) {
                memory_block_t *next_block = block->next;
                
                // 合并块
                block->size += next_block->size;
                block->next = next_block->next;
                if (next_block->next) {
                    next_block->next->prev = block;
                }
                
                free(next_block);
                merged_count++;
                continue;
            }
        }
        block = block->next;
    }
    
    return merged_count;
}

// 移动内存块
static int move_memory_block(memory_block_t *src_block, void *dest_addr) {
    if (!src_block || !dest_addr || src_block->is_pinned) {
        return -1;
    }
    
    // 标记为移动中
    src_block->state = BLOCK_MOVING;
    
    // 复制数据
    memcpy(dest_addr, src_block->address, src_block->size);
    
    // 更新地址
    src_block->address = dest_addr;
    src_block->state = BLOCK_ALLOCATED;
    
    g_defragmenter.stats.bytes_moved += src_block->size;
    
    return 0;
}

// 压缩内存
static int compact_memory(void) {
    memory_block_t *allocated_blocks[MAX_MEMORY_BLOCKS];
    int allocated_count = 0;
    
    // 收集所有已分配的块
    memory_block_t *block = g_defragmenter.block_list;
    while (block && allocated_count < MAX_MEMORY_BLOCKS) {
        if (block->state == BLOCK_ALLOCATED && !block->is_pinned) {
            allocated_blocks[allocated_count++] = block;
        }
        block = block->next;
    }
    
    // 按地址排序（简化实现）
    for (int i = 0; i < allocated_count - 1; i++) {
        for (int j = i + 1; j < allocated_count; j++) {
            if (allocated_blocks[i]->address > allocated_blocks[j]->address) {
                memory_block_t *temp = allocated_blocks[i];
                allocated_blocks[i] = allocated_blocks[j];
                allocated_blocks[j] = temp;
            }
        }
    }
    
    // 压缩：将所有已分配块移动到内存开始处
    void *current_addr = g_defragmenter.block_list->address;
    for (int i = 0; i < allocated_count; i++) {
        if (allocated_blocks[i]->address != current_addr) {
            move_memory_block(allocated_blocks[i], current_addr);
        }
        current_addr = (char*)current_addr + allocated_blocks[i]->size;
    }
    
    return allocated_count;
}

// 执行碎片整理
static int perform_defragmentation(void) {
    pthread_mutex_lock(&g_defragmenter.defrag_lock);
    
    clock_t start_time = clock();
    
    printf("Starting memory defragmentation...\n");
    
    // 1. 合并相邻空闲块
    int merged = merge_free_blocks();
    printf("Merged %d free blocks\n", merged);
    
    // 2. 更新统计信息
    update_defrag_stats();
    
    // 3. 如果碎片率仍然过高，执行压缩
    if (g_defragmenter.stats.fragmentation_ratio > g_defragmenter.threshold) {
        printf("Fragmentation ratio %.2f%% exceeds threshold, compacting...\n", 
               g_defragmenter.stats.fragmentation_ratio * 100);
        
        int compacted = compact_memory();
        printf("Compacted %d memory blocks\n", compacted);
        
        // 重新合并空闲块
        merge_free_blocks();
    }
    
    // 4. 最终更新统计信息
    update_defrag_stats();
    g_defragmenter.stats.defrag_operations++;
    
    clock_t end_time = clock();
    g_defragmenter.stats.defrag_time_ms += 
        (end_time - start_time) * 1000 / CLOCKS_PER_SEC;
    
    printf("Defragmentation completed. New fragmentation ratio: %.2f%%\n", 
           g_defragmenter.stats.fragmentation_ratio * 100);
    
    pthread_mutex_unlock(&g_defragmenter.defrag_lock);
    
    return 0;
}

// 碎片整理线程
static void* defrag_thread_func(void *arg) {
    while (g_defragmenter.defrag_enabled) {
        // 检查是否需要碎片整理
        update_defrag_stats();
        
        if (g_defragmenter.stats.fragmentation_ratio > g_defragmenter.threshold) {
            g_defragmenter.defrag_running = true;
            perform_defragmentation();
            g_defragmenter.defrag_running = false;
        }
        
        // 休眠1秒
        sleep(1);
    }
    
    return NULL;
}

// 初始化碎片整理器
int init_defragmenter(double threshold) {
    memset(&g_defragmenter, 0, sizeof(defragmenter_t));
    
    g_defragmenter.threshold = threshold;
    g_defragmenter.defrag_enabled = true;
    
    if (pthread_mutex_init(&g_defragmenter.defrag_lock, NULL) != 0) {
        return -1;
    }
    
    // 启动碎片整理线程
    if (pthread_create(&g_defragmenter.defrag_thread, NULL, 
                      defrag_thread_func, NULL) != 0) {
        pthread_mutex_destroy(&g_defragmenter.defrag_lock);
        return -1;
    }
    
    printf("Memory defragmenter initialized with threshold %.2f%%\n", 
           threshold * 100);
    
    return 0;
}

// 手动触发碎片整理
int trigger_defragmentation(void) {
    if (g_defragmenter.defrag_running) {
        return -1;  // 已在运行中
    }
    
    return perform_defragmentation();
}

// 获取碎片整理统计信息
defrag_stats_t get_defrag_stats(void) {
    pthread_mutex_lock(&g_defragmenter.defrag_lock);
    defrag_stats_t stats = g_defragmenter.stats;
    pthread_mutex_unlock(&g_defragmenter.defrag_lock);
    
    return stats;
}

// 设置碎片整理阈值
void set_defrag_threshold(double threshold) {
    pthread_mutex_lock(&g_defragmenter.defrag_lock);
    g_defragmenter.threshold = threshold;
    pthread_mutex_unlock(&g_defragmenter.defrag_lock);
}

// 启用/禁用自动碎片整理
void enable_auto_defrag(bool enable) {
    g_defragmenter.defrag_enabled = enable;
}

// 清理碎片整理器
void cleanup_defragmenter(void) {
    g_defragmenter.defrag_enabled = false;
    
    // 等待线程结束
    pthread_join(g_defragmenter.defrag_thread, NULL);
    
    // 清理资源
    pthread_mutex_destroy(&g_defragmenter.defrag_lock);
    
    // 释放内存块链表
    memory_block_t *block = g_defragmenter.block_list;
    while (block) {
        memory_block_t *next = block->next;
        free(block);
        block = next;
    }
    
    printf("Memory defragmenter cleaned up\n");
}

// 打印碎片整理统计信息
void print_defrag_stats(void) {
    defrag_stats_t stats = get_defrag_stats();
    
    printf("\n=== Memory Defragmentation Statistics ===\n");
    printf("Total Memory: %llu bytes\n", (unsigned long long)stats.total_memory);
    printf("Allocated Memory: %llu bytes\n", (unsigned long long)stats.allocated_memory);
    printf("Free Memory: %llu bytes\n", (unsigned long long)stats.free_memory);
    printf("Total Blocks: %u\n", stats.total_blocks);
    printf("Free Blocks: %u\n", stats.free_blocks);
    printf("Largest Free Block: %u bytes\n", stats.largest_free_block);
    printf("Fragmentation Ratio: %.2f%%\n", stats.fragmentation_ratio * 100);
    printf("Defragmentation Operations: %u\n", stats.defrag_operations);
    printf("Bytes Moved: %llu\n", (unsigned long long)stats.bytes_moved);
    printf("Total Defrag Time: %llu ms\n", (unsigned long long)stats.defrag_time_ms);
    printf("==========================================\n\n");
}