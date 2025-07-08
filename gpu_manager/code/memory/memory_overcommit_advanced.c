/**
 * GPU内存过量分配(Overcommit)高级实现
 * 功能：实现GPU内存的智能过量分配和动态管理
 * 特性：内存压缩、交换、统一地址空间、QoS保障
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

// 内存页面大小
#define PAGE_SIZE 4096
#define GPU_PAGE_SIZE (64 * 1024)  // 64KB GPU页面
#define MAX_MEMORY_POOLS 16
#define MAX_VIRTUAL_PAGES 1048576  // 最大虚拟页面数

// 内存页面状态
typedef enum {
    PAGE_STATE_FREE = 0,
    PAGE_STATE_ALLOCATED,
    PAGE_STATE_COMPRESSED,
    PAGE_STATE_SWAPPED,
    PAGE_STATE_LOCKED
} page_state_t;

// 内存压缩算法类型
typedef enum {
    COMPRESS_NONE = 0,
    COMPRESS_LZ4,
    COMPRESS_ZSTD,
    COMPRESS_SNAPPY
} compression_type_t;

// QoS等级定义
typedef enum {
    QOS_CRITICAL = 0,  // 关键任务
    QOS_HIGH = 1,      // 高优先级
    QOS_NORMAL = 2,    // 普通优先级
    QOS_LOW = 3        // 低优先级
} memory_qos_level_t;

// 虚拟内存页面描述符
typedef struct {
    uint64_t virtual_addr;     // 虚拟地址
    uint64_t physical_addr;    // 物理地址
    page_state_t state;        // 页面状态
    uint32_t ref_count;        // 引用计数
    uint32_t access_count;     // 访问次数
    uint64_t last_access_time; // 最后访问时间
    compression_type_t compress_type; // 压缩类型
    size_t compressed_size;    // 压缩后大小
    void *compressed_data;     // 压缩数据
    memory_qos_level_t qos_level; // QoS等级
    uint32_t tenant_id;        // 租户ID
    bool is_pinned;            // 是否锁定
} virtual_page_t;

// 内存池描述符
typedef struct {
    uint32_t pool_id;
    size_t total_size;         // 总大小
    size_t allocated_size;     // 已分配大小
    size_t compressed_size;    // 压缩节省的大小
    size_t swapped_size;       // 交换出的大小
    double overcommit_ratio;   // 过量分配比例
    memory_qos_level_t min_qos; // 最低QoS等级
    bool compression_enabled;   // 是否启用压缩
    bool swap_enabled;         // 是否启用交换
    pthread_mutex_t lock;      // 池锁
} memory_pool_t;

// 内存统计信息
typedef struct {
    uint64_t total_virtual_memory;   // 总虚拟内存
    uint64_t total_physical_memory;  // 总物理内存
    uint64_t allocated_memory;       // 已分配内存
    uint64_t compressed_memory;      // 压缩内存
    uint64_t swapped_memory;         // 交换内存
    uint64_t free_memory;           // 空闲内存
    double compression_ratio;        // 压缩比
    double overcommit_ratio;        // 过量分配比
    uint32_t page_faults;           // 页面错误数
    uint32_t compression_ops;       // 压缩操作数
    uint32_t swap_ops;              // 交换操作数
} memory_stats_t;

// 全局内存管理器
typedef struct {
    virtual_page_t *page_table;     // 页面表
    memory_pool_t pools[MAX_MEMORY_POOLS]; // 内存池
    memory_stats_t stats;           // 统计信息
    uint32_t active_pools;          // 活跃池数
    uint32_t total_pages;           // 总页面数
    uint32_t free_pages;            // 空闲页面数
    bool overcommit_enabled;        // 是否启用过量分配
    double global_overcommit_ratio; // 全局过量分配比例
    pthread_rwlock_t global_lock;   // 全局读写锁
    void *swap_space;               // 交换空间
    size_t swap_space_size;         // 交换空间大小
} advanced_memory_manager_t;

// 全局内存管理器实例
static advanced_memory_manager_t g_memory_manager = {0};

// 函数声明
static uint64_t get_current_time_us(void);
static int compress_page(virtual_page_t *page, compression_type_t type);
static int decompress_page(virtual_page_t *page);
static int swap_out_page(virtual_page_t *page);
static int swap_in_page(virtual_page_t *page);
static virtual_page_t* find_victim_page(memory_qos_level_t min_qos);
static double calculate_compression_ratio(void);
static void update_memory_stats(void);
static bool check_qos_constraints(memory_qos_level_t qos, size_t size);

// 初始化高级内存管理器
int init_advanced_memory_manager(size_t total_memory, double overcommit_ratio) {
    // 初始化页面表
    g_memory_manager.total_pages = total_memory / GPU_PAGE_SIZE;
    g_memory_manager.page_table = calloc(g_memory_manager.total_pages, 
                                        sizeof(virtual_page_t));
    if (!g_memory_manager.page_table) {
        return -1;
    }
    
    // 初始化统计信息
    g_memory_manager.stats.total_physical_memory = total_memory;
    g_memory_manager.stats.total_virtual_memory = 
        (uint64_t)(total_memory * overcommit_ratio);
    g_memory_manager.global_overcommit_ratio = overcommit_ratio;
    g_memory_manager.overcommit_enabled = true;
    g_memory_manager.free_pages = g_memory_manager.total_pages;
    
    // 初始化交换空间
    g_memory_manager.swap_space_size = total_memory / 2; // 50%的交换空间
    g_memory_manager.swap_space = malloc(g_memory_manager.swap_space_size);
    
    // 初始化锁
    pthread_rwlock_init(&g_memory_manager.global_lock, NULL);
    
    // 初始化内存池
    for (int i = 0; i < MAX_MEMORY_POOLS; i++) {
        pthread_mutex_init(&g_memory_manager.pools[i].lock, NULL);
        g_memory_manager.pools[i].pool_id = i;
        g_memory_manager.pools[i].compression_enabled = true;
        g_memory_manager.pools[i].swap_enabled = true;
        g_memory_manager.pools[i].overcommit_ratio = overcommit_ratio;
    }
    
    printf("Advanced memory manager initialized:\n");
    printf("  Physical memory: %zu MB\n", total_memory / (1024*1024));
    printf("  Virtual memory: %llu MB\n", 
           g_memory_manager.stats.total_virtual_memory / (1024*1024));
    printf("  Overcommit ratio: %.2f\n", overcommit_ratio);
    printf("  Swap space: %zu MB\n", g_memory_manager.swap_space_size / (1024*1024));
    
    return 0;
}

// 智能内存分配
void* advanced_memory_alloc(size_t size, memory_qos_level_t qos, uint32_t tenant_id) {
    pthread_rwlock_wrlock(&g_memory_manager.global_lock);
    
    // 检查QoS约束
    if (!check_qos_constraints(qos, size)) {
        pthread_rwlock_unlock(&g_memory_manager.global_lock);
        return NULL;
    }
    
    // 计算需要的页面数
    uint32_t pages_needed = (size + GPU_PAGE_SIZE - 1) / GPU_PAGE_SIZE;
    
    // 检查是否有足够的虚拟内存
    if (g_memory_manager.stats.allocated_memory + size > 
        g_memory_manager.stats.total_virtual_memory) {
        pthread_rwlock_unlock(&g_memory_manager.global_lock);
        return NULL;
    }
    
    // 如果物理内存不足，尝试压缩或交换
    if (g_memory_manager.free_pages < pages_needed) {
        // 尝试压缩低优先级页面
        for (uint32_t i = 0; i < g_memory_manager.total_pages && 
             g_memory_manager.free_pages < pages_needed; i++) {
            virtual_page_t *page = &g_memory_manager.page_table[i];
            if (page->state == PAGE_STATE_ALLOCATED && 
                page->qos_level > qos && !page->is_pinned) {
                if (compress_page(page, COMPRESS_LZ4) == 0) {
                    g_memory_manager.free_pages++;
                }
            }
        }
        
        // 如果压缩后仍不足，尝试交换
        if (g_memory_manager.free_pages < pages_needed) {
            virtual_page_t *victim = find_victim_page(qos);
            if (victim && swap_out_page(victim) == 0) {
                g_memory_manager.free_pages++;
            }
        }
    }
    
    // 分配虚拟页面
    virtual_page_t *allocated_pages = NULL;
    uint32_t allocated_count = 0;
    
    for (uint32_t i = 0; i < g_memory_manager.total_pages && 
         allocated_count < pages_needed; i++) {
        virtual_page_t *page = &g_memory_manager.page_table[i];
        if (page->state == PAGE_STATE_FREE) {
            page->state = PAGE_STATE_ALLOCATED;
            page->virtual_addr = (uint64_t)page * GPU_PAGE_SIZE;
            page->physical_addr = page->virtual_addr; // 简化实现
            page->qos_level = qos;
            page->tenant_id = tenant_id;
            page->ref_count = 1;
            page->last_access_time = get_current_time_us();
            page->is_pinned = (qos == QOS_CRITICAL);
            
            if (allocated_count == 0) {
                allocated_pages = page;
            }
            allocated_count++;
        }
    }
    
    if (allocated_count < pages_needed) {
        // 回滚分配
        for (uint32_t i = 0; i < allocated_count; i++) {
            allocated_pages[i].state = PAGE_STATE_FREE;
        }
        pthread_rwlock_unlock(&g_memory_manager.global_lock);
        return NULL;
    }
    
    // 更新统计信息
    g_memory_manager.stats.allocated_memory += size;
    g_memory_manager.free_pages -= pages_needed;
    
    update_memory_stats();
    
    pthread_rwlock_unlock(&g_memory_manager.global_lock);
    
    printf("Allocated %zu bytes (QoS: %d, Tenant: %u) at virtual address: 0x%llx\n",
           size, qos, tenant_id, (unsigned long long)allocated_pages->virtual_addr);
    
    return (void*)allocated_pages->virtual_addr;
}

// 内存释放
int advanced_memory_free(void *ptr, size_t size) {
    pthread_rwlock_wrlock(&g_memory_manager.global_lock);
    
    uint64_t addr = (uint64_t)ptr;
    uint32_t page_index = addr / GPU_PAGE_SIZE;
    uint32_t pages_to_free = (size + GPU_PAGE_SIZE - 1) / GPU_PAGE_SIZE;
    
    for (uint32_t i = 0; i < pages_to_free; i++) {
        virtual_page_t *page = &g_memory_manager.page_table[page_index + i];
        
        if (page->state == PAGE_STATE_COMPRESSED && page->compressed_data) {
            free(page->compressed_data);
            g_memory_manager.stats.compressed_memory -= page->compressed_size;
        }
        
        page->state = PAGE_STATE_FREE;
        page->ref_count = 0;
        page->compressed_data = NULL;
        page->compressed_size = 0;
        
        g_memory_manager.free_pages++;
    }
    
    g_memory_manager.stats.allocated_memory -= size;
    update_memory_stats();
    
    pthread_rwlock_unlock(&g_memory_manager.global_lock);
    
    printf("Freed %zu bytes at address: 0x%llx\n", size, (unsigned long long)addr);
    return 0;
}

// 内存压缩实现
static int compress_page(virtual_page_t *page, compression_type_t type) {
    if (page->state != PAGE_STATE_ALLOCATED || page->is_pinned) {
        return -1;
    }
    
    // 简化的压缩实现（实际应使用LZ4、ZSTD等算法）
    size_t original_size = GPU_PAGE_SIZE;
    size_t compressed_size = original_size / 2; // 假设50%压缩率
    
    void *compressed_data = malloc(compressed_size);
    if (!compressed_data) {
        return -1;
    }
    
    // 模拟压缩过程
    memcpy(compressed_data, (void*)page->physical_addr, compressed_size);
    
    page->compressed_data = compressed_data;
    page->compressed_size = compressed_size;
    page->compress_type = type;
    page->state = PAGE_STATE_COMPRESSED;
    
    g_memory_manager.stats.compressed_memory += compressed_size;
    g_memory_manager.stats.compression_ops++;
    
    printf("Compressed page at 0x%llx: %zu -> %zu bytes\n",
           (unsigned long long)page->virtual_addr, original_size, compressed_size);
    
    return 0;
}

// 内存解压缩实现
static int decompress_page(virtual_page_t *page) {
    if (page->state != PAGE_STATE_COMPRESSED) {
        return -1;
    }
    
    // 检查是否有足够的物理内存
    if (g_memory_manager.free_pages < 1) {
        // 尝试找到牺牲页面
        virtual_page_t *victim = find_victim_page(page->qos_level);
        if (!victim || swap_out_page(victim) != 0) {
            return -1;
        }
    }
    
    // 解压缩数据
    memcpy((void*)page->physical_addr, page->compressed_data, page->compressed_size);
    
    // 清理压缩数据
    free(page->compressed_data);
    g_memory_manager.stats.compressed_memory -= page->compressed_size;
    
    page->compressed_data = NULL;
    page->compressed_size = 0;
    page->state = PAGE_STATE_ALLOCATED;
    page->last_access_time = get_current_time_us();
    
    g_memory_manager.free_pages--;
    
    printf("Decompressed page at 0x%llx\n", 
           (unsigned long long)page->virtual_addr);
    
    return 0;
}

// 页面交换出实现
static int swap_out_page(virtual_page_t *page) {
    if (page->state != PAGE_STATE_ALLOCATED || page->is_pinned) {
        return -1;
    }
    
    // 检查交换空间
    if (g_memory_manager.stats.swapped_memory + GPU_PAGE_SIZE > 
        g_memory_manager.swap_space_size) {
        return -1;
    }
    
    // 将页面数据复制到交换空间
    size_t swap_offset = g_memory_manager.stats.swapped_memory;
    memcpy((char*)g_memory_manager.swap_space + swap_offset,
           (void*)page->physical_addr, GPU_PAGE_SIZE);
    
    page->physical_addr = swap_offset; // 存储交换偏移
    page->state = PAGE_STATE_SWAPPED;
    
    g_memory_manager.stats.swapped_memory += GPU_PAGE_SIZE;
    g_memory_manager.stats.swap_ops++;
    
    printf("Swapped out page at 0x%llx to offset %zu\n",
           (unsigned long long)page->virtual_addr, swap_offset);
    
    return 0;
}

// 页面交换入实现
static int swap_in_page(virtual_page_t *page) {
    if (page->state != PAGE_STATE_SWAPPED) {
        return -1;
    }
    
    // 检查物理内存可用性
    if (g_memory_manager.free_pages < 1) {
        virtual_page_t *victim = find_victim_page(page->qos_level);
        if (!victim || swap_out_page(victim) != 0) {
            return -1;
        }
    }
    
    // 分配新的物理页面
    uint64_t new_physical_addr = page->virtual_addr; // 简化实现
    
    // 从交换空间恢复数据
    size_t swap_offset = page->physical_addr;
    memcpy((void*)new_physical_addr,
           (char*)g_memory_manager.swap_space + swap_offset, GPU_PAGE_SIZE);
    
    page->physical_addr = new_physical_addr;
    page->state = PAGE_STATE_ALLOCATED;
    page->last_access_time = get_current_time_us();
    
    g_memory_manager.stats.swapped_memory -= GPU_PAGE_SIZE;
    g_memory_manager.free_pages--;
    
    printf("Swapped in page at 0x%llx from offset %zu\n",
           (unsigned long long)page->virtual_addr, swap_offset);
    
    return 0;
}

// 查找牺牲页面（LRU算法）
static virtual_page_t* find_victim_page(memory_qos_level_t min_qos) {
    virtual_page_t *victim = NULL;
    uint64_t oldest_time = UINT64_MAX;
    
    for (uint32_t i = 0; i < g_memory_manager.total_pages; i++) {
        virtual_page_t *page = &g_memory_manager.page_table[i];
        
        if (page->state == PAGE_STATE_ALLOCATED && 
            page->qos_level >= min_qos && 
            !page->is_pinned &&
            page->last_access_time < oldest_time) {
            victim = page;
            oldest_time = page->last_access_time;
        }
    }
    
    return victim;
}

// 检查QoS约束
static bool check_qos_constraints(memory_qos_level_t qos, size_t size) {
    // 关键任务总是允许分配
    if (qos == QOS_CRITICAL) {
        return true;
    }
    
    // 检查当前内存使用率
    double usage_ratio = (double)g_memory_manager.stats.allocated_memory / 
                        g_memory_manager.stats.total_physical_memory;
    
    switch (qos) {
    case QOS_HIGH:
        return usage_ratio < 0.9; // 高优先级在90%以下可分配
    case QOS_NORMAL:
        return usage_ratio < 0.8; // 普通优先级在80%以下可分配
    case QOS_LOW:
        return usage_ratio < 0.7; // 低优先级在70%以下可分配
    default:
        return false;
    }
}

// 更新内存统计信息
static void update_memory_stats(void) {
    g_memory_manager.stats.free_memory = 
        g_memory_manager.free_pages * GPU_PAGE_SIZE;
    
    if (g_memory_manager.stats.allocated_memory > 0) {
        g_memory_manager.stats.compression_ratio = 
            (double)g_memory_manager.stats.compressed_memory / 
            g_memory_manager.stats.allocated_memory;
    }
    
    g_memory_manager.stats.overcommit_ratio = 
        (double)g_memory_manager.stats.total_virtual_memory / 
        g_memory_manager.stats.total_physical_memory;
}

// 获取当前时间（微秒）
static uint64_t get_current_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

// 打印内存统计信息
void print_memory_stats(void) {
    pthread_rwlock_rdlock(&g_memory_manager.global_lock);
    
    printf("\n=== Advanced Memory Manager Statistics ===\n");
    printf("Physical Memory: %llu MB\n", 
           g_memory_manager.stats.total_physical_memory / (1024*1024));
    printf("Virtual Memory: %llu MB\n", 
           g_memory_manager.stats.total_virtual_memory / (1024*1024));
    printf("Allocated Memory: %llu MB\n", 
           g_memory_manager.stats.allocated_memory / (1024*1024));
    printf("Compressed Memory: %llu MB\n", 
           g_memory_manager.stats.compressed_memory / (1024*1024));
    printf("Swapped Memory: %llu MB\n", 
           g_memory_manager.stats.swapped_memory / (1024*1024));
    printf("Free Memory: %llu MB\n", 
           g_memory_manager.stats.free_memory / (1024*1024));
    printf("Compression Ratio: %.2f%%\n", 
           g_memory_manager.stats.compression_ratio * 100);
    printf("Overcommit Ratio: %.2f\n", 
           g_memory_manager.stats.overcommit_ratio);
    printf("Page Faults: %u\n", g_memory_manager.stats.page_faults);
    printf("Compression Operations: %u\n", 
           g_memory_manager.stats.compression_ops);
    printf("Swap Operations: %u\n", g_memory_manager.stats.swap_ops);
    printf("==========================================\n\n");
    
    pthread_rwlock_unlock(&g_memory_manager.global_lock);
}

// 清理内存管理器
void cleanup_advanced_memory_manager(void) {
    pthread_rwlock_wrlock(&g_memory_manager.global_lock);
    
    // 清理压缩数据
    for (uint32_t i = 0; i < g_memory_manager.total_pages; i++) {
        virtual_page_t *page = &g_memory_manager.page_table[i];
        if (page->compressed_data) {
            free(page->compressed_data);
        }
    }
    
    // 释放资源
    free(g_memory_manager.page_table);
    free(g_memory_manager.swap_space);
    
    pthread_rwlock_unlock(&g_memory_manager.global_lock);
    pthread_rwlock_destroy(&g_memory_manager.global_lock);
    
    for (int i = 0; i < MAX_MEMORY_POOLS; i++) {
        pthread_mutex_destroy(&g_memory_manager.pools[i].lock);
    }
    
    printf("Advanced memory manager cleaned up\n");
}