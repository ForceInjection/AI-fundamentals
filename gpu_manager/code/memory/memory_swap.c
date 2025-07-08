/**
 * GPU内存交换机制实现
 * 功能：实现GPU内存与系统内存/存储设备之间的交换
 * 支持：多级存储、智能预取、异步交换
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

// 函数前向声明
void cleanup_swap_system(void);

// 交换存储类型
typedef enum {
    SWAP_SYSTEM_MEMORY = 0,  // 系统内存
    SWAP_SSD_STORAGE,        // SSD存储
    SWAP_HDD_STORAGE,        // HDD存储
    SWAP_NVME_STORAGE,       // NVMe存储
    SWAP_REMOTE_STORAGE      // 远程存储
} swap_storage_type_t;

// 交换页状态
typedef enum {
    PAGE_STATE_ACTIVE = 0,   // 活跃状态
    PAGE_STATE_INACTIVE,     // 非活跃状态
    PAGE_STATE_SWAPPING_OUT, // 正在换出
    PAGE_STATE_SWAPPED_OUT,  // 已换出
    PAGE_STATE_SWAPPING_IN,  // 正在换入
    PAGE_STATE_PREFETCHING   // 预取中
} page_state_t;

// 访问模式
typedef enum {
    ACCESS_SEQUENTIAL = 0,   // 顺序访问
    ACCESS_RANDOM,           // 随机访问
    ACCESS_TEMPORAL,         // 时间局部性
    ACCESS_SPATIAL           // 空间局部性
} access_pattern_t;

// 交换页描述符
typedef struct swap_page {
    uint64_t page_id;           // 页面ID
    void *gpu_addr;             // GPU地址
    void *swap_addr;            // 交换地址
    size_t page_size;           // 页面大小
    page_state_t state;         // 页面状态
    swap_storage_type_t storage_type; // 存储类型
    uint64_t last_access_time;  // 最后访问时间
    uint32_t access_count;      // 访问次数
    access_pattern_t pattern;   // 访问模式
    bool is_dirty;              // 是否脏页
    bool is_pinned;             // 是否锁定
    struct swap_page *next;     // 链表指针
    struct swap_page *prev;     // 链表指针
    pthread_mutex_t page_lock;  // 页面锁
} swap_page_t;

// 交换存储描述符
typedef struct {
    swap_storage_type_t type;   // 存储类型
    char *path;                 // 存储路径
    int fd;                     // 文件描述符
    size_t total_size;          // 总大小
    size_t used_size;           // 已使用大小
    size_t free_size;           // 空闲大小
    uint32_t bandwidth_mbps;    // 带宽（MB/s）
    uint32_t latency_us;        // 延迟（微秒）
    bool is_available;          // 是否可用
    pthread_mutex_t storage_lock; // 存储锁
} swap_storage_t;

// LRU链表
typedef struct {
    swap_page_t *head;          // 链表头
    swap_page_t *tail;          // 链表尾
    size_t count;               // 页面数量
    pthread_mutex_t lru_lock;   // LRU锁
} lru_list_t;

// 交换统计信息
typedef struct {
    uint64_t total_swap_out;    // 总换出次数
    uint64_t total_swap_in;     // 总换入次数
    uint64_t total_prefetch;    // 总预取次数
    uint64_t swap_out_bytes;    // 换出字节数
    uint64_t swap_in_bytes;     // 换入字节数
    uint64_t swap_out_time_us;  // 换出总时间
    uint64_t swap_in_time_us;   // 换入总时间
    uint32_t page_faults;       // 页面错误次数
    uint32_t prefetch_hits;     // 预取命中次数
    uint32_t prefetch_misses;   // 预取失误次数
} swap_stats_t;

// 交换管理器
typedef struct {
    swap_page_t *page_table;    // 页面表
    size_t page_table_size;     // 页面表大小
    swap_storage_t storages[5]; // 存储设备数组
    lru_list_t active_list;     // 活跃页面链表
    lru_list_t inactive_list;   // 非活跃页面链表
    size_t page_size;           // 页面大小
    size_t max_gpu_memory;      // 最大GPU内存
    size_t current_gpu_memory;  // 当前GPU内存使用
    double swap_threshold;      // 交换阈值
    bool prefetch_enabled;      // 是否启用预取
    uint32_t prefetch_window;   // 预取窗口大小
    swap_stats_t stats;         // 统计信息
    pthread_t swap_thread;      // 交换线程
    pthread_mutex_t manager_lock; // 管理器锁
    bool shutdown;              // 关闭标志
} swap_manager_t;

// 全局交换管理器
static swap_manager_t g_swap_manager = {0};

// 函数声明
static uint64_t get_current_time_us(void);
static int init_swap_storage(swap_storage_type_t type, const char *path, size_t size);
static swap_page_t* allocate_swap_page(size_t size);
static void free_swap_page(swap_page_t *page);
static int swap_out_page(swap_page_t *page);
static int swap_in_page(swap_page_t *page);
static void lru_add_page(lru_list_t *list, swap_page_t *page);
static void lru_remove_page(lru_list_t *list, swap_page_t *page);
static void lru_move_to_head(lru_list_t *list, swap_page_t *page);
static swap_page_t* lru_get_tail(lru_list_t *list);
static void update_access_pattern(swap_page_t *page);
static void prefetch_pages(swap_page_t *current_page);
static void* swap_worker_thread(void *arg);
static swap_storage_t* select_best_storage(size_t size);
static int write_to_storage(swap_storage_t *storage, const void *data, size_t size, off_t offset);
static int read_from_storage(swap_storage_t *storage, void *data, size_t size, off_t offset);

// 初始化交换系统
int init_swap_system(size_t max_gpu_memory, size_t page_size, double swap_threshold) {
    memset(&g_swap_manager, 0, sizeof(swap_manager_t));
    
    g_swap_manager.page_size = page_size;
    g_swap_manager.max_gpu_memory = max_gpu_memory;
    g_swap_manager.current_gpu_memory = 0;
    g_swap_manager.swap_threshold = swap_threshold;
    g_swap_manager.prefetch_enabled = true;
    g_swap_manager.prefetch_window = 4; // 预取4个页面
    g_swap_manager.shutdown = false;
    
    // 初始化页面表
    g_swap_manager.page_table_size = max_gpu_memory / page_size;
    g_swap_manager.page_table = calloc(g_swap_manager.page_table_size, sizeof(swap_page_t));
    if (!g_swap_manager.page_table) {
        return -1;
    }
    
    // 初始化LRU链表
    pthread_mutex_init(&g_swap_manager.active_list.lru_lock, NULL);
    pthread_mutex_init(&g_swap_manager.inactive_list.lru_lock, NULL);
    pthread_mutex_init(&g_swap_manager.manager_lock, NULL);
    
    // 初始化存储设备
    init_swap_storage(SWAP_SYSTEM_MEMORY, "/tmp/gpu_swap_mem", 2ULL * 1024 * 1024 * 1024); // 2GB
    init_swap_storage(SWAP_SSD_STORAGE, "/tmp/gpu_swap_ssd", 8ULL * 1024 * 1024 * 1024);   // 8GB
    init_swap_storage(SWAP_HDD_STORAGE, "/tmp/gpu_swap_hdd", 32ULL * 1024 * 1024 * 1024);  // 32GB
    
    // 启动交换工作线程
    if (pthread_create(&g_swap_manager.swap_thread, NULL, swap_worker_thread, NULL) != 0) {
        cleanup_swap_system();
        return -1;
    }
    
    printf("Swap system initialized:\n");
    printf("  Max GPU Memory: %zu MB\n", max_gpu_memory / (1024*1024));
    printf("  Page Size: %zu KB\n", page_size / 1024);
    printf("  Swap Threshold: %.1f%%\n", swap_threshold * 100);
    printf("  Page Table Size: %zu entries\n", g_swap_manager.page_table_size);
    printf("  Prefetch: %s (window: %u)\n", 
           g_swap_manager.prefetch_enabled ? "enabled" : "disabled",
           g_swap_manager.prefetch_window);
    
    return 0;
}

// 分配GPU内存页面
void* allocate_gpu_memory(size_t size) {
    if (size == 0) {
        return NULL;
    }
    
    // 对齐到页面大小
    size_t aligned_size = (size + g_swap_manager.page_size - 1) & 
                         ~(g_swap_manager.page_size - 1);
    
    pthread_mutex_lock(&g_swap_manager.manager_lock);
    
    // 检查是否需要交换
    double usage_ratio = (double)g_swap_manager.current_gpu_memory / g_swap_manager.max_gpu_memory;
    if (usage_ratio > g_swap_manager.swap_threshold) {
        // 触发交换
        printf("GPU memory usage %.1f%% > threshold %.1f%%, triggering swap\n",
               usage_ratio * 100, g_swap_manager.swap_threshold * 100);
        
        // 选择要交换的页面（LRU策略）
        swap_page_t *victim = lru_get_tail(&g_swap_manager.active_list);
        if (victim && !victim->is_pinned) {
            swap_out_page(victim);
        }
    }
    
    // 分配新页面
    swap_page_t *page = allocate_swap_page(aligned_size);
    if (!page) {
        pthread_mutex_unlock(&g_swap_manager.manager_lock);
        return NULL;
    }
    
    // 模拟GPU内存分配
    page->gpu_addr = malloc(aligned_size); // 实际应调用CUDA API
    if (!page->gpu_addr) {
        free_swap_page(page);
        pthread_mutex_unlock(&g_swap_manager.manager_lock);
        return NULL;
    }
    
    page->state = PAGE_STATE_ACTIVE;
    page->last_access_time = get_current_time_us();
    
    // 添加到活跃链表
    lru_add_page(&g_swap_manager.active_list, page);
    
    g_swap_manager.current_gpu_memory += aligned_size;
    
    pthread_mutex_unlock(&g_swap_manager.manager_lock);
    
    printf("Allocated GPU memory: %zu bytes (total: %zu MB)\n", 
           aligned_size, g_swap_manager.current_gpu_memory / (1024*1024));
    
    return page->gpu_addr;
}

// 释放GPU内存页面
void free_gpu_memory(void *ptr) {
    if (!ptr) {
        return;
    }
    
    pthread_mutex_lock(&g_swap_manager.manager_lock);
    
    // 查找页面
    swap_page_t *page = NULL;
    for (size_t i = 0; i < g_swap_manager.page_table_size; i++) {
        if (g_swap_manager.page_table[i].gpu_addr == ptr) {
            page = &g_swap_manager.page_table[i];
            break;
        }
    }
    
    if (!page) {
        pthread_mutex_unlock(&g_swap_manager.manager_lock);
        return;
    }
    
    // 从LRU链表中移除
    if (page->state == PAGE_STATE_ACTIVE) {
        lru_remove_page(&g_swap_manager.active_list, page);
    } else if (page->state == PAGE_STATE_INACTIVE) {
        lru_remove_page(&g_swap_manager.inactive_list, page);
    }
    
    // 释放GPU内存
    if (page->gpu_addr) {
        free(page->gpu_addr); // 实际应调用CUDA API
        g_swap_manager.current_gpu_memory -= page->page_size;
    }
    
    // 释放交换空间
    if (page->swap_addr) {
        free(page->swap_addr);
    }
    
    // 清理页面
    pthread_mutex_destroy(&page->page_lock);
    memset(page, 0, sizeof(swap_page_t));
    
    pthread_mutex_unlock(&g_swap_manager.manager_lock);
    
    printf("Freed GPU memory: %p (remaining: %zu MB)\n", 
           ptr, g_swap_manager.current_gpu_memory / (1024*1024));
}

// 访问GPU内存页面
void* access_gpu_memory(void *ptr) {
    if (!ptr) {
        return NULL;
    }
    
    pthread_mutex_lock(&g_swap_manager.manager_lock);
    
    // 查找页面
    swap_page_t *page = NULL;
    for (size_t i = 0; i < g_swap_manager.page_table_size; i++) {
        if (g_swap_manager.page_table[i].gpu_addr == ptr) {
            page = &g_swap_manager.page_table[i];
            break;
        }
    }
    
    if (!page) {
        pthread_mutex_unlock(&g_swap_manager.manager_lock);
        return NULL;
    }
    
    pthread_mutex_lock(&page->page_lock);
    
    // 检查页面状态
    if (page->state == PAGE_STATE_SWAPPED_OUT) {
        // 页面已被换出，需要换入
        printf("Page fault: swapping in page %llu\n", page->page_id);
        
        g_swap_manager.stats.page_faults++;
        
        if (swap_in_page(page) != 0) {
            pthread_mutex_unlock(&page->page_lock);
            pthread_mutex_unlock(&g_swap_manager.manager_lock);
            return NULL;
        }
    }
    
    // 更新访问信息
    page->last_access_time = get_current_time_us();
    page->access_count++;
    update_access_pattern(page);
    
    // 移动到LRU链表头部
    if (page->state == PAGE_STATE_ACTIVE) {
        lru_move_to_head(&g_swap_manager.active_list, page);
    } else if (page->state == PAGE_STATE_INACTIVE) {
        // 从非活跃链表移动到活跃链表
        lru_remove_page(&g_swap_manager.inactive_list, page);
        page->state = PAGE_STATE_ACTIVE;
        lru_add_page(&g_swap_manager.active_list, page);
    }
    
    // 触发预取
    if (g_swap_manager.prefetch_enabled) {
        prefetch_pages(page);
    }
    
    pthread_mutex_unlock(&page->page_lock);
    pthread_mutex_unlock(&g_swap_manager.manager_lock);
    
    return page->gpu_addr;
}

// 初始化交换存储
static int init_swap_storage(swap_storage_type_t type, const char *path, size_t size) {
    swap_storage_t *storage = &g_swap_manager.storages[type];
    
    storage->type = type;
    storage->path = strdup(path);
    storage->total_size = size;
    storage->used_size = 0;
    storage->free_size = size;
    storage->is_available = false;
    
    // 设置存储特性
    switch (type) {
    case SWAP_SYSTEM_MEMORY:
        storage->bandwidth_mbps = 25600; // 25.6 GB/s
        storage->latency_us = 100;
        break;
    case SWAP_SSD_STORAGE:
        storage->bandwidth_mbps = 3500;  // 3.5 GB/s
        storage->latency_us = 100;
        break;
    case SWAP_HDD_STORAGE:
        storage->bandwidth_mbps = 150;   // 150 MB/s
        storage->latency_us = 10000;
        break;
    case SWAP_NVME_STORAGE:
        storage->bandwidth_mbps = 7000;  // 7 GB/s
        storage->latency_us = 50;
        break;
    case SWAP_REMOTE_STORAGE:
        storage->bandwidth_mbps = 1000;  // 1 GB/s
        storage->latency_us = 1000;
        break;
    }
    
    // 创建交换文件
    storage->fd = open(path, O_CREAT | O_RDWR, 0644);
    if (storage->fd < 0) {
        printf("Failed to create swap file: %s\n", path);
        return -1;
    }
    
    // 预分配空间
    if (ftruncate(storage->fd, size) != 0) {
        printf("Failed to allocate swap space: %s\n", path);
        close(storage->fd);
        return -1;
    }
    
    pthread_mutex_init(&storage->storage_lock, NULL);
    storage->is_available = true;
    
    printf("Initialized %s storage: %s (%zu MB, %u MB/s, %u us)\n",
           type == SWAP_SYSTEM_MEMORY ? "Memory" :
           type == SWAP_SSD_STORAGE ? "SSD" :
           type == SWAP_HDD_STORAGE ? "HDD" :
           type == SWAP_NVME_STORAGE ? "NVMe" : "Remote",
           path, size / (1024*1024), storage->bandwidth_mbps, storage->latency_us);
    
    return 0;
}

// 分配交换页面
static swap_page_t* allocate_swap_page(size_t size) {
    // 查找空闲页面
    for (size_t i = 0; i < g_swap_manager.page_table_size; i++) {
        swap_page_t *page = &g_swap_manager.page_table[i];
        if (page->page_id == 0) { // 空闲页面
            page->page_id = i + 1;
            page->page_size = size;
            page->state = PAGE_STATE_ACTIVE;
            page->storage_type = SWAP_SYSTEM_MEMORY;
            page->last_access_time = get_current_time_us();
            page->access_count = 0;
            page->pattern = ACCESS_SEQUENTIAL;
            page->is_dirty = false;
            page->is_pinned = false;
            page->next = NULL;
            page->prev = NULL;
            
            pthread_mutex_init(&page->page_lock, NULL);
            
            return page;
        }
    }
    
    return NULL; // 页面表已满
}

// 释放交换页面
static void free_swap_page(swap_page_t *page) {
    if (!page) {
        return;
    }
    
    pthread_mutex_destroy(&page->page_lock);
    memset(page, 0, sizeof(swap_page_t));
}

// 换出页面
static int swap_out_page(swap_page_t *page) {
    if (!page || page->is_pinned) {
        return -1;
    }
    
    uint64_t start_time = get_current_time_us();
    
    pthread_mutex_lock(&page->page_lock);
    
    if (page->state != PAGE_STATE_ACTIVE && page->state != PAGE_STATE_INACTIVE) {
        pthread_mutex_unlock(&page->page_lock);
        return -1;
    }
    
    page->state = PAGE_STATE_SWAPPING_OUT;
    
    // 选择最佳存储设备
    swap_storage_t *storage = select_best_storage(page->page_size);
    if (!storage) {
        page->state = PAGE_STATE_ACTIVE;
        pthread_mutex_unlock(&page->page_lock);
        return -1;
    }
    
    // 分配交换空间
    page->swap_addr = malloc(page->page_size);
    if (!page->swap_addr) {
        page->state = PAGE_STATE_ACTIVE;
        pthread_mutex_unlock(&page->page_lock);
        return -1;
    }
    
    // 复制数据到交换空间
    memcpy(page->swap_addr, page->gpu_addr, page->page_size);
    
    // 写入存储设备
    off_t offset = storage->used_size;
    if (write_to_storage(storage, page->swap_addr, page->page_size, offset) != 0) {
        free(page->swap_addr);
        page->swap_addr = NULL;
        page->state = PAGE_STATE_ACTIVE;
        pthread_mutex_unlock(&page->page_lock);
        return -1;
    }
    
    // 释放GPU内存
    free(page->gpu_addr);
    page->gpu_addr = NULL;
    
    page->state = PAGE_STATE_SWAPPED_OUT;
    page->storage_type = storage->type;
    
    // 更新存储使用量
    pthread_mutex_lock(&storage->storage_lock);
    storage->used_size += page->page_size;
    storage->free_size -= page->page_size;
    pthread_mutex_unlock(&storage->storage_lock);
    
    // 更新统计信息
    uint64_t end_time = get_current_time_us();
    g_swap_manager.stats.total_swap_out++;
    g_swap_manager.stats.swap_out_bytes += page->page_size;
    g_swap_manager.stats.swap_out_time_us += (end_time - start_time);
    
    pthread_mutex_unlock(&page->page_lock);
    
    printf("Swapped out page %llu (%zu bytes) to %s in %llu us\n",
           page->page_id, page->page_size,
           storage->type == SWAP_SYSTEM_MEMORY ? "Memory" :
           storage->type == SWAP_SSD_STORAGE ? "SSD" :
           storage->type == SWAP_HDD_STORAGE ? "HDD" : "Storage",
           end_time - start_time);
    
    return 0;
}

// 换入页面
static int swap_in_page(swap_page_t *page) {
    if (!page || page->state != PAGE_STATE_SWAPPED_OUT) {
        return -1;
    }
    
    uint64_t start_time = get_current_time_us();
    
    page->state = PAGE_STATE_SWAPPING_IN;
    
    // 分配GPU内存
    page->gpu_addr = malloc(page->page_size);
    if (!page->gpu_addr) {
        page->state = PAGE_STATE_SWAPPED_OUT;
        return -1;
    }
    
    // 从存储设备读取数据
    swap_storage_t *storage = &g_swap_manager.storages[page->storage_type];
    
    // 简化实现：直接从交换地址复制
    if (page->swap_addr) {
        memcpy(page->gpu_addr, page->swap_addr, page->page_size);
        free(page->swap_addr);
        page->swap_addr = NULL;
    }
    
    page->state = PAGE_STATE_ACTIVE;
    
    // 更新存储使用量
    pthread_mutex_lock(&storage->storage_lock);
    storage->used_size -= page->page_size;
    storage->free_size += page->page_size;
    pthread_mutex_unlock(&storage->storage_lock);
    
    // 更新统计信息
    uint64_t end_time = get_current_time_us();
    g_swap_manager.stats.total_swap_in++;
    g_swap_manager.stats.swap_in_bytes += page->page_size;
    g_swap_manager.stats.swap_in_time_us += (end_time - start_time);
    
    printf("Swapped in page %llu (%zu bytes) from %s in %llu us\n",
           page->page_id, page->page_size,
           storage->type == SWAP_SYSTEM_MEMORY ? "Memory" :
           storage->type == SWAP_SSD_STORAGE ? "SSD" :
           storage->type == SWAP_HDD_STORAGE ? "HDD" : "Storage",
           end_time - start_time);
    
    return 0;
}

// LRU链表操作
static void lru_add_page(lru_list_t *list, swap_page_t *page) {
    pthread_mutex_lock(&list->lru_lock);
    
    page->next = list->head;
    page->prev = NULL;
    
    if (list->head) {
        list->head->prev = page;
    } else {
        list->tail = page;
    }
    
    list->head = page;
    list->count++;
    
    pthread_mutex_unlock(&list->lru_lock);
}

static void lru_remove_page(lru_list_t *list, swap_page_t *page) {
    pthread_mutex_lock(&list->lru_lock);
    
    if (page->prev) {
        page->prev->next = page->next;
    } else {
        list->head = page->next;
    }
    
    if (page->next) {
        page->next->prev = page->prev;
    } else {
        list->tail = page->prev;
    }
    
    page->next = NULL;
    page->prev = NULL;
    list->count--;
    
    pthread_mutex_unlock(&list->lru_lock);
}

static void lru_move_to_head(lru_list_t *list, swap_page_t *page) {
    lru_remove_page(list, page);
    lru_add_page(list, page);
}

static swap_page_t* lru_get_tail(lru_list_t *list) {
    pthread_mutex_lock(&list->lru_lock);
    swap_page_t *tail = list->tail;
    pthread_mutex_unlock(&list->lru_lock);
    return tail;
}

// 更新访问模式
static void update_access_pattern(swap_page_t *page) {
    // 简化的访问模式检测
    if (page->access_count < 10) {
        page->pattern = ACCESS_SEQUENTIAL;
    } else if (page->access_count % 2 == 0) {
        page->pattern = ACCESS_TEMPORAL;
    } else {
        page->pattern = ACCESS_RANDOM;
    }
}

// 预取页面
static void prefetch_pages(swap_page_t *current_page) {
    // 简化的预取策略：预取相邻页面
    for (uint32_t i = 1; i <= g_swap_manager.prefetch_window; i++) {
        uint64_t next_page_id = current_page->page_id + i;
        if (next_page_id <= g_swap_manager.page_table_size) {
            swap_page_t *next_page = &g_swap_manager.page_table[next_page_id - 1];
            if (next_page->state == PAGE_STATE_SWAPPED_OUT) {
                // 异步预取
                next_page->state = PAGE_STATE_PREFETCHING;
                // 这里应该启动异步预取任务
                g_swap_manager.stats.total_prefetch++;
            }
        }
    }
}

// 选择最佳存储设备
static swap_storage_t* select_best_storage(size_t size) {
    swap_storage_t *best = NULL;
    uint32_t best_score = 0;
    
    for (int i = 0; i < 5; i++) {
        swap_storage_t *storage = &g_swap_manager.storages[i];
        if (!storage->is_available || storage->free_size < size) {
            continue;
        }
        
        // 计算存储评分（带宽权重更高）
        uint32_t score = storage->bandwidth_mbps / (storage->latency_us / 100 + 1);
        
        if (score > best_score) {
            best_score = score;
            best = storage;
        }
    }
    
    return best;
}

// 写入存储设备
static int write_to_storage(swap_storage_t *storage, const void *data, size_t size, off_t offset) {
    pthread_mutex_lock(&storage->storage_lock);
    
    ssize_t written = pwrite(storage->fd, data, size, offset);
    
    pthread_mutex_unlock(&storage->storage_lock);
    
    return (written == (ssize_t)size) ? 0 : -1;
}

// 从存储设备读取
static int read_from_storage(swap_storage_t *storage, void *data, size_t size, off_t offset) {
    pthread_mutex_lock(&storage->storage_lock);
    
    ssize_t read_bytes = pread(storage->fd, data, size, offset);
    
    pthread_mutex_unlock(&storage->storage_lock);
    
    return (read_bytes == (ssize_t)size) ? 0 : -1;
}

// 交换工作线程
static void* swap_worker_thread(void *arg) {
    (void)arg;
    
    while (!g_swap_manager.shutdown) {
        usleep(100000); // 100ms
        
        // 检查内存使用情况
        double usage_ratio = (double)g_swap_manager.current_gpu_memory / g_swap_manager.max_gpu_memory;
        
        if (usage_ratio > g_swap_manager.swap_threshold) {
            // 执行后台交换
            swap_page_t *victim = lru_get_tail(&g_swap_manager.active_list);
            if (victim && !victim->is_pinned) {
                swap_out_page(victim);
            }
        }
    }
    
    return NULL;
}

// 获取当前时间（微秒）
static uint64_t get_current_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

// 打印交换统计信息
void print_swap_stats(void) {
    pthread_mutex_lock(&g_swap_manager.manager_lock);
    
    printf("\n=== Swap Statistics ===\n");
    printf("Total Swap Out: %llu\n", g_swap_manager.stats.total_swap_out);
    printf("Total Swap In: %llu\n", g_swap_manager.stats.total_swap_in);
    printf("Total Prefetch: %llu\n", g_swap_manager.stats.total_prefetch);
    printf("Swap Out Bytes: %llu MB\n", g_swap_manager.stats.swap_out_bytes / (1024*1024));
    printf("Swap In Bytes: %llu MB\n", g_swap_manager.stats.swap_in_bytes / (1024*1024));
    printf("Page Faults: %u\n", g_swap_manager.stats.page_faults);
    printf("Prefetch Hits: %u\n", g_swap_manager.stats.prefetch_hits);
    printf("Prefetch Misses: %u\n", g_swap_manager.stats.prefetch_misses);
    
    if (g_swap_manager.stats.total_swap_out > 0) {
        printf("Avg Swap Out Speed: %.2f MB/s\n",
               (double)g_swap_manager.stats.swap_out_bytes / 
               (g_swap_manager.stats.swap_out_time_us / 1000000.0) / (1024*1024));
    }
    
    if (g_swap_manager.stats.total_swap_in > 0) {
        printf("Avg Swap In Speed: %.2f MB/s\n",
               (double)g_swap_manager.stats.swap_in_bytes / 
               (g_swap_manager.stats.swap_in_time_us / 1000000.0) / (1024*1024));
    }
    
    printf("Current GPU Memory: %zu MB / %zu MB (%.1f%%)\n",
           g_swap_manager.current_gpu_memory / (1024*1024),
           g_swap_manager.max_gpu_memory / (1024*1024),
           (double)g_swap_manager.current_gpu_memory / g_swap_manager.max_gpu_memory * 100);
    
    printf("Active Pages: %zu\n", g_swap_manager.active_list.count);
    printf("Inactive Pages: %zu\n", g_swap_manager.inactive_list.count);
    
    printf("======================\n\n");
    
    pthread_mutex_unlock(&g_swap_manager.manager_lock);
}

// 清理交换系统
void cleanup_swap_system(void) {
    g_swap_manager.shutdown = true;
    
    // 等待工作线程结束
    pthread_join(g_swap_manager.swap_thread, NULL);
    
    // 清理存储设备
    for (int i = 0; i < 5; i++) {
        swap_storage_t *storage = &g_swap_manager.storages[i];
        if (storage->is_available) {
            close(storage->fd);
            unlink(storage->path);
            free(storage->path);
            pthread_mutex_destroy(&storage->storage_lock);
        }
    }
    
    // 清理页面表
    if (g_swap_manager.page_table) {
        for (size_t i = 0; i < g_swap_manager.page_table_size; i++) {
            swap_page_t *page = &g_swap_manager.page_table[i];
            if (page->page_id != 0) {
                if (page->gpu_addr) {
                    free(page->gpu_addr);
                }
                if (page->swap_addr) {
                    free(page->swap_addr);
                }
                pthread_mutex_destroy(&page->page_lock);
            }
        }
        free(g_swap_manager.page_table);
    }
    
    // 清理锁
    pthread_mutex_destroy(&g_swap_manager.active_list.lru_lock);
    pthread_mutex_destroy(&g_swap_manager.inactive_list.lru_lock);
    pthread_mutex_destroy(&g_swap_manager.manager_lock);
    
    printf("Swap system cleaned up\n");
}