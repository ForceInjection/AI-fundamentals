/**
 * GPU内存热迁移实现
 * 功能：支持GPU内存的在线迁移，无需停止应用程序
 * 特性：增量迁移、一致性保障、故障恢复
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdbool.h>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#define MAX_MIGRATION_SESSIONS 32
#define MIGRATION_CHUNK_SIZE (1024 * 1024)  // 1MB chunks
#define MAX_DIRTY_PAGES 10000
#define MIGRATION_TIMEOUT_SEC 300  // 5 minutes

// 迁移状态
typedef enum {
    MIGRATION_IDLE = 0,
    MIGRATION_PREPARING,
    MIGRATION_COPYING,
    MIGRATION_SYNCING,
    MIGRATION_FINALIZING,
    MIGRATION_COMPLETED,
    MIGRATION_FAILED,
    MIGRATION_CANCELLED
} migration_state_t;

// 迁移策略
typedef enum {
    MIGRATION_STRATEGY_STOP_AND_COPY = 0,  // 停止并复制
    MIGRATION_STRATEGY_PRE_COPY,           // 预复制
    MIGRATION_STRATEGY_POST_COPY,          // 后复制
    MIGRATION_STRATEGY_HYBRID              // 混合策略
} migration_strategy_t;

// 脏页跟踪
typedef struct {
    uint64_t page_addr;     // 页面地址
    size_t page_size;       // 页面大小
    uint32_t dirty_count;   // 脏页计数
    time_t last_modified;   // 最后修改时间
    bool is_dirty;          // 是否为脏页
} dirty_page_t;

// 迁移统计信息
typedef struct {
    uint64_t total_bytes;           // 总字节数
    uint64_t transferred_bytes;     // 已传输字节数
    uint64_t dirty_bytes;           // 脏页字节数
    uint32_t total_pages;           // 总页数
    uint32_t transferred_pages;     // 已传输页数
    uint32_t dirty_pages;           // 脏页数
    uint32_t iterations;            // 迁移轮次
    double transfer_rate_mbps;      // 传输速率 MB/s
    uint64_t downtime_us;           // 停机时间 微秒
    time_t start_time;              // 开始时间
    time_t end_time;                // 结束时间
} migration_stats_t;

// 迁移会话
typedef struct {
    uint32_t session_id;            // 会话ID
    migration_state_t state;        // 迁移状态
    migration_strategy_t strategy;  // 迁移策略
    
    // 源和目标信息
    int source_gpu;                 // 源GPU
    int target_gpu;                 // 目标GPU
    void *source_ptr;               // 源内存指针
    void *target_ptr;               // 目标内存指针
    size_t memory_size;             // 内存大小
    
    // 脏页跟踪
    dirty_page_t dirty_pages[MAX_DIRTY_PAGES];
    uint32_t dirty_page_count;      // 脏页数量
    pthread_mutex_t dirty_lock;     // 脏页锁
    
    // 迁移控制
    pthread_t migration_thread;     // 迁移线程
    pthread_mutex_t session_lock;   // 会话锁
    pthread_cond_t state_cond;      // 状态条件变量
    bool is_active;                 // 是否活跃
    bool stop_requested;            // 是否请求停止
    
    // 统计信息
    migration_stats_t stats;        // 迁移统计
    
    // 回调函数
    void (*progress_callback)(uint32_t session_id, double progress);
    void (*completion_callback)(uint32_t session_id, int result);
} migration_session_t;

// 热迁移管理器
typedef struct {
    migration_session_t sessions[MAX_MIGRATION_SESSIONS];
    uint32_t next_session_id;       // 下一个会话ID
    pthread_mutex_t manager_lock;   // 管理器锁
    bool is_initialized;            // 是否已初始化
} hot_migration_manager_t;

// 全局热迁移管理器
static hot_migration_manager_t g_migration_manager = {0};

// 获取当前时间（微秒）
static uint64_t get_current_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

// 标记页面为脏页
static void mark_page_dirty(migration_session_t *session, uint64_t page_addr, size_t page_size) {
    pthread_mutex_lock(&session->dirty_lock);
    
    // 查找现有脏页
    for (uint32_t i = 0; i < session->dirty_page_count; i++) {
        if (session->dirty_pages[i].page_addr == page_addr) {
            session->dirty_pages[i].dirty_count++;
            session->dirty_pages[i].last_modified = time(NULL);
            session->dirty_pages[i].is_dirty = true;
            pthread_mutex_unlock(&session->dirty_lock);
            return;
        }
    }
    
    // 添加新脏页
    if (session->dirty_page_count < MAX_DIRTY_PAGES) {
        dirty_page_t *page = &session->dirty_pages[session->dirty_page_count];
        page->page_addr = page_addr;
        page->page_size = page_size;
        page->dirty_count = 1;
        page->last_modified = time(NULL);
        page->is_dirty = true;
        session->dirty_page_count++;
    }
    
    pthread_mutex_unlock(&session->dirty_lock);
}

// 复制内存块
static int copy_memory_chunk(void *src, void *dst, size_t size, int src_gpu, int dst_gpu) {
#ifdef CUDA_ENABLED
    cudaError_t err;
    
    // 设置源GPU
    err = cudaSetDevice(src_gpu);
    if (err != cudaSuccess) {
        return -1;
    }
    
    // 执行内存复制
    if (src_gpu == dst_gpu) {
        // 同一GPU内复制
        err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    } else {
        // 跨GPU复制
        err = cudaMemcpyPeer(dst, dst_gpu, src, src_gpu, size);
    }
    
    return (err == cudaSuccess) ? 0 : -1;
#else
    // CUDA未启用时的模拟实现
    printf("Simulating memory copy from GPU %d to GPU %d, size %zu\n", 
           src_gpu, dst_gpu, size);
    memcpy(dst, src, size);
    return 0;
#endif
}

// 预复制策略实现
static int pre_copy_migration(migration_session_t *session) {
    uint64_t start_time = get_current_time_us();
    size_t chunk_size = MIGRATION_CHUNK_SIZE;
    size_t remaining = session->memory_size;
    char *src_ptr = (char*)session->source_ptr;
    char *dst_ptr = (char*)session->target_ptr;
    
    session->stats.iterations = 0;
    
    while (remaining > 0 && !session->stop_requested) {
        session->stats.iterations++;
        size_t current_chunk = (remaining < chunk_size) ? remaining : chunk_size;
        
        // 复制当前块
        if (copy_memory_chunk(src_ptr, dst_ptr, current_chunk, 
                             session->source_gpu, session->target_gpu) != 0) {
            return -1;
        }
        
        session->stats.transferred_bytes += current_chunk;
        session->stats.transferred_pages++;
        
        // 更新进度
        double progress = (double)session->stats.transferred_bytes / session->memory_size;
        if (session->progress_callback) {
            session->progress_callback(session->session_id, progress);
        }
        
        src_ptr += current_chunk;
        dst_ptr += current_chunk;
        remaining -= current_chunk;
        
        // 检查脏页
        pthread_mutex_lock(&session->dirty_lock);
        session->stats.dirty_pages = session->dirty_page_count;
        pthread_mutex_unlock(&session->dirty_lock);
        
        // 如果脏页太多，重新开始
        if (session->stats.dirty_pages > session->stats.total_pages * 0.1) {
            printf("Too many dirty pages (%u), restarting iteration\n", 
                   session->stats.dirty_pages);
            remaining = session->memory_size;
            src_ptr = (char*)session->source_ptr;
            dst_ptr = (char*)session->target_ptr;
            session->stats.transferred_bytes = 0;
            session->stats.transferred_pages = 0;
            
            // 清理脏页
            pthread_mutex_lock(&session->dirty_lock);
            session->dirty_page_count = 0;
            pthread_mutex_unlock(&session->dirty_lock);
        }
        
        usleep(1000);  // 1ms延迟，避免过度占用资源
    }
    
    // 最终同步阶段
    session->state = MIGRATION_SYNCING;
    uint64_t sync_start = get_current_time_us();
    
    // 复制剩余脏页
    pthread_mutex_lock(&session->dirty_lock);
    for (uint32_t i = 0; i < session->dirty_page_count; i++) {
        dirty_page_t *page = &session->dirty_pages[i];
        if (page->is_dirty) {
            copy_memory_chunk((void*)page->page_addr, 
                            (void*)(page->page_addr - (uint64_t)session->source_ptr + 
                                   (uint64_t)session->target_ptr),
                            page->page_size, session->source_gpu, session->target_gpu);
            page->is_dirty = false;
        }
    }
    pthread_mutex_unlock(&session->dirty_lock);
    
    uint64_t sync_end = get_current_time_us();
    session->stats.downtime_us = sync_end - sync_start;
    
    // 计算传输速率
    uint64_t total_time = get_current_time_us() - start_time;
    if (total_time > 0) {
        session->stats.transfer_rate_mbps = 
            (double)session->stats.transferred_bytes / (total_time / 1000000.0) / (1024 * 1024);
    }
    
    return 0;
}

// 停止并复制策略实现
static int stop_and_copy_migration(migration_session_t *session) {
    uint64_t start_time = get_current_time_us();
    
    // 直接复制整个内存区域
    if (copy_memory_chunk(session->source_ptr, session->target_ptr, 
                         session->memory_size, session->source_gpu, 
                         session->target_gpu) != 0) {
        return -1;
    }
    
    session->stats.transferred_bytes = session->memory_size;
    session->stats.transferred_pages = session->stats.total_pages;
    session->stats.iterations = 1;
    
    uint64_t end_time = get_current_time_us();
    session->stats.downtime_us = end_time - start_time;
    
    // 计算传输速率
    if (session->stats.downtime_us > 0) {
        session->stats.transfer_rate_mbps = 
            (double)session->memory_size / (session->stats.downtime_us / 1000000.0) / (1024 * 1024);
    }
    
    return 0;
}

// 迁移线程函数
static void* migration_thread_func(void *arg) {
    migration_session_t *session = (migration_session_t*)arg;
    int result = 0;
    
    pthread_mutex_lock(&session->session_lock);
    session->state = MIGRATION_PREPARING;
    session->stats.start_time = time(NULL);
    pthread_cond_broadcast(&session->state_cond);
    pthread_mutex_unlock(&session->session_lock);
    
    // 准备阶段
    printf("Starting migration session %u: %zu bytes from GPU %d to GPU %d\n", 
           session->session_id, session->memory_size, 
           session->source_gpu, session->target_gpu);
    
    // 分配目标内存
#ifdef CUDA_ENABLED
    cudaSetDevice(session->target_gpu);
    cudaError_t err = cudaMalloc(&session->target_ptr, session->memory_size);
    if (err != cudaSuccess) {
        result = -1;
        goto cleanup;
    }
#else
    // CUDA未启用时的模拟实现
    session->target_ptr = malloc(session->memory_size);
    if (!session->target_ptr) {
        result = -1;
        goto cleanup;
    }
#endif
    
    pthread_mutex_lock(&session->session_lock);
    session->state = MIGRATION_COPYING;
    pthread_cond_broadcast(&session->state_cond);
    pthread_mutex_unlock(&session->session_lock);
    
    // 执行迁移策略
    switch (session->strategy) {
        case MIGRATION_STRATEGY_STOP_AND_COPY:
            result = stop_and_copy_migration(session);
            break;
        case MIGRATION_STRATEGY_PRE_COPY:
            result = pre_copy_migration(session);
            break;
        default:
            result = -1;
            break;
    }
    
cleanup:
    pthread_mutex_lock(&session->session_lock);
    
    if (result == 0 && !session->stop_requested) {
        session->state = MIGRATION_COMPLETED;
        printf("Migration session %u completed successfully\n", session->session_id);
    } else {
        session->state = MIGRATION_FAILED;
        printf("Migration session %u failed\n", session->session_id);
        
        // 清理目标内存
        if (session->target_ptr) {
#ifdef CUDA_ENABLED
            cudaSetDevice(session->target_gpu);
            cudaFree(session->target_ptr);
#else
            free(session->target_ptr);
#endif
            session->target_ptr = NULL;
        }
    }
    
    session->stats.end_time = time(NULL);
    session->is_active = false;
    
    pthread_cond_broadcast(&session->state_cond);
    pthread_mutex_unlock(&session->session_lock);
    
    // 调用完成回调
    if (session->completion_callback) {
        session->completion_callback(session->session_id, result);
    }
    
    return NULL;
}

// 初始化热迁移管理器
int init_hot_migration_manager(void) {
    if (g_migration_manager.is_initialized) {
        return 0;
    }
    
    memset(&g_migration_manager, 0, sizeof(hot_migration_manager_t));
    
    if (pthread_mutex_init(&g_migration_manager.manager_lock, NULL) != 0) {
        return -1;
    }
    
    // 初始化所有会话
    for (int i = 0; i < MAX_MIGRATION_SESSIONS; i++) {
        migration_session_t *session = &g_migration_manager.sessions[i];
        pthread_mutex_init(&session->session_lock, NULL);
        pthread_mutex_init(&session->dirty_lock, NULL);
        pthread_cond_init(&session->state_cond, NULL);
    }
    
    g_migration_manager.next_session_id = 1;
    g_migration_manager.is_initialized = true;
    
    printf("Hot migration manager initialized\n");
    return 0;
}

// 开始内存迁移
uint32_t start_memory_migration(void *source_ptr, size_t size, 
                               int source_gpu, int target_gpu,
                               migration_strategy_t strategy,
                               void (*progress_cb)(uint32_t, double),
                               void (*completion_cb)(uint32_t, int)) {
    if (!g_migration_manager.is_initialized) {
        return 0;
    }
    
    pthread_mutex_lock(&g_migration_manager.manager_lock);
    
    // 查找空闲会话
    migration_session_t *session = NULL;
    for (int i = 0; i < MAX_MIGRATION_SESSIONS; i++) {
        if (!g_migration_manager.sessions[i].is_active) {
            session = &g_migration_manager.sessions[i];
            break;
        }
    }
    
    if (!session) {
        pthread_mutex_unlock(&g_migration_manager.manager_lock);
        return 0;  // 没有可用会话
    }
    
    // 初始化会话
    memset(session, 0, sizeof(migration_session_t));
    session->session_id = g_migration_manager.next_session_id++;
    session->state = MIGRATION_IDLE;
    session->strategy = strategy;
    session->source_gpu = source_gpu;
    session->target_gpu = target_gpu;
    session->source_ptr = source_ptr;
    session->memory_size = size;
    session->is_active = true;
    session->progress_callback = progress_cb;
    session->completion_callback = completion_cb;
    
    // 初始化统计信息
    session->stats.total_bytes = size;
    session->stats.total_pages = (size + 4095) / 4096;  // 假设4KB页面
    
    pthread_mutex_unlock(&g_migration_manager.manager_lock);
    
    // 启动迁移线程
    if (pthread_create(&session->migration_thread, NULL, 
                      migration_thread_func, session) != 0) {
        session->is_active = false;
        return 0;
    }
    
    return session->session_id;
}

// 取消内存迁移
int cancel_memory_migration(uint32_t session_id) {
    pthread_mutex_lock(&g_migration_manager.manager_lock);
    
    migration_session_t *session = NULL;
    for (int i = 0; i < MAX_MIGRATION_SESSIONS; i++) {
        if (g_migration_manager.sessions[i].session_id == session_id &&
            g_migration_manager.sessions[i].is_active) {
            session = &g_migration_manager.sessions[i];
            break;
        }
    }
    
    if (!session) {
        pthread_mutex_unlock(&g_migration_manager.manager_lock);
        return -1;
    }
    
    pthread_mutex_lock(&session->session_lock);
    session->stop_requested = true;
    session->state = MIGRATION_CANCELLED;
    pthread_cond_broadcast(&session->state_cond);
    pthread_mutex_unlock(&session->session_lock);
    
    pthread_mutex_unlock(&g_migration_manager.manager_lock);
    
    // 等待线程结束
    pthread_join(session->migration_thread, NULL);
    
    return 0;
}

// 获取迁移状态
migration_state_t get_migration_state(uint32_t session_id) {
    pthread_mutex_lock(&g_migration_manager.manager_lock);
    
    for (int i = 0; i < MAX_MIGRATION_SESSIONS; i++) {
        if (g_migration_manager.sessions[i].session_id == session_id) {
            migration_state_t state = g_migration_manager.sessions[i].state;
            pthread_mutex_unlock(&g_migration_manager.manager_lock);
            return state;
        }
    }
    
    pthread_mutex_unlock(&g_migration_manager.manager_lock);
    return MIGRATION_IDLE;
}

// 获取迁移统计信息
migration_stats_t get_migration_stats(uint32_t session_id) {
    migration_stats_t stats = {0};
    
    pthread_mutex_lock(&g_migration_manager.manager_lock);
    
    for (int i = 0; i < MAX_MIGRATION_SESSIONS; i++) {
        if (g_migration_manager.sessions[i].session_id == session_id) {
            stats = g_migration_manager.sessions[i].stats;
            break;
        }
    }
    
    pthread_mutex_unlock(&g_migration_manager.manager_lock);
    return stats;
}

// 等待迁移完成
int wait_for_migration(uint32_t session_id, int timeout_sec) {
    pthread_mutex_lock(&g_migration_manager.manager_lock);
    
    migration_session_t *session = NULL;
    for (int i = 0; i < MAX_MIGRATION_SESSIONS; i++) {
        if (g_migration_manager.sessions[i].session_id == session_id) {
            session = &g_migration_manager.sessions[i];
            break;
        }
    }
    
    if (!session) {
        pthread_mutex_unlock(&g_migration_manager.manager_lock);
        return -1;
    }
    
    pthread_mutex_lock(&session->session_lock);
    pthread_mutex_unlock(&g_migration_manager.manager_lock);
    
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_sec;
    
    int result = 0;
    while (session->state != MIGRATION_COMPLETED && 
           session->state != MIGRATION_FAILED &&
           session->state != MIGRATION_CANCELLED) {
        
        if (pthread_cond_timedwait(&session->state_cond, 
                                  &session->session_lock, &timeout) != 0) {
            result = -1;  // 超时
            break;
        }
    }
    
    pthread_mutex_unlock(&session->session_lock);
    return result;
}

// 清理热迁移管理器
void cleanup_hot_migration_manager(void) {
    if (!g_migration_manager.is_initialized) {
        return;
    }
    
    // 取消所有活跃的迁移
    for (int i = 0; i < MAX_MIGRATION_SESSIONS; i++) {
        migration_session_t *session = &g_migration_manager.sessions[i];
        if (session->is_active) {
            cancel_memory_migration(session->session_id);
        }
        
        pthread_mutex_destroy(&session->session_lock);
        pthread_mutex_destroy(&session->dirty_lock);
        pthread_cond_destroy(&session->state_cond);
    }
    
    pthread_mutex_destroy(&g_migration_manager.manager_lock);
    g_migration_manager.is_initialized = false;
    
    printf("Hot migration manager cleaned up\n");
}