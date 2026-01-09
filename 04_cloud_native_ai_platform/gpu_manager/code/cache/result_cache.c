#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sys/time.h>
#include <openssl/sha.h>

#include "result_cache.h"
#include "../common/gpu_common.h"

// 缓存管理器全局实例
static cache_manager_t g_cache_manager = {0};
static bool g_cache_initialized = false;
static pthread_rwlock_t g_cache_rwlock = PTHREAD_RWLOCK_INITIALIZER;
static pthread_mutex_t g_stats_mutex = PTHREAD_MUTEX_INITIALIZER;

// 哈希表实现
static uint32_t hash_function(const char *key) {
    uint32_t hash = 5381;
    int c;
    
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }
    
    return hash % CACHE_HASH_TABLE_SIZE;
}

// 计算数据的SHA256哈希
static void compute_data_hash(const void *data, size_t size, char *hash_str) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, data, size);
    SHA256_Final(hash, &sha256);
    
    // 转换为十六进制字符串
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        sprintf(hash_str + (i * 2), "%02x", hash[i]);
    }
    hash_str[SHA256_DIGEST_LENGTH * 2] = '\0';
}

// 创建缓存键
static void create_cache_key(const cache_request_t *request, char *key, size_t key_size) {
    char data_hash[SHA256_DIGEST_LENGTH * 2 + 1];
    
    if (request->input_data && request->input_size > 0) {
        compute_data_hash(request->input_data, request->input_size, data_hash);
    } else {
        strcpy(data_hash, "no_input");
    }
    
    snprintf(key, key_size, "%s_%s_%s", 
             request->operation_type, request->parameters, data_hash);
}

// 创建缓存条目
static cache_entry_t* create_cache_entry(const char *key, const void *data, 
                                        size_t data_size, uint32_t ttl_seconds) {
    cache_entry_t *entry = malloc(sizeof(cache_entry_t));
    if (!entry) {
        return NULL;
    }
    
    memset(entry, 0, sizeof(cache_entry_t));
    
    // 复制键
    strncpy(entry->key, key, sizeof(entry->key) - 1);
    
    // 复制数据
    entry->data = malloc(data_size);
    if (!entry->data) {
        free(entry);
        return NULL;
    }
    memcpy(entry->data, data, data_size);
    entry->data_size = data_size;
    
    // 设置时间戳
    entry->created_time = gpu_get_timestamp_us();
    entry->last_access_time = entry->created_time;
    entry->ttl_seconds = ttl_seconds;
    entry->access_count = 0;
    entry->next = NULL;
    
    return entry;
}

// 释放缓存条目
static void free_cache_entry(cache_entry_t *entry) {
    if (entry) {
        free(entry->data);
        free(entry);
    }
}

// 检查缓存条目是否过期
static bool is_entry_expired(const cache_entry_t *entry) {
    if (entry->ttl_seconds == 0) {
        return false; // 永不过期
    }
    
    uint64_t now = gpu_get_timestamp_us();
    uint64_t age_seconds = (now - entry->created_time) / 1000000;
    
    return age_seconds >= entry->ttl_seconds;
}

// 从哈希表中查找条目
static cache_entry_t* find_cache_entry(const char *key) {
    uint32_t hash = hash_function(key);
    cache_entry_t *entry = g_cache_manager.hash_table[hash];
    
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            if (is_entry_expired(entry)) {
                return NULL; // 已过期
            }
            return entry;
        }
        entry = entry->next;
    }
    
    return NULL;
}

// 向哈希表中插入条目
static cache_result_t insert_cache_entry(cache_entry_t *entry) {
    uint32_t hash = hash_function(entry->key);
    
    // 检查是否已存在
    cache_entry_t *existing = find_cache_entry(entry->key);
    if (existing) {
        return CACHE_ERROR_KEY_EXISTS;
    }
    
    // 检查缓存容量
    if (g_cache_manager.current_entries >= g_cache_manager.config.max_entries) {
        // 需要执行LRU淘汰
        if (cache_evict_lru() != CACHE_SUCCESS) {
            return CACHE_ERROR_EVICTION_FAILED;
        }
    }
    
    // 插入到链表头部
    entry->next = g_cache_manager.hash_table[hash];
    g_cache_manager.hash_table[hash] = entry;
    
    g_cache_manager.current_entries++;
    g_cache_manager.current_memory_usage += entry->data_size;
    
    return CACHE_SUCCESS;
}

// 从哈希表中移除条目
static cache_result_t remove_cache_entry(const char *key) {
    uint32_t hash = hash_function(key);
    cache_entry_t *entry = g_cache_manager.hash_table[hash];
    cache_entry_t *prev = NULL;
    
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            // 找到条目，移除它
            if (prev) {
                prev->next = entry->next;
            } else {
                g_cache_manager.hash_table[hash] = entry->next;
            }
            
            g_cache_manager.current_entries--;
            g_cache_manager.current_memory_usage -= entry->data_size;
            
            free_cache_entry(entry);
            return CACHE_SUCCESS;
        }
        prev = entry;
        entry = entry->next;
    }
    
    return CACHE_ERROR_KEY_NOT_FOUND;
}

// LRU淘汰算法
cache_result_t cache_evict_lru(void) {
    cache_entry_t *lru_entry = NULL;
    uint64_t oldest_access = UINT64_MAX;
    
    // 遍历所有条目找到最久未访问的
    for (int i = 0; i < CACHE_HASH_TABLE_SIZE; i++) {
        cache_entry_t *entry = g_cache_manager.hash_table[i];
        while (entry) {
            if (entry->last_access_time < oldest_access) {
                oldest_access = entry->last_access_time;
                lru_entry = entry;
            }
            entry = entry->next;
        }
    }
    
    if (lru_entry) {
        gpu_log(GPU_LOG_DEBUG, "Evicting LRU cache entry: %s", lru_entry->key);
        return remove_cache_entry(lru_entry->key);
    }
    
    return CACHE_ERROR_NO_ENTRIES_TO_EVICT;
}

// 初始化结果缓存
cache_result_t result_cache_init(const cache_config_t *config) {
    if (!config) {
        return CACHE_ERROR_INVALID_PARAM;
    }
    
    if (g_cache_initialized) {
        gpu_log(GPU_LOG_WARNING, "Result cache already initialized");
        return CACHE_SUCCESS;
    }
    
    memset(&g_cache_manager, 0, sizeof(g_cache_manager));
    g_cache_manager.config = *config;
    
    // 验证配置参数
    if (g_cache_manager.config.max_entries == 0) {
        g_cache_manager.config.max_entries = 1000; // 默认1000个条目
    }
    if (g_cache_manager.config.max_memory_mb == 0) {
        g_cache_manager.config.max_memory_mb = 512; // 默认512MB
    }
    if (g_cache_manager.config.default_ttl_seconds == 0) {
        g_cache_manager.config.default_ttl_seconds = 3600; // 默认1小时
    }
    
    g_cache_initialized = true;
    
    gpu_log(GPU_LOG_INFO, "Result cache initialized: max_entries=%u, max_memory=%uMB", 
            g_cache_manager.config.max_entries, g_cache_manager.config.max_memory_mb);
    
    return CACHE_SUCCESS;
}

// 清理结果缓存
void result_cache_cleanup(void) {
    if (!g_cache_initialized) {
        return;
    }
    
    pthread_rwlock_wrlock(&g_cache_rwlock);
    
    // 清理所有缓存条目
    for (int i = 0; i < CACHE_HASH_TABLE_SIZE; i++) {
        cache_entry_t *entry = g_cache_manager.hash_table[i];
        while (entry) {
            cache_entry_t *next = entry->next;
            free_cache_entry(entry);
            entry = next;
        }
        g_cache_manager.hash_table[i] = NULL;
    }
    
    memset(&g_cache_manager, 0, sizeof(g_cache_manager));
    g_cache_initialized = false;
    
    pthread_rwlock_unlock(&g_cache_rwlock);
    
    gpu_log(GPU_LOG_INFO, "Result cache cleaned up");
}

// 存储计算结果
cache_result_t result_cache_store(const cache_request_t *request, 
                                 const void *result_data, size_t result_size) {
    if (!g_cache_initialized || !request || !result_data || result_size == 0) {
        return CACHE_ERROR_INVALID_PARAM;
    }
    
    char cache_key[CACHE_MAX_KEY_LEN];
    create_cache_key(request, cache_key, sizeof(cache_key));
    
    // 检查内存限制
    size_t max_memory_bytes = (size_t)g_cache_manager.config.max_memory_mb * 1024 * 1024;
    if (g_cache_manager.current_memory_usage + result_size > max_memory_bytes) {
        gpu_log(GPU_LOG_WARNING, "Cache memory limit exceeded, evicting entries");
        // 可以实现更智能的内存管理策略
        while (g_cache_manager.current_memory_usage + result_size > max_memory_bytes && 
               g_cache_manager.current_entries > 0) {
            if (cache_evict_lru() != CACHE_SUCCESS) {
                break;
            }
        }
    }
    
    uint32_t ttl = request->ttl_seconds > 0 ? request->ttl_seconds : 
                   g_cache_manager.config.default_ttl_seconds;
    
    cache_entry_t *entry = create_cache_entry(cache_key, result_data, result_size, ttl);
    if (!entry) {
        return CACHE_ERROR_OUT_OF_MEMORY;
    }
    
    pthread_rwlock_wrlock(&g_cache_rwlock);
    
    cache_result_t result = insert_cache_entry(entry);
    if (result != CACHE_SUCCESS) {
        free_cache_entry(entry);
    } else {
        pthread_mutex_lock(&g_stats_mutex);
        g_cache_manager.stats.total_stores++;
        pthread_mutex_unlock(&g_stats_mutex);
        
        gpu_log(GPU_LOG_DEBUG, "Cached result for key: %s (size: %zu bytes)", 
                cache_key, result_size);
    }
    
    pthread_rwlock_unlock(&g_cache_rwlock);
    
    return result;
}

// 检索计算结果
cache_result_t result_cache_retrieve(const cache_request_t *request, 
                                    void **result_data, size_t *result_size) {
    if (!g_cache_initialized || !request || !result_data || !result_size) {
        return CACHE_ERROR_INVALID_PARAM;
    }
    
    char cache_key[CACHE_MAX_KEY_LEN];
    create_cache_key(request, cache_key, sizeof(cache_key));
    
    pthread_rwlock_rdlock(&g_cache_rwlock);
    
    cache_entry_t *entry = find_cache_entry(cache_key);
    if (!entry) {
        pthread_rwlock_unlock(&g_cache_rwlock);
        
        pthread_mutex_lock(&g_stats_mutex);
        g_cache_manager.stats.cache_misses++;
        pthread_mutex_unlock(&g_stats_mutex);
        
        return CACHE_ERROR_KEY_NOT_FOUND;
    }
    
    // 复制数据
    *result_data = malloc(entry->data_size);
    if (!*result_data) {
        pthread_rwlock_unlock(&g_cache_rwlock);
        return CACHE_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(*result_data, entry->data, entry->data_size);
    *result_size = entry->data_size;
    
    // 更新访问统计
    entry->last_access_time = gpu_get_timestamp_us();
    entry->access_count++;
    
    pthread_rwlock_unlock(&g_cache_rwlock);
    
    pthread_mutex_lock(&g_stats_mutex);
    g_cache_manager.stats.cache_hits++;
    pthread_mutex_unlock(&g_stats_mutex);
    
    gpu_log(GPU_LOG_DEBUG, "Cache hit for key: %s (size: %zu bytes)", 
            cache_key, *result_size);
    
    return CACHE_SUCCESS;
}

// 使缓存条目无效
cache_result_t result_cache_invalidate(const cache_request_t *request) {
    if (!g_cache_initialized || !request) {
        return CACHE_ERROR_INVALID_PARAM;
    }
    
    char cache_key[CACHE_MAX_KEY_LEN];
    create_cache_key(request, cache_key, sizeof(cache_key));
    
    pthread_rwlock_wrlock(&g_cache_rwlock);
    
    cache_result_t result = remove_cache_entry(cache_key);
    
    if (result == CACHE_SUCCESS) {
        pthread_mutex_lock(&g_stats_mutex);
        g_cache_manager.stats.total_invalidations++;
        pthread_mutex_unlock(&g_stats_mutex);
        
        gpu_log(GPU_LOG_DEBUG, "Invalidated cache entry: %s", cache_key);
    }
    
    pthread_rwlock_unlock(&g_cache_rwlock);
    
    return result;
}

// 清空所有缓存
cache_result_t result_cache_clear(void) {
    if (!g_cache_initialized) {
        return CACHE_ERROR_NOT_INITIALIZED;
    }
    
    pthread_rwlock_wrlock(&g_cache_rwlock);
    
    int cleared_entries = 0;
    
    // 清理所有缓存条目
    for (int i = 0; i < CACHE_HASH_TABLE_SIZE; i++) {
        cache_entry_t *entry = g_cache_manager.hash_table[i];
        while (entry) {
            cache_entry_t *next = entry->next;
            free_cache_entry(entry);
            cleared_entries++;
            entry = next;
        }
        g_cache_manager.hash_table[i] = NULL;
    }
    
    g_cache_manager.current_entries = 0;
    g_cache_manager.current_memory_usage = 0;
    
    pthread_rwlock_unlock(&g_cache_rwlock);
    
    gpu_log(GPU_LOG_INFO, "Cleared %d cache entries", cleared_entries);
    
    return CACHE_SUCCESS;
}

// 获取缓存统计信息
cache_result_t result_cache_get_stats(cache_stats_t *stats) {
    if (!g_cache_initialized || !stats) {
        return CACHE_ERROR_INVALID_PARAM;
    }
    
    pthread_mutex_lock(&g_stats_mutex);
    
    *stats = g_cache_manager.stats;
    stats->current_entries = g_cache_manager.current_entries;
    stats->current_memory_usage = g_cache_manager.current_memory_usage;
    
    // 计算命中率
    uint64_t total_requests = stats->cache_hits + stats->cache_misses;
    if (total_requests > 0) {
        stats->hit_rate = (double)stats->cache_hits / total_requests;
    } else {
        stats->hit_rate = 0.0;
    }
    
    pthread_mutex_unlock(&g_stats_mutex);
    
    return CACHE_SUCCESS;
}

// 获取错误字符串
const char* cache_get_error_string(cache_result_t error) {
    static const char* error_strings[] = {
        [CACHE_SUCCESS] = "Success",
        [CACHE_ERROR_INVALID_PARAM] = "Invalid parameter",
        [CACHE_ERROR_NOT_INITIALIZED] = "Cache not initialized",
        [CACHE_ERROR_OUT_OF_MEMORY] = "Out of memory",
        [CACHE_ERROR_KEY_NOT_FOUND] = "Key not found",
        [CACHE_ERROR_KEY_EXISTS] = "Key already exists",
        [CACHE_ERROR_EVICTION_FAILED] = "Eviction failed",
        [CACHE_ERROR_NO_ENTRIES_TO_EVICT] = "No entries to evict"
    };
    
    if (error >= 0 && error < sizeof(error_strings) / sizeof(error_strings[0])) {
        return error_strings[error];
    }
    return "Unknown error";
}