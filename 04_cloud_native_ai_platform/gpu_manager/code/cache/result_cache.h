#ifndef RESULT_CACHE_H
#define RESULT_CACHE_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// 常量定义
#define CACHE_MAX_KEY_LEN 512
#define CACHE_MAX_OPERATION_TYPE_LEN 64
#define CACHE_MAX_PARAMETERS_LEN 256
#define CACHE_HASH_TABLE_SIZE 1024

// 缓存结果枚举
typedef enum {
    CACHE_SUCCESS = 0,
    CACHE_ERROR_INVALID_PARAM,
    CACHE_ERROR_NOT_INITIALIZED,
    CACHE_ERROR_OUT_OF_MEMORY,
    CACHE_ERROR_KEY_NOT_FOUND,
    CACHE_ERROR_KEY_EXISTS,
    CACHE_ERROR_EVICTION_FAILED,
    CACHE_ERROR_NO_ENTRIES_TO_EVICT
} cache_result_t;

// 缓存策略枚举
typedef enum {
    CACHE_POLICY_LRU = 0,     // 最近最少使用
    CACHE_POLICY_LFU,         // 最少使用频率
    CACHE_POLICY_FIFO,        // 先进先出
    CACHE_POLICY_RANDOM       // 随机淘汰
} cache_policy_t;

// 缓存配置结构
typedef struct {
    uint32_t max_entries;           // 最大缓存条目数
    uint32_t max_memory_mb;         // 最大内存使用量（MB）
    uint32_t default_ttl_seconds;   // 默认生存时间（秒）
    cache_policy_t eviction_policy; // 淘汰策略
    bool enable_compression;        // 是否启用压缩
    bool enable_persistence;        // 是否启用持久化
} cache_config_t;

// 缓存请求结构
typedef struct {
    char operation_type[CACHE_MAX_OPERATION_TYPE_LEN];  // 操作类型
    char parameters[CACHE_MAX_PARAMETERS_LEN];          // 参数字符串
    const void *input_data;                             // 输入数据
    size_t input_size;                                  // 输入数据大小
    uint32_t ttl_seconds;                              // 生存时间（0表示使用默认值）
} cache_request_t;

// 缓存条目结构
typedef struct cache_entry {
    char key[CACHE_MAX_KEY_LEN];        // 缓存键
    void *data;                         // 缓存数据
    size_t data_size;                   // 数据大小
    uint64_t created_time;              // 创建时间戳
    uint64_t last_access_time;          // 最后访问时间戳
    uint32_t access_count;              // 访问次数
    uint32_t ttl_seconds;               // 生存时间
    struct cache_entry *next;           // 链表指针
} cache_entry_t;

// 缓存统计信息
typedef struct {
    uint64_t cache_hits;                // 缓存命中次数
    uint64_t cache_misses;              // 缓存未命中次数
    uint64_t total_stores;              // 总存储次数
    uint64_t total_invalidations;       // 总失效次数
    uint32_t current_entries;           // 当前条目数
    size_t current_memory_usage;        // 当前内存使用量
    double hit_rate;                    // 命中率
} cache_stats_t;

// 缓存管理器结构
typedef struct {
    cache_entry_t *hash_table[CACHE_HASH_TABLE_SIZE];  // 哈希表
    cache_config_t config;                             // 配置
    cache_stats_t stats;                               // 统计信息
    uint32_t current_entries;                          // 当前条目数
    size_t current_memory_usage;                       // 当前内存使用量
    pthread_rwlock_t rwlock;                           // 读写锁
} cache_manager_t;

// 预取预测器结构
typedef struct {
    char pattern[CACHE_MAX_KEY_LEN];    // 访问模式
    uint32_t frequency;                 // 访问频率
    uint64_t last_access;              // 最后访问时间
    double confidence;                  // 预测置信度
} prefetch_predictor_t;

// 核心API函数

/**
 * 初始化结果缓存
 * @param config 缓存配置
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t result_cache_init(const cache_config_t *config);

/**
 * 清理结果缓存
 */
void result_cache_cleanup(void);

/**
 * 存储计算结果到缓存
 * @param request 缓存请求
 * @param result_data 结果数据
 * @param result_size 结果数据大小
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t result_cache_store(const cache_request_t *request, 
                                 const void *result_data, size_t result_size);

/**
 * 从缓存中检索计算结果
 * @param request 缓存请求
 * @param result_data 输出结果数据指针
 * @param result_size 输出结果数据大小
 * @return CACHE_SUCCESS 成功，其他值表示错误
 * @note 调用者负责释放result_data指向的内存
 */
cache_result_t result_cache_retrieve(const cache_request_t *request, 
                                    void **result_data, size_t *result_size);

/**
 * 使缓存条目无效
 * @param request 缓存请求
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t result_cache_invalidate(const cache_request_t *request);

/**
 * 清空所有缓存
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t result_cache_clear(void);

/**
 * 获取缓存统计信息
 * @param stats 输出统计信息
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t result_cache_get_stats(cache_stats_t *stats);

// 缓存管理函数

/**
 * 执行LRU淘汰
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t cache_evict_lru(void);

/**
 * 执行缓存压缩
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t cache_compress_entries(void);

/**
 * 执行缓存持久化
 * @param file_path 持久化文件路径
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t cache_persist_to_file(const char *file_path);

/**
 * 从文件加载缓存
 * @param file_path 缓存文件路径
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t cache_load_from_file(const char *file_path);

// 智能预取函数

/**
 * 初始化预取预测器
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t prefetch_predictor_init(void);

/**
 * 清理预取预测器
 */
void prefetch_predictor_cleanup(void);

/**
 * 记录访问模式
 * @param request 缓存请求
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t prefetch_record_access(const cache_request_t *request);

/**
 * 预测下一个可能的请求
 * @param current_request 当前请求
 * @param predicted_request 输出预测的请求
 * @param confidence 输出预测置信度
 * @return CACHE_SUCCESS 成功，其他值表示错误
 */
cache_result_t prefetch_predict_next(const cache_request_t *current_request,
                                    cache_request_t *predicted_request,
                                    double *confidence);

// 错误处理函数

/**
 * 获取错误字符串
 * @param error 错误码
 * @return 错误描述字符串
 */
const char* cache_get_error_string(cache_result_t error);

// 工具宏

#define CACHE_CHECK(call) do { \
    cache_result_t result = (call); \
    if (result != CACHE_SUCCESS) { \
        fprintf(stderr, "Cache error in %s: %s\n", #call, \
                cache_get_error_string(result)); \
        return result; \
    } \
} while(0)

#define CACHE_CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Null pointer: %s\n", #ptr); \
        return CACHE_ERROR_INVALID_PARAM; \
    } \
} while(0)

// 默认配置值
#define DEFAULT_MAX_ENTRIES 1000
#define DEFAULT_MAX_MEMORY_MB 512
#define DEFAULT_TTL_SECONDS 3600
#define DEFAULT_EVICTION_POLICY CACHE_POLICY_LRU

// 创建默认缓存配置的辅助宏
#define CACHE_CONFIG_DEFAULT() { \
    .max_entries = DEFAULT_MAX_ENTRIES, \
    .max_memory_mb = DEFAULT_MAX_MEMORY_MB, \
    .default_ttl_seconds = DEFAULT_TTL_SECONDS, \
    .eviction_policy = DEFAULT_EVICTION_POLICY, \
    .enable_compression = false, \
    .enable_persistence = false \
}

// 创建缓存请求的辅助宏
#define CACHE_REQUEST_INIT(op_type, params, input, input_sz, ttl) { \
    .operation_type = op_type, \
    .parameters = params, \
    .input_data = input, \
    .input_size = input_sz, \
    .ttl_seconds = ttl \
}

#ifdef __cplusplus
}
#endif

#endif // RESULT_CACHE_H