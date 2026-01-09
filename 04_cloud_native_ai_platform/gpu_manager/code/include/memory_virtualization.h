#ifndef MEMORY_VIRTUALIZATION_H
#define MEMORY_VIRTUALIZATION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Memory Compression API
// ============================================================================

typedef enum {
    COMPRESS_NONE = 0,
    COMPRESS_LZ4,
    COMPRESS_ZSTD,
    COMPRESS_SNAPPY,
    COMPRESS_ADAPTIVE
} compression_algorithm_t;

typedef enum {
    QUALITY_FAST = 1,
    QUALITY_BALANCED = 5,
    QUALITY_BEST = 9
} compression_quality_t;

int init_compression_system(compression_algorithm_t algorithm, compression_quality_t quality, bool parallel_enabled);
int compress_memory(const void *input, size_t input_size, void **output, size_t *output_size, compression_algorithm_t algorithm);
int decompress_memory(const void *input, size_t input_size, void **output, size_t *output_size, compression_algorithm_t algorithm);
void print_compression_stats(void);
void cleanup_compression_system(void);

// ============================================================================
// Memory Swap API
// ============================================================================

typedef enum {
    SWAP_SYSTEM_MEMORY = 0,
    SWAP_SSD_STORAGE,
    SWAP_HDD_STORAGE,
    SWAP_NVME_STORAGE,
    SWAP_REMOTE_STORAGE
} swap_storage_type_t;

int init_swap_system(size_t max_gpu_memory, size_t page_size, double swap_threshold);
void* allocate_gpu_memory(size_t size);
void free_gpu_memory(void *ptr);
void* access_gpu_memory(void *ptr);
void print_swap_stats(void);
void cleanup_swap_system(void);

// ============================================================================
// Unified Address Space API
// ============================================================================

typedef enum {
    MEMORY_TYPE_HOST = 0,
    MEMORY_TYPE_DEVICE,
    MEMORY_TYPE_MANAGED,
    MEMORY_TYPE_PINNED
} memory_type_t;

typedef enum {
    ACCESS_NONE = 0,
    ACCESS_READ = 1,
    ACCESS_WRITE = 2,
    ACCESS_EXECUTE = 4,
    ACCESS_RW = ACCESS_READ | ACCESS_WRITE,
    ACCESS_ALL = ACCESS_READ | ACCESS_WRITE | ACCESS_EXECUTE
} access_permission_t;

int init_unified_address_space(size_t total_size, size_t page_size);
void* allocate_unified_memory(size_t size, memory_type_t type, access_permission_t permissions);
void free_unified_memory(void *ptr);
void* access_unified_memory(void *ptr, access_permission_t required_permissions);
int sync_memory_region(void *ptr, size_t size);
int set_memory_permissions(void *ptr, size_t size, access_permission_t permissions);
void print_address_space_stats(void);
void cleanup_unified_address_space(void);

// ============================================================================
// Memory QoS API
// ============================================================================

typedef enum {
    QOS_LEVEL_CRITICAL = 0,
    QOS_LEVEL_HIGH,
    QOS_LEVEL_NORMAL,
    QOS_LEVEL_LOW,
    QOS_LEVEL_BACKGROUND
} qos_level_t;

typedef enum {
    ACCESS_TYPE_READ = 0,
    ACCESS_TYPE_WRITE,
    ACCESS_TYPE_ATOMIC,
    ACCESS_TYPE_PREFETCH
} memory_access_type_t;

typedef enum {
    BANDWIDTH_POLICY_FAIR = 0,
    BANDWIDTH_POLICY_PRIORITY,
    BANDWIDTH_POLICY_WEIGHTED,
    BANDWIDTH_POLICY_ADAPTIVE
} bandwidth_policy_t;

int init_memory_qos(uint32_t total_bandwidth_mbps, bandwidth_policy_t policy);
int submit_memory_request(void *address, size_t size, memory_access_type_t type, qos_level_t qos_level, uint32_t client_id);
void get_qos_stats(void);
void cleanup_memory_qos(void);

// ============================================================================
// Memory Overcommit API
// ============================================================================

int init_memory_overcommit(size_t physical_memory_size, double overcommit_ratio);
void* allocate_overcommit_memory(size_t size, int priority);
void free_overcommit_memory(void *ptr);
void print_overcommit_stats(void);
void cleanup_memory_overcommit(void);

#ifdef __cplusplus
}
#endif

#endif // MEMORY_VIRTUALIZATION_H
