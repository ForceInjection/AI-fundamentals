/**
 * NUMA感知GPU内存管理实现
 * 功能：优化NUMA架构下的GPU内存访问性能
 * 特性：NUMA拓扑感知、智能内存分配、跨节点迁移
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdbool.h>

#ifdef NUMA_ENABLED
#include <numa.h>
#include <numaif.h>
#endif

#define MAX_NUMA_NODES 8
#define MAX_GPUS_PER_NODE 4
#define NUMA_MIGRATION_THRESHOLD 0.7
#define MEMORY_HOTNESS_THRESHOLD 100

// NUMA节点信息
typedef struct {
    int node_id;                    // 节点ID
    size_t total_memory;            // 总内存
    size_t free_memory;             // 空闲内存
    size_t allocated_memory;        // 已分配内存
    int gpu_count;                  // GPU数量
    int gpu_ids[MAX_GPUS_PER_NODE]; // GPU ID列表
    double memory_bandwidth;        // 内存带宽 GB/s
    double access_latency;          // 访问延迟 ns
    bool is_available;              // 是否可用
} numa_node_t;

// 内存访问统计
typedef struct {
    uint64_t local_accesses;        // 本地访问次数
    uint64_t remote_accesses;       // 远程访问次数
    uint64_t total_bytes_accessed;  // 总访问字节数
    double avg_access_time;         // 平均访问时间
    int hotness_score;              // 热度评分
    time_t last_access_time;        // 最后访问时间
} memory_access_stats_t;

// NUMA感知内存块
typedef struct numa_memory_block {
    void *address;                  // 内存地址
    size_t size;                    // 块大小
    int numa_node;                  // 所在NUMA节点
    int preferred_node;             // 首选节点
    memory_access_stats_t stats;    // 访问统计
    pthread_mutex_t access_lock;    // 访问锁
    struct numa_memory_block *next; // 下一个块
} numa_memory_block_t;

// NUMA拓扑信息
typedef struct {
    int node_count;                 // 节点数量
    numa_node_t nodes[MAX_NUMA_NODES]; // 节点信息
    double distance_matrix[MAX_NUMA_NODES][MAX_NUMA_NODES]; // 节点间距离矩阵
    bool topology_initialized;      // 拓扑是否已初始化
} numa_topology_t;

// NUMA感知内存管理器
typedef struct {
    numa_topology_t topology;       // NUMA拓扑
    numa_memory_block_t *memory_blocks; // 内存块链表
    pthread_rwlock_t manager_lock;  // 管理器锁
    bool migration_enabled;         // 是否启用迁移
    pthread_t migration_thread;     // 迁移线程
    bool running;                   // 是否运行中
} numa_memory_manager_t;

// 全局NUMA内存管理器
static numa_memory_manager_t g_numa_manager = {0};

// 初始化NUMA拓扑
static int init_numa_topology(void) {
#ifdef NUMA_ENABLED
    if (!numa_available()) {
        printf("NUMA not available on this system\n");
        return -1;
    }
    
    g_numa_manager.topology.node_count = numa_max_node() + 1;
#else
    // NUMA未启用时的模拟实现
    printf("NUMA support not compiled in, using single node\n");
    g_numa_manager.topology.node_count = 1;
#endif
    
    for (int i = 0; i < g_numa_manager.topology.node_count; i++) {
        numa_node_t *node = &g_numa_manager.topology.nodes[i];
        node->node_id = i;
        
        // 获取节点内存信息
#ifdef NUMA_ENABLED
        long long free_memory;
        long long total_memory = numa_node_size64(i, &free_memory);
#else
        // 模拟内存信息
        long long total_memory = 8LL * 1024 * 1024 * 1024; // 8GB
        long long free_memory = total_memory / 2;
#endif
        
        if (total_memory > 0) {
            node->total_memory = total_memory;
            node->free_memory = free_memory;
            node->allocated_memory = total_memory - free_memory;
            node->is_available = true;
            
            // 模拟内存带宽和延迟（实际应通过基准测试获得）
            node->memory_bandwidth = 100.0 + i * 10.0; // GB/s
            node->access_latency = 100.0 + i * 20.0;   // ns
            
            printf("NUMA Node %d: %lld MB total, %lld MB free\n", 
                   i, total_memory / (1024*1024), free_memory / (1024*1024));
        } else {
            node->is_available = false;
        }
    }
    
    // 初始化距离矩阵
    for (int i = 0; i < g_numa_manager.topology.node_count; i++) {
        for (int j = 0; j < g_numa_manager.topology.node_count; j++) {
#ifdef NUMA_ENABLED
            g_numa_manager.topology.distance_matrix[i][j] = numa_distance(i, j);
#else
            // 模拟距离矩阵
            g_numa_manager.topology.distance_matrix[i][j] = (i == j) ? 10 : 20;
#endif
        }
    }
    
    g_numa_manager.topology.topology_initialized = true;
    return 0;
}

// 获取当前线程的NUMA节点
static int get_current_numa_node(void) {
#ifdef NUMA_ENABLED
    int cpu = sched_getcpu();
    if (cpu < 0) return -1;
    
    return numa_node_of_cpu(cpu);
#else
    // NUMA未启用时返回节点0
    return 0;
#endif
}

// 选择最佳NUMA节点
static int select_best_numa_node(size_t size, int preferred_node) {
    int best_node = -1;
    double best_score = -1.0;
    
    for (int i = 0; i < g_numa_manager.topology.node_count; i++) {
        numa_node_t *node = &g_numa_manager.topology.nodes[i];
        
        if (!node->is_available || node->free_memory < size) {
            continue;
        }
        
        // 计算节点评分
        double score = 0.0;
        
        // 内存可用性权重
        double memory_ratio = (double)node->free_memory / node->total_memory;
        score += memory_ratio * 0.4;
        
        // 带宽权重
        score += (node->memory_bandwidth / 200.0) * 0.3;
        
        // 延迟权重（越低越好）
        score += (1.0 - node->access_latency / 500.0) * 0.2;
        
        // 首选节点权重
        if (i == preferred_node) {
            score += 0.1;
        } else if (preferred_node >= 0) {
            // 距离权重
            double distance = g_numa_manager.topology.distance_matrix[preferred_node][i];
            score += (1.0 - distance / 255.0) * 0.1;
        }
        
        if (score > best_score) {
            best_score = score;
            best_node = i;
        }
    }
    
    return best_node;
}

// NUMA感知内存分配
void* numa_aware_malloc(size_t size, int preferred_node) {
    if (!g_numa_manager.topology.topology_initialized) {
        return NULL;
    }
    
    pthread_rwlock_wrlock(&g_numa_manager.manager_lock);
    
    // 如果没有指定首选节点，使用当前线程所在节点
    if (preferred_node < 0) {
        preferred_node = get_current_numa_node();
    }
    
    // 选择最佳节点
    int target_node = select_best_numa_node(size, preferred_node);
    if (target_node < 0) {
        pthread_rwlock_unlock(&g_numa_manager.manager_lock);
        return NULL;
    }
    
    // 在指定节点分配内存
#ifdef NUMA_ENABLED
    void *ptr = numa_alloc_onnode(size, target_node);
#else
    void *ptr = malloc(size);
#endif
    if (!ptr) {
        pthread_rwlock_unlock(&g_numa_manager.manager_lock);
        return NULL;
    }
    
    // 创建内存块记录
    numa_memory_block_t *block = malloc(sizeof(numa_memory_block_t));
    if (!block) {
#ifdef NUMA_ENABLED
        numa_free(ptr, size);
#else
        free(ptr);
#endif
        pthread_rwlock_unlock(&g_numa_manager.manager_lock);
        return NULL;
    }
    
    block->address = ptr;
    block->size = size;
    block->numa_node = target_node;
    block->preferred_node = preferred_node;
    memset(&block->stats, 0, sizeof(memory_access_stats_t));
    pthread_mutex_init(&block->access_lock, NULL);
    
    // 添加到链表
    block->next = g_numa_manager.memory_blocks;
    g_numa_manager.memory_blocks = block;
    
    // 更新节点统计
    g_numa_manager.topology.nodes[target_node].free_memory -= size;
    g_numa_manager.topology.nodes[target_node].allocated_memory += size;
    
    pthread_rwlock_unlock(&g_numa_manager.manager_lock);
    
    printf("Allocated %zu bytes on NUMA node %d (preferred: %d)\n", 
           size, target_node, preferred_node);
    
    return ptr;
}

// NUMA感知内存释放
void numa_aware_free(void *ptr) {
    if (!ptr) return;
    
    pthread_rwlock_wrlock(&g_numa_manager.manager_lock);
    
    numa_memory_block_t *prev = NULL;
    numa_memory_block_t *block = g_numa_manager.memory_blocks;
    
    while (block) {
        if (block->address == ptr) {
            // 从链表中移除
            if (prev) {
                prev->next = block->next;
            } else {
                g_numa_manager.memory_blocks = block->next;
            }
            
            // 更新节点统计
            int node = block->numa_node;
            g_numa_manager.topology.nodes[node].free_memory += block->size;
            g_numa_manager.topology.nodes[node].allocated_memory -= block->size;
            
            // 释放内存
#ifdef NUMA_ENABLED
            numa_free(ptr, block->size);
#else
            free(ptr);
#endif
            pthread_mutex_destroy(&block->access_lock);
            free(block);
            
            break;
        }
        prev = block;
        block = block->next;
    }
    
    pthread_rwlock_unlock(&g_numa_manager.manager_lock);
}

// 记录内存访问
static void record_memory_access(numa_memory_block_t *block, bool is_local) {
    pthread_mutex_lock(&block->access_lock);
    
    if (is_local) {
        block->stats.local_accesses++;
    } else {
        block->stats.remote_accesses++;
    }
    
    block->stats.last_access_time = time(NULL);
    
    // 更新热度评分
    uint64_t total_accesses = block->stats.local_accesses + block->stats.remote_accesses;
    if (total_accesses > 0) {
        double local_ratio = (double)block->stats.local_accesses / total_accesses;
        block->stats.hotness_score = (int)(local_ratio * 100 + 
                                          (total_accesses / 1000.0) * 50);
    }
    
    pthread_mutex_unlock(&block->access_lock);
}

// 内存访问包装函数
void* numa_memory_access(void *ptr, size_t offset) {
    if (!ptr) return NULL;
    
    pthread_rwlock_rdlock(&g_numa_manager.manager_lock);
    
    numa_memory_block_t *block = g_numa_manager.memory_blocks;
    while (block) {
        if (block->address <= ptr && 
            (char*)ptr < (char*)block->address + block->size) {
            
            int current_node = get_current_numa_node();
            bool is_local = (current_node == block->numa_node);
            
            record_memory_access(block, is_local);
            break;
        }
        block = block->next;
    }
    
    pthread_rwlock_unlock(&g_numa_manager.manager_lock);
    
    return (char*)ptr + offset;
}

// 内存迁移
static int migrate_memory_block(numa_memory_block_t *block, int target_node) {
    if (block->numa_node == target_node) {
        return 0;  // 已在目标节点
    }
    
    printf("Migrating memory block from node %d to node %d\n", 
           block->numa_node, target_node);
    
    // 在目标节点分配新内存
#ifdef NUMA_ENABLED
    void *new_ptr = numa_alloc_onnode(block->size, target_node);
#else
    void *new_ptr = malloc(block->size);
#endif
    if (!new_ptr) {
        return -1;
    }
    
    // 复制数据
    memcpy(new_ptr, block->address, block->size);
    
    // 更新节点统计
    g_numa_manager.topology.nodes[block->numa_node].free_memory += block->size;
    g_numa_manager.topology.nodes[block->numa_node].allocated_memory -= block->size;
    g_numa_manager.topology.nodes[target_node].free_memory -= block->size;
    g_numa_manager.topology.nodes[target_node].allocated_memory += block->size;
    
    // 释放旧内存
#ifdef NUMA_ENABLED
    numa_free(block->address, block->size);
#else
    free(block->address);
#endif
    
    // 更新块信息
    block->address = new_ptr;
    block->numa_node = target_node;
    
    return 0;
}

// 内存迁移线程
static void* migration_thread_func(void *arg) {
    while (g_numa_manager.running) {
        pthread_rwlock_rdlock(&g_numa_manager.manager_lock);
        
        numa_memory_block_t *block = g_numa_manager.memory_blocks;
        while (block) {
            pthread_mutex_lock(&block->access_lock);
            
            uint64_t total_accesses = block->stats.local_accesses + 
                                    block->stats.remote_accesses;
            
            if (total_accesses > MEMORY_HOTNESS_THRESHOLD) {
                double remote_ratio = (double)block->stats.remote_accesses / total_accesses;
                
                // 如果远程访问比例过高，考虑迁移
                if (remote_ratio > NUMA_MIGRATION_THRESHOLD) {
                    int current_node = get_current_numa_node();
                    if (current_node >= 0 && current_node != block->numa_node) {
                        pthread_mutex_unlock(&block->access_lock);
                        pthread_rwlock_unlock(&g_numa_manager.manager_lock);
                        
                        pthread_rwlock_wrlock(&g_numa_manager.manager_lock);
                        migrate_memory_block(block, current_node);
                        pthread_rwlock_unlock(&g_numa_manager.manager_lock);
                        
                        pthread_rwlock_rdlock(&g_numa_manager.manager_lock);
                        pthread_mutex_lock(&block->access_lock);
                        
                        // 重置统计
                        block->stats.local_accesses = 0;
                        block->stats.remote_accesses = 0;
                    }
                }
            }
            
            pthread_mutex_unlock(&block->access_lock);
            block = block->next;
        }
        
        pthread_rwlock_unlock(&g_numa_manager.manager_lock);
        
        sleep(5);  // 每5秒检查一次
    }
    
    return NULL;
}

// 初始化NUMA感知内存管理器
int init_numa_memory_manager(void) {
    memset(&g_numa_manager, 0, sizeof(numa_memory_manager_t));
    
    if (init_numa_topology() != 0) {
        return -1;
    }
    
    if (pthread_rwlock_init(&g_numa_manager.manager_lock, NULL) != 0) {
        return -1;
    }
    
    g_numa_manager.migration_enabled = true;
    g_numa_manager.running = true;
    
    // 启动迁移线程
    if (pthread_create(&g_numa_manager.migration_thread, NULL, 
                      migration_thread_func, NULL) != 0) {
        pthread_rwlock_destroy(&g_numa_manager.manager_lock);
        return -1;
    }
    
    printf("NUMA-aware memory manager initialized with %d nodes\n", 
           g_numa_manager.topology.node_count);
    
    return 0;
}

// 获取NUMA统计信息
void print_numa_stats(void) {
    pthread_rwlock_rdlock(&g_numa_manager.manager_lock);
    
    printf("\n=== NUMA Memory Statistics ===\n");
    
    for (int i = 0; i < g_numa_manager.topology.node_count; i++) {
        numa_node_t *node = &g_numa_manager.topology.nodes[i];
        if (node->is_available) {
            printf("Node %d:\n", i);
            printf("  Total Memory: %zu MB\n", node->total_memory / (1024*1024));
            printf("  Free Memory: %zu MB\n", node->free_memory / (1024*1024));
            printf("  Allocated Memory: %zu MB\n", node->allocated_memory / (1024*1024));
            printf("  Memory Bandwidth: %.1f GB/s\n", node->memory_bandwidth);
            printf("  Access Latency: %.1f ns\n", node->access_latency);
        }
    }
    
    printf("\nMemory Blocks:\n");
    numa_memory_block_t *block = g_numa_manager.memory_blocks;
    int block_count = 0;
    while (block) {
        printf("  Block %d: %zu bytes on node %d (hotness: %d)\n", 
               ++block_count, block->size, block->numa_node, 
               block->stats.hotness_score);
        block = block->next;
    }
    
    printf("==============================\n\n");
    
    pthread_rwlock_unlock(&g_numa_manager.manager_lock);
}

// 清理NUMA感知内存管理器
void cleanup_numa_memory_manager(void) {
    g_numa_manager.running = false;
    
    // 等待迁移线程结束
    pthread_join(g_numa_manager.migration_thread, NULL);
    
    pthread_rwlock_wrlock(&g_numa_manager.manager_lock);
    
    // 释放所有内存块
    numa_memory_block_t *block = g_numa_manager.memory_blocks;
    while (block) {
        numa_memory_block_t *next = block->next;
#ifdef NUMA_ENABLED
        numa_free(block->address, block->size);
#else
        free(block->address);
#endif
        pthread_mutex_destroy(&block->access_lock);
        free(block);
        block = next;
    }
    
    pthread_rwlock_unlock(&g_numa_manager.manager_lock);
    pthread_rwlock_destroy(&g_numa_manager.manager_lock);
    
    printf("NUMA-aware memory manager cleaned up\n");
}