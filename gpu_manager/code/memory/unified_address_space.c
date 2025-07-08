/**
 * 统一虚拟地址空间管理实现
 * 功能：实现GPU和CPU之间的统一虚拟地址空间
 * 支持：地址转换、内存映射、跨设备访问
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

// 地址空间类型
typedef enum {
    ADDR_SPACE_CPU = 0,      // CPU地址空间
    ADDR_SPACE_GPU,          // GPU地址空间
    ADDR_SPACE_UNIFIED,      // 统一地址空间
    ADDR_SPACE_SHARED        // 共享地址空间
} address_space_type_t;

// 内存类型
typedef enum {
    MEMORY_TYPE_HOST = 0,    // 主机内存
    MEMORY_TYPE_DEVICE,      // 设备内存
    MEMORY_TYPE_MANAGED,     // 托管内存
    MEMORY_TYPE_PINNED       // 锁页内存
} memory_type_t;

// 访问权限
typedef enum {
    ACCESS_NONE = 0,         // 无访问权限
    ACCESS_READ = 1,         // 读权限
    ACCESS_WRITE = 2,        // 写权限
    ACCESS_EXECUTE = 4,      // 执行权限
    ACCESS_RW = ACCESS_READ | ACCESS_WRITE,
    ACCESS_ALL = ACCESS_READ | ACCESS_WRITE | ACCESS_EXECUTE
} access_permission_t;

// 内存区域描述符
typedef struct memory_region {
    uint64_t region_id;         // 区域ID
    void *virtual_addr;         // 虚拟地址
    void *physical_addr;        // 物理地址
    size_t size;                // 区域大小
    address_space_type_t space_type; // 地址空间类型
    memory_type_t memory_type;  // 内存类型
    access_permission_t permissions; // 访问权限
    uint32_t ref_count;         // 引用计数
    bool is_mapped;             // 是否已映射
    bool is_coherent;           // 是否缓存一致
    uint64_t last_access_time;  // 最后访问时间
    struct memory_region *next; // 链表指针
    pthread_mutex_t region_lock; // 区域锁
} memory_region_t;

// 地址转换表项
typedef struct {
    void *virtual_addr;         // 虚拟地址
    void *physical_addr;        // 物理地址
    size_t size;                // 映射大小
    address_space_type_t source_space; // 源地址空间
    address_space_type_t target_space;  // 目标地址空间
    bool is_valid;              // 是否有效
    uint64_t creation_time;     // 创建时间
} address_translation_t;

// 页表项
typedef struct {
    uint64_t virtual_page;      // 虚拟页号
    uint64_t physical_page;     // 物理页号
    access_permission_t permissions; // 访问权限
    bool is_present;            // 是否在内存中
    bool is_dirty;              // 是否脏页
    bool is_accessed;           // 是否被访问
    uint32_t device_id;         // 设备ID
} page_table_entry_t;

// 统一地址空间管理器
typedef struct {
    void *base_address;         // 基地址
    size_t total_size;          // 总大小
    size_t page_size;           // 页面大小
    size_t num_pages;           // 页面数量
    
    memory_region_t *regions;   // 内存区域链表
    uint32_t region_count;      // 区域数量
    
    page_table_entry_t *page_table; // 页表
    address_translation_t *translation_table; // 地址转换表
    size_t translation_table_size; // 转换表大小
    
    void *cpu_heap_start;       // CPU堆起始地址
    void *gpu_heap_start;       // GPU堆起始地址
    size_t cpu_heap_size;       // CPU堆大小
    size_t gpu_heap_size;       // GPU堆大小
    
    pthread_mutex_t manager_lock; // 管理器锁
    pthread_rwlock_t page_table_lock; // 页表读写锁
    
    // 统计信息
    struct {
        uint64_t total_allocations;  // 总分配次数
        uint64_t total_mappings;     // 总映射次数
        uint64_t total_translations; // 总转换次数
        uint64_t cache_hits;         // 缓存命中次数
        uint64_t cache_misses;       // 缓存失误次数
        uint64_t page_faults;        // 页面错误次数
    } stats;
} unified_address_manager_t;

// 全局统一地址管理器
static unified_address_manager_t g_addr_manager = {0};

// 函数声明
static uint64_t get_current_time_us(void);
static uint64_t virtual_to_page_number(void *virtual_addr);
static void* page_number_to_virtual(uint64_t page_number);
static memory_region_t* find_region_by_address(void *addr);
static memory_region_t* allocate_memory_region(size_t size, memory_type_t type);
static void free_memory_region(memory_region_t *region);
static int map_memory_region(memory_region_t *region);
static int unmap_memory_region(memory_region_t *region);
static void* translate_address(void *addr, address_space_type_t from, address_space_type_t to);
static int update_page_table(void *virtual_addr, void *physical_addr, access_permission_t permissions);
static bool check_access_permission(void *addr, access_permission_t required);
static void invalidate_translation_cache(void);
static void update_access_time(memory_region_t *region);

// 初始化统一地址空间管理器
int init_unified_address_space(size_t total_size, size_t page_size) {
    if (total_size == 0 || page_size == 0 || (page_size & (page_size - 1)) != 0) {
        return -1; // 页面大小必须是2的幂
    }
    
    memset(&g_addr_manager, 0, sizeof(unified_address_manager_t));
    
    g_addr_manager.total_size = total_size;
    g_addr_manager.page_size = page_size;
    g_addr_manager.num_pages = total_size / page_size;
    
    // 分配虚拟地址空间
    g_addr_manager.base_address = mmap(NULL, total_size, PROT_NONE, 
                                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g_addr_manager.base_address == MAP_FAILED) {
        printf("Failed to allocate virtual address space: %s\n", strerror(errno));
        return -1;
    }
    
    // 分配页表
    g_addr_manager.page_table = calloc(g_addr_manager.num_pages, sizeof(page_table_entry_t));
    if (!g_addr_manager.page_table) {
        munmap(g_addr_manager.base_address, total_size);
        return -1;
    }
    
    // 分配地址转换表
    g_addr_manager.translation_table_size = 1024; // 初始大小
    g_addr_manager.translation_table = calloc(g_addr_manager.translation_table_size, 
                                             sizeof(address_translation_t));
    if (!g_addr_manager.translation_table) {
        free(g_addr_manager.page_table);
        munmap(g_addr_manager.base_address, total_size);
        return -1;
    }
    
    // 设置CPU和GPU堆区域
    g_addr_manager.cpu_heap_size = total_size / 2;
    g_addr_manager.gpu_heap_size = total_size / 2;
    g_addr_manager.cpu_heap_start = g_addr_manager.base_address;
    g_addr_manager.gpu_heap_start = (char*)g_addr_manager.base_address + g_addr_manager.cpu_heap_size;
    
    // 初始化锁
    pthread_mutex_init(&g_addr_manager.manager_lock, NULL);
    pthread_rwlock_init(&g_addr_manager.page_table_lock, NULL);
    
    printf("Unified address space initialized:\n");
    printf("  Total Size: %zu MB\n", total_size / (1024*1024));
    printf("  Page Size: %zu KB\n", page_size / 1024);
    printf("  Total Pages: %zu\n", g_addr_manager.num_pages);
    printf("  Base Address: %p\n", g_addr_manager.base_address);
    printf("  CPU Heap: %p - %p (%zu MB)\n", 
           g_addr_manager.cpu_heap_start,
           (char*)g_addr_manager.cpu_heap_start + g_addr_manager.cpu_heap_size,
           g_addr_manager.cpu_heap_size / (1024*1024));
    printf("  GPU Heap: %p - %p (%zu MB)\n", 
           g_addr_manager.gpu_heap_start,
           (char*)g_addr_manager.gpu_heap_start + g_addr_manager.gpu_heap_size,
           g_addr_manager.gpu_heap_size / (1024*1024));
    
    return 0;
}

// 分配统一内存
void* allocate_unified_memory(size_t size, memory_type_t type, access_permission_t permissions) {
    if (size == 0) {
        return NULL;
    }
    
    // 对齐到页面大小
    size_t aligned_size = (size + g_addr_manager.page_size - 1) & 
                         ~(g_addr_manager.page_size - 1);
    
    pthread_mutex_lock(&g_addr_manager.manager_lock);
    
    // 分配内存区域
    memory_region_t *region = allocate_memory_region(aligned_size, type);
    if (!region) {
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return NULL;
    }
    
    // 选择地址空间
    void *virtual_addr = NULL;
    if (type == MEMORY_TYPE_HOST || type == MEMORY_TYPE_PINNED) {
        // 从CPU堆分配
        static size_t cpu_heap_offset = 0;
        if (cpu_heap_offset + aligned_size <= g_addr_manager.cpu_heap_size) {
            virtual_addr = (char*)g_addr_manager.cpu_heap_start + cpu_heap_offset;
            cpu_heap_offset += aligned_size;
        }
    } else {
        // 从GPU堆分配
        static size_t gpu_heap_offset = 0;
        if (gpu_heap_offset + aligned_size <= g_addr_manager.gpu_heap_size) {
            virtual_addr = (char*)g_addr_manager.gpu_heap_start + gpu_heap_offset;
            gpu_heap_offset += aligned_size;
        }
    }
    
    if (!virtual_addr) {
        free_memory_region(region);
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return NULL;
    }
    
    // 分配物理内存
    void *physical_addr = NULL;
    switch (type) {
    case MEMORY_TYPE_HOST:
        physical_addr = malloc(aligned_size);
        break;
    case MEMORY_TYPE_DEVICE:
        // 模拟GPU内存分配
        physical_addr = malloc(aligned_size);
        break;
    case MEMORY_TYPE_MANAGED:
        // 托管内存，可以在CPU和GPU之间迁移
        physical_addr = malloc(aligned_size);
        break;
    case MEMORY_TYPE_PINNED:
        // 锁页内存
        physical_addr = mmap(NULL, aligned_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (physical_addr == MAP_FAILED) {
            physical_addr = NULL;
        }
        break;
    }
    
    if (!physical_addr) {
        free_memory_region(region);
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return NULL;
    }
    
    // 设置区域属性
    region->virtual_addr = virtual_addr;
    region->physical_addr = physical_addr;
    region->size = aligned_size;
    region->memory_type = type;
    region->permissions = permissions;
    region->space_type = (type == MEMORY_TYPE_HOST || type == MEMORY_TYPE_PINNED) ? 
                        ADDR_SPACE_CPU : ADDR_SPACE_GPU;
    region->is_coherent = (type == MEMORY_TYPE_MANAGED);
    region->last_access_time = get_current_time_us();
    
    // 映射内存区域
    if (map_memory_region(region) != 0) {
        if (type == MEMORY_TYPE_PINNED) {
            munmap(physical_addr, aligned_size);
        } else {
            free(physical_addr);
        }
        free_memory_region(region);
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return NULL;
    }
    
    // 更新页表
    for (size_t offset = 0; offset < aligned_size; offset += g_addr_manager.page_size) {
        void *page_virtual = (char*)virtual_addr + offset;
        void *page_physical = (char*)physical_addr + offset;
        update_page_table(page_virtual, page_physical, permissions);
    }
    
    // 添加到区域链表
    region->next = g_addr_manager.regions;
    g_addr_manager.regions = region;
    g_addr_manager.region_count++;
    
    g_addr_manager.stats.total_allocations++;
    
    pthread_mutex_unlock(&g_addr_manager.manager_lock);
    
    printf("Allocated unified memory: %p (%zu bytes, type: %d)\n", 
           virtual_addr, aligned_size, type);
    
    return virtual_addr;
}

// 释放统一内存
void free_unified_memory(void *ptr) {
    if (!ptr) {
        return;
    }
    
    pthread_mutex_lock(&g_addr_manager.manager_lock);
    
    // 查找内存区域
    memory_region_t *region = find_region_by_address(ptr);
    if (!region) {
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return;
    }
    
    pthread_mutex_lock(&region->region_lock);
    
    // 减少引用计数
    region->ref_count--;
    if (region->ref_count > 0) {
        pthread_mutex_unlock(&region->region_lock);
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return;
    }
    
    // 取消映射
    unmap_memory_region(region);
    
    // 清理页表
    for (size_t offset = 0; offset < region->size; offset += g_addr_manager.page_size) {
        uint64_t page_num = virtual_to_page_number((char*)region->virtual_addr + offset);
        if (page_num < g_addr_manager.num_pages) {
            pthread_rwlock_wrlock(&g_addr_manager.page_table_lock);
            memset(&g_addr_manager.page_table[page_num], 0, sizeof(page_table_entry_t));
            pthread_rwlock_unlock(&g_addr_manager.page_table_lock);
        }
    }
    
    // 释放物理内存
    if (region->memory_type == MEMORY_TYPE_PINNED) {
        munmap(region->physical_addr, region->size);
    } else {
        free(region->physical_addr);
    }
    
    // 从区域链表中移除
    memory_region_t **current = &g_addr_manager.regions;
    while (*current) {
        if (*current == region) {
            *current = region->next;
            break;
        }
        current = &(*current)->next;
    }
    
    g_addr_manager.region_count--;
    
    pthread_mutex_unlock(&region->region_lock);
    free_memory_region(region);
    
    pthread_mutex_unlock(&g_addr_manager.manager_lock);
    
    printf("Freed unified memory: %p\n", ptr);
}

// 访问统一内存
void* access_unified_memory(void *ptr, access_permission_t required_permissions) {
    if (!ptr) {
        return NULL;
    }
    
    // 检查访问权限
    if (!check_access_permission(ptr, required_permissions)) {
        printf("Access denied: insufficient permissions\n");
        return NULL;
    }
    
    pthread_mutex_lock(&g_addr_manager.manager_lock);
    
    // 查找内存区域
    memory_region_t *region = find_region_by_address(ptr);
    if (!region) {
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return NULL;
    }
    
    pthread_mutex_lock(&region->region_lock);
    
    // 更新访问时间
    update_access_time(region);
    
    // 检查是否需要地址转换
    void *physical_addr = region->physical_addr;
    if (region->space_type != ADDR_SPACE_UNIFIED) {
        // 执行地址转换
        physical_addr = translate_address(ptr, region->space_type, ADDR_SPACE_UNIFIED);
        if (!physical_addr) {
            pthread_mutex_unlock(&region->region_lock);
            pthread_mutex_unlock(&g_addr_manager.manager_lock);
            return NULL;
        }
    }
    
    // 更新页表访问位
    uint64_t page_num = virtual_to_page_number(ptr);
    if (page_num < g_addr_manager.num_pages) {
        pthread_rwlock_wrlock(&g_addr_manager.page_table_lock);
        g_addr_manager.page_table[page_num].is_accessed = true;
        if (required_permissions & ACCESS_WRITE) {
            g_addr_manager.page_table[page_num].is_dirty = true;
        }
        pthread_rwlock_unlock(&g_addr_manager.page_table_lock);
    }
    
    pthread_mutex_unlock(&region->region_lock);
    pthread_mutex_unlock(&g_addr_manager.manager_lock);
    
    return physical_addr;
}

// 同步内存区域
int sync_memory_region(void *ptr, size_t size) {
    if (!ptr || size == 0) {
        return -1;
    }
    
    pthread_mutex_lock(&g_addr_manager.manager_lock);
    
    memory_region_t *region = find_region_by_address(ptr);
    if (!region) {
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return -1;
    }
    
    pthread_mutex_lock(&region->region_lock);
    
    // 对于托管内存，执行同步操作
    if (region->memory_type == MEMORY_TYPE_MANAGED) {
        // 模拟GPU到CPU的数据同步
        if (region->is_coherent) {
            // 缓存一致性内存，无需显式同步
            printf("Memory region is cache coherent, no sync needed\n");
        } else {
            // 执行显式同步
            printf("Syncing memory region: %p (%zu bytes)\n", ptr, size);
            // 这里应该调用实际的GPU同步API
            usleep(100); // 模拟同步延迟
        }
    }
    
    pthread_mutex_unlock(&region->region_lock);
    pthread_mutex_unlock(&g_addr_manager.manager_lock);
    
    return 0;
}

// 设置内存区域权限
int set_memory_permissions(void *ptr, size_t size, access_permission_t permissions) {
    if (!ptr || size == 0) {
        return -1;
    }
    
    pthread_mutex_lock(&g_addr_manager.manager_lock);
    
    memory_region_t *region = find_region_by_address(ptr);
    if (!region) {
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return -1;
    }
    
    pthread_mutex_lock(&region->region_lock);
    
    // 更新区域权限
    region->permissions = permissions;
    
    // 更新页表权限
    for (size_t offset = 0; offset < size; offset += g_addr_manager.page_size) {
        uint64_t page_num = virtual_to_page_number((char*)ptr + offset);
        if (page_num < g_addr_manager.num_pages) {
            pthread_rwlock_wrlock(&g_addr_manager.page_table_lock);
            g_addr_manager.page_table[page_num].permissions = permissions;
            pthread_rwlock_unlock(&g_addr_manager.page_table_lock);
        }
    }
    
    // 更新系统页面权限
    int prot = PROT_NONE;
    if (permissions & ACCESS_READ) prot |= PROT_READ;
    if (permissions & ACCESS_WRITE) prot |= PROT_WRITE;
    if (permissions & ACCESS_EXECUTE) prot |= PROT_EXEC;
    
    if (mprotect(ptr, size, prot) != 0) {
        printf("Failed to update memory protection: %s\n", strerror(errno));
        pthread_mutex_unlock(&region->region_lock);
        pthread_mutex_unlock(&g_addr_manager.manager_lock);
        return -1;
    }
    
    pthread_mutex_unlock(&region->region_lock);
    pthread_mutex_unlock(&g_addr_manager.manager_lock);
    
    printf("Updated memory permissions: %p (%zu bytes, permissions: 0x%x)\n", 
           ptr, size, permissions);
    
    return 0;
}

// 分配内存区域
static memory_region_t* allocate_memory_region(size_t size, memory_type_t type) {
    memory_region_t *region = malloc(sizeof(memory_region_t));
    if (!region) {
        return NULL;
    }
    
    memset(region, 0, sizeof(memory_region_t));
    
    static uint64_t next_region_id = 1;
    region->region_id = next_region_id++;
    region->size = size;
    region->memory_type = type;
    region->ref_count = 1;
    region->is_mapped = false;
    region->is_coherent = false;
    
    pthread_mutex_init(&region->region_lock, NULL);
    
    return region;
}

// 释放内存区域
static void free_memory_region(memory_region_t *region) {
    if (!region) {
        return;
    }
    
    pthread_mutex_destroy(&region->region_lock);
    free(region);
}

// 映射内存区域
static int map_memory_region(memory_region_t *region) {
    if (!region || region->is_mapped) {
        return -1;
    }
    
    // 设置内存保护属性
    int prot = PROT_NONE;
    if (region->permissions & ACCESS_READ) prot |= PROT_READ;
    if (region->permissions & ACCESS_WRITE) prot |= PROT_WRITE;
    if (region->permissions & ACCESS_EXECUTE) prot |= PROT_EXEC;
    
    // 映射虚拟地址到物理地址
    void *mapped = mmap(region->virtual_addr, region->size, prot,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (mapped == MAP_FAILED) {
        printf("Failed to map memory region: %s\n", strerror(errno));
        return -1;
    }
    
    region->is_mapped = true;
    g_addr_manager.stats.total_mappings++;
    
    return 0;
}

// 取消映射内存区域
static int unmap_memory_region(memory_region_t *region) {
    if (!region || !region->is_mapped) {
        return -1;
    }
    
    if (munmap(region->virtual_addr, region->size) != 0) {
        printf("Failed to unmap memory region: %s\n", strerror(errno));
        return -1;
    }
    
    region->is_mapped = false;
    return 0;
}

// 地址转换
static void* translate_address(void *addr, address_space_type_t from, address_space_type_t to) {
    // 查找转换表缓存
    for (size_t i = 0; i < g_addr_manager.translation_table_size; i++) {
        address_translation_t *entry = &g_addr_manager.translation_table[i];
        if (entry->is_valid && 
            entry->virtual_addr <= addr && 
            (char*)addr < (char*)entry->virtual_addr + entry->size &&
            entry->source_space == from && 
            entry->target_space == to) {
            
            // 缓存命中
            g_addr_manager.stats.cache_hits++;
            size_t offset = (char*)addr - (char*)entry->virtual_addr;
            return (char*)entry->physical_addr + offset;
        }
    }
    
    // 缓存失误，执行地址转换
    g_addr_manager.stats.cache_misses++;
    g_addr_manager.stats.total_translations++;
    
    // 简化的地址转换实现
    memory_region_t *region = find_region_by_address(addr);
    if (!region) {
        return NULL;
    }
    
    size_t offset = (char*)addr - (char*)region->virtual_addr;
    void *translated_addr = (char*)region->physical_addr + offset;
    
    // 添加到转换表缓存
    for (size_t i = 0; i < g_addr_manager.translation_table_size; i++) {
        address_translation_t *entry = &g_addr_manager.translation_table[i];
        if (!entry->is_valid) {
            entry->virtual_addr = region->virtual_addr;
            entry->physical_addr = region->physical_addr;
            entry->size = region->size;
            entry->source_space = from;
            entry->target_space = to;
            entry->is_valid = true;
            entry->creation_time = get_current_time_us();
            break;
        }
    }
    
    return translated_addr;
}

// 更新页表
static int update_page_table(void *virtual_addr, void *physical_addr, access_permission_t permissions) {
    uint64_t page_num = virtual_to_page_number(virtual_addr);
    if (page_num >= g_addr_manager.num_pages) {
        return -1;
    }
    
    pthread_rwlock_wrlock(&g_addr_manager.page_table_lock);
    
    page_table_entry_t *entry = &g_addr_manager.page_table[page_num];
    entry->virtual_page = page_num;
    entry->physical_page = (uint64_t)physical_addr / g_addr_manager.page_size;
    entry->permissions = permissions;
    entry->is_present = true;
    entry->is_dirty = false;
    entry->is_accessed = false;
    entry->device_id = 0; // 默认设备
    
    pthread_rwlock_unlock(&g_addr_manager.page_table_lock);
    
    return 0;
}

// 检查访问权限
static bool check_access_permission(void *addr, access_permission_t required) {
    uint64_t page_num = virtual_to_page_number(addr);
    if (page_num >= g_addr_manager.num_pages) {
        return false;
    }
    
    pthread_rwlock_rdlock(&g_addr_manager.page_table_lock);
    
    page_table_entry_t *entry = &g_addr_manager.page_table[page_num];
    bool has_permission = (entry->permissions & required) == required;
    
    pthread_rwlock_unlock(&g_addr_manager.page_table_lock);
    
    return has_permission;
}

// 查找内存区域
static memory_region_t* find_region_by_address(void *addr) {
    memory_region_t *current = g_addr_manager.regions;
    while (current) {
        if ((char*)addr >= (char*)current->virtual_addr && 
            (char*)addr < (char*)current->virtual_addr + current->size) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

// 虚拟地址转页号
static uint64_t virtual_to_page_number(void *virtual_addr) {
    uintptr_t offset = (uintptr_t)virtual_addr - (uintptr_t)g_addr_manager.base_address;
    return offset / g_addr_manager.page_size;
}

// 页号转虚拟地址
static void* page_number_to_virtual(uint64_t page_number) {
    uintptr_t offset = page_number * g_addr_manager.page_size;
    return (char*)g_addr_manager.base_address + offset;
}

// 更新访问时间
static void update_access_time(memory_region_t *region) {
    region->last_access_time = get_current_time_us();
}

// 失效转换缓存
static void invalidate_translation_cache(void) {
    for (size_t i = 0; i < g_addr_manager.translation_table_size; i++) {
        g_addr_manager.translation_table[i].is_valid = false;
    }
}

// 获取当前时间（微秒）
static uint64_t get_current_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

// 打印地址空间统计信息
void print_address_space_stats(void) {
    pthread_mutex_lock(&g_addr_manager.manager_lock);
    
    printf("\n=== Unified Address Space Statistics ===\n");
    printf("Total Allocations: %llu\n", g_addr_manager.stats.total_allocations);
    printf("Total Mappings: %llu\n", g_addr_manager.stats.total_mappings);
    printf("Total Translations: %llu\n", g_addr_manager.stats.total_translations);
    printf("Cache Hits: %llu\n", g_addr_manager.stats.cache_hits);
    printf("Cache Misses: %llu\n", g_addr_manager.stats.cache_misses);
    printf("Page Faults: %llu\n", g_addr_manager.stats.page_faults);
    
    if (g_addr_manager.stats.cache_hits + g_addr_manager.stats.cache_misses > 0) {
        double hit_ratio = (double)g_addr_manager.stats.cache_hits / 
                          (g_addr_manager.stats.cache_hits + g_addr_manager.stats.cache_misses);
        printf("Cache Hit Ratio: %.2f%%\n", hit_ratio * 100);
    }
    
    printf("Active Regions: %u\n", g_addr_manager.region_count);
    printf("Total Pages: %zu\n", g_addr_manager.num_pages);
    printf("Page Size: %zu KB\n", g_addr_manager.page_size / 1024);
    
    // 统计页面使用情况
    uint32_t used_pages = 0;
    uint32_t dirty_pages = 0;
    uint32_t accessed_pages = 0;
    
    pthread_rwlock_rdlock(&g_addr_manager.page_table_lock);
    for (size_t i = 0; i < g_addr_manager.num_pages; i++) {
        page_table_entry_t *entry = &g_addr_manager.page_table[i];
        if (entry->is_present) {
            used_pages++;
            if (entry->is_dirty) dirty_pages++;
            if (entry->is_accessed) accessed_pages++;
        }
    }
    pthread_rwlock_unlock(&g_addr_manager.page_table_lock);
    
    printf("Used Pages: %u / %zu (%.1f%%)\n", 
           used_pages, g_addr_manager.num_pages,
           (double)used_pages / g_addr_manager.num_pages * 100);
    printf("Dirty Pages: %u\n", dirty_pages);
    printf("Accessed Pages: %u\n", accessed_pages);
    
    printf("========================================\n\n");
    
    pthread_mutex_unlock(&g_addr_manager.manager_lock);
}

// 清理统一地址空间管理器
void cleanup_unified_address_space(void) {
    pthread_mutex_lock(&g_addr_manager.manager_lock);
    
    // 清理所有内存区域
    memory_region_t *current = g_addr_manager.regions;
    while (current) {
        memory_region_t *next = current->next;
        
        if (current->is_mapped) {
            unmap_memory_region(current);
        }
        
        if (current->memory_type == MEMORY_TYPE_PINNED) {
            munmap(current->physical_addr, current->size);
        } else {
            free(current->physical_addr);
        }
        
        free_memory_region(current);
        current = next;
    }
    
    // 释放页表
    if (g_addr_manager.page_table) {
        free(g_addr_manager.page_table);
    }
    
    // 释放转换表
    if (g_addr_manager.translation_table) {
        free(g_addr_manager.translation_table);
    }
    
    // 释放虚拟地址空间
    if (g_addr_manager.base_address && g_addr_manager.base_address != MAP_FAILED) {
        munmap(g_addr_manager.base_address, g_addr_manager.total_size);
    }
    
    pthread_mutex_unlock(&g_addr_manager.manager_lock);
    
    // 清理锁
    pthread_mutex_destroy(&g_addr_manager.manager_lock);
    pthread_rwlock_destroy(&g_addr_manager.page_table_lock);
    
    printf("Unified address space cleaned up\n");
}