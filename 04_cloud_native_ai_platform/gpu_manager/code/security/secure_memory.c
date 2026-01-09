/**
 * GPU安全内存管理实现
 * 来源：GPU 管理相关技术深度解析 - 3.2.3 多租户环境下的安全性
 * 功能：提供安全的GPU内存分配和访问控制
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

// 租户访问控制列表结构
typedef struct {
    uint32_t tenant_id;
    uint32_t device_mask;     // 允许访问的设备掩码
    size_t memory_limit;      // 内存使用限制
    uint32_t compute_limit;   // 计算资源限制
} tenant_acl_t;

// 时间常量定义
#define MIN_EXECUTION_TIME 1000  // 最小执行时间(微秒)
#define MAX_RANDOM_DELAY 500     // 最大随机延迟(微秒)

// GPU内存分配时强制清零
cudaError_t secure_cuda_malloc(void **devPtr, size_t size) {
    cudaError_t result = cudaMalloc(devPtr, size);
    if (result == cudaSuccess) {
        cudaMemset(*devPtr, 0, size);  // 强制清零
    }
    return result;
}

// 检查访问权限
bool check_access_permission(uint32_t tenant_id, uint32_t device_id) {
    // 应用时间混淆防止侧信道攻击
    timing_obfuscation();
    
    tenant_acl_t *acl = get_tenant_acl(tenant_id);
    if (!acl) {
        return false;
    }
    
    // 检查设备访问权限
    bool has_device_access = (acl->device_mask & (1 << device_id)) != 0;
    
    // 记录访问尝试（用于审计）
    if (!has_device_access) {
        // 在实际实现中，这里应该记录到安全日志
        printf("Access denied: tenant %u attempted to access device %u\n", 
               tenant_id, device_id);
    }
    
    return has_device_access;
}

// 时间混淆机制防止侧信道攻击
void timing_obfuscation(void) {
    static uint64_t last_time = 0;
    static uint32_t call_count = 0;
    
    uint64_t current_time = get_timestamp();
    uint64_t elapsed = current_time - last_time;
    
    call_count++;
    
    // 增强的随机延迟策略
    uint32_t base_delay = MIN_EXECUTION_TIME;
    uint32_t random_component = rand() % (MAX_RANDOM_DELAY * 4); // 增加随机范围
    
    // 基于调用频率的自适应延迟
    if (elapsed < MIN_EXECUTION_TIME) {
        uint32_t adaptive_delay = base_delay + random_component;
        
        // 每隔一定次数添加额外的混淆延迟
        if (call_count % 7 == 0) {
            adaptive_delay += rand() % 1000; // 额外0-1ms延迟
        }
        
        usleep(adaptive_delay);
    } else {
        // 即使时间间隔足够，也偶尔添加随机延迟
        if (call_count % 13 == 0) {
            usleep(rand() % (MAX_RANDOM_DELAY / 2));
        }
    }
    
    last_time = get_timestamp();
}

// 全局租户ACL表
static tenant_acl_t tenant_acl_table[256]; // 支持最多256个租户
static int acl_table_size = 0;
static bool acl_initialized = false;

// 初始化默认ACL表
static void init_default_acl_table(void) {
    if (acl_initialized) return;
    
    // 添加一些默认租户配置
    tenant_acl_table[0] = (tenant_acl_t){
        .tenant_id = 1001,
        .device_mask = 0x0F,  // 允许访问设备0-3
        .memory_limit = 2ULL * 1024 * 1024 * 1024,  // 2GB
        .compute_limit = 50   // 50%计算资源
    };
    
    tenant_acl_table[1] = (tenant_acl_t){
        .tenant_id = 1002,
        .device_mask = 0x03,  // 允许访问设备0-1
        .memory_limit = 1ULL * 1024 * 1024 * 1024,  // 1GB
        .compute_limit = 25   // 25%计算资源
    };
    
    acl_table_size = 2;
    acl_initialized = true;
}

// 获取租户ACL
tenant_acl_t* get_tenant_acl(uint32_t tenant_id) {
    init_default_acl_table();
    
    // 在ACL表中查找租户
    for (int i = 0; i < acl_table_size; i++) {
        if (tenant_acl_table[i].tenant_id == tenant_id) {
            return &tenant_acl_table[i];
        }
    }
    
    // 如果找不到，返回默认的受限权限
    static tenant_acl_t default_acl = {
        .tenant_id = 0,
        .device_mask = 0x01,  // 只允许访问设备0
        .memory_limit = 512 * 1024 * 1024,  // 512MB
        .compute_limit = 10   // 10%计算资源
    };
    
    return &default_acl;
}

// 获取时间戳（需要实现）
uint64_t get_timestamp(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000ULL;
}