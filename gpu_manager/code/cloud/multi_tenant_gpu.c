/**
 * 云计算多租户GPU资源管理
 * 来源：GPU 管理相关技术深度解析 - 7.1.1 多租户GPU资源管理
 * 功能：实现云环境下的多租户GPU资源分配和管理
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stdbool.h>

// 云计算多租户管理框架
typedef struct {
    char tenant_id[64];          // 租户ID
    int allocated_gpu_memory_mb; // 分配的GPU内存(MB)
    int max_compute_units;       // 最大计算单元数
    double cpu_quota;            // CPU配额
    int priority_level;          // 优先级(1-10)
    uint64_t usage_start_time;   // 使用开始时间
    uint64_t billing_duration;   // 计费时长
    char service_level[32];      // 服务等级
    bool is_active;              // 是否活跃
    double current_usage_percent; // 当前使用率
} cloud_tenant_t;

// 云租户管理器
typedef struct {
    cloud_tenant_t tenants[256]; // 最多256个租户
    int active_tenant_count;     // 活跃租户数
    double total_gpu_memory_gb;  // 总GPU内存
    int total_compute_units;     // 总计算单元数
    char resource_allocation_policy[64]; // 资源分配策略
    double memory_utilization;   // 内存利用率
    double compute_utilization;  // 计算利用率
    uint64_t total_billing_time; // 总计费时间
} cloud_gpu_manager_t;

// 服务等级定义
enum service_level {
    SLA_BASIC = 1,
    SLA_STANDARD = 2,
    SLA_PREMIUM = 3,
    SLA_ENTERPRISE = 4
};

// 资源分配策略
enum allocation_policy {
    POLICY_FAIR_SHARE,
    POLICY_PRIORITY_BASED,
    POLICY_DEMAND_BASED,
    POLICY_SLA_BASED
};

// 全局云GPU管理器
static cloud_gpu_manager_t g_cloud_manager = {
    .active_tenant_count = 0,
    .total_gpu_memory_gb = 32.0,  // 32GB总内存
    .total_compute_units = 128,   // 128个计算单元
    .memory_utilization = 0.0,
    .compute_utilization = 0.0,
    .total_billing_time = 0
};

// 函数声明
static uint64_t get_current_timestamp(void);
static int find_tenant_by_id(const char *tenant_id);
static double calculate_memory_utilization(void);
static double calculate_compute_utilization(void);
static int get_sla_priority(const char *service_level);
static double get_sla_multiplier(const char *service_level);
static void update_billing_info(cloud_tenant_t *tenant);

// 创建新租户
int create_cloud_tenant(const char *tenant_id, const char *service_level, 
                       int memory_mb, int compute_units) {
    if (g_cloud_manager.active_tenant_count >= 256) {
        printf("Error: Maximum tenant limit reached\n");
        return -1;
    }
    
    // 检查资源可用性
    double required_memory_gb = memory_mb / 1024.0;
    if (required_memory_gb > g_cloud_manager.total_gpu_memory_gb * 0.8) {
        printf("Error: Insufficient GPU memory for tenant %s\n", tenant_id);
        return -2;
    }
    
    if (compute_units > g_cloud_manager.total_compute_units * 0.8) {
        printf("Error: Insufficient compute units for tenant %s\n", tenant_id);
        return -3;
    }
    
    // 创建租户
    int index = g_cloud_manager.active_tenant_count;
    cloud_tenant_t *tenant = &g_cloud_manager.tenants[index];
    
    strncpy(tenant->tenant_id, tenant_id, sizeof(tenant->tenant_id) - 1);
    tenant->allocated_gpu_memory_mb = memory_mb;
    tenant->max_compute_units = compute_units;
    tenant->priority_level = get_sla_priority(service_level);
    tenant->usage_start_time = get_current_timestamp();
    tenant->billing_duration = 0;
    strncpy(tenant->service_level, service_level, sizeof(tenant->service_level) - 1);
    tenant->is_active = true;
    tenant->current_usage_percent = 0.0;
    
    // 根据服务等级设置CPU配额
    tenant->cpu_quota = get_sla_multiplier(service_level) * 2.0; // 基础2核
    
    g_cloud_manager.active_tenant_count++;
    
    printf("Created tenant %s with %dMB GPU memory, %d compute units, SLA: %s\n",
           tenant_id, memory_mb, compute_units, service_level);
    
    return 0;
}

// 删除租户
int remove_cloud_tenant(const char *tenant_id) {
    int index = find_tenant_by_id(tenant_id);
    if (index < 0) {
        printf("Error: Tenant %s not found\n", tenant_id);
        return -1;
    }
    
    cloud_tenant_t *tenant = &g_cloud_manager.tenants[index];
    
    // 更新计费信息
    update_billing_info(tenant);
    
    printf("Removing tenant %s (total billing time: %llu seconds)\n",
           tenant_id, (unsigned long long)tenant->billing_duration);
    
    // 移动后续租户
    for (int i = index; i < g_cloud_manager.active_tenant_count - 1; i++) {
        g_cloud_manager.tenants[i] = g_cloud_manager.tenants[i + 1];
    }
    
    g_cloud_manager.active_tenant_count--;
    
    return 0;
}

// 分配GPU资源给租户
int allocate_gpu_resources(const char *tenant_id, int additional_memory_mb, int additional_compute_units) {
    int index = find_tenant_by_id(tenant_id);
    if (index < 0) {
        printf("Error: Tenant %s not found\n", tenant_id);
        return -1;
    }
    
    cloud_tenant_t *tenant = &g_cloud_manager.tenants[index];
    
    // 检查资源限制
    int new_memory = tenant->allocated_gpu_memory_mb + additional_memory_mb;
    int new_compute = tenant->max_compute_units + additional_compute_units;
    
    double total_allocated_memory = 0;
    int total_allocated_compute = 0;
    
    // 计算当前总分配量
    for (int i = 0; i < g_cloud_manager.active_tenant_count; i++) {
        if (i != index) {
            total_allocated_memory += g_cloud_manager.tenants[i].allocated_gpu_memory_mb;
            total_allocated_compute += g_cloud_manager.tenants[i].max_compute_units;
        }
    }
    
    total_allocated_memory += new_memory;
    total_allocated_compute += new_compute;
    
    // 检查是否超过总容量
    if (total_allocated_memory > g_cloud_manager.total_gpu_memory_gb * 1024 * 0.9) {
        printf("Error: Memory allocation would exceed capacity\n");
        return -2;
    }
    
    if (total_allocated_compute > g_cloud_manager.total_compute_units * 0.9) {
        printf("Error: Compute allocation would exceed capacity\n");
        return -3;
    }
    
    // 更新分配
    tenant->allocated_gpu_memory_mb = new_memory;
    tenant->max_compute_units = new_compute;
    
    printf("Allocated additional resources to %s: +%dMB memory, +%d compute units\n",
           tenant_id, additional_memory_mb, additional_compute_units);
    
    return 0;
}

// 更新租户使用情况
int update_tenant_usage(const char *tenant_id, double usage_percent) {
    int index = find_tenant_by_id(tenant_id);
    if (index < 0) {
        return -1;
    }
    
    cloud_tenant_t *tenant = &g_cloud_manager.tenants[index];
    tenant->current_usage_percent = usage_percent;
    
    // 更新计费时间
    update_billing_info(tenant);
    
    return 0;
}

// 获取云GPU管理统计信息
void get_cloud_gpu_statistics() {
    // 更新利用率
    g_cloud_manager.memory_utilization = calculate_memory_utilization();
    g_cloud_manager.compute_utilization = calculate_compute_utilization();
    
    printf("\n=== Cloud GPU Management Statistics ===\n");
    printf("Active Tenants: %d\n", g_cloud_manager.active_tenant_count);
    printf("Total GPU Memory: %.1f GB\n", g_cloud_manager.total_gpu_memory_gb);
    printf("Total Compute Units: %d\n", g_cloud_manager.total_compute_units);
    printf("Memory Utilization: %.1f%%\n", g_cloud_manager.memory_utilization * 100);
    printf("Compute Utilization: %.1f%%\n", g_cloud_manager.compute_utilization * 100);
    
    printf("\n=== Tenant Details ===\n");
    printf("%-20s %-10s %-12s %-10s %-8s %-10s\n", 
           "Tenant ID", "SLA", "Memory(MB)", "Compute", "Usage", "Priority");
    printf("--------------------------------------------------------------------\n");
    
    for (int i = 0; i < g_cloud_manager.active_tenant_count; i++) {
        cloud_tenant_t *tenant = &g_cloud_manager.tenants[i];
        printf("%-20s %-10s %-12d %-10d %-7.1f%% %-10d\n",
               tenant->tenant_id,
               tenant->service_level,
               tenant->allocated_gpu_memory_mb,
               tenant->max_compute_units,
               tenant->current_usage_percent,
               tenant->priority_level);
    }
}

// 执行资源重新平衡
int rebalance_resources() {
    printf("\nExecuting resource rebalancing...\n");
    
    // 按优先级排序租户
    for (int i = 0; i < g_cloud_manager.active_tenant_count - 1; i++) {
        for (int j = i + 1; j < g_cloud_manager.active_tenant_count; j++) {
            if (g_cloud_manager.tenants[i].priority_level < g_cloud_manager.tenants[j].priority_level) {
                cloud_tenant_t temp = g_cloud_manager.tenants[i];
                g_cloud_manager.tenants[i] = g_cloud_manager.tenants[j];
                g_cloud_manager.tenants[j] = temp;
            }
        }
    }
    
    // 重新分配资源
    double available_memory = g_cloud_manager.total_gpu_memory_gb * 1024;
    int available_compute = g_cloud_manager.total_compute_units;
    
    for (int i = 0; i < g_cloud_manager.active_tenant_count; i++) {
        cloud_tenant_t *tenant = &g_cloud_manager.tenants[i];
        
        // 根据优先级和SLA调整分配
        double sla_multiplier = get_sla_multiplier(tenant->service_level);
        int suggested_memory = (int)(available_memory * 0.1 * sla_multiplier);
        int suggested_compute = (int)(available_compute * 0.1 * sla_multiplier);
        (void)suggested_compute; // 避免未使用变量警告
        
        if (suggested_memory < tenant->allocated_gpu_memory_mb) {
            printf("Reducing memory for %s: %d -> %d MB\n",
                   tenant->tenant_id, tenant->allocated_gpu_memory_mb, suggested_memory);
            tenant->allocated_gpu_memory_mb = suggested_memory;
        }
        
        available_memory -= tenant->allocated_gpu_memory_mb;
        available_compute -= tenant->max_compute_units;
    }
    
    printf("Resource rebalancing completed\n");
    return 0;
}

// 生成计费报告
void generate_billing_report() {
    printf("\n=== Billing Report ===\n");
    printf("%-20s %-10s %-15s %-12s %-10s\n", 
           "Tenant ID", "SLA", "Usage Time(s)", "Memory(MB)", "Cost($)");
    printf("---------------------------------------------------------------\n");
    
    double total_revenue = 0.0;
    
    for (int i = 0; i < g_cloud_manager.active_tenant_count; i++) {
        cloud_tenant_t *tenant = &g_cloud_manager.tenants[i];
        
        // 更新计费信息
        update_billing_info(tenant);
        
        // 计算费用 (简化计费模型)
        double hourly_rate = get_sla_multiplier(tenant->service_level) * 0.5; // $0.5/hour base rate
        double memory_rate = tenant->allocated_gpu_memory_mb * 0.001; // $0.001/MB/hour
        double hours = tenant->billing_duration / 3600.0;
        double cost = (hourly_rate + memory_rate) * hours;
        
        total_revenue += cost;
        
        printf("%-20s %-10s %-15llu %-12d $%-9.2f\n",
               tenant->tenant_id,
               tenant->service_level,
               (unsigned long long)tenant->billing_duration,
               tenant->allocated_gpu_memory_mb,
               cost);
    }
    
    printf("---------------------------------------------------------------\n");
    printf("Total Revenue: $%.2f\n", total_revenue);
}

// 辅助函数实现
static uint64_t get_current_timestamp() {
    return (uint64_t)time(NULL);
}

static int find_tenant_by_id(const char *tenant_id) {
    for (int i = 0; i < g_cloud_manager.active_tenant_count; i++) {
        if (strcmp(g_cloud_manager.tenants[i].tenant_id, tenant_id) == 0) {
            return i;
        }
    }
    return -1;
}

static double calculate_memory_utilization() {
    double total_allocated = 0;
    for (int i = 0; i < g_cloud_manager.active_tenant_count; i++) {
        total_allocated += g_cloud_manager.tenants[i].allocated_gpu_memory_mb;
    }
    return total_allocated / (g_cloud_manager.total_gpu_memory_gb * 1024);
}

static double calculate_compute_utilization() {
    int total_allocated = 0;
    for (int i = 0; i < g_cloud_manager.active_tenant_count; i++) {
        total_allocated += g_cloud_manager.tenants[i].max_compute_units;
    }
    return (double)total_allocated / g_cloud_manager.total_compute_units;
}

static int get_sla_priority(const char *service_level) {
    if (strcmp(service_level, "ENTERPRISE") == 0) return 10;
    if (strcmp(service_level, "PREMIUM") == 0) return 8;
    if (strcmp(service_level, "STANDARD") == 0) return 5;
    if (strcmp(service_level, "BASIC") == 0) return 3;
    return 1;
}

static double get_sla_multiplier(const char *service_level) {
    if (strcmp(service_level, "ENTERPRISE") == 0) return 4.0;
    if (strcmp(service_level, "PREMIUM") == 0) return 2.5;
    if (strcmp(service_level, "STANDARD") == 0) return 1.5;
    if (strcmp(service_level, "BASIC") == 0) return 1.0;
    return 0.5;
}

static void update_billing_info(cloud_tenant_t *tenant) {
    uint64_t current_time = get_current_timestamp();
    if (tenant->usage_start_time > 0) {
        tenant->billing_duration = current_time - tenant->usage_start_time;
    }
}

int main() {
    printf("Multi-Tenant GPU Management Module\n");
    return 0;
}