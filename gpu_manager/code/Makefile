# GPU管理技术代码编译配置
# 来源：GPU 管理相关技术深度解析 - 虚拟化、切分及远程调用

CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -std=c99 -O2 -g -fPIC
CXXFLAGS = -Wall -Wextra -O2 -std=c++11 -fPIC
LDFLAGS = -lpthread -lm
SHARED_LDFLAGS = -shared
MEMORY_LIBS = -lz

# CUDA相关配置（如果可用）
NVCC = nvcc
CUDA_FLAGS = -arch=sm_70

# 目录结构
SRC_DIRS = virtualization partitioning scheduling remote security testing cloud monitoring cache memory optimization testing/integration testing/performance
BUILD_DIR = build
BIN_DIR = bin
LIB_DIR = lib
INC_DIR = include

# 源文件
VIRTUALIZATION_SRCS = virtualization/vgpu_context.c virtualization/cuda_api_intercept.c virtualization/kernel_intercept.c
PARTITIONING_SRCS = partitioning/hybrid_slicing.c partitioning/memory_overcommit.c
SCHEDULING_SRCS = scheduling/gpu_scheduler.c scheduling/priority_scheduler.c scheduling/concurrent_executor.c scheduling/error_handler.c scheduling/qos_manager.c scheduling/multi_queue_scheduler.c
REMOTE_SRCS = remote/remote_gpu_protocol.c remote/remote_client.c remote/remote_call.c remote/connection_monitor.c
SECURITY_SRCS = security/secure_memory.c
TESTING_SRCS = testing/performance_security_test.c
CLOUD_SRCS = cloud/multi_tenant_gpu.c
MONITORING_SRCS = monitoring/performance_monitor.c
INTEGRATION_TEST_SRCS = testing/integration/integration_test.c
PERFORMANCE_TEST_SRCS = testing/performance/performance_test.c
CACHE_SRCS = cache/result_cache.c
MEMORY_SRCS = memory/memory_pool.c memory/memory_overcommit_advanced.c memory/memory_compression.c memory/memory_swap.c memory/unified_address_space.c memory/memory_qos.c memory/unified_memory_manager.c memory/memory_defragmentation.c memory/numa_aware_memory.c memory/memory_hot_migration.c memory/memory_fault_recovery.c
MEMORY_DEMO_SRCS = memory/memory_virtualization_demo.c memory/advanced_memory_demo.c
OPTIMIZATION_SRCS = optimization/bandwidth_optimizer.c optimization/latency_optimizer.c
COMMON_SRCS = common/gpu_common.c
SCRIPTS = partitioning/mig_management.sh

# 目标文件
VIRTUALIZATION_OBJS = $(VIRTUALIZATION_SRCS:%.c=$(BUILD_DIR)/%.o)
PARTITIONING_OBJS = $(PARTITIONING_SRCS:%.c=$(BUILD_DIR)/%.o)
SCHEDULING_OBJS = $(SCHEDULING_SRCS:%.c=$(BUILD_DIR)/%.o)
REMOTE_OBJS = $(REMOTE_SRCS:%.c=$(BUILD_DIR)/%.o)
SECURITY_OBJS = $(SECURITY_SRCS:%.c=$(BUILD_DIR)/%.o)
TESTING_OBJS = $(TESTING_SRCS:%.c=$(BUILD_DIR)/%.o)
CLOUD_OBJS = $(CLOUD_SRCS:%.c=$(BUILD_DIR)/%.o)
MONITORING_OBJS = $(MONITORING_SRCS:%.c=$(BUILD_DIR)/%.o)
INTEGRATION_TEST_OBJS = $(INTEGRATION_TEST_SRCS:%.c=$(BUILD_DIR)/%.o)
PERFORMANCE_TEST_OBJS = $(PERFORMANCE_TEST_SRCS:%.c=$(BUILD_DIR)/%.o)
CACHE_OBJS = $(CACHE_SRCS:%.c=$(BUILD_DIR)/%.o)
MEMORY_OBJS = $(MEMORY_SRCS:%.c=$(BUILD_DIR)/%.o)
MEMORY_DEMO_OBJS = $(MEMORY_DEMO_SRCS:%.c=$(BUILD_DIR)/%.o)
OPTIMIZATION_OBJS = $(OPTIMIZATION_SRCS:%.c=$(BUILD_DIR)/%.o)
COMMON_OBJS = $(COMMON_SRCS:%.c=$(BUILD_DIR)/%.o)

# Memory库文件
MEMORY_SHARED_LIB = $(LIB_DIR)/libmemory_virtualization.so
MEMORY_STATIC_LIB = $(LIB_DIR)/libmemory_virtualization.a
MEMORY_HEADER = $(INC_DIR)/memory_virtualization.h

# 可执行文件
EXECUTABLES = $(BIN_DIR)/hybrid_slicing \
             $(BIN_DIR)/gpu_scheduler \
             $(BIN_DIR)/remote_gpu_protocol \
             $(BIN_DIR)/performance_security_test \
             $(BIN_DIR)/multi_tenant_gpu \
             $(BIN_DIR)/performance_monitor \
             $(BIN_DIR)/integration_test \
             $(BIN_DIR)/performance_test \
             $(BIN_DIR)/memory_virtualization_demo \
             $(BIN_DIR)/advanced_memory_demo \
             $(BIN_DIR)/defrag_demo \
             $(BIN_DIR)/numa_demo \
             $(BIN_DIR)/migration_demo \
             $(BIN_DIR)/fault_demo

# 所有目标文件
ALL_OBJS = $(VIRTUALIZATION_OBJS) $(PARTITIONING_OBJS) $(SCHEDULING_OBJS) $(REMOTE_OBJS) $(SECURITY_OBJS) $(TESTING_OBJS) $(CLOUD_OBJS) $(MONITORING_OBJS) $(INTEGRATION_TEST_OBJS) $(PERFORMANCE_TEST_OBJS) $(CACHE_OBJS) $(MEMORY_OBJS) $(OPTIMIZATION_OBJS) $(COMMON_OBJS)

# 默认目标
all: directories $(MEMORY_SHARED_LIB) $(MEMORY_STATIC_LIB) $(MEMORY_HEADER) $(EXECUTABLES) scripts

# 创建目录
directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR) $(LIB_DIR) $(INC_DIR)
	@for dir in $(SRC_DIRS); do mkdir -p $(BUILD_DIR)/$$dir; done

# 编译规则
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# 可执行文件链接规则
$(BIN_DIR)/hybrid_slicing: $(BUILD_DIR)/partitioning/hybrid_slicing.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/gpu_scheduler: $(BUILD_DIR)/scheduling/gpu_scheduler.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/remote_gpu_protocol: $(BUILD_DIR)/remote/remote_gpu_protocol.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/performance_security_test: $(BUILD_DIR)/testing/performance_security_test.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/multi_tenant_gpu: $(BUILD_DIR)/cloud/multi_tenant_gpu.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/performance_monitor: $(BUILD_DIR)/monitoring/performance_monitor.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/integration_test: $(BUILD_DIR)/testing/integration/integration_test.o $(COMMON_OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/performance_test: $(BUILD_DIR)/testing/performance/performance_test.o $(COMMON_OBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

# Memory库构建规则
$(MEMORY_SHARED_LIB): $(MEMORY_OBJS)
	$(CC) $(SHARED_LDFLAGS) -o $@ $^ $(LDFLAGS) $(MEMORY_LIBS)
	@echo "Memory shared library created: $@"

$(MEMORY_STATIC_LIB): $(MEMORY_OBJS)
	ar rcs $@ $^
	@echo "Memory static library created: $@"

# Memory演示程序
$(BIN_DIR)/memory_virtualization_demo: $(BUILD_DIR)/memory/memory_virtualization_demo.o $(MEMORY_SHARED_LIB)
	$(CC) -o $@ $< -L$(LIB_DIR) -lmemory_virtualization $(MEMORY_LIBS) -Wl,-rpath,$(shell pwd)/$(LIB_DIR)

$(BIN_DIR)/advanced_memory_demo: $(BUILD_DIR)/memory/advanced_memory_demo.o $(MEMORY_SHARED_LIB)
	$(CC) -o $@ $< -L$(LIB_DIR) -lmemory_virtualization $(MEMORY_LIBS) -Wl,-rpath,$(shell pwd)/$(LIB_DIR)

$(BIN_DIR)/defrag_demo: $(MEMORY_SHARED_LIB)
	$(CC) $(CFLAGS) -o $@ -DDEFRAG_DEMO_MAIN memory/memory_defragmentation.c -L$(LIB_DIR) -lmemory_virtualization $(MEMORY_LIBS) -Wl,-rpath,$(shell pwd)/$(LIB_DIR)

$(BIN_DIR)/numa_demo: $(MEMORY_SHARED_LIB)
	$(CC) $(CFLAGS) -o $@ -DNUMA_DEMO_MAIN memory/numa_aware_memory.c -L$(LIB_DIR) -lmemory_virtualization $(MEMORY_LIBS) -Wl,-rpath,$(shell pwd)/$(LIB_DIR)

$(BIN_DIR)/migration_demo: $(MEMORY_SHARED_LIB)
	$(CC) $(CFLAGS) -o $@ -DMIGRATION_DEMO_MAIN memory/memory_hot_migration.c -L$(LIB_DIR) -lmemory_virtualization $(MEMORY_LIBS) -Wl,-rpath,$(shell pwd)/$(LIB_DIR)

$(BIN_DIR)/fault_demo: $(MEMORY_SHARED_LIB)
	$(CC) $(CFLAGS) -o $@ -DFAULT_DEMO_MAIN memory/memory_fault_recovery.c -L$(LIB_DIR) -lmemory_virtualization $(MEMORY_LIBS) -Wl,-rpath,$(shell pwd)/$(LIB_DIR)

# Memory头文件生成
$(MEMORY_HEADER): $(MEMORY_SRCS)
	@echo "Creating unified memory virtualization header file..."
	@echo "#ifndef MEMORY_VIRTUALIZATION_H" > $@
	@echo "#define MEMORY_VIRTUALIZATION_H" >> $@
	@echo "" >> $@
	@echo "#include <stdint.h>" >> $@
	@echo "#include <stddef.h>" >> $@
	@echo "#include <stdbool.h>" >> $@
	@echo "#include <pthread.h>" >> $@
	@echo "" >> $@
	@echo "#ifdef __cplusplus" >> $@
	@echo "extern \"C\" {" >> $@
	@echo "#endif" >> $@
	@echo "" >> $@
	@echo "// ============================================================================" >> $@
	@echo "// Memory Compression API" >> $@
	@echo "// ============================================================================" >> $@
	@echo "" >> $@
	@echo "typedef enum {" >> $@
	@echo "    COMPRESS_NONE = 0," >> $@
	@echo "    COMPRESS_LZ4," >> $@
	@echo "    COMPRESS_ZSTD," >> $@
	@echo "    COMPRESS_SNAPPY," >> $@
	@echo "    COMPRESS_ADAPTIVE" >> $@
	@echo "} compression_algorithm_t;" >> $@
	@echo "" >> $@
	@echo "typedef enum {" >> $@
	@echo "    QUALITY_FAST = 1," >> $@
	@echo "    QUALITY_BALANCED = 5," >> $@
	@echo "    QUALITY_BEST = 9" >> $@
	@echo "} compression_quality_t;" >> $@
	@echo "" >> $@
	@echo "int init_compression_system(compression_algorithm_t algorithm, compression_quality_t quality, bool parallel_enabled);" >> $@
	@echo "int compress_memory(const void *input, size_t input_size, void **output, size_t *output_size, compression_algorithm_t algorithm);" >> $@
	@echo "int decompress_memory(const void *input, size_t input_size, void **output, size_t *output_size, compression_algorithm_t algorithm);" >> $@
	@echo "void print_compression_stats(void);" >> $@
	@echo "void cleanup_compression_system(void);" >> $@
	@echo "" >> $@
	@echo "// ============================================================================" >> $@
	@echo "// Memory Swap API" >> $@
	@echo "// ============================================================================" >> $@
	@echo "" >> $@
	@echo "typedef enum {" >> $@
	@echo "    SWAP_SYSTEM_MEMORY = 0," >> $@
	@echo "    SWAP_SSD_STORAGE," >> $@
	@echo "    SWAP_HDD_STORAGE," >> $@
	@echo "    SWAP_NVME_STORAGE," >> $@
	@echo "    SWAP_REMOTE_STORAGE" >> $@
	@echo "} swap_storage_type_t;" >> $@
	@echo "" >> $@
	@echo "int init_swap_system(size_t max_gpu_memory, size_t page_size, double swap_threshold);" >> $@
	@echo "void* allocate_gpu_memory(size_t size);" >> $@
	@echo "void free_gpu_memory(void *ptr);" >> $@
	@echo "void* access_gpu_memory(void *ptr);" >> $@
	@echo "void print_swap_stats(void);" >> $@
	@echo "void cleanup_swap_system(void);" >> $@
	@echo "" >> $@
	@echo "// ============================================================================" >> $@
	@echo "// Unified Address Space API" >> $@
	@echo "// ============================================================================" >> $@
	@echo "" >> $@
	@echo "typedef enum {" >> $@
	@echo "    MEMORY_TYPE_HOST = 0," >> $@
	@echo "    MEMORY_TYPE_DEVICE," >> $@
	@echo "    MEMORY_TYPE_MANAGED," >> $@
	@echo "    MEMORY_TYPE_PINNED" >> $@
	@echo "} memory_type_t;" >> $@
	@echo "" >> $@
	@echo "typedef enum {" >> $@
	@echo "    ACCESS_NONE = 0," >> $@
	@echo "    ACCESS_READ = 1," >> $@
	@echo "    ACCESS_WRITE = 2," >> $@
	@echo "    ACCESS_EXECUTE = 4," >> $@
	@echo "    ACCESS_RW = ACCESS_READ | ACCESS_WRITE," >> $@
	@echo "    ACCESS_ALL = ACCESS_READ | ACCESS_WRITE | ACCESS_EXECUTE" >> $@
	@echo "} access_permission_t;" >> $@
	@echo "" >> $@
	@echo "int init_unified_address_space(size_t total_size, size_t page_size);" >> $@
	@echo "void* allocate_unified_memory(size_t size, memory_type_t type, access_permission_t permissions);" >> $@
	@echo "void free_unified_memory(void *ptr);" >> $@
	@echo "void* access_unified_memory(void *ptr, access_permission_t required_permissions);" >> $@
	@echo "int sync_memory_region(void *ptr, size_t size);" >> $@
	@echo "int set_memory_permissions(void *ptr, size_t size, access_permission_t permissions);" >> $@
	@echo "void print_address_space_stats(void);" >> $@
	@echo "void cleanup_unified_address_space(void);" >> $@
	@echo "" >> $@
	@echo "// ============================================================================" >> $@
	@echo "// Memory QoS API" >> $@
	@echo "// ============================================================================" >> $@
	@echo "" >> $@
	@echo "typedef enum {" >> $@
	@echo "    QOS_LEVEL_CRITICAL = 0," >> $@
	@echo "    QOS_LEVEL_HIGH," >> $@
	@echo "    QOS_LEVEL_NORMAL," >> $@
	@echo "    QOS_LEVEL_LOW," >> $@
	@echo "    QOS_LEVEL_BACKGROUND" >> $@
	@echo "} qos_level_t;" >> $@
	@echo "" >> $@
	@echo "typedef enum {" >> $@
	@echo "    ACCESS_TYPE_READ = 0," >> $@
	@echo "    ACCESS_TYPE_WRITE," >> $@
	@echo "    ACCESS_TYPE_ATOMIC," >> $@
	@echo "    ACCESS_TYPE_PREFETCH" >> $@
	@echo "} memory_access_type_t;" >> $@
	@echo "" >> $@
	@echo "typedef enum {" >> $@
	@echo "    BANDWIDTH_POLICY_FAIR = 0," >> $@
	@echo "    BANDWIDTH_POLICY_PRIORITY," >> $@
	@echo "    BANDWIDTH_POLICY_WEIGHTED," >> $@
	@echo "    BANDWIDTH_POLICY_ADAPTIVE" >> $@
	@echo "} bandwidth_policy_t;" >> $@
	@echo "" >> $@
	@echo "int init_memory_qos(uint32_t total_bandwidth_mbps, bandwidth_policy_t policy);" >> $@
	@echo "int submit_memory_request(void *address, size_t size, memory_access_type_t type, qos_level_t qos_level, uint32_t client_id);" >> $@
	@echo "void get_qos_stats(void);" >> $@
	@echo "void cleanup_memory_qos(void);" >> $@
	@echo "" >> $@
	@echo "// ============================================================================" >> $@
	@echo "// Memory Overcommit API" >> $@
	@echo "// ============================================================================" >> $@
	@echo "" >> $@
	@echo "int init_memory_overcommit(size_t physical_memory_size, double overcommit_ratio);" >> $@
	@echo "void* allocate_overcommit_memory(size_t size, int priority);" >> $@
	@echo "void free_overcommit_memory(void *ptr);" >> $@
	@echo "void print_overcommit_stats(void);" >> $@
	@echo "void cleanup_memory_overcommit(void);" >> $@
	@echo "" >> $@
	@echo "#ifdef __cplusplus" >> $@
	@echo "}" >> $@
	@echo "#endif" >> $@
	@echo "" >> $@
	@echo "#endif // MEMORY_VIRTUALIZATION_H" >> $@
	@echo "Memory header file created: $@"

# 脚本处理
scripts:
	@chmod +x $(SCRIPTS)
	@echo "Scripts made executable"

# 清理
clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR) $(LIB_DIR) $(INC_DIR)
	@echo "Cleaned build, binary, library and include directories"

# 安装（可选）
install: all
	@echo "Installing GPU management tools..."
	@sudo cp $(BIN_DIR)/* /usr/local/bin/
	@sudo cp $(SCRIPTS) /usr/local/bin/
	@sudo cp $(LIB_DIR)/* /usr/local/lib/ 2>/dev/null || true
	@sudo cp $(INC_DIR)/* /usr/local/include/ 2>/dev/null || true
	@sudo ldconfig 2>/dev/null || true
	@echo "Installation completed"

# 测试
test: all
	@echo "Running GPU management tests..."
	@$(BIN_DIR)/performance_security_test

# Memory模块测试
test-memory: $(BIN_DIR)/memory_virtualization_demo $(BIN_DIR)/advanced_memory_demo
	@echo "Running memory virtualization tests..."
	@$(BIN_DIR)/memory_virtualization_demo
	@$(BIN_DIR)/advanced_memory_demo

# Memory演示程序运行
run-memory-demo: $(BIN_DIR)/memory_virtualization_demo
	@echo "Running memory virtualization demo..."
	@$(BIN_DIR)/memory_virtualization_demo

run-memory-benchmark: $(BIN_DIR)/memory_virtualization_demo
	@echo "Running memory performance benchmark..."
	@$(BIN_DIR)/memory_virtualization_demo --duration 60 --threads 8 --size 2048

run-defrag-demo: $(BIN_DIR)/defrag_demo
	@echo "Running memory defragmentation demo..."
	@$(BIN_DIR)/defrag_demo

run-numa-demo: $(BIN_DIR)/numa_demo
	@echo "Running NUMA-aware memory demo..."
	@$(BIN_DIR)/numa_demo

run-migration-demo: $(BIN_DIR)/migration_demo
	@echo "Running memory hot migration demo..."
	@$(BIN_DIR)/migration_demo

run-fault-demo: $(BIN_DIR)/fault_demo
	@echo "Running memory fault recovery demo..."
	@$(BIN_DIR)/fault_demo

# Memory压力测试
run-memory-stress: $(BIN_DIR)/memory_virtualization_demo
	@echo "Running memory stress test..."
	@$(BIN_DIR)/memory_virtualization_demo --duration 120 --threads 16 --size 4096

# Memory单模块测试
run-compression-only: $(BIN_DIR)/memory_virtualization_demo
	@echo "Testing compression module only..."
	@$(BIN_DIR)/memory_virtualization_demo --no-swap --no-unified --no-qos --duration 30

run-swap-only: $(BIN_DIR)/memory_virtualization_demo
	@echo "Testing swap module only..."
	@$(BIN_DIR)/memory_virtualization_demo --no-compression --no-unified --no-qos --duration 30

run-unified-only: $(BIN_DIR)/memory_virtualization_demo
	@echo "Testing unified address space module only..."
	@$(BIN_DIR)/memory_virtualization_demo --no-compression --no-swap --no-qos --duration 30

run-qos-only: $(BIN_DIR)/memory_virtualization_demo
	@echo "Testing QoS module only..."
	@$(BIN_DIR)/memory_virtualization_demo --no-compression --no-swap --no-unified --duration 30

# 文档生成
docs:
	@echo "Generating documentation..."
	@doxygen Doxyfile 2>/dev/null || echo "Doxygen not available, skipping documentation generation"

# 代码格式化
format:
	@echo "Formatting code..."
	@find . -name "*.c" -o -name "*.h" | xargs clang-format -i 2>/dev/null || echo "clang-format not available"

# 静态分析
analyze:
	@echo "Running static analysis..."
	@cppcheck --enable=all --suppress=missingIncludeSystem . 2>/dev/null || echo "cppcheck not available"

# Memory模块代码检查
check-memory:
	@echo "Running memory module static analysis..."
	@which cppcheck > /dev/null && cppcheck --enable=all --std=c99 memory/*.c || echo "cppcheck not found, skipping static analysis"
	@which clang-tidy > /dev/null && clang-tidy memory/*.c -- $(CFLAGS) || echo "clang-tidy not found, skipping linting"

# Memory模块内存检查
valgrind-check: $(BIN_DIR)/memory_virtualization_demo
	@echo "Running memory leak detection..."
	@which valgrind > /dev/null && valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(BIN_DIR)/memory_virtualization_demo --duration 10 --threads 2 || echo "valgrind not found, skipping memory check"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  all              - Build all executables and libraries"
	@echo "  clean            - Remove build artifacts"
	@echo "  install          - Install binaries and libraries to /usr/local"
	@echo "  test             - Run test programs"
	@echo "  test-memory      - Run memory virtualization tests"
	@echo "  run-memory-demo  - Run memory virtualization demo"
	@echo "  run-memory-benchmark - Run memory performance benchmark"
	@echo "  run-memory-stress - Run memory stress test"
	@echo "  run-*-demo       - Run specific memory demos (defrag, numa, migration, fault)"
	@echo "  run-*-only       - Run individual module tests (compression, swap, unified, qos)"
	@echo "  check-memory     - Run memory module static analysis"
	@echo "  valgrind-check   - Run memory leak detection"
	@echo "  docs             - Generate documentation"
	@echo "  format           - Format source code"
	@echo "  analyze          - Run static analysis"
	@echo "  help             - Show this help message"

# 调试版本
debug: CFLAGS += -DDEBUG -g3
debug: all

# 发布版本
release: CFLAGS += -DNDEBUG -O3
release: all

.PHONY: all clean install test test-memory run-memory-demo run-memory-benchmark run-memory-stress run-defrag-demo run-numa-demo run-migration-demo run-fault-demo run-compression-only run-swap-only run-unified-only run-qos-only check-memory valgrind-check docs format analyze help debug release directories scripts