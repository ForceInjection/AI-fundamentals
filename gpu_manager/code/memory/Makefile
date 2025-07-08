# GPU内存虚拟化技术增强模块 Makefile
# 编译所有内存虚拟化相关的组件

CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -O2 -std=c99 -fPIC
CXXFLAGS = -Wall -Wextra -O2 -std=c++11 -fPIC
LDFLAGS = -shared
LIBS = -lpthread -lm -lz -llz4 -lzstd

# 目录定义
SRCDIR = .
OBJDIR = obj
BINDIR = bin
LIBDIR = lib
INCDIR = include

# 源文件
SOURCES = memory_overcommit_advanced.c \
          memory_compression.c \
          memory_swap.c \
          unified_address_space.c \
          memory_qos.c \
          unified_memory_manager.c \
          memory_defragmentation.c \
          numa_aware_memory.c \
          memory_hot_migration.c \
          memory_fault_recovery.c

DEMO_SOURCE = memory_virtualization_demo.c
ADVANCED_DEMO_SOURCE = advanced_memory_demo.c

# 目标文件
OBJECTS = $(SOURCES:%.c=$(OBJDIR)/%.o)
DEMO_OBJ = $(OBJDIR)/$(DEMO_SOURCE:%.c=%.o)
ADVANCED_DEMO_OBJ = $(OBJDIR)/$(ADVANCED_DEMO_SOURCE:%.c=%.o)

# 目标库和可执行文件
LIBTARGET = $(LIBDIR)/libmemory_virtualization.so
DEMO_TARGET = $(BINDIR)/memory_virtualization_demo
STATIC_LIB = $(LIBDIR)/libmemory_virtualization.a
DEFRAG_DEMO = $(BINDIR)/defrag_demo
NUMA_DEMO = $(BINDIR)/numa_demo
MIGRATION_DEMO = $(BINDIR)/migration_demo
FAULT_DEMO = $(BINDIR)/fault_demo
ADVANCED_DEMO = $(BINDIR)/advanced_memory_demo

# 头文件
HEADERS = $(INCDIR)/memory_virtualization.h

# 默认目标
all: directories $(LIBTARGET) $(STATIC_LIB) $(DEMO_TARGET) $(HEADERS) demos $(ADVANCED_DEMO)

# 创建目录
directories:
	@mkdir -p $(OBJDIR) $(BINDIR) $(LIBDIR) $(INCDIR)

# 编译目标文件
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

# 创建动态库
$(LIBTARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Dynamic library created: $@"

# 创建静态库
$(STATIC_LIB): $(OBJECTS)
	ar rcs $@ $^
	@echo "Static library created: $@"

# 编译演示程序
$(DEMO_TARGET): $(DEMO_OBJ) $(LIBTARGET)
	$(CC) -o $@ $< -L$(LIBDIR) -lmemory_virtualization $(LIBS) -Wl,-rpath,$(shell pwd)/$(LIBDIR)
	@echo "Demo executable created: $@"

# 创建演示程序
demos: $(DEFRAG_DEMO) $(NUMA_DEMO) $(MIGRATION_DEMO) $(FAULT_DEMO)

$(DEFRAG_DEMO): $(LIBTARGET)
	$(CC) $(CFLAGS) -o $@ -DDEFRAG_DEMO_MAIN memory_defragmentation.c -L$(LIBDIR) -lmemory_virtualization $(LIBS) -Wl,-rpath,$(shell pwd)/$(LIBDIR)
	@echo "Defragmentation demo created: $@"

$(NUMA_DEMO): $(LIBTARGET)
	$(CC) $(CFLAGS) -o $@ -DNUMA_DEMO_MAIN numa_aware_memory.c -L$(LIBDIR) -lmemory_virtualization $(LIBS) -lnuma -Wl,-rpath,$(shell pwd)/$(LIBDIR)
	@echo "NUMA demo created: $@"

$(MIGRATION_DEMO): $(LIBTARGET)
	$(CC) $(CFLAGS) -o $@ -DMIGRATION_DEMO_MAIN memory_hot_migration.c -L$(LIBDIR) -lmemory_virtualization $(LIBS) -Wl,-rpath,$(shell pwd)/$(LIBDIR)
	@echo "Migration demo created: $@"

$(FAULT_DEMO): $(LIBTARGET)
	$(CC) $(CFLAGS) -o $@ -DFAULT_DEMO_MAIN memory_fault_recovery.c -L$(LIBDIR) -lmemory_virtualization $(LIBS) -Wl,-rpath,$(shell pwd)/$(LIBDIR)
	@echo "Fault recovery demo created: $@"

# 综合演示程序
$(ADVANCED_DEMO): $(ADVANCED_DEMO_OBJ) $(LIBTARGET)
	$(CC) -o $@ $< -L$(LIBDIR) -lmemory_virtualization $(LIBS) -Wl,-rpath,$(shell pwd)/$(LIBDIR)
	@echo "Advanced demo executable created: $@"

# 创建头文件
$(HEADERS): $(SOURCES)
	@echo "Creating unified header file..."
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
	@echo "Header file created: $@"

# 安装目标
install: all
	@echo "Installing memory virtualization library..."
	@mkdir -p /usr/local/lib /usr/local/include /usr/local/bin
	@cp $(LIBTARGET) /usr/local/lib/
	@cp $(STATIC_LIB) /usr/local/lib/
	@cp $(HEADERS) /usr/local/include/
	@cp $(DEMO_TARGET) /usr/local/bin/
	@ldconfig
	@echo "Installation completed"

# 运行演示
run-demo: $(DEMO_TARGET)
	@echo "Running memory virtualization demo..."
	@$(DEMO_TARGET)

# 运行性能测试
run-benchmark: $(DEMO_TARGET)
	@echo "Running performance benchmark..."
	@$(DEMO_TARGET) --duration 60 --threads 8 --size 2048

# 运行压力测试
run-stress: $(DEMO_TARGET)
	@echo "Running stress test..."
	@$(DEMO_TARGET) --duration 120 --threads 16 --size 4096

# 运行单模块测试
run-compression-only: $(DEMO_TARGET)
	@echo "Testing compression module only..."
	@$(DEMO_TARGET) --no-swap --no-unified --no-qos --duration 30

run-swap-only: $(DEMO_TARGET)
	@echo "Testing swap module only..."
	@$(DEMO_TARGET) --no-compression --no-unified --no-qos --duration 30

run-unified-only: $(DEMO_TARGET)
	@echo "Testing unified address space module only..."
	@$(DEMO_TARGET) --no-compression --no-swap --no-qos --duration 30

run-qos-only: $(DEMO_TARGET)
	@echo "Testing QoS module only..."
	@$(DEMO_TARGET) --no-compression --no-swap --no-unified --duration 30

# 代码检查
check: $(SOURCES) $(DEMO_SOURCE)
	@echo "Running static analysis..."
	@which cppcheck > /dev/null && cppcheck --enable=all --std=c99 $(SOURCES) $(DEMO_SOURCE) || echo "cppcheck not found, skipping static analysis"
	@which clang-tidy > /dev/null && clang-tidy $(SOURCES) $(DEMO_SOURCE) -- $(CFLAGS) || echo "clang-tidy not found, skipping linting"

# 内存检查
valgrind-check: $(DEMO_TARGET)
	@echo "Running memory leak detection..."
	@which valgrind > /dev/null && valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes $(DEMO_TARGET) --duration 10 --threads 2 || echo "valgrind not found, skipping memory check"

# 生成文档
docs:
	@echo "Generating documentation..."
	@which doxygen > /dev/null && doxygen Doxyfile || echo "doxygen not found, skipping documentation generation"

# 清理
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(OBJDIR) $(BINDIR) $(LIBDIR) $(INCDIR)
	@rm -f *.o *.so *.a core.*

# 深度清理
distclean: clean
	@echo "Deep cleaning..."
	@rm -rf build/ dist/ *.log *.tmp

# 创建发布包
dist: all
	@echo "Creating distribution package..."
	@mkdir -p dist/memory_virtualization
	@cp $(LIBTARGET) $(STATIC_LIB) dist/memory_virtualization/
	@cp $(HEADERS) dist/memory_virtualization/
	@cp $(DEMO_TARGET) dist/memory_virtualization/
	@cp $(SOURCES) $(DEMO_SOURCE) dist/memory_virtualization/
	@cp Makefile README.md dist/memory_virtualization/
	@cd dist && tar -czf memory_virtualization.tar.gz memory_virtualization/
	@echo "Distribution package created: dist/memory_virtualization.tar.gz"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  all              - Build all targets (default)"
	@echo "  clean            - Remove build artifacts"
	@echo "  distclean        - Deep clean"
	@echo "  install          - Install library and headers"
	@echo "  run-demo         - Run basic demo"
	@echo "  run-benchmark    - Run performance benchmark"
	@echo "  run-stress       - Run stress test"
	@echo "  run-*-only       - Run individual module tests"
	@echo "  check            - Run static analysis"
	@echo "  valgrind-check   - Run memory leak detection"
	@echo "  docs             - Generate documentation"
	@echo "  dist             - Create distribution package"
	@echo "  help             - Show this help message"

# 显示构建信息
info:
	@echo "Build Information:"
	@echo "  Compiler: $(CC)"
	@echo "  Flags: $(CFLAGS)"
	@echo "  Libraries: $(LIBS)"
	@echo "  Sources: $(SOURCES)"
	@echo "  Objects: $(OBJECTS)"
	@echo "  Targets: $(LIBTARGET) $(STATIC_LIB) $(DEMO_TARGET)"

# 依赖关系
$(OBJDIR)/memory_overcommit_advanced.o: memory_overcommit_advanced.c
$(OBJDIR)/memory_compression.o: memory_compression.c
$(OBJDIR)/memory_swap.o: memory_swap.c
$(OBJDIR)/unified_address_space.o: unified_address_space.c
$(OBJDIR)/memory_qos.o: memory_qos.c
$(OBJDIR)/unified_memory_manager.o: unified_memory_manager.c
$(OBJDIR)/memory_defragmentation.o: memory_defragmentation.c
$(OBJDIR)/numa_aware_memory.o: numa_aware_memory.c
$(OBJDIR)/memory_hot_migration.o: memory_hot_migration.c
$(OBJDIR)/memory_fault_recovery.o: memory_fault_recovery.c
$(OBJDIR)/memory_virtualization_demo.o: memory_virtualization_demo.c

.PHONY: all clean distclean install run-demo run-benchmark run-stress \
        run-compression-only run-swap-only run-unified-only run-qos-only \
        check valgrind-check docs dist help info directories

# 确保目录存在
$(OBJECTS): | directories
$(DEMO_OBJ): | directories
$(LIBTARGET): | directories
$(STATIC_LIB): | directories
$(DEMO_TARGET): | directories
$(HEADERS): | directories