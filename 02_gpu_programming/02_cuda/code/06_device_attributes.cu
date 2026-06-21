/**
 * GPU 设备属性查询工具
 *
 * 配套文章: 06_device_attributes.md
 *
 * 使用 cudaDeviceGetAttribute 查询 20+ 种底层硬件能力，覆盖:
 *   - PCIe 原子操作 & GPUDirect 支持
 *   - Managed Memory 能力
 *   - Cooperative Launch 支持
 *   - SM 资源限制
 *   - 数值精度特性
 *   - 多 GPU 拓扑信息
 *
 * 编译: nvcc -arch=sm_80 -O3 -o device_attrs 06_device_attributes.cu
 * 运行: ./device_attrs
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_ATTR(attr, id) do {                              \
    int v;                                                      \
    cudaError_t e = cudaDeviceGetAttribute(&v, attr, id);      \
    if (e == cudaSuccess)                                       \
        printf("  %-50s = %d%s\n", #attr, v, v ? " ✓" : "");   \
    else                                                        \
        printf("  %-50s = N/A (%s)\n", #attr, cudaGetErrorString(e)); \
} while(0)

#define SECTION(title) printf("\n  --- %s ---\n", title)

int main() {
    int count;
    cudaGetDeviceCount(&count);
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  GPU Device Attributes                              ║\n");
    printf("║  Found %d GPU(s)                                      ║\n", count);
    printf("╚══════════════════════════════════════════════════════╝\n");

    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("\nGPU %d: %s (CC %d.%d, %zu MB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024 * 1024));

        SECTION("PCIe & Atomic (GPUDirect P2P 前提)");
        CHECK_ATTR(cudaDevAttrHostNativeAtomicSupported, i);
        CHECK_ATTR(cudaDevAttrCanUseHostPointerForRegisteredMem, i);
        CHECK_ATTR(cudaDevAttrCanMapHostMemory, i);

        SECTION("Managed Memory");
        CHECK_ATTR(cudaDevAttrConcurrentManagedAccess, i);
        CHECK_ATTR(cudaDevAttrPageableMemoryAccess, i);
        CHECK_ATTR(cudaDevAttrPageableMemoryAccessUsesHostPageTables, i);
        CHECK_ATTR(cudaDevAttrDirectManagedMemAccessFromHost, i);

        SECTION("Launch Capability");
        CHECK_ATTR(cudaDevAttrCooperativeLaunch, i);
        CHECK_ATTR(cudaDevAttrCooperativeMultiDeviceLaunch, i);

        SECTION("SM Resources");
        CHECK_ATTR(cudaDevAttrMaxRegistersPerMultiprocessor, i);
        CHECK_ATTR(cudaDevAttrMaxBlocksPerMultiprocessor, i);
        CHECK_ATTR(cudaDevAttrMaxThreadsPerMultiProcessor, i);
        CHECK_ATTR(cudaDevAttrAsyncEngineCount, i);
        CHECK_ATTR(cudaDevAttrWarpSize, i);
        CHECK_ATTR(cudaDevAttrMaxSharedMemoryPerBlockOptin, i);

        SECTION("Precision");
        CHECK_ATTR(cudaDevAttrSingleToDoublePrecisionPerfRatio, i);

        SECTION("Compute");
        CHECK_ATTR(cudaDevAttrMultiGpuBoardGroupID, i);
        CHECK_ATTR(cudaDevAttrComputeCapabilityMajor, i);
        CHECK_ATTR(cudaDevAttrComputeCapabilityMinor, i);

        SECTION("Multi-GPU Topology");
        for (int j = 0; j < count; j++) {
            if (i == j) continue;
            int access;
            cudaDeviceCanAccessPeer(&access, i, j);
            printf("  GPU %d → GPU %d: P2P %s\n", i, j,
                   access ? "✓ enabled" : "✗ disabled");
        }
    }

    printf("\n  nvcc -arch=sm_80 -O3 -o device_attrs 06_device_attributes.cu\n");
    printf("  ./device_attrs\n");
    return 0;
}
