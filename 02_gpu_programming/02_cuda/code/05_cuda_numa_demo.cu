/**
 * CUDA NUMA API Demo
 *
 * 配套文章: 05_cuda_numa_api.md
 *
 * 演示:
 *   1. cudaMallocHost 的 NUMA 亲和性
 *   2. cudaHostAlloc 的 Portable / WriteCombined 策略
 *   3. cudaMemAdvise + cudaMemPrefetchAsync (Managed Memory)
 *   4. 查看 GPU 所在 NUMA 节点的系统命令
 *
 * 编译: nvcc -arch=sm_80 -O3 -o numa_demo 05_cuda_numa_demo.cu
 * 运行: numactl --cpunodebind=0 ./numa_demo   # 绑定到 NUMA node 0
 *       numactl --cpunodebind=1 ./numa_demo   # 绑定到 NUMA node 1
 *       对比两次的 H2D/D2H 带宽差异
 *
 * 前置: 双路 CPU 系统 + apt install numactl (单路系统也可运行，只是看不到 NUMA 差异)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define SIZE (256 * 1024 * 1024)  // 256 MB
#define ITERS 10

// 简单 kernel：对输入做轻量计算
__global__ void dummy_kernel(float *d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d[idx] = d[idx] * 1.0001f;
}

double measure_h2d_d2h(float *h_buf, float *d_buf, const char *label) {
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    int threads = 256;
    int blocks = (SIZE / sizeof(float) + threads - 1) / threads;

    // warmup
    checkCuda(cudaMemcpy(d_buf, h_buf, SIZE, cudaMemcpyHostToDevice));
    dummy_kernel<<<blocks, threads>>>(d_buf, SIZE / sizeof(float));
    checkCuda(cudaMemcpy(h_buf, d_buf, SIZE, cudaMemcpyDeviceToHost));
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        checkCuda(cudaMemcpy(d_buf, h_buf, SIZE, cudaMemcpyHostToDevice));
        dummy_kernel<<<blocks, threads>>>(d_buf, SIZE / sizeof(float));
        checkCuda(cudaMemcpy(h_buf, d_buf, SIZE, cudaMemcpyDeviceToHost));
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaDeviceSynchronize());

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    double bw = (SIZE * 2.0 * ITERS) / (ms / 1000.0) / 1e9;  // H2D + D2H 总 GB/s

    printf("  %-35s: %7.2f ms/iter, %.1f GB/s\n", label, ms / ITERS, bw);

    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    return bw;
}

int main() {
    int device; checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop; checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  CUDA NUMA API Demo                                  ║\n");
    printf("║  GPU: %-40s ║\n", prop.name);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // ================================================================
    // 0. 显示当前线程的 NUMA 信息
    // ================================================================
    printf("  [0] 系统 NUMA 拓扑\n\n");
#ifdef __linux__
    int cpu = sched_getcpu();
    printf("    当前 CPU: %d\n", cpu);
#endif
    printf("    查看完整拓扑: numactl --hardware\n");
    printf("    查看 GPU NUMA:  nvidia-smi topo -m\n\n");

    // ================================================================
    // 1. cudaMallocHost (默认 NUMA)
    // ================================================================
    printf("  [1] cudaMallocHost (分配在调用线程的 NUMA 节点)\n\n");

    float *h_default, *d_buf;
    checkCuda(cudaMallocHost(&h_default, SIZE));
    checkCuda(cudaMalloc(&d_buf, SIZE));
    for (int i = 0; i < (int)(SIZE / sizeof(float)); i++)
        h_default[i] = (float)i;

    measure_h2d_d2h(h_default, d_buf, "默认 cudaMallocHost");

    // ================================================================
    // 2. cudaHostAlloc + Portable / WriteCombined
    // ================================================================
    printf("\n  [2] cudaHostAlloc — 不同策略\n\n");

    float *h_portable, *h_wc, *h_portable_wc;
    checkCuda(cudaHostAlloc(&h_portable, SIZE, cudaHostAllocPortable));
    checkCuda(cudaHostAlloc(&h_wc, SIZE, cudaHostAllocWriteCombined));
    checkCuda(cudaHostAlloc(&h_portable_wc, SIZE,
                             cudaHostAllocPortable | cudaHostAllocWriteCombined));
    for (int i = 0; i < (int)(SIZE / sizeof(float)); i++) {
        h_portable[i] = (float)i;
        h_wc[i] = (float)i;
        h_portable_wc[i] = (float)i;
    }

    measure_h2d_d2h(h_portable, d_buf, "Portable (跨 NUMA 可映射)");
    measure_h2d_d2h(h_wc, d_buf, "WriteCombined (绕过 cache,纯 H2D)");
    measure_h2d_d2h(h_portable_wc, d_buf, "Portable | WriteCombined");

    // ================================================================
    // 3. Managed Memory: cudaMemAdvise + cudaMemPrefetchAsync
    // ================================================================
    printf("\n  [3] Managed Memory — cudaMemAdvise + cudaMemPrefetchAsync\n\n");

    int managed_supported;
    checkCuda(cudaDeviceGetAttribute(&managed_supported,
                                      cudaDevAttrConcurrentManagedAccess, device));
    if (managed_supported) {
        float *d_um;
        checkCuda(cudaMallocManaged(&d_um, SIZE));

        // Advise: 告诉驱动这块内存主要由 GPU 访问
        checkCuda(cudaMemAdvise(d_um, SIZE, cudaMemAdviseSetPreferredLocation, device));
        checkCuda(cudaMemAdvise(d_um, SIZE, cudaMemAdviseSetAccessedBy, device));

        // Prefetch: 主动将数据迁移到 GPU
        checkCuda(cudaMemPrefetchAsync(d_um, SIZE, device, 0));
        checkCuda(cudaDeviceSynchronize());

        printf("    cudaMemAdviseSetPreferredLocation → GPU %d\n", device);
        printf("    cudaMemAdviseSetAccessedBy       → GPU %d\n", device);
        printf("    cudaMemPrefetchAsync             → GPU %d (主动迁移)\n\n", device);

        checkCuda(cudaFree(d_um));
    } else {
        printf("    Managed Memory 不支持此 GPU\n");
    }

    // cleanup
    checkCuda(cudaFreeHost(h_default));
    checkCuda(cudaFreeHost(h_portable));
    checkCuda(cudaFreeHost(h_wc));
    checkCuda(cudaFreeHost(h_portable_wc));
    checkCuda(cudaFree(d_buf));

    printf("\n  ───────────────────────────────────────────────────\n");
    printf("  验证 NUMA 效果:\n");
    printf("    numactl --cpunodebind=0 ./numa_demo   # 从远端 NUMA 分配\n");
    printf("    numactl --cpunodebind=1 ./numa_demo   # 从 GPU 所在 NUMA 分配\n");
    printf("    对比两次的带宽差异，验证 NUMA 亲和性的影响\n");
    printf("  ───────────────────────────────────────────────────\n\n");

    printf("  nvcc -arch=sm_80 -O3 -o numa_demo 05_cuda_numa_demo.cu\n");
    printf("  ./numa_demo\n");
    return 0;
}
