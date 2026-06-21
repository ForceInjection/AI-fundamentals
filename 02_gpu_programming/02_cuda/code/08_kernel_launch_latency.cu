/**
 * Kernel Launch 开销测量
 *
 * 配套文章: 08_kernel_launch_latency.md
 *
 * 测量:
 *   1. 空 kernel 的 launch 延迟
 *   2. 不同 grid/block 配置对 launch 开销的影响
 *   3. CPU vs GPU 的决策边界
 *
 * 编译: nvcc -arch=sm_80 -O3 -o launch_latency 08_kernel_launch_latency.cu
 * 运行: ./launch_latency
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// 空 kernel
__global__ void empty_kernel() {}

// 轻量 kernel (做一个操作，确认 launch overhead 独立于 kernel 执行)
__global__ void tiny_kernel(float *d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d[idx] = d[idx] * 1.001f;
}

__global__ void bigger_kernel(float *d, int n, int loops) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = d[idx];
    for (int i = 0; i < loops; i++) val = val * 0.999f + 0.001f;
    d[idx] = val;
}

double measure_empty_launch(int blocks, int threads, int iters) {
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // warmup
    for (int i = 0; i < 100; i++)
        empty_kernel<<<blocks, threads>>>();
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        empty_kernel<<<blocks, threads>>>();
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaDeviceSynchronize());

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    return ms * 1000 / iters;  // 返回 μs
}

int main() {
    int device; checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop; checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Kernel Launch 开销 — %s                ║\n", prop.name);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // ================================================================
    // 1. 空 kernel launch 延迟
    // ================================================================
    printf("  [1] 空 kernel 单次 launch 延迟\n\n");
    printf("  %-12s | %-12s | %10s\n", "Blocks", "Threads", "延迟 (μs)");
    printf("  --------------|--------------|------------\n");

    int configs[][2] = {{1,1}, {1,32}, {1,128}, {1,512}, {1,1024},
                         {10,256}, {100,256}};
    for (int c = 0; c < 7; c++) {
        int b = configs[c][0], t = configs[c][1];
        int iters = (b * t <= 32) ? 10000 : 1000;
        double lat = measure_empty_launch(b, t, iters);
        printf("  %12d | %12d | %8.2f\n", b, t, lat);
    }

    // ================================================================
    // 2. CPU vs GPU 决策边界
    // ================================================================
    printf("\n  [2] 决策边界: 多少元素值得 launch GPU 一次?\n\n");

    int N = 1024;
    float *h = (float*)malloc(N * sizeof(float));
    float *d;
    checkCuda(cudaMalloc(&d, N * sizeof(float)));

    for (int i = 0; i < N; i++) h[i] = (float)i;
    checkCuda(cudaMemcpy(d, h, N * sizeof(float), cudaMemcpyHostToDevice));

    // CPU baseline
    cudaEvent_t cpustart, cpustop;
    checkCuda(cudaEventCreate(&cpustart));
    checkCuda(cudaEventCreate(&cpustop));

    checkCuda(cudaEventRecord(cpustart));
    for (int iter = 0; iter < 10000; iter++)
        for (int i = 0; i < N; i++) h[i] = h[i] * 1.001f;
    checkCuda(cudaEventRecord(cpustop));
    checkCuda(cudaEventSynchronize(cpustop));

    float cpu_ms;
    checkCuda(cudaEventElapsedTime(&cpu_ms, cpustart, cpustop));

    // GPU
    checkCuda(cudaEventRecord(cpustart));
    for (int iter = 0; iter < 10000; iter++)
        tiny_kernel<<<4, 256>>>(d, N);
    checkCuda(cudaEventRecord(cpustop));
    checkCuda(cudaEventSynchronize(cpustop));

    float gpu_ms;
    checkCuda(cudaEventElapsedTime(&gpu_ms, cpustart, cpustop));

    printf("  %d elements × 10000 iters:\n", N);
    printf("    CPU: %.2f ms\n", cpu_ms);
    printf("    GPU: %.2f ms\n", gpu_ms);
    printf("    单次 GPU launch overhead ≈ %.2f μs (前 1000 次 empty kernel)\n\n",
           measure_empty_launch(4, 256, 1000));

    printf("  结论: 当 N < ~%d 时, GPU launch 开销 > 计算时间,\n",
           (int)(N * (gpu_ms - cpu_ms) / cpu_ms + N));
    printf("        对于小数据量的 kernel, CPU 端直接算可能更快。\n\n");

    free(h); checkCuda(cudaFree(d));
    checkCuda(cudaEventDestroy(cpustart));
    checkCuda(cudaEventDestroy(cpustop));

    printf("  nvcc -arch=sm_80 -O3 -o launch_latency 08_kernel_launch_latency.cu\n");
    printf("  ./launch_latency\n");
    return 0;
}
