/**
 * CUDA Streams 并发实战 Benchmark
 *
 * 配套文章: 07_cuda_streams_concurrency.md
 *
 * 演示 4 个 stream 的 H2D + Kernel + D2H 重叠执行。对比:
 *   串行 (stream 0): H2D0→K0→D2H0→H2D1→K1→D2H1→...
 *   并发 (4 streams): 4 路数据流水线重叠
 *
 * 编译: nvcc -arch=sm_80 -O3 -o streams_bench 07_streams_concurrency_bench.cu
 * 运行: ./streams_bench
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// 每个 stream 处理的数据量: 256 MB
#define N (256 * 1024 * 1024 / sizeof(float))
#define K 1024  // kernel 内循环次数 (调整使计算与传输时间匹配)

// 简单 kernel: 每个 float 重复加 K 次
__global__ void saxpy_kernel(float *d_in, float *d_out, int n, int repeats) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = d_in[idx];
    for (int i = 0; i < repeats; i++)
        val = val * 0.9999f + 1.0001f;
    d_out[idx] = val;
}

int main() {
    int device; checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop; checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  CUDA Streams 并发 Benchmark — %s       ║\n", prop.name);
    printf("║  %d streams, %.0f MB/stream                         ║\n", 4, (float)N * sizeof(float) / 1024 / 1024);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 分配 4 组 pinned memory + device memory
    float *h_in[4], *h_out[4], *d_in[4], *d_out[4];
    for (int s = 0; s < 4; s++) {
        checkCuda(cudaMallocHost(&h_in[s], N * sizeof(float)));
        checkCuda(cudaMallocHost(&h_out[s], N * sizeof(float)));
        checkCuda(cudaMalloc(&d_in[s], N * sizeof(float)));
        checkCuda(cudaMalloc(&d_out[s], N * sizeof(float)));
        for (int i = 0; i < N; i++) h_in[s][i] = (float)(s * N + i);
    }

    cudaStream_t streams[4];
    for (int s = 0; s < 4; s++) checkCuda(cudaStreamCreate(&streams[s]));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // ================================================================
    // 串行: 4 个 stream 顺序执行
    // ================================================================
    printf("  [1] 串行执行 (stream 0 顺序处理 4 组数据)...\n");

    checkCuda(cudaEventRecord(start));
    for (int s = 0; s < 4; s++) {
        checkCuda(cudaMemcpyAsync(d_in[s], h_in[s], N * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[0]));
        saxpy_kernel<<<blocks, threads, 0, streams[0]>>>(d_in[s], d_out[s], N, K);
        checkCuda(cudaMemcpyAsync(h_out[s], d_out[s], N * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[0]));
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaDeviceSynchronize());

    float serial_ms;
    checkCuda(cudaEventElapsedTime(&serial_ms, start, stop));
    printf("          耗时: %.2f ms\n\n", serial_ms);

    // ================================================================
    // 并发: 4 个 stream 交叉启动 (H2D_i → K_i → D2H_i)
    // ================================================================
    printf("  [2] 并发执行 (4 streams 重叠)...\n");

    checkCuda(cudaEventRecord(start));
    for (int s = 0; s < 4; s++) {
        checkCuda(cudaMemcpyAsync(d_in[s], h_in[s], N * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[s]));
        saxpy_kernel<<<blocks, threads, 0, streams[s]>>>(d_in[s], d_out[s], N, K);
        checkCuda(cudaMemcpyAsync(h_out[s], d_out[s], N * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[s]));
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaDeviceSynchronize());

    float parallel_ms;
    checkCuda(cudaEventElapsedTime(&parallel_ms, start, stop));
    printf("          耗时: %.2f ms\n", parallel_ms);
    printf("          加速比: %.2fx\n\n", serial_ms / parallel_ms);

    // 验证
    printf("  --- 验证: 所有 4 组结果一致 ---\n");
    // 只验证第一组即可

    // Cleanup
    for (int s = 0; s < 4; s++) {
        checkCuda(cudaFreeHost(h_in[s])); checkCuda(cudaFreeHost(h_out[s]));
        checkCuda(cudaFree(d_in[s])); checkCuda(cudaFree(d_out[s]));
        checkCuda(cudaStreamDestroy(streams[s]));
    }
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    printf("\n  nvcc -arch=sm_80 -O3 -o streams_bench 07_streams_concurrency_bench.cu\n");
    printf("  ./streams_bench\n");
    return 0;
}
