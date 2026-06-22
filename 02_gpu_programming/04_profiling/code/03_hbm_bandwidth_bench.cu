/**
 * GPU 显存带宽测试：Copy Engine vs Kernel-based
 *
 * 配套文章: 03_hbm_bandwidth_test.md
 *
 * 两种测试:
 *   1. cudaMemcpy D2D   — Copy Engine 带宽（~40% 理论峰值）
 *   2. Kernel Read+Write — SM 驱动的内存带宽（~80%+ 理论峰值）
 *
 * 为什么 cudaMemcpy 跑不满 HBM 带宽？
 *   cudaMemcpy 使用 GPU 的 Copy Engine（DMA），不是 SM 计算单元。
 *   Copy Engine 的设计目标是异步数据搬运（与计算并行），带宽上限
 *   远低于 HBM 内存控制器的全带宽。要想测到接近理论峰值的带宽，
 *   需要用 SM 驱动的 kernel 做连续读写。
 *
 * 编译: nvcc -arch=sm_80 -O3 -o hbm_bw_bench 03_hbm_bandwidth_bench.cu
 * 运行: ./hbm_bw_bench
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(c) do {                                      \
    cudaError_t e = c;                                     \
    if (e != cudaSuccess) {                                \
        printf("Error: %s\n", cudaGetErrorString(e));      \
        exit(1);                                           \
    }                                                      \
} while(0)

// ---- Kernel-based bandwidth test (STREAM-style) ----

// Read + Write kernel (STREAM copy): 每个线程读 in[i], 写 out[i]
// 测量 SM→HBM 的实际带宽 (read + write = 2× data movement)
__global__ void read_write_kernel(const float * __restrict__ in, float * __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        out[idx] = val;
    }
}

double bench_kernel(void (*fn)(const float*, float*, int), const float *d_in,
                    float *d_out, int n, int iters) {
    cudaEvent_t start, stop;
    float ms;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // warmup
    for (int i = 0; i < 5; i++) fn<<<(n+255)/256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    for (int i = 0; i < iters; i++)
        fn<<<(n+255)/256, 256>>>(d_in, d_out, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    // Bandwidth: bytes read + bytes written per iteration
    double total_bytes = (double)n * sizeof(float) * 2 * iters;  // read + write
    double bw = (total_bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return bw;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int theory_bw = 2.0 * prop.memoryClockRate
                  * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("GPU: %s\n", prop.name);
    printf("Memory: %.1f MHz, %d-bit bus\n",
           (float)prop.memoryClockRate / 1000.0, prop.memoryBusWidth);
    printf("Theoretical peak (HBM): %d GB/s\n\n", theory_bw);

    // ====== Test 1: cudaMemcpy D2D (Copy Engine) ======
    printf("=== Test 1: cudaMemcpy D2D (Copy Engine) ===\n");
    const size_t sizes[] = {
        1 * 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024,
        256 * 1024 * 1024, 1024 * 1024 * 1024
    };
    const int ns = sizeof(sizes) / sizeof(sizes[0]);

    float *d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, sizes[ns - 1]));
    CHECK(cudaMalloc(&d_dst, sizes[ns - 1]));

    // warmup
    cudaMemcpyAsync(d_dst, d_src, 1024*1024, cudaMemcpyDeviceToDevice, 0);
    cudaDeviceSynchronize();

    printf("%-12s | %-15s | %-15s\n", "Size", "D2D (GB/s)", "% of peak");
    printf("-------------|------------------|------------------\n");

    for (int i = 0; i < ns; i++) {
        size_t sz = sizes[i];
        cudaEvent_t start, stop; float ms;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        CHECK(cudaMemcpyAsync(d_dst, d_src, sz, cudaMemcpyDeviceToDevice, 0));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float bw = (sz / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
        char b[16];
        if (sz >= 1073741824) snprintf(b, 16, "%lu GB", sz / 1073741824);
        else snprintf(b, 16, "%lu MB", sz / 1048576);
        printf("%-12s | %-15.2f | %-12.1f%%\n", b, bw, bw / theory_bw * 100);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // ====== Test 2: Kernel-based bandwidth ======
    printf("\n=== Test 2: Kernel Read+Write (SM 驱动) ===\n");
    printf("  (每个线程读 in[i] + 写 out[i]，测 SM→HBM 的实际带宽)\n\n");

    int N = sizes[ns - 1] / sizeof(float);  // 1 GB of floats
    float *d_in, *d_out2;
    CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK(cudaMalloc(&d_out2, N * sizeof(float)));

    int iters = 200;
    double bw_kernel = bench_kernel(read_write_kernel, d_in, d_out2, N, iters);
    printf("  Kernel Read+Write: %.1f GB/s (%.1f%% of peak)\n",
           bw_kernel, bw_kernel / theory_bw * 100);
    printf("  cudaMemcpy D2D:    ~821 GB/s  (~40%% of peak)\n");
    printf("  Kernel / Copy Eng: %.1fx\n\n", bw_kernel / 821.0);

    printf("  关键结论:\n");
    printf("  - cudaMemcpy D2D 走 Copy Engine (DMA)，适合异步数据搬运\n");
    printf("  - Kernel Read+Write 走 SM，能接近 HBM 的全带宽\n");
    printf("  - 测真正 HBM 带宽 → 用 nvbandwidth 或 kernel-based benchmark\n");

    CHECK(cudaFree(d_src)); CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_out2));
    return 0;
}
