/**
 * GPU 显存带宽测试：片内 (D2D) vs 片外 (PCIe)
 *
 * 配套文章: 03_hbm_bandwidth_test.md
 *
 * 测量 device-to-device (D2D) 内部 Copy 带宽，覆盖 1 MB → 1 GB。
 * 自动计算理论峰值带宽并与实测值对比。
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

int main() {
    const size_t sizes[] = {
        1 * 1024 * 1024,      // 1 MB
        16 * 1024 * 1024,     // 16 MB
        64 * 1024 * 1024,     // 64 MB
        256 * 1024 * 1024,    // 256 MB
        1024 * 1024 * 1024    // 1 GB
    };
    const int n = sizeof(sizes) / sizeof(sizes[0]);

    float *d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, sizes[n - 1]));
    CHECK(cudaMalloc(&d_dst, sizes[n - 1]));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int theory_bw = 2.0 * prop.memoryClockRate
                  * (prop.memoryBusWidth / 8) / 1.0e6;
    printf("GPU: %s\n", prop.name);
    printf("Memory clock: %.1f MHz | Bus: %d-bit\n",
           (float)prop.memoryClockRate / 1000.0,
           prop.memoryBusWidth);
    printf("Theoretical peak: %d GB/s\n\n", theory_bw);

    printf("%-12s | %-15s | %-15s\n",
           "Size", "D2D (GB/s)", "% of peak");
    printf("-------------|------------------|------------------\n");

    for (int i = 0; i < n; i++) {
        size_t sz = sizes[i];
        cudaEvent_t start, stop;
        float ms;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);
        CHECK(cudaMemcpyAsync(d_dst, d_src, sz, cudaMemcpyDeviceToDevice, 0));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        float bw = (sz / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        char b[16];
        if (sz >= 1073741824)
            snprintf(b, 16, "%lu GB", sz / 1073741824);
        else
            snprintf(b, 16, "%lu MB", sz / 1048576);

        printf("%-12s | %-15.2f | %-15.1f%%\n",
               b, bw, bw / theory_bw * 100);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    return 0;
}
