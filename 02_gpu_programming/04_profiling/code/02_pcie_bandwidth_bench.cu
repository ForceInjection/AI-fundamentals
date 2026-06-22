/**
 * PCIe 链路状态与主机-设备带宽实测
 *
 * 配套文章: 02_pcie_bandwidth_measurement.md
 *
 * 零依赖 CUDA 程序，测量 Host↔Device 传输带宽。
 * 覆盖 1 MB → 1 GB 的 5 个数据点，展示 PCIe 带宽随传输大小的变化。
 *
 * 编译: nvcc -arch=sm_80 -O3 -o pcie_bw_bench 02_pcie_bandwidth_bench.cu
 * 运行: ./pcie_bw_bench
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(cmd) do {                                    \
    cudaError_t e = cmd;                                   \
    if (e != cudaSuccess) {                                \
        printf("CUDA error: %s\n", cudaGetErrorString(e)); \
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
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // 显示 GPU 与 PCIe 信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("PCIe Gen: (check with: nvidia-smi --query-gpu=pcie.link.gen.max --format=csv)\n\n");

    float *h_buf, *d_buf;
    CHECK(cudaMallocHost(&h_buf, sizes[num_sizes - 1]));
    CHECK(cudaMalloc(&d_buf, sizes[num_sizes - 1]));

    // warmup: avoid CUDA context init overhead affecting first measurement
    CHECK(cudaMemcpyAsync(d_buf, h_buf, 1024*1024, cudaMemcpyHostToDevice, 0));
    cudaDeviceSynchronize();

    printf("%-12s | %-15s | %-15s\n", "Size", "H2D (GB/s)", "D2H (GB/s)");
    printf("-------------|------------------|------------------\n");

    for (int i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];
        cudaEvent_t start, stop;
        float ms;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Host -> Device
        cudaEventRecord(start, 0);
        CHECK(cudaMemcpyAsync(d_buf, h_buf, n, cudaMemcpyHostToDevice, 0));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        float h2d = (n / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        // Device -> Host
        cudaEventRecord(start, 0);
        CHECK(cudaMemcpyAsync(h_buf, d_buf, n, cudaMemcpyDeviceToHost, 0));
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        float d2h = (n / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

        char size_str[16];
        if (n >= 1024 * 1024 * 1024)
            snprintf(size_str, 16, "%lu GB", n / (1024 * 1024 * 1024));
        else
            snprintf(size_str, 16, "%lu MB", n / (1024 * 1024));

        printf("%-12s | %-15.2f | %-15.2f\n", size_str, h2d, d2h);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    CHECK(cudaFreeHost(h_buf));
    CHECK(cudaFree(d_buf));
    return 0;
}
