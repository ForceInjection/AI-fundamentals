/**
 * PCIe 传输效率曲线：从小包到大块
 *
 * 配套文章: 05_pcie_transfer_efficiency.md
 *
 * 展示 1 KB → 1 MB 区间 PCIe H2D/D2H 的带宽爬升曲线，
 * 揭示"多大才够"的效率拐点。
 *
 * 编译: nvcc -arch=sm_80 -O3 -o pcie_transfer_efficiency 05_pcie_transfer_efficiency.cu
 * 运行: ./pcie_transfer_efficiency
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(c) do {                                      \
    cudaError_t r = c;                                     \
    if (r != cudaSuccess) {                                \
        printf("Error: %s\n", cudaGetErrorString(r));      \
        exit(1);                                           \
    }                                                      \
} while(0)

int main() {
    const size_t sizes[] = {
        1024,                    // 1 KB
        4 * 1024,                // 4 KB
        16 * 1024,               // 16 KB
        64 * 1024,               // 64 KB
        256 * 1024,              // 256 KB
        512 * 1024,              // 512 KB
        1024 * 1024,             // 1 MB
        4 * 1024 * 1024,         // 4 MB
        16 * 1024 * 1024,        // 16 MB
        64 * 1024 * 1024,        // 64 MB
        256 * 1024 * 1024,       // 256 MB
    };
    const int n = sizeof(sizes) / sizeof(sizes[0]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    float *h, *d;
    cudaEvent_t s, e;
    float t;
    size_t max_sz = sizes[n - 1];
    CHECK(cudaMallocHost(&h, max_sz));
    CHECK(cudaMalloc(&d, max_sz));

    // warmup: avoid CUDA init overhead on first measurement
    CHECK(cudaMemcpy(d, h, 1024*1024, cudaMemcpyHostToDevice));

    printf("%-10s | %-12s | %-12s | %-12s\n",
           "Size", "H2D (GB/s)", "D2H (GB/s)", "Lat (us)");
    printf("-----------|--------------|--------------|-------------\n");

    for (int i = 0; i < n; i++) {
        size_t sz = sizes[i];
        int reps = sz < 65536 ? 100000 :
                   sz < 262144 ? 10000 :
                   sz < 1048576 ? 5000 :
                   sz < 16777216 ? 2000 : 100;

        cudaEventCreate(&s);
        cudaEventCreate(&e);

        // H2D bandwidth (many iterations for accuracy)
        cudaEventRecord(s, 0);
        for (int j = 0; j < reps; j++)
            CHECK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        cudaEventRecord(e, 0);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&t, s, e);
        float h2d = (sz * reps / (t / 1000.0))
                  / (1024.0 * 1024.0 * 1024.0);

        // D2H bandwidth
        cudaEventRecord(s, 0);
        for (int j = 0; j < reps; j++)
            CHECK(cudaMemcpy(h, d, sz, cudaMemcpyDeviceToHost));
        cudaEventRecord(e, 0);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&t, s, e);
        float d2h = (sz * reps / (t / 1000.0))
                  / (1024.0 * 1024.0 * 1024.0);

        // Single transfer latency
        cudaEventRecord(s, 0);
        CHECK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
        cudaEventRecord(e, 0);
        cudaEventSynchronize(e);
        cudaEventElapsedTime(&t, s, e);
        float lat = t * 1000;  // ms -> us

        char b[16];
        if (sz >= 1048576)
            snprintf(b, 16, "%lu MB", sz / 1048576);
        else if (sz >= 1024)
            snprintf(b, 16, "%lu KB", sz / 1024);
        printf("%-10s | %-12.2f | %-12.2f | %-12.1f\n",
               b, h2d, d2h, lat);

        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }

    CHECK(cudaFreeHost(h));
    CHECK(cudaFree(d));
    return 0;
}
