/**
 * Async Copy & Pipeline Benchmark
 *
 * 配套文章: 16_async_copy_pipeline.md
 *
 * 演示:
 *   1. 同步 Load + __syncthreads (1-stage, baseline)
 *   2. Double Buffer (2-stage, Load[k+1] || Compute[k] 重叠)
 *
 * 编译: nvcc -arch=sm_80 -O3 -o async_copy_bench 16_async_copy_bench.cu
 * 运行: ./async_copy_bench
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

// Config
#define TILE   64
#define K_LOOP 2048
#define ITERS  2000

// ---- Test 1: 同步 1-stage (baseline) ----

__global__ void sync_1stage(float *C, const float *A, const float *B, int N) {
    extern __shared__ float smem[];
    float *As = smem;
    float *Bs = smem + TILE * TILE;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < K_LOOP; k += TILE) {
        if (row < N && (k + threadIdx.x) < K_LOOP)
            As[threadIdx.y * TILE + threadIdx.x] = A[row * K_LOOP + k + threadIdx.x];
        if (col < N && (k + threadIdx.y) < K_LOOP)
            Bs[threadIdx.y * TILE + threadIdx.x] = B[(k + threadIdx.y) * N + col];
        __syncthreads();

        for (int kk = 0; kk < TILE; kk++)
            sum += As[threadIdx.y * TILE + kk] * Bs[kk * TILE + threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

// ---- Test 2: Double Buffer 2-stage ----

__global__ void dbuf_2stage(float *C, const float *A, const float *B, int N) {
    extern __shared__ float smem[];
    float *As0 = smem;
    float *Bs0 = smem + TILE * TILE;
    float *As1 = smem + 2 * TILE * TILE;
    float *Bs1 = smem + 3 * TILE * TILE;

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    // Preload stage 0
    if (row < N && threadIdx.x < K_LOOP)
        As0[threadIdx.y * TILE + threadIdx.x] = A[row * K_LOOP + threadIdx.x];
    if (col < N && threadIdx.y < K_LOOP)
        Bs0[threadIdx.y * TILE + threadIdx.x] = B[threadIdx.y * N + col];
    __syncthreads();

    for (int k = TILE; k < K_LOOP; k += TILE) {
        int stage = (k / TILE) % 2;
        float *As_load = (stage == 0) ? As0 : As1;
        float *Bs_load = (stage == 0) ? Bs0 : Bs1;

        // Load next tile (stage 1 or 0 — opposite of current compute)
        if (row < N && (k + threadIdx.x) < K_LOOP)
            As_load[threadIdx.y * TILE + threadIdx.x] = A[row * K_LOOP + k + threadIdx.x];
        if (col < N && (k + threadIdx.y) < K_LOOP)
            Bs_load[threadIdx.y * TILE + threadIdx.x] = B[(k + threadIdx.y) * N + col];
        __syncthreads();

        // Compute previous stage
        float *As_prev = (stage == 1) ? As0 : As1;
        float *Bs_prev = (stage == 1) ? Bs0 : Bs1;
        for (int kk = 0; kk < TILE; kk++)
            sum += As_prev[threadIdx.y * TILE + kk] * Bs_prev[kk * TILE + threadIdx.x];
        __syncthreads();
    }
    // Compute last stage
    int last = (K_LOOP / TILE) % 2;
    float *As_last = (last == 0) ? As1 : As0;
    float *Bs_last = (last == 0) ? Bs1 : Bs0;
    for (int kk = 0; kk < TILE; kk++)
        sum += As_last[threadIdx.y * TILE + kk] * Bs_last[kk * TILE + threadIdx.x];

    if (row < N && col < N) C[row * N + col] = sum;
}

// ---- Benchmark ----

double bench(void (*kernel)(float*,const float*,const float*,int),
             int N, int smem, const char *label) {
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, N * K_LOOP * sizeof(float)));
    checkCuda(cudaMalloc(&d_B, K_LOOP * N * sizeof(float)));
    checkCuda(cudaMalloc(&d_C, N * N * sizeof(float)));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    for (int i = 0; i < 10; i++) kernel<<<grid, block, smem>>>(d_C, d_A, d_B, N);
    checkCuda(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));
    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) kernel<<<grid, block, smem>>>(d_C, d_A, d_B, N);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaDeviceSynchronize());

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    printf("  %-28s: %8.3f ms total (%dx iters)\n", label, ms, ITERS);

    checkCuda(cudaFree(d_A)); checkCuda(cudaFree(d_B)); checkCuda(cudaFree(d_C));
    checkCuda(cudaEventDestroy(start)); checkCuda(cudaEventDestroy(stop));
    return ms / ITERS;
}

int main() {
    int device; checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop; checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Async Copy & Pipeline Benchmark — %s    ║\n", prop.name);
    printf("║  %dx%d tile, K_LOOP=%d, %d iters            ║\n",
           TILE, TILE, K_LOOP, ITERS);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    int N = 1024;

    double t1 = bench(sync_1stage, N, 2 * TILE * TILE * 4, "Sync 1-stage");
    double t2 = bench(dbuf_2stage, N, 4 * TILE * TILE * 4, "Double Buffer 2-stage");
    printf("\n  加速比: %.2fx\n\n", t1 / t2);

    printf("  SMEM 用量: Sync=%d KB, DoubleBuf=%d KB\n",
           2 * TILE * TILE * 4 / 1024, 4 * TILE * TILE * 4 / 1024);
    printf("  nvcc -arch=sm_80 -O3 -o async_copy_bench 16_async_copy_bench.cu\n");
    printf("  ./async_copy_bench\n");
    return 0;
}
