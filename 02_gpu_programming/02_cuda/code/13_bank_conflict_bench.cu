/**
 * Bank Conflict 验证程序 — 合并版
 *
 * 在 A100 (CC 8.0) 上实测三个关键场景，数据用于文章 13_shared_memory_bank_conflict.md：
 *   1. Stride 访问 — 验证 gcd(stride, 32) 与实测带宽的关系
 *   2. Transpose + Padding — 验证 32-way conflict → 1-way 的收益（实测 2.08x）
 *   3. Reduction: Interleaved vs Sequential — 算法层面的无冲突设计（实测 1.24x）
 *
 * 编译: nvcc -arch=sm_80 -O3 -o bank_conflict_bench bank_conflict_bench.cu
 * 运行: ./bank_conflict_bench
 * 分析: ncu --set memory --section MemoryWorkloadAnalysis_Tables ./bank_conflict_bench
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// ============================================================
// Test 1: Stride 访问 — 每个 warp 内用 lane_id * stride 索引
// ============================================================

// 用 lane_id（而非全局 tid）做索引，保证每个 warp 内 32 线程的 bank conflict
// 模式完全一致。108 blocks × 256 threads = 864 warps 并发 → 充足的延迟隐藏。
#define STRIDE_BLOCKS  108     // 每个 SM 一个 block
#define STRIDE_THREADS 256     // 8 warps/block
#define SMEM_SIZE      4096
#define LOOP_COUNT     8192

__global__ void stride_test(float *dummy, int stride) {
    __shared__ float sdata[SMEM_SIZE];
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    if (tid < SMEM_SIZE) sdata[tid] = (float)tid;
    __syncthreads();

    float sum = 0.0f;
#pragma unroll 1
    for (int i = 0; i < LOOP_COUNT; i++) {
        int idx = (lane_id * stride + i * 32) % SMEM_SIZE;
        sum += sdata[idx];
    }
    if (dummy) dummy[blockIdx.x * blockDim.x + tid] = sum;
}

double measure_stride(int stride) {
    float *d_dummy;
    checkCuda(cudaMalloc(&d_dummy, STRIDE_BLOCKS * STRIDE_THREADS * sizeof(float)));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    for (int i = 0; i < 3; i++)
        stride_test<<<STRIDE_BLOCKS, STRIDE_THREADS>>>(d_dummy, stride);
    checkCuda(cudaDeviceSynchronize());

    const int runs = 10;
    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < runs; i++)
        stride_test<<<STRIDE_BLOCKS, STRIDE_THREADS>>>(d_dummy, stride);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms_total;
    checkCuda(cudaEventElapsedTime(&ms_total, start, stop));
    double ms_per_run = ms_total / runs;

    double total_bytes = (double)STRIDE_BLOCKS * STRIDE_THREADS * LOOP_COUNT * sizeof(float);
    double bw = (total_bytes / (ms_per_run / 1000.0)) / 1e9;

    checkCuda(cudaFree(d_dummy));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    return bw;
}

int gcd(int a, int b) { while (b) { int t = b; b = a % b; a = t; } return a; }

void test1_stride() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Stride 访问 — gcd(stride,32) vs 实测带宽\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("A100 CC 8.0, 每个线程做 %d 次 shared memory 读\n", LOOP_COUNT);
    printf("%d blocks × %d threads = %d 并发线程\n\n",
           STRIDE_BLOCKS, STRIDE_THREADS, STRIDE_BLOCKS * STRIDE_THREADS);

    printf("  %-8s | %-8s | %-14s | %-14s | %s\n",
           "stride", "gcd(s,32)", "实测带宽", "理论占比", "conflict");
    printf("  --------|----------|----------------|----------------|-------------\n");

    int strides[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 31, 32, 33, 64, 96, 128};
    double bw_baseline = 0;

    for (int si = 0; si < (int)(sizeof(strides)/sizeof(strides[0])); si++) {
        int s = strides[si];
        double bw = measure_stride(s);
        if (s == 1) bw_baseline = bw;

        int d = gcd(s, 32);
        double expected_pct = 100.0 / d;

        char conflict_str[32];
        if (d == 1) snprintf(conflict_str, sizeof(conflict_str), "无冲突");
        else snprintf(conflict_str, sizeof(conflict_str), "%d-way", d);

        printf("  %-8d | %-8d | %12.1f GB/s | %12.1f%% | %s\n",
               s, d, bw, expected_pct, conflict_str);
    }

    // stride 1~65 带宽退化连续曲线
    printf("\n  --- stride 1~65 带宽占比 (vs stride=1) ---\n\n  ");
    for (int s = 1; s <= 65; s++) {
        double bw = measure_stride(s);
        double pct = bw_baseline > 0 ? 100.0 * bw / bw_baseline : 0;
        int d = gcd(s, 32);
        if (d > 1) printf("\033[1;31m");
        printf("%3.0f%%", pct);
        if (d > 1) printf("\033[0m");
        if (s % 16 == 0) printf("\n  ");
    }
    printf("\n\n  (红色 = gcd(s,32) > 1, 理论上存在 Bank Conflict)\n\n");
}

// ============================================================
// Test 2: Transpose — 有/无 Padding 的性能对比
// ============================================================

#define TILE_DIM       32
#define TRANSPOSE_ITERS 2000

__global__ void transpose_naive(float *odata, const float *idata, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    if (x < N && y < N)
        tile[threadIdx.x][threadIdx.y] = idata[y * N + x];
    __syncthreads();
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (out_x < N && out_y < N)
        odata[out_y * N + out_x] = tile[threadIdx.y][threadIdx.x];
}

__global__ void transpose_padded(float *odata, const float *idata, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // ← +1 padding
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    if (x < N && y < N)
        tile[threadIdx.x][threadIdx.y] = idata[y * N + x];
    __syncthreads();
    int out_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (out_x < N && out_y < N)
        odata[out_y * N + out_x] = tile[threadIdx.y][threadIdx.x];
}

void test2_transpose() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: Transpose — Padding 消除 Bank Conflict\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("无 padding: tile[32][32] → 列写入 32-way conflict\n");
    printf("有 padding: tile[32][33] → gcd(33,32)=1, 无冲突\n");
    printf("Shared Memory: 4KB vs 4.125KB (浪费 3.1%%)\n");
    printf("%d 次 transpose 取平均\n\n", TRANSPOSE_ITERS);

    int N = 2048;
    dim3 grid(N / TILE_DIM, N / TILE_DIM);

    float *d_idata, *d_odata_naive, *d_odata_padded;
    checkCuda(cudaMalloc(&d_idata, N * N * sizeof(float)));
    checkCuda(cudaMalloc(&d_odata_naive, N * N * sizeof(float)));
    checkCuda(cudaMalloc(&d_odata_padded, N * N * sizeof(float)));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    float time_naive = 0, time_padded = 0;

    // warmup
    transpose_naive<<<grid, dim3(TILE_DIM, TILE_DIM)>>>(d_odata_naive, d_idata, N);
    transpose_padded<<<grid, dim3(TILE_DIM, TILE_DIM)>>>(d_odata_padded, d_idata, N);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < TRANSPOSE_ITERS; i++)
        transpose_naive<<<grid, dim3(TILE_DIM, TILE_DIM)>>>(d_odata_naive, d_idata, N);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&time_naive, start, stop));
    time_naive /= TRANSPOSE_ITERS;

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < TRANSPOSE_ITERS; i++)
        transpose_padded<<<grid, dim3(TILE_DIM, TILE_DIM)>>>(d_odata_padded, d_idata, N);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&time_padded, start, stop));
    time_padded /= TRANSPOSE_ITERS;

    printf("  %-25s: %8.4f ms\n", "无 Padding (32-way)", time_naive);
    printf("  %-25s: %8.4f ms\n", "有 Padding (+1 column)", time_padded);
    printf("  %-25s: %7.2fx\n\n", "加速比", time_naive / time_padded);

    checkCuda(cudaFree(d_idata));
    checkCuda(cudaFree(d_odata_naive));
    checkCuda(cudaFree(d_odata_padded));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
}

// ============================================================
// Test 3: Reduction — Interleaved vs Sequential Addressing
// ============================================================

#define REDUCE_SIZE   2048
#define REDUCE_THREADS 256
#define REDUCE_ITERS  5000

__global__ void reduce_interleaved(float *g_idata, float *g_odata, unsigned int n) {
    __shared__ float sdata[REDUCE_THREADS];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_sequential(float *g_idata, float *g_odata, unsigned int n) {
    __shared__ float sdata[REDUCE_THREADS];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

void test3_reduction() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 3: Reduction — Interleaved vs Sequential Addressing\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("Interleaved: tid * stride → stride 递增，后期 bank conflict\n");
    printf("Sequential:  tid + offset → 连续访问，全程无冲突\n");
    printf("(注意: 差异包含 Bank Conflict + Warp Divergence 两者的叠加)\n");
    printf("%d 次 reduction 取平均\n\n", REDUCE_ITERS);

    int blocks = REDUCE_SIZE / REDUCE_THREADS;

    float *d_idata, *d_odata;
    checkCuda(cudaMalloc(&d_idata, REDUCE_SIZE * sizeof(float)));
    checkCuda(cudaMalloc(&d_odata, blocks * sizeof(float)));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    float time_interleaved = 0, time_sequential = 0;

    reduce_interleaved<<<blocks, REDUCE_THREADS>>>(d_idata, d_odata, REDUCE_SIZE);
    reduce_sequential<<<blocks, REDUCE_THREADS>>>(d_idata, d_odata, REDUCE_SIZE);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < REDUCE_ITERS; i++)
        reduce_interleaved<<<blocks, REDUCE_THREADS>>>(d_idata, d_odata, REDUCE_SIZE);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&time_interleaved, start, stop));
    time_interleaved /= REDUCE_ITERS;

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < REDUCE_ITERS; i++)
        reduce_sequential<<<blocks, REDUCE_THREADS>>>(d_idata, d_odata, REDUCE_SIZE);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&time_sequential, start, stop));
    time_sequential /= REDUCE_ITERS;

    printf("  %-30s: %8.4f ms\n", "Interleaved (有 conflict)", time_interleaved);
    printf("  %-30s: %8.4f ms\n", "Sequential (无 conflict)", time_sequential);
    printf("  %-30s: %7.2fx\n\n", "加速比", time_interleaved / time_sequential);

    checkCuda(cudaFree(d_idata));
    checkCuda(cudaFree(d_odata));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
}

// ============================================================
// main
// ============================================================

int main() {
    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   Bank Conflict Benchmark                            ║\n");
    printf("║   GPU: %-40s ║\n", prop.name);
    printf("║   SM: %d, SMEM/Block: %zu KB                              ║\n",
           prop.multiProcessorCount, prop.sharedMemPerBlock / 1024);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    test1_stride();
    test2_transpose();
    test3_reduction();

    printf("═══════════════════════════════════════════════════════\n");
    printf("如需查看 ncu Bank Conflict 指标:\n");
    printf("  ncu --set memory --section MemoryWorkloadAnalysis_Tables ./bank_conflict_bench\n");
    printf("═══════════════════════════════════════════════════════\n");
    return 0;
}
