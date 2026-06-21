/**
 * Reduction Benchmark — 从朴素实现到 Warp Shuffle (8 个 kernel 变体)
 *
 * 配套文章: 10_reduction.md
 *
 * 展示的优化级别:
 *   K0: Interleaved Addressing (朴素实现)
 *   K1: 修复 Warp Divergence (位运算替代取模)
 *   K2: Sequential Addressing (stride 从大到小)
 *   K3: 展开最后一个 Warp
 *   K4: Warp Shuffle (寄存器级 reduce)
 *   K5: Template 编译期展开
 *   K6: Cooperative Groups + Grid-stride Loop
 *
 * 编译: nvcc -arch=sm_80 -O3 -o reduction_bench 10_reduction_bench.cu
 * 运行: ./reduction_bench [--n 1048576]
 *
 * 基于 NVIDIA cuda-samples reduction 改写，适配文章讲解结构。
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

namespace cg = cooperative_groups;

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// ============================================================
// Utility
// ============================================================

inline int nextPow2(int n) { n--; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16; return ++n; }
inline bool isPow2(int n) { return (n & (n - 1)) == 0; }
unsigned int floorPow2(unsigned int n) { n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16; return n - (n >> 1); }

// ============================================================
// K0: Interleaved Addressing (朴素实现)
// ============================================================

__global__ void reduce0(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2 * s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// K1: 修复 Warp Divergence (位运算替代取模)
// ============================================================

__global__ void reduce1(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if ((tid & (2 * s - 1)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// K2: Sequential Addressing (stride 从大到小)
// ============================================================

__global__ void reduce2(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// K3: 展开最后一个 Warp
// ============================================================

__global__ void reduce3(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// K4: Warp Shuffle (寄存器级 reduce)
// ============================================================

__global__ void reduce4(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    sdata[tid] = mySum;
    __syncthreads();

    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            mySum += sdata[tid + s];
            sdata[tid] = mySum;
        }
        cg::sync(cta);
    }

    if (cta.thread_rank() < 32) {
        if (blockDim.x >= 64) mySum += sdata[tid + 32];
        for (int offset = 16; offset > 0; offset /= 2)
            mySum += tile32.shfl_down(mySum, offset);
    }
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

// ============================================================
// K5: Template 编译期展开
// ============================================================

template <unsigned int blockSize>
__global__ void reduce5(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize + tid;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    if (tid < 32) {
        if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
        if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
        if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
        if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
        if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
        if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// K6: Cooperative Groups + Grid-stride Loop (综合最优)
// ============================================================

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    float mySum = 0;

    unsigned int i = blockIdx.x * blockSize * 2 + tid;
    if (nIsPow2) {
        while (i < n) {
            mySum += g_idata[i];
            if (i + blockSize < n) mySum += g_idata[i + blockSize];
            i += gridSize;
        }
    } else {
        while (i < n) {
            mySum += g_idata[i];
            if (i + blockSize < n) mySum += g_idata[i + blockSize];
            i += gridSize;
        }
    }

    sdata[tid] = mySum;
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
        if (blockSize >=  64) { mySum += sdata[tid + 32]; }
        for (int offset = 16; offset > 0; offset /= 2)
            mySum += tile32.shfl_down(mySum, offset);
    }
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

// ============================================================
// Host-side reduce (最终结果 CPU 汇总)
// ============================================================

void reduce_cpu(float *data, int n, float *result) {
    *result = 0;
    for (int i = 0; i < n; i++) *result += data[i];
}

// ============================================================
// Benchmark runner
// ============================================================

typedef void (*reduce_fn_shared)(float*, float*, unsigned int);
typedef void (*reduce_fn_tmpl)(float*, float*, unsigned int);

double bench_kernel(int kernel_id, int blockSize, float *d_idata, float *d_odata,
                    int numBlocks, int n, int iters, int sharedMem) {
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // warmup
    for (int i = 0; i < 3; i++) {
        switch (kernel_id) {
            case 0: reduce0<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 1: reduce1<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 2: reduce2<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 3: reduce3<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 4: reduce4<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 5: reduce5<256><<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 6: reduce6<256, false><<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
        }
    }
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        switch (kernel_id) {
            case 0: reduce0<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 1: reduce1<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 2: reduce2<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 3: reduce3<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 4: reduce4<<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 5: reduce5<256><<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
            case 6: reduce6<256, false><<<numBlocks, blockSize, sharedMem>>>(d_idata, d_odata, n); break;
        }
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    return ms / iters;
}

int main(int argc, char **argv) {
    int device; checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop; checkCuda(cudaGetDeviceProperties(&prop, device));

    int N = 1048576;  // default 1M elements
    if (argc > 1 && strcmp(argv[1], "--n") == 0 && argc > 2) N = atoi(argv[2]);
    int maxThreads = (prop.maxThreadsPerBlock > 256) ? 256 : prop.maxThreadsPerBlock;
    int numBlocks = (N + maxThreads * 2 - 1) / (maxThreads * 2);
    if (numBlocks > prop.maxGridSize[0]) numBlocks = prop.maxGridSize[0];
    int sharedMem = maxThreads * sizeof(float);
    int iters = 100;

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Reduction Benchmark — %s               ║\n", prop.name);
    printf("║  %d elements, %d threads/block, %d blocks         ║\n", N, maxThreads, numBlocks);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // Allocate
    float *h_idata = (float*)malloc(N * sizeof(float));
    float *d_idata, *d_odata;
    checkCuda(cudaMalloc(&d_idata, N * sizeof(float)));
    checkCuda(cudaMalloc(&d_odata, numBlocks * sizeof(float)));

    for (int i = 0; i < N; i++) h_idata[i] = 1.0f;  // 所有元素 = 1，结果应为 N
    checkCuda(cudaMemcpy(d_idata, h_idata, N * sizeof(float), cudaMemcpyHostToDevice));

    // Benchmark all kernels
    const char *names[] = {
        "K0: Interleaved  ", "K1: Div Fix      ", "K2: Sequential   ",
        "K3: Unroll Warp  ", "K4: Warp Shuffle ", "K5: Template     ",
        "K6: CG + Grid    "
    };
    double times[7];
    double baseline = 0;

    printf("  %-20s | %10s | %10s\n", "Kernel", "Time (ms)", "加速比");
    printf("  ----------------------|------------|------------\n");

    for (int k = 0; k < 7; k++) {
        times[k] = bench_kernel(k, maxThreads, d_idata, d_odata, numBlocks, N, iters, sharedMem);
        if (k == 0) baseline = times[0];
        printf("  %-20s | %8.4f   | %7.2fx\n", names[k], times[k], baseline / times[k]);
    }

    // Verify correctness (K6)
    printf("\n  --- 验证: K6 reduce = %.1f (expected %d) ---\n",
           /* read back */ ({
               float result[1] = {0};
               reduce6<256,false><<<numBlocks, maxThreads, sharedMem>>>(d_idata, d_odata, N);
               float *h_odata = (float*)malloc(numBlocks * sizeof(float));
               cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
               float sum = 0; for (int i = 0; i < numBlocks; i++) sum += h_odata[i];
               free(h_odata);
               sum;
           }), N);

    free(h_idata);
    checkCuda(cudaFree(d_idata));
    checkCuda(cudaFree(d_odata));

    printf("\n  nvcc -arch=sm_80 -O3 -o reduction_bench 10_reduction_bench.cu\n");
    printf("  ./reduction_bench [--n 1048576]\n");
    return 0;
}
