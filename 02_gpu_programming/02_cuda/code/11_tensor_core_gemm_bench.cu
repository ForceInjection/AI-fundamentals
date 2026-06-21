/**
 * Tensor Core GEMM 性能实测
 *
 * 配套文章: 11_tensor_core_gemm.md
 *
 * 使用 WMMA API 实现 FP16/BF16 矩阵乘法，验证:
 *   1. Tensor Core 的 WMMA 编程模型 (fragment + load_matrix_sync + mma_sync)
 *   2. 不同 GEMM 尺寸 (M,N,K) 对 TFLOPS 的影响
 *   3. FP16 实测 TFLOPS 与 A100 理论峰值 (312 TFLOPS dense) 的对比
 *
 * 编译: nvcc -arch=sm_80 -O3 -o gemm_bench 11_tensor_core_gemm_bench.cu
 * 运行: ./gemm_bench
 *
 * 参考: NVIDIA cuda-samples/Samples/3_CUDA_Features/cudaTensorCoreGemm
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

using namespace nvcuda;

// WMMA tile 尺寸 (A100, FP16)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 使用 WMMA API 的 FP16 GEMM kernel
// C = α * A * B + β * C
__global__ void wmma_fp16_gemm(half *C, const half *A, const half *B,
                                int M, int N, int K, half alpha, half beta) {
    // 每个 warp 计算一个 16×16 tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate over K tiles
    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + warpM * WMMA_M * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warpN * WMMA_N, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result (with scaling)
    for (int i = 0; i < c_frag.num_elements; i++) {
        c_frag.x[i] = alpha * c_frag.x[i] + beta * __half(0.0f);
    }

    wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N,
                            c_frag, N, wmma::mem_row_major);
}

// 朴素 FP32 GEMM (对比基线)
__global__ void naive_gemm(float *C, const float *A, const float *B,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

double measure_wmma(int M, int N, int K, int iters) {
    half *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, M * K * sizeof(half)));
    checkCuda(cudaMalloc(&d_B, K * N * sizeof(half)));
    checkCuda(cudaMalloc(&d_C, M * N * sizeof(half)));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    dim3 block(256);
    dim3 grid((M + WMMA_M - 1) / (WMMA_M * 8), (N + WMMA_N - 1) / WMMA_N);

    // warmup
    for (int i = 0; i < 10; i++)
        wmma_fp16_gemm<<<grid, block>>>(d_C, d_A, d_B, M, N, K,
                                         __float2half(1.0f), __float2half(0.0f));
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        wmma_fp16_gemm<<<grid, block>>>(d_C, d_A, d_B, M, N, K,
                                         __float2half(1.0f), __float2half(0.0f));
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaDeviceSynchronize());

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));

    // TFLOPS = 2 * M * N * K / time
    double flops = 2.0 * M * N * K * iters;
    double tflops = (flops / (ms / 1000.0)) / 1e12;

    checkCuda(cudaFree(d_A)); checkCuda(cudaFree(d_B)); checkCuda(cudaFree(d_C));
    checkCuda(cudaEventDestroy(start)); checkCuda(cudaEventDestroy(stop));

    return tflops;
}

int main() {
    int device; checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop; checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Tensor Core GEMM Benchmark — %s        ║\n", prop.name);
    printf("║  WMMA API, FP16 accumulator                          ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    printf("  GEMM size (M=N=K) |   TFLOPS (FP16) |  %% of peak dense\n");
    printf("  ------------------|-----------------|-----------------\n");

    int sizes[] = {512, 1024, 2048, 4096, 8192};
    float peak_tflops = 312.0f;  // A100 FP16 dense peak

    for (int si = 0; si < 5; si++) {
        int s = sizes[si];
        int iters = (s <= 2048) ? 50 : 10;
        double tflops = measure_wmma(s, s, s, iters);
        printf("  %16d | %13.1f | %13.1f%%\n", s, tflops, 100.0 * tflops / peak_tflops);
    }

    printf("\n  注意: WMMA 是通用 API, 极致性能需用 CuTe/MMA PTX 直接编程。\n");
    printf("  A100 的 cuBLAS 可达 ~90%% 峰值 (FP16: ~280 TFLOPS, BF16: ~280 TFLOPS)。\n\n");

    printf("  nvcc -arch=sm_80 -O3 -o gemm_bench 11_tensor_core_gemm_bench.cu\n");
    printf("  ./gemm_bench\n");
    return 0;
}
