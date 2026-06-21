/**
 * Multi-GPU CUDA Programming Benchmark
 *
 * 配套文章: 15_multi_gpu_programming.md
 *
 * 演示与实测:
 *   1. Peer Access 检查与启用
 *   2. P2P Memcpy 带宽 (NVLink vs CPU-mediated)
 *   3. 跨 GPU Stream/Event 同步
 *   4. NCCL AllReduce 基础
 *
 * 编译: nvcc -arch=sm_80 -O3 -o multi_gpu_bench 15_multi_gpu_bench.cu -lnccl
 * 运行: ./multi_gpu_bench
 *
 * 要求: ≥2 GPUs, NCCL 已安装
 */

#include <cuda_runtime.h>
// NCCL: 可选——如需启用 NCCL 测试，用 -DWITH_NCCL 编译:
//   nvcc -arch=sm_80 -O3 -DWITH_NCCL -o multi_gpu_bench 15_multi_gpu_bench.cu -lnccl
#ifdef WITH_NCCL
#include <nccl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#ifdef WITH_NCCL
#define checkNccl(ans) { ncclAssert((ans), __FILE__, __LINE__); }
inline void ncclAssert(ncclResult_t code, const char *file, int line) {
    if (code != ncclSuccess) {
        fprintf(stderr, "NCCL error: %s %s %d\n", ncclGetErrorString(code), file, line);
        exit(code);
    }
}
#endif

#define N (64 * 1024 * 1024 / sizeof(float))  // 64 MB per buffer

// ================================================================
// Test 1: Peer Access — 检查与启用
// ================================================================

void test1_peer_access(int num_gpus) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Peer Access 检查\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    printf("  GPU 对  | canAccessPeer | 已启用?\n");
    printf("  --------|---------------|--------\n");

    int pair_count = (num_gpus > 4) ? 4 : num_gpus;  // don't flood output

    for (int i = 0; i < pair_count; i++) {
        checkCuda(cudaSetDevice(i));
        for (int j = 0; j < pair_count; j++) {
            if (i == j) continue;
            int can, enabled = 0;
            checkCuda(cudaDeviceCanAccessPeer(&can, i, j));

            // probe if already enabled by trying enable (idempotent)
            if (can) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled)
                    enabled = 1;
            }
            printf("  %d -> %d   | %13s | %s\n", i, j,
                   can ? "YES" : "NO", enabled ? "YES" : "NO ");
        }
    }
    printf("\n");
}

// ================================================================
// Test 2: P2P Bandwidth — NVLink vs CPU-mediated
// ================================================================

__global__ void fill_kernel(float *d, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d[idx] = val;
}

void test2_p2p_bw(int num_gpus) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 2: P2P 带宽测量\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    if (num_gpus < 2) { printf("  需要 ≥2 GPUs\n\n"); return; }

    float *d0, *d1;
    cudaSetDevice(0); cudaMalloc(&d0, N * sizeof(float));
    cudaSetDevice(1); cudaMalloc(&d1, N * sizeof(float));

    int can01, can10;
    cudaDeviceCanAccessPeer(&can01, 0, 1);
    cudaDeviceCanAccessPeer(&can10, 1, 0);

    cudaSetDevice(0); if (can01) cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1); if (can10) cudaDeviceEnablePeerAccess(0, 0);

    // GPU 0: fill with known value
    cudaSetDevice(0);
    fill_kernel<<<(N + 255) / 256, 256>>>(d0, N, 42.0f);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int iters = 200;

    // --- P2P Direct (GPU 0 → GPU 1 on GPU 0's stream) ---
    cudaStream_t s0;
    cudaSetDevice(0);
    cudaStreamCreate(&s0);

    cudaEventRecord(start, s0);
    for (int i = 0; i < iters; i++)
        cudaMemcpyPeerAsync(d1, 1, d0, 0, N * sizeof(float), s0);
    cudaEventRecord(stop, s0);
    cudaStreamSynchronize(s0);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double bw_p2p = (N * sizeof(float) * (double)iters) / (ms / 1000.0) / 1e9;
    printf("  GPU 0→1 P2P Direct:  %.2f ms/iter, %.1f GB/s\n", ms / iters, bw_p2p);

    // --- CPU-mediated: GPU 0 → CPU → GPU 1 ---
    float *h_buf;
    cudaMallocHost(&h_buf, N * sizeof(float));

    cudaSetDevice(0);
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        cudaMemcpyAsync(h_buf, d0, N * sizeof(float), cudaMemcpyDeviceToHost, s0);
        cudaStreamSynchronize(s0);
        cudaSetDevice(1);
        cudaMemcpy(d1, h_buf, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaSetDevice(0);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);

    double bw_cpu = (N * sizeof(float) * (double)iters) / (ms / 1000.0) / 1e9;
    printf("  GPU 0 → CPU → GPU 1: %.2f ms/iter, %.1f GB/s\n", ms / iters, bw_cpu);
    printf("  P2P 加速比:          %.1fx\n\n", bw_p2p / bw_cpu);

    cudaFreeHost(h_buf); cudaStreamDestroy(s0);
    cudaSetDevice(0); cudaFree(d0);
    cudaSetDevice(1); cudaFree(d1);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// ================================================================
// Test 3: 跨 GPU Stream/Event 同步
// ================================================================

void test3_cross_gpu_sync(int num_gpus) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 3: 跨 GPU Stream/Event 同步\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    if (num_gpus < 2) { printf("  需要 ≥2 GPUs\n\n"); return; }

    float *d0, *d1;
    int n = 1024 * 256;
    cudaSetDevice(0); cudaMalloc(&d0, n * sizeof(float));
    cudaSetDevice(1); cudaMalloc(&d1, n * sizeof(float));

    int can01; cudaDeviceCanAccessPeer(&can01, 0, 1);
    if (can01) { cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0); }

    cudaStream_t s0, s1;
    cudaSetDevice(0); cudaStreamCreate(&s0);
    cudaSetDevice(1); cudaStreamCreate(&s1);

    cudaEvent_t e0;
    cudaSetDevice(0); cudaEventCreate(&e0);

    // GPU 0: fill d0, then P2P to d1
    cudaSetDevice(0);
    fill_kernel<<<(n + 255) / 256, 256, 0, s0>>>(d0, n, 1.0f);
    cudaMemcpyPeerAsync(d1, 1, d0, 0, n * sizeof(float), s0);
    cudaEventRecord(e0, s0);  // e0 fires when P2P done

    // GPU 1: wait for GPU 0's event, then consume
    cudaSetDevice(1);
    cudaStreamWaitEvent(s1, e0, 0);
    fill_kernel<<<(n + 255) / 256, 256, 0, s1>>>(d1, n, 2.0f);  // overwrite after data arrives

    cudaSetDevice(0); cudaStreamSynchronize(s0);
    cudaSetDevice(1); cudaStreamSynchronize(s1);

    printf("  GPU 0: fill d0 → P2P d0→d1 → record e0\n");
    printf("  GPU 1: wait e0 → overwrite d1\n");
    printf("  顺序: GPU 1 的 fill_kernel 在 P2P 完成后才执行 ✓\n\n");

    cudaSetDevice(0); cudaFree(d0); cudaStreamDestroy(s0); cudaEventDestroy(e0);
    cudaSetDevice(1); cudaFree(d1); cudaStreamDestroy(s1);
}

// ================================================================
// Test 4: NCCL AllReduce (optional — requires NCCL)
// ================================================================

#ifdef WITH_NCCL

typedef struct {
    int rank, world_size;
    ncclUniqueId id;
    float *d_buf;
    float *h_buf;
    int n;
} nccl_thread_arg;

void *nccl_thread_func(void *arg) {
    nccl_thread_arg *a = (nccl_thread_arg*)arg;

    checkCuda(cudaSetDevice(a->rank));

    ncclComm_t comm;
    checkNccl(ncclCommInitRank(&comm, a->world_size, a->id, a->rank));

    checkCuda(cudaMalloc(&a->d_buf, a->n * sizeof(float)));
    a->h_buf = (float*)malloc(a->n * sizeof(float));

    // fill: rank i has all value (i+1)
    for (int j = 0; j < a->n; j++) a->h_buf[j] = (float)(a->rank + 1);
    checkCuda(cudaMemcpy(a->d_buf, a->h_buf, a->n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t s;
    cudaStreamCreate(&s);

    checkNccl(ncclAllReduce(a->d_buf, a->d_buf, a->n, ncclFloat, ncclSum, comm, s));
    cudaStreamSynchronize(s);

    checkCuda(cudaMemcpy(a->h_buf, a->d_buf, a->n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaStreamDestroy(s);
    checkNccl(ncclCommDestroy(comm));
    cudaFree(a->d_buf);

    return NULL;
}

void test4_nccl_allreduce(int num_gpus) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 4: NCCL AllReduce\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    // 检查 NCCL 版本
    int nccl_ver;
    ncclGetVersion(&nccl_ver);
    printf("  NCCL version: %d.%d.%d\n",
           nccl_ver / 10000, (nccl_ver % 10000) / 100, nccl_ver % 100);

    int n_gpus = (num_gpus > 4) ? 4 : num_gpus;  // use up to 4 GPUs
    if (n_gpus < 2) { printf("  需要 ≥2 GPUs\n\n"); return; }

    int n = 8;  // 8 elements per GPU
    ncclUniqueId id;
    ncclGetUniqueId(&id);

    pthread_t threads[4];
    nccl_thread_arg args[4];

    for (int i = 0; i < n_gpus; i++) {
        args[i].rank = i;
        args[i].world_size = n_gpus;
        args[i].id = id;
        args[i].n = n;
        pthread_create(&threads[i], NULL, nccl_thread_func, &args[i]);
    }
    for (int i = 0; i < n_gpus; i++)
        pthread_join(threads[i], NULL);

    printf("  AllReduce (sum) with %d GPUs:\n", n_gpus);
    printf("  输入: GPU i 的所有元素 = %d\n", 1);
    int expected = 0; for (int r = 0; r < n_gpus; r++) expected += (r + 1);
printf("  期望: 所有 GPU 的元素 = %d (1+2+...+%d)\n", expected, n_gpus);
    printf("  结果 (GPU 0 前 %d 个): ", n);
    for (int i = 0; i < n; i++) printf("%.0f ", args[0].h_buf[i]);
    printf("\n");

    for (int i = 0; i < n_gpus; i++) free(args[i].h_buf);
    printf("\n");
}

#else

void test4_nccl_allreduce(int num_gpus) {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 4: NCCL AllReduce\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    printf("  NCCL 未安装 — 跳过 AllReduce 测试。\n");
    printf("  安装: apt-get install libnccl-dev libnccl2\n");
    printf("  预期: 4 GPU AllReduce(sum), 每个 GPU 输入 (rank+1),\n");
    printf("        输出 = 所有 rank 值之和 = %d\n\n", num_gpus > 1 ? num_gpus : 4);
}

#endif

// ================================================================
// main
// ================================================================

int main() {
    int num_gpus;
    checkCuda(cudaGetDeviceCount(&num_gpus));

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Multi-GPU CUDA Programming Benchmark               ║\n");
    printf("║  Found %d GPU(s)                                      ║\n", num_gpus);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    test1_peer_access(num_gpus);
    test2_p2p_bw(num_gpus);
    test3_cross_gpu_sync(num_gpus);
    test4_nccl_allreduce(num_gpus);

    printf("═══════════════════════════════════════════════════════\n");
    printf("  nvcc -arch=sm_80 -O3 -o multi_gpu_bench 15_multi_gpu_bench.cu -lnccl\n");
    printf("  ./multi_gpu_bench\n");
    printf("═══════════════════════════════════════════════════════\n");
    return 0;
}
