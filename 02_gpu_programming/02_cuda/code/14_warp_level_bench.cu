/**
 * Warp-level Programming Benchmark
 *
 * 配套文章: 14_warp_level_programming.md
 *
 * 演示与实测:
 *   1. Shuffle Reduction vs Shared Memory Reduction — 延迟对比
 *   2. Shuffle Prefix Sum (Scan) — __shfl_up_sync
 *   3. Ballot + popc — Warp 内 Filter / Compact
 *   4. Match — Warp 内去重
 *   5. A100 Hardware Reduce — __reduce_add_sync vs 手写 shuffle
 *   6. Cooperative Groups vs 原始 _sync 原语
 *
 * 编译: nvcc -arch=sm_80 -O3 -o warp_bench 14_warp_level_bench.cu
 * 运行: ./warp_bench
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// ============================================================
// Test 1: Shuffle Reduction vs Shared Memory Reduction
// ============================================================

// Shared Memory 版 Warp Reduce
__device__ float smemWarpReduce(float val, volatile float *sdata, int tid) {
    sdata[tid] = val;
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
    return sdata[tid];
}

// Shuffle 版 Warp Reduce
__device__ inline float shuffleWarpReduce(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// A100 硬件 Reduce (CC 8.0+) — 仅支持 int/unsigned, float 需 reinterpret cast
__device__ inline float hwWarpReduce(float val) {
    int ival = __float_as_int(val);
    int result = __reduce_add_sync(0xffffffff, ival);
    return __int_as_float(result);
}

// 使用 Cooperative Groups 的 Warp Reduce
__device__ inline float cgWarpReduce(const cg::thread_block_tile<32, cg::thread_block> &tile, float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += tile.shfl_down(val, offset);
    return val;
}

__global__ void reduce_bench(float *dummy, int iters) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    auto cta = cg::this_thread_block();
    auto tile32 = cg::tiled_partition<32>(cta);
    int lane_id = tile32.thread_rank();
    float val = (float)(lane_id + 1);

    // Warmup: skip timing for this block
    if (blockIdx.x == 0) return;

    float sum = 0.0f;

    // SMEM reduce
    for (int i = 0; i < iters; i++) {
        sum += smemWarpReduce(val, sdata, tid);
    }
    // Shuffle reduce
    for (int i = 0; i < iters; i++) {
        sum += shuffleWarpReduce(val);
    }
    // CG reduce
    for (int i = 0; i < iters; i++) {
        sum += cgWarpReduce(tile32, val);
    }
    // HW reduce (A100+)
    for (int i = 0; i < iters; i++) {
        sum += hwWarpReduce(val);
    }

    if (dummy) dummy[blockIdx.x * 256 + tid] = sum;
}

void test1_reduce_comparison() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("Test 1: Shuffle vs Shared Memory Warp Reduce\n");
    printf("═══════════════════════════════════════════════════════\n");

    int blocks = 108, threads = 256, iters = 10000;
    float *d_dummy;
    checkCuda(cudaMalloc(&d_dummy, blocks * threads * sizeof(float)));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // warmup
    reduce_bench<<<blocks, threads, threads*sizeof(float)>>>(d_dummy, iters);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    reduce_bench<<<blocks, threads, threads*sizeof(float)>>>(d_dummy, iters);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop));
    printf("  4 种 reduce 各 %d 次, 107 blocks × 256 threads:\n", iters);
    printf("    总耗时: %.3f ms\n", ms);

    checkCuda(cudaFree(d_dummy));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
}

// ============================================================
// Test 2: Shuffle Prefix Sum (Scan)
// ============================================================

__device__ float warpPrefixSum(float val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        float tmp = __shfl_up_sync(0xffffffff, val, offset);
        int lane_id = threadIdx.x % 32;
        if (lane_id >= offset) val += tmp;
    }
    return val;
}

__global__ void scan_bench(float *out) {
    int lane_id = threadIdx.x % 32;
    float val = (float)(lane_id + 1);
    float sum = warpPrefixSum(val);
    if (out) out[blockIdx.x * 256 + threadIdx.x] = sum;
}

void test2_scan() {
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Test 2: Shuffle Prefix Sum (Scan)\n");
    printf("═══════════════════════════════════════════════════════\n");

    int blocks = 1, threads = 32;
    float *d_out, h_out[32];
    checkCuda(cudaMalloc(&d_out, 32 * sizeof(float)));

    scan_bench<<<blocks, threads>>>(d_out);
    checkCuda(cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  输入: 1,2,3,...,32\n");
    printf("  Scan 输出 (前 8 个): ");
    for (int i = 0; i < 8; i++) printf("%.0f ", h_out[i]);
    printf("\n  期望: 1,3,6,10,15,21,28,36\n");
    printf("  最后一个 (warp sum): %.0f (期望 528)\n", h_out[31]);

    checkCuda(cudaFree(d_out));
}

// ============================================================
// Test 3: Ballot + popc — Warp Filter
// ============================================================

__global__ void ballot_bench(float *out, float threshold) {
    int lane_id = threadIdx.x % 32;
    float val = (float)(lane_id + 1);
    int keep = (val > threshold) ? 1 : 0;

    unsigned mask = __ballot_sync(0xffffffff, keep);
    int prefix = __popc(mask & ((1u << lane_id) - 1));

    if (keep && out) {
        out[prefix] = val;  // 紧凑写入（仅演示，实际会多 warp 写冲突）
        // 注：这里只用一个 warp 做 demo，所以不会冲突
    }
    // 在全 block 场景中，还需 shared memory 处理跨 warp compact
}

void test3_ballot() {
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Test 3: Ballot + popc — Warp Filter\n");
    printf("═══════════════════════════════════════════════════════\n");

    int blocks = 1, threads = 32;
    float *d_out, h_out[32] = {0};
    checkCuda(cudaMalloc(&d_out, 32 * sizeof(float)));

    ballot_bench<<<blocks, threads>>>(d_out, 15.0f);
    checkCuda(cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  输入: 1-32, threshold=15 (keep elements > 15)\n");
    printf("  keep count: 17 个 (16-32)\n");
    printf("  紧凑输出 (前 17 个): ");
    for (int i = 0; i < 17; i++) printf("%.0f ", h_out[i]);
    printf("\n  期望: 16,17,18,...,32\n");

    checkCuda(cudaFree(d_out));
}

// ============================================================
// Test 4: Match — Warp 内去重
// ============================================================

__global__ void match_bench(float *out) {
    int lane_id = threadIdx.x % 32;
    // 模拟哈希冲突：相邻 4 个线程共享同一个 key
    int key = lane_id / 4;

    unsigned same_key = __match_any_sync(0xffffffff, key);
    int is_first = ((same_key & ((1u << lane_id) - 1)) == 0) ? 1 : 0;

    if (out) out[lane_id] = (float)is_first;
}

void test4_match() {
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("Test 4: Match — Warp 内去重\n");
    printf("═══════════════════════════════════════════════════════\n");

    int blocks = 1, threads = 32;
    float *d_out, h_out[32];
    checkCuda(cudaMalloc(&d_out, 32 * sizeof(float)));

    match_bench<<<blocks, threads>>>(d_out);
    checkCuda(cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  输入: key = lane_id/4 → 4 线程共享 1 个 key\n");
    printf("  期望: 每 4 个线程中只有第 1 个标记为 1 (lane 0,4,8,...)\n");
    printf("  输出: ");
    int count = 0;
    for (int i = 0; i < 32; i++) {
        if (h_out[i] > 0.5f) count++;
    }
    printf("%d 个 unique keys\n", count);
    printf("  期望: 8 个\n");

    checkCuda(cudaFree(d_out));
}

// ============================================================
// main
// ============================================================

int main() {
    int device; checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop; checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Warp-level Programming Benchmark — %s    ║\n", prop.name);
    printf("║  CC %d.%d, Warp Size: %d                             ║\n",
           prop.major, prop.minor, prop.warpSize);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    test1_reduce_comparison();
    test2_scan();
    test3_ballot();
    test4_match();

    printf("\n  nvcc -arch=sm_80 -O3 -o warp_bench 14_warp_level_bench.cu\n");
    printf("  ./warp_bench\n");
    return 0;
}
