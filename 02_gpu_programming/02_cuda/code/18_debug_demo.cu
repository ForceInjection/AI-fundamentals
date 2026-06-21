/**
 * CUDA Debug Demo — 故意引入 Bug 用于验证调试工具
 *
 * 配套文章: 18_cuda_debugging.md
 *
 * 编译: nvcc -lineinfo -arch=sm_80 -o debug_demo 18_debug_demo.cu
 * 验证: compute-sanitizer ./debug_demo
 *       compute-sanitizer --tool racecheck ./debug_demo
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ---- Bug 1: 越界 Shared Memory (tid=31 时访问 sdata[62]) ----
__global__ void smem_oob(float *d) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    sdata[tid * 2] = d[tid];  // tid=31 → sdata[62] 越界!
    __syncthreads();
    d[tid] = sdata[tid];
}

// ---- Bug 2: Race Condition (缺少 __syncthreads) ----
__global__ void race_condition(float *d) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = d[tid];
    // 缺少 __syncthreads()!
    float val = sdata[(tid + 1) % 256];
    d[tid] = val;
}

// ---- Bug 3: 未初始化 Shared Memory 读 ----
__global__ void uninit_smem(float *d) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    d[tid] = sdata[tid];  // sdata 未初始化!
}

// ---- Bug 4: 除零 → NaN ----
__global__ void produce_nan(float *d, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float x = 0.0f, y = 0.0f;
    d[tid] = x / y;  // 0/0 = NaN
}

// ---- Bug 5: 非法地址 (null pointer deref) ----
__global__ void null_ptr() {
    float *bad = (float*)0;
    *bad = 1.0f;  // 写 null 指针
}

int main() {
    float *d;
    cudaMalloc(&d, 256 * sizeof(float));

    printf("=== 1. Shared Memory OOB ===\n");
    smem_oob<<<1, 32>>>(d);
    cudaDeviceSynchronize();
    printf("   cudaError: %s\n\n", cudaGetErrorString(cudaGetLastError()));

    printf("=== 2. Race Condition ===\n");
    race_condition<<<1, 256>>>(d);
    cudaDeviceSynchronize();
    printf("   (race 不触发同步错误，但结果可能不正确)\n\n");

    printf("=== 3. Uninit Shared Memory ===\n");
    uninit_smem<<<1, 32>>>(d);
    cudaDeviceSynchronize();
    printf("   (读未初始化数据不触发同步错误)\n\n");

    printf("=== 4. NaN ===\n");
    produce_nan<<<1, 64>>>(d, 4);
    cudaDeviceSynchronize();
    float h[4];
    cudaMemcpy(h, d, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("   result[0]=%f, isnan=%d\n\n", h[0], isnan(h[0]));

    printf("=== 5. Null Pointer (expected error) ===\n");
    null_ptr<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    printf("   cudaError: %s (%s)\n\n",
           cudaGetErrorName(err), cudaGetErrorString(err));

    cudaFree(d);
    return 0;
}
