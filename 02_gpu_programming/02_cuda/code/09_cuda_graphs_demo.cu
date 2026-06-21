/**
 * CUDA Graphs 编程 Demo
 *
 * 配套文章: 09_cuda_graphs.md
 *
 * 演示 Stream Capture 方式录制 graph + 实例化 + 重放的完整生命周期。
 * 关键点: capture 和 launch 必须使用不同的 stream。
 *
 * 编译: nvcc -arch=sm_80 -O3 -o cuda_graphs_demo 09_cuda_graphs_demo.cu
 * 运行: ./cuda_graphs_demo
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

#define N (1024 * 64)
#define THREADS 256

__global__ void saxpy_kernel(float *d_out, const float *d_in, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_out[idx] = a * d_in[idx] + d_out[idx];
}

__global__ void scale_kernel(float *d, float s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d[idx] *= s;
}

int main() {
    int device; checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop; checkCuda(cudaGetDeviceProperties(&prop, device));

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  CUDA Graphs Demo — %s                  ║\n", prop.name);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    int blocks = (N + THREADS - 1) / THREADS;
    int iters = 10000;

    float *d_in, *d_out;
    checkCuda(cudaMalloc(&d_in, N * sizeof(float)));
    checkCuda(cudaMalloc(&d_out, N * sizeof(float)));

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    cudaStream_t s;
    checkCuda(cudaStreamCreate(&s));

    // ================================================================
    // 1. 传统 stream launch
    // ================================================================
    printf("  [1] 传统 stream launch (%d 次, 每次 2 个 kernel)\n", iters);

    for (int i = 0; i < 100; i++) {
        saxpy_kernel<<<blocks, THREADS>>>(d_out, d_in, 2.0f, N);
        scale_kernel<<<blocks, THREADS>>>(d_out, 0.9f, N);
    }
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        saxpy_kernel<<<blocks, THREADS>>>(d_out, d_in, 2.0f, N);
        scale_kernel<<<blocks, THREADS>>>(d_out, 0.9f, N);
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaDeviceSynchronize());

    float traditional_ms;
    checkCuda(cudaEventElapsedTime(&traditional_ms, start, stop));
    printf("    总: %.3f ms (%.3f μs/iter)\n\n",
           traditional_ms, traditional_ms * 1000 / iters);

    // ================================================================
    // 2. Stream Capture (使用独立 capture stream)
    // ================================================================
    printf("  [2] Stream Capture + Instantiate\n");
    printf("      注意: capture stream 和 launch stream 必须分开!\n");

    cudaStream_t capStream;
    checkCuda(cudaStreamCreate(&capStream));

    checkCuda(cudaStreamBeginCapture(capStream, cudaStreamCaptureModeGlobal));
    float la = 2.0f, ls = 0.9f;
    saxpy_kernel<<<blocks, THREADS, 0, capStream>>>(d_out, d_in, la, N);
    scale_kernel<<<blocks, THREADS, 0, capStream>>>(d_out, ls, N);
    cudaGraph_t graph;
    checkCuda(cudaStreamEndCapture(capStream, &graph));

    cudaGraphExec_t instance;
    checkCuda(cudaEventRecord(start));
    checkCuda(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float instantiate_ms;
    checkCuda(cudaEventElapsedTime(&instantiate_ms, start, stop));
    printf("    Instantiate: %.3f ms\n\n", instantiate_ms);

    // ================================================================
    // 3. Graph Re-launch (在普通 stream 上)
    // ================================================================
    printf("  [3] Graph Re-launch (%d 次, 复用已实例化 graph)\n", iters);

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        checkCuda(cudaGraphLaunch(instance, s));
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaStreamSynchronize(s));

    float graph_ms;
    checkCuda(cudaEventElapsedTime(&graph_ms, start, stop));
    printf("    总: %.3f ms (%.3f μs/iter)\n\n",
           graph_ms, graph_ms * 1000 / iters);

    // Summary
    printf("  ═══════════════════════════════════════════════\n");
    printf("  性能对比 (A100):\n");
    printf("    传统 launch / iter:    %.3f μs\n", traditional_ms * 1000 / iters);
    printf("    Graph re-launch / iter: %.3f μs\n", graph_ms * 1000 / iters);
    printf("    加速比:                %.1fx\n", traditional_ms / graph_ms);
    printf("  ═══════════════════════════════════════════════\n\n");

    checkCuda(cudaGraphExecDestroy(instance));
    checkCuda(cudaGraphDestroy(graph));
    checkCuda(cudaStreamDestroy(s));
    checkCuda(cudaStreamDestroy(capStream));
    checkCuda(cudaFree(d_in)); checkCuda(cudaFree(d_out));
    checkCuda(cudaEventDestroy(start)); checkCuda(cudaEventDestroy(stop));

    printf("  nvcc -arch=sm_80 -O3 -o cuda_graphs_demo 09_cuda_graphs_demo.cu\n");
    printf("  ./cuda_graphs_demo\n");
    return 0;
}
