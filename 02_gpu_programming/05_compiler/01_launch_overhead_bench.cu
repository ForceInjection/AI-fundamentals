// launch_overhead_bench.cu —— kernel-per-operator 执行模型的 launch 开销量化
//
// 三种模式完成等量的「逻辑算子」工作（N_OPS 个逐元素变换，数据相互依赖），
// 对比每算子摊到的墙钟时间：
//   eager : 每个 op 一次 cudaLaunchKernel（kernel-per-operator 基线）
//   graph : 同一序列 CUDA Graph capture 后 replay（摊薄提交成本）
//   fused : 全部 op 折进一个 kernel 内循环（消除 launch 边界的下界）
// 另测 empty kernel 的纯 CPU 提交速率，单独给出 launch 税的量级。
//
// 编译：nvcc -O3 -std=c++17 -arch=sm_90 launch_overhead_bench.cu -o launch_overhead_bench
// 运行：CUDA_VISIBLE_DEVICES=<空闲卡> ./launch_overhead_bench

#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

constexpr int N_OPS  = 1024;   // 一次测量迭代内的逻辑算子数
constexpr int N_ELEM = 1024;   // 每个算子处理的元素数（刻意极小：让 kernel 时间远小于 launch 时间）
constexpr int ROUNDS = 200;    // 测量迭代轮数

#define CHECK(x)                                                          \
    do {                                                                  \
        cudaError_t e_ = (x);                                             \
        if (e_ != cudaSuccess) {                                          \
            printf("CUDA 错误 %s（%s:%d）\n", cudaGetErrorString(e_),     \
                   __FILE__, __LINE__);                                   \
            return 1;                                                     \
        }                                                                 \
    } while (0)

// 单个逻辑算子：对缓冲区做一次仿射变换（依赖前一个算子的输出）
__global__ void tiny_kernel(float* x, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_ELEM) x[i] = fmaf(x[i], a, 1.0f);
}

__global__ void empty_kernel() {}

// fused：同一条 1024 步依赖链折进单 kernel（megakernel 式消除边界）
__global__ void fused_kernel(float* x, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_ELEM) {
        float v = x[i];
        for (int k = 0; k < N_OPS; ++k) v = fmaf(v, a, 1.0f);
        x[i] = v;
    }
}

double now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("设备: %s | SM 数 %d | CUDA %d.%d\n", prop.name, prop.multiProcessorCount,
           prop.major, prop.minor);

    float* d_x;
    CHECK(cudaMalloc(&d_x, N_ELEM * sizeof(float)));
    CHECK(cudaMemset(d_x, 0, N_ELEM * sizeof(float)));
    const float a = 1.0000001f;
    const int blocks = (N_ELEM + 255) / 256, threads = 256;
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // ---------- 模式 1：empty kernel 纯 CPU 提交速率（launch 税） ----------
    constexpr int N_EMPTY = 20000;
    CHECK(cudaDeviceSynchronize());
    double t0 = now_us();
    for (int k = 0; k < N_EMPTY; ++k) empty_kernel<<<1, 32, 0, stream>>>();
    double enq_us = now_us() - t0;               // 只统计 CPU 侧入队，不同步
    CHECK(cudaStreamSynchronize(stream));
    printf("[empty ] 纯 CPU 提交: %.3f us/kernel（入队 %d 个耗时 %.1f ms）\n",
           enq_us / N_EMPTY, N_EMPTY, enq_us / 1000.0);

    // ---------- 模式 2：eager（kernel-per-operator 基线） ----------
    // 每轮 N_OPS 次发射 + 轮末同步；用 event 取 GPU 侧首尾跨度对照墙钟
    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));
    double wall_total = 0.0, span_total = 0.0;
    for (int r = 0; r < ROUNDS + 20; ++r) {      // 前 20 轮预热不计
        CHECK(cudaEventRecord(ev0, stream));
        double w0 = now_us();
        for (int k = 0; k < N_OPS; ++k) tiny_kernel<<<blocks, threads, 0, stream>>>(d_x, a);
        CHECK(cudaEventRecord(ev1, stream));
        CHECK(cudaStreamSynchronize(stream));
        double w1 = now_us();
        float span_ms;
        CHECK(cudaEventElapsedTime(&span_ms, ev0, ev1));
        if (r >= 20) {
            wall_total += (w1 - w0);
            span_total += span_ms * 1000.0;
        }
    }
    printf("[eager ] 墙钟 %.3f us/op | GPU 首尾跨度 %.3f us/op（差值 = 提交+同步税）\n",
           wall_total / ROUNDS / N_OPS, span_total / ROUNDS / N_OPS);

    // ---------- 模式 3：CUDA Graph capture + replay ----------
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int k = 0; k < N_OPS; ++k) tiny_kernel<<<blocks, threads, 0, stream>>>(d_x, a);
    CHECK(cudaStreamEndCapture(stream, &graph));
    CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    // 预热
    for (int r = 0; r < 20; ++r) CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    wall_total = 0.0;
    t0 = now_us();
    for (int r = 0; r < ROUNDS; ++r) CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    wall_total = now_us() - t0;
    printf("[graph ] 墙钟 %.3f us/op（整图 %d op 一次提交）\n",
           wall_total / ROUNDS / N_OPS, N_OPS);
    CHECK(cudaGraphExecDestroy(exec));
    CHECK(cudaGraphDestroy(graph));

    // ---------- 模式 4：fused（单 kernel 内联整条依赖链） ----------
    for (int r = 0; r < 20; ++r) fused_kernel<<<blocks, threads, 0, stream>>>(d_x, a);
    CHECK(cudaStreamSynchronize(stream));
    wall_total = 0.0;
    t0 = now_us();
    for (int r = 0; r < ROUNDS; ++r)
        fused_kernel<<<blocks, threads, 0, stream>>>(d_x, a);
    CHECK(cudaStreamSynchronize(stream));
    wall_total = now_us() - t0;
    printf("[fused ] 墙钟 %.3f us/op（%d 步依赖链折进 1 个 kernel）\n",
           wall_total / ROUNDS / N_OPS, N_OPS);

    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaEventDestroy(ev1));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(d_x));
    return 0;
}
