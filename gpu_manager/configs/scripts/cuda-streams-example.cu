#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define STREAMS 4

__global__ void vectorAdd(int *a, int *b, int *c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);
    
    // 分配主机内存
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 创建 CUDA 流
    cudaStream_t streams[STREAMS];
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    int chunk_size = N / STREAMS;
    
    // 使用多个流并行执行
    for (int i = 0; i < STREAMS; i++) {
        int offset = i * chunk_size;
        
        // 异步内存拷贝
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], chunk_size * sizeof(int), 
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], chunk_size * sizeof(int), 
                       cudaMemcpyHostToDevice, streams[i]);
        
        // 启动内核
        dim3 blockSize(256);
        dim3 gridSize((chunk_size + blockSize.x - 1) / blockSize.x);
        vectorAdd<<<gridSize, blockSize, 0, streams[i]>>>(&d_a[offset], &d_b[offset], 
                                                         &d_c[offset], chunk_size);
        
        // 异步拷贝回结果
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], chunk_size * sizeof(int), 
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // 同步所有流
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // 验证结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: expected %d, got %d\n", 
                   i, h_a[i] + h_b[i], h_c[i]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("CUDA 流并行计算成功完成！\n");
    }
    
    // 清理资源
    for (int i = 0; i < STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}