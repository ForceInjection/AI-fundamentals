#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CS(c) do { cudaError_t r = c; if(r != cudaSuccess) { \
    printf("CUDA Err at %d: %s\n", __LINE__, cudaGetErrorString(r)); exit(1); } } while(0)
#define N (128 * 1024 * 1024)
#define IT 10

float *d_a, *d_b;
double run(const char* name, void (*fn)()) {
    for (int i = 0; i < 3; i++) fn();
    cudaEvent_t start, stop; float ms;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    CS(cudaEventRecord(start, 0));
    for (int i = 0; i < IT; i++) fn();
    CS(cudaEventRecord(stop, 0));
    CS(cudaEventSynchronize(stop));
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    double bw = (double)N * IT / (ms / 1000.0) / (1024*1024*1024);
    printf("  %-30s  %8.2f ms  %8.2f GB/s\n", name, ms, bw);
    return bw;
}

/* ---- Method 1: cudaMemcpyPeer ---- */
void m1_peer() {
    cudaMemcpyPeer(d_b, 1, d_a, 0, N);
    cudaMemcpyPeer(d_a, 0, d_b, 1, N);
}

/* ---- Method 2: cudaMemcpy D2D (after cudaDeviceEnablePeerAccess) ---- */
void m2_d2d() {
    cudaMemcpy(d_b, d_a, N, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_a, d_b, N, cudaMemcpyDeviceToDevice);
}

/* ---- Method 3: CPU relay (always works, no P2P needed) ---- */
float *h_relay, *dr_a, *dr_b;
void m3_cpu() {
    cudaMemcpy(h_relay, dr_a, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(dr_b, h_relay, N, cudaMemcpyHostToDevice);
    cudaMemcpy(h_relay, dr_b, N, cudaMemcpyDeviceToHost);
    cudaMemcpy(dr_a, h_relay, N, cudaMemcpyHostToDevice);
}

/* ---- Method 4: Zero-Copy mapped host memory ---- */
float *h_zc, *d_zc;
float *d_zc_gpu1;
void m4_zc() {
    cudaSetDevice(0);
    cudaMemset(d_zc, 42, N);
    CS(cudaDeviceSynchronize());
    cudaSetDevice(1);
    cudaMemcpy(d_zc_gpu1, h_zc, N, cudaMemcpyHostToDevice);
    CS(cudaDeviceSynchronize());
}

/* ---- Method 5: Unified Memory with explicit prefetch ---- */
float *d_um, *d_um2;
void m5_um() {
    cudaSetDevice(0);
    {cudaMemLocation loc = {cudaMemLocationTypeDevice, 0};
     cudaMemPrefetchAsync(d_um, N, loc, 0);}
    cudaMemsetAsync(d_um, 0, N, 0);
    CS(cudaDeviceSynchronize());
    {cudaMemLocation loc2 = {cudaMemLocationTypeDevice, 1};
     cudaMemPrefetchAsync(d_um, N, loc2, 0);}
    cudaSetDevice(1);
    cudaMemcpy(d_um2, d_um, N, cudaMemcpyDeviceToDevice);
    CS(cudaDeviceSynchronize());
}

int main() {
    int devCount, canPeer;
    cudaGetDeviceCount(&devCount);
    if (devCount < 2) { printf("Need >=2 GPUs\n"); return 1; }

    for (int i = 0; i < 2; i++) {
        cudaDeviceProp p; cudaGetDeviceProperties(&p, i);
        printf("Device %d: %s\n", i, p.name);
    }
    CS(cudaDeviceCanAccessPeer(&canPeer, 0, 1));
    printf("P2P available: %s\n\n", canPeer ? "YES" : "NO");
    printf("%-30s  %8s  %8s\n", "Method", "Time", "Bandwidth");
    printf("%-30s  %8s  %8s\n", "------", "----", "--------");

    double bw2 = 0, bw3 = 0, bw4 = 0, bw5 = 0;

    /* Methods 1 & 2: requires P2P. Must cudaSetDevice before enablePeerAccess! */
    if (canPeer) {
        CS(cudaSetDevice(0)); CS(cudaMalloc(&d_a, N));
        CS(cudaSetDevice(1)); CS(cudaMalloc(&d_b, N));
        CS(cudaSetDevice(0)); CS(cudaDeviceEnablePeerAccess(1, 0));
        CS(cudaSetDevice(1)); CS(cudaDeviceEnablePeerAccess(0, 0));

        CS(cudaSetDevice(0));
        run("1. cudaMemcpyPeer (NVLink)", m1_peer);
        bw2 = run("2. cudaMemcpy D2D (P2P on)",  m2_d2d);

        CS(cudaSetDevice(0)); CS(cudaFree(d_a));
        CS(cudaSetDevice(1)); CS(cudaFree(d_b));
    }

    /* Method 3: CPU relay */
    CS(cudaSetDevice(0)); CS(cudaMalloc(&dr_a, N));
    CS(cudaSetDevice(1)); CS(cudaMalloc(&dr_b, N));
    CS(cudaMallocHost(&h_relay, N));
    bw3 = run("3. CPU relay (G->CPU->G)", m3_cpu);
    CS(cudaFreeHost(h_relay));
    CS(cudaSetDevice(0)); CS(cudaFree(dr_a));
    CS(cudaSetDevice(1)); CS(cudaFree(dr_b));

    /* Method 4: Zero-Copy */
    CS(cudaSetDevice(0));
    CS(cudaHostAlloc(&h_zc, N, cudaHostAllocPortable | cudaHostAllocMapped));
    CS(cudaHostGetDevicePointer(&d_zc, h_zc, 0));
    CS(cudaSetDevice(1)); CS(cudaMalloc(&d_zc_gpu1, N));
    bw4 = run("4. Zero-Copy (mapped host)", m4_zc);
    CS(cudaSetDevice(0)); CS(cudaFreeHost(h_zc));
    CS(cudaSetDevice(1)); CS(cudaFree(d_zc_gpu1));

    /* Method 5: Unified Memory */
    CS(cudaSetDevice(0)); CS(cudaMallocManaged(&d_um, N));
    CS(cudaSetDevice(1)); CS(cudaMalloc(&d_um2, N));
    bw5 = run("5. Unified Memory (prefetch)", m5_um);
    CS(cudaSetDevice(0)); CS(cudaFree(d_um));
    CS(cudaSetDevice(1)); CS(cudaFree(d_um2));

    printf("\n=== Summary (128 MB) ===\n");
    if (canPeer) {
        printf("  P2P / CPU-relay:    %.0fx\n", bw2 / bw3);
        printf("  P2P / Zero-Copy:    %.0fx\n", bw2 / bw4);
        printf("  P2P / Unified Mem:  %.0fx\n", bw2 / bw5);
    }
    return 0;
}
