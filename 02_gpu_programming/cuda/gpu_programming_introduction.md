# GPU Architecture and Programming — An Introduction

作者：Najeeb Khan，[原文链接](https://medium.com/@najeebkan/gpu-architecture-and-programming-an-introduction-561bfcb51f54)

Explore Kernel Grids, Blocks, Warps, and Threads to Accelerate Your Code

探索内核网格、块、Warps 和线程以加速我们的代码

![gpu execution hierarchical](../../img/gpu_threads.jpeg)

A GPU executes code in a hierarchical fashion with the kernel as an independent unit of compute. The kernel is executed using independent thread blocks and each thread block consist of subgroups that work in lockstep called warps or wavefronts.

GPU 以分层方式执行代码，内核是独立的计算单元。内核使用独立的线程块执行，每个线程块由同步工作的子组（称为 warp 或 wavefront）组成。

A general purpose graphics processing unit (GPU) provide parallel cores designed to process data simultaneously. Similar to the single instruction multiple data (SIMD) in CPUs, GPUs use several threads to execute instructions in parallel, the paradigm is known as single-instruction multiple threads (SIMT). Modern GPUs can theoretically run one instruction across thousands of data points in a single cycle. However, practical applications often require data exchange between operations, which can consume hundreds of clock cycles. To address this, GPUs use a hierarchical structure to manage communication latency.

通用图形处理单元 (GPU) 提供并行核心，用于同时处理数据。与 CPU 中的单指令多数据 (SIMD) 类似，GPU 使用多个线程并行执行指令，这种模式称为单指令多线程 (SIMT)。现代 GPU 理论上可以在一个周期内跨数千个数据点运行一条指令。然而，实际应用通常需要在操作之间进行数据交换，这可能会消耗数百个时钟周期。为了解决这个问题，GPU 使用分层结构来管理通信延迟。

# Architecture（架构）

In this section, we explore the architecture of a typical GPU. While each GPU generation introduces unique optimizations, we focus on the core concepts that are common across most GPUs.

在本节中，我们将探索典型 GPU 的架构。虽然每一代 GPU 都会引入独特的优化，但我们重点关注大多数 GPU 中常见的核心概念。

## Warps

At the core level, each **_thread_** operates on individual scalar values with private registers. While running thousands of threads simultaneously is impractical, individual threads are not efficient on their own either. Instead, threads are organized into small groups called **_warps or wavefronts_**, typically consisting of 32 threads. Each warp executes a single instruction across 32 data points. For example, in matrix multiplication, a warp might process a row and column from two matrices, performing multiplication and accumulation to generate results as shown in the figure below.

在核心层面，每个 **_thread_** 使用私有寄存器对单个标量值进行操作。虽然同时运行数千个线程是不切实际的，但单个线程本身也效率不高。相反，线程被组织成称为 **_warps 或 wavefronts_** 的小组，通常由 32 个线程组成。每个 warp 跨 32 个数据点执行单个指令。例如，在矩阵乘法中，warp 可能会处理两个矩阵的行和列，执行乘法和累加以生成结果，如下图所示。

![thread warp](../../img/thread_warp.png)

A four thread warp performing matrix multiplication. The warp first performs independent point-wise multiply operation and then performs accumulate operation using some shared memory.

执行矩阵乘法的四线程 Warp。Warp 首先执行独立的逐点乘法运算，然后使用一些共享内存执行累积运算。

## Thread Blocks（线程块）

When operations exceed the warp size of 32 threads, GPUs use tiling to manage larger dimensions. This involves dividing the input into chunks or tiles that fit the warp size, processing these chunks, and then combining the results from all warps. To accumulate partial results, a placeholder is needed, which is where **_thread blocks_** come in. A thread block groups multiple warps, allowing them to share memory and synchronize their execution, as illustrated in the figure below.

当操作超过 32 个线程的 Warp 大小时，GPU 会使用平铺来管理更大的维度。这涉及将输入分成适合 Warp 大小的块或平铺，处理这些块，然后合并所有 Warp 的结果。要累积部分结果，需要一个占位符，这就是 **_thread block_** 的作用所在。线程块将多个 Warp 分组，使它们能够共享内存并同步执行，如下图所示。

![thread block](../../img/thread_block.png)

A thread block comprising of two warps computing matrix multiplication. All the warps in a thread block need to complete execution in order to compute the final output.

一个线程块由两个计算矩阵乘法的 warp 组成。线程块中的所有 warp 都需要完成执行才能计算出最终的输出。

## Grid

The hierarchy from warps to blocks is repeated one more level: if the matrix is larger than what a single thread block can handle, we use a **_grid_** of thread blocks that share global memory. The grid enables the GPU to process large datasets by distributing the workload across multiple thread blocks.

从 warp 到 block 的层次结构又重复了一层：如果矩阵大于单个线程块可以处理的范围，我们将使用共享全局内存的线程块 **_grid_**。网格使 GPU 能够通过将工作负载分布在多个线程块上来处理大型数据集。

All GPU programs, known as **_kernels_**, are executed within this grid structure. When you launch a kernel, you specify both the grid size (the number of thread blocks) and the block size (the number of threads per block). This hierarchical approach ensures efficient computation and data management, allowing the GPU to handle extensive and complex tasks effectively.

所有 GPU 程序（称为 **_kernels_**）都在此网格结构中执行。启动内核时，我们可以指定网格大小（线程块数）和块大小（每个块的线程数）。这种分层方法可确保高效的计算和数据管理，从而使 GPU 能够有效地处理大量复杂的任务。

## Memory Hierarchy（内存层次结构）

Following the structure of the computations, memory is organized into a hierarchy starting from the small and fast registers with ultra low latency and a few kilobytes in size. Registers are private to threads. Next, warps within a thread block share state using **_shared memory_** comprising several hundred kilobytes. Finally, global memory is accessible across the device and provides large capacity on the order of tens of gigabytes with high throughput approaching a terabyte per second. Global memory has higher latency and thus caching is used to reduce latency. The figure below shows the relative scope of each memory type.

根据计算结构，内存被组织成一个层次结构，从具有超低延迟和几千字节大小的小型快速寄存器开始。寄存器是线程私有的。接下来，线程块内的 warp 使用包含数百千字节的 **_shared memory_** 共享状态。最后，全局内存可在整个设备中访问，并提供数十 GB 级的大容量，高吞吐量接近每秒 1 TB。全局内存具有更高的延迟，因此使用缓存来减少延迟。下图显示了每种内存类型的相对范围。

![Memory Hierarchy](../../img/memory_hierarchy.png)

# Programming（编程）

Programming GPUs are supported by dedicated software libraries in C/C++ depending on the make of the GPU: NVIDIA GPUs can be programmed using Compute Unified Device Architecture (CUDA) interface whereas AMD GPUs offer a similar SDK known as HIP.

根据 GPU 的品牌，编程 GPU 由 C/C++ 中的专用软件库支持：NVIDIA GPU 可以使用计算统一设备架构 (CUDA) 接口进行编程，而 AMD GPU 提供类似的 SDK，称为 HIP。

In this section we will briefly show how to run a hello world program on multiple threads using CUDA and how to multiply two matrices.

在本节中，我们将简要展示如何使用 CUDA 在多个线程上运行 hello world 程序以及如何将两个矩阵相乘。

## Hello World!

The entry point of a GPU program is called a kernel. The global ID of a thread can be calculated using three compiler intrinsics — _blockIdx, blockDim, and threadIdx,_ representing the id of the block, the total number of threads in a block, and the thread id within the thread block, respectively. A kernel is defined by the _\_\_global\_\__ qualifier as shown in the listing below. To launch a kernel the _<<<numBlocks, blockSize>>>_ is used. The kernel is executed asynchronously, i.e., the host code will continue to run right after making the kernel call. To sync memory between the host and the GPU device the _cudaDeviceSynchronize_ function is called, which blocks the execution on the host until the kernel finishes its work.

GPU 程序的入口点称为内核。可以使用三个编译器内部函数（_blockIdx、blockDim 和 threadIdx_）计算线程的全局 ID，它们分别表示块的 ID、块中的线程总数以及线程块内的线程 ID。内核由 _\_\_global\_\__ 限定符定义，如下面的列表所示。要启动内核，请使用 _<<<numBlocks, blockSize>>>_。内核是异步执行的，即主机代码将在进行内核调用后继续运行。要同步主机和 GPU 设备之间的内存，请调用 _cudaDeviceSynchronize_ 函数，该函数会阻止主机上的执行，直到内核完成其工作。

```cpp
#include <cuda_runtime.h>  
#include <iostream>  
  
__global__ void helloFromGPU() {  
    printf("Hello World from Thread %d, Block %d, BlockDim %d\\n",   
            threadIdx.x, blockIdx.x, blockDim.x);  
}  
  
int main() {  
    // Launch the kernel with 2 blocks of 4 threads each  
    helloFromGPU<<<2, 4>>>();  
    cudaDeviceSynchronize();  // Wait for the GPU to finish  
    return 0;  
}
```

The above code can be compiled using the NVIDIA compiler and run as follows:

上述代码可以使用NVIDIA编译器进行编译并运行，如下所示：

```bash

> nvcc hello_gpu.cu -o hello_gpu  
> ./hello_gpu  
Hello World from Thread 0, Block 0, BlockDim 4  
Hello World from Thread 1, Block 0, BlockDim 4  
Hello World from Thread 2, Block 0, BlockDim 4  
Hello World from Thread 3, Block 0, BlockDim 4  
Hello World from Thread 0, Block 1, BlockDim 4  
Hello World from Thread 1, Block 1, BlockDim 4  
Hello World from Thread 2, Block 1, BlockDim 4  
Hello World from Thread 3, Block 1, BlockDim 4
```

## Matrix Multiplication（矩阵乘法）

Now that we know the basic structure of a CUDA program, let’s look at a more involved example of matrix multiplication. The CUDA kernel for matrix multiplication is given in the listing below. CUDA provides block IDs and thread IDs in three dimensions. In our case, since we’re dealing with matrices, we use only two dimensions: x and y for the row and column indices.

现在我们已经了解了 CUDA 程序的基本结构，让我们来看一个更复杂的矩阵乘法示例。下面的列表给出了矩阵乘法的 CUDA 内核。CUDA 以三维形式提供块 ID 和线程 ID。在我们的例子中，由于我们处理的是矩阵，因此我们仅使用两个维度：x 和 y 分别表示行和列索引。

The kernel calculates the global row and column indices of each thread by combining the block index and thread index. Each thread then performs the dot product of the corresponding row from matrix A and the column from matrix B, storing the result in matrix C. This approach ensures that each element of the output matrix is computed in parallel, leveraging the GPU’s ability to handle many threads simultaneously.

内核通过结合块索引和线程索引来计算每个线程的全局行和列索引。然后，每个线程对矩阵 A 中相应的行和矩阵 B 中的列进行点积，并将结果存储在矩阵 C 中。这种方法确保输出矩阵的每个元素都是并行计算的，从而充分利用 GPU 同时处理多个线程的能力。

```cpp
__global__ void matrixMul(const float* A, const float* B, float* C, int n) {  
    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  
  
    if (row < n && col < n) {  
        float value = 0.0f;  
        for (int k = 0; k < n; ++k) {  
            value += A[row * n + k] * B[k * n + col];  
        }  
        C[row * n + col] = value;  
    }  
}
```

While this example provides a straightforward implementation of matrix multiplication, it is not optimized for performance. In real-world applications, achieving efficient computation requires careful consideration of memory access patterns and cache utilization. Techniques such as tiling and shared memory usage can significantly enhance performance by reducing memory access latency and improving data locality. Proper cache planning and optimization strategies are essential for scaling these algorithms to handle larger datasets and more complex computations efficiently.

虽然此示例提供了矩阵乘法的直接实现，但它并未针对性能进行优化。在实际应用中，实现高效计算需要仔细考虑内存访问模式和缓存利用率。平铺和共享内存使用等技术可以通过减少内存访问延迟和改善数据局部性来显著提高性能。适当的缓存规划和优化策略对于扩展这些算法以有效处理更大的数据集和更复杂的计算至关重要。