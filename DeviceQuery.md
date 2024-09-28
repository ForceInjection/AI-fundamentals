# 查询 GPU 卡详细参数

## CUDA 示例代码库

Nvidia 官方提供了一个面向 CUDA 开发人员的示例代码库，演示了 CUDA 工具包中的功能，包括以下内容：

- **简介**：面向初学者的基本 CUDA 示例，说明使用 CUDA 和 CUDA 运行时 API 的关键概念(初学的同学先看这个)；
- **实用程序**：实用程序示例演示如何查询设备功能以及测量 GPU/CPU 带宽(非常有用，用于学习和理解 GPU 卡的详细参数)；
- **概念和技术**：展示 CUDA 相关概念和常见问题解决技术的示例；
- **CUDA 功能**：展示 CUDA 功能（协作组、CUDA 动态并行、CUDA 图等）的示例;
- **CUDA 库**：演示如何使用 CUDA 平台库 (NPP、NVJPEG、NVGRAPH cuBLAS、cuFFT、cuSPARSE、cuSOLVER 和 cuRAND) 的示例。
- **特定领域示例**：特定于领域的示例（图形、金融、图像处理）。
- **性能**：展示性能优化的示例。
- **libNVVM**：演示如何使用 libNVVM 和 NVVM IR 的示例。

[代码库地址: (https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples).

今天我们就来使用示例代码库提供的 `deviceQueryDrv` 示例代码来查询 GPU 卡的详细信息。

## 使用 `deviceQuery` 查询 GPU 卡的详细信息

### `deviceQuery` vs `deviceQueryDrv`

- `deviceQuery`: 此示例枚举系统中存在的 CUDA 设备的属性。
`deviceQuery.cpp`:

``` C++
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string>

int *pArgc = NULL;
char **pArgv = NULL;

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>
```

- `deviceQueryDrv`: 此示例使用 CUDA 驱动程序 API 调用枚举存在的 CUDA 设备的属性
`deviceQueryDrv.cpp`:

``` C++
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda.h>
#include <helper_cuda_drvapi.h>
```
我们可以看到 `deviceQueryDrv.cpp` 用到了 `helper_cuda_drvapi.h`.

### 编译代码

```bash
ls -l
total 80
-rw-r--r-- 1 root root 15028 Sep 27 10:00 Makefile
-rw-r--r-- 1 root root  2194 Apr 25 16:44 NsightEclipse.xml
-rw-r--r-- 1 root root  3563 Sep 27 10:00 README.md
-rw-r--r-- 1 root root 16520 Apr 25 16:44 deviceQueryDrv.cpp
-rw-r--r-- 1 root root   857 Apr 25 16:44 deviceQueryDrv_vs2017.sln
-rw-r--r-- 1 root root  5054 Sep 27 10:00 deviceQueryDrv_vs2017.vcxproj
-rw-r--r-- 1 root root   857 Apr 25 16:44 deviceQueryDrv_vs2019.sln
-rw-r--r-- 1 root root  4639 Sep 27 10:00 deviceQueryDrv_vs2019.vcxproj
-rw-r--r-- 1 root root   857 Apr 25 16:44 deviceQueryDrv_vs2022.sln
-rw-r--r-- 1 root root  4639 Sep 27 10:00 deviceQueryDrv_vs2022.vcxproj

make

/usr/local/cuda/bin/nvcc -ccbin g++ -I../../../Common  -m64    --threads 0 --std=c++11 -gencode arch=compute_50,code=compute_50 -o deviceQueryDrv.o -c deviceQueryDrv.cpp
/usr/local/cuda/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_50,code=compute_50 -o deviceQueryDrv deviceQueryDrv.o  -L/usr/local/cuda/lib64/stubs -lcuda
mkdir -p ../../../bin/x86_64/linux/release
cp deviceQueryDrv ../../../bin/x86_64/linux/release

# 本地目录也会有 deviceQueryDrv 可执行文件

ll
total 2484
-rw-r--r-- 1 root root   15028 Sep 27 10:00 Makefile
-rw-r--r-- 1 root root    2194 Apr 25 16:44 NsightEclipse.xml
-rw-r--r-- 1 root root    3563 Sep 27 10:00 README.md
-rwxr-xr-x 1 root root 2441040 Sep 28 07:46 deviceQueryDrv # 可执行文件
-rw-r--r-- 1 root root   16520 Apr 25 16:44 deviceQueryDrv.cpp
-rw-r--r-- 1 root root   16664 Sep 28 07:46 deviceQueryDrv.o
-rw-r--r-- 1 root root     857 Apr 25 16:44 deviceQueryDrv_vs2017.sln
-rw-r--r-- 1 root root    5054 Sep 27 10:00 deviceQueryDrv_vs2017.vcxproj
-rw-r--r-- 1 root root     857 Apr 25 16:44 deviceQueryDrv_vs2019.sln
-rw-r--r-- 1 root root    4639 Sep 27 10:00 deviceQueryDrv_vs2019.vcxproj
-rw-r--r-- 1 root root     857 Apr 25 16:44 deviceQueryDrv_vs2022.sln
-rw-r--r-- 1 root root    4639 Sep 27 10:00 deviceQueryDrv_vs2022.vcxproj

```

### 查询 GPU 卡详细信息

```bash
./deviceQueryDrv
./deviceQueryDrv Starting...

CUDA Device Query (Driver API) statically linked version
Detected 2 CUDA Capable device(s)

Device 0: "Tesla T4"
  CUDA Driver Version:                           12.2
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 14931 MBytes (15655829504 bytes)
  (40) Multiprocessors, ( 64) CUDA Cores/MP:     2560 CUDA Cores
  GPU Max Clock rate:                            1590 MHz (1.59 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Max Texture Dimension Sizes                    1D=(131072) 2D=(131072, 65536) 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size (x,y,z):    (2147483647, 65535, 65535)
  Texture alignment:                             512 bytes
  Maximum memory pitch:                          2147483647 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Concurrent kernel execution:                   Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 59 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

Device 1: "Tesla T4"
  CUDA Driver Version:                           12.2
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 14931 MBytes (15655829504 bytes)
  (40) Multiprocessors, ( 64) CUDA Cores/MP:     2560 CUDA Cores
  GPU Max Clock rate:                            1590 MHz (1.59 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Max Texture Dimension Sizes                    1D=(131072) 2D=(131072, 65536) 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size (x,y,z):    (2147483647, 65535, 65535)
  Texture alignment:                             512 bytes
  Maximum memory pitch:                          2147483647 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Concurrent kernel execution:                   Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 175 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
> Peer-to-Peer (P2P) access from Tesla T4 (GPU0) -> Tesla T4 (GPU1) : Yes
> Peer-to-Peer (P2P) access from Tesla T4 (GPU1) -> Tesla T4 (GPU0) : Yes
Result = PASS

```

以下是 Nvidia Tesla T4 GPU 参数的详细解析：

1. **CUDA Driver Version: 12.2**
   - **CUDA 驱动版本**：表示该 GPU 所使用的 CUDA 驱动程序的版本号。CUDA 是 Nvidia 提供的一个用于开发 GPU 计算应用程序的平台，驱动版本必须与软件版本兼容。

2. **CUDA Capability Major/Minor version number: 7.5**
   - **CUDA 计算能力版本**：CUDA 计算能力决定了 GPU 支持的 CUDA 功能和特性。7.5 表示这张 Tesla T4 支持较新的特性，包括高级的并行处理和硬件优化。

3. **Total amount of global memory: 14931 MBytes (15655829504 bytes)**
   - **总全局内存**：GPU 上的总显存大小，单位为字节和 MB。显存主要用于存储数据集、计算过程中的中间结果等，对于机器学习、深度学习等任务至关重要。Tesla T4 有约 15 GB 的显存。

4. **Multiprocessors, CUDA Cores/MP: 2560 CUDA Cores (40 Multiprocessors, 64 CUDA Cores/MP)**
   - **多处理器和 CUDA 核心数**：Tesla T4 有 40 个流式多处理器（SM），每个 SM 包含 64 个 CUDA 核心，总共 2560 个 CUDA 核心。CUDA 核心是 GPU 执行并行计算的基础单元。

5. **GPU Max Clock rate: 1590 MHz (1.59 GHz)**
   - **最大 GPU 时钟频率**：表示 GPU 核心的运行频率，单位为 MHz。1590 MHz 说明 GPU 在满负载下的最大运行速度。

6. **Memory Clock rate: 5001 MHz**
   - **显存时钟频率**：指显存运行时的时钟频率，单位为 MHz。显存时钟频率决定了显存的带宽和数据传输速度。

7. **Memory Bus Width: 256-bit**
   - **显存总线宽度**：指数据从 GPU 内核到显存之间的通道宽度。256-bit 表示每次传输时可以同时传输 256 位数据。

8. **L2 Cache Size: 4194304 bytes**
   - **二级缓存大小**：GPU 内的二级缓存大小，单位为字节。二级缓存用于加速内存访问，减少频繁的显存读写操作。

9. **Max Texture Dimension Sizes**
   - **最大纹理维度大小**：
     - 1D 最大纹理尺寸：131072
     - 2D 最大纹理尺寸：131072 x 65536
     - 3D 最大纹理尺寸：16384 x 16384 x 16384
     纹理用于 GPU 中的图像处理和渲染。最大纹理尺寸决定了 GPU 能够处理的纹理数据的大小。

10. **Maximum Layered 1D/2D Texture Size, (num) layers**
   - **最大分层 1D/2D 纹理尺寸**：
     - 1D 最大为 32768，每个分层最多支持 2048 层
     - 2D 最大为 32768 x 32768，每个分层最多支持 2048 层
     这是分层纹理的最大支持值，通常用于需要在多个层之间处理不同纹理的应用中。

11. **Total amount of constant memory: 65536 bytes**
   - **常量内存大小**：常量内存是专门存储只读数据的显存部分，大小为 64 KB。常量内存的读写性能比全局内存更高，但其容量有限。

12. **Total amount of shared memory per block: 49152 bytes**
   - **每个块的共享内存大小**：每个线程块（block）可以访问的共享内存大小为 48 KB。共享内存是线程块内部用于数据交换的高速内存。

13. **Total number of registers available per block: 65536**
   - **每个块可用的寄存器数量**：每个线程块可以使用的寄存器数为 65536。寄存器是 GPU 最快的存储单元，主要用于存储临时变量。

14. **Warp size: 32**
   - **Warp 大小**：Warp 是 CUDA 中的基本调度单元，包含 32 个并行执行的线程。每次调度时，GPU 以 Warp 为单位来执行指令。

15. **Maximum number of threads per multiprocessor: 1024**
   - **每个多处理器的最大线程数**：每个 SM 可以同时执行的最大线程数为 1024。这决定了每个 SM 上并行工作的最大线程量。

16. **Maximum number of threads per block: 1024**
   - **每个块的最大线程数**：一个线程块中允许的最大线程数为 1024。线程块是 CUDA 中并行计算的基本单位。

17. **Max dimension size of a thread block (x,y,z): (1024, 1024, 64)**
   - **线程块的最大尺寸 (x, y, z)**：每个线程块在 x 轴和 y 轴上的最大维度是 1024，在 z 轴上最大为 64。这表示每个线程块的尺寸限制。

18. **Max dimension size of a grid size (x,y,z): (2147483647, 65535, 65535)**
   - **网格尺寸的最大值 (x, y, z)**：每个维度上网格大小的最大值，表示一个网格可以包含的最大线程块数。x 维度支持的最大值非常大，达到了 2147483647。

19. **Texture alignment: 512 bytes**
   - **纹理对齐**：纹理内存的对齐单位是 512 字节，这对纹理内存的读取性能有重要影响。

20. **Maximum memory pitch: 2147483647 bytes**
   - **最大内存步幅**：在 2D 内存复制操作中，两个连续行之间的最大字节距离为 2147483647。

21. **Concurrent copy and kernel execution: Yes with 3 copy engine(s)**
   - **并发拷贝与核函数执行**：Tesla T4 支持同时进行数据拷贝和核函数执行，且具有 3 个独立的拷贝引擎，提升了同时执行多个任务时的数据吞吐量。

22. **Run time limit on kernels: No**
   - **核函数运行时间限制**：核函数在 Tesla T4 上没有运行时间限制，这在运行长时间任务时非常有用。

23. **Integrated GPU sharing Host Memory: No**
   - **集成 GPU 共享主机内存**：Tesla T4 不支持集成显卡共享主机内存，意味着它不依赖主机的 RAM。

24. **Support host page-locked memory mapping: Yes**
   - **支持主机页锁定内存映射**：Tesla T4 支持将主机的页锁定内存直接映射到 GPU，从而提高主机与 GPU 之间的数据传输效率。

25. **Concurrent kernel execution: Yes**
   - **并发核函数执行**：Tesla T4 支持同时执行多个核函数，提升了任务的并行执行能力。

26. **Alignment requirement for Surfaces: Yes**
   - **表面对齐要求**：在 CUDA 中，表面内存访问要求数据对齐。

27. **Device has ECC support: Enabled**
   - **设备支持 ECC**：纠错码（ECC）是用于检测和修复内存中的错误，Tesla T4 开启了 ECC 支持，确保在计算任务中数据的准确性和稳定性。

28. **Device supports Unified Addressing (UVA): Yes**
   - **支持统一寻址 (UVA)**：支持统一寻址模式，GPU 和 CPU 共享统一的地址空间，方便数据在 CPU 和 GPU 之间传输。

29. **Device supports Managed Memory: Yes**
   - **支持托管内存**：托管内存允许开发者不用关心 CPU 和 GPU 之间的数据传输，系统会自动在两者之间管理内存。

30. **Device supports Compute Preemption: Yes**
   - **支持计算抢占**：在必要时，GPU 可以暂停正在执行的任务并切换到更高优先级的任务。

31. **Supports Cooperative Kernel Launch: Yes**
   - **支持协同核函数启动**：允许多个核函数在同一个网格中协同工作，从而加速任务执行。

32. **Supports MultiDevice Co-op Kernel Launch: Yes**
   - **支持多设备协同核函数启动**：Tesla T4 支持在多个 GPU 设备上协同启动核函数，即不同的 GPU 可以协同工作，处理同一个大型任务。这种功能对于需要大规模并行计算的任务非常重要。

33. **Device PCI Domain ID / Bus ID / location ID: 0 / 175 / 0**
   - **设备 PCI 域 ID / 总线 ID / 位置 ID**：这是 GPU 在主板上的物理位置，通过 PCI 总线进行连接。PCI 域 ID 为 0，总线 ID 为 175，位置 ID 为 0。这个信息有助于开发人员在多 GPU 系统中确定每个 GPU 的物理位置。

34. **Compute Mode: Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)**
   - **计算模式：默认**：在默认计算模式下，多个主机线程可以同时调用 `cudaSetDevice()` 来使用这块 GPU。这意味着可以多个线程并行运行在同一 GPU 上，适合并行任务场景。

35. **Peer-to-Peer (P2P) access from Tesla T4 (GPU0) -> Tesla T4 (GPU1): Yes**
   - **Tesla T4 (GPU0) 到 Tesla T4 (GPU1) 的 P2P 访问：是**：表示两块 Tesla T4 GPU 之间支持点对点（Peer-to-Peer）访问，允许 GPU 直接交换数据，而无需通过主机内存。这提高了多 GPU 系统中数据传输的效率。

36. **Peer-to-Peer (P2P) access from Tesla T4 (GPU1) -> Tesla T4 (GPU0): Yes**
   - **Tesla T4 (GPU1) 到 Tesla T4 (GPU0) 的 P2P 访问：是**：与上述参数类似，这说明 Tesla T4 在多 GPU 系统中具有双向的 P2P 访问功能，允许 GPU 之间高效地相互传输数据。

37. **Result = PASS**
   - **结果：通过**：该参数表明 Tesla T4 的所有功能都经过了验证，并且其硬件和软件环境都正常工作，没有检测到错误或不兼容问题。

通过这些参数，Tesla T4 作为 Nvidia 专为推理、机器学习和加速计算设计的 GPU，不仅具有强大的计算性能，还支持多种先进的并行计算和内存管理功能。这些功能（如 Peer-to-Peer 访问、托管内存、ECC 内存和协同核函数启动等）确保了它在需要高效、稳定计算的场景下能够发挥出色的性能。