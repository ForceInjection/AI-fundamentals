# GPU Memory (GPU 内存)

**原文：[Cornell University -> Cornell Virtual Workshop -> Understanding GPU Architecture -> GPU Memory](https://cvw.cac.cornell.edu/gpu-architecture/gpu-memory/index)**

Just like a CPU, the GPU relies on a memory hierarchy—from RAM, through cache levels—to ensure that its processing engines are kept supplied with the data they need to do useful work. And just like the cores in a CPU, the streaming multiprocessors (SMs) in a GPU ultimately require the data to be in registers to be available for computations. This topic looks at the sizes and properties of the different elements of the GPU's memory hierarchy and how they compare to those found in CPUs.

就像 CPU 一样，GPU 依赖于内存层次结构——从 RAM 到各级缓存——以确保其处理引擎能够持续获得数据以进行有用工作。就像 CPU 中的核心一样，GPU 中的流式多处理器（SMs）最终也需要数据位于寄存器中才能用于计算。本主题探讨了 GPU 内存层次结构中不同元素的大小和属性，以及它们与 CPU 中发现的元素的比较。

## Memory Levels（内存级别）

Compared to a CPU core, a streaming multiprocessor (SM) in a GPU has many more registers. For example, an SM in the NVIDIA Tesla V100 has 65536 registers in its register file. Taken as a whole, its register file is larger in terms of its total capacity, too, despite the fact that the SM's 4-byte registers hold just a single float, whereas the vector registers in an Intel AVX-512 core hold 16 floats.

与 CPU 核心相比，GPU 中的流式多处理器（SM）拥有更多的寄存器。例如，NVIDIA Tesla V100 中的 SM 拥有 65536 个寄存器在其寄存器文件中。总体而言，尽管 SM 的 4 字节寄存器仅保存一个浮点数，而 Intel AVX-512 核心的矢量寄存器可以保存 16 个浮点数，但其寄存器文件在总容量上仍然更大。

But CPUs have an advantage in total cache size. Every Intel CPU core comes with L1 and L2 data caches, and the size of these caches plus its share of the shared L3 cache is easily larger than the equivalent caches in a GPU, which has nothing beyond a shared L2 cache. The figure shows the full picture of the memory hierarchy in the Tesla V100. The roles of the different caches and memory types are explained on the [Memory Types](#Memory_Types).

但 CPU 在总缓存大小上有优势。每个 Intel CPU 核心都带有 L1 和 L2 数据缓存，这些缓存的大小加上其共享的 L3 缓存的份额，很容易就比 GPU 中的等效缓存大，GPU 除了共享的 L2 缓存之外没有其他缓存。下图展示了 Tesla V100 的完整内存层次结构。不同缓存和内存类型的作用在“内存类型”中进行了解释。

![GPU memory levels and sizes for the NVIDIA Tesla V100](img/GPUMemLevels.png)

GPU memory levels and sizes for the NVIDIA Tesla V100.
(Based on diagrams by NVIDIA and Citadel at GTC 2018)

Arrows in the above diagram show how the layers of memory are linked together. Depending on where the data start, they may have to hop through several layers of cache to enter the registers of an SM and become accessible to the CUDA cores. Global memory (including areas for local, texture, and constant memory) is by far the largest layer, but it is also furthest from the SMs.

上图中的箭头显示了内存层是如何相互连接的。根据数据的起始位置，它们可能需要通过几个缓存层才能进入 SM 的寄存器并成为 CUDA 核心可访问的数据。全局内存（包括局部、纹理和常量内存区域）是迄今为止最大的层，但它也距离 SM 最远。

Clearly it would be favorable for 4-byte operands to travel together in groups of 32 as they move back and forth between caches and registers and CUDA cores. Why? A 32-wide group is exactly right to supply a warp of 32 threads, all at once. Therefore, it makes perfect sense that the size of the cache line in a GPU is 32 x (4 bytes) = 128 bytes.

显然，对于 4 字节的操作数来说，当它们在缓存和寄存器以及 CUDA 核心之间来回移动时，以 32 组的形式一起移动是有利的。为什么？一个 32 宽的组正好可以一次性供应一个包含 32 个线程的 warp。因此，GPU 中的缓存行大小是 32 x (4 字节) = 128 字节是完全合理的。

Notice that data transfers onto and off of the device are mediated by the L2 cache. In most cases, the incoming data will proceed from the L2 into the large global memory of the device.

请注意，数据传输到设备上和从设备传输出去都是由 L2 缓存介导的。在大多数情况下，传入的数据将从 L2 进入设备的大容量全局内存。

## <span id="Memory_Types">Memory Types(内存类型)</span>

The picture on the preceding page is more complex than it would be for a CPU, because the GPU reserves certain areas of memory for specialized use during rendering. Here, we summarize the roles of each type of GPU memory for doing GPGPU computations.

前一页的图比 CPU 的情况要复杂，因为 GPU 为渲染过程中的专门使用预留了某些内存区域。这里，我们总结了每种类型的 GPU 内存在进行 GPGPU 计算时的作用。

The first list covers the on-chip memory areas that are closest to the CUDA cores. They are part of every SM.

第一个列表涵盖了最接近 CUDA 核心的片上内存区域。它们是每个 SM 的一部分。

* **Register File** - denotes the area of memory that feeds directly into the CUDA cores. Accordingly, it is organized into 32 _banks_, matching the 32 threads in a warp. Think of the register file as a big matrix of 4-byte elements, having many rows and 32 columns. A warp operates on full rows; within a given row, each thread (CUDA core) operates on a different column (bank).
* **L1 Cache** - refers to the usual on-chip storage location providing fast access to data that are recently read from, or written to, main memory (RAM). Additionally, L1 serves as the overflow region when the amount of active data exceeeds what an SM's register file can hold, a condition which is termed "register spilling". In L1, the cache lines and spilled registers are organized into banks, just as in the register file.
* **Shared Memory** - is a memory area that physically resides in the same memory as the L1 cache, but differs from L1 in that all its data may be accessed by any thread in a thread block. This allows threads to communicate and share data with each other. Variables that occupy it must be declared explicitly by an application. The application can also set the dividing line between L1 and shared memory.
* **Constant Caches** - are special caches pertaining to variables declared as read-only constants in global memory. Such variables can be read by any thread in a thread block. The main and best use of these caches is to broadcast a single constant value to all the threads in a warp.

* **寄存器文件** - 表示直接进入 CUDA 核心的内存区域。相应地，它被组织成 32 个银行，与一个 warp 中的 32 个线程相匹配。将寄存器文件视为一个由 4 字节元素组成的大矩阵，有很多行和 32 列。一个 warp 在完整的行上操作；在给定的行中，每个线程（CUDA 核心）在不同的列（银行）上操作。

* **L1 缓存** - 指的是通常的片上存储位置，提供对最近从主内存（RAM）读取或写入的数据的快速访问。此外，当活动数据量超过 SM 的寄存器文件可以容纳的量时，L1 还充当溢出区域，这种情况被称为“寄存器溢出”。在 L1 中，缓存行和溢出的寄存器被组织成银行，就像寄存器文件一样。

* **共享内存** - 是一个物理上位于与 L1 缓存相同的内存中的内存区域，但它与 L1 不同，因为它的所有数据都可以被一个线程块中的任何线程访问。这允许线程相互通信和共享数据。占用它的变量必须由应用程序显式声明。应用程序还可以设置 L1 和共享内存之间的分界线。

* **常量缓存** - 是专门用于全局内存中声明为只读常量的变量的特殊缓存。这些变量可以被线程块中的任何线程读取。这些缓存的主要和最佳用途是将单个常量值广播到 warp 中的所有线程。

The second list pertains to the more distant, larger memory areas that are shared by all the SMs.

第二个列表涉及更遥远、更大的内存区域，这些区域由所有 SM 共享。

* **L2 Cache** - is a further on-chip cache for retaining copies of the data that travel back and forth between the SMs and main memory. Like the L1, the L2 cache is intended to speed up subsequent reloads. But unlike the L1 cache(s), there is just one L2 that is shared by all the SMs. The L2 cache is also situated in the path of data moving on or off the device via PCIe or NVLink.
* **Global Memory** - represents the bulk of the main memory of the device, equivalent to RAM in a CPU-based processor. For performance reasons, the Tesla V100 has special HBM2 high-bandwidth memory, while the Quadro RTX 5000 has fast GDDR6 graphics memory.
* **Local Memory** - corresponds to specially mapped regions of main memory that are assigned to each SM. Whenever "register spilling" overflows the L1 cache on a particular SM, the excess data are further offloaded to L2, then to "local memory". The performance penalty for reloading a spilled register becomes steeper for every memory level that must be traversed in order to retrieve it.
* **Texture and Constant Memory** - are regions of main memory that are treated as read-only by the device. When fetched to an SM, variables with a "texture" or "constant" declaration can be read by any thread in a thread block, much like shared memory. Texture memory is cached in L1, while constant memory is cached in the constant caches.

* **L2 缓存** - 是一个进一步的片上缓存，用于保留在 SM 和主内存之间来回传输的数据的副本。像 L1 一样，L2 缓存旨在加速后续的重新加载。但与 L1 缓存不同，只有一个 L2 被所有 SM 共享。L2 缓存也位于通过 PCIe 或 NVLink 在设备上或下移动的数据的路径上。

* **全局内存** - 代表设备的大部分主内存，相当于基于 CPU 的处理器中的 RAM。出于性能原因，Tesla V100 拥有特殊的 HBM2 高带宽内存，而 Quadro RTX 5000 拥有快速的 GDDR6 图形内存。

* **局部内存** - 对应于分配给每个 SM 的主内存的特定映射区域。每当“寄存器溢出”溢出特定 SM 的 L1 缓存时，超出的数据会被进一步卸载到 L2，然后是“局部内存”。重新加载溢出寄存器的性能惩罚随着必须遍历的每个内存级别而变得更加严重。

* **纹理和常量内存** - 是被视为设备只读的内存区域。当它们被传输到 SM 时，具有“纹理”或“常量”声明的变量可以被线程块中的任何线程读取，就像共享内存一样。纹理内存被缓存在 L1 中，而常量内存被缓存在常量缓存中。

## <span id="comparison_cpu_mem">Comparison to CPU Memory(与 CPU 内存对比)</span>

The organization of memory in a GPU largely resembles a CPU's—but there are significant differences as well. This is particularly true of the total capacities available at each level of the memory hierarchy. As mentioned previously, the GPU is characterized by very large register files, while the CPU is much more cache-heavy and has generally wider data paths.

GPU 的内存组织在很大程度上类似于 CPU 的，但也存在显著差异。这在内存层次结构的每个级别上可用的总容量上尤为明显。如前所述，GPU 的特点是非常大的寄存器文件，而 CPU 更多地依赖缓存，并且通常具有更宽的数据路径。

The table below makes a comparison between the memory hierarchies found in the NVIDIA GPUs in TACC's Longhorn system and the Intel Xeon CPUs in TACC's Frontera and Stampede2 systems. In each row, the entry shown in bold highlights the type of device that has the bigger capacity _per computational unit_ at that level.

下表比较了 TACC 的 Longhorn 系统中的 NVIDIA GPU 和 TACC 的 Frontera 和 Stampede2 系统中的 Intel Xeon CPU 的内存层次结构。在每一行中，加粗的条目突出显示了在该级别上每个计算单元可用的更大容量的设备类型。

Available memory per SM or core at each level in the memory hierarchy of the NVIDIA Tesla V100 vs. Intel Xeon SP processors.

NVIDIA Tesla V100 与 Intel Xeon SP 处理器的每个内存层次结构级别的每个 SM 或核心可用内存。

| **Memory Type** | **NVIDIA Tesla V100<br>- per SM -** | **Intel Cascade Lake SP, Skylake SP<br>- per core -** |
|---|---|---|
| **Register file** | 256 kB | 10.5 kB |
| **L1 cache** | 128 kB (max) | 32 kB |
| **Constant caches** | 64 kB | - N/A - |
| **L2 cache** | 0.075 MB | 1 MB |
| **L3 cache** | - N/A - | 1.375 MB |
| **Global or RAM** | 0.4 GB | >3.4 GB |

Since the table compares what is available to a single GPU SM to the equivalent for a single CPU core, the processor-wide memory levels are treated as though they are distributed evenly among all the SMs or cores. A more precise comparison might break down these computational units even further and look at the sizes of the memory slices that pertain to a GPU CUDA core vs. a CPU vector lane. For those interested, this approach is pursued in the [Appendix](#Appendix).

由于表格比较的是一个 GPU SM 与一个 CPU 核心的等效情况，因此将处理器范围的内存级别视为均匀分布在所有 SM 或核心之间。更精确的比较可能会进一步细分这些计算单元，并查看与 GPU CUDA 核心相关的内存切片的大小与 CPU 矢量通道的大小。对于感兴趣的人，这种方法在附录中进行了探讨。

The memory levels do not match up exactly because in the Xeon processors, every core has a private L2 cache plus a slice of the shared L3, while in the V100 GPU, every SM has special constant caches, but no caches beyond a slice of the shared L2. Generally, though, the table shows that the GPU has the greater concentration at the closest memory levels, while the CPU has an evident size advantage as one moves further out in memory.

内存级别并不完全匹配，因为在 Xeon 处理器中，每个核心都有一个私有的 L2 缓存加上共享的 L3 的一部分，而在 V100 GPU 中，每个 SM 都有特殊的常量缓存，但没有超出共享的 L2 的缓存。通常，表格显示 GPU 在最接近的内存级别上有更大的集中度，而 CPU 在内存向外移动时有明显的尺寸优势。

In considering memory _speed_, the roles of GPUs and CPUs become reversed. Latency and bandwidth are two relevant measures of memory speed. The next table shows that the Intel Xeon CPUs feature superior latencies for the caches that are closest to the cores, while the Tesla V100 has an edge in bandwidth at the more distant, global memory level.

在考虑内存速度时，GPU 和 CPU 的角色发生了逆转。延迟和带宽是衡量内存速度的两个相关指标。下一个表格显示，Intel Xeon CPU 的最近核心的缓存具有更优越的延迟，而 Tesla V100 在更遥远的全局内存级别上具有带宽优势。

Latency and available bandwidth per SM or core at each level in the memory hierarchy of the NVIDIA Tesla V100 vs. Intel Xeon SP processors.

NVIDIA Tesla V100 与 Intel Xeon SP 处理器的每个内存层次结构级别的每个 SM 或核心的延迟和可用带宽。

| **Memory Type** | **NVIDIA Tesla V1001[[1](#ref1)]<br>- per SM -** | **Intel Cascade Lake SP, Skylake SP[[2](#ref2)]<br>- per core -** |
|---|---|---|
| **Latency** | Bandwidth | Latency | Bandwidth |
| **L1 cache** | 28 cycles | 128 B/cycle | 4 cycles | 192 B/cycle |
| **Private L2 cache** | - N/A - | - N/A - | 14 cycles | 64 B/cycle |
| **Shared L2 or L3** | 193 cycles | 17.6 B/cycle | 50-70 cycles | 14.3 B/cycle |
| **Global or RAM** | 220-350 cycles | 7.4 B/cycle | 190-220 cycles | 1.9-2.5 B/cycle |

Latencies are not stated for registers because latency is dependent on the particular instruction being executed, not on data movement. Likewise, bandwidth is not an appropriate metric for registers; instead, one speaks of instruction throughput, which may reach or exceed one per cycle (if the pipelines of operands are kept full).

没有为寄存器声明延迟，因为延迟取决于正在执行的特定指令，而不是数据移动。同样，带宽不是寄存器的适当指标；相反，人们谈论的是指令吞吐量，它可能达到或超过每个周期一个（如果操作数的管道保持满）。

Despite the impressive bandwidth of the GPU's global memory, reads or writes from individual threads have high read/write latency. The SM's shared memory and L1 cache can be used to avoid the latency of direct interactions with with DRAM, to an extent. But in GPU programming, the best way to avoid the high latency penalty associated with global memory is to launch very large numbers of threads. That way, at least one warp is able to grab its next instruction from the instruction buffer and go, whenever another warp is stalled waiting for data. This technique is known as _latency hiding_.

尽管 GPU 的全局内存带宽令人印象深刻，但来自个别线程的读写操作具有很高的读写延迟。SM 的共享内存和 L1 缓存可以在一定程度上避免与 DRAM 直接交互的延迟。但在 GPU 编程中，避免与全局内存相关的高延迟惩罚的最佳方法是启动大量的线程。这样，每当另一个 warp 因等待数据而停滞时，至少有一个 warp 能够从指令缓冲区获取其下一个指令并继续执行。这种技术被称为 _延迟隐藏_。

Effectively, there is one additional layer of memory that ought to be considered for a GPU: the memory of the host. The need to transfer data between the host and the GPU can place a heavy constraint on the GPU's overall performance—particularly if the GPU is attached to the host via PCIe. Ideally, once the necessary data are transferred to the GPU, they remain there for computations as much as possible.

实际上，对于 GPU 来说，还有一个额外的内存层应该被考虑：主机的内存。主机和 GPU 之间传输数据的需求可能会严重限制 GPU 的整体性能——特别是如果 GPU 通过 PCIe 连接到主机的话。理想情况下，一旦将必要的数据传输到 GPU，它们就尽可能地保留在那里进行计算。

Since the latency cost for communicating data to and from the GPU is relatively high, data should be sent in big chunks or batches. Furthermore, different batches may be sent over separate _streams_, so that computation can begin on some batches while others are in transit. CUDA provides techniques for using streams to overlap computation with communication.

由于与 GPU 通信数据的延迟成本相对较高，因此应该以大块或批次发送数据。此外，不同的批次可以通过不同的流发送，以便在某些批次在传输时开始计算。CUDA 提供了使用流来重叠计算和通信的技术。

In the next section, we will continue to dig into the unique properties of the specific NVIDIA GPU devices that are incorporated into Frontera and contemporary HPC systems.

在下一节中，我们将继续深入探讨 Frontera 和当代 HPC 系统中集成的特定 NVIDIA GPU 设备的独特属性。

<span id="ref1">1</span>. Reference for Tesla V100 data: Z. Jia et al., [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826.pdf), Tables 3.1, 3.2, and 3.4, and Fig. 3.12. Theoretical peak rates are shown above; if the peak rate is unknown, the measured value is shown instead. Global bandwidths for the V100, such as 2155 GB/s for the L2 cache bandwidth, are converted to B/cycle/SM by dividing by 1.53 GHz and 80 SMs. (Note, the L1 "upper bound" rate in Table 3.2 of the reference is incorrect and should be computed by the formula in the accompanying text, i.e., LSUs\*bytes.) - Tesla V100 数据参考：Z. Jia 等人，通过微基准测试解剖 NVIDIA Volta GPU 架构，表 3.1、3.2 和 3.4，以及图 3.12。上面显示的是理论峰值速率；如果峰
值速率未知，则显示测量值。V100 的全局带宽，如 L2 缓存带宽的 2155 GB/s，通过除以 1.53 GHz 和 80 个 SM 转换为 B/cycle/SM。（注意，参考表 3.2 中的 L1 “上限”速率是不正确的，应该使用随附文本中的公式计算，即 LSUs \* 字节。）

<span id="ref2">2</span>. The memory hierarchies of [Cascade Lake](https://en.wikichip.org/wiki/intel/microarchitectures/cascade_lake#Memory_Hierarchy) and [Skylake](https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(server)#Memory_Hierarchy) are identical up to RAM, according to WikiChip. Published data puts the minimum RAM latency of [Cascade Lake](https://images.anandtech.com/doci/16594/ICX_briefing_51.png) at 81 nsec and [Skylake](https://www.anandtech.com/show/11544/intel-skylake-ep-vs-amd-epyc-7000-cpu-battle-of-the-decade/13) at 90 nsec; these values are converted to cycles by multiplying by 2.7 GHz and 2.1 GHz, respectively. The L3 bandwidth is taken to be the 30 GB/s asymptote seen in Fig. 3 of [Hammond, Vaughn, and Hughes (2018)](https://www.osti.gov/biblio/1515123), normalized by 2.1 GHz for their Xeon 8160 (same model as in Stampede2); the result should apply to Cascade Lake as well. The RAM bandwidths assume that all 6 channels are populated with the maximum [2933 MT/s DDR4](https://en.wikipedia.org/wiki/DDR4_SDRAM) for Cascade Lake or [2666 MT/s DDR4](https://en.wikipedia.org/wiki/DDR4_SDRAM) for Skylake; the global results are then divided by the number of cores per chip, either 28 for Cascade Lake or 24 for Skylake, and normalized by the same frequencies as above (to match [Frontera](https://frontera-portal.tacc.utexas.edu/user-guide/system/#cascade-lake-clx-compute-nodes) and [Stampede2](https://portal.tacc.utexas.edu/user-guides/stampede2#overview-skxcomputenodes), respectively). - 根据 WikiChip，Cascade Lake 和 Skylake 的内存层次结构直到 RAM 都是相同的。发布的数据显示 Cascade Lake 的最小 RAM 延迟为 81 纳秒，Skylake 为 90 纳秒；这些值通过乘以 2.7 GHz 和 2.1 GHz 分别转换为周期。L3 带宽取自 Hammond、Vaughn 和 Hughes（2018 年）图 3 中看到的 30 GB/s 渐近值，按 2.1 GHz 归一化，适用于他们的 Xeon 8160（与 Stampede2 中的型号相同）；结果应该也适用于 Cascade Lake。RAM 带宽假设所有 6 个通道都装满了最大 2933 MT/s DDR4 对于 Cascade Lake 或 2666 MT/s DDR4 对于 Skylake；全局结果然后除以每个芯片的核心数，分别是 28 个对于 Cascade Lake 或 24 个对于 Skylake，并通过上述相同的频率归一化（以匹配 Frontera 和 Stampede2）。

## <span id="Appendix">Appendix: Finer Memory Slices(附录：更细的内存切片)</span>

The table [in the main text](#comparison_cpu_mem) illuminates the per-SM or per-core capacities that pertain to different memory levels. However, it is perhaps fairer to look at how large a slice of each memory type is available to a single CUDA core in a GPU, vs. a single vector lane in a CPU. We again take the NVIDIA Tesla V100 and a couple of contemporary Intel Xeon server-grade processors as the examples.

正文中的表格阐明了与不同内存级别相关的每个 SM 或每个核心的容量。然而，也许更公平地看待的是，查看 GPU 中的单个 CUDA 核心与 CPU 中的单个矢量通道可用的每种内存类型的切片有多大。我们再次以 NVIDIA Tesla V100 和一些当代 Intel Xeon 服务器级处理器为例。

1. **_Register file._** In the NVIDIA Tesla V100, the register file of an SM stores (65536)/(2x32) = 1024 floats per CUDA core. In the Intel Cascade Lake and Skylake chips, a [CPU core](https://en.wikichip.org/wiki/intel/microarchitectures/cascade_lake#Individual_Core) has a [physical register file](https://travisdowns.github.io/blog/2020/05/26/kreg2.html#the-register-files) that stores [168 vector registers](https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(server)#Scheduler_.26_512-SIMD_addition) of 64B, or 10.5 kB in total. This works out to (16x168)/(2x16) = 84 floats per vector lane, which is an order of magnitude _less_ than what is available to a CUDA core. (The factors of 2 appear because a CPU core can handle 2x16-float vectors/cycle, and a GPU SM can handle 2x32-float warps/cycle.) - **_寄存器文件。_** 在 NVIDIA Tesla V100 中，SM 的寄存器文件存储 (65536)/(2x32) = 1024 个每个 CUDA 核心的浮点数。在 Intel Cascade Lake 和 Skylake 芯片中，CPU 核心有一个物理寄存器文件，存储 168 个 64B 的矢量寄存器，总共 10.5 kB。这相当于 (16x168)/(2x16) = 84 个每个矢量通道的浮点数，这比 CUDA 核心可用的少一个数量级。（2 的因子是因为 CPU 核心可以处理 2x16-float 向量/周期，GPU SM 可以处理 2x32-float warps/周期。）

2. **_Cache sizes._** In the NVIDIA Tesla V100, an SM has 128 kB (max) in its L1 data cache, and 64 kB in its constant cache. Adding these to its share of the 6 MB shared L2 (6/80 = 0.075 MB) yields 0.26 MB per SM, or 0.0041 MB per CUDA core. In the Intel Cascade Lake and Skylake chips, a CPU core has 32 kB in its L1 plus 1 MB in its L2 data cache. Adding these to its share of the shared L3 cache (1.375 MB) yields 2.4 MB, or 0.75 MB per vector lane, which is _two_ orders of magnitude _more_ than what is available to a CUDA core. - **_缓存大小。_** 在 NVIDIA Tesla V100 中，SM 在其 L1 数据缓存中有 128 kB（最大）和在其常量缓存中有 64 kB。将这些添加到其共享的 6 MB L2（6/80 = 0.075 MB）中，每个 SM 有 0.26 MB，或者每个 CUDA 核心有 0.0041 MB。在 Intel Cascade Lake 和 Skylake 芯片中，CPU 核心在其 L1 中有 32 kB，在其 L2 数据缓存中有 1 MB。将这些添加到其共享的 L3 缓存（1.375 MB）中，得到 2.4 MB，或者每个矢量通道有 0.75 MB，这比 CUDA 核心可用的多两个数量级。

3. **_Cache lines._** In any GPU, the 128-byte cache lines consist of four 32-byte sectors. In the event of a cache miss, not all 4 sectors in the cache line have to be filled, just the ones that need updating. This means that 4 sectors from 4 different cache lines can be fetched just as readily as 1 full cache line. The [32-byte sector size persists today](https://forums.developer.nvidia.com/t/pascal-l1-cache/49571/20), even though the Volta's HBM2 memory has a 64-byte interface. By contrast, in an Intel Cascade Lake or Skylake processor, data always travel all the way from RAM to registers in full cache lines of 64 bytes. - **_缓存行。_** 在任何 GPU 中，128 字节的缓存行由四个 32 字节的扇区组成。在缓存未命中的情况下，不必填充缓存行中的所有 4 个扇区，只需要更新需要更新的扇区。这意味着可以像获取一个完整的缓存行一样轻松地获取来自 4 个不同缓存行的 4 个扇区。32 字节的扇区大小即使在 Volta 的 HBM2 内存有 64 字节的接口的情况下也持续存在。相比之下，在 Intel Cascade Lake 或 Skylake 处理器中，数据总是以完整的 64 字节缓存行从 RAM 传输到寄存器。

---
