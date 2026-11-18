# 素材快照：Mixture-of-Experts kernel engineering on Amazon SageMaker AI

- 来源: https://aws.amazon.com/cn/blogs/machine-learning/mixture-of-experts-kernel-engineering-on-amazon-sagemaker-ai/
- 发布时间: 2026-10-13（article:modified_time 同日）
- 抓取日期: 2026-11-18（web_reader 全文）
- 定位: AWS 工程师对 vLLM fused MoE 的三项 kernel 优化，全部上游进 vLLM 主仓；应用于 SageMaker LMI 容器
- 配图: ../assets/vllm_fused_moe_swapab.png（2230×570，博客 Figure 1）

## 正文全文

Mixture-of-Experts (MoE) is a sparsely-activated architecture that scales large language model (LLM) capacity while keeping per-token compute low. Because its operations account for a large share of total compute time, inefficiencies there directly raise request latency and cost. Therefore, optimizing MoE operations through kernel engineering offers the largest opportunity for performance gains.

Fused MoE is a key optimization technique for the MoE architecture. A fused MoE system merges token routing, data shuffling, and specialized network calculations into a single batched operation on the GPU. This replaces executing those routing and expert computations as separate, sequential loop operations. While this provides latency advantages, it also brings complexity in kernel implementations and debugging capability. High GPU utilization for this architecture requires hardware-specific optimizations.

In this post, we describe the different kernel optimizations we implemented for fused MoE models and show you how you can benefit from this work. We used a representative MoE model as our primary example.

### Swap-AB for WGMMA utilization

To maximize tensor processing throughput, the NVIDIA Hopper architecture relies on the Warp Group Matrix Multiply Accumulate (WGMMA) instruction protocol, which executes collectively across a cooperative warpgroup of 128 threads. This operand asymmetry is fundamentally an instruction set architecture (ISA)-level property. In the underlying `wgmma.mma_async` assembly definition, the hardware fixes the matrix row dimension (M) at a constant scale of 64 to define its parallel thread-mapping topology. The column dimension (N) remains highly variable. Operand A strictly maps across this fixed M dimension (using registers or shared memory), and Operand B maps across the variable N dimension (through a shared memory descriptor). These operand slots are therefore structurally non-interchangeable. Consequently, peak hardware tile utilization is governed by these rigid shape encodings, requiring that the M dimension equals or exceeds 64 to prevent compute execution lanes from sitting idle.

This structural preference stems from how Hopper Tensor Cores partition memory blocks. The hardware enforces a fixed minimum layout shape for the leading dimension of the first operand, locking the block height to exactly 64 (M=64). When a mathematical matrix possesses a large, continuous leading dimension in this primary slot, Hopper's internal hardware schedulers can cleanly map and distribute data across the physical computing grid.

This structural asymmetry becomes a severe performance bottleneck during the decode phase of MoE networks. In a standard MoE layer, the linear algebra governing expert computation is formulated using traditional matrix notation:

A[M,K] x B[K,N] —> C[M,N]

Where (M) represents the token row count (or batch size), (K) represents the hidden feature dimension, and (N) represents the expansive expert output dimension.

During the decode phase, the system processes only one token per user sequence at a time. Consequently, the batch space or token row size is profoundly thin, regularly dropping to a profile where (M < 64), and frequently down to a single row (M = 1). Under this standard structural layout:

Matrix (A) transforms into a thin, fragmented horizontal vector (for example, 1 x 4096).

Matrix (B) (the core expert weights) remains in a large, dense vertical block (for example, 4096 x 8192).

Because the row count (M) is tiny, Matrix (A) is physically incapable of filling the hardware's fixed (64)-row layout bucket for Operand A. When the WGMMA instruction executes over this layout, it leaves a vast portion of the physical Tensor Core capacity completely starved. The silicon spends clock cycles processing empty spatial slots, leading to poor hardware utilization.

In a standard dense LLM layer, weight matrices are static, unified, and perfectly contiguous. In an MoE layer, however, the workload is fractured across a fleet of independent subnetworks (experts). This creates unique memory and layout bottlenecks that make it hypersensitive to the WGMMA constraints of the Hopper GPU. In a standard dense layer, matrix multiplication multiplies a token vector by a single layer weight matrix. The output dimension (N) is the model's hidden size (for example, 4096).

In a Fused MoE kernel, the GPU isn't only multiplying against one expert. It's grouping and running math for all active experts simultaneously to avoid launching individual kernels for each expert. For a standard dense layer, N is the hidden size (4,096). For a fused MoE layer, N is the number of experts chosen multiplied by the expert intermediate size. For a model running a `topk=8` routine, the effective N dimension inside the fused kernel explodes into tens or hundreds of thousands of parameters. Because the N dimension is so disproportionately large compared to the tiny decode token size (M=1), the layout asymmetry becomes extreme. Transposing the matrices moves this large, combined MoE dimension into the primary slot, unlocking hardware prefetching at a scale that standard dense layers cannot match.

To resolve this hardware misalignment without altering the underlying mathematical integrity of the model, we exploit the algebraic property of matrix transposition:

BT[N,K] x AT[K,M] —> CT[N,M]

[Figure 1: Swap-AB transposition promoting the large expert dimension into the WGMMA operand-A slot for full Tensor Core utilization — ../assets/vllm_fused_moe_swapab.png]

By programmatically flipping the matrix multiplication before it reaches the PTX assembly layer, we realigned the software to match the exact topography of the Hopper silicon. This transposition triggers two immediate architectural advantages on the GPU. First, it saturates the physical layout. The expansive expert dimension (N = 8192) is promoted to the first operand position (BT), and because (N) is large it easily floods the hardware's mandatory (64)-row threshold. The Tensor Cores then operate at maximum local density with no wasted execution cycles. Second, it activates asynchronous prefetching. Hopper's dedicated hardware copy engines can prefetch memory addresses asynchronously from shared memory straight into registers before the math execution block fires. Placing the continuous, large weight blocks (BT) in the leading slot lets these engines trace the data layout smoothly and keep the compute pipeline saturated without CPU intervention. This produced a 2–4 percent end-to-end output throughput improvement across large-scale MoE models on H200. We upstreamed this optimization to vLLM, and it is available in version 0.24.0 or later.

### Backend routing for FP8 tensor quantization

Beyond kernel internals, the choice of which backend executes the MoE computation has a large effect on performance. The vLLM MoE oracle routes different quantization schemes to Triton, FlashInfer+CUTLASS, or other backends. Profiling revealed that for FP8 tensor-quantized models on Hopper, the Triton backend consistently outperforms the default `FLASHINFER_CUTLASS` path.

The Triton backend outperforms CUTLASS in low-concurrency MoE workloads primarily because it replaces rigid, framework-level matrix math configurations with custom, operator-level register fusion. CUTLASS relies on standard C++ template structures that treat FP8 tensor scaling, matrix multiplication, and expert aggregation as discrete mathematical operations. The Triton backend instead consolidates this entire pipeline into a single hardware pass. Data is pulled from global high-bandwidth memory (HBM) exactly once and processed entirely within the GPU's ultra-fast local SRAM registers and compute execution files. It's then streamed back to the final matrix destination. This removes the slow global VRAM read/write roundtrips that compromise CUTLASS when it is forced to constantly step through intermediate tensor scaling steps.

Furthermore, this register-level fusion is specifically paired with dynamic scheduling layouts designed around variable token bucketing, resolving a severe structural flaw in traditional compilation templates. CUTLASS is natively optimized for dense, large, and predictable matrix dimensions. When it encounters the fluid, ultra-thin batch profiles (M < 64 or M = 1) typical of the live decode generation phase, its default block-tiling and thread-mapping structures become profoundly inefficient. The result is heavy warp under-utilization or expensive memory padding overhead. Conversely, the Triton MoE backend excels at dynamically mapping these small, fragmented token arrays across the physical boundaries of the active expert weights. This keeps the hardware's internal prefetch streams and Tensor Core pipelines fully saturated even when processing highly unstable workloads at very low batch sizes.

Updating the routing policy is a pure configuration change with no kernel code modified. Still, it yielded a **10 percent end-to-end throughput improvement** for FP8 quantized fused MoE models on H200. This improvement is available in the AWS Large Model Inference (LMI) v28 container.

### Triton JIT recompilation elimination

One of the most impactful performance pathologies in GPU inference is also one of the least obvious: unnecessary kernel recompilation. Triton, the compiler framework underlying many of vLLM's custom kernels, specializes integer arguments by default. Every time a kernel is invoked with a previously unseen value for a specialized parameter, Triton triggers a fresh just-in-time (JIT) compilation. In serving workloads where sequence lengths and batch sizes vary continuously, this creates a steady stream of compilation overhead that inflates latency, particularly Time to First Token (TTFT).

In MoE model serving, the `_fused_inv_rope_fp8_quant_per_head` kernel (responsible for fused rotary position embedding and FP8 quantization) was recompiling on every new `num_tokens` value. Adding `do_not_specialize` annotations to the variable-length parameters removed these recompilations entirely, improving TTFT by **9 percent** on H200 at 8-way tensor parallelism (PR #42849).

In a multi-GPU cluster, all 8 GPUs must execute their kernels in perfect, microsecond-level synchronization. If even one GPU must stop to recompile its local `_fused_inv_rope_fp8_quant_per_head` kernel because its local token batch count changed, the other seven GPUs sit idle and wait for it. By eliminating recompilations across the entire cluster, we smoothed out the multi-GPU coordination overhead, delivering a 9 percent speedup for the user's initial response latency. The principle here is to treat the compiler as a first-class optimization surface. Understanding how Triton specializes semantics and recognizing when parameters are dead or variable unlocks latency improvements with minimal code change.

We upstreamed this optimization to vLLM, and it is available in version 0.22.0 or later.

### Conclusion

Through our collaboration with the vLLM community, we implemented and open sourced multiple kernel optimizations for fused MoE models. We apply these optimizations in the LMI Deep Learning Container on Amazon SageMaker AI. Together, these optimizations have delivered up to 10 percent higher throughput and up to 9 percent faster TTFT. To get started with custom model hosting, see the Amazon SageMaker AI hosting documentation.

（文末作者介绍略——与正文技术内容无关。）
