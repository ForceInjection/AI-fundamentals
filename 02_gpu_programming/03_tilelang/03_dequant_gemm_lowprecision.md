# 把权重压到 4-bit：TileLang 反量化 GEMM 源码级拆解

> 版本说明：本文基于 tilelang v0.1.15（tag a35f8ddf，2026-09-30 发布），行号取自该 tag 的代码。2026-11 复核时 main 已推进至 71a7c311，`dequantize_gemm` 目录在两个版本之间只有三处小修（`quantize/nvfp4.py` 的 UE4M3 NaN 处理、`quantize/utils.py` 的 interleave_weight 修复、cdna4 例子的 target 写法），正文引用的行号在 main 上同样适用。

一个 7B 参数的模型用 fp16 存权重要占约 14 GB 显存。自回归解码的 batch size 通常很小，每生成一个 token 都要把全部权重从 HBM 读一遍：按 H100 SXM 约 3.35 TB/s 的带宽粗算，光读权重就要 4 ms 上下，算术单元大部分时间在等数据。把权重压到 4-bit，读取量直接降到四分之一，这是 weight-only 量化在推理侧的立足点。

weight-only 的含义是只压权重，激活保持 fp16/bf16（或 int8）。decode 是访存受限阶段：权重矩阵每个 token 读一次，参与上百亿次乘加；激活向量只有几千个数，体积可以忽略。收益直接来自权重读取量的下降，激活保持高精度又避免了量化误差在多步自回归里累积。代价是 Tensor Core 不认识 4-bit 权重，计算前必须先把权重解回高精度格式，这一步就是反量化（dequantize）。

TileLang 仓库的 `examples/dequantize_gemm/` 目录集中放了这套 weight-only 反量化 GEMM 的参考实现：六个 CUDA 例子加一个 AMD 移植，配套一个 `quantize/` 工具包。本文逐文件拆解这些例子的实现路径、三个容易踩的坑，以及它们在 Hopper 上的指令形态。

## 1. 格式矩阵：目录里有什么

七个例子一张表就能看完：

| 例子                                      | 激活         | 权重                        | scale                                                  | 输出/累加                     |
| ----------------------------------------- | ------------ | --------------------------- | ------------------------------------------------------ | ----------------------------- |
| example_dequant_gemv_fp16xint4.py         | fp16，M=1    | uint4（int8 存储）          | 无（`group_size=-1` 退化为整 K，`with_scaling=False`） | fp16                          |
| example_dequant_gemm_w4a8.py              | int8         | 有符号 int4                 | 无，纯整数域                                           | int32                         |
| example_dequant_gemm_fp4_hopper.py        | fp16         | fp4 E2M1                    | 无                                                     | fp16/fp32                     |
| example_dequant_gemm_bf16_fp4_hopper.py   | bf16         | fp4 E2M1                    | 无                                                     | bf16/fp32                     |
| example_dequant_gemm_bf16_mxfp4_hopper.py | bf16         | mxfp4（E2M1 加 E8M0 scale） | 每 32 个 K 元素一个，形状 (N, K/32) 的 uint8           | bf16/fp32，可选 bias          |
| example_dequant_gemm_bf16_mxfp4_cdna4.py  | 同上         | 同上                        | 同上                                                   | 同上（AMD gfx950/MI350 移植） |
| example_dequant_gemm_fine_grained.py      | fp16 或 int8 | uint4                       | 无                                                     | fp16 或 int32                 |

fine_grained 的「fine_grained」指反量化粒度：每个线程固定取 128-bit 微块解码（example_dequant_gemm_fine_grained.py:42-43 的 `MAX_TRANSACTION_SIZE_IN_BITS=128`），与 scale 粒度无关。

仓库里另有三个目录名字相近，路线不同，先划清边界：

- `examples/gemm_fp8/`：fp8 激活配 fp8 权重，两边都是低精度，走 mma 指令而非 `T.gemm`（该目录 README 明言），不属于 weight-only；
- `examples/gemm_sm120/`：Blackwell 的 NVFP4 blockscaled 示例，fp4 数据直接送进 `T.mma_gemm_blockscaled`，由硬件解码；`quantize/nvfp4.py` 的实际消费者就是它，不服务于本目录；
- `examples/gemm_int4/`：`T.int4` 原生张量直接喂 `T.gemm`，全程不解包。

这三条路线都跳过了「软件反量化」这一步，与本文主题正交。

目录本身经历过一轮洗牌：最早的 example_dequant_gemm.py 在 2025-05-14 随 #494 删除；mxfp4 旧版在 2025-08-19 随 #725 重写；带显式 TMA 的 `_tma` 变体由 #766（2025-08-29）引入，又在 #1509（2025-12-24）删除并回并进非 TMA 版本，Scale 张量的加载从此交给编译器自动决策；grouped MoE 例子删于 #2013（2026-04-05）；`tilelang.quantize` 包在 #2761（2026-07-24）整体迁出为目录内的本地包。现在目录里的七个例子就是洗牌后的存货。

## 2. 反量化的两条路径

先说总原则：这批 CUDA 例子全部用软件解码。没有一条路线动用硬件解码单元，4-bit 数据总是在 SM 里用位运算或 PTX 指令解成 fp16/bf16 再进 Tensor Core。区别只在解码发生在哪个存储层级，由此分出两条路径。

寄存器路径：`B_shared` 先 `T.copy` 进 fragment（寄存器），用 TIR 表达式逐元素解码，产物留在寄存器里直接作为 `T.gemm` 的一个操作数。w4a8（example_dequant_gemm_w4a8.py:137-145）、fp4_hopper（example_dequant_gemm_fp4_hopper.py:205-213）和 gemv（example_dequant_gemv_fp16xint4.py:91-135）走这条路。

shared memory 路径：每个线程从 `B_shared` 取一段 128-bit 压缩数据，调 extern intrinsic 批量解码成 8 个 bf16/fp16，写回一张额外的 `B_dequantize_shared`，再由 `T.gemm(A_shared, B_dequantize_shared)` 消费。bf16_fp4（example_dequant_gemm_bf16_fp4_hopper.py:147-194）、mxfp4（example_dequant_gemm_bf16_mxfp4_hopper.py:196-263）、cdna4（example_dequant_gemm_bf16_mxfp4_cdna4.py:133-170）和 fine_grained（example_dequant_gemm_fine_grained.py:67-86）走这条路。

一个容易被忽略的细节：w4a8 和 fp4_hopper 的调用写的是 `T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)`。解码后的权重块坐在 gemm 的 A 位，激活块在 B 位；输出 Ct 形状是 (N, M)，写回全局内存时再转置。这个摆放与 wgmma 的 RS 形态（第 4 节）能对上：刚出寄存器的权重正好落在 A 的位置上。gemv 更直接：M=1 的场景连 `T.gemm` 都不用，每个线程解出 `micro_size_k` 个值后做标量乘加（fp16 域）或 `T.dp4a`（int8 域），再用 `tvm_thread_allreduce` 跨 32 个线程归约（example_dequant_gemv_fp16xint4.py:126-153）。

### 2.1 slow 路径：TIR 位拼装

每条路径都有快慢两种实现。slow 版本是纯 TIR 表达式，逐元素从 uint8 里抠 nibble、移位、拼出目标位型，共四组工具：

- 无符号整数：`_tir_packed_to_unsigned_convert`（quantize/quantization.py:310-318），移位加掩码，所有 uint4 例子共用；
- 有符号整数：w4a8 自带的 `_tir_u8_to_i4_to_i8`（example_dequant_gemm_w4a8.py:10-21），先 `i4 << 4` 重解释成 int8 再算术右移 4 位，完成符号扩展；quantization.py:333-343 的 `_tir_packed_int_to_int_convert` 是同一技巧的 32-bit 版本（左移 28 位再右移 28 位）；
- fp4 转 fp16：fp4_hopper 例子的 `_tir_u8_to_f4_to_f16`（example_dequant_gemm_fp4_hopper.py:10-29），E2M1 的指数域加偏置 14（fp16 偏置 15 减 fp4 偏置 1），再与符号位、尾数位拼成 fp16 位型；
- fp4 转 bf16：mxfp4 例子的本地副本 `_tir_u8_to_f4_to_bf16`（example_dequant_gemm_bf16_mxfp4_hopper.py:10-49），偏置换成 126；scale 不在函数内处理，由调用方在外面乘。

这类实现生成的代码就是普通的移位掩码指令序列，逐元素执行。好处是行为一目了然，容易和 torch 参考实现对账；坏处是每个元素五六条指令，解码本身就把带宽吃掉一截。fast 路径为省这些指令而生。

### 2.2 fast 路径：把 C 源码注入编译单元

fast 路径的机制是 `T.import_source` 加 `T.call_extern`：Python 侧从 quantize 包取一段写好的 CUDA C 源码注入生成的 .cu 文件，kernel 内部按函数名调用。quantize 包里有两族实现。

第一族是 int4 转 fp16 的 LOP3 变体族（quantize/lop3.py:6-38，选择器 `get_lop3_intrin_group` 从 :1091 起）。`lop3.b32` 一条指令完成三输入查表式位运算，真值表常量 `(0xf0 & 0xcc) | 0xaa` 一次做完「抽 nibble 加 fp16 指数魔数」，再用 `sub.f16x2` 减中值修正符号域：

```c
// examples/dequantize_gemm/quantize/lop3.py:12-30（节选，有符号/无符号共用模板）
static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
static constexpr uint BOTTOM_MASK = 0x000f000f;
static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;   // 1024.0 的 fp16 位型
static constexpr uint MEDIAN_NUM = isSigned ? 0x64086408 : 0x64006400;
for (int i = 0; i < (N / 2); i++) {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[i])
                 : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK),
                   "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[i]) : "r"(h[i]), "r"(MEDIAN_NUM));
}
```

一条 `lop3` 加一条 `sub.f16x2` 解出两个 fp16。lop3.py 全文约 1200 行，按目标 dtype、有无 scale、有无 zero point 展开了约 20 个变体。

第二族是 fp4 转 bf16 的 twiddling（quantize/mxfp.py:6-48）。四个 fp4 nibble 的排布经过特殊交织，解码靠 `prmt.b32` 字节重排、`and.b32` 按位掩码和 `mul.bf16x2` 实现，乘数 `0x7e807e80` 的每个 bf16 通道恰为 `2^126`，掩出指数尾数位后乘上它，隐含的指数域就补齐了：

```c
// examples/dequantize_gemm/quantize/mxfp.py:13-46（节选）
asm volatile(
    "prmt.b32 %13, %4, 0, 0x0123;"      // 字节重排，处理小端序
    "mov.b32 %12, 0x7e807e80;"          // 每个 bf16 通道 = 2^126
    "and.b32 %0, %13, 0b10000001110000001000000111000000;"
    "mul.bf16x2 %0, %0, %12;"
    "shl.b32 %1, %13, 3;"
    "and.b32 %1, %1, 0b10000001110000001000000111000000;"
    "mul.bf16x2 %1, %1, %12;"
    // 其余 nibble 由移位、掩码、或运算拼出，输出仍是 4 个 32-bit（8 个 bf16）
```

文件头注释注明参考了 Triton 的 hopper_value.py 实现。AMD 侧没有对应 PTX，cdna4 例子改用 16 项查找表的 C 实现（quantize/mxfp.py:136-157），表里次正规编码 0x1 对应的值是标准答案 0.5，这个细节在 3.1 节还会用到。

### 2.3 一段完整的 fast 路径骨架

把机制串起来，看 mxfp4 例子的 fast 宏（有删节，行号为原文件位置）：

```python
# example_dequant_gemm_bf16_mxfp4_hopper.py:196-263
MAX_TRANSACTION_SIZE_BITS = 128
local_size = MAX_TRANSACTION_SIZE_BITS // DataType(out_dtype).bits   # 8 个 bf16
local_compress_size = local_size // num_elems_per_byte               # 压缩态 2 字节

@T.macro
def fast_dequant_bf16_fp4_twiddling(B_shared, B_dequantize_shared, Scale, k):
    T.import_source(import_source)                # 注入 mxfp.py 的 PTX 源码 (:223)
    tx = T.get_thread_binding()

    B_local_thread = T.alloc_local((local_compress_size,), storage_dtype)
    B_dequantize_local_thread = T.alloc_local((local_size,), out_dtype)
    Scale_local_thread_exponent = T.alloc_local((1,), out_dtype)

    for i in T.serial(0, block_N * block_K // threads // local_size):
        # 每线程取 128-bit 压缩数据（2 字节 uint8）
        index_base = i * threads * local_compress_size + tx * local_compress_size
        for v in T.vectorized(0, local_compress_size):
            index = index_base + v
            B_local_thread[v] = B_shared[index // Block_QK, index % Block_QK]
        # 取 scale，左移一位得 2^Scale，uint8 字节直接当指数用（:244）
        Scale_local_thread[0] = Scale[bx * block_N + si, k * block_K // scale_size + sj]
        Scale_local_thread_exponent[0] = T.shift_left(1, Scale_local_thread[0])
        # 一次解 8 个 fp4 到 8 个 bf16（:247）
        T.call_extern(func_name, T.access_ptr(B_local_thread, "r"),
                      T.access_ptr(B_dequantize_local_thread, "w"), 1, dtype=out_dtype)
        # 乘 scale 后写回 shared memory
        for v in T.vectorized(local_size):
            B_dequantize_local_thread[v] *= Scale_local_thread_exponent[0]
        for v in T.vectorized(0, local_size):
            index = i * threads * local_size + tx * local_size + v
            B_dequantize_shared[index // block_K, index % block_K] = B_dequantize_local_thread[v]
```

simple 路径的对应物是 `get_simple_dequant_func`（同文件 :279-315）：`T.copy` 整块进 fragment，逐元素调 `_tir_u8_to_f4_to_bf16`，再写回。同一份 kernel 骨架靠 `fast_dequant` 布尔参数在两种实现之间切换（:369-372），例子的 `__main__`（:562-568）把 fast/simple 与有无 bias 组成四种组合逐一验证。

### 2.4 前置条件：interleave_weight 位重排

fast 路径对权重字节序有假设。bf16_fp4 例子的注释写明了 twiddling 排布的位归属：`(0,0)(3,0)(3,3)(1,0)(3,1)(3,2)(2,0)……`（example_dequant_gemm_bf16_fp4_hopper.py:109），括号对表示「第 y 位属于第 x 个 fp4 元素」，即连续 16-bit 里 4 个 fp4 的位是交织排列的。量化工具导出的权重不会长这样，入库前要过一遍 `interleave_weight`（quantize/utils.py:71-125）：按目标 dtype 的位宽把 nibble 重新铺位。gemv 例子在喂 kernel 前调用它（example_dequant_gemv_fp16xint4.py:201-204），LOP3 变体族同样假设这种排布。

## 3. 三个坑

### 3.1 fp4 转 fp16 时，次正规 0.5 被映成 0.75

`_tir_u8_to_f4_to_f16` 对所有编码统一加指数偏置 14。正常编码下这是 IEEE 精确转换，但 E2M1 的次正规编码 0x1（真值 0.5，指数域为 0、尾数 1）也被套用同一公式：得到指数 14、尾数 1，即 `1.5 * 2^(14-15) = 0.75`。文件里留了一行修正代码，被注释掉了（example_dequant_gemm_fp4_hopper.py:28）：

```python
# return tirx.Select(e_f4 == tirx.const(0, "uint32"), tirx.const(0, T.float16), val_f16)
```

同文件的 torch 参考 `torch_convert`（:32-41）用完全相同的公式拼位，核与参考自洽，测试照样通过。换句话说，这份实现和它的参考实现共用了一套非 IEEE 的转换，两者对得上，但与 IEEE 语义的 fp4 解码不一致；AMD 的 16 项 LUT（quantize/mxfp.py:136-157）里 0x1 映射到标准值 0.5，那条路径才是按标准来的。bf16 路径同理（偏置 126，0x1 同样映成 0.75）。拿这套 kernel 做数值实验时，0.5 这一档要记得实际是 0.75。

### 3.2 mxfp4 的 docstring 与代码不一致：2^(Scale-127) 还是 2^Scale

mxfp4 例子的 docstring 反复写 scale 按 `2^(Scale-127)` 解释（:208、:215、:385、:407）。代码不是这么做的：kernel 里是 `T.shift_left(1, Scale_local_thread[0])`（:244），simple 路径同款（:312），torch 参考是 `B *= 2 ** Scale[...]`（:397）。三处一致：Scale 字节被直接当 2 的幂使用，没有减 127。docstring 的「减 127」在代码里找不到对应物。引用语义以代码为准，对账 torch 参考时也按 `2^Scale` 算。

### 3.3 interleave_weight 的 1/2-bit 分支在 v0.1.15 必然 TypeError

quantize/utils.py 的 `interleave_weight` 在 v0.1.15 的 1-bit 和 2-bit 分支里写的是 `new_qweight & torch.int32(0xF0F00F0F)` 这类表达式。`torch.int32` 是 dtype 对象，把它当函数调用会直接抛 TypeError，凡走 1-bit（binary）或 2-bit 权重的代码在这版必然失败。main 分支已把它修成裸字面量（现 utils.py:104-122，修复 PR #3140，2026-10-08 落地）。4-bit fp16 主路径不受影响：`nbits=4` 时函数跳过这三个分支直接返回。

## 4. Hopper：这些 kernel 真的用上 wgmma 了吗

`T.gemm` 在 Hopper 上会尝试发射 wgmma（warpgroup 级异步矩阵乘）。发射条件写在 src/cuda/op/gemm.cc:88-97 的 `AllowWgmma`，四条同时满足才行：

- 目标架构是 Hopper；
- gemm 的 m 维不小于 64；
- 线程数折算的 warp 数能被 4 整除；
- `CheckWgmma` 通过。

`CheckWgmma`（:40-71）先要求 B 操作数位于 `shared.dyn` 或 `shared`，再按 dtype 组合检查 K 对齐（fp16/bf16 对 fp16/fp32 累加要求 `k%16==0`；int8 对 int32 要求 `!transA && transB` 且 `k%32==0`）。语言层同时封装了 SS 与 RS 两个变体（tilelang/cuda/language/intrinsics.py:70 可见 `ptx_wgmma_rs` 的导入），RS 形态允许 A 操作数留在寄存器。

第一步是静态推断，拿这些条件去对每个例子的 `T.gemm` 调用：

- mxfp4（:373）、bf16_fp4（example_dequant_gemm_bf16_fp4_hopper.py:341）：`T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)`，双操作数都在 shared；block_M=256 不小于 64，threads=256 折 8 个 warp，block_K=128 满足 k%16。预期发射 wgmma 的 SS 形态。
- fp4_hopper（:169、:213）：`T.gemm(B_dequantize_prev_local, A_shared, Ct_local)`，首参是寄存器 fragment、次参在 shared，对应 RS 形态；m 维取首参行数即 block_N=128，不小于 64，8 warp。预期发射 wgmma 的 RS 形态。
- w4a8（example_dequant_gemm_w4a8.py:145）：调用形态与 fp4_hopper 同为 RS 方向，但它的 block_N=32，m 维不足 64，`AllowWgmma` 过不了。预期不发射 wgmma。
- fine_grained（example_dequant_gemm_fine_grained.py:86）：同为 SS 布局方向，测试配置 128×128×32、threads=128（4 warp）也过线；它不在 regression 入口里，下面的实测集合不含它。文件后半还有一个 ladder 变体（:135 起），绕过 `T.gemm` 直接用 mma emitter 发射 mma 指令，与 wgmma 无关。

第二步是实测验证。对 regression 同款配置（2026-11-05，H100 sm_90，v0.1.15）编译出的 kernel，用 `JITKernel.get_kernel_source()` 取回生成码，按指令助记符计数：

| kernel     | 路径        | 配置                                           | wgmma | cp.async.bulk（TMA） |
| ---------- | ----------- | ---------------------------------------------- | ----- | -------------------- |
| gemv       | 寄存器，M=1 | 1×8192×8192                                    | 0     | 0                    |
| w4a8       | 寄存器      | 4096 立方，block 32×32×128，1 级流水，128 线程 | 0     | 0                    |
| fp4_hopper | 寄存器      | 4096 立方，128×128×128，2 级流水，256 线程     | 3     | 0                    |
| bf16_fp4   | shared      | 4096 立方，256×128×128，2 级流水，256 线程     | 10    | 0                    |
| mxfp4      | shared      | 4096 立方，256×128×128，2 级流水，256 线程     | 4     | 0                    |

计数与推断一一对上：fp4_hopper 发射了 3 处 wgmma（RS 形态，A 在寄存器）；bf16_fp4 与 mxfp4 分别 10 处、4 处（SS 形态，B 在 shared）；w4a8 的 wgmma 与 mma.sync 计数都是 0，证实 m≥64 这道门槛确实把它挡在 wgmma 之外，生成码里连 mma.sync 都没有；gemv 是 M=1 的点积形态，各线程标量乘加后跨线程归约，不含任何 tensor core 指令，计数为 0 符合预期。

mxfp4 例子里的 `if threads == 512: T.disable_warp_group_reg_alloc()`（:357-358）是一个旁证：warp group 是 Hopper wgmma 的调度单元，这行防御代码说明作者预期配置里会出现 512 线程的 warpgroup 组合。bf16_fp4 例子整个 kernel 显式带 `pass_configs={...TL_DISABLE_WARP_SPECIALIZED: True}`（:44），关掉了 warp specialization 流水，说明这版 kernel 在 Hopper 流水特性上做过取舍。

TMA 的演化与此类似：目录曾有显式用 1D TMA 加载整块 Scale 的 `_tma` 变体（#766 引入，#1509 删除），现在六个 CUDA 例子都没有显式 TMA 调用，全局到 shared 的搬运一律 `T.copy`，swizzle 用 `tilelang.layout.make_swizzled_layout` 标注（如 w4a8 :126-130、mxfp4 :344-348）。是否走 TMA 由编译器决定，上面五例的生成码给出了这批配置下的答案：cp.async.bulk 计数全为 0，编译器一处 TMA 都没有生成，与示例层无显式 API 互为印证。

## 5. 实测

测试环境：NVIDIA H100 80GB（sm_90）单卡，驱动 580.95.05，CUDA 13.0，tilelang 0.1.15（PyPI），torch 2.11.0+cu130，2026-11-05 测得。demo 表的数字是各例子直接运行时自己打印的（时延来自 `do_bench` 的 event 后端均值，warmup 25 ms、rep 100 ms、自动清 L2）；regression 表取自各例子 `run_regression_perf` 的 cupti 后端均值。吞吐为推算值，口径 2MNK/时延，gemv 为权重读取量/时延。

demo 形状（默认 256 立方，w4a8 例外）：

| 例子              | 路径组合                                | 时延                                | TFlops                               |
| ----------------- | --------------------------------------- | ----------------------------------- | ------------------------------------ |
| fp4_hopper        | torch 参考 / TileLang                   | 0.16 ms / 0.01 ms                   | 0.21 / 3.04                          |
| bf16_fp4_hopper   | fast（twiddling）/ simple               | 0.01 ms / 0.01 ms                   | 3.15 / 2.91                          |
| bf16_mxfp4_hopper | fast+bias / simple+bias / fast / simple | 0.03 / 0.03 / 0.01 / 0.01 ms        | 1.16 / 1.11 / 3.03 / 2.78            |
| w4a8              | CLI 默认形状 512 立方                   | 0.0115 ms（先打印 All checks pass） | 不适用（int8 域，例子不打印 TFlops） |
| gemv              | M=1，N=K=1024 默认                      | 不计时（只打印 C 与参考对照）       | 不适用                               |

bf16_fp4 两列的顺序按其 `__main__` 的调用序（:458-460，先 `fast=True` 后 `fast=False`）对应，故 3.15 是 fast、2.91 是 simple。w4a8 的 512 立方来自 argparse 缺省（example_dequant_gemm_w4a8.py:205-207），`main()` 函数签名里的 128×256×256 只是参数缺省，命令行入口总会覆盖它。

regression 形状（统一 4096 立方，gemv 例外 1×8192×8192）：

| 例子              | 形状        | 平均时延  | 推算吞吐                                         |
| ----------------- | ----------- | --------- | ------------------------------------------------ |
| gemv              | 1×8192×8192 | 0.0174 ms | 权重读取约 1.93 TB/s（33.6 MB int4 权重 / 时延） |
| fp4_hopper        | 4096 立方   | 0.5388 ms | 约 255 TFLOPS                                    |
| bf16_fp4_hopper   | 4096 立方   | 0.2658 ms | 约 517 TFLOPS                                    |
| bf16_mxfp4_hopper | 4096 立方   | 0.2597 ms | 约 529 TFLOPS                                    |
| w4a8              | 4096 立方   | 2.161 ms  | 约 63.6 TOPS（int8 域按 2MNK 计）                |

fast 路径对 simple 的优势在 demo 形状上只有个位数百分比（mxfp4 带 bias 时 1.16 对 1.11，不带 bias 时 3.03 对 2.78），bias 融合反而让 256 立方的时延翻倍以上（0.01 到 0.03 ms）。这个形状太小，解码省下的指令和 bias 加法的开销落在同一量级里，demo 数字只当量级参考，4096 立方的 regression 才是有效口径。

规模拉开后差距就清楚了。fp4_hopper 在 4096 立方只有约 255 TFLOPS，bf16_fp4/mxfp4 有 500+，差出一倍；原因在解码方式，fp4_hopper 只有 slow TIR 逐元素解码（寄存器路径），后两者走 fast twiddling（shared memory 路径），解码开销的差异直接反映在吞吐上。

还有一处容易误读：fp4_hopper 例子额外打印的 torch 参考（0.16 ms）是 `qB` 解包成 fp16 后走 torch.matmul 的对账实现，与 TileLang 路线不同源，不要拿它当性能对手。

## 6. 工程要点

统一测试入口 test_example_dequantize_gemm.py（33 行）只覆盖四个例子：gemv、fp4_hopper、mxfp4_hopper、w4a8，后三者都要求 CUDA 计算能力恰为 9.0（:15、:21、:27 的 `requires_cuda_compute_version_eq(9, 0)`），gemv 只要求有 CUDA。bf16_fp4 只进 regression 入口，cdna4 需要 gfx950，fine_grained 自己测自己、其 ladder 变体还要装 bitblas。拿这套代码做二次开发时，CI 能兜住的只有四个例子。

tolerance 各不相同，都有原因。gemv 的断言是 `atol=1e3, rtol=1e-1`（example_dequant_gemv_fp16xint4.py:217），数字大到扎眼，但这是无 scale 的 uint4 权重直接累加 K=1024 项的正常现象，代码注释自己说明了；w4a8 用 1e-2（example_dequant_gemm_w4a8.py:170）；fp4 系与 mxfp4 的四组参考（fast/simple 乘有无 bias）都是 0.01。另一个小笔误：mxfp4 的 `do_bench(warmup=500)`（:534）里 500 的单位是毫秒，docstring 写成「500 warmup iterations」（:482）。

选型的粗略结论：权重是 GPTQ 式整数化（无浮点 scale）就选 int4 域，其中 w4a8 要求激活也整数量化成 int8，fp16 激活配 int4 在 M=1 时走 gemv、M 变大后走 fine_grained 一类；权重是浮点域的（MXFP4 这类）选 fp4/mxfp4 系，mxfp4 带每 32 通道一个 scale，是七个例子里精度语义最完整的形态。

要留意的是，这些例子都没接入真实量化管线的打包格式：pack、unscale 一类预处理要自己补，`interleave_weight` 也只在 gemv 例子里被实际调用，其余例子的输入直接按交织排布生成。

## 7. 源文件索引

| 文件                                                               | 关键内容                                                                                                                       |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py         | GEMV 主 kernel :77-155；fast 用 LOP3 :113-119；interleave 调用 :201-204；atol=1e3 断言 :217；regression 1×8192×8192 :220-232   |
| examples/dequantize_gemm/example_dequant_gemm_w4a8.py              | int4 符号扩展 :10-21；寄存器路径解码与 T.gemm :137-145；tolerance :170；CLI 缺省 512 立方 :205-207                             |
| examples/dequantize_gemm/example_dequant_gemm_fp4_hopper.py        | fp4 转 fp16 TIR 解码 :10-29（修正注释 :28）；torch 参考 :32-41；split 变体 :135-177；寄存器路径 :205-213                       |
| examples/dequantize_gemm/example_dequant_gemm_bf16_fp4_hopper.py   | TL_DISABLE_WARP_SPECIALIZED :44；twiddling 位排布注释 :109；fast 宏 :147-194；T.gemm :341；`__main__` :458-460                 |
| examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py | fp4 转 bf16 本地副本 :10-49；fast 宏 :196-263（2^Scale :244）；simple 宏 :279-315；warp group 防御 :357-358；四组参考 :381-475 |
| examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_cdna4.py  | gfx950 target :44；fast 宏 :133-170；simple 宏 :176 起；T.gemm :238                                                            |
| examples/dequantize_gemm/example_dequant_gemm_fine_grained.py      | 128-bit 微块常量 :42-43；shared 路径与 T.gemm :67-86；ladder 变体 :135 起                                                      |
| examples/dequantize_gemm/quantize/quantization.py                  | TIR 位拼装：无符号 :310-318，带 zeros :321，有符号 :333-343                                                                    |
| examples/dequantize_gemm/quantize/lop3.py                          | int4 转 fp16 LOP3 变体族 :6-38；`get_lop3_intrin_group` :1091                                                                  |
| examples/dequantize_gemm/quantize/mxfp.py                          | fp4 转 bf16 twiddling :6-48（参考 Triton :5-6）；AMD LUT :136-157                                                              |
| examples/dequantize_gemm/quantize/utils.py                         | `interleave_weight` :71-125；1/2-bit 分支 :104-122（v0.1.15 存在 TypeError）                                                   |
| src/cuda/op/gemm.cc                                                | `CheckWgmma` :40-71；`AllowWgmma` :88-97                                                                                       |
| tilelang/cuda/language/intrinsics.py                               | wgmma SS/RS 变体导入 :60-70                                                                                                    |
| examples/dequantize_gemm/test_example_dequantize_gemm.py           | 四个统一测试，sm90 门槛 :15/:21/:27                                                                                            |
| examples/dequantize_gemm/regression_example_dequantize_gemm.py     | 五个 regression 入口（含 bf16_fp4）                                                                                            |

## 8. 参考资料

- tilelang 仓库：<https://github.com/tile-ai/tilelang>，本文锚定 tag v0.1.15（a35f8ddf）
- PR #494（首个 example_dequant_gemm.py 删除）、#725（mxfp4 旧版重写）、#766（Scale 的 1D TMA 加载）、#1509（TMA 变体回并，TMA 转为编译器决策）、#2013（grouped MoE 例删除）、#2761（quantize 包迁出为本地包）、#3140（interleave_weight 1/2-bit 分支修复）
- Triton hopper_value.py（mxfp.py twiddling 的参考来源）：<https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/tensor_details/layout_details/hopper_value.py>
- 本仓对读：[量化 GEMM 算子系列](../06_quant_gemm/)（[Marlin 深拆](../06_quant_gemm/01_marlin-deep-dive.md)与本文的 LOP3/twiddling 写法互为对读，续篇拆 Humming JIT 路线与 tcgen05 kernel 内部）
