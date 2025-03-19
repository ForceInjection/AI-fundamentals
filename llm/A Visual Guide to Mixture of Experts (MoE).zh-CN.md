混合专家系统（MoE）图解指南
==========================================

在追踪最新发布的**大型语言模型**时，您会发现许多模型的标题中都醒目标注着「**MoE**」。 这个「**MoE**」究竟代表何种技术？为何能成为众多 `LLM` 的架构首选？

本图解指南将通过 **50+组可视化解析图**，带您深入探索这一核心组件——**混合专家系统**（`Mixture of Experts`）。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F50a9eba8-8490-4959-8cda-f0855af65d67_1360x972.png)

本文将系统解析MoE架构的两大核心模块：应用于典型`LLM`架构中的**专家系统**与**路由分配器**。

若想获取更多与大型语言模型相关的可视化解析，并支持本专栏持续创作，欢迎查阅笔者编著的《**大型语言模型实战指南**》！

> 附言：若您已品读此书，在[亚马逊平台](https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961) 撰写简短书评将是对作者最珍贵的鼓励——您的反馈切实影响着学术创作生态。
> 
> `Github` 地址：`https://github.com/HandsOnLLM/Hands-On-Large-Language-Models`

**混合专家系统**（`Mixture of Experts`，`MoE`）是一种通过协同多个差异化子模型（即「**专家单元**」）来提升大型语言模型性能的技术范式。

`MoE`架构包含两大核心组件：

*   **专家集群**——每个前馈神经网络层配置了可动态调度的多组专家单元集合这些专家单元本质上仍保持前馈神经网络的结构特性；
*   **路由分配器**或**门控网络**——负责确定文本标记的专家调度路径。

在集成`MoE`的大型语言模型各层级中，均部署了（具有领域特化能力的）专家单元：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7931367a-a4a0-47ac-b363-62907cd6291c_1460x356.png)

需要明确的是，这里所说的“专家单元”并非专攻“心理学”或“生物学”等特定领域。其学习能力主要局限于词汇层面的语法信息：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc6a81780-27c8-45f8-bccc-cc8f1ce3e943_1460x252.png)

更准确地说，这些专家系统的核心能力体现在处理**特定语境中的文本标记**。

**路由决策层**（门控网络）会为当前输入层动态选择最优的专家集群：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb6a623a4-fdbc-4abf-883b-3c2679b4ad4d_1460x640.png)

每个专家单元并**非完整**的语言大模型，而是大模型架构中的**子模块组件**。

要深入理解专家系统的**表征机制**和**工作原理**，我们首先需要剖析`MoE`技术旨在替代的核心结构——密集连接层。

混合专家系统的底层基础源于语言大模型中的前馈神经网络（`FFNN`）模块。

值得注意的是，在标准的仅解码器架构`Transformer`中，`FFNN`应用在层标准化处理之后：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4729d2a-a51a-4224-93fe-c5674b9b38eb_1460x800.png)

前馈神经网络能够利用注意力机制生成的上下文信息，通过深度转换捕捉数据中更复杂的关联模式。

但前馈神经网络的参数量会呈现指数级增长趋势。为了学习这些复杂的关系，模型通常会对接收的输入信息进行扩展处理：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F091ec102-45f0-4456-9e0a-7218a49e01df_1460x732.png)

传统`Transformer`中的前馈神经网络被称为**稠密模型**，因为所有模型参数（包括**权重**和**偏置项**）都会被激活。所有输入信息都会被完整保留并参与输出结果的计算。

仔细分析稠密模型可以发现，输入信息会在不同程度上激活所有参数：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F101e8ddc-9aa7-4e24-92fc-78d25da73399_880x656.png)

与之形成对比的是**稀疏模型**，这类模型仅**激活部分参数**，其原理与混合专家系统密切相关。

具体实现上，可将稠密模型拆分为多个专家单元，经重新训练后，每次推理仅激活部分专家集群：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc4eeaf8-166b-419f-896c-463498af5692_880x656.png)


核心原理在于：**每个专家单元在训练过程中会习得不同的信息特征**。**在执行推理时，仅调用与当前任务最相关的特定专家系统**。

当处理具体问题时，可选取最适合当前任务的专家单元：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fce63e5cc-9b82-45b4-b3dc-9db0cac47da3_880x748.png)

如前所述，专家集群学习的信息粒度比整个领域更细致[1](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts#footnote-1-148217245) ，因此将其统称为'专家'有时可能不够准确。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F04123f9e-b798-4712-bcfb-70a26438f3b9_2240x1588.png)

`ST-MoE`论文中编码器模型的专家单元展现出显著的专业化特性。

然而在解码器模型中，专家集群的专业化特征表现相对模糊。但这并不代表所有专家单元都具备相同效能。

[Mixtral 8x7B论文](https://arxiv.org/pdf/2401.04088) 提供了典型范例，论文中每个文本标记均按首选专家单元进行颜色编码。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd03e32b4-5830-4d98-8514-0c1a28127ed9_1028x420.png)

可视化分析同时揭示，专家系统更多聚焦**语法结构**而非特定领域特征。

因此，虽然解码器专家单元未显现明确专业方向，但其对特定类型标记的处理仍保持稳定模式。

尽管将专家单元视为分割后的稠密模型隐藏层有助于直观理解，但实际上它们通常是完整的前馈神经网络结构：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe51561eb-f3d6-45ca-a2f8-c71abfa7c2a9_880x748.png)

鉴于多数大型语言模型配置多个解码器模块，文本生成需经历多层级联的专家系统处理流程：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F89b1caad-5201-43fe-b7de-04ebe877eb2d_1196x836.png)

不同文本标记选择的专家单元存在差异，最终形成多样化的'处理路径'：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcde4794d-8b3e-454d-9a1c-88c1999fdd45_1372x932.png)

若重构解码器模块的可视化模型，当前应呈现多个独立的前馈神经网络（每个对应一个专家单元）：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb97a8ac7-db97-497f-866d-10400729d51e_1248x764.png)

当前解码器模块集成了多个前馈神经网络（每个作为独立的'专家单元'），可在推理过程中灵活调用。

在构建专家集群后，模型如何智能选择适用的专家系统？

专家层前端配置的**路由分配器**（亦称**门控网络**），经专项训练可为特定文本标记智能匹配最优专家单元。

**路由分配器**（或称**门控网络**）本身作为前馈神经网络，基于输入特征动态选择专家单元。该机制通过概率输出实现专家单元的最优匹配：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Facc49abf-bc55-45fd-9697-99c9434087d0_864x916.png)

专家层将选定专家单元的输出结果与门控值（选择概率）进行加权后输出。

路由分配器与专家集群（仅部分激活）共同组成**混合专家架构层**：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa6fcabc6-78cd-477f-ac4e-2260cb06e230_1160x688.png)

混合专家架构层提供两种形态：_**稀疏型**_ 与 _**密集型**_ 专家系统组合方案。

两者虽均采用路由分配机制，但稀疏混合架构仅激活少数专家单元，而密集型架构则全量调用并实施差异化权重分配。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46aadf17-3afe-4c98-b57c-83b7b38918b2_1004x720.png)

例如，当给定一组文本标记时，混合专家架构会将标记分配给所有专家集群进行处理，而稀疏混合专家架构则仅会选取少量专家单元参与计算。

在当前大型语言模型的技术现状中，所提及的`MoE`通常特指稀疏混合专家架构，这种设计允许系统仅调用部分专家单元进行运算。这种架构显著降低了计算成本，这对大型语言模型的运行效率具有关键意义。

门控网络堪称混合专家架构的核心组件，它不仅主导 _**推理过程**_ 中的专家选择机制，更在 _**训练流程**_ 中发挥着关键调控作用。

在最基础的数学表达中，输入向量(_**x**_)将与路由权重矩阵(_**W**_)进行矩阵乘法运算：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58234ce0-bf96-49ab-b414-674a710a1c3c_1164x368.png)

随后通过**SoftMax函数**对输出值进行处理，生成针对各专家单元的概率分布函数**G**(_**x**_)：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb888a32f-acef-4fff-9d4b-cc70e148a8f2_1164x384.png)

路由分配器基于该概率分布，为每个输入选择最适配的专家单元进行处理。

最终系统将每个路由通道的输出结果与对应专家单元的计算输出相乘，并进行聚合求和操作。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe6e46ea4-dbd4-4cc4-aa2b-2c5474917f31_1164x464.png)

让我们将各模块整合，完整解析输入数据在路由决策层与专家集群间的流动路径：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd5d24a0b-2d78-4c69-b6fe-d75ba34bdd0c_2080x2240.png)

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d1122aa-7248-47d0-8e01-caa941ce0aa9_2080x2240.png)

然而，这种简单的函数设计往往导致路由决策层持续选择同一专家单元，因为部分专家单元可能具有更快的参数更新速度：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9233733c-c152-428a-ae99-1ed185fc3d50_1164x660.png)

这不仅会造成专家选择分布失衡，部分专家甚至可能完全无法获得有效训练，最终导致训练流程与推理过程的双重问题。

我们的优化目标是确保各专家单元在训练和推理阶段保持均衡激活状态，该机制被称为**负载均衡**。本质上，这种设计旨在防止模型对固定专家单元产生路径依赖式的过拟合。

要实现专家单元的重要性均衡，必须重点优化路由分配器——这个决定专家选择时序的核心控制组件。

实现路由层负载均衡的有效方法之一是采用[KeepTopK](https://arxiv.org/pdf/1701.06538) [2](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts#footnote-2-148217245) 扩展机制。通过注入可训练的高斯噪声参数，系统能够打破专家选择的固化模式：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1b95b020-ae34-40f0-a5c4-9542343beea9_1164x412.png)

随后，除需要激活的前`k`个专家单元（例如2个）外，其余专家单元的权重将被设置为 **-∞**:

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F66bea40e-3fb0-4937-88d5-2852af456cf3_1164x488.png)

通过将对应权重设为 **-∞**，这些权重经 `SoftMax` 函数处理后得到的概率值将为 **0**：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F687d2279-1d8b-4af1-b55e-55d618ee877f_1164x496.png)

尽管存在诸多优质替代方案，`KeepTopK`策略仍是当前多数大型语言模型采用的核心路由机制。需注意，KeepTopK策略可不依赖额外噪声独立运作。

`KeepTopK`策略通过为每个文本标记选择特定专家单元实现定向处理。该方法称为_标记择路_[3](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts#footnote-3-148217245) ，支持将文本标记路由至单一专家（ _Top-1路由_ ）：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdf7a9988-d4c8-4b1b-a968-073a6b3bfc6a_1004x648.png)


或同时分配给多个专家单元（top-k路由）：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb3f283f1-c359-4baf-8d01-8ebb2a90665f_1004x720.png)

关键优势在于可实现专家系统贡献度的动态加权与有机融合。

为确保训练过程中专家集群的均衡调用，网络在常规损失函数外特别引入了辅助损失（亦称 _负载均衡损失_ ）。

该约束条件强制要求所有专家单元必须保持同等的重要性权重。

该辅助损失函数的第一部分计算方式为：对整批数据中各专家单元的路由分配值进行求和。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3624da0-3137-42ba-95e8-88fcbddb5f9f_1108x288.png)

由此可获得各专家单元的 _重要性评分_，该指标反映了不考虑输入层时专家单元被选中的概率分布。

基于此可计算 _变异系数_ (**CV**)，用于量化不同专家单元间重要性评分的离散程度。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F94def8dc-2a65-4a02-855f-219f0df2a119_916x128.png)

举例说明，当专家单元间重要性评分差异显著时，**CV**值将呈现较高水平：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fab71b90c-ba29-42a9-944b-3dee52fc5c32_916x372.png)

反之，若所有专家单元重要性评分趋近一致，**CV**值则处于较低区间（这正是预期的优化目标）：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc5cb91ac-4aab-4eb5-80bf-84e2bd4dc576_916x324.png)

通过引入**CV**指标，我们可以在训练流程中动态调整 _辅助损失函数_，使其以最小化**CV**值为优化方向（从而实现各专家单元的重要性权重均衡化）：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4aac801-af89-44e7-aaea-c57a55ff282c_916x312.png)

最终，该辅助损失函数将作为独立的优化目标项整合到整体训练流程中。

不平衡现象不仅存在于被选中的专家集群中，也体现在分配到各专家的文本标记分布上。

例如，若输入层的文本标记被不成比例地分配给某个专家单元，则可能导致该专家训练不足：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F749eac8e-36e5-450f-a6fc-fbe48b7a1312_1004x484.png)

在此场景下，关键不仅在于选用哪些专家系统，更在于其被调用的**强度**。

该问题的解决方案是设定 _专家容量_ [4](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts#footnote-4-148217245) ，即限制单个专家单元可处理的文本标记数量。当专家达到容量上限时，溢出的文本标记将自动路由至下一顺位专家：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdf67563f-755a-47a7-bebc-c1ac81a01f8f_1004x568.png)

若所有专家单元均达到容量上限，未处理的文本标记将绕过专家层直接传递至后续网络层。该机制被称为 _**标记溢出现象**_。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe92ce4c5-affa-454d-8fd2-4debf9a08ce2_1004x544.png)

`Switch Transformer`[5](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts#footnote-5-148217245) 作为首个基于`Transformer`架构的`MoE`模型，成功解决了训练稳定性问题（如负载均衡）。该模型在提升训练稳定性的同时，显著简化了系统架构和训练流程。

`Switch Transformer`是基于`T5`模型（编码器-解码器架构）改进的模型，通过用交换层替代传统的前馈神经网络层。该交换层采用稀疏混合专家架构（`Sparse MoE`），基于`Top-1`路由选择机制为每个文本标记分配唯一专家单元。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F024d1788-9007-4953-9bf7-883da0db7f8d_1160x688.png)

路由分配器采用直接计算方法确定专家选择：将输入数据与专家权重矩阵相乘后，通过`SoftMax`函数处理得到路由决策（与前期处理方法保持一致）。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff0758a7f-e26b-44b9-9d75-80ac6caa9802_1104x384.png)

这种`Top-1`路由选择架构的设计假设是：路由分配器仅需调用单一专家单元即可完成输入数据的有效路由学习。这与我们先前讨论的`top-k`路由机制形成理论对比——后者认为需要通过多个专家集群的协同工作才能实现最佳路由行为学习。

容量因子作为核心参数，直接决定单个专家单元可处理的文本标记最大数量。`Switch Transformer`通过创新性地引入**容量因子**这一关键参数，实现了对专家系统处理能力的动态调控与优化扩展。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F22715139-3955-4e00-bed7-c45cffa52744_964x128.png)

专家系统容量的构成要素包含以下简明维度：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4b399c6-723b-4de6-94ca-7020cd1bb181_908x380.png)

当容量因子数值增大时，每个专家单元可处理的文本标记数量将获得线性提升。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7fd2aea0-fddf-4a43-ac79-7c5e5194c115_1240x472.png)

然而，若容量因子设置过大，会导致计算资源浪费。相反，若容量因子过小，模型性能将因 _标记溢出现象_ 而降低。

为进一步防范标记丢失问题，研究团队采用了简化版的辅助损失函数。

相较于计算变异系数，该简化损失函数通过权衡专家单元的标记分配比例与其路由概率占比来实现优化：

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F608da44a-7510-4ab6-97c9-e8ab212a567d_836x388.png)

由于目标是在**N**个专家单元间实现标记的均匀路由分配，因此需要确保向量**P**和**f**的取值均为**1/N**。

超参数**α**可用于在训练过程中精确调节该损失函数的影响权重。参数值过高会压制主损失函数，过低则对负载均衡作用甚微。

混合专家技术并非语言模型专属的解决方案。基于`Transformer`架构的视觉模型（如`ViT`）天然具备集成混合专家系统的可能性。

简而言之，`ViT`（视觉`Transformer`）是一种将图像分割为图像分块进行处理的架构，其处理逻辑与文本标记高度相似。[6](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts#footnote-6-148217245)

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F11b64fce-4069-4c73-995d-c3059fda0dcc_1300x828.png)

这些图像分块（或文本标记）在输入标准编码器前，会被转换为附带位置编码的嵌入表示。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc0b2ea60-238b-446a-ab59-503efb6ca061_1228x1232.png)

当这些分块进入编码器时，其处理方式与文本标记类似，这种设计使得架构能够完美适配混合专家系统。

视觉混合专家架构（`V-MoE`）是图像领域首批成功应用混合专家技术的模型之一[7](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts#footnote-7-148217245) ，其在`ViT`架构基础上，将编码层的稠密前馈网络替换为稀疏混合专家系统。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F10e9721d-4b3f-4062-ad72-97ffd1049077_1160x944.png)

这使得原本规模较小的`ViT`模型，通过扩展专家集群实现了参数量的显著提升。

考虑到图像通常包含大量分块，为缓解硬件限制，每个专家单元都预设了较小的处理容量。但容量过低容易导致分块被丢弃（类似 _标记溢出_ 问题）。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F219141c9-51ff-4d85-9f8a-c705c6e9ece4_1720x744.png)

为维持低容量配置，网络通过优先级机制——先处理高重要性评分分块，使得溢出分块通常具有较低语义重要性。这种方法被称作批量 _优先级路由_ 机制。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa0ef5323-4b4c-4ee7-8a53-51fbe4213283_1720x772.png)

因此，即使文本标记的比例降低，我们仍能确保重要图像分块被正确路由。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F65f972b9-640b-4a76-b77d-2d2ef1b40609_1736x420.png)


优先级路由机制通过聚焦最关键的分块，可显著减少需要处理的图像分块数量。

在`V-MoE`架构中，优先级评分器能有效区分图像分块的重要性差异。但每个专家单元仅处理被分配的分块，导致未处理图块的信息丢失。

`Soft-MoE`通过混合图块的方式，实现了从离散分配到柔性分配的转变（适用于图块和文本标记）。[8](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts#footnote-8-148217245)

首先将输入特征 **x**（即图块嵌入表示）与可训练参数矩阵Φ进行矩阵相乘。由此生成 _路由决策信息_，该信息量化了特定文本标记与各专家单元的相关性强度。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F644c0c1c-24d3-491b-a9a2-fdd9658ad589_1032x516.png)

通过对路由信息矩阵按列执行`SoftMax`运算，我们动态更新每个图块的嵌入表示。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6c3187d8-5bf2-4c73-8c22-547107fe1152_1032x456.png)

更新后的图块嵌入表示本质上是所有图块嵌入表示的加权平均结果。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7cfeb30a-1b04-4b9a-8f5a-d3c5d47e6499_1376x400.png)

从视觉角度观察，所有图块呈现出混合交融的状态。这些整合后的图块数据将被并行传输至各专家单元进行处理。输出结果生成后，需再次与路由决策矩阵进行乘积运算。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F86020d75-c881-4418-82a6-a228f091abe8_808x844.png)

路由决策机制在文本标记层级调控输入数据，并在专家系统层级调控输出结果。

由此我们得到经过处理的**柔性**图块/文本标记，而非原始的离散输入形式。

`MoE`架构的独特价值主要体现在其创新的计算资源配置策略。由于在任意时刻仅调用部分专家集群，实际可调用的模型参数规模远超即时使用需求。

虽然`MoE`需要加载全部参数（稀疏化参数），但推理过程仅激活部分专家单元（激活参数），显著降低实际计算开销。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe1fd47bb-9ced-42e4-8f6c-536f7a65fbf7_1376x1252.png)

简而言之，尽管需要将完整模型（包含所有专家单元）加载至设备（稀疏化参数），但实际推理过程仅需调用部分参数子集（激活参数）。 混合专家模型需要更多显存来加载全部专家集群，但在推理过程中运算效率更高。

我们以`Mixtral 8x7B`模型为例，具体解析稀疏参数与激活参数的量化对比。[9](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts#footnote-9-148217245)

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc3d48d5-8afc-4477-af98-5817b1a145ae_1376x988.png)

此处可见每个专家单元的参数规模为**5.6B**而非`7B`（尽管包含`8`个专家单元）。

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1dfd20b4-d3b7-433b-8072-2e67fc70afaa_1376x544.png)

模型需要 _加载_ **8×5.6B（ **_46.7B_** ）**参数（含全部共享参数），但实际推理仅需激活**2×5.6B（ **_12.8B_** ）**参数。

至此我们完成了对专家混合模型的全面解析！相信通过本文阐述，您能更深入理解这项创新技术的应用潜力。当前几乎所有主流模型集合都包含至少一个混合专家变体，这标志着该技术已成为行业标准配置。

若想获取更多与大型语言模型相关的可视化解析，并支持本专栏持续创作，欢迎查阅笔者编著的《**大型语言模型实战指南**》！

希望本文能为您提供关于混合专家系统的入门指引。若您希望深入研究，以下资源值得参考：

*   [这篇](https://arxiv.org/pdf/2209.01667) 与[这篇](https://arxiv.org/pdf/2407.06204) 论文详尽阐述了混合专家系统领域的最新突破。
    
*   关于[专家选择路由机制](https://arxiv.org/pdf/2202.09368) 的研究论文已在学术界引起广泛讨论。
    
*   [这篇优质博客](https://cameronrwolfe.substack.com/p/conditional-computation-the-birth) 深入剖析了多篇关键论文及其研究成果。
    
*   同类[技术解析博客](https://brunomaga.github.io/Mixture-of-Experts) 则完整呈现了混合专家系统的演进历程。