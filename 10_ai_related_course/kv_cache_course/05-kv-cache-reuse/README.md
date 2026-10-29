# 第 5 讲：KV Cache 怎么复用与分层

> 120 分钟 | 课程第 5 讲 | 前置：第 2–4 讲的 block 与引擎

第 3、4 讲在 block 和哈希层面做前缀复用：nano-vllm 用前缀哈希链让多个请求指向同一批 block，并带请求级 swap；vLLM 在显存不足时 preempt 单个请求，走 recompute，不是 swap。这两套机制都以 block 为单位，复用跟着块的生命周期走，没有回答“哪些前缀值得留、留不下时先扔谁、能不能放到 HBM 之外”。

本讲把这件事系统化：用 radix tree 管理前缀，讲清命中条件、引用保护和驱逐策略，再往下一层存储走。相同前缀怎么在树里命中，命中后靠什么字段保护，满了以后驱逐谁，HBM 放不下时怎么备份到主机内存和存储，是这一讲的主线。

先分开两个容易混的概念。nano-vllm 的请求级 swap 是调度器把一个运行中请求的 KV 整体搬到 CPU，等它重新调度时再搬回来，搬的是这个请求自己的块；vLLM V1 没有这条 swap 路径，它的 preemption 是 recompute。HiCache 的 host tier 是按前缀保存 KV 副本，供之后任意请求命中，命中后仍要回载到 HBM 才能参与计算。前者是请求维度的让路，后者是内容维度的缓存。

## 学习目标

- 说清 prefix caching 省的是哪部分计算，列出收益与代价。
- 区分 nano-vllm `ref_count`、vLLM `ref_cnt`、SGLang `lock_ref` 三套引用语义。
- 读懂 SGLang 默认的 `UnifiedRadixCache` / `UnifiedTreeCore`，知道经典 `RadixCache` 的用途。
- 用四状态模型判断节点能否被驱逐，走通 split、加锁、驱逐、解锁。
- 区分 HiCache 的 hit、loadback、backup、demote / evict 与 write policy。
- 说明 `host_value` 存的是什么，命中为什么要看 `extra_key`、`cache_salt` 和 `page_size`。

## 本讲产出

一张统一观察表，列为：请求、prompt tokens、device_hit、host_hit、storage_hit、实际 prefill tokens、TTFT、load_back ms、evicted tokens、备注。表后固定写三句：TTFT 变化、吞吐变化、是否值得换出。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)

## 前后衔接

第 3、4 讲解决了“块怎么分配、怎么在请求内复用”；本讲解决“块怎么跨请求复用、满了往哪放”。第 6 讲会把复用和分层带到 GLM-5.3-Flash 的混合注意力结构里，那时 `host_value` 不再只有 FULL 组件一份。
