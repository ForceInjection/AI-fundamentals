# 第 5 讲：KV Cache 怎么复用与分层 — PPT 提纲

每讲 10–12 页，只列页面主题和讲解落点；逐页讲稿待课程定稿后补。

1. 封面：从 block 级复用走到树级复用与分层
   落点：先分清 nano-vllm 的请求级 swap 和前缀缓存 host tier，再进入本讲问题。
2. 收益与代价：前缀缓存省的是 prefill
   落点：省前缀 QKV 和自注意力，不省 decode；代价是常驻内存、查找分裂和搬运。
3. 前缀树：匹配、分裂与默认实现
   落点：HEAD 默认走 `UnifiedRadixCache` / `UnifiedTreeCore`，经典 `RadixCache` 只用来建立直觉。
4. 引用语义：`ref_count`、`ref_cnt`、`lock_ref`
   落点：三套字段的粒度和累加规则不同，读代码时不要混用。
5. 节点状态：locked、evictable leaf、internal、evicted
   落点：只有 evictable leaf 能被驱逐，锁沿路径保护节点和祖先。
6. ABCD 例子：一次 split、加锁、驱逐、解锁
   落点：用四步看清 `lock_ref` 和 value 切片怎么变。
7. 命中条件：token、`extra_key`、`cache_salt`、`page_size`
   落点：同一条 prompt 换了 LoRA 或 salt 也不命中，page 对齐还会截掉尾部。
8. 驱逐：候选、策略与回退遍历
   落点：LRU/LFU/SLRU/priority/TLRU 是策略，叶到根的回退才是“分支”。
9. HiCache 边界：hit、loadback、backup、demote
   落点：命中不等于已在 device，backup 也不等于释放 device。
10. HiCache 数据流：一次 host 命中要走几步
    落点：storage prefetch、host staging、loadback、prefill，每一步都计入延迟。
11. `host_value` 与 `HostPoolGroup`
    落点：`host_value` 是 host pool 槽位索引，不是 DRAM 指针，也不覆盖 L3。
12. 实验与小结：三段实验，三句结论
    落点：CPU 模拟建立直觉，server 指标验证，强制驱逐看代价。
