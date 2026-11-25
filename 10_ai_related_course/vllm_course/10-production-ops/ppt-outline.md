# 第 10 讲：生产运维与生态 — PPT 大纲

## 设计说明

- **总页数**：约 22 页
- **演讲时长**：120 分钟（nano 回顾 10 min + server 运维面 15 min + metrics 15 min + bench 15 min + FT 与生态 20 min + 实验 25 min + 收官 20 min）
- **核心视觉**：请求旅程全景收官图（第 19 页，十讲机制在生产面的对应物）；生态三层图（第 15 页，引擎/部署栈/控制面）；生产十条清单（第 20 页，可直接贴运维手册）
- **教学策略**：收官讲以「合流」为主——每个环节都点名它是前九讲哪一讲的「生产对应物」；生态三层用 arXiv 综述收束；实验走完整「压测→读面板」闭环
- **配色**：运维面蓝色系、metrics 绿色系、FT 红色系、生态金色系；收官页用课程全色回顾条
- **边界**：K8s 部署工程细节(HPA/PDB)不展开；llm-d/AIBrix 内部只讲定位与分工；精度方法论引用方法论课

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：生产运维与生态
- **副标题**：vLLM 十讲 · 第 10 讲（收官）
- **基线信息**：nano-vllm `b9dd55a` + vLLM `2f5d8a2b1` | 核对日期 2026-11-24
- **一张图**：课程全十讲图标环绕一圈，中心是「生产」

### 第 2 页：本讲核心问题与议程

- **核心问题**：一个引擎变成一组服务，K8s、网关、控制面怎么接？前九讲的机制，生产上各对应哪条命令、哪个端点、哪张面板？
- **五段议程**：nano 回顾（10 min，第 3 页）| server 运维面（15 min，第 4–6 页）| metrics + bench（30 min，第 7–12 页）| FT 与生态（20 min，第 13–17 页）| 实验（25 min，第 20 页）+ 收官（第 21–22 页）

## 第一部分：nano 回顾（10 min，第 3 页）

### 第 3 页：nano 回顾：运维面四缺

- **上稿**：nano 的「运维」= bench.py 一个脚本；缺 server 端点、缺 metrics、缺健康检查、缺故障恢复
- **口播**：「生产引擎的四样东西，前九讲的机制都需要它们的对应物——本讲收拢」

## 第二部分：server 运维面（15 min，第 4–6 页）

### 第 4 页：运维端点四件

- **一张表**（蓝）：/health（引擎死→503，`health.py:22-33`）| /metrics（`metrics.py:56-81`，ASGI mount）| /version | /load（需 `--enable-server-load-tracking`）
- **红色陷阱框**：/ping 不是通用端点（仅 SageMaker 适配，`sagemaker/api_router.py:117`）；路由已迁 `serve/instrumentator/`

### 第 5 页：中间件洋葱与取消

- **一张图**：洋葱层次（CORS → 认证 → Request-Id → `elastic_ep` → 调试日志 → 自定义 `--middleware`）
- **上稿**：**无限流中间件**（限流靠网关；引擎侧 `vllm:admission_rejections`，`loggers.py:861`）；断连监听 `listen_for_disconnect`（`api_utils.py:53-111`，不能用 `is_disconnected`）→ abort（`async_llm.py:818-826`）

### 第 6 页：多 API server 与 headless

- **上稿**：`--api-server-count N`（一引擎组 N 前端）；`--headless`（完全不起 HTTP，`cli/serve.py:69-77`，交控制面）；与第 6 讲 DP 部署的组合（`--api-server-count=8` 的 Wide-EP 命令回看）

## 第三部分：metrics + bench（30 min，第 7–12 页）

### 第 7 页：metrics 双出口与数据流

- **一张图**：Scheduler 每步产 `SchedulerStats`（`stats.py:224-262`）→ EngineCoreOutputs → client 侧 `StatLoggerManager`（`loggers.py:1372`）→ 日志出口 + Prometheus 出口

### 第 8 页：指标族归类表

- **一张大表**（绿；README 同款）：调度/吞吐/缓存/延迟/外挂五族，每族代表指标带行号
- **重点行**：`cached_by_source`（`:795-807`，第 3/9 讲的来源标签在此合流）；`request_num_preemptions`（`:1024`，第 2 讲抢占的生产可见面）

### 第 9 页：ITL vs TPOT

- **一张时序图**：流式输出逐 token 记 ITL;TPOT=(e2e−TTFT)/(n−1) 每请求一次；投机解码一步多 token 时分叉（`metrics.md:54-55、74-80`）
- **上稿**：bench 的 TPOT 排除 ≤1 token 请求，直方图记 0——两口径分叉的另一个来源

### 第 10 页：vllm bench 七子命令

- **一张表**：serve / throughput / latency / forward / startup / sweep / mm-processor（`vllm/entrypoints/cli/benchmark/`）；`--dataset-name` 13 种（`datasets.py:1614-1631`）

### 第 11 页：输出字段与互补关系

- **一张表**（`serve.py:352-393`）：throughput + TTFT/TPOT/ITL/E2EL 均值与百分位
- **一张对照图**：bench（黑盒注入负载）vs metrics（白盒自报告）；TPOT 口径分叉再次点名
- **闭环行**：`spec_bench` + `vllm:spec_decode*` 抓取 = 第 7 讲的收益验证（回收）

### 第 12 页：文档滞后两处

- **红色表**：`prompt_tokens_total` 旧名（现 `prompt_tokens`）；`num_requests_swapped` 已删——以 `docs/usage/metrics.md` 为准

## 第四部分：FT 与生态（20 min，第 13–17 页）

### 第 13 页：EngineCoreSentinel 显式协议

- **一张状态机图**：HEALTHY → DIAGNOSING → UNHEALTHY → DEAD（`engine_core_sentinel.py:55`；`on_fault` `:142-194`：清脏块 `:231-255` → 抢占或 finish）
- **上稿**：恢复由外部 orchestrator 发 `POST /v1/fault_tolerance/apply`(retry|`scale_down`)+ `GET /status`（`api_router.py:41-103`）；retry 走幸存引擎 gloo 投票（`:469-511`，MIN 归约）
- **红色结论框**：**不是自动自愈**——worker 死即整 executor shutdown（`start_worker_monitor`，`multiproc_executor.py:389-424`；注意第 1 讲引用的 :287 已漂移），上层 K8s 重拉

### 第 14 页：FT 接口与 API 层反应

- **上稿**：FT 端点两件(apply/status);API 层 `validate_alive`（`core_client.py:617-620`）看到 `ENGINE_CORE_DEAD` → EngineDeadError → /health 503;client 侧 FT 簿记（`:1496-1531`）
- **口播**：「引擎暴露原料（状态/事件/接口），控制面做决策」——下一页的三层图由此而来

### 第 15 页：生态三层图（金色）

- **一张大图**：引擎层（vLLM，本课 1-9 讲）→ 部署栈层（production-stack,Helm+router，轻量参考；社区有迁 llm-d 迹象）→ 控制面层（AIBrix 字节，管到 replica 级 / llm-d Red Hat+CNCF Sandbox,well-lit paths + prefix-cache-aware routing）
- **每层一行**：production-stack=K8s 参考部署栈；AIBrix=控制面全家桶(autoscaler/router/gateway);llm-d=分布式推理框架（**读 vLLM KV 事件做前缀感知路由**——第 3/9 讲 KV 事件的消费方式）
- **口播**：「第 3/9 讲你写的 KV 事件，在 llm-d 里是路由器的输入」

### 第 16 页：arXiv 综述收束

- **上稿**：arXiv:2609.23130（2026-09；**单作者综述，非官方立场**）：推理从「引擎内优化」变成「分布式控制问题」；稀缺资源从 FLOPs 转向「被管理的状态、放置、网络搬运、可靠性」
- **呼应框**：第 2 讲调度 = 引擎内放置；第 9 讲 offload = 状态管理；第 6 讲 EP = 异构加速器——前九讲在综述的大图里各有位置

### 第 17 页：外部课与资料

- **上稿**：DeepLearning.AI 短课（2026-06-03 上线；9 节视频+3 lab;Compress/Serve/Benchmark 三阶段；与第 8 讲量化、第 10 讲压测互补）；vLLM 官方 docs 索引(docs.vllm.ai)

## 第五部分：实验与收官（第 18–22 页）

### 第 18 页：本讲实验

- **三行实验卡**：实验 1 运维面巡检（端点+指标，纯 CPU 可做）；实验 2 基线压测（`vllm bench serve`）；实验 3（可选）故障演练（杀 worker 观察 503 与恢复）
- **产出要求**：巡检清单；bench 百分位表；故障时间线

### 第 19 页：请求旅程全景收官图

- **一张大图**（金色）：课程主页请求旅程图扩展——每个环节旁标注「生产对应物」（调度→waiting 指标；KV→`cached_by_source`；后端→启动日志 kernel 行；图→capture 日志；并行→进程公式；spec→accept length；量化→方法行；offload→host/disk 标签；运维→/health /metrics /bench）
- **口播**：「十讲的机制，一张图收齐」

### 第 20 页：生产十条清单

- **一张清单页**（可直接贴运维手册；README 同款十条，每条附一句机制回指）

### 第 21 页：课程总结

- **一张回顾表**：核心问题九连问与答案所在讲（批怎么拼/2;KV 放哪/3;kernel 谁跑/4；为什么快/5；怎么扩/6；怎么多 token/7；怎么省显存/8；装不下/9；生产怎么盯/本讲）
- **口播**：回到第 1 讲核心问题——现在你能回答它在新问题下的所有变体

### 第 22 页：结课

- 结课作业（计考核）：生产部署方案，见 homework.md。

- **上稿**：结课信息；后续路径（KV 八讲纵深 / nano 仓课程 / 权重开源后的验证清单）；课程仓库链接
- **一条提醒**：基线 commit 与核对日期；换版本按「以代码为准」重核

---

## 图资产清单（自制）

| 图 | 页 | 说明 |
| --- | --- | --- |
| 全十讲环绕图 | 1 | 收官封面 |
| 运维端点四件表 | 4 | 蓝色；陷阱框红色 |
| 中间件洋葱 | 5 | 层次图 |
| metrics 双出口数据流 | 7 | Scheduler→Outputs→Loggers |
| 指标族大表 | 8 | 五族带行号 |
| ITL/TPOT 时序 | 9 | 投机解码分叉标注 |
| bench 输出表 | 11 | 百分位 |
| Sentinel 状态机 | 13 | 红色；四态 |
| 生态三层图 | 15 | 金色；引擎/部署栈/控制面 |
| 请求旅程收官全景 | 19 | 金色；十讲对应物 |
| 生产十条清单 | 20 | 直接贴运维手册 |
| 九连问回顾表 | 21 | 课程总结 |
