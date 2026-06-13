# DeepSeek V4 Flash on DGX Spark 开发历史压缩版

本文是原始开发日志的压缩版，保留关键决策、实验结果、失败路径和最终可复现方案。

完整原始日志已归档到：

- `DevHistory_backup/DevHistory.md`
- `DevHistory_backup/DevHistory2.md`

当前仓库对应 GitHub：

```text
https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark
```

## 0. 当前结论

目标是在两台 NVIDIA DGX Spark 上运行 DeepSeek V4 Flash 280B FP4。

最终推荐方案：

- 双机 TP=2。
- vLLM 使用 jasl `ds4-sm120` 系列 fork。
- MoE 后端使用混合路径：
  - layers 0-41 走 Marlin。
  - layer 42 走 Consumer-DeepGEMM 的 Triton `tl.dot_scaled` 路径。
- 启动参数保留：
  - `--moe-backend marlin`
  - `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42`
  - `--kv-cache-dtype fp8`
  - `--gpu-memory-utilization 0.80`
  - `--max-model-len 4096`
  - `--enforce-eager`

实测状态：

| 方案 | 中文 | 英文 | 速度 | 状态 |
|---|---|---|---:|---|
| jasl/vLLM 纯 Marlin | 正常 | 乱码 | ~14 tok/s | 不可作为通用方案 |
| Consumer-DeepGEMM 全层 | 正常 | 正常 | ~4.1 tok/s | 正确但慢 |
| 混合后端 TP=2 | 正常 | 正常 | ~12 tok/s | 当前推荐 |
| 混合后端 PP=2 | 正常 | 正常 | ~10-11 tok/s | 可用，但不是默认推荐 |

## 1. 硬件和基础环境

硬件：

- DGX Spark x2。
- 每台 128GB 统一内存。
- GPU 为 GB10，架构 `sm_121`。
- 两台通过 ConnectX-7 200Gbps RoCE 直连。
- 实验中 host 为 `169.254.248.35`，worker/slave 为 `169.254.30.81`。

模型：

- DeepSeek V4 Flash 280B FP4。
- 权重约 158-160GB。
- 单台 Spark 放不下完整模型，必须双机。

重要认知：

- NVIDIA 宣传的“256GB 共享内存”不是一台机器里的透明统一内存。
- 实际是两台独立主机，各 128GB，通过网络组成分布式推理。
- vLLM 通过 Ray + NCCL 管理两台机器。
- 模型权重和 KV cache 都要按并行策略分布到两台机器上。

## 2. 早期路线：先跑通，再修正确性

最早目标是先让 DeepSeek V4 Flash 在双 Spark 上跑起来。

遇到的问题：

- 官方 vLLM main 对 `sm_120/sm_121` 的 MXFP4 MoE 支持不足。
- NGC 基础镜像里的 `transformers` 太旧，不认识 `deepseek_v4`。
- Docker BuildKit 内部 clone GitHub 不稳定，因为宿主 TUN 代理不覆盖 BuildKit。
- FlashInfer cubin 下载数量巨大，构建时间不可接受。
- `max_model_len=131072` 时容易在 Marlin 权重加载阶段卡死或 OOM。

处理方式：

- 使用 `spark-vllm-docker` 作为双机部署基础。
- 预先在宿主机 clone 依赖，再在 Dockerfile 中 `COPY` 进去。
- 跳过 FlashInfer 预编译 cubin 下载，改走 JIT。
- 编译后升级 `transformers`，并同步镜像到 slave。
- 初期统一用 `--max-model-len 4096` 降低内存压力。
- `--gpu-memory-utilization` 从 0.85 降到 0.80，避免 worker swap/OOM。

## 3. Consumer-DeepGEMM 路线

为了绕开官方/Marlin 在 SM120+ 上的 MXFP4 问题，尝试了 Consumer-DeepGEMM。

Consumer-DeepGEMM 是本项目使用的 SM120+ Triton FP4 MoE 路径，不是上游原版 DeepGEMM。核心是用 Triton `tl.dot_scaled` 实现 FP8 activation x FP4 weight 的 fused MoE 计算。

优化过程：

- 从最初 CUTLASS grouped GEMM 约 `0.79 tok/s` 开始。
- 改成 Triton dequant + cuBLAS，约 `1.67 tok/s`。
- 做 fused dequant + matmul，约 `1.87 tok/s`。
- 使用 `tl.dot_scaled`，再去掉不必要 FP8 转换后达到约 `4.1 tok/s`。

结论：

- Consumer-DeepGEMM 全层输出正确，中英文都干净。
- 速度明显慢于 Marlin。
- 它适合作为正确性兜底路径，但不适合作为全层最终高性能路径。

## 4. Marlin 路线和 sm_121 问题

jasl/vLLM fork 的纯 Marlin 路径在 DGX Spark 上速度快，约 `14 tok/s`，但英文输出乱码。

最初怀疑是 Marlin 的 `ldmatrix` 数据布局在 `sm_121` 上和 `sm_80/sm_90` 不一致。

验证过程：

- 写 CUDA 探针观察 `ldmatrix.sync.aligned.m8n8.x4.shared.b16` 在线程寄存器中的实际布局。
- 确认 `sm_121` 上的行为与 Marlin 假设的旧架构布局不同。
- 尝试用 `__shfl_sync` 做寄存器补偿。
- 在隔离测试中，A/B operand 的小矩阵和部分共享内存布局可以对齐参考结果。

但接入完整模型后仍然乱码。

原因是 Marlin 内部不是只有 `ldmatrix` 一步：

```text
checkpoint weight
  -> gptq_marlin_repack
  -> shared memory swizzle
  -> ldmatrix
  -> dequant
  -> mma
```

这几步在旧架构上互相抵消，换到 `sm_121` 后单点修 `ldmatrix` 不够，必须整体重新匹配数据布局。继续修 Marlin 底层代价过高。

## 5. 排除底层和上层：不是单点 bug

为了确认乱码来源，做了大量探针。

底层 kernel 排除：

- 单层 Marlin GEMM 对 BF16 dequant reference，误差约 `1e-4` 到 `2e-4`。
- vLLM 风格 TP 分片后，rank0/rank1 分别跑 Marlin，再相加，对完整 BF16 reference 通过。
- 真实 checkpoint MXFP4 权重的 TP partition 探针通过。
- expert map / rank1 local experts / mixed valid-invalid top-k 路由探针通过。

上层调度排除：

- DeepSeek V4 的 hash 路由查表一致。
- 路由缩放系数确认。
- routed experts + shared experts 的合并顺序确认。
- 单层 FusedMoE 路径与手工拆分参考对齐。

结论：

- 不是某个 expert 权重坏。
- 不是某个 rank 的 TP 切片错。
- 不是路由编号映射错。
- 不是单层 Marlin 计算完全错误。
- 更像是全模型 43 层级联后的数值误差累积。

## 6. 关键发现：42 层 Marlin + 最后一层 Consumer-DeepGEMM

使用按层二分验证 Marlin 可承受范围。

新增调试环境变量：

```text
VLLM_MXFP4_MARLIN_LAYER_RANGE=start:end
```

结论：

| Marlin 层范围 | 输出 |
|---|---|
| `0:0`，全 Consumer-DeepGEMM | 正常 |
| `0:10` | 正常 |
| `0:20` | 正常 |
| `0:30` | 正常 |
| `0:42`，前 42 层 Marlin | 正常 |
| `0:43`，全 43 层 Marlin | 乱码 |

进一步测试：

- `42:43` 仅最后一层 Marlin，正常。
- `40:43` 最后三层 Marlin，正常。
- `36:43` 最后七层 Marlin，正常。

解释：

- 不是 layer 42 自身坏。
- 是每层 Marlin 在 `sm_121` 上存在小数值偏差。
- 这些偏差单层可接受，但 43 层串起来后在 logits 前越过临界点。
- 只要最后一层改用 Consumer-DeepGEMM，就能在最终 logits 前兜住误差。

最终方案：

```text
layers 0-41: Marlin
layer 42: Consumer-DeepGEMM
```

启动仍使用：

```text
--moe-backend marlin
VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42
```

注意：这里的 fallback 走的是 Consumer-DeepGEMM 的 `tl.dot_scaled` 路径。

## 7. 过程中修过的真实 bug

这些 bug 不一定单独解释全模型乱码，但都是实际问题，最终混合方案需要它们稳定存在。

### 7.1 TP padding source offset bug

文件：

```text
vllm/model_executor/layers/fused_moe/layer.py
```

问题：

- DeepSeek V4 MoE intermediate size 为 2048。
- TP=2 后每 rank 应加载 1024。
- Marlin 为 kernel 对齐会把目标 buffer pad 到 1536。
- 原逻辑用 padded size 计算 checkpoint source offset，导致 rank1 从错误位置读权重。

修复：

- checkpoint source offset 使用未 padding 的真实 shard size。
- padded size 只用于目标 buffer 内部布局。

### 7.2 swiglu_limit 硬编码

文件：

```text
vllm/model_executor/layers/quantization/mxfp4.py
```

问题：

- DeepSeek V4 Flash config 中 `swiglu_limit=10.0`。
- Marlin 路径硬编码为 `7.0`。
- 会错误截断 7 到 10 之间的激活值。

修复：

- 从 layer/config 读取真实 `swiglu_limit`。

### 7.3 Marlin 输出 buffer 与 workspace 复用

文件：

```text
vllm/model_executor/layers/fused_moe/fused_marlin_moe.py
```

问题：

- `fused_out` 和 `workspace13` 可能共享同一块内存。
- 存在中间缓冲与最终输出读写重叠风险。

修复：

- Marlin 自己分配独立输出 buffer，算完后再拷贝回目标输出。

## 8. 最终 TP=2 推荐启动方式

推荐直接使用预构建镜像：

```bash
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest
```

启动：

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
VLLM_SPARK_EXTRA_DOCKER_ARGS="\
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42 \
" \
./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t lmxxf/vllm-deepseek-v4-dgx-spark exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --enforce-eager \
  --moe-backend marlin
```

预期日志：

```text
Using 'MARLIN' Mxfp4 MoE backend.
MXFP4 layer 42 forced to DEEPGEMM_MXFP4 by VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42
```

停止：

```bash
./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t lmxxf/vllm-deepseek-v4-dgx-spark stop
```

## 9. TP=2 实测

混合后端 TP=2 的典型结果：

| Prompt | 输出 | 速度 |
|---|---|---:|
| `17 * 23 = ?` | `391` | 亚秒级，warmup 后 |
| 中文长城介绍 | 正常中文 | ~12 tok/s |
| English quicksort | 正常英文和代码 | ~12 tok/s |
| Python 代码生成 | 正常 | ~12 tok/s |
| 苹果推理题 | `5` | 正常 |

与早期方案对比：

| 方案 | 速度 | 结果 |
|---|---:|---|
| CUTLASS grouped GEMM | ~0.79 tok/s | 正确但慢 |
| Triton dequant + cuBLAS | ~1.67 tok/s | 正确但慢 |
| Consumer-DeepGEMM fused | ~1.87 tok/s | 正确但慢 |
| Consumer-DeepGEMM `tl.dot_scaled` 优化 | ~4.1 tok/s | 正确 |
| jasl 纯 Marlin | ~14 tok/s | 英文乱码 |
| 42 层 Marlin + layer42 Consumer-DeepGEMM | ~12 tok/s | 正确，当前推荐 |

## 10. PP=2 流水线并行实验

后来尝试把双机从 TP=2 改为 PP=2：

```text
--tensor-parallel-size 1
--pipeline-parallel-size 2
```

vLLM 能识别 DeepSeek V4 的 `SupportsPP`，会把 43 层切成两个 pipeline stage：

```text
rank0: 22 layers
rank1: 21 layers
```

显存观察：

- PP rank0 模型权重约 74.8 GiB。
- PP rank1 模型权重约 71.9 GiB。
- 与 TP=2 接近。

原因：

- TP=2 是 43 层都在，但每层权重张量切半。
- PP=2 是每台只有一半层，但每层权重完整。
- 两者每台常驻权重大致都是半个模型。

## 11. PP=2 的两个问题和修复

### 11.1 默认 Ray PP 路径第二请求卡死

默认 Ray executor 在 PP 下走：

```text
Ray compiled graph + RayPPCommunicator
```

表现：

- 第一个请求可能返回。
- 第二个请求卡住。
- `/health` 也可能超时。

修复：

```text
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
```

生效日志特征：

```text
Asynchronous scheduling is enabled
RayWorkerProc
Worker_PP0
Worker_PP1
```

### 11.2 layer42 fallback 在 PP 路径打不中

TP 路径下，layer name 匹配原正则：

```text
\.layers\.(\d+)\.ffn\.experts$
```

PP 路径中实际可能是：

```text
model.layers.42.ffn
model.layers.42.ffn.experts
```

旧正则过窄，导致：

- `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42` 已传入。
- 但 layer 42 没有真正回退。
- 最后一层继续走 Marlin。
- 输出又回到全 Marlin 乱码。

修复：

```text
(?:^|\.)layers\.(\d+)\.ffn(?:\.experts)?$
```

同时添加调试开关：

```text
VLLM_MXFP4_DEBUG_LAYER_NAMES=1
```

确认日志：

```text
MXFP4 layer backend check layer_name='model.layers.42.ffn.experts' layer_idx=42 backend=MARLIN force_deepgemm='42'
MXFP4 layer 42 forced to DEEPGEMM_MXFP4 by VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42
```

## 12. PP 修复镜像

PP 修复不需要重编完整大镜像。主仓库提供脚本：

```text
scripts/build-pp-fix-image.sh
```

它基于已上传镜像：

```text
lmxxf/vllm-deepseek-v4-dgx-spark:latest
```

在派生镜像里 patch 容器内：

```text
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py
```

生成：

```text
vllm-deepseek-v4-pp-fix:latest
```

使用：

```bash
./scripts/build-pp-fix-image.sh --pull
./scripts/build-pp-fix-image.sh --copy-to 169.254.30.81
```

## 13. PP=2 启动方式

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
VLLM_SPARK_EXTRA_DOCKER_ARGS="\
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42 \
  -e VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 \
" \
./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t vllm-deepseek-v4-pp-fix exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --enforce-eager \
  --moe-backend marlin
```

PP=2 验证结果：

| 测试 | 输出 | 耗时/速度 |
|---|---|---:|
| 第一次 `17 * 23` | `391` | 28.39s，含 warmup |
| 第二/三次 `17 * 23` | `391` | 0.34s |
| Python `sum_even` | 正确代码 | 2-3s |
| 中文长城 | 正常中文 | ~10-11 tok/s |
| English quicksort | 正常英文和代码 | ~10-11 tok/s |

结论：

- PP=2 可用。
- 但单请求长输出不如 TP=2。
- 它是 PP 调优基线，不是当前推荐部署。

## 14. 为什么 PP 在 DGX Spark 上不更快

PP 理论上减少跨机器通信：

- TP=2 每层都要 all-reduce。
- PP=2 只在 stage 之间传 hidden states。

但 PP=2 在单请求 decode 下有流水线气泡：

```text
生成第 N 个 token:
  rank0 跑 layers 0-21
  rank1 等 rank0，然后跑 layers 22-42

生成第 N+1 个 token:
  重复同样串行流程
```

单请求场景下 pipeline 填不满。

DGX Spark 还有一个特殊点：

```text
网线虽慢，内存更慢。
```

说明：

- DGX Spark 的 GB10 统一内存容量大，但带宽不是 H100/H200 的 HBM。
- 很多时间本来就在等本地内存喂权重。
- CX7/RoCE 200Gbps 当然比 HBM 慢，但相对 DGX Spark 的本地内存瓶颈，没有慢到完全主导。
- PP 省掉的网线通信，不够抵消单请求 pipeline bubble 和 stage 串行。

更一般的判断：

```text
PP 只有在“网络瓶颈远大于本地内存瓶颈”时才更可能明显收益。
```

例如多台 Mac/家用机器通过 USB 或普通网络拼推理时，网络可能远慢于本地统一内存。那种情况下 TP 每层 all-reduce 会被链路拖死，PP 少通信可能更有意义。

但双 DGX Spark + CX7/RoCE 不是这个形态。

## 15. PP 的定位

PP 不应写成生产推荐方案。

生产级在线推理如果有足够多的数据中心 GPU 和高速互联，通常更倾向：

- TP：Tensor Parallel。
- EP：Expert Parallel。
- DP：Data Parallel。

这些组合更适合吞吐、并发、调度效率和集群利用率。

PP 在大模型训练中常见，但在线单请求 decode 容易被流水线气泡拖住。

本项目测试 PP 的意义是：

- 面向家用/桌面大内存机器的工程基线。
- 验证“能不能把模型按层拆开塞进去”。
- 为后续高并发/大 batch/microbatch 调优留入口。

当前双 Spark 单请求推理仍推荐 TP=2 mixed backend。

## 16. 当前文件和脚本

重要文件：

| 文件 | 作用 |
|---|---|
| `README.md` | 英文部署入口 |
| `README_CN.md` | 中文部署入口 |
| `BUILD-AND-RUN.md` | 当前构建和启动命令 |
| `DevHistory.md` | 本压缩版开发历史 |
| `DevHistory_backup/DevHistory.md` | 原始长日志 1 |
| `DevHistory_backup/DevHistory2.md` | 原始长日志 2 |
| `scripts/build-pp-fix-image.sh` | 生成 PP 修复镜像 |

注意：

- `spark-vllm-docker/` 是嵌套仓库/构建目录，主仓库 `.gitignore` 会忽略它。
- PP 修复入口必须放在主仓库 `scripts/` 下，不能藏在 `spark-vllm-docker/` 里。

## 17. 后续可能方向

可以继续研究，但不影响当前可用部署：

- 高并发下 PP=2 是否能通过 microbatch 填满 pipeline。
- 更大 `max_model_len` 下 Marlin 权重加载和内存 slack 问题。
- 是否能彻底修复 Marlin 在 `sm_121` 上的全 43 层数值漂移。
- 是否能把 Consumer-DeepGEMM 的调度从 Python 循环进一步融合。
- 更系统地比较 TP/PP 在 Mac/USB/普通网络环境下的收益边界。

## 18. Marlin 全 43 层乱码根因排查（进行中，2026-06-12 起）

针对 §17 第三条的根因排查。

### 已证伪（全部本地单 GPU 实验）

1. **Marlin kernel 数值脏** ❌ — 新探针 `test/test_marlin_err_structure.py`：fp32 金标准 + 幅度扫描（×0.1/×1/×4/outlier±30）+ m∈{1,2,16,256}，Marlin 误差 ≡ bf16 舍入（excess_ratio≈1.0，bad_cols=0/4096）。jasl 的两个 sm_120 补偿（`marlin_template.h` L99 ldsm lane shuffle、L1374 B nibble 重排）是干净的。
2. **fp16 global reduce** ❌ — MoE 路径硬编码 `use_fp32_reduce=True, use_atomic_add=False`。
3. **swiglu clamp 语义** ❌ — 官方 `deepseek-v4-flash/inference/model.py` Expert.forward 确认 clamp（gate max=10，up ±10）是模型本尊语义，Marlin 的 `swiglu_limit_func`（`fused_moe/utils.py:391`，仅 Marlin 路径调用）忠实实现。CDG 路径反而**不 clamp**（默认 FLOAT32 scale fmt → colmajor 分支无 clamp）却输出干净 → clamp 很少触发，纯保险丝。
4. **W4A8-FP8 激活路径** ❌ — `VLLM_MARLIN_INPUT_DTYPE` 未设，部署走 bf16 激活（W4A16），探针测的就是部署 kernel。
5. **E8M0 scale 溢出** ❌ — `test/scan_scales.py` 扫全 43 层 33024 个 scale 张量：全部 118~126（2^-9..2^-1），无悬崖。

### 核心推理

CDG 把中间激活量化 FP8（~3%/元素噪声）跑 43 层干净；Marlin bf16 激活（~0.2%）却乱码。
**→ §5 的"43 层数值误差累积"解释站不住：若是噪声累积，FP8 应先乱。真凶是系统性偏差，且只在真实激活/真实路由下发火（随机输入探针全绿）。**
旁证：乱码挑语言（中文好英文乱）= 偏差打在特定 expert/通道上，随机噪声不挑语言。

### 剩余嫌疑（需真实激活区分）

- 真实激活的 outlier 方向 × 权重列对齐（随机探针摸不到）。
- 真实 top-6 非均匀路由 / 256 expert 满配（探针只测了 8 expert 均匀路由）。
- TP=2 交互（探针单 GPU；但 bisect 时 TP=2 下 42 层 Marlin 也干净，嫌疑低）。
- 官方参考用 fp32 算 silu（"for stability"），Marlin 路径在 bf16 算 — 单独不致命（FP8 act 都没事），可能与上面叠加。

### 下一步：抓真实激活回放

- 镜像 `vllm-deepseek-v4-act-dump:latest` 已构建+语法验证。hook = 镜像内 mxfp4.py 末尾 monkeypatch `Mxfp4MoEMethod.apply/apply_monolithic`，env `VLLM_MXFP4_DUMP_DIR` 控制，rank0 only，每层 max 8 份，存 x/topk/out。构建脚本 `scripts/build-act-dump-image.sh`。
- 抓取脚本 `test/run_act_dump.sh`：`sync`（拷镜像到 slave）→ `serve`（混合后端起服务）→ `probe`（英文 prompt，dump 落 `debug_acts/`）。
- 抓到后本地回放：逐层 真实激活 → Marlin(带clamp) vs CDG 语义 vs fp32 官方 ref，找系统性分歧的层/expert/通道。回放脚本待写（等 dump 数据定 shape）。

### 事故记录

- 2026-06-12 白天：两台 Spark 同时僵死（早上还好，晚上回家全挂，拔电重启）。当天只在 host 本地干活（docker build、单 GPU 探针、磁盘扫描），slave 未碰，抓取脚本未跑过，`debug_acts/` 为空。原因不明——嫌疑：共用电源 / GB10 驱动 / RoCE 链路平台级问题。若 serve 时复现双挂，按平台级 bug 另查。

### 2026-06-13 进展：dump 抓到

- 双机服务起来正常，复现英文乱码（"Write a quicksort..." → "The function should take... Provide a step-by-step..." 把 prompt 续写当自言自语）。
- 抓到 344 个 dump 文件（43 层 × 8 帧），2.9 GiB，落在 `debug_acts/`。
- 首次抓取后发现 hook bug：`args=[]`，topk 信息丢失（vLLM 实际用 kwargs 调 `apply`）。已修 `scripts/build-act-dump-image.sh`，新 hook 同时存 `*args` 和 `**kwargs`，重抓后字段齐全。
- dump 字段确认（layer 0/21/42 frame 0，TP=2 rank0 切片，hidden 4096，intermediate 1024）：
  - `x`: `(2048, 4096) bf16` —— MoE 输入。
  - `kwargs.topk_weights`: `(2048, 6) fp32`，`absmax≈1.0~1.5`。
  - `kwargs.topk_ids`: `(2048, 6) int32`，rank0 是 256 中的一片 expert id（min/max 落在 [23, 254] 范围）。
  - `kwargs.shared_experts_input`: 与 `x` 完全相同（同一 tensor，回放时无需单独管）。
  - `out`: `(2048, 4096) bf16`。

### 意外发现：layer 42 输出 absmax 爆炸

| 层 | `x` absmax | `out` absmax |
|---:|---:|---:|
| 0 | 0.90 | 22.6 |
| 21 | 1.41 | 31.8 |
| 42 | **5.41** | **9344** |

- `x` 沿层数单调上涨（0.9 → 1.4 → 5.4），MoE 输入越来越极端。
- layer 42 输出 `absmax=9344`，比正常层（22~32）大 ~300 倍。
- 注意当前是混合后端配置（前 42 层 Marlin + layer 42 fallback 到 CDG），所以 layer 42 这个极端 out 是 **CDG 跑出来的**，但输入 x 已经被前 42 层 Marlin "污染"。
- 这是新信号：sm_121 上的 Marlin 偏差不是均匀小漂移，而是把某些通道推到 5σ 之外，被 CDG 在最后一层兜回正常 logits 范围才让最终输出能用。
- 旁证 §6 的混合方案为何"恰好 work"：不是 CDG 计算更准，是 CDG 没有 Marlin 在 sm_121 上的系统性偏差，能消化掉前序累积的极端 x。

### 下一步：双配置 bisect dump 对比（推迟到下次）

不急着写完整 Marlin-vs-CDG 回放（要重新加载 MXFP4 权重 + 调 fused_marlin_moe + 调 CDG kernel，工作量大）。先做更快的实验：

- 启动 `VLLM_MXFP4_MARLIN_LAYER_RANGE=0:42` 抓一次（混合，正常输出）。
- 启动 `VLLM_MXFP4_MARLIN_LAYER_RANGE=0:43` 抓一次（全 Marlin，乱码）。
- 同一 prompt，逐层对比两次 dump 的 `x` 和 `out`：
  - 如果两次的 layer N 输入 `x` 已经分叉 → 污染来自 layer N-1 的 Marlin。
  - 如果 `x` 一致但 `out` 分叉 → layer N 的 Marlin 自己在真实激活上偏。
- 沿层向下追到第一个分叉点 → 真凶层。
- 在真凶层上才需要写完整 Marlin/CDG kernel 回放对比。

### 文件清单

- `scripts/build-act-dump-image.sh` —— hook 装载脚本（已修 kwargs）。
- `test/run_act_dump.sh` —— 三段式：`sync | serve | probe`。
- `test/replay_marlin_vs_cdg.py` —— 回放骨架（目前只有 dump 结构 inspect，核心 kernel 对比待写）。
- `debug_acts/` —— 抓到的 dump（344 文件，2.9 GiB，未进 git）。

