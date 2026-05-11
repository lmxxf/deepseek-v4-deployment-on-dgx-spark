# DeepSeek V4 Flash / DGX Spark MXFP4 Marlin 调研报告

日期：2026-05-11  
环境：DGX Spark ×2，GB10，TP=2，vLLM `vllm-node-jasl-fix`

## 结论摘要

当前工程已经把 DeepSeek V4 Flash 跑到可用基线，但进一步提速的 true Marlin 路线仍卡在正确性。`jasl-fix + --moe-backend marlin` 早期看似可用，实际被 Python oracle 强制切回 `DEEPGEMM_MXFP4`，所以输出正常但不是 Marlin。真正启用 Marlin 后，4k 上下文能完整加载并生成，速度约 `4.2-4.4 tok/s`，但输出乱码；DeepGEMM 正常输出约 `3.57 tok/s`，作为正确性基线。

目前证据指向：Marlin 的底层 FP4 GEMM 和 TP/专家映射路径基本正确，错误更可能发生在完整 `FusedMoE` runner / `DeepseekV4MoE` 集成边界，而不是 `ldmatrix` 单点。

## 已完成修复

### SM120/121 Marlin 核修复

已在 `_moe_C.abi3.so` 对 Marlin MXFP4 路径做了两类修正：

- `ldmatrix.x4` 在 SM120/121 上的 lane shuffle 补偿。
- FP4 B operand nibble reorder，修正 MMA fragment 布局。

相关源文件：

- `spark-vllm-docker/vllm-sm120/vllm-sm120/csrc/moe/marlin_moe_wna16/marlin_template.h`
- 产物：`spark-vllm-docker/vllm-sm120/vllm-sm120/build/lib.linux-aarch64-cpython-312/vllm/_moe_C.abi3.so`

### DeepSeek V4 `swiglu_limit`

DeepSeek V4 Flash 的 `config.json` 中 `swiglu_limit=10.0`。原 MXFP4/Marlin 路径硬编码为 `7.0`，已改为读取 layer 配置：

```python
swiglu_limit = getattr(layer, "swiglu_limit", None)
```

这是实际 bug，但单独修复后 full-model Marlin 仍乱码。

### Marlin modular workspace alias

`MarlinExperts.apply()` 中 modular allocator 会让 `fused_out` 与 `workspace13` 共享 `common_workspace`。Marlin 又把 `workspace13` 作为 post-activation cache，直接把 final GEMM 写入 `fused_out` 有潜在输入/输出重叠风险。已改为先让 `fused_marlin_moe()` 自己分配输出，再 `output.copy_(moe_output)`。

这个修复能排除一类 alias 风险，但 full-model 输出仍乱码。

## 已验证通过的 Probe

### Raw Marlin GEMM

`test_real_mxfp4_marlin_probe.py`

用真实 checkpoint MXFP4 权重，单 rank / 非 TP，对 BF16 dequant reference。通过，误差约 `1e-4` 量级。

### TP 分片

`test_real_mxfp4_marlin_tp_probe.py`

模拟 vLLM 风格 TP：

- `w1/w3` 沿 intermediate 输出维切分。
- `w2` 沿 intermediate 输入维切分。
- rank0/rank1 Marlin 输出相加，对比 full BF16 reference。

通过，典型误差：

```text
max_abs 0.000244140625
mean_abs 2.3e-05
```

### Modular wrapper

`test_real_mxfp4_marlin_modular_probe.py`

比较 direct `fused_marlin_moe()` 与 `FusedMoEKernel + MoEPrepareAndFinalizeNoDPEPModular + MarlinExperts`。8 experts、256 experts、offset expert 场景均通过。

### Expert map / mixed routing

`test_real_mxfp4_marlin_expert_map_probe.py`

补上之前漏掉的真实 TP 路径：

- rank1 风格 local experts `128..255`
- `expert_map` 将 global expert id 映射为 local expert id
- mixed valid/invalid top-k，模拟真实 TP 中部分 expert 属于对端 rank
- layer 0 / layer 41 均通过

典型误差：

```text
max_abs 0.0001220703125
mean_abs 1.6e-05
```

因此可以基本排除 expert-map masking 和 mixed-rank routing。

## 已排除方向

当前证据基本排除以下方向作为主因：

- `ldmatrix` lane 布局。
- FP4 nibble reorder。
- 单层 raw Marlin GEMM。
- TP 权重切分。
- 256 experts / high expert id offset。
- `expert_map` global-to-local 映射。
- mixed valid/invalid top-k 清零。
- shared expert aux stream。设置 `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` 后仍乱码。
- CUDA graph。当前测试脚本使用 `--enforce-eager`，日志确认 CUDAGraph disabled。

## 当前现象

### DeepGEMM 基线

`jasl-fix` oracle 强制 `DEEPGEMM_MXFP4` 时输出正常：

```text
220 completion tokens / 61.58s = 3.57 tok/s
```

这不是 Marlin 提速路径，只是正确性基线。

### True Marlin 4k

挂载正常 oracle，真正使用 `--moe-backend marlin`：

```text
Using 'MARLIN' Mxfp4 MoE backend.
Using MoEPrepareAndFinalizeNoDPEPModular
```

模型可加载，可生成，但输出乱码：

```text
96 completion tokens / 21.84s = 4.40 tok/s
output: GPUs�名思义�后续GP后续...
```

速度有提升迹象，但正确性未过，不能算有效提速。

### True Marlin 131k

`max_model_len=131072` 下加载容易卡在 safetensors 约 `30/46`。trace 显示卡点不是固定 tensor，而是在不同 rank 的 `expert_data.copy_()` HtoD 过程中停住。4k 能加载，说明 131k 更像内存 slack / allocator 压力问题，不是 checkpoint 某个专家坏。

## 剩余嫌疑

最可疑位置已经从 kernel 下沉问题转移到完整集成边界：

- `MoERunner.forward()`
- `FusedMoE.forward()`
- `DeepseekV4MoE._forward_fused_moe()`
- shared expert 与 routed expert 合并后的 reduce/scale
- `torch.ops.vllm.moe_forward_shared` custom op 包装层
- DeepSeek V4 的 hash MoE 前 3 层与普通 `noaux_tc` 层之间的 runner 差异

Standalone `fused_marlin_moe()` probe 已覆盖太多底层路径，但仍不能复现 full-model 乱码，所以继续写 raw GEMM probe 收益下降。

## 下一步建议

### 已完成追加调研 1：activation / `gemm1_alpha` / `gemm1_beta`

检查对象：

- `deepseek-v4-flash/config.json`
- `deepseek-v4-flash/inference/config.json`
- `vllm/model_executor/layers/quantization/mxfp4.py`
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
- `vllm/model_executor/layers/fused_moe/utils.py`
- `vllm/model_executor/layers/activation.py`

结论：

- DeepSeek V4 Flash 配置中：

```json
{
  "hidden_act": "silu",
  "swiglu_limit": 10.0,
  "gemm1_alpha": null,
  "gemm1_beta": null,
  "expert_dtype": "fp4"
}
```

- GPT-OSS MXFP4 path 会显式传 `gemm1_alpha=1.702, gemm1_beta=1.0`，但 DeepSeek V4 Flash 不是这个 activation 公式。
- DeepSeek V4 routed expert 的 Marlin path 与 shared expert 的 `SiluAndMulWithClamp` 公式一致：

```python
gate = clamp(gate, max=swiglu_limit)
up = clamp(up, min=-swiglu_limit, max=swiglu_limit)
out = silu(gate) * up
```

- 因此 `gemm1_alpha/beta` 不是当前乱码主因。已修复的 `swiglu_limit=10.0` 仍然是正确修复，但它单独不能解决 full-model Marlin 输出错误。

状态：该方向降级，不再作为第一嫌疑。

### 已完成追加调研 2：`moe_forward_shared` custom op / runner 合并顺序

新增 probe：

```text
test_moe_runner_marlin_wrapper_probe.py
```

测试目的：

- 不加载完整模型。
- 使用真实 Marlin MXFP4 expert 权重。
- 使用固定 router，避免 top-k 随机性。
- 加一个假 shared expert，强制走 `torch.ops.vllm.moe_forward_shared` tuple 返回路径。
- 比较：
  - `MoERunner.forward()` wrapped custom op 路径
  - 直接调用 `MoERunner._forward_impl()` 后手动 `shared + fused`

测试结果：

```text
max_abs 0.0
mean_abs 0.0
```

结论：

- `torch.ops.vllm.moe_forward_shared` 的 tuple 返回顺序正确。
- `_unpack(result)` 没有把 shared/routed 输出拿反。
- runner 外层的 `shared_output + fused_output` 合并顺序在该受控场景下正确。
- 这个边界不能解释 full-model Marlin 乱码。

状态：custom op wrapper / tuple unpack / basic shared+routed 合并顺序降级。

### 已完成追加调研 3：`routed_scaling_factor`

检查对象：

- `DeepseekV4MoE._init_fused_moe_experts()`
- `FusedMoE.__init__()`
- `create_fused_moe_router()`
- `fused_topk_bias()`
- `MoERunner._maybe_apply_routed_scale_to_output()`

DeepSeek V4 Flash 使用：

```text
routed_scaling_factor = 1.5
norm_topk_prob = true
scoring_func = sqrtsoftplus
```

代码路径确认：

- `DeepseekV4MoE` 把 `routed_scaling_factor` 传给 `FusedMoE`。
- `FusedMoE` 默认 `apply_routed_scale_to_output=False`，因此 `self.routed_scaling_factor=1.5` 进入 router。
- `fused_topk_bias(... scoring_func="sqrtsoftplus" ...)` 最终把 `routed_scaling_factor` 传给 `ops.topk_hash_softplus_sqrt()`。
- runner 侧的 `routed_scaling_factor` 被设为 `1.0`，所以不会在 output 端重复乘。

结论：

- DeepSeek V4 的 routed scale 设计是“乘在 top-k weights 上”，不是“专家输出后再乘”。
- Marlin 和 DeepGEMM 都吃同一份 router 产生的 `topk_weights`，除非底层 kernel 对 top-k weights 的解释不同，否则这里不应造成 backend 差异。
- 即使漏乘 1.5，表现也应是幅值比例问题，不会变成当前这种 token 级乱码。

状态：`routed_scaling_factor` 方向降级。

### 已完成追加调研 4：真实 `FusedMoE.forward()` + DeepSeekV4 router

新增 probe：

```text
test_fused_moe_forward_marlin_probe.py
```

测试目的：

- 不加载完整模型，但实例化真实 `FusedMoE`。
- 使用真实 DeepSeekV4 router 参数：
  - `scoring_func="sqrtsoftplus"`
  - `renormalize=True`
  - `routed_scaling_factor=1.5`
  - `swiglu_limit=10.0`
- 使用真实 Marlin MXFP4 expert 权重。
- 加一个假 shared expert，强制覆盖 `FusedMoE.forward()` 的 shared+routed 完整路径。
- 比较：
  - `FusedMoE.forward(hidden, router_logits)`
  - 手动 `router.select_experts()` 后 `FakeShared(hidden) + kernel.apply(hidden, topk_weights, topk_ids)`

测试结果：

```text
topk_ids0 [1, 7, 5, 4, 3, 0]
topk_weights0 [0.2874156, 0.2867938, 0.2558469, 0.2498523, 0.2231440, 0.1969472]
max_abs 0.0
mean_abs 0.0
```

结论：

- 真实 `FusedMoE.forward()` 没有在调用边界引入错误。
- DeepSeekV4 router 产生的 `topk_weights/topk_ids` 与 Marlin kernel 消费方式一致。
- shared expert 与 routed expert 的合流在真实 `FusedMoE.forward()` 边界正确。
- `moe_forward_shared`、runner wrapper、router scale、`FusedMoE.forward` 这一串已经基本排除。

状态：真实 `FusedMoE.forward()` 边界降级。下一步应转向真实权重装配或真实 decoder layer boundary。

### 已完成追加调研 5：镜像内 Python 文件一致性

检查对象：

- 裸 `vllm-node-jasl-fix`
- 工作区源码 `spark-vllm-docker/vllm-sm120/vllm-sm120/...`

结果：

- 镜像内 `quantization/mxfp4.py` 已包含 `swiglu_limit = getattr(layer, "swiglu_limit", None)`，这部分一致。
- 镜像内 `fused_moe/fused_marlin_moe.py` 仍是旧版 `MarlinExperts.apply()`：
  - 非 LoRA 路径把 final GEMM 直接写入 modular allocator 提供的 `output`。
  - 非 LoRA 路径没有传 `clamp_limit=self.gemm1_clamp_limit`。
- 显式 bind mount 工作区 `fused_marlin_moe.py` 后，容器内能看到：

```text
output=None: True
clamp_limit=self.gemm1_clamp_limit: True
copy_(: True
```

结论：

- 当前环境存在“探针使用新 Python 文件、真实服务可能使用旧镜像文件”的一致性风险。
- 如果 full-model 复测没有显式挂载或重新 commit `fused_marlin_moe.py`，则乱码结果不能完全代表当前源码补丁。
- 后续真实服务复测必须同时固定三件东西：
  - 新 `_moe_C.abi3.so`
  - 正常 oracle，确保真正选择 Marlin
  - 新 `fused_marlin_moe.py`

状态：环境一致性升为第一优先级。先复测“新 Python + 新 so”的 4k true Marlin，再决定是否继续做 decoder-layer A/B。

### 已完成追加调研 6：4k true Marlin 干净复测

复测配置：

```text
vllm-node-jasl-fix
+ 新 _moe_C.abi3.so
+ 正常 oracle/mxfp4.py
+ 新 fused_marlin_moe.py
+ 新 quantization/mxfp4.py
+ --moe-backend marlin
+ --max-model-len 4096
+ --enforce-eager
```

日志确认：

```text
Using 'MARLIN' Mxfp4 MoE backend.
Using MoEPrepareAndFinalizeNoDPEPModular
Model loading took 73.85 GiB memory
```

请求结果：

```text
prompt: Write one short English paragraph explaining what a GPU is.
completion_tokens: 96
elapsed: 21.46s
throughput: 4.47 tok/s
output: A **(_,_,_ ... 大量括号/符号重复 ...
```

结论：

- 真 Marlin 在 4k 下可以完整加载并生成。
- 新 Python 补丁和新 `_moe_C` 都固定后，输出仍然乱码。
- 环境一致性不是最终根因。
- `MarlinExperts.apply()` alias/clamp 修复是必要清理，但不能解决 full-model 正确性。

状态：full-model 错误确定仍存在。下一步必须做真实模型边界 A/B，而不是继续修运行环境。

### 已完成追加调研 7：TP padding 下的 checkpoint 切片 offset

发现：

- DeepSeek V4 Flash 原始 MoE intermediate 为 `2048`。
- TP=2 后逻辑每 rank 应为 `1024`。
- MXFP4/Marlin 为 kernel 对齐会把目标参数 round up 到 `1536`。
- 原 `FusedMoE._load_w13/_load_w2()` 用 padded destination `shard_size=1536` 计算 checkpoint source offset。

错误后果：

```text
rank0 source: 0..1536       # 应为 0..1024
rank1 source: 1536..2048    # 应为 1024..2048
```

这会让 TP 权重分片错位。之前 TP probe 没测出来，是因为 probe 手工按正确 `1024` 切片，没有走真实 `FusedMoE.weight_loader()`。

已修复：

- source checkpoint 切片改为按 `loaded_weight.shape[shard_dim] / tp_size` 计算。
- padded destination `shard_size` 只用于目标 buffer 的 w1/w3 半区定位。

复测：

```text
4k true Marlin + 新 layer.py
completion_tokens: 96
throughput: 4.39 tok/s
output: A GPU (or a ,>>... 后续仍符号化崩坏
```

结论：

- 这是实打实的 loader bug，必须保留。
- 修复后输出前缀略有改善，但 full-model 仍不正确。
- 说明至少还有一个上层边界 bug。

状态：TP padding source offset 修复已入账，但不是最终根因。下一重点转向 hash MoE / `input_ids -> tid2eid` routing。

### 已完成追加调研 8：hash MoE router

DeepSeek V4 Flash 前 3 层是 hash MoE：

```text
num_hash_layers = 3
tid2eid: input_ids -> topk expert ids
```

这条路径不走普通 top-k 选专家，之前的真实 `FusedMoE.forward()` probe 没覆盖。

新增对照：

- 随机 `router_logits`
- 随机 `input_ids`
- 随机 `hash_indices_table`
- 比较 `fused_topk_bias(... hash_indices_table=...)` 的 CUDA/custom op 路径与 Python 语义：

```python
topk_ids = hash_indices_table[input_ids]
topk_weights = sqrt(softplus(logits)).gather(1, topk_ids)
topk_weights = topk_weights / sum(topk_weights) * 1.5
```

结果：

```text
ids_equal True
weights max_abs 5.96e-08
weights mean_abs 1.34e-08
```

结论：

- `ops.topk_hash_softplus_sqrt()` / hash routing custom op 本身正确。
- hash MoE 的 top-k ids 和 weights 不是当前乱码主因。

状态：hash router 降级。仍需继续查真实层装配、权重格式、或 Marlin kernel 在真实 padded-TP 权重上的行为。

### 已完成追加调研 9：padded TP Marlin 数值

新增 probe：

```text
test_real_mxfp4_marlin_tp_padded_probe.py
```

测试目的：

- 模拟真实 loader 修复后的形态：
  - checkpoint 逻辑 TP shard：`2048 / 2 = 1024`
  - Marlin 目标 buffer padded：`1024 -> 1536`
- rank0/rank1 分别把 1024 真实权重拷入 1536 padded buffer，其余补零。
- 再走 `prepare_moe_mxfp4_layer_for_marlin()` 与 `fused_marlin_moe()`。
- rank0 + rank1 输出对比完整 BF16 dequant reference。

结果：

```text
experts 8 m 2 topk 6 padded_n 1536
max_abs 0.000244140625
mean_abs 2.92e-05
```

结论：

- Marlin kernel 能正确处理真实 TP padding 后的权重。
- loader offset 修复后的 padded 权重布局本身可算对。
- full-model 剩余乱码不来自 padded TP Marlin GEMM 的基本数值路径。

状态：padded TP kernel 降级。继续上移到完整层/非 MoE 子模块/采样前 logits。

### 已完成追加调研 10：两机源码一致性 + 256 experts padded TP

检查发现：

- head 的 `layer.py` 已有 TP padding source offset 修复。
- slave 的 `layer.py` 仍是旧版。
- 因为 `launch-cluster.sh` 的 bind mount 在每台机器使用各自本地路径，head 改源码不会自动同步到 slave。

已同步：

```text
scp .../fused_moe/layer.py lmxxf@169.254.30.81:.../fused_moe/layer.py
```

同步后 4k true Marlin 复测：

```text
completion_tokens: 96
throughput: 4.50 tok/s
output: 仍为符号乱码
```

补测完整 256 experts padded TP probe：

```text
test_real_mxfp4_marlin_tp_padded_probe.py 256 1 6
max_abs 0.000244140625
mean_abs 2.96e-05
```

结论：

- 两机源码不同步是实际工程坑，但同步后 full-model 仍错。
- 256 experts、M=1、TP padding 后的 Marlin GEMM 仍对 BF16 reference 正确。
- 现在可以更确定：单个 MoE expert GEMM、TP padding、专家全集、hash router 都不是剩余主因。

状态：继续查完整模型层边界，尤其是 full `DeepseekV4MoE` 的真实 loader + shared expert + all-reduce 组合，或者 MoE 之外的 attention/MLA 与 Marlin 改动的交互。

### 1. 做 layer-boundary A/B

在单进程内实例化真实 `DeepseekV4MoE/FusedMoE` 层，用同一组：

- hidden states
- router logits / input_ids
- checkpoint 权重

分别跑：

- DeepGEMM
- true Marlin

比较 layer output。目标是确认错误是否在单个 MoE layer boundary 就出现。

如果这里复现 Marlin 错，继续切：

- router top-k 是否一致
- routed expert output 是否一致
- shared expert output 是否一致
- routed + shared 合并是否一致
- TP all-reduce 前后是否一致

### 2. 如果单层正常，转向 decoder-layer boundary

如果真实 MoE layer 单独正常，则问题上移到完整 decoder layer：

- HC transform
- attention / sparse MLA
- residual / RMSNorm
- 多层状态传播

此时应对比 DeepGEMM vs Marlin 的 layer-by-layer hidden state divergence。

### 3. 131k 加载问题后置

131k 加载卡顿是第二问题。优先级低于 4k 正确性。等 true Marlin 4k 输出正常后，再回头处理 131k 的内存 slack / allocator / staging copy 问题。

## 当前判断

这轮探索最大的价值是把 `ldmatrix` 从“第一嫌疑”降级了。它确实有 SM120/121 兼容 bug，也已经修过；但现在乱码不是它直接导致的。继续优化速度前，必须先把 full-model Marlin 正确性打穿。真正的下一刀是 layer-boundary A/B，而不是继续在 MMA fragment 层打转。

### 已完成追加调研 11：真实加载统计与 scale mapping

补了 debug-only 统计：

```text
VLLM_MXFP4_DEBUG_STATS=1
VLLM_MXFP4_DEBUG_LAYER=layers.0.ffn.experts
```

真实 4k Marlin 服务中，layer0 两个 rank 的权重和 scale 都正常加载：

- `w13_weight`: `(256,2048,2048)`，非零约 10.7 亿，范围 `0..255`
- `w2_weight`: `(256,4096,512)`，非零约 5.35 亿，范围 `0..255`
- `w13_scale`: `(256,2048,128)`，非零全满，范围约 `119..123`
- `w2_scale`: `(256,4096,32)`，非零全满，范围约 `119..123`

同时确认 DeepSeekV4 mapper 会把 checkpoint 的 `.scale` 映射到 `weight_scale`，再进入 expert mapping，因此 scale 参数名不是主因。

结论：真实服务里权重/scale 不是空的，也不是 scale mapping 丢失导致乱码。

### 已完成追加调研 12：按层混合后端二分

新增调试开关：

```text
VLLM_MXFP4_MARLIN_LAYER_RANGE=start:end
```

启动仍使用 `--moe-backend marlin`，但只有指定层走 Marlin，其他层强制 DeepGEMM，用于定位 full-model 乱码边界。

当前结果：

| Range | 含义 | 输出 | 速度 |
|---|---|---|---|
| `0:0` | 全 DeepGEMM | 正确 | 2.48 tok/s |
| `0:1` | 仅 layer0 Marlin | 正确 | 2.28 tok/s |
| `0:3` | hash 层 0..2 Marlin | 正确 | 2.63 tok/s |
| `0:10` | 前 10 层 Marlin | 正确 | 2.48 tok/s |
| `0:20` | 前 20 层 Marlin | 正确 | 2.98 tok/s |
| `0:30` | 前 30 层 Marlin | 正确 | 3.01 tok/s |
| `0:36` | 前 36 层 Marlin | 正确 | 3.03 tok/s |
| `0:40` | 前 40 层 Marlin | 正确 | 3.35 tok/s |
| `0:42` | 前 42 层 Marlin，layer42 DeepGEMM | 正确 | 3.37 tok/s |
| `42:43` | 仅 layer42 Marlin | 正确 | 2.44 tok/s |
| `40:43` | 仅最后三层 Marlin | 正确 | 2.47 tok/s |
| `0:43` | 全 Marlin，range 路径复测 | 乱码 | 4.42 tok/s |

关键结论：

- hash routing、早期 noaux_tc 层、前 42 层 Marlin 前缀都能正确生成。
- layer42 单独 Marlin 也能正确生成。
- 全 Marlin `0:43` 在 range 路径下也会乱码，速度约 4.4 tok/s。

这说明问题不是某个单层 Marlin 必然算错，而是“最后层 Marlin + 前面大量 Marlin 后的累计数值漂移/尾层放大”。下一步应做组合验证：

- `36:43`：只让最后七层 Marlin，继续确认尾段局部组合是否独立触发。
- `0:41` / `0:42` / `0:43` 的 hidden-state divergence：重点看 layer41 -> layer42 -> logits 的误差是否出现相变。
- 如果尾段局部组合仍正确，则转向 layer-by-layer hidden state divergence，比较 DeepGEMM/Marlin 在后段残差流里的误差增长。

目前更像“全模型 Marlin 每层都只有小误差，但最后 logits 对这个方向非常敏感”。这类问题不能再靠单层 GEMM probe 解决，必须看残差流。

### 已完成追加调研 13：hidden-state stats 探针

新增临时探针文件：

```text
debug_runtime/deepseek_v4.py
```

通过 bind mount 覆盖容器内：

```text
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py
```

新增环境变量：

```text
VLLM_DSV4_DEBUG_HIDDEN_STATS=1
VLLM_DSV4_DEBUG_HIDDEN_LAYERS=40:43
VLLM_DSV4_DEBUG_HIDDEN_MAX=20
VLLM_DSV4_DEBUG_HIDDEN_MAX_TOKENS=...
```

已解决的工程坑：

- slave 的 `vllm-sm120-patches/.../deepseek_v4.py` 被旧 Docker bind 误创建成 root-owned 目录，不能直接覆盖。
- 改用双机都可写的 `debug_runtime/deepseek_v4.py` 做 bind mount。
- 第一版 layer index 正则太窄，已从 `\.layers\.(\d+)$` 改成 `layers\.(\d+)`。
- warmup 会用 2048/256 token batch 消耗日志计数，因此探针加入 `VLLM_DSV4_DEBUG_HIDDEN_MAX_TOKENS` 跳过大 batch。

已观察到：

- 全 Marlin `0:43` 的 warmup 仍稳定复现乱码。
- warmup 阶段 `hc_head.out` 量级约 `rms ~= 9.77e2`，`model.norm.out` 约 `rms ~= 3.09e-1`。
- `MAX_TOKENS=128` 仍抓不到真实请求 stats，说明请求实际执行 batch 很可能按 256 token block 走。

下一步：

- 用 `VLLM_DSV4_DEBUG_HIDDEN_MAX_TOKENS=512` 重跑：
  - `0:43` 全 Marlin（坏）
  - `0:42` layer42 DeepGEMM（好）
- 对比 layer40/41/42 的 `after_attn`、`after_ffn`，以及 `hc_head.out`、`model.norm.out`。
- 如果统计尺度接近但输出坏，继续升级为保存少量 checksum/top-k logits；如果尺度已经分叉，直接锁定分叉层。

### 已完成追加调研 14：确定可用提速边界

目标：在正式改代码前，确认“前 42 层 Marlin + 最后一层 DeepGEMM”是否只是单 prompt 偶然正确，还是可以作为阶段性 fast-safe 方案。

`0:42` 小样本结果：

| Prompt | completion tokens | 速度 | 结果 |
|---|---:|---:|---|
| GPU 英文解释 | 95 | 3.83 tok/s | 正常 |
| `17 * 23` | 35 | 9.43 tok/s | 正常，答案 391 |
| Python sum 函数 | 96 | 13.74 tok/s | 正常 |
| 中文注意力机制 | 81 | 13.09 tok/s | 正常 |
| 苹果推理题 | 8 | 8.12 tok/s | 正常，答案 5 |

`0:42` 长输出：

```text
completion_tokens: 190
throughput: 13.44 tok/s
bad_score: 0
output: 五条 GPU/ML bullet points，语义正常
```

`0:43` 同一小样本对照：

| Prompt | completion tokens | 速度 | 结果 |
|---|---:|---:|---|
| GPU 英文解释 | 96 | 4.41 tok/s | 符号循环 |
| `17 * 23` | 96 | 12.16 tok/s | 空洞重复，未给正确答案 |
| Python sum 函数 | 96 | 13.54 tok/s | `first/last/#` 重复 |
| 中文注意力机制 | 90 | 12.75 tok/s | “因此/翻译”循环 |
| 苹果推理题 | 96 | 12.85 tok/s | `_**` / `_,_` 循环 |

结论：

- `0:42` 已从单 prompt 正确提升为小样本正确。
- `0:43` 已确认稳定坏，不是单 prompt 偶发。
- 阶段性确定方案：**默认 Marlin，但 layer42 强制 DeepGEMM**。
- 这个方案牺牲最后一层 MoE 的 Marlin 加速，换取正确性；比全 DeepGEMM 快，比全 Marlin 慢但可用。
- 后续正式改代码时，不应保留人工 `VLLM_MXFP4_MARLIN_LAYER_RANGE` 作为用户接口，而应实现一个针对 DeepSeek V4 Flash 的安全回退策略，例如 `VLLM_MXFP4_MARLIN_LAST_LAYER_DEEPGEMM=1` 或配置内自动识别最后 MoE 层。

### 已完成追加调研 15：尾段局部组合与实现入口

补测 `36:43`：

| Prompt | completion tokens | 速度 | 结果 |
|---|---:|---:|---|
| GPU 英文解释 | 96 | 2.88 tok/s | 正常 |
| `17 * 23` | 35 | 4.85 tok/s | 正常，答案 391 |
| 中文注意力机制 | 82 | 5.44 tok/s | 正常 |

这个结果和之前的 `40:43`、`42:43` 一致：只让尾段 Marlin 不会触发乱码。真正坏组合是“前 42 层 Marlin 形成的残差流 + layer42 Marlin”。这进一步排除了“layer42 kernel 单独错误”和“最后几层局部组合错误”。

实现入口调研：

- `FusedMoE` 有 `layer_name` 和 `layer_id`，可稳定解析 `model.layers.N.ffn.experts`。
- `Mxfp4MoEMethod` 当前只接收 `FusedMoEConfig`，里面没有 `num_hidden_layers` 或模型类型。
- DeepSeekV4 构造 `FusedMoE` 时能拿到 `vllm_config.model_config.hf_config.num_hidden_layers`，但当前没有把这个信息传进 MXFP4 quant method。

因此正式方案有两档：

1. **保守可落地**：新增显式层回退变量，例如 `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42`。优点是改动小、可解释、风险最低；缺点是用户要知道最后层号。
2. **工程化自动策略**：在 DeepSeekV4 的 `get_quant_method()` 或 `FusedMoE` 初始化路径把 `num_hidden_layers - 1` 传给 `Mxfp4MoEMethod`，由它自动对最后 MoE 层回退。优点是用户无感；缺点是侵入面更大。

我的判断：这次先做 **显式层回退变量**，把速度和正确性锁住；等后续确认没有其他模型也依赖这条路径，再做自动策略。现在缺的不是更聪明的抽象，而是把可用方案稳定落地。

### 已完成实现 1：显式层回退开关

正式改动位置：

```text
spark-vllm-docker/vllm-sm120/vllm-sm120/vllm/model_executor/layers/quantization/mxfp4.py
```

新增环境变量：

```text
VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42
```

含义：启动仍使用 `--moe-backend marlin`，但指定 decoder MoE 层在 kernel 初始化时强制回退到 `DEEPGEMM_MXFP4`。支持逗号列表和半开区间：

```text
42
40:43
0,42
0:3,42
```

保留调试变量：

```text
VLLM_MXFP4_MARLIN_LAYER_RANGE=0:42
```

它仍用于二分：range 内 Marlin，range 外 DeepGEMM。正式运行优先使用 `VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42`，因为它表达的是“全 Marlin，只把已知危险层回退”，不是实验切片。

推荐启动策略：

```text
--moe-backend marlin
VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42
```

实测结果：

| Prompt | completion tokens | 速度 | 结果 |
|---|---:|---:|---|
| GPU 英文解释 | 77 | 3.39 tok/s | 正常 |
| `17 * 23` | 35 | 9.35 tok/s | 正常，答案 391 |
| Python sum 函数 | 96 | 12.64 tok/s | 正常 |
| 中文注意力机制 | 71 | 11.85 tok/s | 正常 |
| 苹果推理题 | 8 | 8.02 tok/s | 正常，答案 5 |

启动日志确认两台 rank 均触发：

```text
MXFP4 layer 42 forced to DEEPGEMM_MXFP4 by VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42
```

结论：正式变量路径与之前 `VLLM_MXFP4_MARLIN_LAYER_RANGE=0:42` 的正确性一致，可以作为当前默认推荐方案。
