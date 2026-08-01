# 如何使用 tonyd2wild 的 ds4 推理框架（双 DGX Spark 跑 DeepSeek-V4-Flash-0731）

**2026-08-01 实录。最终战果：双机 TP=2 + DSpark 投机解码 = 51.1 tok/s（512 token、温度 0、单流）。**

速度对照：

| 方案 | 速度 |
|---|---:|
| 自家混合后端（preview 权重，185 期方案） | 12 tok/s |
| production-ready 镜像、无投机（0731） | 24.5 tok/s |
| **hybrid-1.6 镜像 + DSpark（0731）** | **51.1 tok/s** |

---

## 一、这套东西是什么

- **recipe 仓库**：https://github.com/tonyd2wild/deepseek-v4-flash-dgx-spark （本地 clone 在 `~/work/deepseek-v4-flash-dgx-spark/`）
  文档为主：README + TROUBLESHOOTING + `launch/unholy-launch.sh` 模板。方案 = vLLM 多机 TP=2，原生精度（FP8 dense + mxfp4 专家，**不额外量化**），NCCL 走 RoCE/RDMA。
- **Docker 镜像**：`aidendle94/sparkrun-vllm-ds4-gb10`（arm64，为 GB10/sm_121 编译的 vLLM）
  - `production-ready`（06-01，12.3GB）：recipe 原配。跑 0731 **必须关投机**（见坑 1）
  - `production-hybrid-1.6`（07-26，11.4GB）：多模型大杂烩（GLM/Kimi/MiniMax 都有），**带 DSpark speculator**，但自带启动脚本是单机版——双机配置要自己写（本文核心增量）
- **我们的启动脚本**：`~/work/deepseek-v4-flash-dgx-spark/launch/spark-launch.sh`（host）＋ slave 上 `~/work/dgx-spark-slave/spark-launch.sh`（同一文件的拷贝，改完记得 scp 同步）

## 二、前置条件

- 权重：`~/work/models/DeepSeek-V4-Flash-0731/`（167GB，48 shards），**两台都要全量**
  - `hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir ...`（支持断点续传）
  - 同步 slave：`rsync -a --info=progress2 ~/work/models/DeepSeek-V4-Flash-0731 169.254.30.81:~/work/models/`（CX7 实测 439MB/s，约 6 分钟）
- 镜像：两台各 `docker pull aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6`（回退用的 `production-ready` 也建议留着）
- 网络硬参数（我们这对机器，`show_gids` 现查可验证）：
  - host spark-3a10 = 169.254.248.35，slave = 169.254.30.81
  - HCA：**`rocep1s0f1`**（recipe 模板写的是 f0，我们接线的是 f1，照抄必翻车）
  - `NCCL_IB_GID_INDEX=3`（RoCE v2 / IPv4 那条 GID）
  - 控制面网口：`enp1s0f1np1`（我们没有独立有线以太网，控制面和 RDMA 共用 CX7 直连口）
- RDMA 容器权限三件套（缺一 QP 握手就死）：`--privileged --ulimit memlock=-1 --device /dev/infiniband`
- 两台驱动/内核/固件版本必须一致（recipe 实测不一致 prefill 掉一半）

## 三、启动与验证

```bash
# 先 worker（slave）
ssh 169.254.30.81 "bash ~/work/dgx-spark-slave/spark-launch.sh 1"
# 后 head（host）
~/work/deepseek-v4-flash-dgx-spark/launch/spark-launch.sh 0

# 盯日志（权重加载 + CUDA graph 捕获约 6-8 分钟）
docker logs -f vllm_unholy        # 等 "Application startup complete"
# 关键健康标记：
#   "enabling UE8M0 for DeepGEMM"          → FP8 快速路径生效（没有它会掉到 5 tok/s 档）
#   "DSpark draft model loaded: 96 params" → 草稿头挂载成功
#   "Capturing dspark CUDA graphs"         → 投机路径 cudagraph 就绪

# 验活
curl -s http://localhost:8000/v1/models
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
```

脚本开关（环境变量前缀）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `TAG` | production-hybrid-1.6 | 镜像 tag；`production-ready` 走旧入口 `dsv4-vllm-entrypoint` |
| `SPEC` | dspark | `dspark` / `mtp` / `off`；**production-ready 必须 off** |

回退到验证过的稳定组合（24.5 tok/s）：
```bash
ssh 169.254.30.81 "TAG=production-ready SPEC=off bash ~/work/dgx-spark-slave/spark-launch.sh 1"
TAG=production-ready SPEC=off ~/work/deepseek-v4-flash-dgx-spark/launch/spark-launch.sh 0
```

停服务：两台各 `docker rm -f vllm_unholy`。

对话前端：`docker start open-webui` → http://192.168.31.198:3000（对接 8000 端口的 OpenAI 兼容 API）。

## 四、踩坑实录（按时间顺序，每个都附机制）

**坑 1：production-ready + 0731 权重，草稿模型加载 KeyError**
`KeyError: 'model.layers.43.mtp_block.main_norm.weight'`——主模型 48 片全加载完，炸在 drafter。机制：0731 把 preview 的串行 MTP 预测层改造成了 DSpark 结构（config 佐证：compress_ratios 从 44 项加长到 46 项，多出的两层就是草稿模块的），06-01 的镜像只认旧 MTP 参数名。**对策**：旧镜像 `SPEC=off` 可正常跑主干；要投机就换 hybrid-1.6——它有独立的 dspark speculator（`--speculative-config '{"method":"dspark",...}'`，草稿权重就藏在主 checkpoint 里，不用单独文件）。

**坑 2：hybrid-1.6 秒退，`dsv4-vllm-entrypoint: No such file or directory`**
新镜像目录布局全变：老入口没了，vllm 在 `/opt/venv/bin/`，自带一堆 `serve-*.sh`。机制：hybrid 系列是多模型镜像，和 production-ready 不同源。**对策**：直接 `python -m vllm.entrypoints.cli.main serve` 自己拼参数。

**坑 3：`Target SM ARCH sm_120a is not compatible`（cudaErrorNoKernelImageForDevice）**
照抄了它 serve 脚本里的 `CUTE_DSL_ARCH=sm_120a`。机制：**带 a 后缀的 arch-specific 二进制不向前兼容**——sm_120 普通 kernel 能在 sm_121 上跑，sm_120a 的不行（和 182 期勘误的 sm_120/121 刀法同一族）。**对策**：改 `CUTE_DSL_ARCH=sm_121a`。

**坑 4：`io_issue_gather_dsv4_nope() got an unexpected keyword argument 'scale_format'`**
机制：镜像的黄金路径是"sm_120a 预编译 AOT 产物 + 单机启动流程"，换 arch 后走现场编译，B12X 激进路径的 Python 调用方和编译产物接口错位。**对策**：砍掉三个激进项走默认路径——`VLLM_USE_AOT_COMPILE=1`、`B12X_MLA_SM120_UNIFIED=1`、`--attention-backend B12X_MLA_SPARSE`。砍完即通，速度照样 51 tok/s。

**坑 5（设计差异，不是 bug）：hybrid-1.6 自带脚本是单机版**
`serve-ds4-flash.sh` 硬编码 `NCCL_IB_DISABLE=1`、`TP_SIZE=4`、PCIe allreduce——面向单机多卡/新形态。双机要绕开它的脚本自己起 vllm，把 NCCL 环境换成我们的 RoCE 三件套（HCA/GID/IFNAME）+ `--nnodes 2 --node-rank N --master-addr 169.254.248.35`。**双机跑 hybrid-1.6 网上没有先例，此文档即首个记录。**

## 五、遗留与备忘

- `--max-model-len 500000` 稳定；模型上限 1M，没实测（KV cache FP8 下 1M 才约 5GB，内存不是约束，prefill 时间才是）
- 并发吞吐没量（recipe 报 8 并发 92 tok/s，是 MTP 时代的数，DSpark 下应该更高）
- 首请求可能超时（JIT 预热），重试即秒回
- 强杀 CUDA 进程可能搞脏 CX7，症状不对先想想这个（老教训，严重时两台重启）
- 老混合后端（185 期发布的自家 Docker Hub 镜像）和 preview 权重（`~/work/deepseek-v4-flash-deployment/deepseek-v4-flash/`）保留当考古备份

---
*整理：2026-08-01，随 0731 正式版部署实录。相关公众号：185 期（自家混合后端史）、236 期（DSpark 原理）、273 期（0731 评测与家用部署生态）。*
