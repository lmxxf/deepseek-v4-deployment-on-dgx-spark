# 双 DGX Spark 跑 DeepSeek-V4-Flash-0731 完整教程（vLLM TP=2 + DSpark 投机解码）

两台 NVIDIA DGX Spark（GB10，128GB 统一内存）直连组双机，跑 DeepSeek-V4-Flash-0731 正式版**原生精度权重**（FP8 + FP4 混合，不额外量化）。2026-08-01 实测：

| 指标 | 实测值 |
|---|---:|
| decode（DSpark 投机开启） | **51 tok/s** |
| decode（纯自回归，无投机） | 24.5 tok/s |
| prefill（8.9K token，热） | 1630 tok/s |
| prefill（44.4K token） | 1265 tok/s |
| TTFT（短 prompt） | 0.26 s |
| 上下文窗口 | 500K（模型上限 1M） |

方案源头是 [tonyd2wild/deepseek-v4-flash-dgx-spark](https://github.com/tonyd2wild/deepseek-v4-flash-dgx-spark)（vLLM 双机 recipe，针对 preview 权重 + MTP 投机）。本文在它基础上适配了 0731 正式版和 DSpark 投机——这个组合截至写作时没有公开先例。**你不需要 clone 它的仓库**，本仓库的 `spark-launch-0731.sh` 是自包含的；想读原始文档再去看它。

---

## 第 0 步：你需要什么

- 两台 DGX Spark，用 QSFP 线把两台的 ConnectX-7 200G 口直连（NVIDIA 官方叫 Spark Stacking），并配好 link-local IP（两台能互 ping）
- 两台各 ~180GB 空闲磁盘（权重 167GB）
- 两台的驱动 / 内核 / 固件版本**完全一致**（不一致会导致 prefill 性能减半甚至更怪的问题）
- 两台之间 SSH 免密（方便操作，非硬性）

下文把跑 API 服务的那台叫 **head（主机）**，另一台叫 **worker（从机）**。

## 第 1 步：下载权重（两台都要全量）

在 head 上：

```bash
hf download deepseek-ai/DeepSeek-V4-Flash-0731 --local-dir ~/work/models/DeepSeek-V4-Flash-0731
```

167GB，支持断点续传（断了重跑同一条命令）。下完同步给 worker（走直连口，200G 线实测 439MB/s，约 6 分钟）：

```bash
rsync -a --info=progress2 ~/work/models/DeepSeek-V4-Flash-0731 <worker的IP>:~/work/models/
```

权重放别处也行，后面用 `MODEL_DIR` 环境变量指过去即可。

## 第 2 步：拉镜像（两台都要）

```bash
docker pull aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6
```

这是社区为 GB10（sm_121）编译好的 vLLM 镜像（arm64，约 11GB）。`production-hybrid-1.6` 是目前唯一支持 0731 DSpark 草稿头的 tag。建议顺手把回退用的旧稳定版也拉了：

```bash
docker pull aidendle94/sparkrun-vllm-ds4-gb10:production-ready
```

## 第 3 步：拿启动脚本（两台各放一份）

就是本仓库的 [`spark-launch-0731.sh`](./spark-launch-0731.sh)。clone 本仓库或者直接下载单个文件都行：

```bash
git clone https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark.git
cd deepseek-v4-deployment-on-dgx-spark
chmod +x spark-launch-0731.sh
# 复制一份给 worker：
scp spark-launch-0731.sh <worker的IP>:~/
```

## 第 4 步：查你自己的网络参数，改脚本头部五个变量

脚本头部有五个变量，**默认值是我这对机器的，你的大概率不同**，逐个查：

```bash
ibstatus            # 看哪个 RoCE 设备是 ACTIVE —— 那就是接了线的 HCA（如 rocep1s0f1）
show_gids           # 找该 HCA 下类型为 "RoCE v2"、地址是 IPv4 的那行，记下它的 GID index（通常是 3）
ip -br addr         # 找 HCA 对应的网口名（如 enp1s0f1np1）和 head 机在这个口上的 IP
```

注意：DGX Spark 有两个 RoCE 口，**只有接线的那个能用**——没接线的口 GID 3 是空的，配错了 NCCL 的 QP 握手会直接死掉。这是最常见的翻车点。

把查到的值填进脚本头部（两台的脚本都要改，或者跑的时候用环境变量传）：

```bash
MASTER_ADDR=...   # head 机在直连口上的 IP
NCCL_HCA=...      # 接线的 RoCE 设备名
NET_IF=...        # 对应网口名
GID_INDEX=...     # RoCE v2 的 GID index
MODEL_DIR=...     # 权重目录（默认 ~/work/models/DeepSeek-V4-Flash-0731）
```

## 第 5 步：启动（顺序很重要：先 worker，后 head）

```bash
# worker 机上：
./spark-launch-0731.sh 1

# head 机上：
./spark-launch-0731.sh 0
```

容器是后台运行的（`docker run -d`），终端立刻返回。在 head 上盯日志：

```bash
docker logs -f vllm_unholy
```

权重加载 + CUDA graph 捕获约 6 到 8 分钟。依次应该看到这些健康标记：

```
enabling UE8M0 for DeepGEMM            ← FP8 快速路径生效（没有它速度会掉到 5 tok/s 档）
Loading safetensors checkpoint shards  ← 权重加载（48 片）
DSpark draft model loaded: 96 params   ← 投机草稿头挂载成功
Capturing dspark CUDA graphs           ← 投机路径就绪
Application startup complete.          ← 完成
```

## 第 6 步：验证

```bash
curl -s http://localhost:8000/v1/models

curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
```

标准 OpenAI 兼容 API，Open WebUI / 各类客户端直接指向 `http://<head的IP>:8000` 即可。冷启动后第一个请求可能因 JIT 预热而超时，重试就好。

## 第 7 步（排障）：我们踩过的五个坑

照上面流程走理论上不会再遇到（脚本里都规避了），但如果你改配置或换镜像版本，这些是雷区：

**坑 1：`KeyError: 'model.layers.43.mtp_block.main_norm.weight'`**
用 `production-ready` 旧镜像 + 0731 权重 + 开投机时必炸：0731 把 preview 的串行 MTP 预测层改造成了 DSpark 结构（config 里 compress_ratios 从 44 项变 46 项，多出的两层就是草稿模块），旧镜像只认旧参数名。对策：旧镜像必须 `SPEC=off`；要投机用 hybrid-1.6（`--speculative-config '{"method":"dspark",...}'`，草稿权重就在主 checkpoint 里）。

**坑 2：容器秒退，`dsv4-vllm-entrypoint: No such file or directory`**
hybrid-1.6 和 production-ready 目录布局完全不同：老入口没了，vllm 在 `/opt/venv/bin/`。脚本已按 tag 分流。

**坑 3：`Target SM ARCH sm_120a is not compatible`（cudaErrorNoKernelImageForDevice）**
hybrid-1.6 自带脚本设 `CUTE_DSL_ARCH=sm_120a`，但**带 a 后缀的 arch-specific 二进制不向前兼容**——sm_120 普通 kernel 能在 sm_121（GB10）上跑，sm_120a 的不行。必须 `sm_121a`。

**坑 4：`io_issue_gather_dsv4_nope() got an unexpected keyword argument 'scale_format'`**
镜像的黄金路径是 sm_120a 预编译 AOT 产物，换 arch 走现场编译后，B12X 激进路径的调用方和编译产物接口错位。对策：不开 `VLLM_USE_AOT_COMPILE`、`B12X_MLA_SM120_UNIFIED`、`--attention-backend B12X_MLA_SPARSE`，走默认路径（速度不受影响）。

**坑 5：hybrid-1.6 自带的 `serve-ds4-flash.sh` 是单机版**
它硬编码 `NCCL_IB_DISABLE=1`、`TP_SIZE=4`、PCIe allreduce。双机必须绕开它自己拼 vllm 命令：NCCL 走 RoCE（HCA/GID/IFNAME 三件套）+ `--nnodes 2 --node-rank <N> --master-addr <head IP>`。本脚本干的就是这件事。

另外两个通用提醒：
- RDMA 容器权限三件套缺一不可：`--privileged --ulimit memlock=-1 --device /dev/infiniband`（脚本已带）
- 强杀 CUDA 进程可能把 CX7 网卡搞进脏状态，之后 NCCL 莫名不通就先怀疑它，严重时两台重启

## 第 8 步：回退与停止

hybrid-1.6 出问题时，回退到验证过的稳定组合（无投机，24.5 tok/s）：

```bash
# 两台都加同样的前缀，仍然先 worker 后 head
TAG=production-ready SPEC=off ./spark-launch-0731.sh 1   # worker
TAG=production-ready SPEC=off ./spark-launch-0731.sh 0   # head
```

停止服务：两台各执行 `docker stop vllm_unholy`（保留容器，之后 `docker start vllm_unholy` 可原配置复活，模型加载时间照旧）。想彻底删掉用 `docker rm -f vllm_unholy`——不过重跑启动脚本时它自己会先清掉旧容器，一般不用手动删。

---

## 附：性能口径说明

- 51 tok/s 是 DSpark 投机解码加持的生成吞吐（512 token、温度 0、单流；流式复测排除 prefill 后纯 decode 50.4 tok/s，TTFT 0.26s）。纯自回归裸速 24.5 tok/s——投机的原理是草稿头一次猜一块、主模型一次前向并行验证，输出分布与纯自回归完全一致，无质量损失
- prefill 实测（max_tokens=1、随机盐防 prefix cache 命中）：8.9K token 冷 820 / 热 1630 tok/s，44.4K token 1265 tok/s。首个同形状请求含 JIT 编译，速度打半折
- 并发吞吐未测（原 recipe 报 8 并发 92 tok/s，是 preview + MTP 时代的数字）
- `--max-model-len 500000` 稳定；模型上限 1M 未实测（CSA/HCA 压缩下 FP8 KV 的 1M 上下文仅约 5GB，内存不是瓶颈，prefill 时间才是）

*2026-08-01，随 0731 正式版发布当周的部署实录。*
