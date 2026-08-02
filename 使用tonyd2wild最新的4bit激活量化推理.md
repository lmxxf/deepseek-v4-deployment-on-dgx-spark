# 使用 tonyd2wild 最新方案：DSpark + NVFP4 KV cache（双 DGX Spark，峰值 83.8 tok/s）

**2026-08-02 实录。** 注：文件名说的"4bit 量化"准确指 **KV cache 存储压到 NVFP4（4 比特浮点 e2m1 + 块级缩放）**——权重精度没变（还是原生 FP8+FP4），注意力计算精度也没变，压的是"记忆"的存储。官方口径 KV 是 FP8（fp8_ds_mla），NVFP4 是作者的实验性再压。

实测速度（本机双 Spark，TP=2）：

| 测试 | 速度 |
|---|---:|
| 峰值（数数到 300，温度 0，热） | **83.8 tok/s** |
| 代码生成题（512 tok） | 52.6 tok/s |
| 参照：旧栈无投机 | 24.5 tok/s |
| 参照：旧栈残血 DSpark（会崩） | 51 tok/s |

比旧栈快的三个来源：**Patch 4**（0731 草稿头有 12 个张量被 vLLM 加载器静默丢弃，接受率 25.7%→60.2%，作者查明并修复，已烤进本方案镜像）、**k=5 概率草稿**（比 k=2 一次多赚）、**NVFP4 KV**（存储带宽减半）。旧栈 DSpark 长对话必崩的坑也埋了——真凶是 `repetition_penalty`（作者文档明写的 spec-decode crash risk，本方案启动器不带任何采样覆盖）。

---

## 一、仓库与前置

```bash
git clone https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark.git
cd DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
```

前置与旧方案相同：两台的权重全量（`~/work/models/DeepSeek-V4-Flash-0731/`）、SSH 免密、RoCE 直连。镜像**本地构建**（基础镜像 `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` 会在 build 时自动拉取，需翻墙覆盖 Docker）。

## 二、配置 .env.dspark

从模板复制并改我们的值（仓库根目录）：

```bash
cp .env.dspark.example .env.dspark
```

改动项（其余保持模板的 validated 默认，**别动 MTP_NUM_TOKENS=5 和那堆 VLLM_DSPARK_* 开关**）：

```bash
WORKER_HOST=169.254.30.81
MASTER_ADDR=169.254.248.35
NCCL_IB_HCA=rocep1s0f1
NCCL_SOCKET_IFNAME=enp1s0f1np1
NCCL_IB_GID_INDEX=3
HF_CACHE=~/work/models                                   # 挂载点戏法：把 models 目录当 HF cache 挂
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
DSPARK_MODEL=/cache/huggingface/DeepSeek-V4-Flash-0731   # 容器内路径 = HF_CACHE 挂载后的权重目录
VLLM_HOST_IP=169.254.248.35
WORKER_VLLM_HOST_IP=169.254.30.81
```

（HF_CACHE 请写绝对路径。模型不走 HuggingFace 下载，直接把本地权重目录挂成 `/cache/huggingface` 再用容器内路径指过去。）

自检——渲染出的 vLLM 命令会打出来，肉眼过一遍模型路径/master 地址/端口：

```bash
./validate-dspark-config.sh
```

把仓库同步给 worker（同路径镜像；之后 start 脚本每次会自动同步 compose 和 .env，但首次全量要手动）：

```bash
# 把整个 checkout 同步到 worker 的相同路径（示例按本机路径，改成你自己的）
rsync -a "$(pwd)" 169.254.30.81:"$(dirname "$(pwd)")/"
```

## 三、坑：基础镜像烤死了作者的网口名（必改）

直接启动会秒炸：`Unable to find address for: enP7s7`。基础镜像里烤死了作者机器的环境变量（`GLOO_SOCKET_IFNAME=enP7s7`、`TP_SOCKET_IFNAME=enP7s7`、`MN_IF_NAME`、`OMPI_MCA_btl_tcp_if_include`），compose 只覆盖了 NCCL 那一个。修法：在 `docker-compose.dspark.yml` 的 environment 段、`NCCL_SOCKET_IFNAME` 那行后面补四行覆盖：

```yaml
      GLOO_SOCKET_IFNAME: "${NCCL_SOCKET_IFNAME}"
      TP_SOCKET_IFNAME: "${NCCL_SOCKET_IFNAME}"
      MN_IF_NAME: "${NCCL_SOCKET_IFNAME}"
      OMPI_MCA_btl_tcp_if_include: "${NCCL_SOCKET_IFNAME}"
```

（本机的 compose 已改好。start 脚本会把改过的 compose 自动同步到 worker。）

## 四、构建镜像（一条命令管两台）

```bash
./build-dspark-vllm-runtime.sh
```

本地依次构建 overlay → nvfp4 stage a/b/c 四层，然后自动 rsync 到 worker 并 ssh 过去重复构建。基础镜像已缓存时整个过程不到一分钟；首次要拉 ghcr 基础镜像（几个 GB）。结束标志：`dspark nvfp4 stage-c image ok 0.21.1rc1...`。

Patch 4 不用单独操作——已烤在 `recipe/overlay/vllm/v1/spec_decode/dspark.py` 里。

## 五、启动 / 验证 / 停止

```bash
# 启动（自动先 worker 后 head，然后轮询等 API；权重两遍加载 + cudagraph 约 10 分钟，curl 连接拒绝刷屏是正常等待）
./start-deepseek-v4-flash-dspark.sh

# 验活（注意：端口 8888，模型名 deepseek-v4-flash-dspark）
curl -s http://localhost:8888/v1/models

# 冒烟测试（仓库自带）
./smoke-deepseek-v4-flash-dspark.sh

# 看日志 / 状态
./logs-deepseek-v4-flash-dspark.sh
./status-deepseek-v4-flash-dspark.sh

# 停止（一条管两台）
./stop-deepseek-v4-flash-dspark.sh
```

Open WebUI 对接：端点改 `http://192.168.31.198:8888/v1`，模型 `deepseek-v4-flash-dspark`。

速度参考口径：作者的 peak-finder 提示是 `"Count from 1 to 300, separated by commas."`（温度 0）；投机解码的速度**随内容浮动**（结构化内容接受率高、跑得快，散文推理慢），单一数字没有意义，报速度要带内容类型。冷启动后第一发有约 30% 惩罚，热一发再量。

## 六、遗留与注意

- **长对话稳定性待验收**（旧栈 DSpark 就是死在这关）。理论上已修：崩溃真凶 `repetition_penalty` 本方案不带；但 NVFP4 KV 是实验性有损存储，超长上下文远端细节是否变糊需要日常使用检验
- 想较真 NVFP4 质量：同组 prompt 在 fp8 KV（作者文档的保守路线）与 nvfp4 两档对跑输出——224 期式质检
- `MAX_NUM_SEQS=12`、1M 上下文与并发共享一个 KV 池，多个满上下文请求会排队
- 别设 `VLLM_USE_V2_MODEL_RUNNER=1`（与 DSpark 硬冲突）、别传 `--attention-backend`（保持 AUTO）、别把 `VLLM_USE_B12X_MOE` 关掉（关了掉到 29 tok/s 档）——.env 模板注释里都有出处
- 旧栈（8000 端口、`spark-launch-0731.sh` 那套）转预备役，回退路径见《如何使用tonyd2wild的ds4推理框架.md》

---
*2026-08-02，随 tonyd2wild 新仓库（DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark）部署实录。速度阶梯：12（自家混合后端）→ 24.5（无投机）→ 51（残血 DSpark，不稳）→ 83.8 峰值（本方案）。*
