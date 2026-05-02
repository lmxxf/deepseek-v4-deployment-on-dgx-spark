# DeepSeek V4 Flash 双机部署踩坑记录

两台 DGX Spark (128GB×2) 通过 ConnectX-7 200Gbps 跑 DeepSeek V4 Flash (280B, 158GB FP4)。

---

## 第一阶段：下载权重

### 问题1：hf_transfer 在国内网络翻车
- `huggingface-cli download` 默认开了 `hf_transfer` 多线程并行下载
- 国内翻墙网络扛不住多线程，10MB/s 跑到 2% 就断
- **解法**：关掉 hf_transfer（`unset HF_HUB_ENABLE_HF_TRANSFER`），单线程老老实实下
- 1.4MB/s 跑了一晚上，160G 下完

### 问题2：下载中断
- 下到 42/46 文件时连接断了
- **解法**：重跑同样的命令，已下完的自动跳过，只续传剩余文件

### 问题3：国内镜像
- hf-mirror.com 是公益镜像，`export HF_ENDPOINT=https://hf-mirror.com` 即可
- 不走梯子，国内带宽直接下

---

## 第二阶段：双机组网

### 硬件连接
- 两台 Spark 通过 QSFP56 DAC 线缆直连 ConnectX-7 网卡
- 200Gbps 点对点，不需要交换机（交换机是给 3 台以上用的）
- host (spark-3a10): 192.168.31.198 / CX7: 169.254.248.35
- slave (spark-e8bb): 192.168.31.172 / CX7: 169.254.30.81

### 配置步骤
1. `netplan` 配网（两台都执行，官方 playbook 脚本）
2. `discover-sparks` 自动发现 + SSH 免密
3. 互 ping 验证

### 关键认知：256GB 不是"统一内存"
- NVIDIA 宣传"256GB 共享内存"是营销话术
- 实际是**两台独立机器通过 RDMA 网络组成的分布式集群**
- 每台有自己独立的 128GB，通过 NCCL 做 GPU 间通信
- 不像 Mac 的统一内存那样对应用透明
- 两台机器**各需要一份模型权重**

### 权重同步
- rsync 走 CX7 200Gbps：581MB/s，160G 不到五分钟
- `rsync -avP` 从 host 到 slave

---

## 第三阶段：推理框架选型

### 方案1：手动编译 NCCL（放弃）
- 官方 playbook 要求两台都手动编译 NCCL + nccl-tests
- 步骤繁琐：装 libopenmpi-dev → 编译 NCCL（sm_121）→ 设环境变量 → 编译测试套件 → 跑通信测试
- 对比 Mac 一条命令跑模型，这条路太蛋疼

### 方案2：mark-ramsey-ri/vllm-dgx-spark 脚本（采用）
- Docker 容器方案，NCCL/Ray/vLLM 全打包在 NGC 容器里
- 一键 `./start_cluster.sh` 启动 Ray 集群 + vLLM 推理服务
- host 跑 Ray Head + vLLM API (port 8000)，slave 跑 Ray Worker

### 架构理解
- **Ray = 调度员**：管进程启停、资源分配、任务编排
- **NCCL = 对讲机**：管 GPU 之间的张量传输（AllReduce）
- **Tensor Parallel**：每一层的矩阵乘法被水平切分，两台同时算，每层通过 NCCL 同步
- 200Gbps CX7 vs 同机 NVLink 900Gbps：带宽差 4.5 倍，但 MoE 模型通信量小

---

## 第四阶段：踩坑大全

### 坑1：HuggingFace cache 格式不匹配
- 权重直接下载到 `deepseek-v4-flash/` 目录
- vllm-dgx-spark 脚本用 `snapshot_download(local_files_only=True)` 验证模型
- 这个函数要求 HF cache 格式：`hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/<hash>/`
- **解法**：直接改脚本，把 `snapshot_download` 验证替换成 `test -f config.json`
- 两个脚本都要改：`start_cluster.sh`（host）和 `start_worker_vllm.sh`（worker）

### 坑2：vLLM serve 路径问题
- Docker 容器把 `${HF_CACHE}` 挂载到 `/root/.cache/huggingface`
- 配置里 MODEL 用 HF 模型名 `deepseek-ai/DeepSeek-V4-Flash`，容器内离线模式找不到
- **解法**：把 `vllm serve` 命令里的 MODEL 硬改成容器内路径 `/root/.cache/huggingface/deepseek-v4-flash`

### 坑3：slave Docker 权限
- SSH 到 slave 启动 worker 时报 `permission denied while trying to connect to docker API`
- **解法**：`sudo usermod -aG docker lmxxf`，重新登录生效

### 坑4：vLLM 26.03 不认识 DeepSeek V4
- NGC 容器 `nvcr.io/nvidia/vllm:26.03-py3` 是 2026 年 3 月的
- DeepSeek V4 是 2026-04-24 发布的，容器里的 transformers 4.57.5 不认识 `deepseek_v4` 架构
- **解法**：在容器里 `pip install --upgrade transformers vllm`
- transformers 4.57.5 → 5.6.2，vLLM 0.17.1 → 0.20.0

### 坑5：start_cluster.sh 每次重建容器
- 脚本每次启动都删除旧容器、创建新容器
- 之前在容器里 pip 升级的全丢了
- **解法**：`docker commit` 保存升级后的容器为新镜像 `nvcr.io/nvidia/vllm:26.03-py3-dsv4`
- `docker save | ssh ... docker load` 走 CX7 传到 slave

### 坑6：脚本每次 docker pull 远程
- 新镜像名在远程 registry 不存在，pull 报错
- **解法**：改脚本 pull 逻辑，先 `docker image inspect` 检查本地有没有，有就跳过

### 坑7：worker 脚本用了旧镜像名
- `start_cluster.sh` 配置了新镜像名，但没传给 worker 脚本
- `start_worker_vllm.sh` 里 IMAGE 默认值写死了 `nvcr.io/nvidia/vllm:26.03-py3`
- **解法**：在 WORKER_ENV 里加上 `IMAGE=${IMAGE}`

### 坑8：flashinfer 版本不匹配
- 容器里 flashinfer-jit-cache 是 0.6.7，pip 升级装了 flashinfer 0.6.8
- **解法**：加环境变量 `FLASHINFER_DISABLE_VERSION_CHECK=1`

### 坑9：--swap-space 参数被移除
- vLLM 0.20.0 移除了 `--swap-space` 参数
- **解法**：从脚本里注释掉这个参数

### 坑10：GPU 显存不足
- `gpu_memory_utilization=0.9` 要求 109.52GB，但可用只有 106.1GB
- 128GB 统一内存里有一部分被系统占了
- **解法**：降到 0.85

### 坑11：kv-cache 格式
- DeepSeek V4 只支持 fp8 kv-cache，默认 `auto` 报错
- **解法**：加参数 `--kv-cache-dtype fp8`

### 坑12：PyTorch inductor 编译器冲突（致命）
- pip 升级 vllm 0.20.0 时连带把 torch 从 NGC 定制版换成了社区版
- NGC 的 torch 针对 ARM64 + Blackwell sm_121 做过深度定制
- 社区版 torch 2.11.0 的 inductor 编译器和 NGC 容器里的其他组件不兼容
- `AssertionError: auto_functionalized was not removed` — torch inductor 内部 bug
- **结论**：在 NGC 26.03 容器里 pip 升级这条路走不通

---

## 第五阶段：换方案

### eugr/spark-vllm-docker（构建成功，运行失败）
- 社区专门给 DGX Spark 做的 Docker 构建方案
- 从源码编译 vLLM 0.20.1rc1，针对 ARM64 + Blackwell sm_121 优化
- 有 Transformers v5 版本（`vllm-node-tf5`），支持 DeepSeek V4
- `./build-and-copy.sh -t vllm-node-tf5 -c` 构建成功，17 分钟，自动复制到 slave
- Dockerfile 修改：nccl 和 DeepGEMM 都改成宿主机预先 clone + COPY（Docker 内访问不了 GitHub）

### 坑13：DeepGEMM 缺失
- vLLM 的 Sparse Attention Indexer 需要 DeepGEMM 库
- eugr 镜像默认不含 DeepGEMM
- **解法**：在 Dockerfile 里加 `COPY deepgemm-src/ + pip install`
- 注意 git submodule（cutlass）也要一起 clone

### 坑14：torch inductor `auto_functionalized` 错误（再次）
- DeepGEMM 安装成功后，`--enforce-eager` 模式绕过 torch.compile
- 但仍然在 dummy_run 阶段触发 inductor 编译路径

### 坑15：DeepGEMM hyperconnection 不支持 sm_121（致命）
- `RuntimeError: Assertion error (csrc/apis/hyperconnection.hpp:56): Unsupported architecture`
- DeepGEMM 的 `tf32_hc_prenorm_gemm` kernel 硬编码了架构检查
- 只支持数据中心级 Blackwell (sm_100) 和 Hopper (sm_90)
- DGX Spark 的 GB10 是 sm_121（桌面级 Blackwell），不在支持列表
- 环境变量（`VLLM_USE_DEEP_GEMM=0` 等）无法绕过——hyperconnection 是独立的代码路径
- 论坛上有人用 Triton kernel 补丁成功，但补丁和 eugr 镜像的 vLLM 版本不兼容

### 坑16：jasl/vllm fork 补丁版本不匹配
- jasl 的 `ds4-sm120` 分支专门做了 sm_120 系列支持
- 尝试用 `-v` 挂载替换个别 .py 文件到 eugr 容器
- 失败：`ImportError: cannot import name 'dequantize_combined_sparse_mla_decode_kv'`
- 两个版本的 vllm 内部 API 不兼容，部分替换行不通

### 坑17：jasl fork 从源码编译的坑
- 改 eugr Dockerfile 把预编译 whl 换成 jasl 源码编译
- Docker 不跟软链——`ln -s` 不行，必须 `cp -r`
- 缺 `setuptools_scm`、`pybind11`、`cmake`——逐个加
- `setuptools` 版本冲突——torch 要 `<82`，vllm 要 `<81`，新 pyproject.toml 要 `>=75`
- requirements.txt 里的 hash 锁定——直接 `sed` 删掉所有 `--hash` 行
- 每次编译约 1 小时，共编译 4 次

### 坑18：CUDA 架构不匹配（最后一个坑）
- 编译成功，hyperconnection Triton fallback 生效——不再报 `Unsupported architecture`
- 但运行时 `CUDA error: no kernel image is available for execution on the device`
- 根因：容器里的 torch 2.11.0（PyPI 社区版）只支持到 sm_120，不包含 sm_121
- DGX Spark GPU 实际是 sm_121 (Capability 12.1)
- **解法**：vLLM C++ 扩展编译时用 `TORCH_CUDA_ARCH_LIST="12.0"` + `CMAKE_CUDA_ARCHITECTURES="120-real"`
- sm_120 的 kernel 在 sm_121 上前向兼容——sm_121 是 sm_120 的超集
- 重新编译一次，43 分钟，**跑通了**

### ✅ 最终成功（2026-04-29）
- 首次请求返回：*"你好，我是DeepSeek，一个由深度求索公司打造的AI助手，乐于为你解答问题、提供信息与创意灵感。"*
- 从 4-28 晚开始到 4-29 下午出字，约 48 小时
- 核心方案：eugr 基础设施 + jasl sm_120 fork 源码编译 + sm_120 前向兼容
- 镜像名：`vllm-node-sm120`

---

## 第六阶段：Docker 构建中的代理坑

### Docker 代理配置
- Docker build 容器内通过 `~/.docker/config.json` 配代理
- `127.0.0.1` 在容器内不可达，必须用宿主机 IP（`192.168.31.198`）
- `noProxy` 要排除不需要翻墙的域名：`*.ubuntu.com`、`*.nvidia.com`、`*.nvidia.cn`、`*.pypi.org`、`*.pythonhosted.org`、`*.pytorch.org`、`download-r2.pytorch.org`
- 最终方案：**去掉 Docker 代理**，GitHub 相关的在宿主机预先 clone 好 COPY 进去
- ShellCrash 的 Redir 模式（iptables 透明代理）让宿主机不需要设 proxy 环境变量就能访问 GitHub，但 Docker build 的网络是隔离的不走 iptables

---

## 日常运维命令速查

### 启动推理服务

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
./launch-cluster.sh -t vllm-node-sm120 exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-model-len 1000000 \
  --enforce-eager
```

### 停止推理服务

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker
./launch-cluster.sh -t vllm-node-sm120 stop
```

**不要 ctrl-c**——只杀 host 进程，slave 容器还在跑。必须用 stop。

### 启动聊天网页（Open WebUI）

```bash
docker start open-webui
```

浏览器打开 http://192.168.31.198:3000 ，首次启动需要注册本地账号（不联网），注册后是管理员。

如果是第一次安装 Open WebUI：

```bash
docker run -d --name open-webui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://192.168.31.198:8000/v1 \
  -e OPENAI_API_KEY=none \
  --restart unless-stopped \
  ghcr.io/open-webui/open-webui:main
```

ARM64 上首次启动初始化数据库要好几分钟，耐心等。

### 测试 API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/.cache/huggingface/deepseek-v4-flash",
    "messages": [{"role":"user","content":"你好"}],
    "max_tokens": 100
  }'
```

### 关键路径

| 路径 | 内容 |
|------|------|
| `/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash/` | 模型权重（158GB，两台都有） |
| `/home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker/` | eugr 构建工具 + 改过的 Dockerfile |
| `/home/lmxxf/work/deepseek-v4-flash-deployment/vllm-sm120/` | jasl vllm ds4-sm120 源码 |

### 网络地址

| 节点 | WiFi IP | CX7 IP |
|------|---------|--------|
| host (spark-3a10) | 192.168.31.198 | 169.254.248.35 |
| slave (spark-e8bb) | 192.168.31.172 | 169.254.30.81 |

### 已开源

- Docker Hub：`docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest`
- GitHub：`https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark`

### 显存分配

- 总可用：~206GB（两台各 103GB，0.85 利用率）
- 模型权重：~148GB（每台 73.85GB）
- KV cache：~58GB，V4 的 KV cache 仅传统模型 2%，1M 上下文只需约 5GB，绰绰有余

---

## 待解决：中间层激活值提取

### 当前状态
vLLM 是推理服务框架，不暴露中间层激活值。要做自我意识实验（dump 中间层激活、开灯/关灯对比等），需要在 forward 过程中 hook 中间层，vLLM 不支持。

### 可能的方案（等生态成熟）

1. **vLLM 加 hook 支持**——社区有需求，未来版本可能加 `--return-hidden-states`
2. **SGLang**——已有 `return_hidden_states` 实验性支持，等它支持 sm_121 + V4
3. **transformers + accelerate 双机**——等 accelerate 支持 CX7 RDMA，158GB 权重分两台各半加载，注册 forward hook
4. **NGC 官方容器**——等 26.05/26.06 NGC 容器原生支持 V4 + sm_121，用 NGC 的 torch 加 hook

### 临时方案（现在就能试，但慢）
进 vLLM 容器，用 transformers `device_map="auto"` 加载到单机 128GB（CPU+GPU 混合放置），注册 forward hook 提取激活值。速度很慢但能拿到数据。

---

## 第七阶段：长输出垃圾问题排查（2026-05-02，未解决）

### 现象
短回复（<100 token）正常，长输出（>100 token）开头固定出垃圾（重复符号、多语言乱码），几十 token 后模型自己拉回来输出正常内容。

### 排查过程

1. **chat template 排查**：tokenizer_config.json 没有 chat_template 字段（V4 设计如此，用 `--tokenizer-mode deepseek_v4` 代替）
2. **发现 `</think>` bug**：`deepseek_v4_encoding.py` 第 403 行，chat 模式下错误在 prompt 末尾插入 `</think>`。已修复（`docker commit` 保存）
3. **修复后仍然出垃圾**：prompt_tokens 从 14 降到 13（修复生效），但输出模式不变
4. **tokenizer 验证**：`PreTrainedTokenizerFast` 直接编码特殊 token 完全正确（`<｜User｜>` → 128803 单 token）
5. **completions 端点验证**：手动拼正确格式的 prompt 走 `/v1/completions`，也出垃圾——排除 chat template 问题
6. **Docker Hub 原版镜像验证**：`lmxxf/vllm-deepseek-v4-dgx-spark` 同样出垃圾——bug 一直存在，前天短对话没暴露
7. **权重文件验证**：46 shard、149GB，config 正常
8. **两台镜像一致性验证**：image ID 完全相同

### 根因确认 ✅

**Marlin MXFP4 MoE kernel 在 SM120/SM121 上产生错误计算结果。** 这是已知问题（vllm#40928、cutlass#3096）。

日志：`Using 'MARLIN' Mxfp4 MoE backend` — Marlin 的 NVFP4 kernel 没有原生 SM120 支持，PTX fallback 从 sm_80 JIT 提升到 sm_121，cubin 产生**静默错误**。

### 排除过程

| 尝试 | 结果 |
|------|------|
| `--kv-cache-dtype` 去掉 | ❌ V4 强制要求 fp8 KV cache |
| `--max-model-len 8192` | 无改善 |
| `--moe-backend triton` | ❌ Triton MoE 不支持 SM120 |
| `--moe-backend emulation` | ❌ emulation 不支持 MXFP4 权重转换 |
| `--moe-backend cutlass` | ❌ cutlass 不在 MXFP4 可选列表 |
| `--moe-backend flashinfer_cutlass` | ❌ 不支持 SM120 量化格式 |
| 清页缓存 + 降 gpu-memory-utilization | 无改善 |
| 两台重启 | 无改善 |

**结论：SM120 上只有 Marlin 能跑 FP4 MoE，但 Marlin 在 SM120 上算错。其他所有 backend 都不支持 SM120。**

### 为什么前天看起来"正常"

前天只测了短对话（"你好"，<100 token）。短回复时 MoE 专家激活少、误差小，输出碰巧能自洽。长输出（>100 token）误差累积就跑飞。

### 速度基线（参考，输出含垃圾）
- 短回复：33 tokens / 3.8s = ~8.7 tok/s（含 prefill）
- 长输出（重复 token）：1000 tokens / 75.3s = ~13.3 tok/s
- 长输出（混合垃圾+正常）：500 tokens / 54s = ~9.3 tok/s

### 进一步排查：不是全坏，是部分坏（2026-05-02 晚）

| Prompt | 结果 | tokens | finish |
|--------|------|--------|--------|
| "2+2" | ✅ 正确 "4" | 2 | stop |
| "用200字介绍北京的历史" | ✅ 完美 | 135 | stop |
| "1+1等于几" | ✅ 正确 | 33 | stop |
| "写一个Python快速排序函数" | ❌ 垃圾 | 500 | length |
| "唐朝著名诗人" | ❌ 垃圾 | 500 | length |
| "What is quicksort" | ❌ 垃圾 | 500 | length |
| "请用500字介绍万里长城" | ❌ 垃圾 | 600 | length |
| "print hello world in python" | ❌ 垃圾 | 50 | length |
| 加 system prompt | ❌ 垃圾 | 16 | length |

**模式**：知识问答类 prompt 正常（模型自然生成 EOS 停止），代码类/长文类从第一个 token 开始跑飞（被 max_tokens 截断）。不同 prompt 激活不同的 MoE 专家组合，某些专家的 FP4 计算在 SM120 上是错的。

**Marlin .so 确认编译了 sm_120 cubin**（不是 PTX fallback）——问题是 Marlin 的 FP4 mma.sync kernel 在 SM120 上计算结果错误，可能和 SM120 的 FP4 MMA 行为差异有关。

**其他 MoE backend 排查**：
- `--moe-backend triton`：不支持 SM120
- `--moe-backend emulation`：权重转换不支持 / 反量化依赖 amd-quark / quant scheme 不匹配
- `--moe-backend cutlass/flashinfer_cutlass`：不支持 MXFP4 或 SM120

**另外修复了 `</think>` bug**：`deepseek_v4_encoding.py` 第 403 行，chat 模式下错误插入 `</think>`。已在容器内修复。

### vLLM 框架内修复尝试（全部失败）

| 方案 | 结果 |
|------|------|
| `--moe-backend triton` | Triton MoE 不支持 SM120 设备 |
| `--moe-backend emulation` (OCP) | 需要 amd-quark，quark 依赖 `torch.ao.quantization.pt2e`，容器 PyTorch 没有 |
| `--moe-backend emulation` (Nvfp4) | `global_scale` 为 None，初始化路径不匹配 |
| `--moe-backend cutlass` | 不在 MXFP4 可选列表 |
| `--moe-backend flashinfer_cutlass` | 不支持 SM120 量化格式 |
| 绕过 `_return_or_raise` 检查 | ✅ 生效但后续反量化函数缺依赖 |
| 绕过 `convert_weight_to_mxfp4_moe_kernel_format` | ✅ 生效但后续 forward 缺依赖 |

**结论：vLLM 框架内的所有 fallback 路径在 SM120 + 当前容器环境下都不可用。Marlin 是唯一能跑的 backend，但计算结果部分错误。**

### 附加修复

- **`</think>` bug**：`deepseek_v4_encoding.py` 第 403 行，chat 模式下错误插入 `</think>`。已修复（`docker commit`）
- **`autodiscover` 超时**：`169.254.0.0/16` 网段 65534 个 IP 扫描超时，加 `-n` 参数手动指定节点解决

### 解法方向

1. **Consumer-DeepGEMM**（新项目）：用 CUTLASS SM120 模板实现正确的 FP4 MoE kernel，替换 Marlin——CUTLASS Example 79 有现成的 SM120 FP4 Grouped GEMM
2. **替换 `dequant_mxfp4` 函数**：用实验平台验证过的 FP4 查表法重写反量化，不依赖 quark
3. 等 vLLM 上游修复 Marlin SM120 支持
4. 等 NVIDIA NGC 容器原生支持 V4 + SM121

### Consumer-DeepGEMM 进展（2026-05-03）

**主线确定：不再继续在 vLLM 里硬 patch Marlin/emulation，老老实实做 Consumer-DeepGEMM。**

已完成：

1. **Python fallback 正确性修正**
   - `fp8_fp4_*` 不再把 packed FP4 权重当 FP8 解码
   - 新增 E2M1 查表解包 + E8M0 block scale 反量化
   - 这是正确性兜底路径，不追求速度

2. **CUTLASS native extension 骨架**
   - 新增 `consumer_deep_gemm._C`
   - 默认安装不编 CUDA，避免普通环境卡死
   - 设置 `CONSUMER_DEEP_GEMM_BUILD_CUDA=1` 才编 CUDA extension
   - `scripts/build_native_sm120.sh` 自动设置：
     - `CUDA_HOME=/usr/local/cuda`
     - `CUTLASS_PATH=../DeepGEMM/third-party/cutlass`
     - `CONSUMER_DEEP_GEMM_CUDA_ARCH=120a`
   - 兼容容器里只有 `python3` 没有 `python` 的情况

3. **Docker 编译验证通过**
   - base conda 的 PyTorch 是 CPU/非 CUDA 链接环境，不能编 extension：缺 `libc10_cuda` / `libtorch_cuda`
   - 正确编译环境是 `vllm-node-sm120:latest`
   - 用临时容器挂载源码编译成功：

```bash
docker run --rm \
  -v /home/lmxxf/work/deepseek-v4-flash-deployment:/work \
  -w /work/Consumer-DeepGEMM \
  vllm-node-sm120:latest \
  bash -lc './scripts/build_native_sm120.sh'
```

验证命令：

```bash
docker run --rm \
  -v /home/lmxxf/work/deepseek-v4-flash-deployment:/work \
  -w /work/Consumer-DeepGEMM \
  vllm-node-sm120:latest \
  python3 -c "import consumer_deep_gemm as dg; print(dg.native_build_info())"
```

输出：

```text
{'available': True, 'cutlass_sm120_probe': True, 'arch': 'sm_121a'}
```

**当前状态：编译链路打通，但还没接真实 GEMM。**  
现在 `_C` 里只有 CUTLASS SM120/SM121 探针，证明 CUDA 13.2 + PyTorch 2.11.0+cu130 + CUTLASS + aarch64 编译链路可用。真正的 `m_grouped_fp8_fp4_gemm_nt_contiguous` 还没从 CUTLASS Example 79d 拆出来。

**2026-05-03 追加：grouped fallback 语义修正**

- 修复 `Consumer-DeepGEMM/consumer_deep_gemm/gemm.py`：
  - `m_grouped_fp8_fp4_gemm_nt_contiguous` 不再把真实 MoE grouped B 当普通 2D GEMM 处理
  - 新增 per-row `m_indices` 分组路径，支持 `-1` padding 行清零
  - grouped B 反量化后按 group 选择对应专家权重，逐组执行 `A_group @ B_group.T`
  - 对 B 的 `[G, N, K]` / `[G, K, N]` 两种布局做最小推断
- 新增测试：`tests/test_fp4_fallback.py::test_m_grouped_fp8_fp4_gemm_nt_contiguous_uses_grouped_b`
- 本地验证：

```bash
python3 Consumer-DeepGEMM/tests/test_fp4_fallback.py
```

结果通过。当前环境没有 `pytest`，所以没有跑 `pytest -q`。

这个修复不是性能解法，但它把 Python fallback 的 MoE 语义接正了。后续 CUTLASS 79d kernel 可以替换 `_m_grouped_fp8_fp4_fallback_nt` 的内部实现，不需要再改 vLLM-facing API。

**2026-05-03 追加：Docker 可装载路径打通**

- 新增顶层 `deep_gemm` compatibility package：
  - `import deep_gemm` 会转发到 `consumer_deep_gemm`
  - 补了最小 `deep_gemm.utils.math`，避免 vLLM/测试 import 直接炸
- 新增 `consumer_deep_gemm/mega.py`：
  - 实现纯张量的 `transform_weights_for_mega_moe`
  - `get_symm_buffer_for_mega_moe` / `fp8_fp4_mega_moe` 先明确 `NotImplementedError`
  - 这不是最终 MegaMoE 解法，只是让 import 链和权重 transform 链先落地
- 修复 FP4 fallback 支持 vLLM 实际传入的 `int8` packed FP4 view
- 修复 `get_mk_alignment_for_contiguous_layout()` 返回类型：Consumer 包内部返回 int，vLLM wrapper 再包装成 `[align, align]`
- 新增 Docker 内安装脚本：

```bash
cd /work/Consumer-DeepGEMM
./scripts/install_in_vllm_container.sh
```

脚本动作：

1. `pip install -e .`
2. 构建 `consumer_deep_gemm._C`
3. 写入 `vllm.third_party.deep_gemm` shim
4. patch vLLM 的 DeepGEMM 支持检查，让 SM120/SM121 通过（原来只认 SM90/SM100）
5. 打印 `deep_gemm` 和 `vllm.third_party.deep_gemm` 的 native probe

已验证：

```bash
docker run --rm \
  -v /home/lmxxf/work/deepseek-v4-flash-deployment:/work \
  -w /work/Consumer-DeepGEMM \
  vllm-node-sm120:latest \
  bash -lc './scripts/install_in_vllm_container.sh'
```

输出确认：

```text
deep_gemm: {'available': True, 'cutlass_sm120_probe': True, 'arch': 'sm_121a'}
vllm.third_party.deep_gemm: {'available': True, 'cutlass_sm120_probe': True, 'arch': 'sm_121a'}
```

中途踩到 Docker bind mount owner 不一致导致 `git safe.directory` 报错，已在 installer 里自动处理。

**当前边界**：现在已经能“放进 vLLM Docker 里安装、import、通过 SM121 support gate、构建 native 扩展探针”。还不能证明长输出垃圾消失，因为真实 CUTLASS 79d grouped FP4 GEMM kernel 仍未接入，native `_C` 仍只有 probe。

下一步：

1. 从 CUTLASS `79d_blackwell_geforce_nvfp4_grouped_gemm.cu` 拆出库函数
2. 先实现 `m_grouped_fp8_fp4_gemm_nt_contiguous`
3. Python API 已经接好 native 优先、fallback 兜底
4. 接上真实 kernel 后，再集成 vLLM 服务验证长输出垃圾是否消失

---

## 经验总结

1. **DGX Spark 双机 ≠ 一台大机器**——是分布式集群，所有分布式的坑一个不少
2. **DeepSeek V4 太新**——2026-04-24 发布，NGC 26.03 容器不支持，生态还没跟上
3. **不要在 NGC 容器里 pip 升级 torch**——NGC 的 torch 是深度定制的，社区版不兼容
4. **Mac 的优势是零配置**——统一内存 + MLX/llama.cpp，一条命令跑模型；代价是没有 Blackwell FP4 加速
5. **200Gbps CX7 够用但不是 NVLink**——跨机 TP 每层通信延迟是微秒级（NVLink 是纳秒级），MoE 模型通信量小所以影响可控
6. **rsync 走 CX7 很快**——581MB/s，160G 五分钟，比 WiFi 快几十倍
7. **sm_121 是孤儿架构**——DGX Spark 的 GB10 (sm_121) 既不是数据中心 Blackwell (sm_100) 也不是桌面 RTX (sm_120)，DeepGEMM/vLLM 的 CUDA kernel 两边都没覆盖到，需要 Triton fallback
8. **Docker build 里的代理是大坑**——容器网络隔离，宿主机的透明代理不生效；GitHub 需要翻墙但 apt/pip/PyTorch 不需要，`noProxy` 配置是场噩梦；最终方案：不用代理，GitHub 相关的在宿主机 clone 好 COPY 进去
9. **"256GB 共享内存"是营销话术**——实际是两台独立机器通过 RDMA 网络组成的分布式集群，模型权重两边各存一份，GPU 通信走 NCCL + CX7
10. **sm_120 前向兼容 sm_121**——torch 社区版只编译到 sm_120，但 sm_121 能跑 sm_120 的 kernel。编译时指定 `TORCH_CUDA_ARCH_LIST="12.0"` 就行
11. **Docker 里编译 vLLM 需要约 1 小时**——ARM64 Grace CPU 10 核，MAX_JOBS=16。每次改一行重新编译都是一小时。一定要一次改对
