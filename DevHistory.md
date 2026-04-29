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
