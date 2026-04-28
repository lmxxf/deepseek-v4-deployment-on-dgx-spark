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

### eugr/spark-vllm-docker（当前进行中）
- 社区专门给 DGX Spark 做的 Docker 构建方案
- 从源码编译 vLLM，针对 ARM64 + Blackwell sm_121 优化
- 有 Transformers v5 版本（`vllm-node-tf5`），支持 DeepSeek V4
- `./build-and-copy.sh -t vllm-node-tf5 -c` 自动构建 + 复制到 slave
- 构建中...

---

## 经验总结

1. **DGX Spark 双机 ≠ 一台大机器**——是分布式集群，所有分布式的坑一个不少
2. **DeepSeek V4 太新**——2026-04-24 发布，NGC 26.03 容器不支持，生态还没跟上
3. **不要在 NGC 容器里 pip 升级 torch**——NGC 的 torch 是深度定制的，社区版不兼容
4. **Mac 的优势是零配置**——统一内存 + MLX/llama.cpp，一条命令跑模型；代价是没有 Blackwell FP4 加速
5. **200Gbps CX7 够用但不是 NVLink**——跨机 TP 每层通信延迟是微秒级（NVLink 是纳秒级），MoE 模型通信量小所以影响可控
6. **rsync 走 CX7 很快**——581MB/s，160G 五分钟，比 WiFi 快几十倍
