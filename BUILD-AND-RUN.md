# DeepSeek V4 Flash 双机部署：编译启动流程

christopherowen MXFP4 方案，基于 eugr/spark-vllm-docker。

## 前置条件

- 两台 DGX Spark 已组网（CX7 200Gbps 直连）
- host: 169.254.248.35 / slave: 169.254.30.81
- ShellCrash TUN 模式已开启（Docker BuildKit 仍然不走 TUN，需要预先 clone）
- 模型权重已下载到两台的 `/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash/`

## 第一步：Clone 依赖（宿主机，只需做一次）

所有依赖都在 `spark-vllm-docker/` 下。

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

# christopherowen 三件套（flashinfer + cutlass 内嵌）
git clone https://github.com/christopherowen/vllm.git vllm-mxfp4-src
cd vllm-mxfp4-src && git checkout 045293d82b832229560ac4a13152a095af603b6e && git submodule update --init --recursive && cd ..

git clone https://github.com/christopherowen/flashinfer.git flashinfer-mxfp4-src
cd flashinfer-mxfp4-src && git checkout f349e52496a72a00d8c4ac02c7a1e38523ff7194 && git submodule update --init 3rdparty/spdlog && cd ..
cd flashinfer-mxfp4-src && rm -rf 3rdparty/cutlass && git clone https://github.com/christopherowen/cutlass.git 3rdparty/cutlass && cd 3rdparty/cutlass && git checkout fede53000a962b46e05bafe0c86311778caeb380 && cd ../../../

# vLLM CMake FetchContent 依赖（Docker BuildKit 内 clone GitHub 不稳定）
git clone --depth 1 --branch v4.2.1 https://github.com/nvidia/cutlass.git cutlass-v421-src
git clone --depth 1 --branch v3.5.0 https://github.com/triton-lang/triton.git triton-mxfp4-src
git clone https://github.com/IST-DASLab/qutlass.git qutlass-mxfp4-src && cd qutlass-mxfp4-src && git checkout 830d2c4537c7396e14a02a46fbddd18b5d107c65 && cd ..
git clone https://github.com/vllm-project/FlashMLA.git flashmla-mxfp4-src && cd flashmla-mxfp4-src && git checkout 46d64a8ebef03fa50b4ae74937276a5c940e3f95 && cd ..
git clone https://github.com/vllm-project/flash-attention.git flash-attn-mxfp4-src && cd flash-attn-mxfp4-src && git checkout 86f8f157cf82aa2342743752b97788922dd7de43 && git submodule update --init csrc/cutlass && cd ..
```

## 第二步：编译镜像

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker
./build-and-copy.sh --exp-mxfp4 -t vllm-node-mxfp4 -c 169.254.30.81
```

约 20-25 分钟。自动 copy 到 slave。

**注意**：Dockerfile.mxfp4 已改成 COPY 本地依赖（不走 GitHub clone），同时跳过了 flashinfer-cubin 下载（5299 个 cubin 太慢，JIT 替代）。

## 第三步：升级 transformers

NGC 26.01 基础镜像的 transformers 太旧，不认识 `deepseek_v4` 架构。编译完需要升级：

```bash
# host 上升级
docker run --name tf-upgrade -d vllm-node-mxfp4:latest sleep 60
docker exec tf-upgrade pip install -U transformers
docker commit tf-upgrade vllm-node-mxfp4:latest
docker rm -f tf-upgrade

# 同步到 slave（走 CX7，约 2 分钟）
docker save vllm-node-mxfp4:latest | ssh lmxxf@169.254.30.81 "docker load"
```

## 第四步：启动服务

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker

HF_HOME=/home/lmxxf/work/deepseek-v4-flash-deployment \
VLLM_SPARK_EXTRA_DOCKER_ARGS="-e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1" \
./launch-cluster.sh -n 169.254.248.35,169.254.30.81 -t vllm-node-mxfp4 exec \
  vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --max-model-len 131072 \
  --enforce-eager
```

## 第五步：测试

```bash
# 等服务起来
curl -s http://localhost:8000/health

# 短测
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"2+2等于几？"}],"max_tokens":20}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])"

# 英文长输出（验证 MXFP4 正确性）
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"What is quicksort? Explain with code."}],"max_tokens":500}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:300]); u=d['usage']; print(f'\ntokens: {u[\"completion_tokens\"]}')"

# 中文长输出
time curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash","messages":[{"role":"user","content":"请用500字介绍万里长城"}],"max_tokens":600}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:200]); u=d['usage']; print(f'\ntokens: {u[\"completion_tokens\"]}')"
```

## 停止服务

```bash
cd /home/lmxxf/work/deepseek-v4-flash-deployment/spark-vllm-docker
./launch-cluster.sh -t vllm-node-mxfp4 stop
```

**不要 ctrl-c**——只杀 host 进程，slave 容器还在跑。必须用 stop。

## 聊天网页（Open WebUI）

```bash
docker start open-webui
# 浏览器打开 http://192.168.31.198:3000
```

## 坑速查

| 问题 | 原因 | 解法 |
|------|------|------|
| `deepseek_v4` architecture not recognized | transformers 太旧 | `pip install -U transformers` + 同步 slave |
| slave 报 `No module named transformers.models.deepseek_v4` | slave 镜像没更新 | `docker save \| ssh ... docker load` |
| CMake FetchContent clone 超时 | Docker BuildKit 不走 TUN | 宿主机预先 clone + COPY（已配置） |
| flashinfer-cubin 下载 9 小时 | 5299 个 cubin 从 NVIDIA 下载 | 跳过，用 JIT 替代（已配置） |
| `setuptools-scm` 版本检测失败 | .dockerignore 排除了 .git | 不排除 .git（已修复） |
| gpu-memory-utilization OOM | 0.85 时 slave swap 爆满 | 降到 0.80 |

## Dockerfile.mxfp4 改动汇总（相对 eugr 原版）

1. FlashInfer + CUTLASS：git clone → `COPY flashinfer-mxfp4-src/`
2. vLLM：git clone → `COPY vllm-mxfp4-src/`
3. CMake FetchContent 5 个依赖：COPY + ENV 环境变量指向本地路径
4. flashinfer-cubin 构建：注释掉（跳过 5299 cubin 下载）
5. .dockerignore：只排 `__pycache__` 和 `.pyc`，保留 `.git`（setuptools-scm 需要）

## 镜像对照

| 镜像 | 方案 | 中文 | 英文 | 速度 |
|------|------|------|------|------|
| `vllm-node-jasl:latest` | jasl fork Triton MLA | ✅ | ❌ | 14 tok/s |
| `vllm-node-sm121-cdg:latest` | Consumer-DeepGEMM fused Triton | ✅ | ✅ | 1.87 tok/s |
| `vllm-node-mxfp4:latest` | christopherowen CUTLASS MXFP4 | ? | ? | ? |
