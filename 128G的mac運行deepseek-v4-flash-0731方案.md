# 128GB Mac 运行 DeepSeek-V4-Flash-0731 方案（2026-08-02 调研）

**注意：本机没有 Mac，以下是调研整理，未亲测**——命令以各项目 README 为准，速度数字来自社区公开实测。

前提：Apple Silicon（M3 Max / M4 Max / M5 Max 级）+ **128GB 统一内存**。64GB 及以下别想全量方案（48GB 有 SSD 专家流式的邪道，另说）。

## 路线一：mlx-lm + 社区量化（最省事）

- 权重：HuggingFace 上已有针对 0731 的 MLX 量化，例如
  - `thetom-ai/DeepSeek-V4-Flash-ConfigI-MLX`——专为 128GB 天花板设计（边界专家压 4bit，给 KV cache 留余量）
  - mlx-community 的 2bit-DQ 系（约 90-96GB 档，专家 2bit、其余 4bit 起步；preview 时代的配方是听了 antirez 建议定的）
- 跑法（标准 mlx-lm 流程）：

```bash
pip install mlx-lm
mlx_lm.chat --model thetom-ai/DeepSeek-V4-Flash-ConfigI-MLX
# 或起 OpenAI 兼容服务：mlx_lm.server --model <模型名>
```

- 社区实测参考：M5 Max 约 30 到 34 tok/s，M3 Ultra 约 30 到 34 tok/s
- 优点：生态最正统、命令最少；缺点：速度中档，量化配方各家不一，下载前看清磁盘大小能不能进 128GB

## 路线二：ds4 / DwarfStar（最快，antirez 的 C 引擎）

- 项目：https://github.com/antirez/ds4 （Redis 作者写的专用推理引擎，Metal 原生，自带 OpenAI/Anthropic 兼容 server 和编码 agent）
- 权重：GGUF 格式，作者自己发布（`antirez/deepseek-v4-gguf`），q2 量化（只压路由专家 2bit，attention/共享专家原精度）
- 跑法（照它 README）：

```bash
git clone https://github.com/antirez/ds4 && cd ds4
make                       # macOS 默认 Metal 后端
./download_model.sh q2-imatrix
./ds4 -p "你好"            # 单发；./ds4 交互；./ds4-server 起 API
```

- 社区实测参考：128GB M5 Max 生成约 30+ tok/s、M3 Max 约 27（短 prompt、贪心解码口径）；prefill 300+ tok/s
- 0731 与 DSpark：项目持续跟进 0731，DSpark 投机解码有实验性支持（另下约 5.6GB 支持文件，`--dspark` 开关）
- 优点：最快、零 Python 依赖、开箱带 agent；缺点：专家 2bit（智力打折比 4bit 狠），"自包含且故意狭隘"——只伺候这一个模型系

## 坑：别下"bit-exact 原生精度 MLX 版"

`Vontra/DeepSeek-V4-Flash-0731-MXFP4-MLX` 这类"逐比特精确"的原生精度转换 = 完整 305B / 约 167GB，**128GB Mac 装不下**——那是给 192GB+ Mac Studio 准备的。看到 "bit-exact" 别激动，先看文件总大小。

## 选择建议

| 需求 | 选 |
|---|---|
| 省事、生态熟 | 路线一（mlx-lm + Config-I 或 2bit-DQ） |
| 快、想要自带 agent/API | 路线二（ds4 q2） |
| 精度洁癖 | 128GB 满足不了，上双机 DGX Spark 或 192GB+ Mac Studio |

参考（调研来源）：antirez/ds4 README、HuggingFace 各量化模型页、modelfit.io 与 kingy.ai 的 0731 本地部署指南、MindStudio 的 DwarfStar 教程。

---
*2026-08-02 整理。同目录另有双 DGX Spark 的两套方案文档（稳定档 24.5 / 快档 83.8 tok/s）。*
