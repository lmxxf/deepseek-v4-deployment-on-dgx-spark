import json
import struct
import sys
import types
from types import SimpleNamespace

import torch

sys.path.append("/work/spark-vllm-docker/vllm-sm120/vllm-sm120")


class _DummyMark:
    def __getattr__(self, _name):
        def deco(*_args, **_kwargs):
            if _args and callable(_args[0]) and len(_args) == 1 and not _kwargs:
                return _args[0]
            return lambda f: f

        return deco


sys.modules.setdefault("pytest", types.SimpleNamespace(mark=_DummyMark()))

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    MarlinExperts,
    fused_marlin_moe,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from vllm.scalar_type import scalar_types
from vllm.v1.worker.workspace import init_workspace_manager


CKPT = "/work/deepseek-v4-flash/model-00002-of-00046.safetensors"


def _metadata(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return header_len, json.loads(f.read(header_len))


def _read_u8(path, header_len, meta, name):
    item = meta[name]
    begin, end = item["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_len + begin)
        data = f.read(end - begin)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(item["shape"])


def _load_marlin(experts):
    header_len, meta = _metadata(CKPT)
    w1s, w2s, w3s = [], [], []
    s1s, s2s, s3s = [], [], []
    for expert in range(experts):
        base = f"layers.0.ffn.experts.{expert}"
        w1s.append(_read_u8(CKPT, header_len, meta, f"{base}.w1.weight"))
        w2s.append(_read_u8(CKPT, header_len, meta, f"{base}.w2.weight"))
        w3s.append(_read_u8(CKPT, header_len, meta, f"{base}.w3.weight"))
        s1s.append(_read_u8(CKPT, header_len, meta, f"{base}.w1.scale"))
        s2s.append(_read_u8(CKPT, header_len, meta, f"{base}.w2.scale"))
        s3s.append(_read_u8(CKPT, header_len, meta, f"{base}.w3.scale"))

    w13 = torch.stack([torch.cat([w1, w3], dim=0) for w1, w3 in zip(w1s, w3s)]).cuda()
    w2 = torch.stack(w2s).cuda()
    s13 = torch.stack([torch.cat([s1, s3], dim=0) for s1, s3 in zip(s1s, s3s)]).cuda()
    s2 = torch.stack(s2s).cuda()

    layer = SimpleNamespace(params_dtype=torch.bfloat16)
    return prepare_moe_mxfp4_layer_for_marlin(layer, w13, w2, s13, s2, None, None)


def _make_kernel(w13, w2, s13, s2, experts, topk):
    quant_config = mxfp4_w4a16_moe_quant_config(
        w1_scale=s13,
        w2_scale=s2,
        gemm1_clamp_limit=10.0,
    )
    moe_config = FusedMoEConfig(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=4096,
        intermediate_size_per_partition=1536,
        num_local_experts=experts,
        num_logical_experts=experts,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=torch.bfloat16,
        disable_inplace=True,
    )
    kernel = mk.FusedMoEKernel(
        MoEPrepareAndFinalizeNoDPEPModular(),
        MarlinExperts(moe_config=moe_config, quant_config=quant_config),
        inplace=False,
    )
    return kernel


def main():
    init_workspace_manager(torch.device("cuda"))

    experts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    topk = min(experts, int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    w13, w2, s13, s2, _, _ = _load_marlin(experts)
    torch.manual_seed(123)
    a = (torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16) / 10).contiguous()
    ids = ((torch.arange(topk, dtype=torch.int32, device="cuda") + offset) % experts).repeat(m, 1)
    weights = torch.arange(1, topk + 1, device="cuda", dtype=torch.float32)
    weights = (weights / weights.sum()).repeat(m, 1).contiguous()

    with set_current_vllm_config(VllmConfig()):
        direct = fused_marlin_moe(
            a,
            w13,
            w2,
            None,
            None,
            s13,
            s2,
            topk_weights=weights,
            topk_ids=ids,
            global_num_experts=experts,
            quant_type_id=scalar_types.float4_e2m1f.id,
            input_dtype=torch.bfloat16,
            is_k_full=True,
            clamp_limit=10.0,
        )
        kernel = _make_kernel(w13, w2, s13, s2, experts, topk)
        modular = kernel.apply(
            a.clone(),
            w13,
            w2,
            weights,
            ids,
            activation=MoEActivation.SILU,
            global_num_experts=experts,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

    diff = (direct.float() - modular.float()).abs()
    print("experts", experts, "m", m, "topk", topk, "offset", offset)
    print("direct0", direct[0, :8].float().tolist())
    print("modular0", modular[0, :8].float().tolist())
    print("max_abs", diff.max().item())
    print("mean_abs", diff.mean().item())
    torch.testing.assert_close(modular, direct, atol=0, rtol=0)


if __name__ == "__main__":
    main()
