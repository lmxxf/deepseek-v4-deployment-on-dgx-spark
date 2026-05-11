import torch
import sys
import types
import os

sys.path.append("/work/spark-vllm-docker/vllm-sm120/vllm-sm120")

class _DummyMark:
    def __getattr__(self, _name):
        def deco(*_args, **_kwargs):
            if _args and callable(_args[0]) and len(_args) == 1 and not _kwargs:
                return _args[0]
            return lambda f: f

        return deco


pytest_stub = types.SimpleNamespace(mark=_DummyMark())
sys.modules.setdefault("pytest", pytest_stub)

from tests.kernels.moe.test_moe import MarlinMoEWeightData
from tests.kernels.utils import torch_experts
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed


def main() -> None:
    set_random_seed(1)
    dtype = torch.bfloat16
    quant_type = scalar_types.float4_e2m1f
    group_size = int(os.environ.get("PROBE_GROUP", "32"))
    m = int(os.environ.get("PROBE_M", "8"))
    n = int(os.environ.get("PROBE_N", "1024"))
    k = int(os.environ.get("PROBE_K", "1024"))
    e = int(os.environ.get("PROBE_E", "8"))
    topk = int(os.environ.get("PROBE_TOPK", "2"))

    print(f"shape m={m} n={n} k={k} e={e} topk={topk} group={group_size}")

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    score = torch.randn((m, e), device="cuda", dtype=dtype)

    w1_data = MarlinMoEWeightData.make(w1, quant_type, group_size, act_order=False)
    w2_data = MarlinMoEWeightData.make(w2, quant_type, group_size, act_order=False)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)

    with set_current_vllm_config(VllmConfig()):
        score_sm = torch.softmax(score, dim=-1, dtype=torch.float32)
        topk_weight_ref, topk_ids_ref = torch.topk(score_sm, topk)
        ref = torch_experts(
            a,
            w1_data.w_ref,
            w2_data.w_ref,
            topk_weight=topk_weight_ref,
            topk_ids=topk_ids_ref,
            global_num_experts=e,
            expert_map=None,
            quant_dtype=dtype,
            per_act_token_quant=True,
        )

    out = fused_marlin_moe(
        a,
        w1_data.qweight,
        w2_data.qweight,
        None,
        None,
        w1_data.scales,
        w2_data.scales,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        quant_type_id=quant_type.id,
        global_scale1=w1_data.global_scale,
        global_scale2=w2_data.global_scale,
        input_dtype=dtype,
        is_k_full=True,
    )

    diff = (out.float() - ref.float()).abs()
    print("max_abs", diff.max().item())
    print("mean_abs", diff.mean().item())
    print("out0", out[0, :8].float().tolist())
    print("ref0", ref[0, :8].float().tolist())
    torch.testing.assert_close(out, ref, atol=4e-2, rtol=0)


if __name__ == "__main__":
    main()
