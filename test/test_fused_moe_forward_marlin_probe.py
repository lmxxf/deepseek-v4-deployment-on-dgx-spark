import sys
import types

import torch
import torch.nn as nn


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
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.v1.worker.workspace import init_workspace_manager

from test_real_mxfp4_marlin_modular_probe import _load_marlin


class FakeShared(nn.Module):
    def forward(self, x):
        return x * torch.tensor(0.125, device=x.device, dtype=x.dtype)


class MarlinQuantShim(FusedMoEMethodBase):
    def __init__(self, kernel):
        super().__init__(kernel.fused_experts.moe_config)
        self.moe_kernel = kernel
        self.moe_quant_config = kernel.fused_experts.quant_config

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    def get_fused_moe_quant_config(self, layer):
        return self.moe_quant_config


def main():
    init_workspace_manager(torch.device("cuda"))
    experts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    topk = min(experts, int(sys.argv[3]) if len(sys.argv) > 3 else 6)

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        layer = FusedMoE(
            num_experts=experts,
            top_k=topk,
            hidden_size=4096,
            intermediate_size=2048,
            params_dtype=torch.bfloat16,
            renormalize=True,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
            e_score_correction_bias=torch.zeros(experts, device="cuda", dtype=torch.float32),
            activation="silu",
            shared_experts=FakeShared(),
            tp_size=1,
            ep_size=1,
            dp_size=1,
            pcp_size=1,
            prefix="probe.fused_moe",
        ).cuda()

        w13, w2, s13, s2, _, _ = _load_marlin(experts)
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
            shared_experts=None,
            inplace=False,
        )
        shim = MarlinQuantShim(kernel)
        layer._modules.pop("quant_method", None)
        layer._replace_quant_method(shim)
        layer.w13_weight = nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = nn.Parameter(w2, requires_grad=False)

        torch.manual_seed(123)
        hidden = (torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16) / 10).contiguous()
        router_logits = torch.randn((m, experts), device="cuda", dtype=torch.float32)

        topk_weights, topk_ids = layer.router.select_experts(
            hidden_states=hidden,
            router_logits=router_logits,
        )

        fused = kernel.apply(
            hidden_states=hidden.clone(),
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=hidden.clone(),
        )
        expected = FakeShared().cuda()(hidden) + fused

        with set_forward_context(None, vllm_config):
            actual = layer(hidden.clone(), router_logits)

    diff = (actual.float() - expected.float()).abs()
    print("topk_ids0", topk_ids[0].tolist())
    print("topk_weights0", topk_weights[0].float().tolist())
    print("actual0", actual[0, :8].float().tolist())
    print("expected0", expected[0, :8].float().tolist())
    print("max_abs", diff.max().item())
    print("mean_abs", diff.mean().item())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    main()
