import sys
import types

import torch
import torch.nn as nn

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
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.v1.worker.workspace import init_workspace_manager

from test_real_mxfp4_marlin_modular_probe import _load_marlin


class FixedRouter:
    routing_method_type = RoutingMethodType.DeepseekV4

    def __init__(self, topk_weights, topk_ids):
        self.topk_weights = topk_weights
        self.topk_ids = topk_ids

    def select_experts(self, hidden_states, router_logits, input_ids=None):
        return self.topk_weights, self.topk_ids


class FakeShared(nn.Module):
    def forward(self, x):
        return x * torch.tensor(0.125, device=x.device, dtype=x.dtype)


class FakeQuantMethod(FusedMoEMethodBase):

    def __init__(self, kernel):
        super().__init__(kernel.fused_experts.moe_config)
        self.moe_kernel = kernel
        self.moe_quant_config = kernel.fused_experts.quant_config

    @property
    def mk_owns_shared_expert(self):
        return self.moe_kernel.owns_shared_experts

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


class FakeLayer:
    def __init__(self, name, w13, w2, kernel, router):
        self.layer_name = name
        self.w13_weight = w13
        self.w2_weight = w2
        self.activation = MoEActivation.SILU
        self.global_num_experts = w13.shape[0]
        self.expert_map = None
        self.apply_router_weight_on_input = False
        self.runner = None
        self.quant_method = FakeQuantMethod(kernel)
        moe_config = kernel.fused_experts.moe_config
        self.runner = MoERunner(
            layer_name=name,
            moe_config=moe_config,
            router=router,
            routed_input_transform=None,
            routed_output_transform=None,
            gate=None,
            shared_experts=FakeShared(),
            quant_method=self.quant_method,
            enable_dbo=False,
            routed_scaling_factor=1.0,
        )

    def ensure_moe_quant_config_init(self):
        return None


def main():
    init_workspace_manager(torch.device("cuda"))

    experts = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    topk = min(experts, int(sys.argv[3]) if len(sys.argv) > 3 else 6)

    w13, w2, s13, s2, _, _ = _load_marlin(experts)
    torch.manual_seed(123)
    a = (torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16) / 10).contiguous()
    ids = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(m, 1)
    weights = torch.arange(1, topk + 1, device="cuda", dtype=torch.float32)
    weights = (weights / weights.sum()).repeat(m, 1).contiguous()
    router = FixedRouter(weights, ids)

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
    layer = FakeLayer("probe.moe", w13, w2, kernel, router)

    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context[layer.layer_name] = layer
    with set_current_vllm_config(vllm_config):
        with set_forward_context(None, vllm_config):
            wrapped = layer.runner.forward(a.clone(), torch.empty((m, experts), device="cuda"))
            direct_tuple = layer.runner._forward_impl(
                layer,
                a.clone(),
                torch.empty((m, experts), device="cuda"),
                a.clone(),
            )

    shared, fused = direct_tuple
    direct = shared + fused
    diff = (wrapped.float() - direct.float()).abs()
    print("wrapped0", wrapped[0, :8].float().tolist())
    print("direct0", direct[0, :8].float().tolist())
    print("max_abs", diff.max().item())
    print("mean_abs", diff.mean().item())
    torch.testing.assert_close(wrapped, direct, atol=0, rtol=0)


if __name__ == "__main__":
    main()
