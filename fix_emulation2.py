"""Patch mxfp4.py: EMULATION backend use Nvfp4 instead of OCP (no amd-quark needed).

Run inside the vllm container:
  python3 /workspace/fix_emulation2.py
"""

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"

with open(TARGET, "r") as f:
    content = f.read()

old = '''    elif backend == Mxfp4MoeBackend.EMULATION:
        from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (
            OCP_MXQuantizationEmulationTritonExperts,
        )

        return [OCP_MXQuantizationEmulationTritonExperts]'''

new = '''    elif backend == Mxfp4MoeBackend.EMULATION:
        from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
            Nvfp4QuantizationEmulationTritonExperts,
        )

        return [Nvfp4QuantizationEmulationTritonExperts]'''

if old not in content:
    if "Nvfp4QuantizationEmulationTritonExperts" in content:
        print("Already patched!")
    else:
        print("ERROR: Could not find target string")
        exit(1)
else:
    content = content.replace(old, new)
    with open(TARGET, "w") as f:
        f.write(content)
    print("Patched successfully!")

with open(TARGET, "r") as f:
    verify = f.read()
print(f"Nvfp4 emulation: {'Nvfp4QuantizationEmulationTritonExperts' in verify}")
