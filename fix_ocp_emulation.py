"""Revert EMULATION backend to OCP version (needs amd-quark installed)."""

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"

with open(TARGET) as f:
    c = f.read()

# Revert Nvfp4 -> OCP if needed
if "Nvfp4QuantizationEmulationTritonExperts" in c:
    c = c.replace(
        "from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import",
        "from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import",
    ).replace(
        "Nvfp4QuantizationEmulationTritonExperts,",
        "OCP_MXQuantizationEmulationTritonExperts,",
    ).replace(
        "return [Nvfp4QuantizationEmulationTritonExperts]",
        "return [OCP_MXQuantizationEmulationTritonExperts]",
    )
    with open(TARGET, "w") as f:
        f.write(c)
    print("Reverted to OCP emulation")
else:
    print("Already using OCP emulation")

# Verify all patches are in place
with open(TARGET) as f:
    c = f.read()
print(f"SM120 skip check: {'SM120 patch' in c}")
print(f"EMULATION weight bypass: {'SM120 fallback' in c}")
print(f"OCP emulation: {'OCP_MXQuantizationEmulationTritonExperts' in c}")
