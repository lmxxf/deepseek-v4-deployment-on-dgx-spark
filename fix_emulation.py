"""Patch mxfp4.py to support EMULATION backend on SM120.

Run inside the vllm container:
  python3 /workspace/fix_emulation.py
"""
import re

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"

with open(TARGET, "r") as f:
    content = f.read()

# Find the "Unsupported mxfp4_backend" error and insert EMULATION branch before it
old = '''    else:
        raise ValueError(
            f"Unsupported mxfp4_backend for Mxfp4MoEMethod: {mxfp4_backend}. "
            f"Expected TRTLLM or Triton backend."
        )'''

new = '''    elif mxfp4_backend in (Mxfp4MoeBackend.EMULATION,):
        # SM120 fallback: no weight transform needed for emulation
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )
    else:
        raise ValueError(
            f"Unsupported mxfp4_backend for Mxfp4MoEMethod: {mxfp4_backend}. "
            f"Expected TRTLLM or Triton backend."
        )'''

if old not in content:
    if "EMULATION" in content and "SM120 fallback" in content:
        print("Already patched!")
    else:
        print("ERROR: Could not find target string to patch")
        exit(1)
else:
    content = content.replace(old, new)
    with open(TARGET, "w") as f:
        f.write(content)
    print("Patched successfully!")

# Verify
with open(TARGET, "r") as f:
    verify = f.read()
print(f"EMULATION branch exists: {'SM120 fallback' in verify}")
