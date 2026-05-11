"""Patch: bypass is_supported_config check for EMULATION backend on SM120."""

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py"

with open(TARGET, "r") as f:
    content = f.read()

old = '''    def _return_or_raise(
        backend: Mxfp4MoeBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts]]:
        reason: str | None = None
        for k_cls in backend_to_kernel_cls(backend):'''

new = '''    def _return_or_raise(
        backend: Mxfp4MoeBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts]]:
        # SM120 patch: skip is_supported_config for EMULATION
        if backend == Mxfp4MoeBackend.EMULATION:
            k_cls_list = backend_to_kernel_cls(backend)
            return backend, k_cls_list[0]
        reason: str | None = None
        for k_cls in backend_to_kernel_cls(backend):'''

if old not in content:
    if "SM120 patch: skip is_supported_config for EMULATION" in content:
        print("Already patched!")
    else:
        print("ERROR: Could not find target string")
        print("Looking for:")
        print(repr(old[:100]))
        # Debug: show what's actually around line 455
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '_return_or_raise' in line:
                print(f"Line {i+1}: {line}")
        exit(1)
else:
    content = content.replace(old, new)
    with open(TARGET, "w") as f:
        f.write(content)
    print("Patched successfully!")
