# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Static guards for the target-aware installer's CUDA dependency boundary."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
INSTALL = REPO_ROOT / "src" / "hyperloom" / "inference_optimizer" / "assets" / "install.sh"


def test_cuda_installer_uses_nvidia_extra_and_skips_amd_chains():
    text = INSTALL.read_text(encoding="utf-8")
    assert '"${REPO_ROOT}[nvidia]"' in text
    assert '"hyperloom-inference_optimizer[nvidia]"' in text
    guarded_calls = (
        "ensure_forge_gemm_tune",
        "ensure_inferencex",
        "ensure_bench_serving_deps",
        "ensure_scriptable_quality_deps",
    )
    call_region = text[text.index("HYPERLOOM_BENCHMARK_BACKEND_LC=") :]
    for call in guarded_calls:
        position = call_region.index(call)
        prefix = call_region[max(0, position - 500) : position]
        assert '!= "vllm_cuda"' in prefix, f"{call} must stay outside the CUDA install path"
    rocprof_body = text[text.index("ensure_rocprof_compute() {") : text.index("ensure_magpie() {")]
    assert 'HYPERLOOM_BENCHMARK_BACKEND_LC:-}" = "vllm_cuda"' in rocprof_body
    kernel_body = text[text.index("chain_kernel_agent() {") : text.index("HYPERLOOM_BENCHMARK_BACKEND_LC=")]
    assert 'HYPERLOOM_BENCHMARK_BACKEND_LC:-}" = "vllm_cuda"' in kernel_body
    framework_body = text[text.index("ensure_framework_deps() {") : text.index("chain_kernel_agent() {")]
    assert 'HYPERLOOM_BENCHMARK_BACKEND_LC:-}" = "vllm_cuda"' in framework_body


def test_installer_shell_is_syntax_valid():
    import subprocess

    result = subprocess.run(["bash", "-n", str(INSTALL)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
