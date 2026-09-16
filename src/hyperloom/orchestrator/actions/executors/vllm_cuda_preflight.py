# SPDX-License-Identifier: MIT
"""Framework checks owned by the native vLLM CUDA backend."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any

from hyperloom.inference_optimizer.target_registry import TargetValidationError, _rehash_fingerprint

VLLM_CUDA_REQUIRED_SERVER_FLAGS = (
    "--host",
    "--port",
    "--served-model-name",
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--max-model-len",
)
VLLM_CUDA_REQUIRED_BENCH_FLAGS = (
    "--backend",
    "--base-url",
    "--endpoint",
    "--model",
    "--tokenizer",
    "--dataset-name",
    "--random-input-len",
    "--random-output-len",
    "--num-prompts",
    "--num-warmups",
    "--max-concurrency",
    "--request-rate",
    "--ignore-eos",
    "--percentile-metrics",
    "--metric-percentiles",
    "--save-result",
    "--result-dir",
    "--result-filename",
    "--disable-tqdm",
)


def _probe_vllm_cli_flags(
    *subcommand: str, python_exe: str = sys.executable, env: dict[str, str] | None = None
) -> set[str]:
    """Return long options exposed by one installed vLLM CLI command.

    Recent vLLM releases group most serving flags by config class, so the
    probe uses ``--help=all``. It runs through ``sys.executable`` to verify the
    exact interpreter that will launch the CUDA runner.
    """
    command = [python_exe, "-m", "vllm.entrypoints.cli.main", *subcommand, "--help=all"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        rendered = " ".join(command)
        raise TargetValidationError(f"cannot probe vLLM CLI ({rendered}): {exc}") from exc
    output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        detail = output.strip()[-1000:]
        raise TargetValidationError(
            f"vLLM CLI probe failed for {' '.join(subcommand)} (rc={proc.returncode}): {detail}"
        )
    flags = set(re.findall(r"(?<![A-Za-z0-9_-])(--[a-z0-9][a-z0-9-]*)", output))
    if not flags:
        raise TargetValidationError(f"vLLM CLI probe returned no long options for {' '.join(subcommand)}")
    return flags


_STACK_PROBE = r"""
import importlib.metadata, json, torch
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is false")
for index in range(torch.cuda.device_count()):
    # Exercise an actual kernel, not only the driver/library version strings.
    assert torch.ones(1, device=f"cuda:{index}").sum().item() == 1
try:
    nccl_version = str(torch.cuda.nccl.version() or "")
except Exception:
    nccl_version = ""
print(json.dumps({"vllm_version": importlib.metadata.version("vllm"),
                  "torch_version": str(torch.__version__), "torch_cuda_version": str(torch.version.cuda or ""),
                  "nccl_version": nccl_version, "device_count": torch.cuda.device_count()}))
"""


def validate_execution_stack(
    fingerprint: dict[str, Any], *, python_exe: str = sys.executable, pythonpath: str = ""
) -> dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(r["uuid"] for r in fingerprint["devices"])
    env["VLLM_PLUGINS"] = ""
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [pythonpath, env.get("PYTHONPATH", "")]))
    try:
        proc = subprocess.run(
            [python_exe, "-c", _STACK_PROBE], env=env, capture_output=True, text=True, timeout=90, check=True
        )
        stack = json.loads(proc.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise TargetValidationError(
            f"vllm_cuda execution environment failed: {getattr(exc, 'stderr', '') or exc}"
        ) from exc
    if stack.pop("device_count") != len(fingerprint["devices"]):
        raise TargetValidationError("vllm_cuda interpreter sees a different GPU pool")
    cli = {}
    for key, command, required in (
        ("server_flags", ("serve",), VLLM_CUDA_REQUIRED_SERVER_FLAGS),
        ("bench_flags", ("bench", "serve"), VLLM_CUDA_REQUIRED_BENCH_FLAGS),
    ):
        available = _probe_vllm_cli_flags(*command, python_exe=python_exe, env=env)
        missing = sorted(set(required) - available)
        if missing:
            raise TargetValidationError(f"vLLM {' '.join(command)} missing required flag(s): {', '.join(missing)}")
        cli[key] = sorted(available)
    result = {
        **fingerprint,
        **stack,
        "vllm_cli": cli,
        "runtime_python": python_exe,
        "runtime_pythonpath": pythonpath,
        "benchmark_backend": "vllm_cuda",
    }
    _rehash_fingerprint(result)
    return result


def validate_profile_runtime(
    fingerprint: dict[str, Any], *, backend: str = "torch", roofline: bool = False
) -> dict[str, Any]:
    surface = fingerprint.get("vllm_cli", {})
    if "--profiler-config" not in surface.get("server_flags", []):
        raise TargetValidationError("NVIDIA profiling requires vLLM serve --profiler-config")
    if "--profile" not in surface.get("bench_flags", []):
        raise TargetValidationError("NVIDIA profiling requires vLLM bench serve --profile")
    if backend == "nsys":
        from .cuda_nsight import tool_fingerprint

        try:
            return tool_fingerprint(roofline=roofline)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            raise TargetValidationError(f"Nsight preflight failed: {exc}") from exc
    return {}


def validate_execution_identity(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Framework identity is a backend concern, including legacy flat metadata."""
    keys = ("torch_version", "torch_cuda_version", "nccl_version", "vllm_version", "vllm_cli")
    for key in keys:
        if key not in saved:
            raise TargetValidationError(f"Saved execution environment lacks {key}; start a new session")
        if saved[key] != current.get(key):
            raise TargetValidationError(f"Execution environment changed ({key}); start a new session")
    if saved.get("schema_version", 1) >= 2:
        for key in ("runtime_python", "runtime_pythonpath", "benchmark_backend"):
            if key not in saved or saved[key] != current.get(key):
                raise TargetValidationError(f"Execution environment changed ({key}); start a new session")
