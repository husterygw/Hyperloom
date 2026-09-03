# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution-target registry and hardware contracts.

Targets describe a complete runtime boundary (vendor, benchmark backend and
capabilities).  They intentionally do not extend the AMD ``--gpu-type`` board
enum: a CUDA host is a different execution target, not an AMD runner alias.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


TARGET_ENV = "HYPERLOOM_TARGET"
TARGET_RUNTIME_ENV = "HYPERLOOM_TARGET_RUNTIME"
HARDWARE_FINGERPRINT_ENV = "HYPERLOOM_HARDWARE_FINGERPRINT"
DEFAULT_TARGET = "amd_auto"
NVIDIA_LOCAL_TARGET = "nvidia_rtx4090_8x_local"


@dataclass(frozen=True)
class TargetCapabilities:
    """Feature families admitted for a target."""

    baseline: bool = True
    config_explore: bool = True
    sweep: bool = True
    report: bool = True
    profile: bool = True
    source_patch: bool = True
    kernel_patch: bool = True
    quantization: bool = True
    multinode: bool = True

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class TargetDescriptor:
    """Immutable runtime target description."""

    target_id: str
    vendor: str
    runtime: str
    benchmark_backend: str
    capabilities: TargetCapabilities
    expected_gpu_name: str = ""
    expected_gpu_count: int = 0
    expected_compute_capability: str = ""
    min_memory_mib: int = 0
    required_vllm_server_flags: tuple[str, ...] = ()
    required_vllm_bench_flags: tuple[str, ...] = ()
    default_cuda_home: str = ""
    experimental: bool = False


@dataclass(frozen=True)
class NvidiaDevice:
    """Stable identity and placement data for one NVIDIA device."""

    index: int
    uuid: str
    name: str
    memory_mib: int
    compute_capability: str
    pci_bus_id: str
    numa_node: int | None


_ALL = TargetCapabilities()
_NVIDIA_MVP = TargetCapabilities(
    baseline=True,
    config_explore=True,
    sweep=True,
    report=True,
    profile=False,
    source_patch=False,
    kernel_patch=False,
    quantization=False,
    multinode=False,
)

# The CUDA runner owns these flags and cannot operate safely without them.
# Compatibility is determined by the installed CLI surface, not by an exact
# vLLM package version: vLLM releases can advance while retaining this contract.
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

_TARGETS: dict[str, TargetDescriptor] = {
    DEFAULT_TARGET: TargetDescriptor(
        target_id=DEFAULT_TARGET,
        vendor="amd",
        runtime="rocm",
        benchmark_backend="magpie",
        capabilities=_ALL,
    ),
    NVIDIA_LOCAL_TARGET: TargetDescriptor(
        target_id=NVIDIA_LOCAL_TARGET,
        vendor="nvidia",
        runtime="cuda",
        benchmark_backend="vllm_cuda",
        capabilities=_NVIDIA_MVP,
        expected_gpu_name="NVIDIA GeForce RTX 4090",
        expected_gpu_count=8,
        expected_compute_capability="8.9",
        min_memory_mib=24000,
        required_vllm_server_flags=VLLM_CUDA_REQUIRED_SERVER_FLAGS,
        required_vllm_bench_flags=VLLM_CUDA_REQUIRED_BENCH_FLAGS,
        default_cuda_home="/usr/local/cuda-13.0",
        experimental=True,
    ),
}


class TargetValidationError(RuntimeError):
    """The selected target cannot run on the current host/configuration."""


def target_names() -> tuple[str, ...]:
    """Return registered target ids in stable order."""
    return tuple(_TARGETS)


def get_target(target_id: str) -> TargetDescriptor:
    """Resolve a target id or raise a useful error."""
    key = str(target_id or "").strip().lower() or DEFAULT_TARGET
    try:
        return _TARGETS[key]
    except KeyError as exc:
        raise TargetValidationError(
            f"unknown target {target_id!r}; expected one of {', '.join(target_names())}"
        ) from exc


def resolve_target(requested: str | None = None, *, persisted: str | None = None) -> TargetDescriptor:
    """Resolve CLI > environment > persisted state > AMD default."""
    selected = (
        str(requested or "").strip()
        or str(os.environ.get(TARGET_ENV) or "").strip()
        or str(persisted or "").strip()
        or DEFAULT_TARGET
    )
    return get_target(selected)


def is_cuda_target(value: str | TargetDescriptor | None) -> bool:
    """Return whether ``value`` identifies a CUDA target."""
    if isinstance(value, TargetDescriptor):
        return value.runtime == "cuda"
    try:
        return get_target(str(value or DEFAULT_TARGET)).runtime == "cuda"
    except TargetValidationError:
        return False


def _decode_nvml_text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _numa_node_for_pci(pci_bus_id: str) -> int | None:
    bus_id = pci_bus_id.strip().lower()
    domain, sep, rest = bus_id.partition(":")
    if sep and len(domain) > 4:
        bus_id = f"{domain[-4:]}:{rest}"
    path = Path("/sys/bus/pci/devices") / bus_id / "numa_node"
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return value if value >= 0 else None


def discover_nvidia_devices() -> tuple[NvidiaDevice, ...]:
    """Discover NVIDIA devices through NVML, importing it lazily."""
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on host packaging
        raise TargetValidationError("NVIDIA target requires nvidia-ml-py (pynvml); install the [nvidia] extra") from exc

    try:
        pynvml.nvmlInit()
        count = int(pynvml.nvmlDeviceGetCount())
        devices: list[NvidiaDevice] = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus_id = _decode_nvml_text(getattr(pci, "busId", "")).lower()
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                NvidiaDevice(
                    index=index,
                    uuid=_decode_nvml_text(pynvml.nvmlDeviceGetUUID(handle)),
                    name=_decode_nvml_text(pynvml.nvmlDeviceGetName(handle)),
                    memory_mib=int(memory.total // (1024 * 1024)),
                    compute_capability=f"{int(major)}.{int(minor)}",
                    pci_bus_id=pci_bus_id,
                    numa_node=_numa_node_for_pci(pci_bus_id),
                )
            )
        return tuple(devices)
    except Exception as exc:  # noqa: BLE001 - normalized as a target error
        raise TargetValidationError(f"NVML NVIDIA discovery failed: {type(exc).__name__}: {exc}") from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - shutdown is best effort
            pass


def _topology_diagnostic() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def build_hardware_fingerprint(
    target: TargetDescriptor,
    devices: tuple[NvidiaDevice, ...],
) -> dict[str, Any]:
    """Build the JSON-persisted hardware identity for a CUDA session."""
    payload: dict[str, Any] = {
        "target_id": target.target_id,
        "runtime": target.runtime,
        "devices": [asdict(device) for device in devices],
        "topology": _topology_diagnostic(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _rehash_fingerprint(payload: dict[str, Any]) -> None:
    canonical_payload = {key: value for key, value in payload.items() if key != "sha256"}
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _nvidia_driver_version() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    versions = sorted({line.strip() for line in proc.stdout.splitlines() if line.strip()})
    return ",".join(versions)


def _nvcc_release(cuda_home: Path) -> str:
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise TargetValidationError(f"CUDA compiler not found at {nvcc}")
    try:
        proc = subprocess.run(
            [str(nvcc), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TargetValidationError(f"cannot execute CUDA compiler {nvcc}: {exc}") from exc
    text = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0 or "release 13.0" not in text:
        raise TargetValidationError(f"{nvcc} is not the required CUDA 13.0 compiler (rc={proc.returncode})")
    return text.strip().splitlines()[-1] if text.strip() else "CUDA 13.0"


def _probe_vllm_cli_flags(*subcommand: str) -> set[str]:
    """Return long options exposed by one installed vLLM CLI command.

    Recent vLLM releases group most serving flags by config class, so the
    probe uses ``--help=all``. It runs through ``sys.executable`` to verify the
    exact interpreter that will launch the CUDA runner.
    """
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", *subcommand, "--help=all"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
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


def _validate_vllm_cli(target: TargetDescriptor) -> dict[str, list[str]]:
    """Verify the command surface required by the native CUDA runner."""
    server_flags = _probe_vllm_cli_flags("serve")
    bench_flags = _probe_vllm_cli_flags("bench", "serve")
    missing_server = sorted(set(target.required_vllm_server_flags) - server_flags)
    missing_bench = sorted(set(target.required_vllm_bench_flags) - bench_flags)
    errors: list[str] = []
    if missing_server:
        errors.append("vLLM serve is missing required flag(s): " + ", ".join(missing_server))
    if missing_bench:
        errors.append("vLLM bench serve is missing required flag(s): " + ", ".join(missing_bench))
    if errors:
        raise TargetValidationError("; ".join(errors))
    return {"server_flags": sorted(server_flags), "bench_flags": sorted(bench_flags)}


def validate_nvidia_host(target: TargetDescriptor) -> dict[str, Any]:
    """Fail closed unless the current interpreter and host match ``target``."""
    if target.runtime != "cuda":
        return {}
    devices = discover_nvidia_devices()
    errors: list[str] = []
    if len(devices) != target.expected_gpu_count:
        errors.append(f"expected {target.expected_gpu_count} GPUs, found {len(devices)}")
    for device in devices:
        if device.name != target.expected_gpu_name:
            errors.append(f"GPU {device.index} is {device.name!r}, expected {target.expected_gpu_name!r}")
        if device.compute_capability != target.expected_compute_capability:
            errors.append(
                f"GPU {device.index} compute capability is {device.compute_capability}, "
                f"expected {target.expected_compute_capability}"
            )
        if device.memory_mib < target.min_memory_mib:
            errors.append(
                f"GPU {device.index} memory is {device.memory_mib} MiB, expected at least {target.min_memory_mib} MiB"
            )
    torch_version = ""
    torch_cuda_version = ""
    nccl_version = ""
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_cuda_version = str(torch.version.cuda or "")
        try:
            nccl_version = str(torch.cuda.nccl.version() or "")
        except Exception:  # noqa: BLE001 - version metadata is diagnostic
            nccl_version = ""
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
        elif int(torch.cuda.device_count()) != target.expected_gpu_count:
            errors.append(f"torch sees {torch.cuda.device_count()} GPUs, expected {target.expected_gpu_count}")
    except ImportError:
        errors.append("torch is not importable from the selected interpreter")
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = ""
    vllm_cli: dict[str, list[str]] = {}
    if not vllm_version:
        errors.append("vLLM is not installed in the selected interpreter")
    else:
        try:
            vllm_cli = _validate_vllm_cli(target)
        except TargetValidationError as exc:
            errors.append(str(exc))

    explicit_cuda_home = str(os.environ.get("CUDA_HOME") or "").strip()
    cuda_home = Path(explicit_cuda_home or target.default_cuda_home).expanduser().resolve()
    try:
        nvcc_version = _nvcc_release(cuda_home)
    except TargetValidationError as exc:
        errors.append(str(exc))
        nvcc_version = ""
    if errors:
        raise TargetValidationError("NVIDIA target preflight failed:\n- " + "\n- ".join(errors))

    fingerprint = build_hardware_fingerprint(target, devices)
    fingerprint["cuda_home"] = str(cuda_home)
    fingerprint["nvcc"] = nvcc_version
    fingerprint["vllm_version"] = vllm_version
    fingerprint["vllm_cli"] = vllm_cli
    fingerprint["driver_version"] = _nvidia_driver_version()
    fingerprint["torch_version"] = torch_version
    fingerprint["torch_cuda_version"] = torch_cuda_version
    fingerprint["nccl_version"] = nccl_version
    _rehash_fingerprint(fingerprint)
    return fingerprint


def configure_target_environment(
    target: TargetDescriptor,
    *,
    fingerprint: Mapping[str, Any] | None = None,
) -> None:
    """Publish the target contract to Hyperloom and its child processes."""
    os.environ[TARGET_ENV] = target.target_id
    os.environ[TARGET_RUNTIME_ENV] = target.runtime
    if target.runtime != "cuda":
        return
    os.environ["HYPERLOOM_BENCHMARK_BACKEND"] = target.benchmark_backend
    # The native CUDA runner already owns physical GPU leases and its complete
    # server process group.  Wrapping it in the legacy single-node Ray serving
    # actor loses the explicit CUDA_VISIBLE_DEVICES mapping and can fail before
    # the runner starts when no Ray head exists.  This config-only target is
    # single-node by contract, so keep benchmark execution local.
    os.environ["INFERENCE_OPTIMIZER_RAY_EXEC"] = "0"
    cuda_home = str((fingerprint or {}).get("cuda_home") or target.default_cuda_home)
    os.environ["CUDA_HOME"] = cuda_home
    cuda_bin = str(Path(cuda_home) / "bin")
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    os.environ["PATH"] = os.pathsep.join([cuda_bin, *[part for part in path_parts if part != cuda_bin]])
    # Empty is an explicit allow-list: discover plugins for diagnostics but load none.
    os.environ["VLLM_PLUGINS"] = ""
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    if fingerprint:
        os.environ[HARDWARE_FINGERPRINT_ENV] = json.dumps(dict(fingerprint), sort_keys=True)


def validate_target_arguments(args: Any, target: TargetDescriptor) -> None:
    """Apply deterministic target policy to parsed CLI arguments."""
    if target.runtime != "cuda":
        return
    if int(getattr(args, "nodes", 1) or 1) != 1:
        raise TargetValidationError(f"target {target.target_id} is single-node only")
    framework = str(getattr(args, "framework", None) or "").strip().lower()
    if framework and framework != "vllm":
        raise TargetValidationError(f"target {target.target_id} requires --framework vllm")
    if getattr(args, "gpu_type", None):
        raise TargetValidationError("--gpu-type is AMD-only and cannot be combined with a NVIDIA target")
    if getattr(args, "quantize", None) or str(getattr(args, "quantize_scheme", "") or "") not in ("", "none"):
        raise TargetValidationError(f"target {target.target_id} does not support quantization in P0-P2")

    args.framework = "vllm"
    args.no_kernel = True
    args.enable_roofline = False
    # Keep OPTIMIZE enabled for config exploration. The source arm is marked
    # exhausted in SharedState and PolicyGate denies patch-capable actions.
    args.no_framework_agent = False
    args.no_framework_local_explore = True
    args.enablement = "off"
    args.no_warm_replay = True
    args.no_eval = True


__all__ = [
    "DEFAULT_TARGET",
    "HARDWARE_FINGERPRINT_ENV",
    "NVIDIA_LOCAL_TARGET",
    "NvidiaDevice",
    "TARGET_ENV",
    "TARGET_RUNTIME_ENV",
    "TargetCapabilities",
    "TargetDescriptor",
    "TargetValidationError",
    "VLLM_CUDA_REQUIRED_BENCH_FLAGS",
    "VLLM_CUDA_REQUIRED_SERVER_FLAGS",
    "build_hardware_fingerprint",
    "configure_target_environment",
    "discover_nvidia_devices",
    "get_target",
    "is_cuda_target",
    "resolve_target",
    "target_names",
    "validate_nvidia_host",
    "validate_target_arguments",
]
