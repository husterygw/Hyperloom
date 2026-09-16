# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution-target registry and hardware contracts.

Targets describe the GPU platform (vendor, runtime and capabilities).  They intentionally do not extend the AMD ``--gpu-type`` board
enum: a CUDA host is a different execution target, not an AMD runner alias.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


TARGET_ENV = "HYPERLOOM_TARGET"
TARGET_RUNTIME_ENV = "HYPERLOOM_TARGET_RUNTIME"
HARDWARE_FINGERPRINT_ENV = "HYPERLOOM_HARDWARE_FINGERPRINT"
DEFAULT_TARGET = "amd_auto"
NVIDIA_CUDA_TARGET = "nvidia_cuda"
NVIDIA_LOCAL_TARGET = "nvidia_rtx4090_8x_local"  # Accepted legacy CLI/session alias.


@dataclass(frozen=True)
class TargetCapabilities:
    """Feature families admitted for a target."""

    baseline: bool = True
    config_explore: bool = True
    sweep: bool = True
    report: bool = True
    profile: bool = True
    roofline: bool = True
    trace_analysis: bool = True
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
    capabilities: TargetCapabilities
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
    roofline=False,
    trace_analysis=False,
    source_patch=False,
    kernel_patch=False,
    quantization=False,
    multinode=False,
)

_TARGETS: dict[str, TargetDescriptor] = {
    DEFAULT_TARGET: TargetDescriptor(DEFAULT_TARGET, "amd", "rocm", _ALL),
    NVIDIA_CUDA_TARGET: TargetDescriptor(NVIDIA_CUDA_TARGET, "nvidia", "cuda", _NVIDIA_MVP, experimental=True),
}


class TargetValidationError(RuntimeError):
    """The selected target cannot run on the current host/configuration."""


def target_names() -> tuple[str, ...]:
    """Return registered target ids in stable order."""
    return (*_TARGETS, NVIDIA_LOCAL_TARGET)


def get_target(target_id: str) -> TargetDescriptor:
    """Resolve a target id or raise a useful error."""
    key = str(target_id or "").strip().lower() or DEFAULT_TARGET
    if key == NVIDIA_LOCAL_TARGET:
        key = NVIDIA_CUDA_TARGET
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
    try:
        proc = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise TargetValidationError(f"cannot execute CUDA compiler {nvcc}: {exc}") from exc
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


def discover_cuda_toolkit() -> dict[str, str]:
    """Compiler discovery is diagnostic; serving does not require nvcc."""
    explicit = os.environ.get("CUDA_HOME", "").strip()
    nvcc = shutil.which("nvcc")
    root = (
        Path(explicit).expanduser()
        if explicit
        else (Path(nvcc).resolve().parent.parent if nvcc else Path("/usr/local/cuda"))
    )
    result = {"cuda_home": str(root.resolve()) if explicit or root.is_dir() else "", "nvcc": ""}
    if (root / "bin" / "nvcc").is_file():
        try:
            result["nvcc"] = _nvcc_release(root)
        except TargetValidationError as exc:
            result["toolkit_error"] = str(exc)
    return result


# Probe the CUDA driver in a fresh process: importing a framework must not fix
# the parent's device enumeration before the final visibility mask is chosen.
_CUDA_ENUMERATOR = r"""
import ctypes, json, uuid
cuda = ctypes.CDLL("libcuda.so.1")
def check(code):
    if code:
        raise RuntimeError("CUDA driver error %s" % code)
check(cuda.cuInit(0))
count = ctypes.c_int()
version = ctypes.c_int()
check(cuda.cuDeviceGetCount(ctypes.byref(count)))
check(cuda.cuDriverGetVersion(ctypes.byref(version)))
rows = []
for ordinal in range(count.value):
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), ordinal))
    identity = (ctypes.c_ubyte * 16)()
    uuid_fn = getattr(cuda, "cuDeviceGetUuid_v2", cuda.cuDeviceGetUuid)
    check(uuid_fn(ctypes.byref(identity), device))
    rows.append({"uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identity))), "cuda_index": ordinal})
print(json.dumps({"devices": rows, "cuda_driver_api_version": version.value}))
"""


def probe_cuda_devices(*, unmasked: bool = False) -> dict[str, Any]:
    env = dict(os.environ)
    if unmasked:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CUDA_ENUMERATOR], env=env, capture_output=True, text=True, timeout=30, check=True
        )
        return json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise TargetValidationError(f"CUDA device enumeration failed: {detail}") from exc


def select_cuda_devices(rows: list[dict[str, Any]], mask: str) -> list[dict[str, Any]]:
    """Resolve a CUDA mask against actual driver ordinals/UUIDs, never NVML indices."""
    if not mask.strip() or mask.strip() == "-1":
        return []
    selected = []
    for token in mask.split(","):
        token = token.strip()
        if token.startswith("MIG-"):
            raise TargetValidationError("MIG partition scheduling is not implemented")
        if token.isdecimal():
            matches = [r for r in rows if r["cuda_index"] == int(token)]
        elif token.startswith("GPU-"):
            matches = [r for r in rows if r["uuid"].startswith(token)]
        else:
            matches = []
        if len(matches) != 1:
            raise TargetValidationError(f"invalid or ambiguous CUDA_VISIBLE_DEVICES entry: {token!r}")
        if any(r["uuid"] == matches[0]["uuid"] for r in selected):
            raise TargetValidationError("CUDA_VISIBLE_DEVICES contains duplicate devices")
        selected.append(matches[0])
    return selected


def allocate_cuda_devices(
    fingerprint: Mapping[str, Any], world_size: int, mask: str | None = None
) -> list[dict[str, Any]]:
    pool = fingerprint.get("devices") or []
    if not pool or any("cuda_index" not in r for r in pool):
        raise TargetValidationError("CUDA pool is missing validated ordinal/UUID mapping; repeat preflight")
    selected = pool
    if mask is not None:
        selected = select_cuda_devices(fingerprint.get("cuda_devices") or pool, mask)
        allowed = {r["uuid"]: r for r in pool}
        if any(r["uuid"] not in allowed for r in selected):
            raise TargetValidationError("candidate CUDA_VISIBLE_DEVICES escapes the allowed GPU pool")
        selected = [allowed[r["uuid"]] for r in selected]
    if world_size <= 0 or len(selected) < world_size:
        raise TargetValidationError(
            f"CUDA_VISIBLE_DEVICES pool has {len(selected)} devices but TP*PP requires {world_size}"
        )
    selected = selected[:world_size]
    if len({r["compute_capability"] for r in selected}) != 1:
        raise TargetValidationError("cross-architecture CUDA parallel execution is not implemented")
    return selected


def validate_nvidia_host(target: TargetDescriptor, *, capacity: int | None = None) -> dict[str, Any]:
    """Discover a usable CUDA pool independently of any inference framework."""
    if target.runtime != "cuda":
        return {}
    inventory = discover_nvidia_devices()
    by_uuid = {device.uuid: asdict(device) for device in inventory}
    unmasked = probe_cuda_devices(unmasked=True)
    visible = probe_cuda_devices()
    all_cuda = unmasked["devices"]
    actual = visible["devices"]
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        expected = select_cuda_devices(all_cuda, os.environ["CUDA_VISIBLE_DEVICES"])
        if [r["uuid"] for r in actual] != [r["uuid"] for r in expected]:
            raise TargetValidationError("CUDA visibility does not match the requested device order")
    if not actual:
        raise TargetValidationError("NVIDIA target has no visible CUDA devices")
    if capacity is not None and (capacity <= 0 or capacity > len(actual)):
        raise TargetValidationError(f"GPU capacity {capacity} exceeds or invalidates visible pool of {len(actual)}")
    base_indices = {r["uuid"]: r["cuda_index"] for r in all_cuda}
    rows = []
    for logical, device in enumerate(actual[:capacity]):
        uuid = device["uuid"]
        if uuid not in by_uuid:
            raise TargetValidationError(f"CUDA device {uuid} has no physical NVML identity; MIG is not supported")
        rows.append({**by_uuid[uuid], "cuda_index": base_indices[uuid], "logical_index": logical})
    payload = build_hardware_fingerprint(target, inventory)
    payload.update(
        schema_version=2,
        inventory=payload["devices"],
        devices=rows,
        cuda_devices=all_cuda,
        visible_uuids=[r["uuid"] for r in actual],
        cuda_driver_api_version=visible["cuda_driver_api_version"],
        driver_version=_nvidia_driver_version(),
        **discover_cuda_toolkit(),
    )
    _rehash_fingerprint(payload)
    return payload


def validate_resume_environment(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Compare selected identities and software, not transient NVML ordinals."""

    def identity(value: Mapping[str, Any]) -> dict[str, Any]:
        rows = value.get("devices") or []
        if not rows or any(not r.get("uuid") for r in rows):
            raise TargetValidationError("Saved NVIDIA session lacks device identities; start a new session")
        keys = ("uuid", "name", "compute_capability", "memory_mib", "pci_bus_id", "numa_node")
        result = {"devices": [{k: r.get(k) for k in keys} for r in rows]}
        for key in ("driver_version", "nvcc", "cuda_home"):
            if key not in value:
                raise TargetValidationError(f"Saved NVIDIA session lacks {key}; start a new session")
            result[key] = value[key]
        return result

    if identity(saved) != identity(current):
        raise TargetValidationError("NVIDIA device pool or execution environment changed; start a new session")


def configure_target_environment(target: TargetDescriptor, *, fingerprint: Mapping[str, Any] | None = None) -> None:
    """Publish platform identity and tool paths, without framework-specific policy."""
    os.environ[TARGET_ENV] = target.target_id
    os.environ[TARGET_RUNTIME_ENV] = target.runtime
    if target.runtime != "cuda":
        return
    cuda_home = str((fingerprint or {}).get("cuda_home") or "")
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        cuda_bin = str(Path(cuda_home) / "bin")
        path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and p != cuda_bin]
        os.environ["PATH"] = os.pathsep.join([cuda_bin, *path_parts])
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    if fingerprint:
        os.environ[HARDWARE_FINGERPRINT_ENV] = json.dumps(dict(fingerprint), sort_keys=True)


def effective_target_capabilities(
    target: TargetDescriptor, level: str = "config", backend: str = "torch", *, roofline: bool = True
) -> dict[str, bool]:
    """Grant only implemented, explicitly selected features for a new session."""
    capabilities = target.capabilities.to_dict()
    if target.runtime == "cuda":
        if level not in ("config", "profile"):
            raise TargetValidationError(f"NVIDIA optimization level {level!r} is not implemented")
        capabilities["profile"] = level == "profile"
        capabilities["trace_analysis"] = level == "profile" and backend == "nsys"
        capabilities["roofline"] = capabilities["trace_analysis"] and roofline
    return capabilities


def validate_target_arguments(args: Any, target: TargetDescriptor) -> None:
    """Apply deterministic target policy to parsed CLI arguments."""
    level = getattr(args, "optimization_level", None)
    backend = getattr(args, "profile_backend", None)
    if target.runtime != "cuda":
        if level is not None or backend is not None:
            raise TargetValidationError("--optimization-level and --profile-backend currently require a NVIDIA target")
        return
    args.optimization_level = level or "config"
    args.profile_backend = backend or "torch"
    if args.profile_backend not in ("torch", "nsys"):
        raise TargetValidationError("unsupported NVIDIA profile backend")
    if args.profile_backend == "nsys" and args.optimization_level != "profile":
        raise TargetValidationError("NVIDIA nsys requires --optimization-level profile")
    args.target_capabilities = effective_target_capabilities(
        target, args.optimization_level, args.profile_backend, roofline=getattr(args, "enable_roofline", True)
    )
    if int(getattr(args, "nodes", 1) or 1) != 1:
        raise TargetValidationError(f"target {target.target_id} is single-node only")
    if getattr(args, "gpu_type", None):
        raise TargetValidationError("--gpu-type is AMD-only and cannot be combined with a NVIDIA target")
    if getattr(args, "quantize", None) or str(getattr(args, "quantize_scheme", "") or "") not in ("", "none"):
        raise TargetValidationError(f"target {target.target_id} does not support quantization")

    args.no_kernel = True
    args.enable_roofline = args.target_capabilities["roofline"]
    # Keep OPTIMIZE enabled for config exploration. The source arm is marked
    # exhausted in SharedState and PolicyGate denies patch-capable actions.
    args.no_framework_agent = False
    args.no_framework_local_explore = True
    args.enablement = "off"
    args.no_warm_replay = True


__all__ = [
    "DEFAULT_TARGET",
    "NVIDIA_CUDA_TARGET",
    "NVIDIA_LOCAL_TARGET",
    "HARDWARE_FINGERPRINT_ENV",
    "NvidiaDevice",
    "TARGET_ENV",
    "TARGET_RUNTIME_ENV",
    "TargetCapabilities",
    "TargetDescriptor",
    "TargetValidationError",
    "build_hardware_fingerprint",
    "configure_target_environment",
    "discover_nvidia_devices",
    "get_target",
    "is_cuda_target",
    "resolve_target",
    "target_names",
    "validate_nvidia_host",
    "validate_target_arguments",
    "validate_resume_environment",
    "select_cuda_devices",
]
