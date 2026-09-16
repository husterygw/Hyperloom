"""Portable CUDA platform contracts; these tests require no GPU or framework."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer import target_registry as registry
from hyperloom.orchestrator.actions.executors import benchmark_backend, cuda_nsight, vllm_cuda_preflight
from hyperloom.orchestrator.actions.executors.cuda_host_lock import CudaHostLock
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture(autouse=True)
def clean_visibility(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)


def hardware(monkeypatch, count=4, name="NVIDIA H100", cc="9.0", visible=None):
    # CUDA enumeration is deliberately the reverse of NVML ordering.
    inventory = tuple(
        registry.NvidiaDevice(i, f"GPU-{i:04d}", name, 80000, cc, f"0000:{i + 1:02x}:00.0", i % 2) for i in range(count)
    )
    all_cuda = [{"uuid": row.uuid, "cuda_index": i} for i, row in enumerate(reversed(inventory))]
    visible = all_cuda if visible is None else [all_cuda[i] for i in visible]
    monkeypatch.setattr(registry, "discover_nvidia_devices", lambda: inventory)
    monkeypatch.setattr(
        registry,
        "probe_cuda_devices",
        lambda *, unmasked=False: {"devices": all_cuda if unmasked else visible, "cuda_driver_api_version": 12080},
    )
    monkeypatch.setattr(registry, "_nvidia_driver_version", lambda: "driver")
    monkeypatch.setattr(registry, "_topology_diagnostic", lambda: "topology")
    monkeypatch.setattr(registry, "discover_cuda_toolkit", lambda: {"cuda_home": "", "nvcc": ""})
    return registry.get_target("nvidia_cuda")


@pytest.mark.parametrize("count", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "name,cc", [("NVIDIA RTX 4090", "8.9"), ("NVIDIA A100", "8.0"), ("NVIDIA H100", "9.0"), ("NVIDIA B200", "10.0")]
)
def test_platform_discovers_any_model_and_count_without_framework(monkeypatch, count, name, cc):
    target = hardware(monkeypatch, count, name, cc)
    result = registry.validate_nvidia_host(target)
    assert len(result["devices"]) == count
    assert result["devices"][0]["index"] == count - 1
    assert result["devices"][0]["cuda_index"] == 0
    assert result["devices"][0]["name"] == name
    assert "vllm_version" not in result
    assert result["nvcc"] == ""


def test_mask_and_capacity_preserve_actual_cuda_order(monkeypatch):
    target = hardware(monkeypatch, visible=[2, 0, 1])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0,1")
    result = registry.validate_nvidia_host(target, capacity=2)
    assert [r["index"] for r in result["devices"]] == [1, 3]
    assert [r["logical_index"] for r in result["devices"]] == [0, 1]
    selected = registry.allocate_cuda_devices(result, 2, "GPU-0003,GPU-0001")
    assert [r["index"] for r in selected] == [3, 1]
    with pytest.raises(registry.TargetValidationError, match="escapes"):
        registry.allocate_cuda_devices(result, 1, "1")
    with pytest.raises(registry.TargetValidationError, match=r"TP\*PP"):
        registry.allocate_cuda_devices(result, 3)


@pytest.mark.parametrize("mask", ["", "-1"])
def test_empty_mask_never_expands_to_host(monkeypatch, mask):
    target = hardware(monkeypatch, visible=[])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(registry.TargetValidationError, match="no visible"):
        registry.validate_nvidia_host(target)


@pytest.mark.parametrize("mask", ["0,0", "99", "GPU-", "0,invalid", "MIG-123"])
def test_invalid_masks_rejected_even_if_cuda_silently_truncates(monkeypatch, mask):
    target = hardware(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(registry.TargetValidationError):
        registry.validate_nvidia_host(target)


@pytest.mark.parametrize("capacity", [-1, 0, 5])
def test_invalid_capacity(monkeypatch, capacity):
    with pytest.raises(registry.TargetValidationError, match="capacity"):
        registry.validate_nvidia_host(hardware(monkeypatch), capacity=capacity)


def test_mixed_host_allows_homogeneous_subset(monkeypatch):
    target = hardware(monkeypatch)
    inventory = registry.discover_nvidia_devices()
    monkeypatch.setattr(
        registry, "discover_nvidia_devices", lambda: (replace(inventory[0], compute_capability="8.0"), *inventory[1:])
    )
    result = registry.validate_nvidia_host(target)
    assert len(registry.allocate_cuda_devices(result, 2)) == 2
    with pytest.raises(registry.TargetValidationError, match="cross-architecture"):
        registry.allocate_cuda_devices(result, 4)


@pytest.mark.parametrize("release", ["12.4", "12.8", "13.0"])
def test_toolkit_version_is_recorded_not_pinned(monkeypatch, tmp_path, release):
    root = tmp_path / "cuda"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "nvcc").touch()
    monkeypatch.setenv("CUDA_HOME", str(root))
    monkeypatch.setattr(registry.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=f"release {release}"))
    assert registry.discover_cuda_toolkit()["nvcc"] == f"release {release}"


def test_explicit_toolkit_path_does_not_fall_back_to_other_version(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "no-compiler"))
    monkeypatch.setattr(registry.shutil, "which", lambda _: "/different/cuda/bin/nvcc")
    result = registry.discover_cuda_toolkit()
    assert result["cuda_home"] == str(tmp_path / "no-compiler")
    assert result["nvcc"] == ""


def test_backend_selection_is_separate_and_does_not_fall_back(monkeypatch):
    assert benchmark_backend.select_platform_backend("cuda", "vllm") == "vllm_cuda"
    assert benchmark_backend.select_platform_backend("rocm", "sglang") == "magpie"
    for framework, requested in [("sglang", ""), ("vllm", "magpie"), ("vllm", "typo")]:
        with pytest.raises(registry.TargetValidationError, match="No CUDA execution backend"):
            benchmark_backend.select_platform_backend("cuda", framework, requested)
    monkeypatch.setenv("HYPERLOOM_TARGET_RUNTIME", "cuda")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "typo")
    with pytest.raises(registry.TargetValidationError):
        benchmark_backend.resolve_backend_name()


def test_backend_checks_actual_interpreter_and_required_flags(monkeypatch):
    fp = registry.validate_nvidia_host(hardware(monkeypatch))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "-c" in command:
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "vllm_version": "v",
                        "torch_version": "t",
                        "torch_cuda_version": "12.8",
                        "nccl_version": "n",
                        "device_count": 4,
                    }
                )
            )
        flags = (
            vllm_cuda_preflight.VLLM_CUDA_REQUIRED_SERVER_FLAGS
            if command[3] == "serve"
            else vllm_cuda_preflight.VLLM_CUDA_REQUIRED_BENCH_FLAGS
        )
        return SimpleNamespace(stdout=" ".join(flags), stderr="", returncode=0)

    monkeypatch.setattr(vllm_cuda_preflight.subprocess, "run", run)
    result = vllm_cuda_preflight.validate_execution_stack(fp, python_exe="/runtime/python", pythonpath="/source")
    assert all(c[0][0] == "/runtime/python" for c in calls)
    assert all(c[1]["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-0003,GPU-0002,GPU-0001,GPU-0000" for c in calls)
    assert result["runtime_python"] == "/runtime/python"
    monkeypatch.setattr(vllm_cuda_preflight, "_probe_vllm_cli_flags", lambda *a, **kw: set())
    with pytest.raises(registry.TargetValidationError, match="missing required"):
        vllm_cuda_preflight.validate_execution_stack(fp)


def full_identity(monkeypatch):
    fp = registry.validate_nvidia_host(hardware(monkeypatch))
    fp.update(torch_version="t", torch_cuda_version="c", nccl_version="n", vllm_version="v", vllm_cli={})
    return fp


def test_resume_accepts_alias_and_index_renumbering_but_not_changed_identity(monkeypatch):
    current = full_identity(monkeypatch)
    saved = copy.deepcopy(current)
    saved["target_id"] = registry.NVIDIA_LOCAL_TARGET
    for row in saved["devices"]:
        row["index"] += 10
    registry.validate_resume_environment(saved, current)
    for key, value in [("driver_version", "different")]:
        changed = {**saved, key: value}
        with pytest.raises(registry.TargetValidationError, match="changed"):
            registry.validate_resume_environment(changed, current)
    saved["devices"].reverse()
    with pytest.raises(registry.TargetValidationError, match="changed"):
        registry.validate_resume_environment(saved, current)
    with pytest.raises(registry.TargetValidationError, match="lacks"):
        registry.validate_resume_environment({}, current)


def test_legacy_state_migration_canonicalizes_target_without_inventing_mapping():
    state = SharedState.from_dict(
        {"schema_version": 9, "target_id": registry.NVIDIA_LOCAL_TARGET, "benchmark_backend": "vllm_cuda"}
    )
    assert state.target_id == "nvidia_cuda"
    assert state.target_runtime == "cuda"
    assert state.device_pool == []
    assert state.benchmark_backend == "vllm_cuda"


def test_metric_selection_intersects_actual_device_capabilities(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
        text = "gpu__time_duration.sum\ndram__bytes.sum.per_second\n"
        if len(calls) == 1:
            text += "dram__bytes.sum.peak_sustained\n"
        return SimpleNamespace(stdout=text, returncode=0, stderr="")

    monkeypatch.setattr(cuda_nsight.subprocess, "run", run)
    result = cuda_nsight.query_roofline_metrics(["compute"], ["GPU-a", "GPU-b"])
    assert calls == ["GPU-a"] * 3 + ["GPU-b"] * 3
    assert result["selected"] == ["gpu__time_duration.sum", "dram__bytes.sum.per_second"]
    assert "dram__bytes.sum.peak_sustained" in result["unavailable"]
    assert cuda_nsight.roofline_point({"gpu__time_duration.sum": 1})["status"] == "unavailable"


def test_host_lock_excludes_other_processes_and_releases(tmp_path):
    path = tmp_path / "host.lock"
    with CudaHostLock("first", path):
        with pytest.raises(RuntimeError, match="reserved"):
            with CudaHostLock("second", path):
                pass
    with CudaHostLock("third", path):
        pass


def test_resume_backend_identity_rejects_changed_software(monkeypatch):
    current = full_identity(monkeypatch)
    current.update(runtime_python="/python", runtime_pythonpath="", benchmark_backend="vllm_cuda")
    legacy = copy.deepcopy(current)
    legacy.pop("schema_version")
    legacy.pop("runtime_python")
    vllm_cuda_preflight.validate_execution_identity(legacy, current)
    for field in ("torch_version", "runtime_python", "runtime_pythonpath"):
        changed = {**current, field: "different"}
        with pytest.raises(registry.TargetValidationError, match="changed"):
            vllm_cuda_preflight.validate_execution_identity(changed, current)


def test_scheduler_and_specialists_use_capped_physical_pool(monkeypatch):
    from hyperloom.orchestrator.bus.gpu_pool import resolve_gpu_specialist_devices, resolve_whole_machine_devices
    from hyperloom.orchestrator.policy.gate import detect_gpu_count

    fp = registry.validate_nvidia_host(hardware(monkeypatch), capacity=2)
    monkeypatch.setenv("HYPERLOOM_TARGET_RUNTIME", "cuda")
    monkeypatch.setenv("HYPERLOOM_HARDWARE_FINGERPRINT", json.dumps(fp))
    assert detect_gpu_count() == 2
    assert [d.index for d in resolve_whole_machine_devices()] == [3, 2]
    assert resolve_gpu_specialist_devices(8, serving_tp=1) == [2]
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES", "0")
    with pytest.raises(ValueError, match="escape"):
        resolve_gpu_specialist_devices(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_name", ["nvidia_cuda", "nvidia_rtx4090_8x_local"])
async def test_cli_uses_detected_capacity_and_canonical_target(monkeypatch, target_name, capsys):
    from hyperloom.inference_optimizer import cli
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    monkeypatch.delenv("HYPERLOOM_TARGET", raising=False)
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_BACKEND", raising=False)
    fp = registry.validate_nvidia_host(hardware(monkeypatch, count=1))
    monkeypatch.setattr(registry, "validate_nvidia_host", lambda *a, **kw: fp)
    monkeypatch.setattr(vllm_cuda_preflight, "validate_execution_stack", lambda value: value)
    monkeypatch.setattr(registry, "configure_target_environment", lambda *a, **kw: None)
    # Stop at the real CLI resource check, before setup or model loading.
    args = _build_parser().parse_args(["optimize", "--target", target_name, "--model", "/model", "--tp", "2"])
    before = dict(os.environ)
    try:
        with pytest.raises(SystemExit) as error:
            await cli._run_optimize(args)
        assert error.value.code == 2
        assert args.gpus_per_node == 1
        assert args.target == "nvidia_cuda"
        assert "the 1 GPUs available" in capsys.readouterr().err
    finally:
        os.environ.clear()
        os.environ.update(before)


def test_platform_publication_has_no_framework_policy(monkeypatch):
    for key in (
        "HYPERLOOM_TARGET",
        "HYPERLOOM_TARGET_RUNTIME",
        "HYPERLOOM_HARDWARE_FINGERPRINT",
        "CUDA_HOME",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "PATH",
    ):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("VLLM_PLUGINS", "custom-plugin")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_BACKEND", "custom-backend")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    registry.configure_target_environment(
        registry.get_target("nvidia_cuda"), fingerprint={"cuda_home": "", "devices": []}
    )
    assert os.environ["VLLM_PLUGINS"] == "custom-plugin"
    assert os.environ["HYPERLOOM_BENCHMARK_BACKEND"] == "custom-backend"
    assert os.environ["INFERENCE_OPTIMIZER_RAY_EXEC"] == "1"


def test_metric_query_keeps_hardware_metrics_when_optional_collection_is_unavailable(monkeypatch):
    def run(command, **kwargs):
        if "--query-metrics-collection" in command:
            return SimpleNamespace(returncode=1, stdout="", stderr="unsupported collection")
        return SimpleNamespace(returncode=0, stdout="gpu__time_duration.sum", stderr="")

    monkeypatch.setattr(cuda_nsight.subprocess, "run", run)
    result = cuda_nsight.query_roofline_metrics(["compute"], ["GPU-a"])
    assert result["selected"] == ["gpu__time_duration.sum"]
    assert len(result["query_errors"]) == 2
    assert "device__attribute_pci_bus_id" in result["unavailable"]
