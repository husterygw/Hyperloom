# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution-target contracts for the local 8x RTX 4090 CUDA MVP."""

from __future__ import annotations

import argparse
import json
import os

import pytest

from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.inference_optimizer.cli import preflight as cli_preflight
from hyperloom.inference_optimizer.target_registry import (
    DEFAULT_TARGET,
    HARDWARE_FINGERPRINT_ENV,
    NVIDIA_LOCAL_TARGET,
    NvidiaDevice,
    TargetValidationError,
    build_hardware_fingerprint,
    configure_target_environment,
    get_target,
    resolve_target,
    target_names,
    validate_nvidia_host,
    validate_target_arguments,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.policy.gate import PolicyDenied, PolicyGate
from hyperloom.orchestrator.prompts.prompt_builder import default_enabled_actions
from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.orchestrator.state.shared_state import LATEST_STATE_SCHEMA_VERSION, SharedState


@pytest.fixture(autouse=True)
def _restore_environment():
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def _device(index: int) -> NvidiaDevice:
    return NvidiaDevice(
        index=index,
        uuid=f"GPU-{index:02d}",
        name="NVIDIA GeForce RTX 4090",
        memory_mib=24564,
        compute_capability="8.9",
        pci_bus_id=f"0000:{index + 1:02x}:00.0",
        numa_node=0 if index < 4 else 1,
    )


def _args(**overrides):
    values = {
        "nodes": 1,
        "framework": None,
        "gpu_type": None,
        "quantize": None,
        "quantize_scheme": "none",
        "no_kernel": False,
        "enable_roofline": True,
        "no_framework_agent": True,
        "no_framework_local_explore": False,
        "enablement": "auto",
        "no_warm_replay": False,
        "no_eval": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_registry_keeps_amd_default_and_explicit_nvidia_target(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_TARGET", raising=False)
    assert target_names() == (DEFAULT_TARGET, NVIDIA_LOCAL_TARGET)
    assert resolve_target().target_id == DEFAULT_TARGET
    target = get_target(NVIDIA_LOCAL_TARGET)
    assert target.runtime == "cuda"
    assert target.benchmark_backend == "vllm_cuda"
    assert target.expected_gpu_count == 8
    assert target.capabilities.config_explore is True
    assert target.capabilities.profile is False
    assert target.capabilities.source_patch is False
    assert target.capabilities.kernel_patch is False


def test_cli_parses_target_and_pipeline_parallelism():
    args = _build_parser().parse_args(
        ["optimize", "--model", "/models/qwen", "--target", NVIDIA_LOCAL_TARGET, "--tp", "1", "--pp", "8"]
    )
    assert args.target == NVIDIA_LOCAL_TARGET
    assert args.tp == 1
    assert args.pp == 8
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["optimize", "--model", "/models/qwen", "--pp", "0"])


def test_cuda_target_arguments_are_forced_to_config_only():
    args = _args()
    validate_target_arguments(args, get_target(NVIDIA_LOCAL_TARGET))
    assert args.framework == "vllm"
    assert args.no_kernel is True
    assert args.enable_roofline is False
    assert args.no_framework_agent is False
    assert args.no_framework_local_explore is True
    assert args.enablement == "off"
    assert args.no_warm_replay is True
    assert args.no_eval is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"nodes": 2}, "single-node"),
        ({"framework": "sglang"}, "requires --framework vllm"),
        ({"gpu_type": "mi300x"}, "AMD-only"),
        ({"quantize": "fp8"}, "does not support quantization"),
    ],
)
def test_cuda_target_rejects_unsupported_cli_combinations(overrides, message):
    with pytest.raises(TargetValidationError, match=message):
        validate_target_arguments(_args(**overrides), get_target(NVIDIA_LOCAL_TARGET))


def test_configure_cuda_environment_is_vendor_clean(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/usr/local/cuda-13.0/bin")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    fingerprint = {"target_id": NVIDIA_LOCAL_TARGET, "cuda_home": "/cuda", "sha256": "abc"}
    configure_target_environment(get_target(NVIDIA_LOCAL_TARGET), fingerprint=fingerprint)
    assert os.environ["HYPERLOOM_TARGET_RUNTIME"] == "cuda"
    assert os.environ["HYPERLOOM_BENCHMARK_BACKEND"] == "vllm_cuda"
    assert os.environ["INFERENCE_OPTIMIZER_RAY_EXEC"] == "0"
    assert os.environ["CUDA_HOME"] == "/cuda"
    assert os.environ["PATH"].split(":")[0] == "/cuda/bin"
    assert os.environ["VLLM_PLUGINS"] == ""
    assert "ROCR_VISIBLE_DEVICES" not in os.environ
    assert "HIP_VISIBLE_DEVICES" not in os.environ
    assert json.loads(os.environ[HARDWARE_FINGERPRINT_ENV])["sha256"] == "abc"


def test_cli_auth_cli_presence_check_does_not_require_claude_or_node(monkeypatch, capsys):
    monkeypatch.setenv("HYPERLOOM_CODEX_CLI_AUTH", "1")
    monkeypatch.setattr(
        cli_preflight.shutil,
        "which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )

    cli_preflight._check_node_claude_cli()

    assert capsys.readouterr().out == ""


def test_hardware_fingerprint_is_deterministic(monkeypatch):
    from hyperloom.inference_optimizer import target_registry as registry

    monkeypatch.setattr(registry, "_topology_diagnostic", lambda: "GPU0 GPU1\nGPU0 X PHB")
    devices = tuple(_device(i) for i in range(8))
    first = build_hardware_fingerprint(get_target(NVIDIA_LOCAL_TARGET), devices)
    second = build_hardware_fingerprint(get_target(NVIDIA_LOCAL_TARGET), devices)
    assert first == second
    assert len(first["sha256"]) == 64
    assert first["devices"][4]["numa_node"] == 1


def test_nvidia_host_validation_records_pinned_stack(monkeypatch):
    import torch
    from hyperloom.inference_optimizer import target_registry as registry

    monkeypatch.setattr(registry, "discover_nvidia_devices", lambda: tuple(_device(i) for i in range(8)))
    monkeypatch.setattr(registry, "_nvcc_release", lambda path: "Cuda compilation tools, release 13.0")
    monkeypatch.setattr(registry, "_nvidia_driver_version", lambda: "590.48.01")
    monkeypatch.setattr(registry.importlib.metadata, "version", lambda name: "0.27.0rc1" if name == "vllm" else "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    fingerprint = validate_nvidia_host(get_target(NVIDIA_LOCAL_TARGET))
    assert fingerprint["target_id"] == NVIDIA_LOCAL_TARGET
    assert fingerprint["cuda_home"] == "/usr/local/cuda-13.0"
    assert fingerprint["vllm_version"] == "0.27.0rc1"
    assert fingerprint["driver_version"] == "590.48.01"
    assert fingerprint["torch_cuda_version"]
    assert fingerprint["nccl_version"]


def test_v6_state_migrates_target_and_pp_defaults():
    state = SharedState.from_dict({"schema_version": 6, "session_id": "old", "tp": 4})
    assert state.schema_version == LATEST_STATE_SCHEMA_VERSION == 7
    assert state.target_id == DEFAULT_TARGET
    assert state.target_capabilities == {}
    assert state.hardware_fingerprint == {}
    assert state.pp == 1


def test_prompt_and_policy_filter_disabled_target_actions():
    capabilities = get_target(NVIDIA_LOCAL_TARGET).capabilities.to_dict()
    actions = default_enabled_actions(no_kernel=False, target_capabilities=capabilities)
    assert {"baseline", "explore", "sweep", "report"} <= set(actions)
    assert {"roofline", "integrate_patch", "kernel_opt", "integrate", "gemm_tuning"}.isdisjoint(actions)

    state = SharedState(
        target_id=NVIDIA_LOCAL_TARGET,
        target_capabilities=capabilities,
        phase="FRAMEWORK_AGENT",
    )
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    with pytest.raises(PolicyDenied, match="source_patch") as proposed:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.PROPOSE_ACTION,
                payload={"action_name": "integrate_patch", "predicted_gain_pct": 1.0},
            ),
        )
    assert proposed.value.rule == "target_capability"
    with pytest.raises(PolicyDenied, match="profile") as dispatched:
        gate.validate_dispatched_task("profile", {})
    assert dispatched.value.rule == "target_capability"


def test_target_capability_rejects_kernel_request_alias():
    state = SharedState(
        target_id=NVIDIA_LOCAL_TARGET,
        target_capabilities=get_target(NVIDIA_LOCAL_TARGET).capabilities.to_dict(),
        phase="KERNEL_AGENT",
        precision="bf16",
        framework="vllm",
    )
    gate = PolicyGate(role_registry=default_role_registry(), shared_state=state)
    with pytest.raises(PolicyDenied, match="kernel_patch") as denied:
        gate.validate_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={"target_agent": "kernel_agent", "kind": "run_gemm_tuning", "params": {}},
            ),
        )
    assert denied.value.rule == "target_capability"


def test_policy_serving_gpu_count_includes_pipeline_parallelism():
    from hyperloom.orchestrator.policy.gate import _serving_tp_for_policy

    assert _serving_tp_for_policy(SharedState(tp=1, pp=8)) == 8
    assert _serving_tp_for_policy(SharedState(tp=2, pp=4)) == 8


def test_config_only_nvidia_target_skips_kernel_agent_env(monkeypatch, tmp_path):
    from hyperloom.inference_optimizer.cli import preflight

    monkeypatch.setenv("HYPERLOOM_TARGET", NVIDIA_LOCAL_TARGET)
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.delenv("HYPERLOOM_KERNEL_AGENT_ROOT", raising=False)
    monkeypatch.delenv("KERNEL_AGENT_ENV", raising=False)

    result = preflight._load_kernel_agent_env_fallback()

    assert result["status"] == "skipped"
    assert result["skip_reason"] == "target_capability_disabled"
    assert result["detail"]["target_id"] == NVIDIA_LOCAL_TARGET


def test_cuda_serving_framework_gate_accepts_cuda_vllm(monkeypatch):
    from hyperloom.inference_optimizer.cli import preflight

    monkeypatch.setenv("HYPERLOOM_TARGET_RUNTIME", "cuda")
    monkeypatch.setattr(preflight, "_framework_probe_interpreters", lambda *_args: ["/cuda/python"])
    monkeypatch.setattr(preflight, "_framework_importable", lambda *_args: preflight._Probe(True))

    result = preflight._check_serving_framework(argparse.Namespace(framework="vllm"), "/cuda/python")

    assert result["status"] == "applied"
    assert result["detail"]["runtime"] == "cuda"
    assert os.environ["HYPERLOOM_RESOLVED_FRAMEWORK"] == "vllm"
