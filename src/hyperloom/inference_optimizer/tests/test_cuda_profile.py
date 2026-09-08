# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CUDA profile capability, trace integrity and cleanup regressions."""

import gzip
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.inference_optimizer.target_registry import (
    NVIDIA_LOCAL_TARGET,
    TargetValidationError,
    get_target,
    validate_target_arguments,
    validate_profile_runtime,
)
from hyperloom.inference_optimizer.protocol.action_surfaces import target_capability_enabled
from hyperloom.orchestrator.actions.executors.cuda_profile import CudaProfileExecutor, validate_cuda_traces
from hyperloom.orchestrator.actions.executors import vllm_cuda_runner as runner
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator
from hyperloom.orchestrator.policy.gate import PolicyGate, PolicyDenied
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.tests.test_vllm_cuda_runner import (
    _cuda_env,
    _config,
    _FinishedServer,
    _fake_client_success,
    isolate_cuda_host_lock,  # noqa: F401 - imported autouse fixture
    _restore_environment,  # noqa: F401 - imported autouse fixture
)


def _trace(root, rank=0, *, kernel=True):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"dp0_pp0_tp0_dcp0_ep0_rank{rank}.123.pt.trace.json.gz"
    with gzip.open(path, "wt") as stream:
        json.dump({"traceEvents": [{"cat": "kernel" if kernel else "cpu_op", "ph": "X", "dur": 5}]}, stream)
    stamp = time.time()
    os.utime(path, (stamp, stamp))  # Explicit time avoids coarse filesystem-clock flakes.
    return path


@pytest.mark.parametrize("level", ["config", "profile"])
def test_nv_profile_is_opt_in_and_cannot_enable_roofline(level):
    args = SimpleNamespace(optimization_level=level)
    validate_target_arguments(args, get_target(NVIDIA_LOCAL_TARGET))
    assert args.target_capabilities["profile"] == (level == "profile")
    assert not args.target_capabilities["roofline"]
    assert not args.target_capabilities["trace_analysis"]
    assert not args.target_capabilities["source_patch"]
    assert not args.target_capabilities["kernel_patch"]
    assert not args.enable_roofline
    gate = PolicyGate.__new__(PolicyGate)
    gate.shared_state = SharedState(target_id=NVIDIA_LOCAL_TARGET, target_capabilities=args.target_capabilities)
    for action in ("roofline", "trace_analyze", "record_trace_analyze", "integrate_patch", "gemm_tuning"):
        with pytest.raises(PolicyDenied):
            gate._validate_target_capability(action)
    if level == "profile":
        gate._validate_target_capability("profile")


@pytest.mark.parametrize(
    "kwargs", [{"optimization_level": "source"}, {"optimization_level": "kernel"}, {"profile_backend": "nsys"}]
)
def test_unimplemented_features_fail_before_gpu_launch(kwargs):
    with pytest.raises(TargetValidationError, match="not implemented"):
        validate_target_arguments(SimpleNamespace(**kwargs), get_target(NVIDIA_LOCAL_TARGET))


def test_legacy_capabilities_keep_semantics():
    for profile in (False, True):
        assert target_capability_enabled({"profile": profile}, "roofline") == profile
        assert target_capability_enabled({"profile": profile}, "trace_analysis") == profile
    assert not target_capability_enabled({"profile": True, "roofline": False}, "roofline")


def test_profile_preflight_requires_both_cli_flags():
    with pytest.raises(TargetValidationError):
        validate_profile_runtime({"vllm_cli": {"server_flags": ["--profiler-config"]}})
    validate_profile_runtime({"vllm_cli": {"server_flags": ["--profiler-config"], "bench_flags": ["--profile"]}})


def test_trace_requires_every_rank_and_real_fresh_cuda_events(tmp_path):
    first = _trace(tmp_path, 0)
    assert not validate_cuda_traces(tmp_path, world_size=2, started_at=0)["passed"]
    _trace(tmp_path, 1, kernel=False)
    assert not validate_cuda_traces(tmp_path, world_size=2, started_at=0)["passed"]
    second = _trace(tmp_path, 1)
    assert validate_cuda_traces(tmp_path, world_size=2, started_at=0)["passed"]
    os.utime(first, (1, 1))
    assert not validate_cuda_traces(tmp_path, world_size=2, started_at=2)["passed"]
    second.write_bytes(b"invalid gzip")
    assert not validate_cuda_traces(tmp_path, world_size=2, started_at=0)["passed"]


def test_profile_cannot_promote_benchmark_score():
    result = {
        "status": "succeeded",
        "measurement_kind": "profile",
        "output_throughput": 99999,
        "trace_health": {"passed": True},
    }
    for kind in ("baseline", "sweep", "explore"):
        assert not WritebackCollaborator._is_promotable_result(None, kind, result)
    assert WritebackCollaborator._is_promotable_result(None, "profile", result)
    result["trace_health"]["passed"] = False
    assert not WritebackCollaborator._is_promotable_result(None, "profile", result)


def test_materialization_preserves_launch_args_and_bounds_capture(tmp_path):
    path = _config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["benchmark"]["envs"]["NUM_PROMPTS"] = 100
    cfg["benchmark"]["envs"]["EXTRA_VLLM_ARGS"] = "--compilation-config '{\"mode\":3}'"
    path.write_text(yaml.safe_dump(cfg))
    CudaProfileExecutor()._after_materialize_config(path, tmp_path)
    updated = yaml.safe_load(path.read_text())
    assert updated["benchmark"]["envs"]["EXTRA_VLLM_ARGS"] == cfg["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]
    assert updated["benchmark"]["envs"]["NUM_PROMPTS"] == 16
    assert updated["benchmark"]["profiler"]["torch_profiler"]["enabled"]
    assert not updated["benchmark"]["server_lifecycle"]["enabled"]


@pytest.mark.parametrize("failure", [None, "missing_rank", "start_failed", "stop_in_progress", "client_failed"])
def test_profile_runner_validates_artifact_and_releases_gpu(tmp_path, monkeypatch, failure):
    _cuda_env(monkeypatch, tmp_path, visible="0,1")
    monkeypatch.setattr(runner, "_wait_ready", lambda *a, **kw: True)
    monkeypatch.setattr(runner, "_verify_model", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_quality_smoke", lambda *a, **kw: {"passed": True})

    class AliveServer(_FinishedServer):
        def poll(self):
            return None

    monkeypatch.setattr(runner.subprocess, "Popen", AliveServer)
    terminated = []
    monkeypatch.setattr(runner, "_terminate_group", lambda proc: terminated.append(proc))
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: pytest.fail("stop_profile retried"))

    def client(argv, **kwargs):
        assert "--profile" in argv
        root = Path(argv[argv.index("--result-dir") + 1])
        _trace(root / "torch_trace", 0)
        if failure != "missing_rank":
            _trace(root / "torch_trace", 1)
        # vLLM bench may omit its success messages despite empty HTTP 200.
        with (root / "server.log").open("a") as log:
            if failure != "start_failed":
                log.write('"POST /start_profile HTTP/1.1" 200 OK\n')
            log.write("Stopping profiler...\n")
            if failure != "stop_in_progress":
                log.write('"POST /stop_profile HTTP/1.1" 200 OK\n')
        result = _fake_client_success(argv, **kwargs)
        if failure == "client_failed":
            result.returncode = 1
        return result

    monkeypatch.setattr(runner.subprocess, "run", client)
    path = _config(tmp_path, pp=2)
    cfg = yaml.safe_load(path.read_text())
    cfg["benchmark"]["profiler"] = {"torch_profiler": {"enabled": True}}
    cfg["benchmark"]["server_lifecycle"] = {"enabled": True, "cleanup": False}
    path.write_text(yaml.safe_dump(cfg))
    assert runner.run_benchmark(path, tmp_path / "out") == (0 if failure is None else 1)
    artifact = next((tmp_path / "out").rglob("vllm_cuda_profile.json"))
    data = json.loads(artifact.read_text())
    assert data["cleanup_status"] == "released"
    assert terminated
    assert data["status"] == ("succeeded" if failure is None else "failed")
    plan = json.loads((artifact.parent / "launch_plan.json").read_text())
    assert "--profiler-config" in plan["server_argv"]


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("optimization_level", "profile"), ("profile_backend", "nsys")])
async def test_resume_cannot_change_feature_tier_before_host_probe(tmp_path, monkeypatch, field, value):
    from hyperloom.inference_optimizer.cli import _run_optimize
    from hyperloom.inference_optimizer.cli.parser import _build_parser
    from hyperloom.inference_optimizer import target_registry

    state = SharedState(
        target_id=NVIDIA_LOCAL_TARGET, target_capabilities=get_target(NVIDIA_LOCAL_TARGET).capabilities.to_dict()
    )
    state.save(tmp_path)
    # Emulate a config-only session saved before feature tiers existed.
    payload = json.loads((tmp_path / "state.json").read_text())
    payload.pop("optimization_level")
    payload.pop("profile_backend")
    (tmp_path / "state.json").write_text(json.dumps(payload))
    monkeypatch.delenv("HYPERLOOM_TARGET", raising=False)
    monkeypatch.setattr(
        target_registry, "validate_nvidia_host", lambda *a: pytest.fail("hardware probed before resume conflict")
    )
    args = _build_parser().parse_args(
        ["optimize", "--resume-from", str(tmp_path), "--" + field.replace("_", "-"), value]
    )
    with pytest.raises(SystemExit) as exc:
        await _run_optimize(args)
    assert exc.value.code == 2
    loaded = SharedState.load_or_init(tmp_path)
    assert loaded.optimization_level == "config"
    assert not loaded.target_capabilities["profile"]


def test_stop_ack_does_not_require_bench_client_success_text(tmp_path):
    log = tmp_path / "server.log"
    log.write_text('"POST /start_profile HTTP/1.1" 200 OK\nStopping profiler...\n')
    assert runner._profile_endpoint_status(log) == (True, False, True)
    with log.open("a") as stream:
        stream.write('"POST /stop_profile HTTP/1.1" 200 OK\n')
    assert runner._profile_endpoint_status(log) == (True, True, True)


@pytest.mark.asyncio
async def test_executor_profiles_current_config_without_sanitizing_graph_flags(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors.baseline import BaselineExecutor

    captured = {}

    async def baseline(self, ctx):
        captured.update(ctx.task.params)
        return {"status": "succeeded", "output_throughput": 10000, "workspace": str(tmp_path)}

    monkeypatch.setattr(BaselineExecutor, "__call__", baseline)
    (tmp_path / "vllm_cuda_profile.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "trace_files": ["rank0.trace"],
                "trace_health": {"passed": True},
                "fingerprints": {"config": "config"},
                "errors": [],
            }
        )
    )
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            params={
                "base_extra_args": "--compilation-config '{\"mode\":3}' --gpu-memory-utilization 0.9",
                "base_extra_envs": {"MAX_MODEL_LEN": "2048"},
                "base_args_mode": "replace",
                "base_remove_args": ["--enforce-eager"],
            }
        )
    )
    result = await CudaProfileExecutor()(ctx)
    assert "--compilation-config" in captured["extra_server_args"]
    assert captured["args_mode"] == "replace"
    assert captured["remove_args"] == ["--enforce-eager"]
    assert captured["extra_envs"]["MAX_MODEL_LEN"] == "2048"
    assert "output_throughput" not in result
    assert result["diagnostic_throughput"] == 10000
    assert result["measurement_kind"] == "profile"


def test_profile_report_cannot_be_salvaged_as_a_benchmark(tmp_path):
    from hyperloom.orchestrator.actions.executors.benchmark_result import (
        extract_benchmark_measurement,
        is_valid_measurement,
    )

    # A positive raw result alongside the profile must not become a fallback score.
    (tmp_path / "benchmark_results.json").write_text(json.dumps({"output_throughput": 10000, "completed": 8}))
    report = {
        "measurement_kind": "profile",
        "success": True,
        "throughput": {"output_throughput": 10000},
        "completed_requests": 8,
    }
    measurement = extract_benchmark_measurement(report, workspace=tmp_path)
    assert not measurement["valid_measurement"]
    assert measurement["measurement_kind"] == "profile"
    assert not is_valid_measurement(
        {"measurement_kind": "profile", "output_throughput": 10000, "completed_requests": 8}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("extra,expected", [([], 0.5), (["--max-hours", "2.0"], 2.0)])
async def test_nv_resume_restores_budget_unless_explicit(tmp_path, monkeypatch, extra, expected):
    from hyperloom.inference_optimizer.cli import _run_optimize
    from hyperloom.inference_optimizer.cli.parser import _build_parser
    from hyperloom.inference_optimizer import target_registry

    SharedState(target_id=NVIDIA_LOCAL_TARGET, max_minutes=30).save(tmp_path)
    monkeypatch.delenv("HYPERLOOM_TARGET", raising=False)
    def end_preflight(*args):
        raise TargetValidationError("test ends at hardware preflight")
    monkeypatch.setattr(target_registry, "validate_nvidia_host", end_preflight)
    args = _build_parser().parse_args(["optimize", "--resume-from", str(tmp_path), *extra])
    with pytest.raises(SystemExit):
        await _run_optimize(args)
    assert args.max_hours == expected
