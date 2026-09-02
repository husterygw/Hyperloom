# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CUDA runner lifecycle, normalization, leasing and fault-injection tests."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import _workload_envs as workload_envs
from hyperloom.orchestrator.actions.executors import benchmark_backend
from hyperloom.orchestrator.actions.executors import vllm_cuda_runner as runner


@pytest.fixture(autouse=True)
def _restore_environment():
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def _hardware() -> dict:
    return {
        "target_id": runner.TARGET_ID,
        "runtime": "cuda",
        "sha256": "f" * 64,
        "devices": [
            {
                "index": index,
                "uuid": f"GPU-{index:02d}",
                "name": "NVIDIA GeForce RTX 4090",
                "memory_mib": 24564,
                "compute_capability": "8.9",
                "pci_bus_id": f"0000:{index + 1:02x}:00.0",
                "numa_node": 0 if index < 4 else 1,
            }
            for index in range(8)
        ],
    }


def _cuda_env(monkeypatch, tmp_path: Path, *, visible: str = "0") -> Path:
    session_dir = tmp_path / "session"
    monkeypatch.setenv("HYPERLOOM_TARGET", runner.TARGET_ID)
    monkeypatch.setenv("HYPERLOOM_TARGET_RUNTIME", "cuda")
    monkeypatch.setenv("HYPERLOOM_HARDWARE_FINGERPRINT", json.dumps(_hardware()))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(session_dir))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-13.0")
    return session_dir


def _config(tmp_path: Path, *, tp: int = 1, pp: int = 1) -> Path:
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "model": "/models/qwen",
                    "timeout_seconds": 120,
                    "envs": {
                        "TP": tp,
                        "PP": pp,
                        "CONC": 2,
                        "ISL": 16,
                        "OSL": 8,
                        "MAX_MODEL_LEN": 64,
                        "NUM_PROMPTS": 4,
                        "NUM_WARMUPS": 1,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _plan(*, world_size: int = 8) -> runner.LaunchPlan:
    return runner.LaunchPlan(
        target_id=runner.TARGET_ID,
        model="/models/qwen",
        served_model_name="hyperloom-qwen",
        topology={"replicas": 1, "tp": 1, "pp": world_size, "world_size": world_size},
        port=18080,
        base_url="http://127.0.0.1:18080",
        server_argv=["python", "-m", "vllm.entrypoints.cli.main", "serve"],
        benchmark_argv=["python", "-m", "vllm.entrypoints.cli.main", "bench"],
        child_env={"CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(world_size))},
        artifacts={"raw_result": "/tmp/raw.json"},
        fingerprints={"hardware": "hardware", "workload": "workload", "config": "config"},
    )


def _raw_result() -> dict:
    return {
        "completed": 4,
        "failed": 0,
        "duration": 2.0,
        "total_input_tokens": 64,
        "total_output_tokens": 32,
        "request_throughput": 2.0,
        "output_throughput": 80.0,
        "median_ttft_ms": 10.0,
        "p90_ttft_ms": 12.0,
        "p99_ttft_ms": 14.0,
        "median_tpot_ms": 3.0,
        "p90_tpot_ms": 4.0,
        "p99_tpot_ms": 5.0,
        "median_itl_ms": 2.0,
        "p90_itl_ms": 2.5,
        "p99_itl_ms": 3.0,
        "median_e2el_ms": 40.0,
        "p90_e2el_ms": 45.0,
        "p99_e2el_ms": 50.0,
    }


def test_normalize_records_total_per_gpu_and_all_percentiles():
    report = runner.normalize_vllm_result(
        _raw_result(),
        plan=_plan(),
        quality_gate={"passed": True},
        cleanup_status="released",
    )
    assert report["schema_version"] == runner.SCHEMA_VERSION
    assert report["status"] == "succeeded"
    assert report["metrics"]["output_tokens_per_second_total"] == 80.0
    assert report["metrics"]["output_tokens_per_second_per_gpu"] == 10.0
    assert report["metrics"]["latency"]["ttft"] == {"p50_ms": 10.0, "p90_ms": 12.0, "p99_ms": 14.0}
    assert report["metrics"]["latency"]["e2el"]["p99_ms"] == 50.0


@pytest.mark.parametrize(
    ("raw_override", "quality_gate", "cleanup_status"),
    [
        ({"failed": 1}, {"passed": True}, "released"),
        ({}, {"passed": False}, "released"),
        ({}, {"passed": True}, "lease_release_failed"),
    ],
)
def test_normalize_fails_closed_on_request_quality_or_cleanup_failure(raw_override, quality_gate, cleanup_status):
    raw = {**_raw_result(), **raw_override}
    report = runner.normalize_vllm_result(
        raw,
        plan=_plan(),
        quality_gate=quality_gate,
        cleanup_status=cleanup_status,
    )
    assert report["status"] == "failed"


def test_visible_device_selection_fails_closed_on_short_duplicate_or_symbolic(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert runner._visible_indices({"CUDA_VISIBLE_DEVICES": "4,5,6,7"}, 4) == (4, 5, 6, 7)
    with pytest.raises(ValueError, match=r"TP\*PP"):
        runner._visible_indices({"CUDA_VISIBLE_DEVICES": "0,1"}, 4)
    with pytest.raises(ValueError, match="unique"):
        runner._visible_indices({"CUDA_VISIBLE_DEVICES": "0,0"}, 2)
    with pytest.raises(ValueError, match="integer"):
        runner._visible_indices({"CUDA_VISIBLE_DEVICES": "GPU-a"}, 1)


def test_extra_args_cannot_override_runner_owned_topology():
    assert runner._tokenize_extra_args({"EXTRA_VLLM_ARGS": "--gpu-memory-utilization 0.85"}) == [
        "--gpu-memory-utilization",
        "0.85",
    ]
    with pytest.raises(ValueError, match="pipeline-parallel-size"):
        runner._tokenize_extra_args({"EXTRA_VLLM_ARGS": "--pipeline-parallel-size=2"})
    with pytest.raises(ValueError, match="unsupported flag"):
        runner._tokenize_extra_args({"EXTRA_VLLM_ARGS": "--future-unpinned-knob 1"})


def test_gpu_lease_persists_uuid_and_numa_and_is_idempotent(tmp_path, monkeypatch):
    session_dir = _cuda_env(monkeypatch, tmp_path, visible="0,1")
    first = runner._acquire_gpu_lease(gpu_ids=(0, 1), stable_key="stable", ttl_sec=120)
    second = runner._acquire_gpu_lease(gpu_ids=(0, 1), stable_key="stable", ttl_sec=120)
    assert second.holder_id == first.holder_id
    assert second.gpu_uuids == ("GPU-00", "GPU-01")
    assert second.numa_nodes == (0, 0)
    with sqlite3.connect(session_dir / "storage" / "coordinator.db") as conn:
        rows = conn.execute("SELECT gpu_id,gpu_uuid,numa_node FROM gpu_leases ORDER BY gpu_id").fetchall()
    assert rows == [(0, "GPU-00", 0), (1, "GPU-01", 0)]
    runner._release_gpu_lease(first)
    assert runner._lease_is_released(first)


def test_gpu_lease_conflict_is_rejected(tmp_path, monkeypatch):
    _cuda_env(monkeypatch, tmp_path, visible="0")
    first = runner._acquire_gpu_lease(gpu_ids=(0,), stable_key="first", ttl_sec=120)
    with pytest.raises(RuntimeError, match="lease conflict"):
        runner._acquire_gpu_lease(gpu_ids=(0,), stable_key="second", ttl_sec=120)
    runner._release_gpu_lease(first)


def test_cuda_materialization_uses_tp_times_pp_and_removes_amd_surface(tmp_path, monkeypatch):
    _cuda_env(monkeypatch, tmp_path, visible="0,1,2,3,4,5,6,7")
    monkeypatch.setenv("TP", "1")
    monkeypatch.setenv("PP", "8")
    source = tmp_path / "source.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "model": "/models/qwen",
                    "runner_type": "mi300x",
                    "benchmark_script": "vllm_mi300x.sh",
                    "inferencex_path": "/opt/InferenceX",
                    "envs": {"ROCR_VISIBLE_DEVICES": "0,1", "HIP_VISIBLE_DEVICES": "0,1"},
                }
            }
        ),
        encoding="utf-8",
    )
    rendered_path = workload_envs.materialize_config_with_envs(
        source,
        tmp_path / "rendered",
        extra_envs={"ROCR_VISIBLE_DEVICES": "malicious", "HIP_VISIBLE_DEVICES": "malicious"},
    )
    bench = yaml.safe_load(rendered_path.read_text(encoding="utf-8"))["benchmark"]
    assert bench["envs"]["TP"] == 1
    assert bench["envs"]["PP"] == 8
    assert bench["envs"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "ROCR_VISIBLE_DEVICES" not in bench["envs"]
    assert "HIP_VISIBLE_DEVICES" not in bench["envs"]
    assert "runner_type" not in bench
    assert "benchmark_script" not in bench
    assert "inferencex_path" not in bench


def test_backend_selects_exact_interpreter_and_native_module(monkeypatch, tmp_path):
    monkeypatch.setenv(benchmark_backend.BENCHMARK_BACKEND_ENV, "vllm_cuda")
    backend = benchmark_backend.resolve_backend()
    assert isinstance(backend, benchmark_backend.VllmCudaBackend)
    command = backend.build_command(
        python_exe="/env/bin/python",
        config_path=tmp_path / "config.yaml",
        output_dir=tmp_path / "out",
    )
    assert command[:3] == ["/env/bin/python", "-m", "hyperloom.orchestrator.actions.executors.vllm_cuda_runner"]
    assert backend.lifecycle_eligibility({"framework": "vllm", "envs": {"PORT": 18080}})["eligible"] is True


class _FinishedServer:
    pid = 999999

    def __init__(self, argv, *, stdout=None, **kwargs):
        self.argv = argv
        if stdout is not None:
            stdout.write("server fixture started\n")
            stdout.flush()

    def poll(self):
        return 0


def _fake_client_success(argv, **kwargs):
    result_dir = Path(argv[argv.index("--result-dir") + 1])
    filename = argv[argv.index("--result-filename") + 1]
    (result_dir / filename).write_text(json.dumps(_raw_result()), encoding="utf-8")
    return SimpleNamespace(returncode=0)


def test_full_runner_lifecycle_writes_launch_and_compatibility_artifacts(tmp_path, monkeypatch):
    _cuda_env(monkeypatch, tmp_path, visible="0,1,2,3,4,5,6,7")
    monkeypatch.setattr(runner, "_pick_port", lambda: 18080)
    monkeypatch.setattr(runner, "_wait_ready", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "_verify_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_quality_smoke", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(runner.subprocess, "Popen", _FinishedServer)
    monkeypatch.setattr(runner.subprocess, "run", _fake_client_success)
    output_dir = tmp_path / "output"
    assert runner.run_benchmark(_config(tmp_path, tp=1, pp=8), output_dir) == 0
    workspace = next(output_dir.glob("benchmark_vllm_*"))
    launch = json.loads((workspace / "launch_plan.json").read_text(encoding="utf-8"))
    unified = json.loads((workspace / "vllm_cuda_benchmark.json").read_text(encoding="utf-8"))
    compatibility = json.loads((workspace / "benchmark_report.json").read_text(encoding="utf-8"))
    assert launch["topology"]["world_size"] == 8
    assert launch["topology"]["gpu_uuids"] == [f"GPU-{i:02d}" for i in range(8)]
    assert launch["server_argv"][launch["server_argv"].index("--pipeline-parallel-size") + 1] == "8"
    assert launch["child_env"]["VLLM_PLUGINS"] == ""
    assert unified["status"] == "succeeded"
    assert unified["cleanup_status"] == "released"
    assert compatibility["output_throughput_total"] == 80.0
    assert compatibility["output_throughput_per_gpu"] == 10.0


@pytest.mark.parametrize("failure", ["timeout", "bad_json", "server_crash", "oom"])
def test_runner_faults_fail_closed_and_release_lease(tmp_path, monkeypatch, failure):
    session_dir = _cuda_env(monkeypatch, tmp_path, visible="0")
    monkeypatch.setattr(runner, "_pick_port", lambda: 18081)
    monkeypatch.setattr(runner, "_verify_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_quality_smoke", lambda *args, **kwargs: {"passed": True})

    class FaultServer(_FinishedServer):
        def __init__(self, argv, *, stdout=None, **kwargs):
            super().__init__(argv, stdout=stdout, **kwargs)
            if failure == "oom" and stdout is not None:
                stdout.write("CUDA out of memory\n")
                stdout.flush()

    monkeypatch.setattr(runner.subprocess, "Popen", FaultServer)
    monkeypatch.setattr(runner, "_wait_ready", lambda *args, **kwargs: failure not in {"server_crash", "oom"})

    def fake_client(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout=1)
        result_dir = Path(argv[argv.index("--result-dir") + 1])
        filename = argv[argv.index("--result-filename") + 1]
        value = "{bad json" if failure == "bad_json" else json.dumps(_raw_result())
        (result_dir / filename).write_text(value, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_client)
    output_dir = tmp_path / f"output-{failure}"
    assert runner.run_benchmark(_config(tmp_path), output_dir) == 1
    workspace = next(output_dir.glob("benchmark_vllm_*"))
    unified = json.loads((workspace / "vllm_cuda_benchmark.json").read_text(encoding="utf-8"))
    assert unified["status"] == "failed"
    assert unified["failure_reason"]
    assert unified["cleanup_status"] == "released"
    with sqlite3.connect(session_dir / "storage" / "coordinator.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM gpu_leases").fetchone()[0] == 0


def test_runner_requires_validated_target_and_hardware(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_TARGET", raising=False)
    monkeypatch.delenv("HYPERLOOM_TARGET_RUNTIME", raising=False)
    with pytest.raises(ValueError, match="requires HYPERLOOM_TARGET"):
        runner.run_benchmark(_config(tmp_path), tmp_path / "output")
