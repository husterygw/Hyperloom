# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CUDA runner lifecycle, normalization, leasing and fault-injection tests."""

from __future__ import annotations

import sys
import types
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
def isolate_cuda_host_lock(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import cuda_host_lock

    monkeypatch.setattr(cuda_host_lock, "LOCK_PATH", tmp_path / "cuda-host.lock")


def test_gpu_memory_sampler_records_per_gpu_summary(monkeypatch):
    class _Info:
        used = 2 * 1024 * 1024

    fake = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetMemoryInfo=lambda _handle: _Info(),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sampler = runner._GpuMemorySampler((0, 1), interval_sec=0.05)
    sampler.start()
    summary = sampler.stop()
    assert summary["status"] == "collected"
    assert summary["per_gpu"]["0"]["max_used_mib"] == 2.0
    assert summary["per_gpu"]["1"]["samples"] >= 1


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
                "cuda_index": index,
                "logical_index": index,
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
    hardware = _hardware()
    hardware["cuda_devices"] = [{"uuid": r["uuid"], "cuda_index": r["cuda_index"]} for r in hardware["devices"]]
    indices = [int(i) for i in visible.split(",") if i]
    hardware["devices"] = [hardware["devices"][i] for i in indices]
    monkeypatch.setenv("HYPERLOOM_HARDWARE_FINGERPRINT", json.dumps(hardware))
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


def test_visible_device_selection_checks_pool_and_supports_uuid(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HYPERLOOM_HARDWARE_FINGERPRINT", json.dumps(_hardware()))
    assert runner._visible_indices({"CUDA_VISIBLE_DEVICES": "4,5,6,7"}, 4) == (4, 5, 6, 7)
    with pytest.raises(ValueError, match=r"TP\*PP"):
        runner._visible_indices({"CUDA_VISIBLE_DEVICES": "0,1"}, 4)
    with pytest.raises(ValueError, match="duplicate"):
        runner._visible_indices({"CUDA_VISIBLE_DEVICES": "0,0"}, 2)
    assert runner._visible_indices({"CUDA_VISIBLE_DEVICES": "GPU-04"}, 1) == (4,)
    with pytest.raises(ValueError, match="invalid or ambiguous"):
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


def test_cuda_runner_normalizes_sigterm_to_its_cleanup_path():
    with pytest.raises(KeyboardInterrupt):
        runner._cleanup_signal_handler(15, None)


def test_qwen3_p3_quality_suite_persists_the_semantic_matrix(tmp_path, monkeypatch):
    """P3 stores all four prompt classes, not just a pass/fail bit."""
    model = tmp_path / "qwen3"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            content = str(messages[0]["content"])
            return f"thinking={enable_thinking}\n{content}"

        def __call__(self, prompt, *, add_special_tokens):
            return {"input_ids": list(range(160 if "背景资料" in prompt or "Background" in prompt else 16))}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, *, local_files_only):
            assert path == str(model)
            assert local_files_only is True
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    def _reply(*args, **kwargs):
        prompt = kwargs["payload"]["prompt"]
        english = "Answer in one English" in prompt or "Background" in prompt
        if "thinking=True" in prompt:
            if english:
                text = (
                    "<think>reasoning</think> Photosynthesis converts water and carbon dioxide into sugar and oxygen."
                )
            else:
                text = "<think>推理</think> 光合作用把二氧化碳和水转化为有机物并释放氧气。"
        elif english:
            text = "Photosynthesis uses water and carbon dioxide to make sugar and oxygen."
        else:
            text = "光合作用把二氧化碳和水转化为有机物并释放氧气。"
        return {"choices": [{"text": text}]}

    monkeypatch.setattr(runner, "_json_request", _reply)
    artifact = tmp_path / "quality_cases.json"

    gate = runner._qwen3_p3_quality_gate(
        "http://127.0.0.1:1",
        "model",
        str(model),
        artifact_path=artifact,
    )

    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert gate["semantic_case_count"] == 4
    assert {(row["language"], row["length"], row["enable_thinking"]) for row in persisted["cases"]} == {
        ("zh", "short", True),
        ("en", "short", False),
        ("zh", "long", False),
        ("en", "long", True),
    }
    assert all(row["semantic_checks"]["passed"] is True for row in persisted["cases"])


def test_qwen3_p3_rejects_nonempty_but_semantically_incomplete_answer(tmp_path, monkeypatch):
    model = tmp_path / "qwen3"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            return str(messages[0]["content"])

        def __call__(self, prompt, *, add_special_tokens):
            return {"input_ids": list(range(160 if "背景资料" in prompt or "Background" in prompt else 16))}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer())),
    )
    monkeypatch.setattr(runner, "_json_request", lambda *args, **kwargs: {"choices": [{"text": "植物很重要。"}]})

    artifact = tmp_path / "quality_cases.json"
    with pytest.raises(RuntimeError, match="failed semantic checks"):
        runner._qwen3_p3_quality_gate(
            "http://127.0.0.1:1",
            "model",
            str(model),
            artifact_path=artifact,
        )
    persisted = json.loads(artifact.read_text(encoding="utf-8"))
    assert persisted["passed"] is False
    assert persisted["cases"][0]["completion"] == "植物很重要。"
    assert persisted["cases"][0]["semantic_checks"]["missing_groups"]


def test_qwen3_p3_accepts_a_semantically_complete_thinking_prefix():
    spec = runner._qwen3_p3_cases()[0]
    checks = runner._qwen3_p3_semantic_checks(
        spec,
        "<think>光合作用把二氧化碳和水转化为有机物，并释放氧气。",
    )
    assert checks["passed"] is True
    assert checks["answer_source"] == "thinking_prefix"


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
    assert bench["envs"]["CUDA_VISIBLE_DEVICES"] == ",".join(f"GPU-{i:02d}" for i in range(8))
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


@pytest.mark.parametrize("failure", ["timeout", "bad_json", "server_crash", "oom", "insufficient_free_memory"])
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
            if failure == "insufficient_free_memory" and stdout is not None:
                stdout.write("Free memory on device cuda:0 is less than desired GPU memory utilization\n")
                stdout.write("subsequent teardown detail\n" * 300)
                stdout.flush()

    monkeypatch.setattr(runner.subprocess, "Popen", FaultServer)
    monkeypatch.setattr(
        runner,
        "_wait_ready",
        lambda *args, **kwargs: failure not in {"server_crash", "oom", "insufficient_free_memory"},
    )

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
    if failure in {"oom", "insufficient_free_memory"}:
        assert "server_cuda_oom" in unified["failure_reason"]
    if failure == "server_crash":
        assert "server_exited_before_ready" in unified["failure_reason"]
    assert unified["cleanup_status"] == "released"
    with sqlite3.connect(session_dir / "storage" / "coordinator.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM gpu_leases").fetchone()[0] == 0


def test_runner_requires_validated_target_and_hardware(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPERLOOM_TARGET", raising=False)
    monkeypatch.delenv("HYPERLOOM_TARGET_RUNTIME", raising=False)
    with pytest.raises(ValueError, match="requires HYPERLOOM_TARGET"):
        runner.run_benchmark(_config(tmp_path), tmp_path / "output")


def test_lifecycle_ownership_is_published_before_server_ready(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors._server_lifecycle import teardown_lifecycle_server

    session_dir = _cuda_env(monkeypatch, tmp_path)
    path = _config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    pid_dir = tmp_path / "pid"
    cfg["benchmark"]["server_lifecycle"] = {"enabled": True, "cleanup": False, "pid_dir": str(pid_dir)}
    cfg["benchmark"]["envs"]["PORT"] = 18080
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runner.subprocess, "Popen", _FinishedServer)
    monkeypatch.setattr(runner.os, "getpgid", lambda pid: pid)

    def killed_before_ready(*args, **kwargs):
        metadata = json.loads((pid_dir / "vllm_18080.json").read_text())
        assert metadata["gpu_lease_holder"].startswith("vllm_cuda:")
        # The supervisor can recover ownership without the runner's finally.
        teardown_lifecycle_server(pid_dir=pid_dir, framework="vllm", port=18080)
        with sqlite3.connect(session_dir / "storage/coordinator.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM gpu_leases").fetchone()[0] == 0
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_wait_ready", killed_before_ready)
    assert runner.run_benchmark(path, tmp_path / "output") == 1


def test_cancel_releases_lease_before_slow_server_teardown(tmp_path, monkeypatch):
    session_dir = _cuda_env(monkeypatch, tmp_path)
    monkeypatch.setattr(runner.subprocess, "Popen", _FinishedServer)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_wait_ready", interrupted)

    def teardown(proc):
        with sqlite3.connect(session_dir / "storage/coordinator.db") as db:
            assert db.execute("SELECT COUNT(*) FROM gpu_leases").fetchone()[0] == 0

    monkeypatch.setattr(runner, "_terminate_group", teardown)
    assert runner.run_benchmark(_config(tmp_path), tmp_path / "out") == 1


def test_lifecycle_requires_an_ownership_directory(tmp_path, monkeypatch):
    _cuda_env(monkeypatch, tmp_path)
    path = _config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["benchmark"]["server_lifecycle"] = {"enabled": True}
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: pytest.fail("server launched"))
    with pytest.raises(ValueError, match="requires pid_dir"):
        runner.run_benchmark(path, tmp_path / "out")


def test_runtime_python_requires_an_executable_absolute_interpreter(tmp_path):
    runtime_python = tmp_path / "runtime" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("#!/bin/sh\n", encoding="utf-8")
    runtime_python.chmod(0o755)

    assert runner._runtime_python({"runtime": {"python": str(runtime_python)}}) == str(runtime_python.resolve())
    with pytest.raises(ValueError, match="executable absolute path"):
        runner._runtime_python({"runtime": {"python": "relative/python"}})


def test_runtime_pythonpath_requires_an_absolute_source_directory(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    assert runner._runtime_pythonpath({"runtime": {"pythonpath": str(source_root)}}) == str(source_root.resolve())
    with pytest.raises(ValueError, match="absolute directory"):
        runner._runtime_pythonpath({"runtime": {"pythonpath": "relative/source"}})


def test_extra_args_accept_vllm_numa_binding_capability(monkeypatch):
    monkeypatch.setenv(
        "HYPERLOOM_HARDWARE_FINGERPRINT",
        json.dumps({"vllm_cli": {"server_flags": ["--numa-bind", "--numa-bind-nodes"]}}),
    )
    assert runner._tokenize_extra_args({"EXTRA_VLLM_ARGS": "--numa-bind --numa-bind-nodes 0 1"}) == [
        "--numa-bind",
        "--numa-bind-nodes",
        "0",
        "1",
    ]
