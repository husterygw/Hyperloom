# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Nsight evidence integrity, capability selection and score isolation."""

import csv
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors.cuda_nsight import (
    analyze_nsys,
    analyze_ncu,
    phase_name,
    profiler_argv,
    roofline_point,
    union_ns,
    write_analysis,
)
from hyperloom.orchestrator.actions.executors.cuda_roofline import CudaRooflineExecutor
from hyperloom.orchestrator.actions.executors.cuda_profile import CudaProfileExecutor
from hyperloom.inference_optimizer.target_registry import (
    get_target,
    NVIDIA_LOCAL_TARGET,
    validate_target_arguments,
    TargetValidationError,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator
from hyperloom.orchestrator.kernel.roofline_snapshot import extract_workload_summary, extract_top_kernel


def timeline(tmp_path):
    path = tmp_path / "capture.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
CREATE TABLE StringIds(id INT,value TEXT);
INSERT INTO StringIds VALUES(1,'compute'),(2,'ncclKernel'),(3,'execute_context_0(0)_generation_2(2)');
CREATE TABLE PROCESSES(globalPid INT,pid INT);
CREATE TABLE TARGET_INFO_GPU(id INT,uuid TEXT);
INSERT INTO TARGET_INFO_GPU VALUES(7,'a'),(2,'b');
CREATE TABLE NVTX_EVENTS(start INT,end INT,globalTid INT,text TEXT,textId INT);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INT,end INT,globalTid INT,correlationId INT);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT,end INT,deviceId INT,globalPid INT,demangledName INT,correlationId INT,gridX INT,gridY INT,gridZ INT,blockX INT,blockY INT,blockZ INT);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INT,end INT,globalPid INT);
""")
        for pid in (11, 12):
            db.execute("INSERT INTO PROCESSES VALUES(?,?)", (pid << 24, pid))
        db.execute("INSERT INTO NVTX_EVENTS VALUES(0,100,?,NULL,3)", ((11 << 24) + 11,))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(5,9,?,1)", ((11 << 24) + 11,))
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,1,1,1,32,1,1)",
            [(10, 40, 7, 11 << 24, 1, 1), (20, 50, 7, 11 << 24, 2, 2), (0, 100, 2, 12 << 24, 1, 3)],
        )
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(35,60,?)", (11 << 24,))
    workers = [{"pid": 11, "rank": 0, "gpu_uuid": "GPU-a"}, {"pid": 12, "rank": 1, "gpu_uuid": "GPU-b"}]
    return path, workers


def test_nsys_uuid_identity_and_interval_unions(tmp_path):
    path, workers = timeline(tmp_path)
    summary = analyze_nsys(path, workers, fingerprints={"model": "abc"})
    rank = summary["ranks"][0]
    assert rank["compute_pct"] == 30
    assert rank["communication_pct"] == 30
    assert rank["overlap_pct"] == 20
    assert rank["copy_pct"] == 25
    assert rank["idle_pct"] == 50
    assert "decode" in summary["hot_kernels"][0]["phases"]
    write_analysis(tmp_path, summary)
    assert extract_workload_summary(tmp_path / "analysis.md")["idle_pct"] == 25
    assert extract_top_kernel(tmp_path / "analysis.md")["name"] == "compute"


def test_nsys_rejects_wrong_uuid_and_missing_rank(tmp_path):
    path, workers = timeline(tmp_path)
    workers[0]["gpu_uuid"] = "GPU-b"
    with pytest.raises(ValueError, match="UUID"):
        analyze_nsys(path, workers, fingerprints={})
    workers[0]["gpu_uuid"] = "GPU-a"
    workers.append({"pid": 13, "rank": 2, "gpu_uuid": "GPU-c"})
    with pytest.raises(ValueError, match="every rank"):
        analyze_nsys(path, workers, fingerprints={})


def test_union_and_phase_do_not_invent_evidence():
    assert union_ns([(0, 10), (2, 5), (5, 12), (20, 23)]) == 15
    assert phase_name("execute_context_3(10)_generation_2(2)") == "mixed"
    assert phase_name("fused_matmul") == "unknown"


def metrics():
    return {
        "gpu__time_duration.sum": "1000",
        "dram__bytes.sum.per_second": "1000000000",
        "dram__bytes.sum.peak_sustained_elapsed.per_second": "2000000000",
        "sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum.per_second": "100000000000",
        "sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum.peak_sustained_elapsed.per_second": "400000000000",
    }


def test_roofline_units_and_missing_metrics():
    data = metrics()
    point = roofline_point(data)
    assert point["operations"] == 100000
    assert point["dram_bytes"] == 1000
    assert point["arithmetic_intensity"] == 100
    assert point["bound_type"] == "memory"
    assert point["efficiency_percent"] == 50
    data["dram__bytes.sum.per_second"] = "n/a"
    assert roofline_point(data)["status"] == "unavailable"


def test_ncu_wide_csv_process_and_unit_validation(tmp_path):
    path = tmp_path / "capture.csv"
    data = {
        "ID": "0",
        "Process ID": "11",
        "Kernel Name": "compute",
        "Grid Size": "(1, 1, 1)",
        "Block Size": "(32, 1, 1)",
        **metrics(),
    }
    units = dict.fromkeys(data, "")
    units["gpu__time_duration.sum"] = "ns"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data))
        writer.writeheader()
        writer.writerow(units)
        writer.writerow(data)
    workers = [{"pid": 11, "rank": 0, "gpu_uuid": "GPU-a"}]
    result = analyze_ncu(path, workers, "compute")
    assert result["passed"]
    assert result["launches"][0]["ncu_roofline"]["arithmetic_intensity"] == 100
    with pytest.raises(ValueError, match="process or kernel"):
        analyze_ncu(path, workers, "different")
    path.write_text(path.read_text().replace("ns,", "us,"))
    with pytest.raises(ValueError, match="ns"):
        analyze_ncu(path, workers, "compute")


@pytest.mark.parametrize("enabled", [False, True])
def test_nsys_capabilities_and_serving_score_isolation(enabled):
    args = SimpleNamespace(optimization_level="profile", profile_backend="nsys", enable_roofline=enabled)
    validate_target_arguments(args, get_target(NVIDIA_LOCAL_TARGET))
    assert args.target_capabilities["trace_analysis"]
    assert args.target_capabilities["roofline"] == enabled
    result = {
        "status": "succeeded",
        "measurement_kind": "profile",
        "trace_health": {"passed": True},
        "counter_health": {"passed": True},
        "output_throughput": 999999,
    }
    for kind in ("baseline", "explore", "sweep"):
        assert not WritebackCollaborator._is_promotable_result(None, kind, result)
    assert WritebackCollaborator._is_promotable_result(None, "roofline", result)
    result["counter_health"]["passed"] = False
    assert not WritebackCollaborator._is_promotable_result(None, "roofline", result)
    with pytest.raises(TargetValidationError, match="requires"):
        validate_target_arguments(SimpleNamespace(profile_backend="nsys"), get_target(NVIDIA_LOCAL_TARGET))


def test_profiler_commands_preserve_graphs_and_exclude_communication(tmp_path):
    server = ["python", "-m", "vllm.entrypoints.cli.main", "serve", "/model"]
    cmd = profiler_argv("nsys", server, tmp_path)
    assert cmd[-len(server) :] == server
    assert "--cuda-graph-trace=node" in cmd
    assert "--enforce-eager" not in cmd
    cmd = profiler_argv("ncu", server, tmp_path, "kernel<T>")
    assert cmd[cmd.index("--launch-count") + 1] == "3"
    assert "--target-processes" in cmd
    with pytest.raises(ValueError, match="non-communication"):
        profiler_argv("ncu", server, tmp_path, "ncclKernel")


@pytest.mark.asyncio
@pytest.mark.parametrize("counter_failed", [False, True, "identity"])
async def test_composite_records_analysis_without_changing_current_best(tmp_path, monkeypatch, counter_failed):
    db, workers = timeline(tmp_path)
    summary = analyze_nsys(db, workers, fingerprints={})
    for hotspot in summary["hot_kernels"]:
        hotspot["first_step_by_rank"] = {"0": 9, "1": 13}
    write_analysis(tmp_path, summary)
    state = SharedState(model_path="/model", framework="vllm", tp=2, pp=4, current_best={"output_throughput": 123})
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.cuda_roofline.session_grid_bounds", lambda s: (None, None)
    )

    async def capture(self, ctx):
        if self.backend == "nsys":
            return {
                "status": "succeeded",
                "workspace": str(tmp_path),
                "main_trace_path": str(db),
                "trace_health": {"passed": True},
                "measurement_kind": "profile",
                "fingerprints": {
                    k: "identity" for k in ("hardware", "model", "workload", "serving_config", "quality_suite")
                },
            }
        assert self.delay_iterations == 12
        assert self.max_iterations == 12
        if counter_failed is True:
            return {"status": "failed", "error": "counter permissions"}
        return {
            "status": "succeeded",
            "profile_artifact": "counter.json",
            "fingerprints": {
                k: ("different" if counter_failed == "identity" and k == "serving_config" else "identity")
                for k in ("hardware", "model", "workload", "serving_config", "quality_suite")
            },
            "trace_files": ["counter.ncu-rep"],
            "trace_health": {
                "ranks": [{"rank": 0}, {"rank": 1}],
                "launches": [
                    {
                        "rank": rank,
                        "grid": "(1, 1, 1)",
                        "block": "(32, 1, 1)",
                        "metrics": {"raw_counter": "retained in counter.json"},
                        "ncu_roofline": roofline_point(metrics()),
                    }
                    for rank in (0, 1)
                ],
            },
        }

    monkeypatch.setattr(CudaProfileExecutor, "__call__", capture)
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="test", params={"output_dir": str(tmp_path / "run")}), extra={})
    (tmp_path / "run").mkdir()
    result = await CudaRooflineExecutor(shared_state=state, session_dir=tmp_path)(ctx)
    assert result["status"] == ("failed" if counter_failed else "succeeded")
    if counter_failed == "identity":
        assert "fingerprint mismatch" in result["error"]
    assert state.current_best == {"output_throughput": 123}
    assert "NVIDIA Nsight" in state.last_trace_analyze["analysis_md_text"]
    assert state.last_profile_status == "succeeded"
    assert state.last_trace_analyze["roofline_snapshot_id"] == 1
    if not counter_failed:
        counters = json.loads((tmp_path / "nsight_summary.json").read_text())["hot_kernels"]
        assert all(
            "metrics" not in launch for row in counters for launch in row.get("ncu_roofline", {}).get("launches", [])
        )


def test_nv_snapshot_never_uses_amd_model_ceiling(tmp_path, monkeypatch):
    from hyperloom.orchestrator.kernel.roofline_ceiling import compute_roofline_breakdown_from_state
    from hyperloom.orchestrator.kernel.roofline_snapshot import attach_perfmodel_breakdown

    state = SharedState(target_id=NVIDIA_LOCAL_TARGET, gpu_type="mi300x")
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.roofline_ceiling.resolve_runtime_workload",
        lambda *a, **kw: pytest.fail("AMD model called for NVIDIA"),
    )
    assert compute_roofline_breakdown_from_state(state).peak_tok_per_sec == 0
    snapshot = {}
    attach_perfmodel_breakdown(snapshot, state, arm="baseline")
    assert snapshot["roofline_provenance"]["e2e_ceiling"] == "unavailable"


@pytest.mark.asyncio
async def test_counter_timeout_cancels_child_and_retains_timeline(tmp_path, monkeypatch):
    import asyncio

    db, workers = timeline(tmp_path)
    summary = analyze_nsys(db, workers, fingerprints={})
    write_analysis(tmp_path, summary)
    state = SharedState(target_id=NVIDIA_LOCAL_TARGET, framework="vllm")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors.cuda_roofline.session_grid_bounds",
        lambda s: (time.monotonic() + 60.05, None),
    )
    cleaned = []

    async def capture(self, ctx):
        if self.backend == "nsys":
            return {"status": "succeeded", "workspace": str(tmp_path), "main_trace_path": str(db)}
        try:
            await asyncio.sleep(10)
        finally:
            cleaned.append(True)

    monkeypatch.setattr(CudaProfileExecutor, "__call__", capture)
    root = tmp_path / "run"
    root.mkdir()
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="timeout", params={"output_dir": str(root)}), extra={})
    result = await CudaRooflineExecutor(shared_state=state, session_dir=tmp_path)(ctx)
    assert cleaned and result["status"] == "failed"
    assert "exhausted" in result["error"]
    assert json.loads((tmp_path / "nsight_summary.json").read_text())["counter_status"] == "failed"
    assert Path(state.last_trace_analyze["analysis_md_path"]).is_file()


def test_nsight_backend_resume_flags_are_explicit():
    from hyperloom.inference_optimizer.cli.parser import _build_parser

    parser = _build_parser()
    defaults = parser.parse_args(["optimize"])
    assert not getattr(defaults, "enable_roofline_explicit", False)
    explicit = parser.parse_args(["optimize", "--no-enable-roofline"])
    assert explicit.enable_roofline_explicit and not explicit.enable_roofline


def test_ncu_export_keeps_identity_and_records_nvtx():
    from hyperloom.orchestrator.actions.executors.cuda_nsight import ncu_export_argv

    command = ncu_export_argv(Path("/capture.ncu-rep"), nvtx=True)
    assert command[command.index("--rename-kernels") + 1] == "off"
    assert command[command.index("--print-units") + 1] == "base"
    assert command[command.index("--print-nvtx-rename") + 1] == "kernel"


def test_live_rank_mapping_uses_log_prefix_when_ncu_suppresses_title(tmp_path, monkeypatch):
    import sys
    from hyperloom.orchestrator.actions.executors.cuda_nsight import rank_processes

    log = tmp_path / "server.log"
    log.write_text("(Worker_PP0_TP0 pid=11) ready\n(Worker_PP0_TP1 pid=12) ready\n")
    real_text = Path.read_text

    def read_text(path, *a, **kw):
        if str(path) in ("/proc/11/stat", "/proc/12/stat"):
            return "11 (python) S 10 10 10"
        return real_text(path, *a, **kw)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"python\0multiprocessing.spawn\0")
    monkeypatch.setitem(
        sys.modules,
        "pynvml",
        SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
            nvmlDeviceGetPciInfo=lambda h: SimpleNamespace(busId="00000000:4F:00.0"),
            nvmlDeviceGetComputeRunningProcesses=lambda h: [SimpleNamespace(pid=11 if h == "GPU-a" else 12)],
        ),
    )
    rows = rank_processes({"tp": 2, "pp": 1, "world_size": 2, "gpu_uuids": ["GPU-a", "GPU-b"]}, 10, log)
    assert [(r["rank"], r["pid"], r["gpu_uuid"]) for r in rows] == [(0, 11, "GPU-a"), (1, 12, "GPU-b")]
    log.write_text("(Worker_PP0_TP0 pid=11) ready\n(Worker_PP1_TP0 pid=11) changed\n")
    with pytest.raises(ValueError, match="conflicting"):
        rank_processes({"tp": 2, "world_size": 2, "gpu_uuids": ["GPU-a", "GPU-b"]}, 10, log)


@pytest.mark.asyncio
async def test_nvidia_trace_request_never_enters_tracelens(tmp_path, monkeypatch):
    from hyperloom.orchestrator.kernel import request_handlers

    db, workers = timeline(tmp_path)
    summary = analyze_nsys(db, workers, fingerprints={"model": "x"})
    write_analysis(tmp_path, summary)
    state = SharedState(
        target_id=NVIDIA_LOCAL_TARGET,
        profile_backend="nsys",
        target_capabilities={"trace_analysis": True},
        last_profile_trace=str(db),
    )
    state.save(tmp_path)
    (tmp_path / "vllm_cuda_profile.json").write_text(
        json.dumps({"backend": "nsys", "status": "succeeded", "fingerprints": {"model": "x"}})
    )
    monkeypatch.setattr(request_handlers, "_kernel_agent_root_error", lambda: pytest.fail("AMD/TraceLens entered"))
    result = await request_handlers.trace_analyze_handler({"trace_input": str(db)}, session_dir=tmp_path)
    assert result["status"] == "succeeded"
    rejected = await request_handlers.trace_analyze_handler(
        {"trace_input": str(tmp_path / "other.sqlite")}, session_dir=tmp_path
    )
    assert rejected["status"] == "failed"


def test_profiler_launcher_publishes_ownership_before_exec(tmp_path):
    import os
    import subprocess
    import sys
    from hyperloom.orchestrator.actions.executors._server_lifecycle import teardown_lifecycle_server

    ownership = tmp_path / "owner.json"
    ownership.write_text(json.dumps({"pid_dir": str(tmp_path), "port": 19081, "model": "test", "metadata": {}}))
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hyperloom.orchestrator.actions.executors.cuda_profiler_launch",
            str(ownership),
            sys.executable,
            "-c",
            "import time;time.sleep(60)",
            "vllm.entrypoints.cli.main",
        ],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        path = tmp_path / "vllm_19081.pid"
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert path.is_file()
        pid, pgid = map(int, path.read_text().split())
        assert pid == process.pid and os.getpgid(pid) == pgid
        teardown_lifecycle_server(pid_dir=tmp_path, framework="vllm", port=19081)
        process.wait(timeout=10)
        assert not path.exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 15)
            process.wait(timeout=10)


def test_owned_server_exec_transition_does_not_look_like_pid_reuse(monkeypatch):
    from hyperloom.orchestrator.actions.executors import _server_lifecycle as lifecycle

    values = iter(["", "", "python -m vllm.entrypoints.cli.main serve /model"])
    monkeypatch.setattr(lifecycle, "_pid_cmdline", lambda pid: next(values))
    monkeypatch.setattr(lifecycle, "_pid_alive_simple", lambda pid: True)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda seconds: None)
    assert lifecycle._looks_like_server_process(11)
    monkeypatch.setattr(lifecycle, "_pid_cmdline", lambda pid: "unrelated-process")
    assert not lifecycle._looks_like_server_process(11)


def test_hotspot_window_counts_empty_pipeline_steps(tmp_path):
    path, workers = timeline(tmp_path)
    with sqlite3.connect(path) as db:
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,NULL)",
            [
                (-20, -15, (11 << 24) + 11, "execute_context_0(0)_generation_0(0)"),
                (-10, -5, (11 << 24) + 11, "execute_context_0(0)_generation_0(0)"),
            ],
        )
    summary = analyze_nsys(path, workers, fingerprints={})
    compute = next(r for r in summary["hot_kernels"] if r["name"] == "compute")
    assert compute["first_step_by_rank"]["0"] == 3


def test_profile_tools_survive_state_roundtrip(tmp_path):
    tools = {"nsys": {"path": "/nsys", "version": "test"}, "ncu": {"path": "/ncu", "version": "test"}}
    state = SharedState(
        target_id=NVIDIA_LOCAL_TARGET, profile_backend="nsys", profile_tool_fingerprint=tools, enable_roofline=False
    )
    state.save(tmp_path)
    restored = SharedState.load_or_init(tmp_path)
    assert restored.profile_tool_fingerprint == tools
    assert restored.profile_backend == "nsys" and not restored.enable_roofline


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", ["roofline", "tools"])
async def test_nsight_resume_rejects_changed_capture_identity(tmp_path, monkeypatch, capsys, conflict):
    from hyperloom.inference_optimizer.cli import _run_optimize
    from hyperloom.inference_optimizer.cli.parser import _build_parser
    from hyperloom.inference_optimizer import target_registry

    saved = {"nsys": {"path": "/nsys", "version": "old"}}
    SharedState(
        target_id=NVIDIA_LOCAL_TARGET,
        optimization_level="profile",
        profile_backend="nsys",
        profile_tool_fingerprint=saved,
        enable_roofline=False,
    ).save(tmp_path)
    monkeypatch.delenv("HYPERLOOM_TARGET", raising=False)
    monkeypatch.setattr(target_registry, "validate_nvidia_host", lambda *a: {})
    monkeypatch.setattr(target_registry, "validate_profile_runtime", lambda *a, **kw: {"nsys": {"version": "new"}})
    monkeypatch.setattr(
        target_registry, "configure_target_environment", lambda *a, **kw: pytest.fail("accepted resume conflict")
    )
    extra = ["--enable-roofline"] if conflict == "roofline" else []
    args = _build_parser().parse_args(["optimize", "--resume-from", str(tmp_path), *extra])
    with pytest.raises(SystemExit) as exc:
        await _run_optimize(args)
    assert exc.value.code == 2
    assert (
        "enable_roofline conflict" if conflict == "roofline" else "tool fingerprint changed"
    ) in capsys.readouterr().err
    assert SharedState.load_or_init(tmp_path).profile_tool_fingerprint == saved


@pytest.mark.parametrize("counter_status", ["succeeded", "failed", "not_requested"])
def test_nsight_reports_keep_diagnostic_evidence(tmp_path, monkeypatch, counter_status):
    from hyperloom.inference_optimizer.breakdown import exporter
    from hyperloom.orchestrator.actions.executors import report

    db, workers = timeline(tmp_path)
    summary = analyze_nsys(db, workers, fingerprints={})
    summary["counter_status"] = counter_status
    state = SharedState(
        target_id=NVIDIA_LOCAL_TARGET,
        profile_backend="nsys",
        framework="vllm",
        baseline_tput=123,
        current_best={"action": "baseline", "tput": 123},
    )
    state.last_profile_trace = str(db)
    state.record_trace_analyze({"trace_input": str(db)}, write_analysis(tmp_path, summary))
    state.save(tmp_path)
    monkeypatch.setattr(exporter, "_crash_safe_platform", lambda gpu: {})
    monkeypatch.setattr(report, "_platform_fingerprint", lambda gpu: {})
    fallback_md = exporter.write_minimal_final_report(tmp_path).read_text()
    fallback = json.loads(exporter.write_minimal_final_json(tmp_path).read_text())
    regular = report._build_summary_dict(state, {}, [], session_dir=tmp_path)
    for data in (fallback, regular):
        assert data["nsight_analysis"]["counter_status"] == counter_status
        assert data["nsight_analysis"]["e2e_ceiling"] == "unavailable"
        assert data["baseline_tput"] == 123
    for text in (fallback_md, report._format_md(regular)):
        assert "NVIDIA Nsight diagnostics" in text
        assert str(tmp_path / "analysis.md") in text
        assert "HBM" not in text


def test_nsight_uses_preflight_executable_after_path_changes(tmp_path, monkeypatch):
    from hyperloom.orchestrator.actions.executors import cuda_nsight

    expected = {"nsys": {"path": "/selected/nsys", "version": "v1"}}
    selected = []
    monkeypatch.setattr(cuda_nsight.shutil, "which", lambda name: selected.append(name) or name)
    monkeypatch.setattr(cuda_nsight.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="v1\n"))
    assert cuda_nsight.tool_fingerprint(roofline=False, expected=expected) == expected
    assert selected == ["/selected/nsys"]
    assert profiler_argv("nsys", ["server"], tmp_path, tool_paths=expected)[0] == "/selected/nsys"
    assert cuda_nsight.ncu_export_argv(tmp_path / "capture.ncu-rep", executable="/selected/ncu")[0] == "/selected/ncu"
    monkeypatch.setattr(cuda_nsight.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="v2\n"))
    with pytest.raises(ValueError, match="identity changed"):
        cuda_nsight.tool_fingerprint(roofline=False, expected=expected)
