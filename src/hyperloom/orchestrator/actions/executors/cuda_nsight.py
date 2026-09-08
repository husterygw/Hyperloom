# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Nsight collection artifacts and deterministic CUDA analysis.

No serving scores are produced here. All timings refer to instrumented runs.
"""

from __future__ import annotations

import csv
from bisect import bisect_right
import hashlib
import json
import math
import re
import shutil
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


def tool_fingerprint(*, roofline: bool, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {}
    for name in ["nsys", "ncu"] if roofline else ["nsys"]:
        identity = (expected or {}).get(name) or {}
        executable = shutil.which(identity.get("path") or name)
        if not executable:
            raise ValueError(f"NVIDIA profiling requires {name} on PATH")
        version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=15, check=True)
        result[name] = {"path": str(Path(executable).resolve()), "version": version.stdout.strip()}
        if identity and result[name] != identity:
            raise ValueError(f"{name} tool identity changed since session preflight")
    return result


def profiler_argv(
    backend: str, server: list[str], workspace: Path, kernel_name: str = "", *, tool_paths: dict[str, Any] | None = None
) -> list[str]:
    executable = ((tool_paths or {}).get(backend) or {}).get("path") or backend
    if backend == "nsys":
        return [
            executable,
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
            "--kill=none",
            "--wait=all",
            "-o",
            str(workspace / "capture"),
            *server,
        ]
    if backend != "ncu" or not kernel_name or is_communication(kernel_name):
        raise ValueError("ncu requires an explicit non-communication kernel")
    return [
        executable,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--nvtx",
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--graph-profiling",
        "node",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        "regex:^" + re.escape(kernel_name) + "$",
        "--rename-kernels",
        "off",
        "--filter-mode",
        "per-gpu",
        "--launch-count",
        "3",
        "--section",
        "SpeedOfLight",
        "--section",
        "SpeedOfLight_RooflineChart",
        "--section",
        "SpeedOfLight_HierarchicalTensorRooflineChart",
        "-o",
        str(workspace / "capture"),
        *server,
    ]


def rank_processes(topology: dict[str, Any], service_pid: int, server_log: Path | None = None) -> list[dict[str, Any]]:
    """Bind live workers to leased UUIDs; validate their process ancestry/rank."""
    import pynvml

    def belongs(pid: int) -> bool:
        visited = set()
        while pid > 1 and pid not in visited:
            if pid == service_pid:
                return True
            visited.add(pid)
            try:
                pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                return False
        return False

    # Nsight Compute can suppress setproctitle; vLLM's own PID-tagged log
    # prefixes retain TP/PP identity. GPU UUID and ancestry are checked below.
    log_names: dict[int, str] = {}
    if server_log is not None:
        for name, pid in re.findall(r"\((Worker[^()\n]*?) pid=(\d+)\)", server_log.read_text(errors="replace")):
            if "_PP" in name or "_TP" in name:
                previous = log_names.get(int(pid))
                if previous is not None and previous != name:
                    raise ValueError(f"conflicting rank names for PID {pid}")
                log_names[int(pid)] = name
    rows = {}
    pynvml.nvmlInit()
    try:
        for uuid in topology["gpu_uuids"]:
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            pci_bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
            if isinstance(pci_bus_id, bytes):
                pci_bus_id = pci_bus_id.decode()
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if not belongs(proc.pid):
                    continue
                title = Path(f"/proc/{proc.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if topology["world_size"] == 1:
                    rank, pp, tp = 0, 0, 0
                else:
                    rank_title = log_names.get(proc.pid) or title
                    if "Worker" not in rank_title:
                        continue
                    ppm = re.search(r"_PP(\d+)", rank_title)
                    tpm = re.search(r"_TP(\d+)", rank_title)
                    pp = int(ppm[1]) if ppm else 0
                    tp = int(tpm[1]) if tpm else 0
                    rank = pp * topology["tp"] + tp
                if rank in rows:
                    raise ValueError(f"ambiguous CUDA worker for rank {rank}")
                rows[rank] = {
                    "rank": rank,
                    "pp_rank": pp,
                    "tp_rank": tp,
                    "pid": proc.pid,
                    "gpu_uuid": uuid,
                    "pci_bus_id": pci_bus_id,
                    "process_title": title.strip(),
                    "rank_log_name": log_names.get(proc.pid, ""),
                }
    finally:
        pynvml.nvmlShutdown()
    if set(rows) != set(range(topology["world_size"])):
        raise ValueError(f"live worker coverage mismatch: {sorted(rows)}")
    return [rows[r] for r in sorted(rows)]


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total, end = 0, -1
    for a, b in sorted(intervals):
        if b > max(end, a):
            total += b - max(end, a)
        end = max(end, b)
    return total


def is_communication(name: str) -> bool:
    return bool(re.search(r"nccl|all.?reduce|all.?gather|reduce.?scatter|send.?recv|nvshmem|cross_device", name, re.I))


def phase_name(name: str) -> str:
    match = re.search(r"context_(\d+).*generation_(\d+)", name)
    if not match:
        return "unknown"
    ctx, gen = map(int, match.groups())
    return "mixed" if ctx and gen else "prefill" if ctx else "decode" if gen else "unknown"


def analyze_nsys(database: Path, workers: list[dict[str, Any]], *, fingerprints: dict[str, Any]) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            raise ValueError("nsys report contains no CUDA kernel table")
        strings = dict(conn.execute("SELECT id,value FROM StringIds"))
        processes = {r["globalPid"]: r["pid"] for r in conn.execute("SELECT * FROM PROCESSES")}
        gpu_uuids = {
            r["id"]: "GPU-" + str(r["uuid"]).removeprefix("GPU-")
            for r in conn.execute("SELECT id,uuid FROM TARGET_INFO_GPU")
        }
        by_pid = {r["pid"]: r for r in workers}
        ranges = defaultdict(list)
        if "NVTX_EVENTS" in tables:
            for raw in conn.execute("SELECT * FROM NVTX_EVENTS WHERE end > start"):
                r = dict(raw)
                name = r.get("text") or strings.get(r.get("textId"), "")
                if "execute_" in name:
                    ranges[r["globalTid"]].append((r["start"], r["end"], phase_name(name)))
        # vLLM's profiler counts execute_model calls, including empty pipeline
        # steps. Preserve their order to select a counter window from evidence.
        range_starts = {}
        for tid in ranges:
            ranges[tid].sort()
            range_starts[tid] = [r[0] for r in ranges[tid]]
        execution_starts = defaultdict(list)
        for tid, entries in ranges.items():
            execution_starts[(tid >> 24) << 24].extend(a for a, _, _ in entries)
        execution_starts = {pid: sorted(set(starts)) for pid, starts in execution_starts.items()}
        runtime = {}
        if ranges and "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
            for r in conn.execute("SELECT start,end,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
                pid_global = (r["globalTid"] >> 24) << 24
                step = bisect_right(execution_starts.get(pid_global, []), r["start"])
                runtime[(pid_global, r["correlationId"])] = ("unknown", step)
                # execute_model annotations are sequential on a worker thread.
                # Binary search avoids scanning thousands of steps per API call.
                index = bisect_right(range_starts.get(r["globalTid"], []), r["start"]) - 1
                if index >= 0:
                    a, b, phase = ranges[r["globalTid"]][index]
                    if a <= r["start"] <= b:
                        runtime[(pid_global, r["correlationId"])] = (phase, step)
        kernels = []
        for raw in conn.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end > start"):
            r = dict(raw)
            pid = processes.get(r["globalPid"])
            worker = by_pid.get(pid)
            if worker is None:
                continue
            if gpu_uuids.get(r["deviceId"]) != worker["gpu_uuid"]:
                raise ValueError(f"GPU UUID mismatch for rank {worker['rank']}")
            name = strings[r["demangledName"]]
            kernels.append(
                {
                    "rank": worker["rank"],
                    "pid": pid,
                    "gpu_uuid": worker["gpu_uuid"],
                    "name": name,
                    "start": r["start"],
                    "end": r["end"],
                    "phase": runtime.get((r["globalPid"], r["correlationId"]), ("unknown", 0))[0],
                    "step": runtime.get((r["globalPid"], r["correlationId"]), ("unknown", 0))[1],
                    "communication": is_communication(name),
                    "grid": [r[f"grid{x}"] for x in "XYZ"],
                    "block": [r[f"block{x}"] for x in "XYZ"],
                }
            )
        if {r["rank"] for r in kernels} != {r["rank"] for r in workers}:
            raise ValueError("nsys CUDA kernel coverage does not include every rank")
        start, end = min(k["start"] for k in kernels), max(k["end"] for k in kernels)
        window = end - start
        copies = defaultdict(list)
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if table not in tables:
                continue
            for raw in conn.execute(f"SELECT * FROM {table} WHERE end > start"):
                r = dict(raw)
                worker = by_pid.get(processes.get(r.get("globalPid")))
                if worker:
                    a, b = max(start, r["start"]), min(end, r["end"])
                    if b > a:
                        copies[worker["rank"]].append((a, b))
        ranks = []
        for worker in workers:
            ks = [k for k in kernels if k["rank"] == worker["rank"]]
            compute = [(k["start"], k["end"]) for k in ks if not k["communication"]]
            comm = [(k["start"], k["end"]) for k in ks if k["communication"]]
            busy = union_ns(compute + comm + copies[worker["rank"]])
            ranks.append(
                {
                    **worker,
                    "kernel_events": len(ks),
                    "window_ns": window,
                    "compute_pct": 100 * union_ns(compute) / window,
                    "communication_pct": 100 * union_ns(comm) / window,
                    "overlap_pct": 100 * (union_ns(compute) + union_ns(comm) - union_ns(compute + comm)) / window,
                    "copy_pct": 100 * union_ns(copies[worker["rank"]]) / window,
                    "idle_pct": 100 * (window - busy) / window,
                }
            )
        grouped = {}
        total = sum(k["end"] - k["start"] for k in kernels)
        for k in kernels:
            name = k["name"]
            row = grouped.setdefault(
                name,
                {
                    "kernel_id": hashlib.sha256(name.encode()).hexdigest()[:16],
                    "name": name,
                    "gpu_time_ns": 0,
                    "calls": 0,
                    "communication": k["communication"],
                    "ranks": set(),
                    "phases": set(),
                    "launches": set(),
                    "first_step_by_rank": {},
                    "reusable_native_kernel": False,
                },
            )
            row["gpu_time_ns"] += k["end"] - k["start"]
            row["calls"] += 1
            if k["step"] > 0:
                key = str(k["rank"])
                row["first_step_by_rank"][key] = min(k["step"], row["first_step_by_rank"].get(key, k["step"]))
            row["ranks"].add(k["rank"])
            row["phases"].add(k["phase"])
            row["launches"].add((k["rank"], tuple(k["grid"]), tuple(k["block"])))
        hot = sorted(grouped.values(), key=lambda r: r["gpu_time_ns"], reverse=True)
        for row in hot:
            row["gpu_pct"] = 100 * row["gpu_time_ns"] / total
            row["ranks"], row["phases"] = sorted(row["ranks"]), sorted(row["phases"])
            row["launches"] = [{"rank": r, "grid": list(g), "block": list(b)} for r, g, b in sorted(row["launches"])]
        return {
            "schema_version": 1,
            "backend": "nsys",
            "measurement_kind": "profile",
            "status": "succeeded",
            "fingerprints": fingerprints,
            "ranks": ranks,
            "hot_kernels": hot,
            "counter_status": "not_requested",
            "window": {"start_ns": start, "end_ns": end},
            "attribution": "native NVTX execution phases; layer identity unavailable for fused/graph kernels",
        }
    finally:
        conn.close()


def number(value: Any) -> float | None:
    try:
        n = float(str(value).replace(",", ""))
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def roofline_point(metrics: dict[str, Any]) -> dict[str, Any]:
    """Use NCU's operation-rate and peak metrics, never spec-sheet TFLOPS."""
    duration = number(metrics.get("gpu__time_duration.sum"))
    bandwidth = number(metrics.get("dram__bytes.sum.per_second"))
    peak_bw = number(metrics.get("dram__bytes.sum.peak_sustained_elapsed.per_second"))
    if peak_bw is None:
        per_cycle = number(metrics.get("dram__bytes.sum.peak_sustained"))
        clocks = number(metrics.get("dram__cycles_elapsed.avg.per_second"))
        peak_bw = per_cycle * clocks if per_cycle and clocks else None
    paths = []
    for key, value in metrics.items():
        if key.startswith("sm__ops_path_tensor_") and key.endswith(".sum.per_second"):
            rate = number(value)
            peak = number(metrics.get(key.replace(".sum.per_second", ".sum.peak_sustained_elapsed.per_second")))
            if rate and rate > 0 and peak:
                paths.append((key, rate, peak))
    if not paths:
        # Scalar FP32 arithmetic: sum across SMSPs, with FMA counting as two ops.
        cycle_rate = number(metrics.get("smsp__cycles_elapsed.avg.per_second"))
        parts = [
            number(metrics.get(f"smsp__sass_thread_inst_executed_op_{op}_pred_on.sum.per_cycle_elapsed"))
            for op in ("fadd", "fmul", "ffma")
        ]
        peak_cycle = number(metrics.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"))
        sm_clock = number(metrics.get("sm__cycles_elapsed.avg.per_second"))
        if cycle_rate and all(x is not None for x in parts) and peak_cycle and sm_clock:
            rate = (parts[0] + parts[1] + 2 * parts[2]) * cycle_rate
            if rate > 0:
                paths.append(("scalar_fp32", rate, 2 * peak_cycle * sm_clock))
    if not duration or not bandwidth or not peak_bw or len(paths) != 1:
        return {"status": "unavailable", "reason": "missing metrics, zero traffic, or mixed arithmetic paths"}
    path, rate, peak = paths[0]
    intensity = rate / bandwidth
    ceiling = min(peak, intensity * peak_bw)
    return {
        "status": "available",
        "arithmetic_path": path,
        "duration_ns": duration,
        "dram_bytes": bandwidth * duration / 1e9,
        "operations": rate * duration / 1e9,
        "arithmetic_intensity": intensity,
        "operations_per_second": rate,
        "compute_ceiling_ops_per_second": peak,
        "bandwidth_ceiling_bytes_per_second": peak_bw,
        "roofline_ops_per_second": ceiling,
        "efficiency_percent": 100 * rate / ceiling,
        "bound_type": "memory" if intensity * peak_bw < peak else "compute",
        "clock_control": "none",
        "cache_control": "none",
        "scope": "instrumented kernel launch",
    }


def ncu_export_argv(report: Path, *, nvtx: bool = False, executable: str = "ncu") -> list[str]:
    command = [
        executable,
        "--import",
        str(report),
        "--page",
        "raw",
        "--csv",
        "--print-units",
        "base",
        "--rename-kernels",
        "off",
    ]
    if nvtx:
        command += ["--print-nvtx-rename", "kernel"]
    return command


def analyze_ncu(csv_path: Path, workers: list[dict[str, Any]], kernel_name: str) -> dict[str, Any]:
    by_pid = {r["pid"]: r for r in workers}
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "Process ID" not in rows[0]:
        raise ValueError("invalid NCU CSV schema")
    units = rows[0]
    if units.get("gpu__time_duration.sum") != "ns":
        raise ValueError("NCU duration must be exported in ns")
    phases = {}
    phase_path = csv_path.with_name("capture_nvtx.csv")
    if phase_path.is_file():
        with phase_path.open(newline="") as stream:
            for r in csv.DictReader(stream):
                phases[(r.get("Process ID"), r.get("ID"))] = phase_name(r.get("Kernel Name") or "")
    launches = []
    for row in rows[1:]:
        worker = by_pid.get(int(row["Process ID"]))
        if worker is None or row["Kernel Name"] != kernel_name:
            raise ValueError("NCU process or kernel does not match capture selection")
        if worker.get("pci_bus_id"):
            expected_bus = int(worker["pci_bus_id"].split(":")[-2], 16)
            if number(row.get("device__attribute_pci_bus_id")) != expected_bus:
                raise ValueError("NCU device PCI identity does not match leased GPU")
        launches.append(
            {
                "rank": worker["rank"],
                "gpu_uuid": worker["gpu_uuid"],
                "pid": worker["pid"],
                "kernel_name": row["Kernel Name"],
                "phase": phases.get((row["Process ID"], row["ID"]), "unknown"),
                "grid": row["Grid Size"],
                "block": row["Block Size"],
                "metrics": row,
                "ncu_roofline": roofline_point(row),
            }
        )
    if not launches:
        raise ValueError("NCU report has no matched launches")
    counts = {rank: sum(r["rank"] == rank for r in launches) for rank in {r["rank"] for r in launches}}
    if any(count > 3 for count in counts.values()):
        raise ValueError("NCU launch count exceeded per-rank limit")
    return {
        "passed": True,
        "ranks": [{"rank": rank, "kernel_events": count} for rank, count in sorted(counts.items())],
        "launches": launches,
        "units": units,
        "trace_files": [str(csv_path)],
        "errors": [],
    }


def write_analysis(workspace: Path, summary: dict[str, Any]) -> dict[str, Any]:
    summary_path, report_path = workspace / "nsight_summary.json", workspace / "analysis.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = [
        "# NVIDIA Nsight analysis",
        "",
        "Instrumented diagnostics; not a serving performance score.",
        "",
        f"Counter status: {summary.get('counter_status', 'not_requested')}",
        "",
        "| Rank | GPU UUID | Compute % | Communication % | Overlap % | Copy % | Idle % |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary["ranks"]:
        lines.append(
            f"| {r['rank']} | {r['gpu_uuid']} | {r['compute_pct']:.2f} | {r['communication_pct']:.2f} | "
            f"{r['overlap_pct']:.2f} | {r['copy_pct']:.2f} | {r['idle_pct']:.2f} |"
        )
    lines += [
        "",
        "Shares use the common capture window and interval unions; compute and communication may overlap.",
        "Hotspot shares below use summed kernel durations, not wall-clock shares.",
        "",
        "| Kernel | Summed GPU % | Ranks | Phases |",
        "|---|---|---|---|",
    ]
    for row in summary["hot_kernels"][:15]:
        name = row["name"].replace("|", "\\|")
        lines.append(f"| {name} | {row['gpu_pct']:.2f} | {row['ranks']} | {row['phases']} |")
    for row in summary["hot_kernels"]:
        counter = row.get("ncu_roofline") or {}
        if not counter:
            continue
        lines += ["", f"## Counter samples: {row['kernel_id']}", "", f"Artifact: {counter['profile_artifact']}"]
        for launch in counter.get("launches", []):
            point = launch["ncu_roofline"]
            lines += [
                "",
                f"Rank {launch['rank']}, phase {launch.get('phase', 'unknown')}, "
                f"grid {launch['grid']}, block {launch['block']}: `{json.dumps(point, ensure_ascii=False)}`",
            ]
    lines += [
        "",
        summary["attribution"],
        "",
        "## Evidence",
        "",
        f"Structured analysis: {summary_path}",
        "",
        f"Fingerprints: `{json.dumps(summary['fingerprints'], sort_keys=True)}`",
    ]
    for error in summary.get("counter_errors", []):
        lines += ["", f"Counter collection incomplete: {error}"]
    report_path.write_text("\n".join(lines) + "\n")
    return {
        "status": "succeeded",
        "trace_report_path": str(report_path),
        "hot_kernels": summary["hot_kernels"],
        "artifact_paths": {"nsight_summary": str(summary_path)},
        "trace_health_warnings": [],
    }


def report_summary(state: Any) -> dict[str, Any]:
    """Small persisted evidence reference, including for interrupted sessions."""
    if getattr(state, "target_id", "") != "nvidia_rtx4090_8x_local" or getattr(state, "profile_backend", "") != "nsys":
        return {}
    analysis = getattr(state, "last_trace_analyze", {}) or {}
    if not analysis.get("analysis_md_path"):
        return {}
    # The normal snapshot keeps a generic top-kernel schema. Read the
    # deterministic status line already cached with the complete analysis.
    status = re.search(
        r"(?m)^Counter status: (succeeded|failed|not_requested|cancelled)$",
        analysis.get("analysis_md_text", ""),
    )
    return {
        "backend": "nsys",
        "analysis_md_path": analysis["analysis_md_path"],
        "trace_input": analysis.get("trace_input"),
        "snapshot_id": analysis.get("roofline_snapshot_id"),
        "counter_status": status[1] if status else "unknown",
        "scope": "instrumented selected kernels",
        "e2e_ceiling": "unavailable",
    }


def report_lines(summary: dict[str, Any]) -> list[str]:
    if not summary:
        return []
    return [
        "",
        "## NVIDIA Nsight diagnostics",
        "",
        f"Counter status: `{summary['counter_status']}`.",
        "",
        f"[Timeline and kernel analysis]({summary['analysis_md_path']})",
        "",
        "Counter samples describe instrumented selected kernels. "
        "They do not establish a model throughput ceiling or a serving performance gain.",
        "",
    ]
