# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native single-node CUDA benchmark runner for the pinned vLLM wheel.

The runner consumes the same materialized YAML and writes the same
``benchmark_report.json`` surface as Magpie while also emitting the richer
``vllm_cuda_benchmark/v1`` artifact.  It owns only process groups and GPU lease
rows it created; it never scans for or kills unrelated GPU processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.env_safety import build_benchmark_env
from hyperloom.common.jsonio import read_json
from hyperloom.orchestrator.bus.storage.schema import ensure_schema

from . import bypass_engine, bypass_report


SCHEMA_VERSION = "vllm_cuda_benchmark/v1"
TARGET_ID = "nvidia_rtx4090_8x_local"

# Config-only search surface validated against the pinned vLLM CLI. Unknown
# flags are rejected here instead of being forwarded to an environment whose
# parser may change across versions.
SUPPORTED_EXTRA_VLLM_FLAGS = frozenset(
    {
        "--async-scheduling",
        "--attention-backend",
        "--block-size",
        "--compilation-config",
        "--cpu-offload-gb",
        "--disable-cascade-attn",
        "--disable-custom-all-reduce",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "--kv-cache-dtype",
        "--kv-cache-memory-bytes",
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--safetensors-load-strategy",
    }
)


@dataclass(frozen=True)
class LaunchPlan:
    """Fully materialized launch contract persisted before server start."""

    target_id: str
    model: str
    served_model_name: str
    topology: dict[str, Any]
    port: int
    base_url: str
    server_argv: list[str]
    benchmark_argv: list[str]
    child_env: dict[str, str]
    artifacts: dict[str, str]
    fingerprints: dict[str, str]


@dataclass(frozen=True)
class _Lease:
    db_path: Path
    holder_id: str
    task_id: str
    gpu_ids: tuple[int, ...]
    gpu_uuids: tuple[str, ...]
    numa_nodes: tuple[int | None, ...]


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _model_fingerprint(model: str) -> str:
    path = Path(model).expanduser()
    payload: dict[str, Any] = {"model": model}
    if path.exists():
        resolved = path.resolve()
        payload["resolved_path"] = str(resolved)
        config_path = resolved / "config.json" if resolved.is_dir() else resolved
        try:
            payload["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        except OSError:
            payload["config_sha256"] = ""
        if resolved.is_dir():
            inventory: list[tuple[str, int]] = []
            for candidate in sorted(resolved.glob("*.safetensors*")):
                try:
                    inventory.append((candidate.name, candidate.stat().st_size))
                except OSError:
                    continue
            payload["checkpoint_inventory"] = inventory
    return _fingerprint(payload)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _workspace(output_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    workspace = (output_dir / f"benchmark_vllm_{stamp}").resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "torch_trace").mkdir()
    (workspace / "system_profile").mkdir()
    return workspace


def _pick_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tokenize_extra_args(envs: dict[str, Any]) -> list[str]:
    import shlex

    raw = str(envs.get("EXTRA_VLLM_ARGS") or "").strip()
    if not raw:
        return []
    tokens = shlex.split(raw)
    controlled = {
        "--host",
        "--port",
        "--served-model-name",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--max-model-len",
        "--model",
    }
    for token in tokens:
        flag = token.split("=", 1)[0]
        if flag in controlled:
            raise ValueError(f"EXTRA_VLLM_ARGS may not override runner-owned flag {flag}")
        if flag.startswith("--") and flag not in SUPPORTED_EXTRA_VLLM_FLAGS:
            raise ValueError(f"EXTRA_VLLM_ARGS contains unsupported flag {flag}")
    return tokens


def _visible_indices(envs: dict[str, Any], world_size: int) -> tuple[int, ...]:
    raw = str(envs.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw:
        return tuple(range(world_size))
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    try:
        indices = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("vllm_cuda MVP requires integer CUDA_VISIBLE_DEVICES indices") from exc
    if len(indices) < world_size:
        raise ValueError(f"CUDA_VISIBLE_DEVICES has {len(indices)} devices but TP*PP requires {world_size}")
    selected = indices[:world_size]
    if any(index < 0 for index in selected) or len(set(selected)) != len(selected):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain unique non-negative device indices")
    return selected


def _hardware_payload() -> dict[str, Any]:
    try:
        payload = json.loads(os.environ.get("HYPERLOOM_HARDWARE_FINGERPRINT", "") or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _hardware_rows() -> dict[int, dict[str, Any]]:
    payload = _hardware_payload()
    rows = payload.get("devices") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    return {int(row["index"]): row for row in rows if isinstance(row, dict) and str(row.get("index", "")).isdigit()}


def _session_db_path() -> Path:
    session_dir = os.environ.get("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", "").strip()
    if not session_dir:
        raise RuntimeError("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR is required for CUDA GPU leasing")
    return Path(session_dir).resolve() / "storage" / "coordinator.db"


def _acquire_gpu_lease(
    *,
    gpu_ids: tuple[int, ...],
    stable_key: str,
    ttl_sec: int,
) -> _Lease:
    db_path = _session_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    holder_id = f"vllm_cuda:{stable_key[:24]}"
    task_id = f"vllm_cuda_benchmark:{stable_key[:24]}"
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat(timespec="microseconds")
    expires_iso = datetime.fromtimestamp(now.timestamp() + max(60, ttl_sec), timezone.utc).isoformat(
        timespec="microseconds"
    )
    hardware = _hardware_rows()
    with sqlite3.connect(db_path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM gpu_leases WHERE expires_at <= ?", (now_iso,))
            rows = conn.execute(
                "SELECT gpu_id FROM gpu_leases WHERE holder_id=? AND task_id=? ORDER BY gpu_id",
                (holder_id, task_id),
            ).fetchall()
            existing = tuple(int(row["gpu_id"]) for row in rows)
            if existing and existing != tuple(sorted(gpu_ids)):
                raise RuntimeError(f"stale CUDA lease topology {existing}, requested {gpu_ids}")
            placeholders = ",".join("?" for _ in gpu_ids)
            conflicts = conn.execute(
                f"SELECT gpu_id, holder_id FROM gpu_leases WHERE gpu_id IN ({placeholders}) "  # nosec B608
                "AND NOT (holder_id=? AND task_id=?)",
                [*gpu_ids, holder_id, task_id],
            ).fetchall()
            if conflicts:
                detail = ", ".join(f"gpu{row['gpu_id']}:{row['holder_id']}" for row in conflicts)
                raise RuntimeError(f"CUDA GPU lease conflict: {detail}")
            if existing:
                conn.execute(
                    "UPDATE gpu_leases SET expires_at=?, heartbeat_at=? WHERE holder_id=? AND task_id=?",
                    (expires_iso, now_iso, holder_id, task_id),
                )
            else:
                for gpu_id in gpu_ids:
                    row = hardware.get(gpu_id, {})
                    conn.execute(
                        "INSERT INTO gpu_leases(gpu_id,gpu_uuid,numa_node,holder_id,task_id,"
                        "acquired_at,expires_at,heartbeat_at) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            gpu_id,
                            str(row.get("uuid") or "") or None,
                            row.get("numa_node"),
                            holder_id,
                            task_id,
                            now_iso,
                            expires_iso,
                            now_iso,
                        ),
                    )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    refs = [hardware.get(gpu_id, {}) for gpu_id in gpu_ids]
    return _Lease(
        db_path=db_path,
        holder_id=holder_id,
        task_id=task_id,
        gpu_ids=gpu_ids,
        gpu_uuids=tuple(str(row.get("uuid") or "") for row in refs),
        numa_nodes=tuple(row.get("numa_node") for row in refs),
    )


def _release_gpu_lease(lease: _Lease) -> None:
    try:
        with sqlite3.connect(lease.db_path, timeout=30) as conn:
            conn.execute(
                "DELETE FROM gpu_leases WHERE holder_id=? AND task_id=?",
                (lease.holder_id, lease.task_id),
            )
            conn.commit()
    except sqlite3.Error:
        # Cleanup status is verified and recorded by the caller.
        pass


def _lease_is_released(lease: _Lease) -> bool:
    try:
        with sqlite3.connect(lease.db_path, timeout=10) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM gpu_leases WHERE holder_id=? AND task_id=?",
                (lease.holder_id, lease.task_id),
            ).fetchone()
        return bool(row and int(row[0]) == 0)
    except sqlite3.Error:
        return False


def _json_request(url: str, *, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback URL
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"endpoint {url} returned non-object JSON")
    return parsed


def _wait_ready(base_url: str, *, timeout_s: float, proc: subprocess.Popen[Any] | None) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=3) as response:  # noqa: S310
                if int(response.status) == 200:
                    return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1.0)
    return False


def _verify_model(base_url: str, served_model_name: str) -> None:
    payload = _json_request(f"{base_url}/v1/models")
    ids = {str(row.get("id") or "") for row in payload.get("data", []) if isinstance(row, dict)}
    if served_model_name not in ids:
        raise RuntimeError(f"model identity mismatch: expected {served_model_name!r}, served={sorted(ids)!r}")


def _quality_smoke(base_url: str, served_model_name: str) -> dict[str, Any]:
    payload = _json_request(
        f"{base_url}/v1/completions",
        payload={
            "model": served_model_name,
            "prompt": "Reply with the word OK.",
            "max_tokens": 8,
            "temperature": 0,
        },
        timeout=60,
    )
    choices = payload.get("choices")
    text = ""
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        text = str(choices[0].get("text") or "").strip()
    if not text:
        raise RuntimeError("quality smoke returned an empty completion")
    return {"passed": True, "nonempty_completion": True, "sample_chars": len(text)}


def _terminate_group(proc: subprocess.Popen[Any] | None, *, grace_s: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    deadline = time.monotonic() + grace_s
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if proc.poll() is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def _latency_percentiles(raw: dict[str, Any], metric: str) -> dict[str, float]:
    return {
        "p50_ms": _float(raw.get(f"p50_{metric}_ms"), _float(raw.get(f"median_{metric}_ms"))),
        "p90_ms": _float(raw.get(f"p90_{metric}_ms")),
        "p99_ms": _float(raw.get(f"p99_{metric}_ms")),
    }


def normalize_vllm_result(
    raw: dict[str, Any],
    *,
    plan: LaunchPlan,
    quality_gate: dict[str, Any],
    cleanup_status: str,
    failure_reason: str = "",
) -> dict[str, Any]:
    """Normalize vLLM 0.27 bench JSON into the stable CUDA schema."""
    world_size = int(plan.topology["world_size"])
    total_output = _float(raw.get("output_throughput"))
    completed = _int(raw.get("completed"), _int(raw.get("completed_requests"), 0))
    failed = _int(raw.get("failed"), _int(raw.get("failed_requests"), 0))
    quality_passed = bool(quality_gate.get("passed"))
    cleanup_ok = cleanup_status in {"released", "deferred"}
    status = (
        "succeeded"
        if total_output > 0 and completed > 0 and failed == 0 and quality_passed and cleanup_ok and not failure_reason
        else "failed"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "target_id": plan.target_id,
        "model_id": plan.served_model_name,
        "system_fingerprint": plan.fingerprints.get("hardware"),
        "workload_fingerprint": plan.fingerprints.get("workload"),
        "config_fingerprint": plan.fingerprints.get("config"),
        "target": {
            "id": plan.target_id,
            "hardware_fingerprint": plan.fingerprints.get("hardware"),
        },
        "model": plan.model,
        "served_model_name": plan.served_model_name,
        "topology": plan.topology,
        "fingerprints": plan.fingerprints,
        "metrics": {
            "completed_requests": completed,
            "failed_requests": failed,
            "output_tokens_per_second_total": total_output,
            "output_tokens_per_second_per_gpu": total_output / world_size if world_size else 0.0,
            "requests_per_second": _float(raw.get("request_throughput")),
            "total_input_tokens": _int(raw.get("total_input_tokens")),
            "total_output_tokens": _int(raw.get("total_output_tokens")),
            "duration_seconds": _float(raw.get("duration")),
            "latency": {metric: _latency_percentiles(raw, metric) for metric in ("ttft", "tpot", "itl", "e2el")},
        },
        "quality_gate": quality_gate,
        "raw_artifacts": plan.artifacts,
        "cleanup_status": cleanup_status,
        "failure_reason": failure_reason or None,
    }


def _compatibility_report(
    unified: dict[str, Any],
    raw: dict[str, Any],
    *,
    workspace: Path,
    execution_time: float,
    errors: list[str],
) -> dict[str, Any]:
    metrics = unified.get("metrics") or {}
    lat = metrics.get("latency") or {}
    compatibility_raw = dict(raw)
    compatibility_raw["output_throughput"] = metrics.get("output_tokens_per_second_per_gpu", 0.0)
    compatibility_raw["completed"] = metrics.get("completed_requests", 0)
    for metric in ("ttft", "tpot", "itl", "e2el"):
        row = lat.get(metric) or {}
        compatibility_raw[f"median_{metric}_ms"] = row.get("p50_ms", 0.0)
        compatibility_raw[f"p90_{metric}_ms"] = row.get("p90_ms", 0.0)
        compatibility_raw[f"p99_{metric}_ms"] = row.get("p99_ms", 0.0)
    report = bypass_report.build_report(
        compatibility_raw,
        framework="vllm",
        model=str(unified.get("model") or ""),
        success=unified.get("status") == "succeeded",
        workspace_dir=str(workspace),
        execution_time=execution_time,
        errors=errors,
    )
    report["vllm_cuda"] = unified
    report["output_throughput_total"] = metrics.get("output_tokens_per_second_total", 0.0)
    report["output_throughput_per_gpu"] = metrics.get("output_tokens_per_second_per_gpu", 0.0)
    return report


def _find_raw_result(workspace: Path) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(workspace.glob("vllm_benchmark_raw*.json"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise RuntimeError("vLLM benchmark produced no result JSON")
    path = candidates[-1]
    raw = read_json(path, default=None, require_dict=True, strict=True)
    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid vLLM benchmark JSON at {path}")
    return path, raw


def run_benchmark(config_path: Path, output_dir: Path) -> int:
    """Run one YAML-configured vLLM CUDA benchmark."""
    target_id = os.environ.get("HYPERLOOM_TARGET", "").strip()
    target_runtime = os.environ.get("HYPERLOOM_TARGET_RUNTIME", "").strip().lower()
    if target_id != TARGET_ID or target_runtime != "cuda":
        raise ValueError(
            f"vllm_cuda runner requires HYPERLOOM_TARGET={TARGET_ID!r} and HYPERLOOM_TARGET_RUNTIME='cuda'"
        )
    hardware_payload = _hardware_payload()
    if hardware_payload.get("target_id") != target_id or not hardware_payload.get("sha256"):
        raise ValueError("vllm_cuda runner requires the validated NVIDIA hardware fingerprint")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    bench = cfg.get("benchmark") or {}
    envs = dict(bench.get("envs") or {})
    if str(bench.get("framework") or "").lower() != "vllm":
        raise ValueError("vllm_cuda runner requires benchmark.framework=vllm")
    model = str(bench.get("model") or os.environ.get("MODEL_PATH") or "").strip()
    if not model:
        raise ValueError("vllm_cuda runner requires a model")
    tp = max(1, _int(envs.get("TP"), 1))
    pp = max(1, _int(envs.get("PP"), 1))
    world_size = tp * pp
    gpu_ids = _visible_indices(envs, world_size)
    hardware_rows = _hardware_rows()
    missing_hardware = [gpu_id for gpu_id in gpu_ids if gpu_id not in hardware_rows]
    if missing_hardware:
        raise ValueError(f"hardware fingerprint has no identity row for CUDA devices {missing_hardware}")
    conc = max(1, _int(envs.get("CONC"), 2))
    isl = max(1, _int(envs.get("ISL"), 128))
    osl = max(1, _int(envs.get("OSL"), 32))
    num_prompts = max(1, _int(envs.get("NUM_PROMPTS"), max(4, conc * 2)))
    num_warmups = max(0, _int(envs.get("NUM_WARMUPS"), 1))
    max_model_len = max(isl + osl, _int(envs.get("MAX_MODEL_LEN"), isl + osl + 32))
    timeout_s = max(60, _int(bench.get("timeout_seconds"), 3600))
    lifecycle = bench.get("server_lifecycle") or {}
    lifecycle_enabled = bool(lifecycle.get("enabled"))
    cleanup_requested = bool(lifecycle.get("cleanup", True)) if lifecycle_enabled else True
    ready_timeout_s = max(30, _int(lifecycle.get("server_ready_timeout_s"), timeout_s))
    pid_dir = str(lifecycle.get("pid_dir") or "")
    port = _int(envs.get("PORT"), 0) or _pick_port()
    served_model_name = f"hyperloom-{_fingerprint({'model': model})[:12]}"
    base_url = f"http://127.0.0.1:{port}"
    workspace = _workspace(output_dir)
    raw_path = workspace / "vllm_benchmark_raw.json"
    unified_path = workspace / "vllm_cuda_benchmark.json"
    server_log_path = workspace / "server.log"
    client_stdout_path = workspace / "client_stdout.log"
    client_stderr_path = workspace / "client_stderr.log"

    child_env = build_benchmark_env(envs)
    child_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    child_env["VLLM_PLUGINS"] = ""
    child_env.pop("ROCR_VISIBLE_DEVICES", None)
    child_env.pop("HIP_VISIBLE_DEVICES", None)
    server_argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        str(tp),
        "--pipeline-parallel-size",
        str(pp),
        "--max-model-len",
        str(max_model_len),
        *_tokenize_extra_args(envs),
    ]
    benchmark_argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        served_model_name,
        "--tokenizer",
        model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(isl),
        "--random-output-len",
        str(osl),
        "--num-prompts",
        str(num_prompts),
        "--num-warmups",
        str(num_warmups),
        "--max-concurrency",
        str(conc),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--result-dir",
        str(workspace),
        "--result-filename",
        raw_path.name,
        "--disable-tqdm",
    ]
    fingerprints = {
        "hardware": str(hardware_payload["sha256"]),
        "model": _model_fingerprint(model),
        "workload": _fingerprint({"isl": isl, "osl": osl, "conc": conc, "prompts": num_prompts}),
        "config": _fingerprint(cfg),
    }
    plan = LaunchPlan(
        target_id=target_id,
        model=model,
        served_model_name=served_model_name,
        topology={
            "replicas": 1,
            "tp": tp,
            "pp": pp,
            "world_size": world_size,
            "gpu_indices": list(gpu_ids),
            "gpu_uuids": [str(hardware_rows[gpu_id].get("uuid") or "") for gpu_id in gpu_ids],
            "numa_nodes": [hardware_rows[gpu_id].get("numa_node") for gpu_id in gpu_ids],
        },
        port=port,
        base_url=base_url,
        server_argv=server_argv,
        benchmark_argv=benchmark_argv,
        child_env={
            key: child_env.get(key, "") for key in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PATH", "VLLM_PLUGINS")
        },
        artifacts={
            "workspace": str(workspace),
            "server_log": str(server_log_path),
            "raw_result": str(raw_path),
            "unified_result": str(unified_path),
            "compatibility_report": str(workspace / "benchmark_report.json"),
        },
        fingerprints=fingerprints,
    )
    _atomic_write_json(workspace / "launch_plan.json", asdict(plan))
    (workspace / "benchmark_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    stable_key = (
        _fingerprint({"pid_dir": pid_dir, "port": port})
        if lifecycle_enabled
        else _fingerprint({"workspace": str(workspace), "pid": os.getpid()})
    )
    lease = _acquire_gpu_lease(
        gpu_ids=gpu_ids,
        stable_key=stable_key,
        ttl_sec=ready_timeout_s + timeout_s + 600,
    )
    start = time.time()
    server_proc: subprocess.Popen[Any] | None = None
    server_log = None
    errors: list[str] = []
    raw: dict[str, Any] = {}
    quality_gate: dict[str, Any] = {"passed": False}
    cleanup_status = "pending"
    persistent = False
    try:
        reuse = lifecycle_enabled and pid_dir and bypass_engine.server_health_ok(base_url)
        if reuse:
            if not bypass_engine.lifecycle_files_present(pid_dir, "vllm", port):
                raise RuntimeError(f"port {port} is healthy but not owned by this Hyperloom lifecycle")
        else:
            server_log = server_log_path.open("a", encoding="utf-8")
            server_proc = subprocess.Popen(  # noqa: S603 - argv is materialized, no shell
                server_argv,
                env=child_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            if not _wait_ready(base_url, timeout_s=ready_timeout_s, proc=server_proc):
                tail = ""
                try:
                    tail = server_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                except OSError:
                    pass
                reason = "server_ready_timeout"
                if "out of memory" in tail.lower() or "cuda oom" in tail.lower():
                    reason = "server_cuda_oom"
                raise RuntimeError(reason)
        _verify_model(base_url, served_model_name)
        quality_gate = _quality_smoke(base_url, served_model_name)
        with (
            client_stdout_path.open("w", encoding="utf-8") as stdout,
            client_stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            try:
                client = subprocess.run(  # noqa: S603 - argv is materialized, no shell
                    benchmark_argv,
                    env=child_env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                    start_new_session=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("benchmark_timeout") from exc
        if client.returncode != 0:
            raise RuntimeError(f"benchmark_nonzero_rc_{client.returncode}")
        _, raw = _find_raw_result(workspace)
        if _float(raw.get("output_throughput")) <= 0 or _int(raw.get("completed"), 0) <= 0:
            raise RuntimeError("benchmark_zero_or_invalid_measurement")
        completed_requests = _int(raw.get("completed"), _int(raw.get("completed_requests"), 0))
        failed_requests = _int(raw.get("failed"), _int(raw.get("failed_requests"), 0))
        if failed_requests > 0 or completed_requests != num_prompts:
            raise RuntimeError(
                f"benchmark_request_failure_completed_{completed_requests}_expected_{num_prompts}_failed_{failed_requests}"
            )
        if lifecycle_enabled and not cleanup_requested and server_proc is not None:
            pgid = os.getpgid(server_proc.pid)
            bypass_engine.write_lifecycle_files(
                pid_dir=pid_dir,
                framework="vllm",
                port=port,
                pid=server_proc.pid,
                pgid=pgid,
                model=served_model_name,
                metadata={
                    "gpu_lease_holder": lease.holder_id,
                    "gpu_lease_task": lease.task_id,
                    "gpu_indices": list(lease.gpu_ids),
                    "gpu_uuids": list(lease.gpu_uuids),
                },
            )
            persistent = True
            cleanup_status = "deferred"
        else:
            if lifecycle_enabled and pid_dir:
                from ._server_lifecycle import teardown_lifecycle_server

                teardown_lifecycle_server(pid_dir=pid_dir, framework="vllm", port=port)
            else:
                _terminate_group(server_proc)
            _release_gpu_lease(lease)
            cleanup_status = "released" if _lease_is_released(lease) else "lease_release_failed"
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - normalized into artifacts
        errors.append(f"{type(exc).__name__}: {exc}")
        if lifecycle_enabled and pid_dir:
            from ._server_lifecycle import teardown_lifecycle_server

            teardown_lifecycle_server(pid_dir=pid_dir, framework="vllm", port=port)
        _terminate_group(server_proc)
        _release_gpu_lease(lease)
        cleanup_status = "released" if _lease_is_released(lease) else "lease_release_failed"
    finally:
        if server_log is not None:
            server_log.close()
        if not persistent and server_proc is not None:
            _terminate_group(server_proc)

    try:
        unified = normalize_vllm_result(
            raw,
            plan=plan,
            quality_gate=quality_gate,
            cleanup_status=cleanup_status,
            failure_reason="; ".join(errors),
        )
        _atomic_write_json(unified_path, unified)
        compatibility = _compatibility_report(
            unified,
            raw,
            workspace=workspace,
            execution_time=time.time() - start,
            errors=errors,
        )
        _atomic_write_json(workspace / "benchmark_report.json", compatibility)
    except BaseException:
        # A lifecycle run may intentionally leave the server alive after the
        # measurement. Artifact persistence is part of success, so a disk or
        # serialization failure must revoke that ownership before propagating.
        if persistent and lifecycle_enabled and pid_dir:
            from ._server_lifecycle import teardown_lifecycle_server

            teardown_lifecycle_server(pid_dir=pid_dir, framework="vllm", port=port)
        _terminate_group(server_proc)
        _release_gpu_lease(lease)
        raise
    try:
        (workspace / "summary.txt").write_text(bypass_report.format_summary_text(compatibility), encoding="utf-8")
    except OSError:
        pass
    return 0 if unified["status"] == "succeeded" else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm_cuda_runner")
    parser.add_argument("command", choices=("benchmark",))
    parser.add_argument("--benchmark-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-mode", choices=("local",), default="local")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return run_benchmark(args.benchmark_config, args.output_dir)
    except Exception as exc:  # noqa: BLE001 - pre-workspace/config failures
        print(f"vllm_cuda_runner: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LaunchPlan",
    "SCHEMA_VERSION",
    "SUPPORTED_EXTRA_VLLM_FLAGS",
    "main",
    "normalize_vllm_result",
    "run_benchmark",
]
