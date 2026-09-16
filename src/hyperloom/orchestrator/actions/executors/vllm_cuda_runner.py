# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native single-node CUDA benchmark runner for a capability-validated vLLM CLI.

The runner consumes the same materialized YAML and writes the same
``benchmark_report.json`` surface as Magpie while also emitting the richer
``vllm_cuda_benchmark/v1`` artifact.  It owns only process groups and GPU lease
rows it created; it never scans for or kills unrelated GPU processes.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.env_safety import build_benchmark_env
from hyperloom.common.jsonio import read_json
from hyperloom.orchestrator.bus.storage.schema import ensure_schema

from . import bypass_engine, bypass_report


SCHEMA_VERSION = "vllm_cuda_benchmark/v1"
TARGET_ID = "nvidia_cuda"
QUALITY_SUITE_SMOKE = "smoke"
QUALITY_SUITE_QWEN3_P3 = "qwen3_p3"
_QUALITY_SUITES = frozenset({QUALITY_SUITE_SMOKE, QUALITY_SUITE_QWEN3_P3})

# Config-only search surface. Unknown flags are rejected here; configured flags
# are additionally checked against the vLLM CLI capabilities captured during
# target preflight before they are forwarded to a newer vLLM release.
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
        "--enable-layerwise-nvtx-tracing",
        "--enable-prefix-caching",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "--kv-cache-dtype",
        "--kv-cache-memory-bytes",
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--numa-bind",
        "--no-numa-bind",
        "--numa-bind-cpus",
        "--numa-bind-nodes",
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


class _GpuMemorySampler:
    """Best-effort NVML memory sampler scoped to one runner-owned benchmark.

    Sampling is diagnostic, never a prerequisite for serving success: the
    target preflight already requires NVML, but an intermittent driver query
    must not discard a valid benchmark or prevent the runner's cleanup path.
    """

    def __init__(self, gpu_ids: tuple[int, ...], *, interval_sec: float = 0.25):
        self.gpu_ids = tuple(gpu_ids)
        self.interval_sec = max(0.05, float(interval_sec))
        self._samples: dict[int, list[float]] = {gpu_id: [] for gpu_id in self.gpu_ids}
        self._errors: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any = None
        self._handles: dict[int, Any] = {}

    def start(self) -> None:
        """Start polling after the runner has spawned or attached to the server."""
        if self._thread is not None or self._nvml is not None:
            return
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = {gpu_id: pynvml.nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in self.gpu_ids}
            self._sample_once()
            self._thread = threading.Thread(target=self._run, name="hyperloom-gpu-memory", daemon=True)
            self._thread.start()
        except Exception as exc:  # noqa: BLE001 - diagnostics are non-fatal
            self._errors.append(f"nvml_init:{type(exc).__name__}:{exc}")
            self._nvml = None
            self._handles = {}

    def _sample_once(self) -> None:
        if self._nvml is None:
            return
        try:
            rows = {
                gpu_id: float(self._nvml.nvmlDeviceGetMemoryInfo(handle).used) / (1024.0 * 1024.0)
                for gpu_id, handle in self._handles.items()
            }
            with self._lock:
                for gpu_id, used_mib in rows.items():
                    self._samples[gpu_id].append(round(used_mib, 3))
        except Exception as exc:  # noqa: BLE001 - one failed poll must not kill serving
            with self._lock:
                if len(self._errors) < 3:
                    self._errors.append(f"nvml_sample:{type(exc).__name__}:{exc}")

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._sample_once()

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return a compact, JSON-safe summary."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 4.0))
        self._sample_once()
        with self._lock:
            per_gpu = {
                str(gpu_id): {
                    "samples": len(values),
                    "min_used_mib": min(values) if values else 0.0,
                    "max_used_mib": max(values) if values else 0.0,
                    "last_used_mib": values[-1] if values else 0.0,
                }
                for gpu_id, values in self._samples.items()
            }
            return {
                "status": "collected" if any(values for values in self._samples.values()) else "unavailable",
                "interval_ms": int(self.interval_sec * 1000),
                "per_gpu": per_gpu,
                "errors": list(self._errors),
            }


def _cleanup_signal_handler(_signum: int, _frame: Any) -> None:
    """Turn SIGTERM into the same cleanup path as Ctrl-C."""
    raise KeyboardInterrupt


def _install_parent_death_cleanup() -> None:
    """Ensure an interrupted coordinator cannot orphan a CUDA server.

    The runner owns a separate vLLM process group.  Its Linux parent-death
    signal interrupts the runner when the executor disappears, and SIGTERM is
    normalized to ``KeyboardInterrupt`` so ``run_benchmark`` releases that
    process group and its GPU lease through its existing cleanup path.
    """
    signal.signal(signal.SIGTERM, _cleanup_signal_handler)
    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_PDEATHSIG is Linux's per-process parent-death notification.
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _runtime_python(bench: dict[str, Any]) -> str:
    """Return the explicitly selected vLLM interpreter for this benchmark."""
    runtime = bench.get("runtime") or {}
    configured = str(runtime.get("python") or "").strip() if isinstance(runtime, dict) else ""
    if not configured:
        return sys.executable
    path = Path(configured).expanduser()
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("benchmark.runtime.python must be an executable absolute path")
    return str(path.resolve())


def _runtime_pythonpath(bench: dict[str, Any]) -> str:
    """Return an optional absolute source root prepended for vLLM children."""
    runtime = bench.get("runtime") or {}
    configured = str(runtime.get("pythonpath") or "").strip() if isinstance(runtime, dict) else ""
    if not configured:
        return ""
    path = Path(configured).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("benchmark.runtime.pythonpath must be an absolute directory")
    return str(path.resolve())


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
    cli_capabilities = _hardware_payload().get("vllm_cli")
    available = cli_capabilities.get("server_flags") if isinstance(cli_capabilities, dict) else None
    if isinstance(available, list):
        available_flags = {str(flag) for flag in available}
        missing = sorted({token.split("=", 1)[0] for token in tokens if token.startswith("--")} - available_flags)
        if missing:
            raise ValueError(
                "EXTRA_VLLM_ARGS contains flag(s) not supported by the installed vLLM CLI: " + ", ".join(missing)
            )
    return tokens


def _visible_indices(envs: dict[str, Any], world_size: int) -> tuple[int, ...]:
    from hyperloom.inference_optimizer.target_registry import allocate_cuda_devices, TargetValidationError

    try:
        selected = allocate_cuda_devices(
            _hardware_payload(),
            world_size,
            str(envs["CUDA_VISIBLE_DEVICES"]) if "CUDA_VISIBLE_DEVICES" in envs else None,
        )
    except TargetValidationError as exc:
        raise ValueError(str(exc)) from exc
    return tuple(r["index"] for r in selected)


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


def _completion_text(payload: dict[str, Any]) -> str:
    """Extract one non-empty text completion from the OpenAI completions shape."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return str(choices[0].get("text") or "").strip()
    return ""


def _qwen3_p3_cases() -> tuple[dict[str, Any], ...]:
    """Return the small, deterministic semantic matrix required by NVIDIA P3."""
    cn_context = "\n".join("背景资料：植物利用光能把二氧化碳和水转化为有机物，并释放氧气。" for _ in range(12))
    en_context = "\n".join(
        "Background: photosynthesis uses light energy to convert water and carbon dioxide into sugars and oxygen."
        for _ in range(12)
    )
    return (
        {
            "id": "cn_short_thinking",
            "language": "zh",
            "length": "short",
            "enable_thinking": True,
            "max_tokens": 512,
            "semantic_groups": (
                ("光合作用",),
                ("二氧化碳",),
                ("水",),
                ("氧气",),
                ("糖", "有机物", "葡萄糖"),
            ),
            "content": ("请用一句中文说明光合作用：必须明确写出它使用二氧化碳和水，生成有机物和氧气。"),
        },
        {
            "id": "en_short_no_thinking",
            "language": "en",
            "length": "short",
            "enable_thinking": False,
            "max_tokens": 64,
            "semantic_groups": (
                ("photosynthesis",),
                ("carbon dioxide",),
                ("water",),
                ("oxygen",),
                ("sugar", "glucose"),
            ),
            "content": (
                "Answer in one English sentence what photosynthesis does. "
                "Explicitly include carbon dioxide, water, sugar, and oxygen."
            ),
        },
        {
            "id": "cn_long_no_thinking",
            "language": "zh",
            "length": "long",
            "enable_thinking": False,
            "max_tokens": 64,
            "semantic_groups": (
                ("二氧化碳",),
                ("水",),
                ("氧气",),
                ("糖", "有机物", "葡萄糖"),
            ),
            "content": f"{cn_context}\n\n只用一句中文总结上述资料，且明确写出二氧化碳、水、有机物和氧气。",
        },
        {
            "id": "en_long_thinking",
            "language": "en",
            "length": "long",
            "enable_thinking": True,
            "max_tokens": 512,
            "semantic_groups": (
                ("carbon dioxide",),
                ("water",),
                ("oxygen",),
                ("sugar", "glucose"),
            ),
            "content": (
                f"{en_context}\n\nGive a one-sentence English summary of the background. "
                "Explicitly include carbon dioxide, water, sugar, and oxygen."
            ),
        },
    )


def _qwen3_p3_semantic_checks(spec: dict[str, Any], completion: str) -> dict[str, Any]:
    """Validate P3's factual answer contract without treating a nonempty reply as correct."""
    normalized = " ".join(completion.casefold().split())
    final_answer = normalized
    answer_source = "completion"
    if "<think>" in normalized:
        if "</think>" not in normalized:
            # Qwen3 can spend the entire bounded completion budget in a
            # well-formed reasoning prefix. It is still a semantic response;
            # validate its factual content and mark that provenance explicitly.
            final_answer = normalized.split("<think>", 1)[1].strip()
            answer_source = "thinking_prefix"
        else:
            final_answer = normalized.split("</think>", 1)[1].strip()
            answer_source = "final_answer"
    if not final_answer:
        return {"passed": False, "reason": "empty_final_answer", "final_answer": "", "matched_terms": []}

    groups = tuple(tuple(str(term).casefold() for term in group) for group in spec["semantic_groups"])
    matched_terms: list[str] = []
    missing_groups: list[list[str]] = []
    for group in groups:
        matched = next((term for term in group if term in final_answer), None)
        if matched is None:
            missing_groups.append(list(group))
        else:
            matched_terms.append(matched)
    return {
        "passed": not missing_groups,
        "reason": "" if not missing_groups else "missing_required_semantic_terms",
        "final_answer": final_answer,
        "answer_source": answer_source,
        "matched_terms": matched_terms,
        "missing_groups": missing_groups,
        "required_groups": [list(group) for group in groups],
    }


def _qwen3_p3_quality_gate(
    base_url: str,
    served_model_name: str,
    model: str,
    *,
    artifact_path: Path,
) -> dict[str, Any]:
    """Run and persist P3's Qwen3 semantic prompt matrix.

    The tokenizer renders each prompt with its explicit thinking mode, then the
    runner uses the stable `/v1/completions` endpoint. This keeps the API-side
    contract independent of optional chat-completions payload keys that can
    drift between vLLM releases.
    """
    config_path = Path(model).expanduser() / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"qwen3_p3 quality suite cannot read model config: {exc}") from exc
    if str(config.get("model_type") or "").lower() != "qwen3":
        raise RuntimeError("qwen3_p3 quality suite requires a Qwen3 checkpoint")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - surface the required local tokenizer evidence
        raise RuntimeError(f"qwen3_p3 quality suite cannot load local tokenizer: {exc}") from exc

    artifacts: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in _qwen3_p3_cases():
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": spec["content"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(spec["enable_thinking"]),
            )
            input_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        except Exception as exc:  # noqa: BLE001 - template behavior is the subject under test
            raise RuntimeError(f"qwen3_p3 could not render {spec['id']}: {exc}") from exc
        if spec["length"] == "long" and input_tokens < 128:
            raise RuntimeError(f"qwen3_p3 long case {spec['id']} rendered only {input_tokens} tokens")
        payload = _json_request(
            f"{base_url}/v1/completions",
            payload={
                "model": served_model_name,
                "prompt": prompt,
                "max_tokens": int(spec["max_tokens"]),
                "temperature": 0,
            },
            timeout=90,
        )
        completion = _completion_text(payload)
        if not completion:
            raise RuntimeError(f"qwen3_p3 case {spec['id']} returned an empty completion")
        semantic_checks = _qwen3_p3_semantic_checks(spec, completion)
        if not semantic_checks["passed"]:
            failed_metadata = {
                "id": spec["id"],
                "language": spec["language"],
                "length": spec["length"],
                "enable_thinking": bool(spec["enable_thinking"]),
                "input_tokens": input_tokens,
                "response_chars": len(completion),
                "semantic_checks": semantic_checks,
                "passed": False,
            }
            artifacts.append(
                {
                    **failed_metadata,
                    "prompt": prompt,
                    "completion": completion,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
                }
            )
            _atomic_write_json(
                artifact_path,
                {
                    "schema_version": "vllm_cuda_quality/v1",
                    "suite": QUALITY_SUITE_QWEN3_P3,
                    "model": model,
                    "cases": artifacts,
                    "passed": False,
                },
            )
            raise RuntimeError(
                f"qwen3_p3 case {spec['id']} failed semantic checks: {semantic_checks['reason']} "
                f"{semantic_checks.get('missing_groups', [])}"
            )
        metadata = {
            "id": spec["id"],
            "language": spec["language"],
            "length": spec["length"],
            "enable_thinking": bool(spec["enable_thinking"]),
            "input_tokens": input_tokens,
            "response_chars": len(completion),
            "semantic_checks": semantic_checks,
            "passed": True,
        }
        summaries.append(metadata)
        artifacts.append(
            {
                **metadata,
                "prompt": prompt,
                "completion": completion,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
            }
        )
    _atomic_write_json(
        artifact_path,
        {
            "schema_version": "vllm_cuda_quality/v1",
            "suite": QUALITY_SUITE_QWEN3_P3,
            "model": model,
            "cases": artifacts,
        },
    )
    return {
        "passed": True,
        "semantic_suite": QUALITY_SUITE_QWEN3_P3,
        "semantic_case_count": len(summaries),
        "semantic_cases": summaries,
    }


def _quality_smoke(
    base_url: str,
    served_model_name: str,
    *,
    model: str,
    quality_suite: str,
    artifact_path: Path,
) -> dict[str, Any]:
    """Run the baseline smoke request and an optional persisted semantic suite."""
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
    text = _completion_text(payload)
    if not text:
        raise RuntimeError("quality smoke returned an empty completion")
    smoke_artifact = {
        "id": "smoke",
        "prompt": "Reply with the word OK.",
        "completion": text,
        "response_chars": len(text),
        "passed": True,
    }
    if quality_suite == QUALITY_SUITE_SMOKE:
        _atomic_write_json(
            artifact_path,
            {"schema_version": "vllm_cuda_quality/v1", "suite": quality_suite, "cases": [smoke_artifact]},
        )
        return {"passed": True, "nonempty_completion": True, "sample_chars": len(text)}
    if quality_suite == QUALITY_SUITE_QWEN3_P3:
        semantic = _qwen3_p3_quality_gate(
            base_url,
            served_model_name,
            model,
            artifact_path=artifact_path,
        )
        persisted = read_json(artifact_path, default={}, require_dict=True, strict=True)
        cases = persisted.get("cases") if isinstance(persisted, dict) else None
        if isinstance(cases, list):
            persisted["cases"] = [smoke_artifact, *cases]
            _atomic_write_json(artifact_path, persisted)
        return {"nonempty_completion": True, "sample_chars": len(text), **semantic}
    raise ValueError(f"unsupported vLLM CUDA quality suite {quality_suite!r}")


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
    gpu_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a capability-validated vLLM bench JSON into the stable CUDA schema."""
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
            "gpu_memory": dict(gpu_memory or {"status": "unavailable", "per_gpu": {}, "errors": []}),
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
    if unified.get("measurement_kind") == "profile":
        report["measurement_kind"] = "profile"
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
    from .cuda_host_lock import CudaHostLock

    with CudaHostLock(f"benchmark:{output_dir}"):
        return _run_benchmark(config_path, output_dir)


def _profile_endpoint_status(server_log_path: Path) -> tuple[bool, bool, bool]:
    """Read server acknowledgements; bench mislabels empty HTTP 200 responses.

    A stop request is not idempotent in torch/vLLM. Never retry a stop that
    reached the server, including one still exporting the trace.
    """
    try:
        log_text = server_log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, False, False
    started = bool(re.search(r'POST /start_profile HTTP/[^"\s]+" 200(?: |$)', log_text, re.MULTILINE))
    stopped = bool(re.search(r'POST /stop_profile HTTP/[^"\s]+" 200(?: |$)', log_text, re.MULTILINE))
    stop_attempted = "Stopping profiler..." in log_text or "POST /stop_profile " in log_text
    return started, stopped, stop_attempted


def _run_benchmark(config_path: Path, output_dir: Path) -> int:
    """Run one YAML-configured vLLM CUDA benchmark."""
    target_id = os.environ.get("HYPERLOOM_TARGET", "").strip()
    target_runtime = os.environ.get("HYPERLOOM_TARGET_RUNTIME", "").strip().lower()
    from hyperloom.inference_optimizer.target_registry import get_target, is_cuda_target

    if not is_cuda_target(target_id) or target_runtime != "cuda":
        raise ValueError(
            f"vllm_cuda runner requires HYPERLOOM_TARGET={TARGET_ID!r} and HYPERLOOM_TARGET_RUNTIME='cuda'"
        )
    target_id = get_target(target_id).target_id
    hardware_payload = _hardware_payload()
    if get_target(hardware_payload.get("target_id", "")).target_id != target_id or not hardware_payload.get("sha256"):
        raise ValueError("vllm_cuda runner requires the validated NVIDIA hardware fingerprint")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    bench = cfg.get("benchmark") or {}
    runtime_python = _runtime_python(bench)
    runtime_pythonpath = _runtime_pythonpath(bench)
    if runtime_python != hardware_payload.get(
        "runtime_python", sys.executable
    ) or runtime_pythonpath != hardware_payload.get("runtime_pythonpath", ""):
        from .vllm_cuda_preflight import validate_execution_stack

        checked = validate_execution_stack(hardware_payload, python_exe=runtime_python, pythonpath=runtime_pythonpath)
        from hyperloom.inference_optimizer.target_registry import validate_resume_environment

        validate_resume_environment(hardware_payload, checked)
        # Record the environment that will actually execute this task.
        hardware_payload = checked
        os.environ["HYPERLOOM_HARDWARE_FINGERPRINT"] = json.dumps(checked, sort_keys=True)
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
    cuda_profiler = (bench.get("profiler") or {}).get("cuda_profiler") or {}
    profile_backend = str(cuda_profiler.get("backend") or "torch")
    external_profile = profile_backend in ("nsys", "ncu")
    profile_enabled = external_profile or bool(
        ((bench.get("profiler") or {}).get("torch_profiler") or {}).get("enabled")
    )
    if profile_enabled and num_prompts > 16:
        raise ValueError("CUDA profiling is limited to 16 requests per trace window")
    lifecycle = bench.get("server_lifecycle") or {}
    if profile_enabled:
        lifecycle = {}  # Profiling always owns a fresh server and releases it.

    lifecycle_enabled = bool(lifecycle.get("enabled"))
    cleanup_requested = bool(lifecycle.get("cleanup", True)) if lifecycle_enabled else True
    ready_timeout_s = max(30, _int(lifecycle.get("server_ready_timeout_s"), timeout_s))
    pid_dir = str(lifecycle.get("pid_dir") or "")
    if lifecycle_enabled and not pid_dir:
        raise ValueError("server_lifecycle.enabled requires pid_dir")
    port = _int(envs.get("PORT"), 0) or _pick_port()
    served_model_name = f"hyperloom-{_fingerprint({'model': model})[:12]}"
    base_url = f"http://127.0.0.1:{port}"
    workspace = _workspace(output_dir)
    raw_path = workspace / "vllm_benchmark_raw.json"
    unified_path = workspace / "vllm_cuda_benchmark.json"
    quality_cases_path = workspace / "quality_cases.json"
    server_log_path = workspace / "server.log"
    client_stdout_path = workspace / "client_stdout.log"
    client_stderr_path = workspace / "client_stderr.log"

    child_env = build_benchmark_env(envs)
    if runtime_pythonpath:
        inherited_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = (
            f"{runtime_pythonpath}{os.pathsep}{inherited_pythonpath}" if inherited_pythonpath else runtime_pythonpath
        )
    child_env["CUDA_VISIBLE_DEVICES"] = ",".join(hardware_rows[gpu_id]["uuid"] for gpu_id in gpu_ids)
    child_env["VLLM_PLUGINS"] = ""
    child_env.pop("ROCR_VISIBLE_DEVICES", None)
    child_env.pop("HIP_VISIBLE_DEVICES", None)
    quality_suite = os.environ.get("INFERENCE_OPTIMIZER_QUALITY_SUITE", QUALITY_SUITE_SMOKE).strip()
    if quality_suite not in _QUALITY_SUITES:
        raise ValueError(f"unsupported vLLM CUDA quality suite {quality_suite!r}")
    server_argv = [
        runtime_python,
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
        runtime_python,
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
    if profile_enabled:
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": str((workspace / "torch_trace").resolve()),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_dump_cuda_time_total": False,
        }
        if external_profile:
            profiler_config = {"profiler": "cuda", "detailed_trace_annotation": True}
            if profile_backend == "ncu":
                profiler_config["max_iterations"] = int(cuda_profiler.get("max_iterations", 0))
                profiler_config["delay_iterations"] = int(cuda_profiler.get("delay_iterations", 0))
        server_argv.extend(["--profiler-config", json.dumps(profiler_config)])
        benchmark_argv.append("--profile")
    serving_args = []
    skip_value = False
    for value in server_argv:
        if skip_value:
            skip_value = False
            continue
        if value in ("--port", "--served-model-name", "--profiler-config"):
            skip_value = True
            continue
        serving_args.append(value)
    fingerprints = {
        "serving_config": _fingerprint({"argv": serving_args, "envs": {k: v for k, v in envs.items() if k != "PORT"}}),
        "hardware": str(hardware_payload["sha256"]),
        "model": _model_fingerprint(model),
        "workload": _fingerprint({"isl": isl, "osl": osl, "conc": conc, "prompts": num_prompts}),
        "config": _fingerprint(cfg),
        "quality_suite": _fingerprint(quality_suite),
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
            "cuda_indices": [hardware_rows[gpu_id]["cuda_index"] for gpu_id in gpu_ids],
            "logical_indices": list(range(world_size)),
            "gpu_uuids": [str(hardware_rows[gpu_id].get("uuid") or "") for gpu_id in gpu_ids],
            "numa_nodes": [hardware_rows[gpu_id].get("numa_node") for gpu_id in gpu_ids],
        },
        port=port,
        base_url=base_url,
        server_argv=server_argv,
        benchmark_argv=benchmark_argv,
        child_env={
            key: child_env.get(key, "")
            for key in ("CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PATH", "PYTHONPATH", "VLLM_PLUGINS")
        },
        artifacts={
            "workspace": str(workspace),
            "server_log": str(server_log_path),
            "raw_result": str(raw_path),
            "unified_result": str(unified_path),
            "quality_cases": str(quality_cases_path),
            "compatibility_report": str(workspace / "benchmark_report.json"),
        },
        fingerprints=fingerprints,
    )
    ncu_worker_filter = None
    if external_profile:
        from .cuda_nsight import roofline_metrics, profiler_argv, tool_fingerprint, query_roofline_metrics

        fingerprints["profiler_tools"] = tool_fingerprint(
            roofline=profile_backend == "ncu", expected=cuda_profiler.get("tools")
        )
        ncu_devices = cuda_profiler.get("devices")
        if profile_backend == "ncu" and ncu_devices is None:
            # NCU's default all-device scope can stall graph replay even when
            # CUDA_VISIBLE_DEVICES restricts the serving process to one GPU.
            ncu_devices = list(range(world_size))
        if any(d not in range(world_size) for d in ncu_devices or []):
            raise ValueError("ncu profiling device is outside the leased CUDA_VISIBLE_DEVICES")
        worker_rank = cuda_profiler.get("worker_rank")
        if profile_backend == "ncu" and worker_rank is not None:
            from .cuda_profiler_launch import prepare_worker_filter

            if worker_rank not in range(world_size) or len(cuda_profiler.get("devices") or []) != 1:
                raise ValueError("ncu worker filtering requires one leased device and a valid worker rank")
            ncu_worker_filter = prepare_worker_filter(workspace, worker_rank)
        metric_selection = None
        if profile_backend == "ncu":
            names = cuda_profiler.get("kernel_names") or [str(cuda_profiler.get("kernel_name") or "")]
            metric_selection = query_roofline_metrics(
                names,
                [plan.topology["gpu_uuids"][d] for d in ncu_devices],
                executable=fingerprints["profiler_tools"]["ncu"]["path"],
            )
            _atomic_write_json(workspace / "ncu_metric_selection.json", metric_selection)
            if "gpu__time_duration.sum" not in metric_selection["selected"]:
                raise ValueError("Nsight Compute duration counters unavailable for the sampled GPU")
        # The launcher publishes the actual serving PID/group before exec.
        server_argv = profiler_argv(
            profile_backend,
            [
                sys.executable,
                "-m",
                "hyperloom.orchestrator.actions.executors.cuda_profiler_launch",
                str(workspace / "serving_ownership.json"),
                *server_argv,
            ],
            workspace,
            str(cuda_profiler.get("kernel_name") or ""),
            tool_paths=fingerprints["profiler_tools"],
            devices=ncu_devices,
            process_name=ncu_worker_filter["process_name"] if ncu_worker_filter else None,
            kernel_names=cuda_profiler.get("kernel_names"),
            metrics=metric_selection["selected"] if metric_selection is not None else None,
        )
        plan = replace(plan, server_argv=server_argv)
        if profile_backend == "ncu":
            _atomic_write_json(
                workspace / "ncu_capture_policy.json",
                {
                    "schema_version": 1,
                    "window": "native_benchmark_start_stop",
                    "delay_iterations": profiler_config["delay_iterations"],
                    "max_iterations": profiler_config["max_iterations"],
                    "launch_limit_per_kernel_per_gpu": 1 if ncu_worker_filter else 3,
                    "devices": ncu_devices,
                    "worker_filter": ncu_worker_filter,
                    "kernel_name": cuda_profiler.get("kernel_name"),
                    "kernel_names": cuda_profiler.get("kernel_names") or [str(cuda_profiler.get("kernel_name"))],
                    "metrics": list(
                        roofline_metrics(cuda_profiler.get("kernel_names") or [str(cuda_profiler.get("kernel_name"))])
                    ),
                    "argv": server_argv,
                    "tools": fingerprints["profiler_tools"],
                },
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
    ownership_metadata = {
        "gpu_lease_holder": lease.holder_id,
        "gpu_lease_task": lease.task_id,
        "gpu_indices": list(lease.gpu_ids),
        "gpu_uuids": list(lease.gpu_uuids),
    }
    if external_profile:
        _atomic_write_json(
            workspace / "serving_ownership.json",
            {
                "pid_dir": str(workspace / "serving"),
                "port": port,
                "model": served_model_name,
                "metadata": ownership_metadata,
                "ncu_worker_filter": ncu_worker_filter,
            },
        )
    start = time.time()
    server_proc: subprocess.Popen[Any] | None = None
    server_log = None
    memory_sampler = _GpuMemorySampler(gpu_ids)
    errors: list[str] = []
    raw: dict[str, Any] = {}
    quality_gate: dict[str, Any] = {"passed": False}
    cleanup_status = "pending"
    persistent = False
    trace_health: dict[str, Any] = {"passed": False, "trace_files": [], "ranks": [], "errors": []}
    profile_stopped = False
    analysis_result: dict[str, Any] = {}
    workers: list[dict[str, Any]] = []
    ncu_progress = None

    def cleanup_profile_processes() -> None:
        if profile_enabled:
            from ._server_lifecycle import teardown_lifecycle_server

            for owner in ("serving", "profiler"):
                teardown_lifecycle_server(pid_dir=workspace / owner, framework="vllm", port=port)
            from .cuda_profiler_launch import cleanup_profile_workers

            cleanup_profile_workers(workspace)

    try:
        reuse = lifecycle_enabled and pid_dir and bypass_engine.server_health_ok(base_url)
        if reuse:
            if not bypass_engine.lifecycle_files_present(pid_dir, "vllm", port):
                raise RuntimeError(f"port {port} is healthy but not owned by this Hyperloom lifecycle")
            memory_sampler.start()
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
            if profile_enabled:
                bypass_engine.write_lifecycle_files(
                    pid_dir=workspace / ("profiler" if external_profile else "serving"),
                    framework="vllm",
                    port=port,
                    pid=server_proc.pid,
                    pgid=os.getpgid(server_proc.pid),
                    model=served_model_name,
                    metadata=ownership_metadata,
                )
            if lifecycle_enabled:
                # The parent can kill this runner while vLLM is still booting.
                # Publish ownership now so outer teardown can release the lease
                # even if this process never reaches its Python cleanup handler.
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
            memory_sampler.start()
            if not _wait_ready(base_url, timeout_s=ready_timeout_s, proc=server_proc):
                tail = ""
                try:
                    tail = server_log_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
                reason = "server_exited_before_ready" if server_proc.poll() is not None else "server_ready_timeout"
                if any(
                    text in tail.lower()
                    for text in ("out of memory", "cuda oom", "less than desired gpu memory utilization")
                ):
                    reason = "server_cuda_oom"
                raise RuntimeError(reason)
        if external_profile:
            from .cuda_nsight import rank_processes

            serving_pid = int((workspace / "serving" / f"vllm_{port}.pid").read_text().split()[0])
            workers = rank_processes(plan.topology, serving_pid, server_log_path)
            _atomic_write_json(workspace / "rank_processes.json", {"workers": workers})
            if ncu_worker_filter:
                selected_worker = next(w for w in workers if w["rank"] == ncu_worker_filter["rank"])
                if Path(f"/proc/{selected_worker['pid']}/exe").resolve() != Path(ncu_worker_filter["executable"]):
                    raise RuntimeError("ncu_worker_process_filter_not_applied")
        _verify_model(base_url, served_model_name)
        quality_gate = _quality_smoke(
            base_url,
            served_model_name,
            model=model,
            quality_suite=quality_suite,
            artifact_path=quality_cases_path,
        )
        if profile_backend == "ncu":
            from .cuda_nsight import NcuProgress

            ncu_progress = NcuProgress(workspace)
            ncu_progress.start()
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
        if profile_enabled:
            # vLLM bench warms up first and brackets measured requests with the
            # native start/stop endpoints. It does not fail on endpoint errors.
            profile_started, profile_stopped, _ = _profile_endpoint_status(server_log_path)
            if not profile_started or not profile_stopped:
                raise RuntimeError("profiler_start_or_stop_failed")
            if external_profile:
                from ._server_lifecycle import teardown_lifecycle_server
                from .cuda_nsight import analyze_nsys, analyze_ncu, write_analysis, ncu_export_argv

                # Let Nsight finish exporting after the application exits. Never
                # terminate the wrapper's group before its report is complete.
                teardown_lifecycle_server(pid_dir=workspace / "serving", framework="vllm", port=port)
                server_proc.wait(timeout=180)
                from .cuda_profiler_launch import cleanup_profile_workers

                cleanup_profile_workers(workspace)
                report = workspace / ("capture.nsys-rep" if profile_backend == "nsys" else "capture.ncu-rep")
                if not report.is_file() or report.stat().st_mtime < start or report.stat().st_size == 0:
                    raise RuntimeError("missing_or_stale_nsight_report")
                with (workspace / "export.log").open("w") as export_log:
                    if profile_backend == "nsys":
                        database = workspace / "capture.sqlite"
                        subprocess.run(
                            [
                                fingerprints["profiler_tools"]["nsys"]["path"],
                                "export",
                                "--type",
                                "sqlite",
                                "--output",
                                str(database),
                                str(report),
                            ],
                            stdout=export_log,
                            stderr=subprocess.STDOUT,
                            timeout=120,
                            check=True,
                        )
                        summary = analyze_nsys(database, workers, fingerprints=fingerprints)
                        analysis_result = write_analysis(workspace, summary)
                        trace_health = {
                            "passed": True,
                            "ranks": summary["ranks"],
                            "trace_files": [str(report), str(database)],
                            "errors": [],
                        }
                    else:
                        csv_path = workspace / "capture.csv"
                        with csv_path.open("w") as csv_stream:
                            subprocess.run(
                                ncu_export_argv(report, executable=fingerprints["profiler_tools"]["ncu"]["path"]),
                                stdout=csv_stream,
                                stderr=export_log,
                                timeout=120,
                                check=True,
                            )
                        with (workspace / "capture_nvtx.csv").open("w") as nvtx_stream:
                            subprocess.run(
                                ncu_export_argv(
                                    report, nvtx=True, executable=fingerprints["profiler_tools"]["ncu"]["path"]
                                ),
                                stdout=nvtx_stream,
                                stderr=export_log,
                                timeout=120,
                                check=True,
                            )
                        trace_health = analyze_ncu(
                            csv_path,
                            workers,
                            str(cuda_profiler.get("kernel_name") or ""),
                            kernel_names=cuda_profiler.get("kernel_names"),
                        )
                        trace_health["trace_files"].insert(0, str(report))
            else:
                from .cuda_profile import validate_cuda_traces

                trace_health = validate_cuda_traces(workspace / "torch_trace", world_size=world_size, started_at=start)
            if not trace_health["passed"]:
                raise RuntimeError("invalid_cuda_trace: " + "; ".join(trace_health["errors"]))
        if lifecycle_enabled and not cleanup_requested and server_proc is not None:
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
        # A profiler flush/server teardown may exceed the supervisor's TERM
        # grace. Release the database row first; the host lock still serializes
        # GPU launches until this runner is reaped.
        _release_gpu_lease(lease)
        profile_started, profile_stopped, stop_attempted = (
            _profile_endpoint_status(server_log_path)
            if profile_enabled and server_log_path.exists()
            else (False, False, False)
        )
        if profile_started and not stop_attempted and server_proc is not None and server_proc.poll() is None:
            # Even a timed-out/interrupted client must release profiler state
            # before the same process-group and lease cleanup as any benchmark.
            import urllib.request

            try:
                request = urllib.request.Request(base_url + "/stop_profile", data=b"", method="POST")
                with urllib.request.urlopen(request, timeout=60):
                    pass
            except Exception as stop_exc:  # noqa: BLE001 - cleanup must continue
                errors.append(f"profiler_cleanup: {stop_exc}")
        if lifecycle_enabled and pid_dir:
            from ._server_lifecycle import teardown_lifecycle_server

            teardown_lifecycle_server(pid_dir=pid_dir, framework="vllm", port=port)
        _terminate_group(server_proc)
        _release_gpu_lease(lease)
        cleanup_status = "released" if _lease_is_released(lease) else "lease_release_failed"
    finally:
        cleanup_profile_processes()
        if server_log is not None:
            server_log.close()
        if ncu_progress is not None:
            ncu_progress.stop()
        if not persistent:
            _terminate_group(server_proc)
            _release_gpu_lease(lease)

    gpu_memory = memory_sampler.stop()

    try:
        unified = normalize_vllm_result(
            raw,
            plan=plan,
            quality_gate=quality_gate,
            cleanup_status=cleanup_status,
            failure_reason="; ".join(errors),
            gpu_memory=gpu_memory,
        )
        if profile_enabled:
            from .cuda_nsight import capture_failure

            unified["measurement_kind"] = "profile"
            _atomic_write_json(
                workspace / "vllm_cuda_profile.json",
                {
                    "schema_version": 1,
                    "status": unified["status"],
                    "measurement_kind": "profile",
                    "backend": profile_backend,
                    "analysis_result": analysis_result,
                    "trace_health": trace_health,
                    "trace_files": trace_health["trace_files"],
                    "fingerprints": fingerprints,
                    "topology": plan.topology,
                    "cleanup_status": cleanup_status,
                    "errors": errors,
                    "capture_failure": capture_failure(errors, server_log_path.read_text(errors="replace"))
                    if profile_backend == "ncu" and server_log_path.exists()
                    else None,
                    "capture_policy_path": str(workspace / "ncu_capture_policy.json")
                    if profile_backend == "ncu"
                    else None,
                    "progress_path": str(workspace / "ncu_progress.jsonl") if profile_backend == "ncu" else None,
                    "diagnostic_metrics": unified,
                },
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
        _install_parent_death_cleanup()
        return run_benchmark(args.benchmark_config, args.output_dir)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - pre-workspace/config failures
        print(f"vllm_cuda_runner: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LaunchPlan",
    "SCHEMA_VERSION",
    "SUPPORTED_EXTRA_VLLM_FLAGS",
    "_cleanup_signal_handler",
    "_install_parent_death_cleanup",
    "main",
    "normalize_vllm_result",
    "run_benchmark",
]
