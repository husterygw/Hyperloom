# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native vLLM torch profiling, with explicit per-rank CUDA trace validation."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .baseline import BaselineExecutor


def validate_cuda_traces(trace_dir: Path, *, world_size: int, started_at: float) -> dict[str, Any]:
    """Require a fresh, parseable trace with real CUDA kernels on every rank."""
    ranks: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for path in sorted(trace_dir.glob("*.pt.trace.json*")):
        match = re.search(r"(?:^|_)rank(\d+)\.", path.name)
        if match is None:
            continue  # Ignore frontend traces; they do not prove GPU coverage.
        rank = int(match.group(1))
        try:
            if path.stat().st_mtime < started_at:
                raise ValueError("stale trace")
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as stream:
                trace = json.load(stream)
            events = trace.get("traceEvents", [])
            kernels = [
                e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X" and float(e.get("dur", 0)) > 0
            ]
            if not kernels:
                raise ValueError("no CUDA kernel events")
            if rank in ranks:
                raise ValueError("duplicate rank trace")
            ranks[rank] = {
                "rank": rank,
                "path": str(path.resolve()),
                "kernel_events": len(kernels),
                "kernel_time_us": sum(float(e["dur"]) for e in kernels),
            }
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append(f"{path.name}: {exc}")
    expected = set(range(world_size))
    if set(ranks) != expected:
        errors.append(f"rank coverage: expected {sorted(expected)}, collected {sorted(ranks)}")
    rows = [ranks[rank] for rank in sorted(ranks)]
    return {"passed": not errors, "ranks": rows, "errors": errors, "trace_files": [row["path"] for row in rows]}


class CudaProfileExecutor(BaselineExecutor):
    """Reuse supervised benchmark launch/cleanup without AMD profiler hooks."""

    def _after_materialize_config(self, config_path: Path, output_dir: Path) -> None:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        bench = cfg["benchmark"]
        bench["profiler"] = {"torch_profiler": {"enabled": True}}
        bench["server_lifecycle"] = {"enabled": False}
        envs = bench.setdefault("envs", {})
        envs["RUN_EVAL"] = "false"
        # Keep the workload shape and launch args; bound only the trace window.
        envs["NUM_PROMPTS"] = min(16, int(envs.get("NUM_PROMPTS", 16)))
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    async def __call__(self, ctx: Any) -> dict[str, Any]:
        from ._grid_runner import merge_server_args

        params = ctx.task.params or {}
        params["extra_server_args"] = merge_server_args(
            str(params.get("base_extra_args") or ""), str(params.get("extra_server_args") or "")
        )
        params["extra_envs"] = {**(params.get("base_extra_envs") or {}), **(params.get("extra_envs") or {})}
        for key in ("remove_args", "unset_envs", "args_mode"):
            if "base_" + key in params:
                params.setdefault(key, params["base_" + key])
        ctx.task.params = params
        result = await super().__call__(ctx)
        result["measurement_kind"] = "profile"
        # Profile overhead is diagnostic only, never an optimization score.
        result["diagnostic_throughput"] = result.get("output_throughput")
        for key in list(result):
            if "throughput" in key and key != "diagnostic_throughput":
                result.pop(key)
        workspace = result.get("workspace")
        artifact = Path(workspace) / "vllm_cuda_profile.json" if workspace else None
        if artifact is None or not artifact.is_file():
            result.update(status="failed", error_class="cuda_profile_missing", error="CUDA profile artifact missing")
            return result
        profile = json.loads(artifact.read_text(encoding="utf-8"))
        result.update({key: profile[key] for key in ("trace_files", "trace_health", "fingerprints")})
        result["profile_artifact"] = str(artifact)
        result["main_trace_path"] = next(iter(profile["trace_files"]), None)
        if profile["status"] != "succeeded":
            result.update(status="failed", error_class="cuda_profile_invalid", error="; ".join(profile["errors"]))
        elif result.get("status") == "succeeded" or result.get("error_class") == "invalid_measurement":
            # The common measurement reader deliberately rejects profile scores;
            # the trace artifact, including its cleanup verdict, owns success.
            result["status"] = "succeeded"
            result.pop("error_class", None)
            result.pop("error", None)
            result["diagnostic_throughput"] = ((profile.get("diagnostic_metrics") or {}).get("metrics") or {}).get(
                "output_tokens_per_second_per_gpu", result.get("diagnostic_throughput")
            )
        return result
