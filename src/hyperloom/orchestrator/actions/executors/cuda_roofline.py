# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""NVIDIA timeline plus bounded, independent hotspot counter captures."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any

from .cuda_profile import CudaProfileExecutor
from .cuda_nsight import write_analysis
from ._grid_runner import session_grid_bounds


class CudaRooflineExecutor:
    def __init__(self, *, shared_state: Any, session_dir: Path | None = None):
        self.shared_state = shared_state
        self.session_dir = Path(session_dir) if session_dir else None

    async def __call__(self, ctx: Any) -> dict[str, Any]:
        session_dir = self.session_dir or getattr(self.shared_state, "_session_dir", None)
        if session_dir is None:
            return {"status": "failed", "error_class": "nsight_session_missing"}
        session_dir = Path(session_dir)
        root = Path((ctx.task.params or {}).get("output_dir") or session_dir / "runs" / "roofline" / ctx.task.task_id)
        deadline, _ = session_grid_bounds(self.shared_state)
        budget = min(2700, deadline - time.monotonic() - 60) if deadline is not None else 2700
        if budget <= 0:
            return {"status": "failed", "measurement_kind": "profile", "error_class": "nsight_budget_exhausted"}
        result = None
        summary = None
        errors = []

        async def capture(
            backend: str,
            directory: str,
            names: list[str] | None = None,
            devices: list[int] | None = None,
            worker_rank: int | None = None,
        ) -> dict[str, Any]:
            child = copy.copy(ctx)
            child.task = copy.copy(ctx.task)
            child.task.params = copy.deepcopy(ctx.task.params or {})
            child.task.params["output_dir"] = str(root / directory)
            child.extra = {**(ctx.extra or {}), "workspace": str(root / directory)}
            return await CudaProfileExecutor(
                session_dir=session_dir,
                shared_state=self.shared_state,
                backend=backend,
                kernel_names=names,
                devices=devices,
                worker_rank=worker_rank,
                delay_iterations=0,
                max_iterations=0,
            )(child)

        try:
            async with asyncio.timeout(budget):
                result = await capture("nsys", "timeline")
                if result.get("status") != "succeeded":
                    return result
                workspace = Path(result["workspace"])
                summary = json.loads((workspace / "nsight_summary.json").read_text())
                selected = [r for r in summary["hot_kernels"] if not r["communication"]][:3]
                rank_uuids = {r["rank"]: r["gpu_uuid"] for r in summary["ranks"]}
                if not selected:
                    errors.append("no non-communication hotspot available")
                # Nsight instrumentation of unselected TP workers can also
                # stall prefill. Profile one worker at a time for all phases.
                # Bound each selected name to one launch in the same run,
                # avoiding a complete service restart for every hotspot.
                jobs = []
                isolate = int(getattr(self.shared_state, "tp", 1) or 1) > 1
                if isolate:
                    for device, uuid in enumerate(result["topology"]["gpu_uuids"]):
                        rank = next(r["rank"] for r in summary["ranks"] if r["gpu_uuid"] == uuid)
                        targets = [h for h in selected if rank in h["ranks"]]
                        if targets:
                            jobs.append((f"counter_device_{device}", targets, [device], {rank}))
                else:
                    for index, hotspot in enumerate(selected):
                        jobs.append((f"counter_{index}", [hotspot], None, set(hotspot["ranks"])))
                summary["counter_strategy"] = "isolated_workers" if isolate else "sequential_hotspots"
                for directory, targets, devices, expected_ranks in jobs:
                    counter = await capture(
                        "ncu",
                        directory,
                        [h["name"] for h in targets],
                        devices,
                        next(iter(expected_ranks)) if devices else None,
                    )
                    health = counter.get("trace_health") or {}
                    ranks = {r["rank"] for r in health.get("ranks", [])}
                    if counter.get("status") != "succeeded":
                        errors.append(
                            f"{directory}: counter capture failed ({counter.get('capture_failure')}): "
                            f"{counter.get('error', '')}"
                        )
                        continue
                    if ranks != expected_ranks:
                        errors.append(
                            f"{directory}: rank coverage mismatch: "
                            f"expected {sorted(expected_ranks)}, collected {sorted(ranks)}"
                        )
                        continue
                    identity_keys = ("hardware", "model", "workload", "serving_config", "quality_suite")
                    expected_identity = result.get("fingerprints") or {}
                    counter_identity = counter.get("fingerprints") or {}
                    if any(
                        not expected_identity.get(k) or counter_identity.get(k) != expected_identity[k]
                        for k in identity_keys
                    ):
                        errors.append(f"{directory}: model/workload/serving configuration fingerprint mismatch")
                        continue
                    for hotspot in targets:
                        expected = {(r["rank"], tuple(r["grid"]), tuple(r["block"])) for r in hotspot["launches"]}
                        matched = []
                        for launch in health.get("launches", []):
                            if launch["kernel_name"] != hotspot["name"]:
                                continue

                            def dims(text: str) -> tuple[int, ...]:
                                return tuple(int(x.strip()) for x in text.strip("()").split(","))

                            signature = (launch["rank"], dims(launch["grid"]), dims(launch["block"]))
                            if (
                                signature in expected
                                and launch["rank"] in expected_ranks
                                and launch.get("gpu_uuid") == rank_uuids[launch["rank"]]
                            ):
                                matched.append(
                                    {
                                        **{k: v for k, v in launch.items() if k != "metrics"},
                                        "profile_artifact": counter["profile_artifact"],
                                    }
                                )
                        if {r["rank"] for r in matched} != expected_ranks:
                            errors.append(f"{directory}/{hotspot['kernel_id']}: launch-shape attribution incomplete")
                            continue
                        evidence = hotspot.setdefault(
                            "ncu_roofline",
                            {
                                "launches": [],
                                "profile_artifact": counter["profile_artifact"],
                                "profile_artifacts": [],
                                "trace_files": [],
                            },
                        )
                        evidence["launches"].extend(matched)
                        evidence["profile_artifacts"].append(counter["profile_artifact"])
                        evidence["trace_files"].extend(counter["trace_files"])
                for hotspot in selected:
                    launches = (hotspot.get("ncu_roofline") or {}).get("launches", [])
                    if {r["rank"] for r in launches if r["ncu_roofline"]["status"] == "available"} != set(
                        hotspot["ranks"]
                    ):
                        errors.append(f"{hotspot['kernel_id']}: complete rank coverage or roofline metrics unavailable")
        except asyncio.CancelledError:
            if result is not None and summary is not None:
                summary["counter_status"] = "cancelled"
                summary["counter_errors"] = ["Counter capture cancelled; valid timeline retained"]
                write_analysis(Path(result["workspace"]), summary)
            raise
        except TimeoutError:
            errors.append("Nsight analysis budget exhausted; capture cancelled and owned processes cleaned up")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        if result is None or summary is None:
            return {
                "status": "failed",
                "measurement_kind": "profile",
                "error_class": "cuda_roofline_failed",
                "error": "; ".join(errors),
            }
        summary["counter_status"] = "failed" if errors else "succeeded"
        summary["counter_errors"] = errors
        analysis = write_analysis(Path(result["workspace"]), summary)
        # Even a failed counter stage preserves the new, valid timeline. It
        # never overwrites serving scores or advertises complete roofline data.
        self.shared_state.last_profile_trace = result["main_trace_path"]
        self.shared_state.last_profile_status = "succeeded"
        self.shared_state.last_profile_args = str((ctx.task.params or {}).get("base_extra_args") or "")
        self.shared_state.record_profile_workload(ctx.task.params or {})
        self.shared_state.record_trace_analyze({"trace_input": result["main_trace_path"]}, analysis)
        result.update(
            status="failed" if errors else "succeeded",
            analysis_result=analysis,
            counter_health={"passed": not errors, "errors": errors},
            snapshot_id=self.shared_state.last_trace_analyze.get("roofline_snapshot_id"),
        )
        if errors:
            result.update(error_class="cuda_roofline_incomplete", error="; ".join(errors))
        (root / "roofline_result.json").write_text(json.dumps(result, indent=2))
        return result
