# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SQLite-backed GPU pool for specialist sub-agents."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from hyperloom.common.timeutil import now_iso
from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS, parse_device_list

from .storage.connection import SqliteConnection


log = logging.getLogger(__name__)


DEFAULT_GPU_LEASE_TTL_SEC = 1800

# Reserved synthetic gpu_id base for single-node Ray pending-observation slots.
_RAY_OBS_ID_BASE = 100000

# GPU-lease / gpu_research_lane TTL grace over the agent wall budget.
GPU_LEASE_TTL_GRACE = 0.1


_now_iso = now_iso


def _parse_gpu_list(raw: str) -> list[int]:
    """Parse a GPU-id list through the shared visible-device parser."""
    return parse_device_list(raw)


def _explicit_pool() -> list[int] | None:
    """Resolve the operator's explicit GPU pool, or ``None`` when unset."""
    raw = os.environ.get("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES")
    if raw is None or not raw.strip():
        return None
    ids = _parse_gpu_list(raw)
    if not ids:
        log.error("INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES=%r has no valid GPU ids", raw)
    return ids


def _visible_device_mask() -> tuple[list[int], bool]:
    """Resolve the process's visible-GPU mask as absolute device ids.

    Checks ``ROCR_VISIBLE_DEVICES`` first (canonical ROCm pinning per the repo
    convention; the CLI preflight drops ``HIP_VISIBLE_DEVICES`` when ROCR is
    set), then ``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES``.

    Returns:
        ``(ids, present)`` where ``present`` is True when any of the masks is
        set (even to an empty string, which means "no visible GPUs" → ``[]``);
        ``present`` is False only when none of the masks is set.
    """
    runtime = os.environ.get("HYPERLOOM_TARGET_RUNTIME", "rocm").strip().lower()
    order = (
        ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
        if runtime == "cuda"
        else ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    )
    for env_name in order:
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        return _parse_gpu_list(raw), True
    return [], False


def resolve_gpu_specialist_devices(
    capacity: int,
    *,
    serving_tp: int = 0,
) -> list[int]:
    """Resolve the absolute GPU ids available to GPU specialists."""
    cap = max(0, int(capacity or 0))
    if cap <= 0:
        return []
    explicit = _explicit_pool()
    if explicit is not None:
        return explicit[:cap]
    serving = max(0, int(serving_tp or 0))
    mask_ids, mask_present = _visible_device_mask()
    if mask_present:
        return mask_ids[serving:][:cap]
    return list(range(cap))[serving:]


def resolve_whole_machine_devices() -> list[int | GpuDeviceRef]:
    """Resolve the full set of GPU ids on this node — **no** serving carve.

    Returns *every* visible card and, unlike
    :func:`resolve_gpu_specialist_devices`, does **not**:

    * subtract the ``serving_tp`` cards, nor
    * gate on ``gpu_specialist_capacity``.

    Id-source precedence mirrors :func:`resolve_gpu_specialist_devices`:

    1. ``INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES`` — explicit operator pool
       (used verbatim, uncapped here); set but unparseable yields ``[]``.
    2. The process visible-device mask (``ROCR_VISIBLE_DEVICES`` →
       ``HIP``/``CUDA``) — used verbatim.
    3. No mask set → ``range(detect_gpu_count())`` (the machine's detected GPU
       count, probed lazily via ``rocm-smi`` only in this branch).

    Returns:
        The absolute GPU ids available to a framework whole-machine lease;
        ``[]`` when the mask is set-but-empty or nothing can be resolved.
    """
    if os.environ.get("HYPERLOOM_TARGET_RUNTIME", "").strip().lower() == "cuda":
        try:
            import json

            fingerprint = json.loads(os.environ.get("HYPERLOOM_HARDWARE_FINGERPRINT", "") or "{}")
            rows = fingerprint.get("devices") if isinstance(fingerprint, dict) else None
            if isinstance(rows, list) and rows:
                return [
                    GpuDeviceRef(
                        index=int(row["index"]),
                        uuid=str(row.get("uuid") or ""),
                        numa_node=(int(row["numa_node"]) if row.get("numa_node") is not None else None),
                    )
                    for row in rows
                    if isinstance(row, dict) and "index" in row
                ]
        except (TypeError, ValueError, json.JSONDecodeError):
            log.warning("invalid HYPERLOOM_HARDWARE_FINGERPRINT; falling back to CUDA indices")
    explicit = _explicit_pool()
    if explicit is not None:
        return explicit
    mask_ids, mask_present = _visible_device_mask()
    if mask_present:
        return mask_ids
    # No mask: fall back to the detected machine GPU count.
    from ..policy.gate import detect_gpu_count

    return list(range(max(0, int(detect_gpu_count() or 0))))


@dataclass(frozen=True)
class GpuDeviceRef:
    """Physical GPU identity retained across index renumbering."""

    index: int
    uuid: str = ""
    numa_node: int | None = None


@dataclass(frozen=True)
class GpuLease:
    holder_id: str
    task_id: str
    gpu_ids: tuple[int, ...]
    acquired_at: str
    expires_at: str
    gpu_uuids: tuple[str, ...] = ()
    numa_nodes: tuple[int | None, ...] = ()


class SpecialistGpuPool:
    """Capacity-limited GPU allocation for specialist tasks."""

    def __init__(
        self,
        db: SqliteConnection,
        *,
        gpu_ids: list[int | GpuDeviceRef] | tuple[int | GpuDeviceRef, ...],
    ):
        """Initialize the pool over a fixed set of GPU ids."""
        self.db = db
        devices: dict[int, GpuDeviceRef] = {}
        for raw in gpu_ids:
            ref = raw if isinstance(raw, GpuDeviceRef) else GpuDeviceRef(index=int(raw))
            if ref.index >= 0:
                devices.setdefault(ref.index, ref)
        self.devices = tuple(devices.values())
        self.gpu_ids = tuple(device.index for device in self.devices)
        self._device_by_id = {device.index: device for device in self.devices}

    @property
    def capacity(self) -> int:
        """Return the number of GPUs the pool manages."""
        return len(self.gpu_ids)

    async def try_acquire(
        self,
        *,
        count: int,
        holder_id: str,
        task_id: str,
        ttl_sec: int = DEFAULT_GPU_LEASE_TTL_SEC,
    ) -> GpuLease | None:
        """Acquire ``count`` GPU ids or return ``None`` if the pool is full."""
        n = int(count or 0)
        if n <= 0 or n > self.capacity:
            return None
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + max(1, int(ttl_sec or DEFAULT_GPU_LEASE_TTL_SEC))
        expires_iso = datetime.fromtimestamp(
            expires_ts,
            tz=timezone.utc,
        ).isoformat(timespec="microseconds")

        async with self.db.transaction() as cur:
            cur.execute(
                "DELETE FROM gpu_leases WHERE expires_at <= ?",
                (now_iso,),
            )
            # Same-holder/task acquire is idempotent when the existing lease already satisfies the request: same count
            # and every id still in this pool.
            cur.execute(
                "SELECT gpu_id, gpu_uuid, numa_node, acquired_at FROM gpu_leases "
                "WHERE holder_id=? AND task_id=? ORDER BY gpu_id",
                (holder_id, task_id),
            )
            existing_rows = cur.fetchall()
            if existing_rows:
                existing_ids = tuple(int(r["gpu_id"]) for r in existing_rows)
                pool_set = set(self.gpu_ids)
                if len(existing_ids) == n and set(existing_ids) <= pool_set:
                    acquired_at = existing_rows[0]["acquired_at"]
                    cur.execute(
                        "UPDATE gpu_leases SET expires_at=?, heartbeat_at=? WHERE holder_id=? AND task_id=?",
                        (expires_iso, now_iso, holder_id, task_id),
                    )
                    return GpuLease(
                        holder_id=holder_id,
                        task_id=task_id,
                        gpu_ids=existing_ids,
                        gpu_uuids=tuple(str(r["gpu_uuid"] or "") for r in existing_rows),
                        numa_nodes=tuple(
                            int(r["numa_node"]) if r["numa_node"] is not None else None for r in existing_rows
                        ),
                        acquired_at=acquired_at,
                        expires_at=expires_iso,
                    )
                # Stale lease — release it and fall through to a fresh acquire.
                cur.execute(
                    "DELETE FROM gpu_leases WHERE holder_id=? AND task_id=?",
                    (holder_id, task_id),
                )
            placeholders = ",".join("?" * len(self.gpu_ids))
            cur.execute(  # nosec B608 - placeholders string is generated from configured GPU id count.
                f"SELECT gpu_id FROM gpu_leases WHERE gpu_id IN ({placeholders})",  # nosec B608 - generated placeholders only.
                list(self.gpu_ids),
            )
            leased = {int(r["gpu_id"]) for r in cur.fetchall()}
            available = [g for g in self.gpu_ids if g not in leased]
            if len(available) < n:
                return None
            selected = available[:n]
            for gpu_id in selected:
                device = self._device_by_id[gpu_id]
                cur.execute(
                    """
                    INSERT INTO gpu_leases(
                        gpu_id, gpu_uuid, numa_node, holder_id, task_id,
                        acquired_at, expires_at, heartbeat_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        gpu_id,
                        device.uuid or None,
                        device.numa_node,
                        holder_id,
                        task_id,
                        now_iso,
                        expires_iso,
                        now_iso,
                    ),
                )
        selected_refs = [self._device_by_id[gpu_id] for gpu_id in selected]
        return GpuLease(
            holder_id=holder_id,
            task_id=task_id,
            gpu_ids=tuple(selected),
            gpu_uuids=tuple(device.uuid for device in selected_refs),
            numa_nodes=tuple(device.numa_node for device in selected_refs),
            acquired_at=now_iso,
            expires_at=expires_iso,
        )

    async def try_acquire_ray_observation(
        self,
        *,
        holder_id: str,
        task_id: str,
        pending_limit: int,
        ttl_sec: int = DEFAULT_GPU_LEASE_TTL_SEC,
    ) -> GpuLease | None:
        """Admit a GPU specialist under single-node Ray by COUNT, not physical id."""
        limit = max(1, int(pending_limit or 1))
        now_ts = time.time()
        now_iso = _now_iso()
        expires_ts = now_ts + max(1, int(ttl_sec or DEFAULT_GPU_LEASE_TTL_SEC))
        expires_iso = datetime.fromtimestamp(
            expires_ts,
            tz=timezone.utc,
        ).isoformat(timespec="microseconds")

        async with self.db.transaction() as cur:
            cur.execute(
                "DELETE FROM gpu_leases WHERE expires_at <= ?",
                (now_iso,),
            )
            cur.execute(
                "SELECT gpu_id FROM gpu_leases WHERE gpu_id >= ?",
                (_RAY_OBS_ID_BASE,),
            )
            used = {int(r["gpu_id"]) for r in cur.fetchall()}
            slot: int | None = None
            for i in range(limit):
                cand = _RAY_OBS_ID_BASE + i
                if cand not in used:
                    slot = cand
                    break
            if slot is None:
                return None
            cur.execute(
                """
                INSERT INTO gpu_leases(
                    gpu_id, holder_id, task_id,
                    acquired_at, expires_at, heartbeat_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (slot, holder_id, task_id, now_iso, expires_iso, now_iso),
            )
        return GpuLease(
            holder_id=holder_id,
            task_id=task_id,
            gpu_ids=(slot,),
            acquired_at=now_iso,
            expires_at=expires_iso,
        )

    async def release(self, lease: GpuLease | None) -> None:
        """Release the GPUs held by a lease."""
        if lease is None or not lease.gpu_ids:
            return
        placeholders = ",".join("?" * len(lease.gpu_ids))
        params = list(lease.gpu_ids) + [lease.holder_id]
        async with self.db.transaction() as cur:
            cur.execute(  # nosec B608 - placeholders string is generated from lease GPU id count.
                f"DELETE FROM gpu_leases WHERE gpu_id IN ({placeholders}) AND holder_id=?",  # nosec B608 - generated placeholders only.
                params,
            )

    async def extend(self, task_id: str, ttl_sec: int) -> int:
        """Push a task's GPU rows out to ``ttl_sec`` from now."""
        expires_iso = datetime.fromtimestamp(time.time() + max(0, int(ttl_sec)), tz=timezone.utc).isoformat()
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            cur.execute(
                "UPDATE gpu_leases SET expires_at=?, heartbeat_at=? WHERE task_id=?",
                (expires_iso, now_iso, task_id),
            )
            return int(cur.rowcount or 0)

    async def reap_expired(self) -> int:
        """Actively delete TTL-expired GPU leases; returns rows reaped."""
        now_iso = _now_iso()
        async with self.db.transaction() as cur:
            cur.execute(
                "DELETE FROM gpu_leases WHERE expires_at <= ?",
                (now_iso,),
            )
            return int(cur.rowcount or 0)


__all__ = [
    "DEFAULT_GPU_LEASE_TTL_SEC",
    "GPU_LEASE_TTL_GRACE",
    "GpuDeviceRef",
    "GpuLease",
    "SpecialistGpuPool",
    "resolve_gpu_specialist_devices",
    "resolve_whole_machine_devices",
]
