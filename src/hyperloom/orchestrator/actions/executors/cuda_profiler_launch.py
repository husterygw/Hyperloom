# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish serving-process ownership before exec under an external profiler."""

from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import shutil
import select
import signal
import sys
import time
from pathlib import Path


def prepare_worker_filter(workspace: Path, rank: int) -> dict:
    """Give one spawned worker a stable executable name for NCU filtering."""
    if importlib.util.find_spec("sitecustomize") is not None:
        raise ValueError("NCU worker filtering cannot override an existing sitecustomize module")
    root = workspace / "ncu_worker"
    executable = root / "bin" / "hyperloom_ncu_worker"
    executable.parent.mkdir(parents=True)
    shutil.copy2(Path(sys.executable).resolve(), executable)
    (root / "lib").symlink_to(Path(sys.prefix) / "lib", target_is_directory=True)
    site = root / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "from hyperloom.orchestrator.actions.executors.cuda_profiler_launch import install_worker_filter\n"
        "install_worker_filter()\n"
    )
    return {
        "rank": rank,
        "executable": str(executable),
        "process_name": executable.name,
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "python_prefix": sys.prefix,
        "site_dir": str(site),
        "manifest_path": str(root / "workers.jsonl"),
    }


def install_worker_filter() -> None:
    # vLLM starts its named workers serially using Python's spawn context.
    # Only the executable name changes; restore the parent's choice afterward.
    import multiprocessing as mp
    import multiprocessing.process
    import multiprocessing.spawn

    original = mp.process.BaseProcess.start

    def start(process):
        previous = mp.spawn.get_executable()
        selected = process.name == "VllmWorker-" + os.environ["HYPERLOOM_NCU_WORKER_RANK"]
        if selected:
            mp.set_executable(os.environ["HYPERLOOM_NCU_WORKER_EXECUTABLE"])
        try:
            result = original(process)
            if process.name.startswith("VllmWorker-"):
                ticks = _start_ticks(process.pid)
                if ticks is not None:
                    with Path(os.environ["HYPERLOOM_NCU_WORKER_MANIFEST"]).open("a") as stream:
                        stream.write(json.dumps({"pid": process.pid, "start_ticks": ticks}) + "\n")
            return result
        finally:
            if selected:
                mp.set_executable(previous)

    mp.process.BaseProcess.start = start


def _start_ticks(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in {"Z", "X"} else fields[19]
    except FileNotFoundError:
        return None


def _pidfd_open(pid: int) -> int:
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    # Conda Python built against older glibc may omit the wrapper even on a
    # recent kernel. These Linux architectures share the pidfd_open syscall.
    import ctypes
    import platform

    if platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("NCU worker cleanup requires Python with os.pidfd_open")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def cleanup_profile_workers(root: Path) -> None:
    """Reap recorded workers even if Nsight moved them out of the serving group."""
    for manifest in root.glob("**/ncu_worker/workers.jsonl"):
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        active = {}
        poller = select.poll()
        signalled = []
        try:
            for row in records:
                try:
                    fd = _pidfd_open(row["pid"])
                except ProcessLookupError:
                    continue
                if _start_ticks(row["pid"]) != row["start_ticks"]:
                    os.close(fd)
                    continue
                active[fd] = row["pid"]
                poller.register(fd, select.POLLIN)
            # pidfds bind signals to the recorded process, never a recycled PID.
            for sig, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
                for fd, pid in active.items():
                    try:
                        signal.pidfd_send_signal(fd, sig)
                        signalled.append({"pid": pid, "signal": sig.name})
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + grace
                while active and time.monotonic() < deadline:
                    for fd, _ in poller.poll(50):
                        poller.unregister(fd)
                        os.close(fd)
                        active.pop(fd)
            with manifest.with_name("cleanup.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {"observed_at": time.time(), "signalled": signalled, "remaining_pids": list(active.values())}
                    )
                    + "\n"
                )
            if active:
                raise RuntimeError(f"NVIDIA profile workers did not exit: {list(active.values())}")
        finally:
            for fd in active:
                os.close(fd)


def main() -> None:
    from . import bypass_engine

    ownership = json.loads(Path(sys.argv[1]).read_text())
    # Nsight may create a separate group for the application. Record the actual
    # group, independently of the profiler PID, before any CUDA initialization.
    if os.getpgrp() != os.getpid():
        os.setsid()
    bypass_engine.write_lifecycle_files(
        pid_dir=ownership["pid_dir"],
        framework="vllm",
        port=ownership["port"],
        pid=os.getpid(),
        pgid=os.getpgrp(),
        model=ownership["model"],
        metadata=ownership["metadata"],
    )
    worker = ownership.get("ncu_worker_filter")
    if worker:
        os.environ.update(
            PYTHONPATH=worker["site_dir"] + os.pathsep + os.environ.get("PYTHONPATH", ""),
            PYTHONHOME=worker["python_prefix"],
            HYPERLOOM_NCU_WORKER_RANK=str(worker["rank"]),
            HYPERLOOM_NCU_WORKER_EXECUTABLE=worker["executable"],
            HYPERLOOM_NCU_WORKER_MANIFEST=worker["manifest_path"],
        )
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == "__main__":
    main()
