# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cooperative host exclusion between CUDA experiments and a resident service."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

LOCK_PATH = Path("/tmp/hyperloom-nvidia-cuda.lock")
LEGACY_LOCK_PATH = Path("/tmp/hyperloom-nvidia-rtx4090-8x.lock")


class CudaHostLock:
    """Hold one inode for the full GPU ownership interval; never unlink it."""

    def __init__(self, owner: str, path: Path | None = None):
        self.owner = owner
        self.path = path or LOCK_PATH
        self.fd: int | None = None
        self._fds: list[int] = []

    def __enter__(self):
        # Retain exclusion with already-running pre-generalization processes.
        paths = [self.path]
        if self.path == Path("/tmp/hyperloom-nvidia-cuda.lock"):
            paths.append(LEGACY_LOCK_PATH)
        try:
            for path in paths:
                fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                self._fds.append(fd)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.ftruncate(fd, 0)
                os.write(fd, json.dumps({"pid": os.getpid(), "owner": self.owner}).encode())
            self.fd = self._fds[0]
        except BaseException:
            self.__exit__(None, None, None)
            raise RuntimeError(f"CUDA host is reserved by another service/benchmark: {self.path}") from None
        return self

    def __exit__(self, *_args):
        for fd in reversed(self._fds):
            os.close(fd)
        self._fds.clear()
        self.fd = None
