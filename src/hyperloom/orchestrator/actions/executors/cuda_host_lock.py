# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cooperative host exclusion between CUDA experiments and a resident service."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

LOCK_PATH = Path("/tmp/hyperloom-nvidia-rtx4090-8x.lock")


class CudaHostLock:
    """Hold one inode for the full GPU ownership interval; never unlink it."""

    def __init__(self, owner: str, path: Path | None = None):
        self.owner = owner
        self.path = path or LOCK_PATH
        self.fd: int | None = None

    def __enter__(self):
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise RuntimeError(f"CUDA host is reserved by another service/benchmark: {self.path}") from None
        self.fd = fd
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "owner": self.owner}).encode())
        return self

    def __exit__(self, *_args):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
