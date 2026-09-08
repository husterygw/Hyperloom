# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish serving-process ownership before exec under an external profiler."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


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
    os.execvpe(sys.argv[2], sys.argv[2:], os.environ)


if __name__ == "__main__":
    main()
