#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Source-checkout example; prepare the CUDA environment as described in SKILL.md.
set -euo pipefail

EXAMPLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
DEMO_PYTHON="${PYTHON:-python}"
DEMO_DRY_RUN=0
DEMO_RESUME=""
DEMO_LEVEL=""
DEMO_BACKEND=""
DEMO_ROOFLINE=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DEMO_DRY_RUN=1; shift ;;
    --optimization-level)
      if [ "$#" -lt 2 ] || [[ "$2" != config && "$2" != profile ]]; then
        echo 'error: --optimization-level requires config or profile' >&2
        exit 2
      fi
      DEMO_LEVEL="$2"
      shift 2
      ;;
    --profile-backend)
      if [ "$#" -lt 2 ] || [[ "$2" != torch && "$2" != nsys ]]; then
        echo 'error: --profile-backend requires torch or nsys' >&2
        exit 2
      fi
      DEMO_BACKEND="$2"; shift 2 ;;
    --enable-roofline|--no-enable-roofline) DEMO_ROOFLINE="$1"; shift ;;
    --resume-from)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        echo 'error: --resume-from requires a session directory' >&2
        exit 2
      fi
      DEMO_RESUME="$2"
      shift 2
      ;;
    --help|-h)
      echo 'Usage: MODEL_PATH=/local/Qwen3-8B USER_DATA_PATH=/data/sessions bash run_example.sh [--dry-run] [--optimization-level config|profile] [--profile-backend torch|nsys] [--enable-roofline|--no-enable-roofline] [--resume-from /absolute/session]'
      echo 'Uses the current Python (override with PYTHON). Dry run prints the command without starting Hyperloom.'
      exit 0
      ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${USER_DATA_PATH:?Set USER_DATA_PATH to the artifact workspace root}"
if [[ "$USER_DATA_PATH" != /* ]]; then
  echo 'error: USER_DATA_PATH must be absolute' >&2
  exit 2
fi
export USER_DATA_PATH
export PYTHONPATH="${EXAMPLE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HYPERLOOM_TARGET=nvidia_rtx4090_8x_local
export HYPERLOOM_BENCHMARK_BACKEND=vllm_cuda
export INFERENCE_OPTIMIZER_RAY_EXEC=0

DEMO_COMMAND=("$DEMO_PYTHON" -m hyperloom.inference_optimizer.cli optimize
  --target nvidia_rtx4090_8x_local --codex-cli-auth)
if [ -n "$DEMO_LEVEL" ]; then
  DEMO_COMMAND+=(--optimization-level "$DEMO_LEVEL")
fi
if [ -n "$DEMO_BACKEND" ]; then
  DEMO_COMMAND+=(--profile-backend "$DEMO_BACKEND")
fi
if [ -n "$DEMO_ROOFLINE" ]; then
  DEMO_COMMAND+=("$DEMO_ROOFLINE")
fi
if [ -n "$DEMO_RESUME" ]; then
  if [[ "$DEMO_RESUME" != /* ]] || [ ! -f "$DEMO_RESUME/state.json" ]; then
    echo 'error: --resume-from must name an absolute session directory containing state.json' >&2
    exit 2
  fi
  # Resume restores the persisted workload and budget; do not reseed defaults.
  DEMO_COMMAND+=(--resume-from "$DEMO_RESUME")
else
  : "${MODEL_PATH:?Set MODEL_PATH to a local Qwen3-8B checkpoint containing config.json}"
  if [[ "$MODEL_PATH" != /* ]] || [ ! -f "$MODEL_PATH/config.json" ]; then
    echo 'error: MODEL_PATH must be absolute and contain config.json' >&2
    exit 2
  fi
  DEMO_PRELUDE_PCT=0.03
  DEMO_FRAMEWORK_PCT=0.90
  if [ "$DEMO_LEVEL" = profile ]; then
    DEMO_PRELUDE_PCT=0.15
    DEMO_FRAMEWORK_PCT=0.78
  fi
  if [ "$DEMO_BACKEND" = nsys ] && [ "$DEMO_ROOFLINE" != --no-enable-roofline ]; then
    DEMO_PRELUDE_PCT=0.35
    DEMO_FRAMEWORK_PCT=0.58
  fi
  if [ "$DEMO_BACKEND" != nsys ] && [ -z "$DEMO_ROOFLINE" ]; then
    DEMO_COMMAND+=(--no-enable-roofline)
  fi
  DEMO_COMMAND+=(
    --model "$MODEL_PATH" --framework vllm
    --tp 1 --pp 1 --ep 1 --precision bf16
    --isl 512 --osl 128 --conc 4 --max-model-len 2048
    --num-prompts 100 --num-warmups 5 --quality-suite qwen3_p3
    --server-args '--gpu-memory-utilization 0.90'
    --target-gain 30 --max-hours 2.75
    --max-minutes-prelude-pct "$DEMO_PRELUDE_PCT"
    --max-minutes-framework-pct "$DEMO_FRAMEWORK_PCT" --max-minutes-sweep-pct 0.01
    --no-kernel --no-enable-conc-sweep
    --no-warm-replay --no-eval
  )
fi

# 165 minutes for optimization; TERM at minute 179 leaves one minute for
# runner-owned cleanup before the process deadline at minute 180.
DEMO_COMMAND=(timeout --signal=TERM --kill-after=60s 179m "${DEMO_COMMAND[@]}")
if [ "$DEMO_DRY_RUN" -eq 1 ]; then
  printf '%q ' "${DEMO_COMMAND[@]}"
  printf '\n'
  exit 0
fi

cd "$EXAMPLE_ROOT"
exec "${DEMO_COMMAND[@]}"
