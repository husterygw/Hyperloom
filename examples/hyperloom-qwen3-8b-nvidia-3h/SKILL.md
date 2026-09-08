---
name: hyperloom-qwen3-8b-nvidia-3h
description: Run a bounded 3-hour Qwen3-8B config optimization demo with optional torch profiling from a Hyperloom source checkout on the local eight-RTX-4090 NVIDIA CUDA/vLLM target. Use for the NVIDIA counterpart of the AMD Qwen3-8B demo.
---

# Qwen3-8B NVIDIA/vLLM 3-hour example

This is the NVIDIA counterpart of `../hyperloom-qwen3-8b-3h/SKILL.md`.
Use this skill's CUDA launch path; the original demo's ROCm setup, Magpie,
kernel-agent environment, and Docker images do not apply.

The current target requires a host with **exactly eight RTX 4090 GPUs**,
SM 8.9, at least 24000 MiB per GPU, and CUDA 13.0 at
`/usr/local/cuda-13.0`. This example uses **one leased GPU (TP1/PP1)** on
that host. It is not a generic target for arbitrary NVIDIA cards or GPU counts.
vLLM must already work in the selected Python environment; preflight checks its
actual CLI capabilities and records its version.

## Workload and supported optimization

| Setting | Value |
|---|---|
| Model | Local Qwen3-8B checkpoint, BF16 |
| Backend | `vllm_cuda`, local host |
| Parallelism | TP1 / PP1 / EP1 |
| Input / output / concurrency | 512 / 128 tokens / 4 |
| Maximum model length | 2048 tokens |
| Measurement | 100 requests, 5 warmups |
| Quality suite | `qwen3_p3` |
| Initial server args | `--gpu-memory-utilization 0.90` |
| Time budget / gain target | 165 minutes optimization + cleanup, 180-minute process cap / 30% stopping target |

The original AMD demo uses concurrency 64 and 1024/1024 tokens. This smaller
starting workload leaves more room for KV cache on a 24 GiB card; it is an
example configuration, not a measured optimum. Keep the default CUDA Graph
behavior so the baseline does not manufacture gains by forcing eager mode.
The 30% target is a stopping condition, not a promised speedup.

The loop can benchmark, explore allowed vLLM configuration changes, keep or
revert candidates, and report results. Add `--optimization-level profile` to
capture native vLLM torch traces, validated for every TP×PP rank. This preset
reserves 15% of the optimization budget for PRELUDE (baseline plus profiling). Profiling
measurements are diagnostic and cannot become performance winners. Roofline,
framework source patches, kernel optimization, quantization, evaluation, warm replay, and
multi-node execution remain disabled. Keep FRAMEWORK_AGENT enabled for config
exploration; do not pass `--no-framework-agent`. As in the original short demo,
the extra post-optimization concurrency sweep is disabled.

## Prepare the existing CUDA environment

Use a source checkout containing the NVIDIA target and this example. Resolve
the user's existing Python environment, model path, artifact root, and auth
settings first. Reuse values already supplied by the user; ask only for missing
required values. Load needed credentials privately from a trusted `.env` when
applicable, preserving exported values. Do not print credentials or source an
old ROCm runtime environment into this launch.

On the validated host the environment is `llm_sim`. From the checkout root:

```bash
conda activate llm_sim
export REPO_ROOT="$(pwd -P)"
export PYTHON="$(command -v python)"
export MODEL_PATH=/absolute/path/to/Qwen3-8B
export USER_DATA_PATH="${USER_DATA_PATH:-/data/ygw/llm_sim/hyperloom_nv_examples}"
export HYPERLOOM_TARGET=nvidia_rtx4090_8x_local
export HYPERLOOM_BENCHMARK_BACKEND=vllm_cuda

# First-time preparation; reuse a prepared installation on subsequent runs.
"$PYTHON" -m pip install -e '.[nvidia]'
bash src/hyperloom/inference_optimizer/assets/install.sh
```

`MODEL_PATH` must be absolute and contain `config.json`. The launcher does not
download a model or silently substitute a different one. Resolve a missing
checkpoint with the user before launching. Preserve an existing
`USER_DATA_PATH`; it names the workspace root, not an individual session.

The launcher uses `--codex-cli-auth` to reuse an existing `codex login` session.
If login is absent, let the user complete it before launch. Explicit configured
gateway credentials retain precedence. Keep the normal sandbox checks; consult
`../../docs/reference/authentication.md` and the NVIDIA section of
`../../src/hyperloom/inference_optimizer/README.md` for environment-specific
setup. Do not automatically enable sandbox bypass.

With `vllm_cuda` selected, the installer skips AMD serving/kernel integrations.
There is no requirement to source `runtime/kernel-agent.env.sh` for this target.
Do not upgrade the existing CUDA/PyTorch/vLLM stack as part of routine example
preparation. On preflight failure, report the failing contract and resolve it
before retrying; do not change the target to bypass the check.

## Preview and launch

The adjacent `run_example.sh` is the canonical command. It resolves the source
root independently of the working directory and passes explicit workload flags.
It does not install dependencies, download checkpoints, or clean up other jobs.

```bash
bash examples/hyperloom-qwen3-8b-nvidia-3h/run_example.sh --dry-run
```

This checks required paths and prints the command only. It does **not** validate
GPU availability, model loading, credentials, or end-to-end performance.
The real optimize command performs target and runtime preflight.

For a foreground run:

```bash
bash examples/hyperloom-qwen3-8b-nvidia-3h/run_example.sh
```

For an agent-managed background run, record the launch separately from the
optimizer's session artifacts:

```bash
mkdir -p "$USER_DATA_PATH/launches"
DEMO_LAUNCH_DIR="$(mktemp -d "$USER_DATA_PATH/launches/qwen3-8b-nvidia.XXXXXXXX")"
setsid nohup bash "$REPO_ROOT/examples/hyperloom-qwen3-8b-nvidia-3h/run_example.sh" \
  >"$DEMO_LAUNCH_DIR/optimizer.log" 2>&1 < /dev/null &
DEMO_PID=$!
printf '%s\n' "$DEMO_PID" > "$DEMO_LAUNCH_DIR/optimizer.pid"
```

Before launch, report the model, interpreter, target, single-GPU workload,
budget, and artifact root. After launch, report the PID and log, then obtain the
actual session directory from the optimizer output. Do not label a started
process as a successful benchmark. Check initial preflight and baseline results.
During the run, monitor approximately every five minutes and report phase,
stop reason, measured throughput, and candidate KEEP/REVERT decisions when
available. The optimizer budget is 165 minutes. GNU `timeout` sends TERM at
179 minutes and KILL after a further minute; inspect cleanup artifacts on timeout.

## Resume and result interpretation

After an unexpected crash, inspect the log and `state.json`, and verify the
previous session's processes have stopped. Resume the same session with the
same environment and workspace root:

```bash
bash examples/hyperloom-qwen3-8b-nvidia-3h/run_example.sh \
  --resume-from /absolute/path/to/existing/session
```

Resume uses persisted workload and budget values rather than the fresh-run
preset, including its optimization level; explicitly changing the level on
resume is rejected. Do not loop on failures, create replacement sessions automatically, or
force a terminal session to resume. Never kill unrelated GPU jobs; the CUDA
runner owns its leases and server process groups.

At completion, inspect `state.json`, `session_breakdown.json`,
`reports/optimization_journal.json`, `reports/final.*`, and the baseline and
candidate `vllm_cuda_benchmark.json` artifacts. Report the actual stop reason,
quality outcomes, accepted configuration, and measured gains. A safety-net
final report does not establish that CLOSE completed. No KEEP is a valid result;
single-GPU synthetic results are not eight-GPU or production-SLO evidence.
