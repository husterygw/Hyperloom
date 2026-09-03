# inference_optimizer

The **inference_optimizer** package is the canonical entry point for
Hyperloom's autonomous LLM inference optimization on AMD GPUs. It is
the CLI/session layer — CLI, session paths, action `_meta` specs,
protocol surfaces, and breakdown export — that launches the Coordinator
(a Python state machine) living in the sibling `hyperloom.orchestrator`
package, which drives the four-agent architecture — Orchestration,
Kernel, Critic, and Robustness — through baseline measurement,
profiling, parameter search, kernel optimization, and validated
promotion.

This is the package referenced by `src/hyperloom/inference_optimizer/SKILL.md`;
it is installed from the `hyperloom-inference_optimizer` wheel published on
GitHub Releases (see `examples/README.md`), not from PyPI.

## Where to read next

* **[SKILL.md](SKILL.md)** — the agent-facing instructions: full
  optimization protocol, prompt templates, failure handling, and
  knowledge-base usage. Cursor and Claw load this on demand.
* **[../../../README.md](../../../README.md)** — repository-level overview,
  quickstart links, and the documentation index.
* **[../../../docs/conceptual/optimization-loop.md](../../../docs/conceptual/optimization-loop.md)**
  — the stateless orchestration loop, the phase chain and per-phase
  contracts, RecipeKB feedback loops, and the retired-names list.
* **[../../../docs/reference/authentication.md](../../../docs/reference/authentication.md)** — credential
  and environment configuration.
* **[../../../docs/reference/environment-variables.md](../../../docs/reference/environment-variables.md)**
  — exhaustive list of every environment variable read by the runtime.
* **[../../../docs/reference/session-breakdown.md](../../../docs/reference/session-breakdown.md)**
  — the `session_breakdown.json` contract for downstream consumers.

## Quick CLI

Use the module entry point; it works both for normal installs and
`pip install --target` layouts where console scripts are not on `PATH`:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
    --model /path/to/model \
    --framework sglang \
    --gpu-type mi300x \
    --model-class moe_mla \
    --isl 1024 --osl 1024 \
    --max-hours 2.0
```

Resume an interrupted session (the session dir is always named explicitly —
take it from the launch-info JSON or the `HYPERLOOM_LAUNCH` line the CLI
printed at launch):

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
    --resume-from "$SESSION_DIR"
```

See `python -m hyperloom.inference_optimizer.cli optimize --help` for the full flag set and
[SKILL.md](SKILL.md) for the prompt-driven launch workflow used inside
Cursor and Claw.

## Experimental NVIDIA/vLLM target

The `nvidia_rtx4090_8x_local` target is the config-only CUDA MVP for the
validated local host: exactly eight RTX 4090 GPUs (compute capability 8.9) and
CUDA 13.0 at `/usr/local/cuda-13.0`. vLLM is operator-managed rather than
version-pinned: preflight probes the selected interpreter's `vllm serve` and
`vllm bench serve` CLI, then fails closed only when it lacks the runner's
required capabilities. The actual version and CLI surface are recorded in the
session hardware fingerprint. It also fails closed on hardware, compiler,
session-resume, or `TP*PP` drift. It does not enter
the ROCm/Magpie/InferenceX/TraceLens/GEAK/Quark paths.

```bash
conda activate llm_sim
codex login                       # one-time ChatGPT login, if not already done
export HYPERLOOM_BENCHMARK_BACKEND=vllm_cuda
bash src/hyperloom/inference_optimizer/assets/install.sh

python -m hyperloom.inference_optimizer.cli optimize \
    --target nvidia_rtx4090_8x_local \
    --codex-cli-auth \
    --model /path/to/model \
    --framework vllm \
    --tp 1 --pp 8 \
    --isl 128 --osl 32 --conc 2 \
    --num-prompts 100 --num-warmups 5 \
    --max-hours 2
```

`--codex-cli-auth` explicitly reuses the current `codex login` session; no
`OPENAI_API_KEY` or `OPENAI_BASE_URL` is required. Hyperloom validates the
owner-only CLI credential file, copies it into a private per-agent home, and
removes that copy when the agent exits. Explicit gateway credentials still take
precedence. Codex sandboxing remains independent: the secure default requires a
working bubblewrap capability probe. On a host that already supplies an
external isolation boundary but blocks bubblewrap namespaces, use the documented
double opt-in `HYPERLOOM_CODEX_SANDBOX_MODE=bypass` plus
`HYPERLOOM_CODEX_EXTERNAL_SANDBOX=1`.

The target supports baseline, config exploration, sweep, and report. Profile,
source patching, kernel patching, quantization, warm replay, evaluation, and
multi-node execution are disabled by target capabilities. Each benchmark emits
an atomic `vllm_cuda_benchmark.json`, the compatibility
`benchmark_report.json`, raw vLLM JSON/logs, a launch plan, total tok/s,
per-GPU tok/s, and p50/p90/p99 latency metrics. GPU leases retain physical
index, UUID, and NUMA identity, and cleanup only terminates the session-owned
process group.

`--num-prompts` and `--num-warmups` optionally pin the serving measurement
protocol for every non-profile run; they are persisted in `state.json` and
restored by `--resume-from`. This is the supported way to request a fixed
sample size (for example, P3's 100 continuous requests); generic `--extra-env`
does not retarget these workload-owned values.

## Layout

```
src/hyperloom/inference_optimizer/
├── SKILL.md                    # Agent instructions (Cursor / Claw entry point)
├── references/                 # SKILL reference chapters (benchmark/cache/critic/…)
├── cli/                        # `python -m hyperloom.inference_optimizer.cli optimize` entry point
│   ├── __init__.py             # main()/_run_optimize()
│   ├── parser.py               # _build_parser()
│   ├── backends/bootstrap/executors/kb/model_gate/preflight.py
│   └── credentials/multi_node/quantization/recover.py
├── model_config_utils.py       # stdlib-only model-config leaf shared by cli/ and the orchestrator
├── session/                    # Session paths, manifest writer, single-optimizer lock
│   ├── manifest.py             # Session manifest writer
│   ├── paths.py                # USER_DATA_PATH-rooted path helpers
│   ├── session_paths.py        # Per-session artifact path helpers
│   └── lock.py                 # Single-optimizer session lock
├── baseline_comparison/        # InferenceX reference fetching & target analysis
├── breakdown/                  # session_breakdown.json producer (downstream contract)
├── actions/                    # Per-action markdown specs + scheduling metadata
├── tools/                      # Operator CLIs (dump_session_breakdown/event_counts/…)
├── experiments/                # A/B and roofline-audit scripts
├── assets/                     # install.sh + bare-metal/profile configs
└── tests/                      # Unit + regression tests
```

The Coordinator + agent roles + action executors live in the sibling
`hyperloom.orchestrator` package (`src/hyperloom/orchestrator/`), not under
`inference_optimizer/`.

## Package metadata

* **License:** MIT (see top-level `LICENSE`).
* **Python:** 3.10+.
* **Distribution:** built from `pyproject.toml` (at the repo root) as the
  `hyperloom-inference_optimizer` wheel and attached to GitHub Releases; it is
  not published to PyPI. Install the release wheel directly (see
  `examples/README.md` for the versioned GitHub Release wheel URL).
