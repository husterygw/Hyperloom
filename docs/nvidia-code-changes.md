# Hyperloom NVIDIA/CUDA 支持代码变更说明

更新日期：2026-09-16

本文说明 NVIDIA/CUDA 支持代码相对于原工程的变化、每组变化解决的问题，以及这些代码共同完成的功能。

## 1. 对比范围

本文使用以下版本作为对比基线：

- 原工程基线：`4aa00d425`（增加 NVIDIA 代码之前的版本）。
- NVIDIA 初始支持：`8710d3131` 至 `039f55535` 的提交链。
- 当前版本：上述通用化改造已完成并随同步提交推送到 `origin/my_dev`。

从原工程基线到 NVIDIA 初始支持，Git 差异涉及 89 个文件，约新增 11,731 行；当前工作区相对于 NVIDIA 初始支持又涉及 27 个文件。这些数字包含测试、文档和实验隔离代码，不能理解为全部是运行时逻辑。

NVIDIA 支持的目标不是重新实现 Coordinator、agent 或已有推理框架，而是把已有优化闭环接到 CUDA 平台：

```text
选择 CUDA 平台
  → 发现设备并检查环境
  → 分配 GPU、启动服务、执行基准
  → 保存吞吐/延迟/质量结果
  → 按需采集 CUDA trace 或 Nsight 数据
  → 将分析证据写回会话
  → 让 agent 在已接通的能力范围内继续决策
```

当前正式接通的是单机 CUDA 配置测量和 profiling 路径；源码补丁、kernel 优化、量化、多机和 MIG 调度仍由能力开关限制。

## 2. 代码变化总览

| 代码层 | 主要文件 | 增加或修改的内容 | 要完成的功能 |
|---|---|---|---|
| 平台目标 | [`target_registry.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/target_registry.py) | 目标描述、能力表、设备发现、硬件指纹、CUDA 预检、目标参数校验 | 让程序知道当前运行在哪个 GPU 平台，以及哪些能力可以安全开放 |
| CLI 与启动 | [`cli/__init__.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/cli/__init__.py)、`cli/parser.py`、`cli/preflight.py`、`cli/bootstrap.py` | 解析目标和 profiling 参数，执行预检，发布平台环境，恢复会话 | 在 Coordinator 启动前确定平台、资源和执行能力 |
| 基准后端 | [`benchmark_backend.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/benchmark_backend.py)、[`vllm_cuda_runner.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/vllm_cuda_runner.py) | 新增 `vllm_cuda` 后端和原生 CUDA runner | 不依赖 Magpie/ROCm 脚本，直接启动 CUDA 环境中的服务和 benchmark |
| 结果与质量 | `benchmark_result.py`、`vllm_cuda_runner.py`、`breakdown/exporter.py` | 统一吞吐、延迟、请求、质量和产物字段 | 让 CUDA 后端的结果能被原有基线、候选和报告流程读取 |
| GPU 资源 | `bus/gpu_pool.py`、`bus/storage/schema.py`、`cuda_host_lock.py`、`specialists/subprocess_.py` | GPU 租约、UUID/NUMA 信息、主机互斥和子进程设备隔离 | 避免任务抢卡、错卡、残留服务或误杀其他进程 |
| CUDA 采集 | `cuda_profile.py`、`cuda_profiler_launch.py`、`cuda_nsight.py`、`cuda_roofline.py` | torch profiler、Nsight Systems、Nsight Compute 和热点 roofline | 找到 CUDA 执行热点，并把 profile 作为诊断证据保存 |
| 调度与能力 | `action_surfaces.py`、`policy/gate.py`、`cli/executors.py`、`kernel/request_handlers.py` | 按目标能力过滤动作和执行任务 | 防止 agent 提出尚未实现的 CUDA 源码或 kernel 任务 |
| 会话状态 | `state/shared_state.py`、`session/manifest.py` | 保存 target、设备、profile 工具、质量协议和并行形状 | 中断后恢复同一实验条件，防止混用不可比较的结果 |
| 依赖与文档 | `pyproject.toml`、`assets/install.sh`、README、适配报告 | NVIDIA 依赖组、安装分流、使用说明和实验隔离 | 让 CUDA 部署不拉入不需要的 ROCm/KernelForge 依赖 |

## 3. 平台目标与设备识别

### 3.1 初始实现：增加独立 NVIDIA target

新增 [`TargetDescriptor`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/target_registry.py:52) 和 [`TargetCapabilities`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/target_registry.py:32)，把“运行平台”和“支持哪些动作”从 AMD 的 `--gpu-type` 中分离出来。初始版本注册了一个 NVIDIA 目标，并使用目标能力表关闭源码、kernel、量化和多机路径。

初始实现还包含以下固定条件：

- 指定 GPU 型号、卡数和计算能力。
- 指定最小显存。
- 指定 CUDA 安装目录。
- 将 vLLM CLI 所需参数列在目标描述中。

这些字段最初用于本地受控环境的 fail-closed 预检：条件不满足就拒绝启动，避免错误地把 AMD 参数或不兼容的 CUDA 环境送入执行器。

### 3.2 通用化实现：目标描述不再绑定硬件型号

当前工作区把规范目标改为 `nvidia_cuda`，旧目标名保留为别名。目标描述只保留 NVIDIA/CUDA 平台和能力，不再保存“必须是某型号、必须有多少张卡、必须是某个 CUDA 版本”等白名单条件。

[`validate_nvidia_host()`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/target_registry.py:351) 现在负责发现实际设备池：

1. 使用 NVML 读取物理设备 UUID、型号、显存、计算能力、PCI 地址和 NUMA 节点。
2. 在独立 CUDA driver 进程中读取 CUDA 自己的设备枚举顺序。
3. 通过 UUID 对齐 NVML 编号和 CUDA ordinal，记录物理、CUDA 和进程逻辑编号。
4. 读取驱动版本、CUDA driver API 版本和可选 toolkit 信息，生成硬件指纹。
5. 把设备池写入环境变量和会话状态，供 runner、GPU 池、profiling 和恢复逻辑共同使用。

这样做是因为 `CUDA_VISIBLE_DEVICES` 会改变 CUDA 应用看到的设备及其枚举顺序。程序不能假设 CUDA ordinal 与 NVML index 永远相同，也不能用主机总卡数代替本次任务实际可用卡数。

### 3.3 资源池和任务分配

[`allocate_cuda_devices()`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/target_registry.py:323) 建立了“发现的设备 → 进程允许的设备池 → 任务分配”关系：

- 支持数字 ordinal 和 GPU UUID 形式的 `CUDA_VISIBLE_DEVICES`。
- 保留用户指定的设备顺序，拒绝重复、非法、空池和越界选择。
- `--gpus-per-node` 只能缩小已发现的设备池，不能扩展设备范围。
- 任务所需设备数量由 `TP × PP` 校验。
- 混合计算架构的并行组当前明确拒绝，避免把不同架构的局部结果混为同一组实验。

`_workload_envs.py`、`vllm_cuda_runner.py`、`gpu_pool.py` 和 Specialist 子进程启动逻辑都使用这份映射。服务进程通过 UUID 接收设备，报告同时保存物理 index、CUDA index 和逻辑 index。

## 4. CLI、预检和会话生命周期

### 4.1 CLI 参数与启动顺序

[`cli/parser.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/cli/parser.py:270) 增加了目标、优化层级、profile backend、pipeline parallel 和请求数量等参数。启动时由 [`cli/__init__.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/cli/__init__.py:1978) 按以下顺序处理：

```text
CLI / 环境变量 / 已保存会话
  → 解析 nvidia_cuda 或兼容别名
  → 发现设备池并生成硬件指纹
  → 由具体 benchmark backend 检查框架运行环境
  → 检查 profile 工具（仅 profile 请求需要）
  → 发布 CUDA 环境并创建/恢复会话
```

平台层不再检查 vLLM 专用 CLI；这类检查移到 [`vllm_cuda_preflight.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/vllm_cuda_preflight.py)，并使用真正启动 runner 的 Python 解释器执行。

### 4.2 会话与恢复

[`SharedState`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/state/shared_state.py:530) 增加了运行平台、设备池、目标能力、profile 工具和工作负载字段，状态 schema 从 v6/v9 继续升级到 v10。恢复时：

- 兼容旧目标名并规范化为 `nvidia_cuda`。
- 检查保存的 GPU UUID、型号、计算能力、显存、PCI/NUMA 信息。
- 检查驱动、toolkit 和执行环境身份。
- 检查设备顺序、profile backend、roofline 设置和工具版本。
- 缺少关键身份信息或发现环境变化时，要求新建会话，不继续比较旧基线。

### 4.3 生命周期和清理

[`cuda_host_lock.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/cuda_host_lock.py:1) 增加主机级互斥；runner 还会记录服务进程组、GPU 租约、端口和工作目录。启动失败、超时、取消和父进程退出都进入统一清理路径，只回收本次任务创建的资源。旧锁名在过渡期间仍被同时锁定，避免新旧进程并行运行。

## 5. CUDA 基准执行和结果接口

### 5.1 `vllm_cuda` 后端

[`benchmark_backend.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/benchmark_backend.py:184) 新增 `VllmCudaBackend`。它遵守公共 benchmark backend 协议，但命令改为启动：

```text
python -m hyperloom.orchestrator.actions.executors.vllm_cuda_runner benchmark ...
```

因此，Magpie 仍是默认公共后端，CUDA 任务可以使用原生 runner；没有实现的平台/框架组合会明确报错，不会静默回退到 AMD/Magpie 路径。

### 5.2 runner 完成的工作

[`vllm_cuda_runner.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/vllm_cuda_runner.py:954) 负责一个完整的单机服务生命周期：

- 读取物化 YAML，校验模型、TP/PP、设备池和额外 vLLM 参数。
- 创建工作目录和启动计划，保存命令、环境和指纹。
- 启动 vLLM server，等待就绪，执行预热和正式请求。
- 可选执行质量 smoke 或语义用例。
- 停止服务，释放 GPU 租约，采集清理状态。
- 保存原始 vLLM JSON、统一结果、兼容 `benchmark_report.json` 和日志。

结果归一化函数 [`normalize_vllm_result()`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/vllm_cuda_runner.py:854) 明确保存：总吞吐、每卡吞吐、请求吞吐、TTFT/TPOT/ITL/E2EL 分位数、完成/失败请求数、质量状态、设备拓扑、原始产物和清理状态。公共解析器 [`benchmark_result.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/benchmark_result.py:739) 再把这些字段转换成原有流程认识的测量对象。

### 5.3 为什么要保留两套吞吐字段

多卡总吞吐与每卡吞吐回答的问题不同。runner 同时保存二者，兼容报告按当前公共口径提供主吞吐字段，并保留总量和每卡值。这样可以避免改变卡数后把总吞吐误当成可比的单卡收益。

质量检查也进入结果状态：服务返回 HTTP 200 或非空文本，只能说明请求成功，不能替代模型质量检查。profile 结果会标记 `measurement_kind=profile` 和 `valid_measurement=false`，不会更新 baseline 或性能赢家。

## 6. CUDA profiling 和 Nsight 分析

### 6.1 torch profiler

[`cuda_profile.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/cuda_profile.py:57) 为 profiling 建立独立服务生命周期：先预热，再在有限请求窗口内采集，最后停止服务并检查每个 rank 的 trace。profile 运行的吞吐仅作为诊断数据保存，因为 profiler 本身会改变运行开销。

### 6.2 Nsight Systems 与 Nsight Compute

[`cuda_nsight.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/cuda_nsight.py:297) 将两个工具串成有界流程：

1. Nsight Systems 采集 CUDA/NVTX 时间线，识别计算、通信、拷贝、等待和热点 kernel。
2. 从时间线选出非通信热点。
3. Nsight Compute 只对选中的 kernel、进程、rank 和设备采集计数器。
4. 将 CSV、时间线、rank/GPU 身份和计数器状态写入分析报告。

当前工作区增加了按实际 GPU 查询指标的逻辑：不同架构或工具版本缺失某项计数器时，保留可用指标和缺失原因，不把缺失指标填成零。

### 6.3 多进程和设备身份

采集结果需要同时匹配 `worker PID → rank → GPU UUID → trace/kernel sample`。`cuda_profiler_launch.py` 管理 worker 的启动和清理；Nsight 分析在发现 PCI 身份、rank 覆盖或 kernel launch shape 不一致时拒绝生成确定性结论。

[`cuda_roofline.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/cuda_roofline.py:20) 负责先时间线、后热点计数器的复合任务。即使计数器阶段失败，也可以保留有效时间线；该失败不会伪造 roofline，也不会进入性能采纳。

## 7. 调度、能力限制和优化决策

### 7.1 能力表控制动作

目标能力通过 [`action_surfaces.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/protocol/action_surfaces.py:145) 进入 agent 动作目录，并由 [`policy/gate.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/policy/gate.py:870) 在执行侧再次检查。初始 CUDA 目标开放基线、配置探索、sweep、报告和 opt-in profile；源码补丁、kernel patch、量化、多机和未实现的评测路径关闭。

这解决的是“提示词说可以做”与“工程实际不能做”之间的不一致：未接通的动作会在提议或派发时被拒绝，而不是运行到中途才失败。

### 7.2 roofline 与 kernel 结果的边界

`roofline_snapshot.py` 和 `roofline_ceiling.py` 增加 CUDA 分支，明确 selected kernel 的分析不等于整模型性能上限。局部 kernel 变快后，调度、通信、数据转换和其他算子仍可能决定端到端表现。因此 kernel 或分析结果只有在实际集成框架并完成普通端到端复测后，才可能进入原有采纳流程。

## 8. 依赖、安装和实验隔离

### 8.1 安装依赖

[`pyproject.toml`](/data/ygw/llm_sim/Hyperloom/pyproject.toml:27) 增加 NVIDIA extra，包含 CUDA 运行所需的 `nvidia-ml-py` 等依赖。`assets/install.sh` 根据 benchmark backend 分流：CUDA 路径不安装或初始化不需要的 Magpie、InferenceX、rocprof 和 KernelForge 链路。

### 8.2 文档与独立实验

README、环境变量说明、认证说明和 NVIDIA 适配计划补充 CUDA 启动方式、profile 选项、资源规则和恢复约定。Qwen3 等硬件实验被放到 `experiments/nvidia_qwen3/` 和独立报告中，与安装后的通用优化器代码分离，避免实验脚本成为平台能力的隐式依赖。

## 9. 测试变化

新增或扩展的测试覆盖以下行为：

- NVIDIA target 解析、能力过滤、环境预检和旧状态迁移。
- GPU UUID、NVML/CUDA ordinal 映射、设备子集、重排、资源不足和重复选择。
- runner 启动、结果归一化、质量检查、租约、进程清理和中断恢复。
- torch profile、Nsight 时间线、NCU 指标、rank/GPU 身份和失败状态。
- profile 不进入性能测量、AMD 路径不被 CUDA 改动破坏。

当前工作区的 NVIDIA 相关回归测试已通过；本机实际验证覆盖单卡可见的普通基线、配置候选、torch profile 及 NCU 指标查询。其他 GPU 型号和卡数主要由模拟测试覆盖，仍需在对应硬件上分别验收。

## 10. 当前能力边界

已经具备的主要能力是：

- 发现并记录 NVIDIA GPU 与 CUDA 环境。
- 在可见设备子集上分配 GPU，并通过 UUID 隔离子进程。
- 启动原生 CUDA benchmark，输出公共可读结果。
- 采集 torch/Nsight 诊断数据并关联到 rank 和 GPU。
- 保存状态，拒绝不可比较的环境恢复。

尚未作为通用正式能力开放的部分是：

- 多机 CUDA 资源、通信和故障恢复。
- MIG 分区调度。
- 跨计算架构混合并行。
- CUDA 框架源码补丁的完整构建、加载、正确性和回滚闭环。
- CUDA/Triton kernel 自动生成、编译、集成和端到端采纳。
- 通用量化和完整模型评测组合。

因此，“增加了 NVIDIA 支持”应理解为 Hyperloom 已经拥有一条可执行的 CUDA 平台接线和测量/分析闭环；它不表示所有 NVIDIA 型号、所有推理框架组合或所有 kernel 优化能力都已经完成硬件验收。

## 相关文档

- [NVIDIA/CUDA 平台适配方案](/data/ygw/llm_sim/Hyperloom_NVIDIA_vLLM适配计划.md)
- [NVIDIA 平台通用化验证记录](/data/ygw/llm_sim/Hyperloom/docs/nvidia-platform-generalization-validation.md)
- [NVIDIA 初始实施记录](/data/ygw/llm_sim/Hyperloom/docs/nvidia-integration-plan.md)
- [独立 NVIDIA/vLLM 实验报告](/data/ygw/llm_sim/Hyperloom/docs/nvidia-vllm-integration-report.md)

## 附录 A：相对原工程的完整文件清单

下面的清单由 `git diff --name-status 4aa00d425..HEAD` 生成，覆盖 NVIDIA 初始支持提交链中的全部 89 个文件。`A` 表示新增，`M` 表示在原文件上修改。实验、测试和文档文件也列在这里，但它们不等于运行时能力。

### 新增文件（A）

```text
.github/workflows/nvidia-experiments.yml
docs/nvidia-integration-plan.md
examples/hyperloom-qwen3-8b-nvidia-3h/SKILL.md
examples/hyperloom-qwen3-8b-nvidia-3h/run_example.sh
experiments/__init__.py
experiments/nvidia_qwen3/README.md
experiments/nvidia_qwen3/__init__.py
experiments/nvidia_qwen3/__main__.py
experiments/nvidia_qwen3/baseline.py
experiments/nvidia_qwen3/cli.py
experiments/nvidia_qwen3/config_search.py
experiments/nvidia_qwen3/process_guard.py
experiments/nvidia_qwen3/requirements.txt
experiments/nvidia_qwen3/service.py
experiments/nvidia_qwen3/stability.py
experiments/nvidia_qwen3/tests/__init__.py
experiments/nvidia_qwen3/tests/test_baseline.py
experiments/nvidia_qwen3/tests/test_config_search.py
experiments/nvidia_qwen3/tests/test_service.py
experiments/nvidia_qwen3/tests/test_stability.py
src/hyperloom/inference_optimizer/target_registry.py
src/hyperloom/inference_optimizer/tests/test_cli_experiment_isolation.py
src/hyperloom/inference_optimizer/tests/test_cuda_nsight.py
src/hyperloom/inference_optimizer/tests/test_cuda_profile.py
src/hyperloom/inference_optimizer/tests/test_install_nvidia_target.py
src/hyperloom/inference_optimizer/tests/test_nvidia_target.py
src/hyperloom/inference_optimizer/tests/test_vllm_cuda_runner.py
src/hyperloom/orchestrator/actions/executors/cuda_host_lock.py
src/hyperloom/orchestrator/actions/executors/cuda_nsight.py
src/hyperloom/orchestrator/actions/executors/cuda_profile.py
src/hyperloom/orchestrator/actions/executors/cuda_profiler_launch.py
src/hyperloom/orchestrator/actions/executors/cuda_roofline.py
src/hyperloom/orchestrator/actions/executors/vllm_cuda_runner.py
```

### 修改文件（M）

```text
docs/reference/authentication.md
docs/reference/environment-variables.md
examples/README.md
pyproject.toml
scripts/check_wheel_contents.py
src/hyperloom/agents/kernel/tests/test_codex_session.py
src/hyperloom/agents/robustness/role/envelope.py
src/hyperloom/common/codex_session.py
src/hyperloom/inference_optimizer/README.md
src/hyperloom/inference_optimizer/assets/install.sh
src/hyperloom/inference_optimizer/breakdown/exporter.py
src/hyperloom/inference_optimizer/cli/__init__.py
src/hyperloom/inference_optimizer/cli/backends.py
src/hyperloom/inference_optimizer/cli/bootstrap.py
src/hyperloom/inference_optimizer/cli/credentials.py
src/hyperloom/inference_optimizer/cli/executors.py
src/hyperloom/inference_optimizer/cli/kb.py
src/hyperloom/inference_optimizer/cli/parser.py
src/hyperloom/inference_optimizer/cli/preflight.py
src/hyperloom/inference_optimizer/protocol/action_surfaces.py
src/hyperloom/inference_optimizer/session/manifest.py
src/hyperloom/inference_optimizer/tests/test_backend_gating.py
src/hyperloom/inference_optimizer/tests/test_breakdown_exporter_unit.py
src/hyperloom/inference_optimizer/tests/test_cli_backends_unit.py
src/hyperloom/inference_optimizer/tests/test_cli_resume_launch_shape.py
src/hyperloom/inference_optimizer/tests/test_cli_workload_envs.py
src/hyperloom/inference_optimizer/tests/test_critic_agent_backend.py
src/hyperloom/inference_optimizer/tests/test_gpu_pool_device_resolution.py
src/hyperloom/inference_optimizer/tests/test_manifest_unit.py
src/hyperloom/inference_optimizer/tests/test_preflight_auth_override.py
src/hyperloom/inference_optimizer/tests/test_resource_lanes.py
src/hyperloom/inference_optimizer/tests/test_shared_state_units.py
src/hyperloom/inference_optimizer/tests/test_specialist_codex_backend.py
src/hyperloom/inference_optimizer/tests/test_workload_envs.py
src/hyperloom/inference_optimizer/tests/test_workload_envs_golden_lock.py
src/hyperloom/orchestrator/actions/executors/_server_lifecycle.py
src/hyperloom/orchestrator/actions/executors/_workload_envs.py
src/hyperloom/orchestrator/actions/executors/baseline.py
src/hyperloom/orchestrator/actions/executors/benchmark_backend.py
src/hyperloom/orchestrator/actions/executors/benchmark_result.py
src/hyperloom/orchestrator/actions/executors/bypass_engine.py
src/hyperloom/orchestrator/actions/executors/bypass_report.py
src/hyperloom/orchestrator/actions/executors/report.py
src/hyperloom/orchestrator/bus/gpu_pool.py
src/hyperloom/orchestrator/bus/storage/schema.py
src/hyperloom/orchestrator/kernel/request_handlers.py
src/hyperloom/orchestrator/kernel/roofline_ceiling.py
src/hyperloom/orchestrator/kernel/roofline_snapshot.py
src/hyperloom/orchestrator/loop/dispatcher.py
src/hyperloom/orchestrator/loop/writeback.py
src/hyperloom/orchestrator/phases/prelude.py
src/hyperloom/orchestrator/policy/gate.py
src/hyperloom/orchestrator/prompts/prompt_builder.py
src/hyperloom/orchestrator/roles/critic_agent.py
src/hyperloom/orchestrator/specialists/subprocess_.py
src/hyperloom/orchestrator/state/shared_state.py
```

## 附录 B：本次同步完成的通用化改造

这些文件相对 NVIDIA 初始支持版本的改动，目的在于去除“8 张 RTX 4090、固定 CUDA 版本”的假设；本次同步已将它们纳入远端分支：

```text
docs/nvidia-integration-plan.md
docs/nvidia-platform-generalization-validation.md
src/hyperloom/inference_optimizer/README.md
src/hyperloom/inference_optimizer/cli/__init__.py
src/hyperloom/inference_optimizer/cli/bootstrap.py
src/hyperloom/inference_optimizer/cli/executors.py
src/hyperloom/inference_optimizer/cli/parser.py
src/hyperloom/inference_optimizer/cli/preflight.py
src/hyperloom/inference_optimizer/target_registry.py
src/hyperloom/inference_optimizer/tests/test_cuda_nsight.py
src/hyperloom/inference_optimizer/tests/test_cuda_platform.py
src/hyperloom/inference_optimizer/tests/test_cuda_profile.py
src/hyperloom/inference_optimizer/tests/test_nvidia_target.py
src/hyperloom/inference_optimizer/tests/test_vllm_cuda_runner.py
src/hyperloom/orchestrator/actions/executors/_workload_envs.py
src/hyperloom/orchestrator/actions/executors/benchmark_backend.py
src/hyperloom/orchestrator/actions/executors/cuda_host_lock.py
src/hyperloom/orchestrator/actions/executors/cuda_nsight.py
src/hyperloom/orchestrator/actions/executors/cuda_profile.py
src/hyperloom/orchestrator/actions/executors/cuda_profiler_launch.py
src/hyperloom/orchestrator/actions/executors/cuda_roofline.py
src/hyperloom/orchestrator/actions/executors/vllm_cuda_preflight.py
src/hyperloom/orchestrator/actions/executors/vllm_cuda_runner.py
src/hyperloom/orchestrator/bus/gpu_pool.py
src/hyperloom/orchestrator/kernel/request_handlers.py
src/hyperloom/orchestrator/kernel/roofline_ceiling.py
src/hyperloom/orchestrator/kernel/roofline_snapshot.py
src/hyperloom/orchestrator/policy/gate.py
src/hyperloom/orchestrator/specialists/subprocess_.py
src/hyperloom/orchestrator/state/shared_state.py
```

附录 B 与附录 A 共同构成当前 NVIDIA 支持实现；其他 GPU 型号、多机、MIG 和跨架构并行仍需在对应硬件上单独验收。
