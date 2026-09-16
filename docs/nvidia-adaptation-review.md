# NVIDIA/CUDA 适配实现审查与改进建议

更新日期：2026-09-16

本文审查 Hyperloom 已提交的 NVIDIA/CUDA 代码，重点判断当前方案是否足够通用、实现之间是否一致，以及下一步应怎样收敛成可维护的平台适配层。

## 1. 审查范围和当前状态

审查基线为提交 `039f55535`（`feat(nvidia): add native Nsight profiling and selected-kernel roofline`），并参考 NVIDIA 支持开始前的 `4aa00d425`。

本文中的问题清单来自该审查时点。随后已完成 `upstream/main` 合并，解决 CLI、执行器、状态和 profiling 相关冲突，并通过编译、差异检查和 NVIDIA 专项回归；当前版本为 `f86eb6caf`，不存在未解决的 Git 冲突标记。

## 2. 结论

当前方案已经实现了一条“固定本机环境的 CUDA/vLLM 测量和 profiling 链路”，但还不是通用 NVIDIA 平台适配。主要原因是平台层仍绑定了具体硬件和软件，设备可见性只在 runner 中做了很薄的整数解析，平台描述还包含 vLLM 后端字段。

更合适的方向是把适配拆成四个独立接口：

```text
Platform（NVIDIA/CUDA）
  → Device inventory（UUID、ordinal、显存、架构、拓扑）
  → Execution backend（vLLM、其他框架）
  → Measurement/Profiler（公共结果和可选采集）
```

这样可以复用 Hyperloom 的 Coordinator、agent、intent、候选调度、状态和报告逻辑，同时让“能否在这台 NVIDIA 机器上执行”由实际设备和后端能力决定。

## 3. 必须优先修复的问题

### P0：完成合并并恢复可运行状态（已完成）

审查时工作区存在未解决冲突标记，涉及 CLI、preflight、执行器、状态和 kernel 分析等关键模块。当前已完成：

1. `python -m compileall src/hyperloom`，确保没有冲突标记和语法错误。
2. Hyperloom 原有 AMD 回归测试。
3. NVIDIA target、runner、profile、Nsight 和恢复测试。
4. `git diff --check` 及打包导入检查。

编译、冲突标记检查和 `git diff --check` 均通过；CUDA/NVIDIA 专项测试在当前环境中可运行的 111 项全部通过。

### P1：目标描述仍然绑定 8 张 RTX 4090

[`target_registry.py`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/target_registry.py:52) 的 `TargetDescriptor` 包含 `expected_gpu_name`、`expected_gpu_count`、`expected_compute_capability`、`min_memory_mib` 和 `default_cuda_home`；注册的唯一 NVIDIA 目标是 `nvidia_rtx4090_8x_local`。[`validate_nvidia_host()`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/target_registry.py:368) 会拒绝卡数、型号、计算能力或显存不同的主机，并强制检查 CUDA 13.0 的 `nvcc`。

这会直接拒绝 A10、A100、H100、L40S、RTX 6000、单卡容器和其他合法 CUDA 环境。建议：

- 新增规范目标 `nvidia_cuda`，旧名称只作为兼容别名。
- `TargetDescriptor` 只描述 vendor/runtime/capabilities，不保存具体型号白名单。
- 设备型号、显存、计算能力和卡数进入运行时 inventory；任务根据实际能力选择或拒绝。
- 普通 serving/benchmark 不要求 `nvcc`；只有 CUDA 扩展或 kernel 编译任务声明编译器依赖时才检查 toolkit。

### P1：平台层和 vLLM 后端耦合

同一个 `TargetDescriptor` 同时保存 `benchmark_backend`、vLLM CLI flags 和 CUDA 路径（见上述文件第 59–67 行）。这使得“有 NVIDIA GPU”被错误地等同于“必须安装某个 vLLM CLI”。原工程已经支持 vLLM，平台适配不应重新定义框架协议。

建议使用组合解析：

```text
resolve_platform()  -> nvidia_cuda
resolve_framework() -> vllm / sglang / ...
resolve_backend(platform, framework) -> vllm_cuda / 其他后端
```

框架专用 CLI 检查放入后端自己的 preflight；平台预检只检查驱动、CUDA driver、设备和请求的公共工具。

### P1：CUDA 可见设备和进程编号不完整

当前 runner 的 [`_visible_indices()`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/orchestrator/actions/executors/vllm_cuda_runner.py:319) 只接受整数；空值时默认使用 `range(world_size)`。它不支持 GPU UUID，不保留 UUID mask 的重排语义，也没有把 CUDA ordinal 与 NVML index 明确对齐。平台 preflight 还按整机 8 卡和 torch 的设备数校验，因此容器中只暴露部分 GPU 时会在启动前失败。

更稳妥的实现应保存三层映射：

| 层 | 内容 | 用途 |
|---|---|---|
| 物理设备 | NVML UUID、PCI、NUMA、型号、显存、计算能力 | 稳定身份和资源租约 |
| CUDA 枚举 | CUDA ordinal 与 UUID | 解释 `CUDA_VISIBLE_DEVICES` 及框架编号 |
| 本次任务 | 分配顺序、逻辑 index、rank | 子进程、TP/PP、profile 和报告 |

解析流程必须支持数字和 UUID mask，保留用户顺序；空 mask 必须表示无设备，不能回退为全机；`TP × PP` 超过分配池时明确失败。服务、GPU 租约、显存采样、rank 记录和 Nsight 解析全部使用同一映射。

### P1：预检检查范围过大且解释器可能不一致

当前 [`validate_nvidia_host()`](/data/ygw/llm_sim/Hyperloom/src/hyperloom/inference_optimizer/target_registry.py:368) 在平台预检中直接 import torch、查询 vLLM 版本、探测 vLLM CLI，并使用当前 Python 进程的 torch 设备数。实际 runner 可能使用另一个 benchmark Python 或不同 `PYTHONPATH`。

建议平台预检返回“设备和驱动能力”，再由实际 backend 使用最终解释器执行：

- `python -c` 检查 torch/vLLM import、版本和 CUDA kernel smoke。
- 用同一个解释器探测 `serve`/`bench` CLI flags。
- 把 `runtime_python`、`PYTHONPATH`、框架版本和 CLI surface 写入执行指纹。
- profiler、Nsight、`nvcc` 等可选工具只在请求相应能力时检查。

## 4. 已有实现中值得保留的部分

以下设计方向是正确的，应在重构时保留：

- `TargetCapabilities` 通过 action surface 和 `PolicyGate` 双重限制未实现能力。
- vLLM runner 有独立的服务启动、就绪等待、请求、停止和清理流程。
- GPU lease、主机锁、服务进程组和异常清理被纳入生命周期。
- benchmark 结果保存总吞吐、请求统计、延迟和原始产物，并有兼容报告。
- profiling 吞吐不应更新 baseline/current best/KEEP。
- Nsight 采用“先时间线、后热点计数器”的顺序，并保留原始报告。
- 状态中保存硬件、模型、工作负载和工具指纹，恢复时拒绝不可比较环境。

这些是通用优化闭环的公共能力，不应因为 NVIDIA 通用化而删除。

## 5. 对通用化尝试（`ee5baa50d`）的审查

这次本地通用化提交已经解决了原始 MVP 的几个核心问题：增加 `nvidia_cuda` 规范名称和旧别名、移除目标描述中的固定型号字段、用 NVML 与 CUDA driver 的 UUID 对齐设备、支持可见设备重排、将 vLLM 检查移到 `vllm_cuda_preflight.py`，并让 Nsight 按实际工具查询指标。这些方向是正确的。

但提交完整性本身还有缺口：`ee5baa50d` 中的测试和 CLI 已经 import `vllm_cuda_preflight`，而该模块没有进入该提交的 Git tree；隔离工作树运行专项测试会在收集阶段直接报 `ModuleNotFoundError`。通用化提交还依赖未提交的测试文件。因此，合入时必须把运行模块、测试和文档作为同一变更提交，不能只合并已修改的文件。

仍需要修正以下细节后再合入：

| 优先级 | 位置 | 风险 | 建议 |
|---|---|---|---|
| 高 | `target_registry.py:251–264` | `CUDA_HOME` 显式指向不存在目录时仍可能写入环境；普通 serving 不需要 toolkit，却会把无效路径传播给子进程 | 只有目录真实存在时设置 `CUDA_HOME`；toolkit 缺失作为编译能力缺失记录，不阻断普通 benchmark |
| 高 | `target_registry.py:380–389` | 恢复指纹只比较设备和少量 toolkit 字段，后端/解释器/框架身份依赖另一条路径，容易出现两条校验不一致 | 定义统一 `PlatformIdentity` 和 `BackendIdentity`，由一次恢复检查同时比较，并把设备顺序、mask 和分配池纳入身份 |
| 高 | `_workload_envs.py`、`vllm_cuda_runner.py` | 环境变量、YAML 候选和会话池都能表达设备选择，多个入口容易产生不同优先级 | 只允许平台分配器生成最终 `CUDA_VISIBLE_DEVICES`；候选只能请求池内 UUID，禁止直接覆盖最终 mask |
| 中 | `cuda_nsight.py` | 指标查询失败与没有匹配 kernel 的失败类型需要保持可区分，否则报告难以判断是工具问题还是 workload 问题 | 统一错误码和 `counter_status`，保留有效时间线，并将不可用指标列为 unavailable |
| 中 | `vllm_cuda_runner.py` | runner 同时承担配置解析、设备分配、租约、服务生命周期、质量和结果转换，文件过大 | 先抽出 `CudaAssignment`、`ServerLifecycle`、`ResultNormalizer`、`QualityGate` 四个模块，保持现有公共结果协议 |

通用化提交中的测试主要是模拟 NVML、CUDA 枚举和子进程；它们能验证分支逻辑，但不能替代不同架构实机验收。应在合并完成后分别记录模拟覆盖和实际 GPU 组合，避免把单一 RTX 4090 结果写成通用支持声明。

## 6. 推荐的目标实现

### 5.1 平台接口

新增一个平台模块，提供以下最小接口：

```python
discover() -> PlatformInventory
validate(inventory, request) -> ValidatedDevicePool
allocate(pool, parallel_shape) -> DeviceAssignment
environment(assignment) -> dict[str, str]
fingerprint(inventory, assignment, tools) -> dict[str, Any]
```

`PlatformInventory` 只包含 NVIDIA/CUDA 信息；框架版本和 benchmark flags 不放进平台对象。

### 5.2 后端接口

后端根据平台和框架组合注册：

```python
preflight(runtime_python, workload) -> BackendCapabilities
build_command(config, assignment) -> list[str]
normalize_result(raw_result) -> PublicBenchmarkResult
cleanup(ownership) -> CleanupResult
```

现有 `vllm_cuda_runner.py` 可以作为第一个后端实现，但应逐步拆出设备分配、服务生命周期、结果归一化和质量检查，避免一个 1400 行文件同时承担所有职责。

### 5.3 profiling 接口

将 torch profiler、Nsight Systems、Nsight Compute 视为可选采集器：

1. 普通 benchmark 产生可采纳测量。
2. profile benchmark 使用独立窗口并标记 `measurement_kind=profile`。
3. Nsight 指标按实际 GPU 和已安装工具查询，不假定所有架构都有同样计数器。
4. 计数器缺失时保留时间线和缺失原因，不填零、不生成伪造 roofline。

### 5.4 状态和恢复

状态至少保存：规范平台名、后端名、物理 UUID 顺序、分配顺序、TP/PP、模型和 workload 指纹、Python/框架版本、profile 工具路径和版本。旧目标名可以迁移为规范名，但缺少 UUID 或关键环境身份时必须拒绝恢复。

## 7. 测试矩阵

通用化完成条件应覆盖：

- 单卡、双卡、四卡、八卡和混合型号主机。
- 数字 mask、UUID mask、重排 mask、部分可见设备和空设备池。
- `TP × PP` 资源不足、重复设备、非法 UUID 和跨架构并行。
- 不同 CUDA driver/toolkit，普通测量无 `nvcc`，profile 工具缺失。
- vLLM backend 缺失或 CLI 不兼容时明确报错；不回退到 Magpie/AMD 路径。
- runner 启动失败、超时、中断、服务复用和租约清理。
- profile 不更新性能赢家，AMD 原有路径保持不变。
- 旧会话迁移、设备变化检测和工具变化检测。

## 8. 推进顺序

1. 完成当前合并并恢复全套测试。
2. 把 `TargetDescriptor` 改成平台描述，加入 `nvidia_cuda` 和旧别名迁移。
3. 实现 NVML/CUDA UUID 映射和任务设备池，先接普通 benchmark。
4. 将 vLLM 预检移到后端，验证实际解释器和公共结果字段。
5. 接通 torch profiler，再接 Nsight Systems/Compute；保持 profile 与性能测量隔离。
6. 补齐状态恢复和测试矩阵。
7. 最后再评估 CUDA 框架源码、Triton/CUDA kernel、量化和多机能力；未完成前继续关闭能力开关。

## 9. 最终判断

当前实现适合作为“固定 8×RTX 4090 环境的 NVIDIA/vLLM MVP”，不适合作为通用 NVIDIA GPU 支持。最关键的改进不是增加更多 GPU 型号枚举，而是建立基于运行时能力的设备池，并将平台、框架后端和 profiling 工具解耦。完成合并清理、平台接口重构和设备 UUID 映射后，现有 Coordinator、agent、intent、候选调度和报告流程才能真正复用到不同 NVIDIA GPU。

## 10. 本次审查的验证结果

- 在隔离工作树中对通用化提交执行 `python -m compileall`，通过。
- `test_nvidia_target.py` 和 `test_vllm_cuda_runner.py` 的非异步用例：43 passed。
- profile/Nsight 专项测试在收集阶段发现 `vllm_cuda_preflight` 未随 `ee5baa50d` 提交进入 Git tree，报 `ModuleNotFoundError`；这验证了上文所述的提交完整性问题。
- 合并后的 `git diff --check` 和冲突标记检查均通过；完整测试收集仅受环境缺少 `hypothesis`、`coverage` 依赖影响。
