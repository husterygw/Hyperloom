# NVIDIA/vLLM 后续适配计划与首批验收

更新日期：2026-09-08。范围固定为本机 8×RTX 4090、CUDA 13.0、现有
`llm_sim` 环境中的 vLLM；暂不扩展其他 GPU、多机或其他 serving framework。

本轮实施边界是阶段 A、B 和 profile/roofline 能力拆分。阶段 C–H 是后续工作，
不能通过打开 capability 标志替代实现和实机验收。

## 完整路线

| 阶段 | 工作 | 验收与依赖 |
|---|---|---|
| A：当前工程回归 | 实验代码隔离；Qwen2.5-3B 优化 smoke；Qwen3-8B 示例；中断/恢复；清理和报告 | 真实服务与代理执行；保留 baseline、workload、target；记录真实停止原因；进程和租约释放 |
| B：torch profiling | 原生 vLLM profiler；单卡、Qwen3-32B PP8 和 TP2/PP4；独立 profile/roofline/trace_analysis 能力 | 每个 rank 有新生成的 CUDA kernel trace；启动/停止确认；profile 吞吐不可成为 baseline/KEEP |
| C：Nsight 分析 | `nsys` 时间线和瓶颈分析；`ncu` 计数器及 roofline；结果映射到模型和 rank | 先验证工具权限和实际采集，再开放对应 backend/capability；采集性能不参与晋升 |
| D：基于测量的配置优化 | 依据 trace 选择调度、CUDA Graph、编译、attention、KV/cache 参数 | 固定 W0–W5 协议，按下面的联合指标裁决；不以 eager baseline 制造收益 |
| E：vLLM 源码优化 | 独立 worktree 和运行环境；确保实际导入修改后的代码；补丁、质量和回退记录 | 源码路径/内容证明、功能检查、端到端复测通过后才 KEEP |
| F：NVIDIA kernel 优化 | KernelForge CUDA backend；先 Triton softmax，再 RMSNorm/fused-add、GEMM；隔离 CUDA C++ 构建 | 数值验证、microbenchmark、端到端收益同时通过；GPU 架构与工具链写入构建身份 |
| G：量化 | 隔离环境中 llmcompressor GPTQ W4A16/compressed-tensors，排除 lm_head | 固定校准集和评估集；任务分数下降 ≤1 个百分点，困惑度增长 ≤5%；再评估吞吐和尾延迟 |
| H：组合回放与回归 | 组合已接受的配置、源码、kernel、量化结果；稳定性和故障恢复 | 完整基线/候选三次重放、资源清理、报告和可复现启动配置；按次拆分长时间 soak |

阶段 E/F/G 需要进一步拆分 source/kernel、GEMM tuning、evaluation、warm replay
能力。当前 CLI 对 `--optimization-level source/kernel`、`--profile-backend nsys`
明确报未实现；AMD 的原有默认能力保持原语义。

## 性能和质量协议

每次真实 GPU 实验最多 180 分钟，包括清理。示例给优化过程 165 分钟，
进程外层在第 179 分钟发 TERM，再给 60 秒退出时间。恢复时不自动改变能力层级；
NV 未显式传 `--max-hours` 时恢复已保存的预算值。已有运行时对记录过停止原因的
会话按新执行段重新计时，因此一次恢复仍须独立满足 180 分钟上限。

| 工作负载 | ISL | OSL | 并发 | 测量请求数 |
|---|---:|---:|---:|---:|
| W0 | 64 | 16 | 1 | 100 |
| W1 | 64 | 128 | 1 | 100 |
| W2 | 512 | 128 | 4 | 100 |
| W3 | 2048 | 32 | 1 | 100 |
| W4 | 2048 | 256 | 2 | 100 |
| W5 | 512 | 128 | 16 | 160 |

最终 baseline 和 candidate 各三次，W2/W5 吞吐几何平均增益须超过
`max(3%, 2×baseline CV)`；全部工作负载的 TTFT、TPOT、E2EL P99 退化不超过 5%，
其他吞吐退化不得超过测量噪声，无失败请求。未满足条件即 REVERT。
本轮 profiling smoke 使用短窗口，只验证采集闭环，不替代这套性能验收。

量化校准固定 WikiText-2 raw train，512×2048 tokens，seed=0；评估固定
WikiText-2 raw test 的 perplexity，以及 GSM8K main 完整 test。
关闭 thinking，确定性生成，最大输出 2048 tokens；与同一套 BF16 基线比较。
校准数据、评估数据和量化产物分别保存指纹。

## 本轮实现

- `--optimization-level config/profile`、`--profile-backend torch`，状态持久化和恢复冲突检查。
- `profile`、`roofline`、`trace_analysis` 分开授权；新 NV profile 会话仍不能进入 AMD TraceLens/roofline。
- `CudaProfileExecutor` 复用现有监督执行和 CUDA runner，保留当前配置的编译/Graph 参数，采集窗口最多 16 个请求。
- vLLM bench 先 warmup，再调用原生 start/stop。当前 vLLM 对空 HTTP 200 的客户端成功判定不可靠，因此检查服务端确认；已到达服务端的 stop 不重试。
- `vllm_cuda_profile.json` 包含 rank、trace health、拓扑、硬件/模型/配置/工作负载指纹和清理状态。
- 通用测量读取和晋升路径拒绝 profile 分数，也禁止从旁边的原始 JSON 补回分数。
- Critic 在 `--codex-cli-auth` 下复用私有 Codex SDK 认证与清理，保留真实 review runtime。
- 原 Qwen3 实验收拢至 `experiments/nvidia_qwen3/`；单独 CPU CI；安装包不依赖实验目录。

入口为 `python -m hyperloom.inference_optimizer.cli optimize`；CLI 的 `_run_optimize`
创建并启动 Coordinator。NV profile 在 `cli/executors.py` 注册，实际服务启动在
`orchestrator/actions/executors/vllm_cuda_runner.py`。

## 验收记录

三种拓扑的独立 profile smoke 使用 ISL=128、OSL=32、并发=2、8 个请求、2 次 warmup；
Qwen3-8B 示例另行使用其固定的 512/128/4 工作负载。

原始文件根目录：
`/data/ygw/llm_sim/benchmark_results/hyperloom_nv_integration_20260908/`。

| 验证 | 结果 | 证据（相对于原始文件根目录） |
|---|---|---|
| CPU 综合回归 | 584 passed；增补用例通过，生命周期专项另有 182 passed | `regression.log`、`regression.xml`、`cleanup_regression.log` |
| wheel 隔离 | 981 entries；无测试包或实验模块；安装后 CLI/CUDA profile 可导入 | `wheel_workspace.txt` 指向构建与安装日志 |
| Qwen2.5-3B baseline | 成功，225.753 output tokens/s/GPU；质量 smoke 通过；测量后释放租约 | `config_smoke.log` 和对应 session 的 baseline 目录 |
| 单卡 torch profile | 成功，rank 0，64,344 个 CUDA kernel 事件 | `profile_single_v3/executor_result.json` |
| Qwen3-32B TP1/PP8 | 成功，rank 0–7 全部有效，租约为零 | `profile_pp8_prefetch/executor_result.json` |
| Qwen3-32B TP2/PP4 | 成功，rank 0–7 全部有效，租约为零 | `profile_tp2pp4/executor_result.json` |
| 启动期强制终止 | 服务就绪前 SIGKILL runner，外层自动回收服务与租约 | `startup_cancel_fixed/acceptance.json` |
| 中断/恢复 | 首段以 signal 停止；恢复保留 baseline、工作负载和 30 分钟预算；三组候选测量成功、无 KEEP；再次受控停止后报告完成、GPU/租约释放 | `config_resume.log` |
| Qwen3-8B | 实际示例 baseline 成功，203.343 tok/s/GPU；自动 profile 通过，253,104 个 kernel 事件；baseline/current_best 未变化；受控停止后报告完成 | `/data/ygw/models/Qwen3-8B` |

开发中的失败也保留：首次 profile 暴露重复 stop 问题；一轮旧模块缺少 import，
对应 GPU 进程已退出并记录手工租约清理（`profile_single_v2/manual_cleanup.json`）。
PP8 默认 lazy 权重加载很慢，受控停止后使用已有 Qwen3-32B 基线的
`--gpu-memory-utilization 0.70 --safetensors-load-strategy prefetch` 通过验证。
最终收尾还发现启动期取消的租约窗口：旧 runner 在服务就绪后才发布生命周期记录，
可能先被监督进程杀死。现已改为启动后立即发布所有权，并以真实 SIGKILL 验证外层回收。
旧执行段的遗留租约按确切 holder/task 清理，见 `startup_cancel_legacy_cleanup.json`。
这些失败记录不计作成功验收，也不覆盖历史实验报告。

两个完整 CLI smoke 都按验收边界受控停止，`stop_reason=signal`，生成了 final report；
没有宣称自然进入 CLOSE、达到 30% 收益或跑满三小时。最终资源检查见
`cleanup_acceptance.json`。首批 A/B 的实现、采集和恢复验证完成，下一批进入阶段 C。

历史 4 小时 P6 soak 的证据属于旧源码身份；本轮没有重新执行该 soak，
不能用其结果宣称当前源码已通过相同的长期稳定性验收。


实现提交：`76b9ddadf`（Critic 认证）、`57b05302a`（实验隔离）、
`521763b01`（CUDA profile/能力/预算恢复）、`a4614f2f8`（示例）、
`0cc7bd40d`（启动取消清理）。当前工作只在本地提交，没有推送。
