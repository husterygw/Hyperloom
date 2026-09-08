# NVIDIA/vLLM 后续适配计划与首批验收

更新日期：2026-09-08。范围固定为本机 8×RTX 4090、CUDA 13.0、现有
`llm_sim` 环境中的 vLLM；暂不扩展其他 GPU、多机或其他 serving framework。

首批完成阶段 A、B 和 profile/roofline 能力拆分。阶段 C 的实现与验收见文末；
阶段 D–H 仍是后续工作，能力开放须有实际执行与验收证据。

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
能力。当前 CLI 对 `--optimization-level source/kernel` 明确报未实现；
NVIDIA Nsight 使用独立执行器，AMD 的原有默认能力保持原语义。

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

## 首批 A/B 实现

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

## 首批 A/B 验收记录

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


## 阶段 C：Nsight 实现

NVIDIA 新会话显式选择 `--optimization-level profile --profile-backend nsys`，
默认执行 Nsight Systems 时间线和 Nsight Compute 热点计数器复合动作。
`--no-enable-roofline` 保留时间线与 `analysis.md`，跳过 ncu；torch 和 config
仍保留原来的默认行为。恢复会话保留 backend、roofline 开关、工具路径/版本和预算，
冲突则拒绝恢复，旧会话不会因升级代码而自动获得新能力。实际启动和导出固定
使用预检选定的绝对路径，CUDA 环境准备对 PATH 的修改不会替换工具。

`CudaRooflineExecutor` 先独立运行 nsys，再为前三个非通信 kernel 名称分别启动
新的 ncu 服务。每次保留模型、工作负载、拓扑、编译/Graph 和 serving 参数，
warmup 后最多采集 16 个请求；ncu 每个相关 rank 最多采集 3 次匹配 launch。
窗口从 nsys 的 native NVTX execution step 确定，包含空 pipeline step，
起点对齐最晚首次出现的 rank；rank 错开时覆盖其差值与后续 pipeline 周期。
复合动作最多 45 分钟，另受会话剩余预算
和阶段取消约束；每次真实实验的 180 分钟上限继续有效。

时间线以实际 worker PID、GPU UUID 和 rank 绑定，计算区间并集、通信重叠、
copy 和 idle；热点占比以 kernel 时长之和为分母，不能当作墙钟比例。
独立 ncu 样本还须匹配模型/硬件/工作负载/serving 配置/质量套件指纹及
rank、grid、block。计数器以 base units 导出，保留原始 metric、单位、
实际时钟对应的 sustained peak、算术路径和样本来源。
无法获得的指标显示 unavailable，不填零；fused/Graph kernel 的层归属未知时
明确保留 unknown。选定 kernel 的 roofline 不外推为整个模型的吞吐上限。

原始 `.nsys-rep`、SQLite、`.ncu-rep`、CSV、`rank_processes.json`、
`vllm_cuda_profile.json`、`nsight_summary.json` 和 `analysis.md` 均保留。
计数器失败会保留已验证时间线，同时把复合结果标记为不完整。
profiling 吞吐只作诊断，不能成为 baseline、current_best 或 KEEP 分数。
source、kernel 和 quantization 能力仍未开放。阶段 C 的混合并行计数器验收
仍需补齐，再进入阶段 D 的配置优化验收。

示例命令：

```bash
PYTHON=/data/ygw/miniconda3/envs/llm_sim/bin/python \
MODEL_PATH=/data/ygw/models/Qwen3-8B \
USER_DATA_PATH=/data/ygw/llm_sim/benchmark_results/my_nsight_run \
bash examples/hyperloom-qwen3-8b-nvidia-3h/run_example.sh \
  --optimization-level profile --profile-backend nsys
```

示例仍自动测量 baseline；优化预算 165 分钟，完整 Nsight 模式的 PRELUDE 占
35%，framework 占 58%，sweep 占 1%。启动入口不变，新增执行器为
`cuda_roofline.py`，采集/解析工具为 `cuda_nsight.py`。

## 阶段 C 验收记录与剩余项

原始证据根目录为
`/data/ygw/llm_sim/benchmark_results/hyperloom_nv_nsight_20260908/`。
环境为 vLLM 0.28.0、torch 2.13.0+cu130、Nsight Systems 2026.1.3、
Nsight Compute 2026.2.1；本机计数器权限和实际小矩阵采集已验证。
拓扑 smoke 使用 ISL=128、OSL=32、并发=2、8 个请求；实际 Qwen3-8B
CLI 使用示例的 512/128/4 工作负载。每次均保留原 serving/CUDA Graph 配置。

| 验证 | 结果 | 证据（相对上述根目录） |
|---|---|---|
| CPU 综合回归 | 571 passed、1 skipped、1 xfailed；生命周期专项另有 182 passed | `final_regression.log`、`final_regression.xml`、`cleanup_regression.log` |
| wheel 隔离 | 984 entries，无测试/实验模块；隔离安装后 CLI、CUDA 执行器和报告辅助函数可导入 | `final_wheel_acceptance.json`、`final_wheel.log` |
| Qwen2.5-3B 单卡复合采集 | 成功；63,758 个 kernel 事件，前三个热点共 9 个有效 roofline 样本 | `roofline_single_final/executor_result.json`、`final_topology_acceptance.json` |
| Qwen3-32B TP1/PP8 复合采集 | 成功；213,696 个 kernel 事件，rank 0–7 有效；三个热点共 50 个有效样本，其中最后一个热点仅出现在 rank 7 | `roofline_pp8_final/executor_result.json`、`final_topology_acceptance.json` |
| Qwen3-8B 实际 CLI | 自动 baseline → nsys → 三次 ncu → state/report；252,596 个 kernel 事件、9 个有效样本；baseline/current_best 保持 203.10486789309786 tok/s/GPU | `cli_acceptance.json`、`cli_writeback_verification.json` |
| 中断/恢复与报告 | 恢复工具身份、能力、工作负载、165 分钟预算及 baseline；受控 SIGINT 后自动更新带分析链接的报告，counter_status=succeeded | `resume_acceptance.json`、`resume_report_verification.json` |
| 启动期 SIGKILL | 服务就绪前强杀 runner，外层回收已启动 GPU worker、端口和租约 | `startup_cancel/acceptance.json` |
| Qwen3-32B TP2/PP4 时间线 | rank 0–7 验证通过；计数器失败后仍保留有效时间线 | `roofline_tp2pp4_final/executor_result.json` |
| TP2/PP4 单个 decode 热点补充采集 | 成功取得 8 个 rank 各 3 个有效样本；只验证了该热点 | `ncu_mixed_minimal/executor_result.json`、`mixed_window_verification.json` |
| TP2/PP4 完整计数器 | **未通过**：重启后热点出现步骤变化会漏采，部分窗口出现 sample_tokens RPC 超时 | `roofline_tp2pp4_final/executor_result.json` |
| 扩展窗口补充实验 | **失败**：尝试按实际采样完成数提前停止，最终仅完成 3/8 个请求；未引入生产代码 | `ncu_mixed_completion/executor_result.json`、`completion_probe_control.json` |
| 最终资源检查 | GPU 进程、实验所有权 PID 文件、GPU 租约和实验服务监听端口均已释放 | `cleanup_acceptance.json` |

Qwen3-8B CLI 和恢复验证均受控停止，`stop_reason=signal`；报告属于中断时的
安全网导出，`report_complete=false`，不能宣称自然进入 CLOSE、得到最终 KEEP、
获得吞吐收益或完成三小时运行。profile 关闭质量评估，仅用于诊断采集，
也不能替代阶段 D 的质量和性能验收。

开发失败的原始报告均保留。包括导出时 kernel 名称被重命名、pipeline 空步骤
导致窗口选早，以及 TP 混合并行采集停滞。前两项已修复并通过上述单卡/PP8
复验；混合并行仍未解决。尝试过的 shmem 参数与外部完成计数控制器未进入
最终实现。名为 `ncu_mixed_minimal` 的补充实验实际使用完整指标集：环境准备
使 PATH 包装器失效，未发生指标缩减；随后已固定预检工具的绝对路径。
恢复控制器也曾过早发信号或误识别 CLI PID，这些失败记录独立保留，
最终按精确 CLI PID 完成了真实恢复验证。

下一步首先解决 TP2/PP4 在相同工作负载和 CUDA Graph 配置下的稳定热点采集，
验收要求是新的完整 nsys → 三热点 ncu 运行全部通过 rank、指纹、launch shape
和算术指标检查。现有固定 execution step 窗口不能保证跨进程重放时同名热点
仍出现在该窗口；不能用补充单热点成功代替完整验收，也不能静默修改并发或
关闭 CUDA Graph 来通过。需要仅使用时间线时，可显式加 `--no-enable-roofline`。
该项通过后，再依据分析结果实施阶段 D，并按 W0–W5 三次复测协议裁决候选。
