# NVIDIA/CUDA 适配计划与实施记录

更新日期：2026-09-16。平台入口已通用化为 `nvidia_cuda`，旧目标名保留为兼容别名。
设备型号、数量、显存、计算能力与 CUDA 软件环境改为运行时发现；平台预检与框架后端检查分开。
设备选择保留 CUDA 可见顺序，通过 UUID 对齐 NVML，并将受限设备池传入配置、租约和采集。
普通测量不再强制 CUDA 13.0 或 nvcc；Nsight 计数器按实际设备查询。
会话状态升级至 v10，保存平台及设备池，恢复时核对设备与软件身份。
自动化与本机验证见[通用化验证记录](nvidia-platform-generalization-validation.md)。

以下阶段 A–L 保留的是截至 2026-09-15 的历史实施与实验记录，其硬件、模型和结果不变。
这些实验当时使用本机 8×RTX 4090、CUDA 13.0 和 `llm_sim` 中的 vLLM，
不能据此宣称其他 GPU 组合已经完成硬件验收。

阶段 A–L 的本轮受控优化闭环已完成。阶段 D、E、I、J 和 L 的候选均按联合性能门槛
回退；F、G 和 H 已完成资格或拓扑审计并停止在没有合格候选的边界。当前没有写入
`current_best` 的 NVIDIA 优化叠加层，TP2/PP4 固定服务配置仍是本轮受控基线。

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
`0cc7bd40d`（启动取消清理）。上述 A/B 实现已随当前同步提交推送到 `origin/my_dev`。


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
窗口改为原生 warmup → start → 完整有界请求 → stop，delay/max iterations 均为 0，
不再用上一次运行的 execution step 推算下一次窗口。TP 场景的 decode/未知阶段热点
逐 GPU、逐热点独立采集；明确属于 prefill/mixed 的热点仍一次采集相关 GPU。
每次均保留完整服务拓扑；decode 采集同时按设备和 worker 进程过滤，
每个选定 worker 采 1 次 launch，其余采集最多 3 次。
复合动作最多 45 分钟，另受会话剩余预算
和阶段取消约束；每次真实实验的 180 分钟上限继续有效。

时间线以实际 worker PID、GPU UUID 和 rank 绑定，计算区间并集、通信重叠、
copy 和 idle；热点占比以 kernel 时长之和为分母，不能当作墙钟比例。
独立 ncu 样本还须匹配模型/硬件/工作负载/serving 配置/质量套件指纹及
rank、GPU UUID、grid、block。计数器以 base units 导出，保留原始 metric、单位、
实际时钟对应的 sustained peak、算术路径和样本来源。
无法获得的指标显示 unavailable，不填零；fused/Graph kernel 的层归属未知时
明确保留 unknown。选定 kernel 的 roofline 不外推为整个模型的吞吐上限。

原始 `.nsys-rep`、SQLite、`.ncu-rep`、CSV、`rank_processes.json`、
`vllm_cuda_profile.json`、`nsight_summary.json` 和 `analysis.md` 均保留。
计数器失败会保留已验证时间线，同时把复合结果标记为不完整。
profiling 吞吐只作诊断，不能成为 baseline、current_best 或 KEEP 分数。
source、kernel 和 quantization 能力仍未开放。阶段 C 的本机混合并行计数器
验收已补齐（见文末），下一阶段为 D 的配置优化验收。

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

## 阶段 C 首轮验收记录（历史）

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
复验；当时混合并行尚未解决，最终修复见下文。尝试过的 shmem 参数与外部完成计数控制器未进入
最终实现。名为 `ncu_mixed_minimal` 的补充实验实际使用完整指标集：环境准备
使 PATH 包装器失效，未发生指标缩减；随后已固定预检工具的绝对路径。
恢复控制器也曾过早发信号或误识别 CLI PID，这些失败记录独立保留，
最终按精确 CLI PID 完成了真实恢复验证。

首轮留下的验收项是 TP2/PP4 在相同工作负载和 CUDA Graph 配置下的稳定热点采集，
验收要求是新的完整 nsys → 三热点 ncu 运行全部通过 rank、指纹、launch shape
和算术指标检查。当时固定 execution step 窗口不能保证跨进程重放时同名热点
仍出现在该窗口；不能用补充单热点成功代替完整验收，也不能静默修改并发或
关闭 CUDA Graph 来通过。需要仅使用时间线时，可显式加 `--no-enable-roofline`。
该项已按下文完成。下一步依据分析结果实施阶段 D，并按 W0–W5 三次复测协议裁决候选。

## TP2/PP4 ncu 修复验收（2026-09-08）

窗口漏采和 TP 并行采集停滞已修复；最终源码在本机固定环境完成连续三轮
全新 nsys → 三热点 ncu → state/report。验收总表为
`/data/ygw/llm_sim/benchmark_results/hyperloom_nv_ncu_fix_20260908/final_acceptance.json`，
以下证据路径均相对该目录。阶段 C 的本机拓扑验收已补齐，阶段 D 尚未开始。
本轮复验也遇到并处理了主机连续页不足，环境处置和适用边界见下文。

### 最终实现

- 取消跨运行 execution step 窗口，使用 warmup → 原生 start → 全部有界请求 → stop；
  profile 最多 16 请求，保持原模型、拓扑、并发及 CUDA Graph/serving 参数。
- ncu 显式限定本次租约的逻辑 GPU。TP 场景每次只让一个实际 worker 被 ncu 注入；
  该 rank 的实际前三热点合并为一次服务运行，每个精确名字采第一次匹配。
  输出仍为逐 kernel roofline，未使用 graph/range 聚合替代。
- 以显式算术/带宽指标替代旧 section 集。BF16 Tensor GEMM 独立指标实测 1 pass，
  含标量 GEMV 的联合指标实测 3 passes，旧 section 集为 16 passes。缺失算术不补零。
  policy 记录实际命令、设备、工具、指标和窗口，progress 记录观察时间；每个样本
  保留 replay passes 和可获取的备份字节数。
- 指定 worker 使用同一 Python 二进制的独立名称，保持 prefix 和包环境；文件仅在
  采集工作区生成，已有 sitecustomize 时拒绝覆盖。记录并核对 executable/SHA256。
  所有 worker 的 PID/启动时间立即写入 manifest；正常退出、失败和 runner 被杀后，
  通过 pidfd 回收脱离服务组的 worker，避免残留显存或 PID 复用误杀。
- 对每个实际热点检查完整相关 rank、PID、精确名字、GPU UUID/PCI、grid/block，
  以及硬件、模型、工作负载、serving、quality 指纹。每个名字/rank 需 1–3 个有效
  样本，缺失或不匹配即失败。计数器失败仍保留有效时间线。

### 最终源码验收

源码身份为 `final_acceptance_source_identity.json`；三轮期间未改代码。
TP2/PP4 固定 Qwen3-32B、ISL128/OSL32/CONC2、8 请求、2 warmup、maxlen2048、
`--gpu-memory-utilization 0.70 --safetensors-load-strategy prefetch`。
每轮 8 次 ncu，每个采集均完成全部请求、无失败请求，三个热点各覆盖 rank 0–7。

| 验证 | 结果 | 证据 |
|---|---|---|
| 连续完整复验 1 | 1678.76 秒，24 个有效样本 | `mixed_final_repeat_3/evidence_validation.json` |
| 连续完整复验 2 | 1272.15 秒，24 个有效样本 | `mixed_final_repeat_4/evidence_validation.json` |
| 连续完整复验 3 | 1241.84 秒，24 个有效样本 | `mixed_final_repeat_5/evidence_validation.json` |
| 连续三轮总计 | 72 个有效逐 kernel 样本；各轮均在 45 分钟内 | `three_consecutive_acceptance.json` |
| Qwen2.5-3B 单卡 | 337.77 秒；3 次 ncu、9 个有效样本 | `single_scoped/evidence_validation.json` |
| Qwen3-32B TP1/PP8 | 641.67 秒；3 次 ncu、51 个有效样本 | `pp8_verified/evidence_validation.json` |
| Qwen3-8B 实际 example | 自动 baseline → nsys → 3 次 ncu → state/report；9 样本，profile 每次 16 请求；baseline/current_best 保持 203.56367939689284 tok/s/GPU | `cli_acceptance.json`、`cli_writeback_verification.json` |
| 恢复与报告 | 工具身份、能力、预算、工作负载和 baseline 不变；报告含有效分析链接 | `resume_acceptance.json`、`resume_report_verification.json` |
| nsys-only 实际 example | roofline 关闭，仅一次 nsys，无 ncu；报告 counter_status=not_requested | `nsys_only_acceptance.json`、`nsys_only_writeback_verification.json` |
| 启动期 SIGKILL | nsys 与进程过滤 ncu 均通过；8 个记录 worker 退出，GPU/端口/租约释放 | `startup_cancel/acceptance.json`、`ncu_startup_verified/acceptance.json` |
| CPU 回归 | 665 passed、1 skipped；唯一跳过项缺少既有外部样例报告 | `scoped_regression.log`、`scoped_regression.xml` |
| wheel / 静态检查 | 984 entries，隔离安装导入和源码逐字节一致；Ruff/格式检查通过 | `scoped_wheel_acceptance.json` |
| 最终资源 | 无实验 GPU/worker 进程、PID 文件、租约或监听端口，源码身份不变 | `final_cleanup_acceptance.json` |

所有 profile 吞吐均不写入 baseline/current_best，也不产生 KEEP。CLI/resume 按验收
边界受控 SIGINT，报告 `stop_reason=signal`、`report_complete=false`；不宣称自然
进入 CLOSE、获得端到端吞吐收益或完成三小时/长期 soak。后续阶段 D 按 W0–W5
基线和候选各三次复测及质量门槛裁决。

### 失败证据与环境边界

旧的仅设备过滤仍会在 GPU 0 GEMV 停滞；进程过滤后首次完整通过，第二轮暴露
worker 显存残留。补充回收后，全卡 prefill 仍间歇停滞，因此 TP 的所有阶段均
改为逐 worker 过滤。旧策略的 `mixed_worker_repeat_1–3` 三轮成功保留在
`grouped_three_consecutive_acceptance.json`，不能替代最终源码三轮。

单卡默认 all-device 范围曾导致采样后 Graph 回放极慢；仅 invocation 过滤无效。
同一模型、工作负载和工具版本下，仅 `--devices 0` 的对照在 6.47 秒完成 8 请求
和 3 样本，见 `single_device_only/executor_result.json`。生产保留显式租约设备范围，
未保留无效的单热点 invocation 补充。失败分别见 `single_final`、`single_verified`。

最终源码的 `mixed_final_repeat_1` 在 worker 3 遇到 ncu backing-store allocation /
LaunchFailed；内核同时记录 order-10 连续页分配失败。主机约 908 GiB 为页缓存，
虽然 MemAvailable 很高，普通内存区仍无所需连续页。单次规整后的同 rank 对照
完成 8 请求、3 样本，五类指纹相同；随后连续页再次耗尽，第二轮提前受控取消。
这些运行均不计入成功验收，原始错误和取消记录保存在各目录。

在 GPU 实验全部退出后，一次性回收干净页缓存并规整内存，再开始最终三轮；
未改变文件内容、持久化 VM 设置、源码、模型或服务参数。操作前后状态见
`host_cache_reclaim.json`，对照见 `rank3_after_compaction/evidence_validation.json`。
生产执行器不会自动执行主机维护；此类主机资源不足仍可能使 ncu 回放失败，
不能把本机验收外推为任意主机内存状态、其他 GPU、其他工具版本或多机的保证。

## 阶段 D 历史实验与实机结论

阶段 D 是一次性、固定合同的本机 Qwen3-32B BF16、TP2/PP4、maxlen 2048 配置 campaign。
其阶段 launcher 已从正式工程移除；以下保留实际参数、门槛与 artifact，供审计使用。它只接受
本机 Qwen3-32B BF16、TP2/PP4、maxlen 2048，自动建立 baseline；不接受通用
agent 的任意参数搜索，也不会写入 `current_best`。

每个 GPU 子步骤由独立的 179 分钟进程执行并写入同一个 campaign manifest：一次
W2/W5 baseline、四个候选的 W2/W5 筛选、三次 baseline/candidate 成对 W0–W5
复验和最终裁决。可重复启动同一 campaign 路径恢复，恢复前会核对硬件、模型、
拓扑、工作负载、baseline、候选和质量套件指纹。

候选仅为 `max-num-batched-tokens`（4096/8192）、`max-num-seqs=16` 和 GPU
memory utilization（0.70/0.80/0.85）的固定组合。保留既有 prefix cache、
chunked prefill、Inductor、CUDA Graph、BF16、attention backend 与 KV dtype；
不在阶段 D 修改 eager、FP8、量化、源码或 kernel。

Qwen3 P3 现在要求四个固定中英文/短长/thinking 用例包含光合作用的反应物和产物
语义组。每次服务启动均执行；thinking 用例正常闭合时验证最终回答，若受限 completion
只返回未闭合的 thinking 前缀，则验证其中已生成的事实内容并明确记录来源。回答、规范化
文本、命中词、缺失组和哈希都会保存；非空但语义不完整也会失败。

筛选候选必须在 W2 与 W5 都有正吞吐增益且所有 P99 不超过筛选 baseline 的 105%。
最终只有 W2/W5 几何平均吞吐增益严格大于 `max(3%, 2×baseline CV)`、其余负载
下降不超过 `2×baseline CV`、全部 TTFT/TPOT/E2EL P99 不超过 5%、无失败请求且
P3 全通过时才输出 `accepted`。实际 campaign 位于
`/data/ygw/llm_sim/benchmark_results/qwen3-32b-tp2-pp4-stage-d/`：baseline 与
四个候选的筛选均已执行，未出现可进入三轮全量复验的候选，因此结论为 `reverted`。
完整输入、每次测量和筛选理由保存在 `stage_d_manifest.json`，最终结论为
`stage_d_final.json`；不写入 `current_best`。

## 阶段 E 历史源码实验（首个通信候选）

阶段 E 是一次性、来源绑定的 vLLM 源码实验；阶段 CLI 和 launcher 已从正式工程移除。它从当前 vLLM
wheel 的 `direct_url.json` 提取构建提交和 SHA-256，在 campaign 私有 worktree
中导入源码 Python 层，并从该同一 immutable wheel 提取经哈希校验的原生扩展；这样
PP Python 补丁实际生效，同时不重编未修改的 CUDA/C++ 扩展。提交或 wheel 不可获取、
源码不干净、导入位置或版本不在该 worktree、扩展身份不一致时均停止，不会以本地 0.27
checkout 或其他 release 替代。

步骤依次为 `prepare`、三轮 W2/W5 `source-baseline`、短窗口 `profile`、
`candidate` 和 `finalize`。唯一候选在 PP sampled-token 通信中 coalesce 原有两次
broadcast，保持 tensor、side stream、event、队列与释放请求过滤语义。候选需在 W2
和 W5 均有正吞吐增益、全部 P99 不超过 105%、P3 通过且无请求或清理失败，才会标记
`ready_for_validation`。随后依次执行三次 `validate --replicate 1|2|3` 和
`validation-finalize`：每次先恢复已记录 SHA-256 的源码基线并测量 W0–W5，再重放同一
PP 补丁并测量 W0–W5。最终在 W2/W5 要求几何平均吞吐增益严格超过
`max(3%, 2×baseline CV)`，其余工作负载的降幅不超过 `2×baseline CV`，全部 P99
回归不超过 5%，且 P3、请求和清理均成功。结论写入
`stage_e_validation_final.json`，为 `accepted` 或 `reverted`；阶段 E 不写入
`current_best`。实际 campaign 位于
`/data/ygw/llm_sim/benchmark_results/qwen3-32b-tp2-pp4-stage-e/`：三轮完整配对
W0–W5 已完成，候选因 W2/W5 吞吐未达门槛及 W2/W5 P99 回归而 `reverted`。
`stage_e_manifest.json` 保留每次测量、源码 SHA-256、补丁与运行时身份，
`stage_e_validation_final.json` 保留最终裁决。

## 阶段 F0 历史 kernel 资格确认

阶段 F0 是一次性 decode GEMV 资格确认；阶段 CLI 和 launcher 已从正式工程移除。它在独立、干净的
vLLM 源码运行时中，对 W2/W5 使用 eager + layerwise NVTX 作诊断采集，关联
cuBLAS `gemvx` 与 vLLM linear layer、BF16 `(M,N,K)` 和 bias 条件；诊断结果不能
作为 serving 性能 baseline。随后针对同一 kernel 的受限 NCU 样本和原始
unquantized GEMM microbenchmark 写入同一 manifest。

只有 W2/W5 均能稳定归因、各自的 GEMV 时间占比至少 10%、三次 microbenchmark
CV 不超过 5%、且 NCU 带宽效率低于 85% 时，才标记 `qualified` 并允许后续独立的
F1 Triton 候选工作。否则 `stage_f_final.json` 写入 `no_eligible_kernel`，停止
kernel authoring 并保留全量采集证据；阶段 F0 永不写入 `current_best`。

## 阶段 G（PP P2P 调度候选）

阶段 G 是一次性 PP P2P 调度实验；阶段 CLI 和 launcher 已从正式工程移除。历史步骤为
`prepare`、`diagnose`、`source-baseline`、`candidate`、`finalize`、三次
`validate`、`validation-finalize` 和可选的 `promote`。

G0 `diagnose` 在 campaign 私有 vLLM worktree 中临时为每个 worker 标记“上一轮
非阻塞 PP send 的等待到当前 irecv 提交”的 NVTX 区间，以 Nsight Systems 原始
SQLite 计算各 rank 的最大区间占 capture window 的比例。采集结束无条件恢复
`gpu_worker.py`。只有 SendRecv 的总 GPU kernel 时间占比至少 10%，且至少一个
PP 边界的标记区间至少为该 rank capture window 的 1%，才允许建立 serving
baseline 并试验唯一候选。

唯一候选保留原来的 send handles，但将其 wait 从 `execute_model` 开头移到当前
`irecv_tensor_dict` 提交之后、模型执行之前；不会改变 tensor、PP group、发送顺序
或所有权。筛选与阶段 E 相同：W2/W5 均须正吞吐增益、所有 P99 不超过 5% 筛选
窗口；最终以三轮成对 W0–W5 复验，W2/W5 几何平均增益必须严格高于
`max(3%, 2×baseline CV)`，其他 workload 的吞吐下降不超过 `2×baseline CV`，
所有 P99 不超过 baseline 最大 P99 的 5%，并要求 P3、请求与清理全部成功。

`promote` 仅在最终 `accepted` 后写入身份绑定的
`pp_p2p_overlap_overlay.json`，声明需要显式环境变量
`HYPERLOOM_NVIDIA_PP_P2P_OVERLAP=1`；阶段 G 不写 `current_best`。

### 阶段 G 实测结论

实际 campaign 位于
`/data/ygw/llm_sim/benchmark_results/qwen3-32b-tp2-pp4-stage-g/`。G0 W2
采集完成 8 个请求、P3 质量通过且无失败请求；`gpu_worker.py` 已恢复到采集前 SHA-256
`5e8faf3e00649289c81c2917dd4557a80b0b351c33295d8e0c4b204a304e6552`。Nsight 结果显示
SendRecv 占 GPU kernel 时间 26.4046%，满足第一项门槛；但 8 个 rank 中最大的标记边界
窗口仅为 capture window 的 0.1329%，低于 1%。因此
`stage_g_final.json` 的结论是 `stopped_no_eligible_p2p_window`，没有运行 serving
baseline、源码候选或配置优化，亦没有更新 `current_best`。

## 阶段 H（TP2/PP4 GPU rank 映射）

阶段 H 是一次性 GPU rank mapping 审计；阶段 CLI 和 launcher 已从正式工程移除。它固定本机
Qwen3-32B、TP2/PP4 与现有 serving 参数，审计 `nvidia-smi topo -m` 的完整 8×8
GPU 链路矩阵和 NUMA 分布，穷举 `CUDA_VISIBLE_DEVICES` 的 8! 个 rank 映射。比较时
PP 六条边的跨 NUMA 数、链路总成本和最差链路优先于 TP 四条边的相同指标；只有某映射
在全部指标不差且至少一项严格更优时，才会执行 W2/W5 筛选以及三轮成对 W0–W5 复验。

实际 audit 位于 `/data/ygw/llm_sim/benchmark_results/qwen3-32b-tp2-pp4-stage-h/`。
默认 `0,1,2,3,4,5,6,7` 映射得分为 PP `(2,18,5)`、TP `(0,6,2)`，其中两个 PP
链路跨 NUMA/SYS；没有严格支配映射。结论为
`stopped_default_mapping_pareto_optimal`，因此未启动 vLLM serving benchmark、未修改
源码或 CPU affinity，也未更新 `current_best`。证据在 `stage_h_final.json`。

## 阶段 I–L 与本轮收口（2026-09-15）

阶段 I–L 补齐了 NUMA、拓扑和 TTFT 归因后的单一源码候选。所有测量均固定本机
Qwen3-32B、8×RTX 4090、TP2/PP4（除阶段 J 拓扑筛选）、maxlen 2048、BF16、
`--gpu-memory-utilization 0.70 --safetensors-load-strategy prefetch`，每次服务启动均
要求 Qwen3 P3 质量门通过。筛选要求 W2/W5 都有严格正吞吐增益且 TTFT/TPOT/E2EL
P99 不超过基线 5%；不满足时不进入三轮 W0–W5 复验，也不更新 `current_best`。

| 阶段 | 实测结论 | 最终证据 |
|---|---|---|
| I：NUMA binding | `--numa-bind` 的 W2 无正吞吐增益，回退。 | `/data/ygw/llm_sim/benchmark_results/qwen3-32b-tp2-pp4-stage-i/stage_i_final.json` |
| J：TP/PP 拓扑 | TP4/PP2 的 W2/W5 吞吐增加 32.93%/11.47%，但 TTFT P99 增加 57.64%/56.02%；TP1/PP8 的 W2 吞吐下降 40.33%。两者回退。 | `/data/ygw/llm_sim/benchmark_results/qwen3-32b-nvidia-stage-j/stage_j_final.json` |
| K：TTFT 归因 | W2/W5 中 PP `irecv` 的 CPU metadata receive 分别均值 4.595/4.361 ms，成为可验证源码候选；诊断插桩在每轮后恢复。 | `/data/ygw/llm_sim/benchmark_results/qwen3-32b-tp2-pp4-stage-k/stage_k_final.json` |
| L：metadata cache | 缓存候选 W2 100/100 请求与 P3 均通过，但 90.5698 tok/s 低于同运行时基线的 90.5847 tok/s（-0.0164%）；跳过 W5 和全量复验，回退。 | `/data/ygw/llm_sim/benchmark_results/qwen3-32b-tp2-pp4-stage-l/stage_l_final.json` |

阶段 K 的元数据热点没有转化为端到端吞吐收益，因此本轮不再继续尝试缺少新独立证据的
源码候选。阶段 L 的私有 vLLM worktree 已恢复干净，实验服务和 Nsight 进程已退出；
候选补丁只保存在 campaign artifact 中，不进入工程源码或生产启动参数。

### 最终状态与适用边界

本轮完成了 NVIDIA/vLLM 的本机可运行接入、baseline/configuration campaign、CUDA
profile/Nsight/roofline 采集、来源绑定的源码实验、GPU 拓扑与 NUMA 审计及质量/清理
验收。它证明这些路径能够产生可恢复、可审计的正反两类证据；它没有证明对该硬件、模型
和工作负载存在可保留的端到端性能提升。

当前受控基线为 TP2/PP4 与上述固定服务参数。任何新的优化工作必须从新的、独立的热点
证据开始，并重新执行 W2/W5 筛选、P3 质量和全部 P99 门槛；不得复用本轮被回退候选的
数字作为收益声明。
