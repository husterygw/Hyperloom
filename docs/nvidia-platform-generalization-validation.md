# NVIDIA/CUDA 平台通用化验证记录

日期：2026-09-16。基于已同步的 `upstream/main` 和本地 NVIDIA 适配提交完成，现已纳入 `origin/my_dev`。

## 实现范围

- 新入口为 `--target nvidia_cuda`；原目标名称仅作为兼容别名。型号、卡数、计算能力、显存及 CUDA 版本均由实际环境提供。
- 平台预检使用 NVML 和独立 CUDA driver 枚举，不依赖推理框架；实际 Python/PyTorch/框架接口检查由执行后端负责。
- 可见设备池、数量上限和任务分配贯通配置生成、runner、GPU 租约、采集及 Specialist 设备分配。子进程使用 GPU UUID，记录对应的物理与逻辑编号。
- 普通运行不强制 nvcc；源码和 kernel 构建能力仍关闭。Nsight 指标按实际设备和工具版本查询。
- 会话状态升级至 v10，记录运行平台和设备池；兼容目标别名，恢复前验证设备顺序及平台、执行环境、采集工具身份。

## 自动化验证

18 个相关测试文件合计 **567 项通过**，执行耗时约 45 秒。使用 `llm_sim` Python 环境，测试集合为：

```text
test_cuda_platform.py
test_nvidia_target.py
test_vllm_cuda_runner.py
test_cuda_profile.py
test_cuda_nsight.py
test_gpu_pool_device_resolution.py
test_shared_state_evolution.py
test_shared_state_persistence.py
test_workload_envs.py
test_workload_envs_golden_lock.py
test_cli_bootstrap.py
test_roofline_snapshot_unit.py
test_roofline_ceiling_env_units.py
test_preflight_serving_framework.py
test_policy_helpers_coverage_unit.py
test_specialist_subprocess.py
test_install_nvidia_target.py
test_cli_experiment_isolation.py
```

以上文件均位于 `src/hyperloom/inference_optimizer/tests/`。覆盖设备发现、CUDA/NVML 编号不一致、设备子集与重排、UUID、空掩码、资源上限、候选越界、不同软件版本、可选编译器、后端缺失、旧会话迁移、诊断结果隔离、清理与 AMD 回归。修改的 Python 文件通过 Ruff 检查，`git diff --check` 通过。

RTX 4090、A100、H100、B200 及 1/2/4/8 卡组合属于模拟覆盖，不能当作这些型号全部通过了硬件验收。

## 本机实际验证

使用现有 CUDA 执行后端，在本机八卡主机上仅开放物理 GPU 7，进程内对应逻辑 GPU 0。模型为本地 Qwen2.5-3B-Instruct，完成以下验证：

| 验证项 | 结果 |
|---|---|
| 平台与执行环境预检 | 识别单卡可见池并通过检查，不再要求八张卡全部可见 |
| 普通基线 | 服务启动、质量 smoke、请求测量、兼容报告和清理成功 |
| 配置候选 | 修改 batch token 上限后成功启动并完成同类测量 |
| torch profile | 成功导出 rank 0 的 CUDA kernel trace，采集健康检查通过 |
| 公共结果解析 | 基线及候选为有效普通测量；profile 返回 `valid_measurement=False` |
| 资源清理 | 三轮均报告 `cleanup_status=released`，会话租约表最终为空；测试 GPU 无残留计算进程 |
| ncu 指标发现 | 在另一张空闲同型号 GPU 上查询 profiling、device、stats 三类指标，所选 BF16 指标及 PCI 身份、replay 统计均可解析 |

原始产物入口：[验证目录](/data/ygw/llm_sim/benchmark_results/nvidia-platform-generalization-20260916)、[执行摘要](/data/ygw/llm_sim/benchmark_results/nvidia-platform-generalization-20260916/summary.json)、[ncu 指标查询](/data/ygw/llm_sim/benchmark_results/nvidia-platform-generalization-20260916/ncu-query.json)。

本次验证用于确认平台接线和测量类型，不构成性能提升结论。未重新执行完整 Nsight 时间线加热点计数器采集，也未进行其他型号、多卡并行或容器组合的硬件验收；这些路径分别由自动化回归覆盖或待后续实际验证。多机、MIG 与跨架构混合并行仍不支持。
