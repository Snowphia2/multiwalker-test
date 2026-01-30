# MultiWalker 批量扰动测试使用指南

## 概述

这个测试框架实现了对 MultiWalker 环境的自适应动作扰动实验，可以：
- 对指定 agent 施加动作扰动
- 自动监测失效（连续10步角度>6.5°）并取消扰动
- 追踪三个关键指标：MTTF、恢复时间、最大角度
- 批量测试多个配置组合
- 为每个配置生成角度-时间变化图

## 实现的功能

### 1. 环境层扰动追踪 (`pettingzoo_mw_env.py`)
- 在 step 100 注入扰动到目标 agent 的所有4个动作维度
- 实时监测杆子角度
- 当连续10步角度超过6.5°时自动取消扰动
- 监测恢复时间（角度降到5°以下）
- 追踪最大倾斜角度

### 2. Logger层指标汇总 (`pettingzoo_mw_logger.py`)
- 收集每个episode的MTTF、恢复时间、最大角度
- 计算统计平均值

### 3. 评估层结果输出 (`eval.py`)
- 将扰动指标添加到评估结果
- 提供绘图函数生成角度-时间图

### 4. 批量测试脚本 (`batch_disturbance_test.py`)
- 自动测试所有agent × 扰动幅度组合（3 × 5 = 15组）
- 生成结果JSON和可视化图表
- 生成汇总报告

## 使用方法

### 测试单个配置

```bash
# 测试默认配置 (Agent 0, Magnitude 0.3)
uv run python test_single_disturbance.py

# 测试指定配置
uv run python test_single_disturbance.py --agent 1 --magnitude 0.5

# 查看帮助
uv run python test_single_disturbance.py --help
```

### 批量测试所有配置

```bash
# 测试所有 3个agent × 5个幅度 = 15组配置
uv run python batch_disturbance_test.py
```

这将测试以下所有组合：
- Agent 0: 幅度 0.1, 0.2, 0.3, 0.4, 0.5
- Agent 1: 幅度 0.1, 0.2, 0.3, 0.4, 0.5
- Agent 2: 幅度 0.1, 0.2, 0.3, 0.4, 0.5

### 通过配置文件测试

如果需要更细粒度的控制，可以在配置文件中设置参数，然后直接运行eval：

```bash
uv run python -m sources.skill.code.eval \
  --config-name=pettingzoo_mw \
  ++environment.env_tweak.disturbance_mode=adaptive \
  ++environment.env_tweak.disturb_target_agent=0 \
  ++environment.env_tweak.disturb_magnitude=0.3 \
  ++environment.env_tweak.disturb_start_step=100
```

## 输出结果

### 单个配置测试

运行后会在 `results/single_disturbance_test/<timestamp>/` 目录下生成：

```
results/single_disturbance_test/20260113_143022/
├── result.json              # 完整的评估结果
└── angle_time_plot.png      # 角度-时间变化图
```

`result.json` 包含：
```json
{
  "disturbance_mttf_avg": 45.2,           // 平均失效时间（步数）
  "disturbance_recovery_time_avg": 23.5,  // 平均恢复时间（步数）
  "disturbance_max_angle": 8.7,           // 最大倾斜角度（度）
  "terminate_cnt": 2,                      // 提前终止次数
  "angle_data_grouped": [[...], [...]]    // 每个episode的角度数据
}
```

### 批量测试

运行后会在 `results/batch_disturbance/<timestamp>/` 目录下生成：

```
results/batch_disturbance/20260113_143022/
├── agent_0_mag_0.1.json
├── agent_0_mag_0.1_angle.png
├── agent_0_mag_0.2.json
├── agent_0_mag_0.2_angle.png
├── ...
├── agent_2_mag_0.5.json
├── agent_2_mag_0.5_angle.png
├── summary.json              # 所有结果汇总
└── summary_report.txt        # 文本格式的汇总报告
```

## 关键参数说明

### 扰动配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `disturbance_mode` | 扰动模式，设为 'adaptive' 启用自适应扰动 | None |
| `disturb_target_agent` | 目标 agent ID (0-2) | -1 |
| `disturb_magnitude` | 扰动幅度（加到动作上的值） | 0.0 |
| `disturb_start_step` | 开始注入扰动的步数 | 100 |

### 失效和恢复阈值

在 `pettingzoo_mw_env.py` 中定义：

```python
self.failure_threshold = 6.5  # 失效角度阈值（度）
self.recovery_threshold = 5.0  # 恢复角度阈值（度）
self.failure_consecutive_steps = 10  # 连续多少步超过阈值算失效
```

### 三个关键指标

1. **MTTF (Mean Time To Failure)**: 从注入扰动到失效的平均步数
   - 失效定义：连续10步角度超过6.5°
   - 数值越大表示系统越稳定

2. **Recovery Time**: 从取消扰动到恢复的步数
   - 恢复定义：角度降到5°以下
   - 数值越小表示恢复能力越强

3. **Max Angle**: 整个episode中的最大倾斜角度
   - 数值越小表示扰动影响越小

## 可视化图表说明

每个配置生成的角度-时间图包含：
- **蓝色/橙色/绿色曲线**: 不同episode的角度变化
- **红色虚线** (y=6.5): 失效阈值线
- **橙色虚线** (y=5.0): 恢复阈值线
- **绿色点线** (x=100): 扰动注入时间点

## 扩展和自定义

### 修改测试范围

编辑 `batch_disturbance_test.py`:

```python
# 测试更多agent
agents = [0, 1, 2, 3]  # 如果环境有4个agent

# 测试更多幅度
magnitudes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# 测试不同的起始时间
cfg.environment.env_tweak.disturb_start_step = 200
```

### 修改失效判定条件

编辑 `packages/HARL/harl/envs/pettingzoo_mw/pettingzoo_mw_env.py`:

```python
self.failure_threshold = 8.0  # 提高失效阈值
self.failure_consecutive_steps = 15  # 需要连续15步
```

### 添加更多指标

在 `_calculate_metrics()` 方法中添加新指标，并在logger中收集。

## 故障排查

### 问题：Hydra配置错误

```
Error: Could not find config file
```

**解决方案**: 确保在项目根目录运行脚本，并且配置文件路径正确。

### 问题：环境初始化失败

```
Error: Could not create environment
```

**解决方案**: 检查环境依赖是否安装完整：
```bash
uv sync
```

### 问题：没有生成图表

**解决方案**: 
1. 检查 matplotlib 是否安装
2. 确保 `angle_data_grouped` 在结果中存在
3. 查看控制台输出的错误信息

## 性能说明

- 单个配置测试时间：约5-10分钟（取决于eval_episodes设置）
- 批量测试（15组）总时间：约1.5-2.5小时
- 可以通过减少 `eval_episodes` 来加快测试速度（用于调试）

## 下一步

1. 运行单个配置测试验证功能
2. 检查生成的结果和图表
3. 根据需要调整参数
4. 运行完整的批量测试
5. 分析汇总报告，比较不同配置的效果
