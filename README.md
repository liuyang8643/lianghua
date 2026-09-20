# WBR 量化交易系统

当前PPO默认：100000轮、20env、n_steps=64、batch_size=640、n_epochs=3、固定learning_rate=3e-4、gamma=0.99、gae_lambda=0.95、target_kl=None、seed=None。其余算法设置继承SB3；标准DiagGaussianDistribution、Linear mean、log_std_init=0，环境动作裁剪到Box[-1,1]后由唯一ActionSchema映射到11个[0,1]权重与[0,0.2]换股比例。


## 环境要求

- Windows + Python 3.12 + [QMT 客户端]
- uv 包管理器

### 安装

```powershell
# 安装 uv
irm https://astral.sh/uv/install.ps1 | iex

# 安装依赖
uv sync
```

## 实盘运行

1. 配置 `configs/env.py`（QMT路径、账号等）
2. 登录 QMT，勾选极简模式
3. 使用已冻结且通过部署校验的 bundle 和本地 runtime 启动：

```powershell
.\run.ps1 -Bundle artifacts/policies/selected -Runtime data/runtime/sealed.npz
```

新账户链需显式传入 `-InitializeNewChain`；已有账户链从 journal 恢复实际成交与配置历史。

## 开发指引

### 单回测

```bash
uv run python -m testback.run_backtest \
  --start-date 20240101 --end-date 20241231 \
  --individual-config configs/config.json
```

### GA 参数搜索

```bash
uv run python -m ai.ga.train --mode ga --runtime data/runtime/runtime_1990-12-19_2026-08-28_rawstate.npz
```

GA默认训练10000代，并从GA/PPO共用的 `configs/evaluation_splits.json` 读取三段诊断日期。可用 `--evaluation-splits <JSON文件>` 覆盖，不要求先训练PPO；该参数与 `--ppo-reference` 互斥，二者共用同一个评估器。每10代及结束评估当代训练冠军，按验证Calmar选择，测试仅诊断。debug模式不隐式打开holdout。快速运行可显式指定 `--generations 100 --population-size 32 --workers 20`。

PPO默认100000个vector rollout。共同分期为训练2004-04-28～2017-12-31、验证2018–2022、测试2023–2026-08-28。长动量固定使用[T-252,T-21)，允许跳过内部缺失的日收益，但首尾必须有效且231行中至少185行有效；不补造价格、不压缩时间轴或外推收益。factor v9下，2004-04-28是11因子各自首次均覆盖至少50%当时成员的日期，用作最早实用起点；2003年已有极稀疏财务值，不当作广覆盖起点，50%也不是后续每日股票池限制。更长历史不保证更好泛化，已查看年份仍是研究诊断。因子或分期改变需新随机root，旧模型、种群和缓存不能作为新分期续训；旧结果按原日期独立保留。PPO在初始化、每50轮及结束评估，GA每10代及结束评估，两者轮/代不代表相同计算预算。

### 添加新因子

研究候选由 `factor_db.discovery` 动态发现并登记，已登记的候选源码保留血缘，不按生产训练是否使用来删除。接纳的因子由 `factor/registry.py` 显式注册；`ActionSchema` 是权重、买卖数量范围及固定控制的唯一来源。`configs/strategy.yaml` 只保留运行配置，静态对照使用 `configs/config.json`；实盘读取 bundle 内冻结的配置。


十一因子 PPO 研究入口（下列命令启动全新随机训练，不用于旧十因子模型续训）：

```powershell
.venv/Scripts/python.exe -m ai.rl.train --runtime data/runtime/runtime_1990-12-19_2026-08-28_rawstate.npz --output artifacts/rl/ppo --rollouts 100
```

当前十一因子在原十项末尾加入高异常毛利润 `HighAbnormalGrossProfit`，GA与PPO均学习其连续[0,1]权重；静态对照新权重为0。公式为（单季毛利润−去年同季毛利润×单季销售收现同比比例）/同季末总资产，公告严格早于决策日，缺失保持NaN。二值因子不参与截面排名，连续因子同分取平均排名。`factor_coverage_train.json` 记录各年覆盖率及全缺失区间，`ppo_diagnostics.jsonl` 记录每轮 loss/KL 与奖励、优势分位数。

GA/PPO共用固定50只目标持仓，`sell_m` 已替换为换股比例 `turnover_rate`。生产搜索比例范围0～0.2，每天最多检查10只；检查数量为 `floor(50 × 比例)`（处理整股边界的浮点舍入）：先取持仓中当日因子排名最差的X只，只换出其中跌出完整PIT股票池前50名且可全部卖出的股票，锁仓不递补检查更好的持仓。买入继续服从原有候选池、过滤器和交易合法性。冷启动可建满50只；每日等权与现金sweep保持，因此比例限制的是替换只数，不是资金换手率硬上限。

动作共12维：11个独立因子权重及1个连续换股比例。GA/PPO均从 `ActionSchema` 读取范围、规范化精度和编解码；PPO使用SB3标准高斯动作头，不保留类别换股或全局/残差动作头。

当前actor输入为完整股票轴、64日原始状态：19个历史字段（已完成K线、原始财报和报告季度）加18个当日已知字段（开盘、股本、发行与生命周期、交易合法性和财报时效）。因子排名和过滤信号不进入actor；原始持仓、现金和实际配置/成交历史继续保留。财报按公告版本因果回放、保留原始负值；normalizer只在训练集拟合。因子计算窗口不受actor的64日输入限制。详细字段见[输入报告](artifacts/raw_state_only_20260917/report.md)。

训练、冻结评估和实盘推理共用一个原始序列Transformer及CUDA执行路径，原始数据通过 `RawMarketStore` 共享，窗口引用在可学习层前移除。可微更新不跨梯度步骤缓存学习结果。仅保留当前Observation/encoder/network契约，形状或语义变化必须新随机root，没有旧checkpoint重解释或CPU fallback。

当前PPO默认：100000轮、20env、n_steps=64、batch_size=640、n_epochs=3、固定learning_rate=3e-4、gamma=0.99、gae_lambda=0.95、target_kl=None、seed=None。其余算法设置继承SB3；标准DiagGaussianDistribution、Linear mean、log_std_init=0，环境动作裁剪到Box[-1,1]后由唯一ActionSchema映射到11个[0,1]权重与[0,0.2]换股比例。

PPO只保留固定周期评估：随机初始化、第50/100/150…轮及结束时，用同一checkpoint分别回放训练、验证、测试完整周期。每个split只准备一次并驻留只读共享内存，评估按顺序串行执行。仅按验证Calmar选模，测试仅诊断，不参与梯度、训练奖励或选模；静态配置仅作参考。没有训练成绩门槛、解封状态、一次测试限制或Calmar 1.5资格门槛。已反复观察的测试期不称为盲测。保留有限值、满仓、数据因果、schema/source/hash及同身份线性续训检查。

各split只准备一次并共享只读数据；账户链逐日串行更新。完整历史预计算后统一通过 `PreparedEpisode.compact_for_replay()` 保留本段与64日输入所需前置数据；投影不能重算长周期因子。GA/PPO共用唯一env交易核。

## 当前目录与边界

| 路径 | 当前职责 |
|---|---|
| `offline_data/` | 本地快照、PIT公告版本、runtime读取和身份校验 |
| `data/` | 仍在使用的数据源下载、更新和runtime构建；同时存放本地数据产物 |
| `factor/` | 固定生产词表、公共因子计算和有效性；共享数值函数 |
| `factor_db/` | 仍被生产注册引用的旧因子，以及动态候选库、登记和研究记录 |
| `env/` | 唯一账户链、评分、合法性、调仓、成交、收益和Observation；Gym类型仅在适配层 |
| `ai/ga/`、`ai/rl/` | GA静态配置搜索、PPO动态决策，各自学习循环共用env |
| `ai/factor_discovery/` | 离线因子与行业研究入口 |
| `trade/` | 券商适配、实盘装配、成交journal和只读replay |
| `testback/` | 固定策略回测入口与只读报告；不另写账户算法 |
| `configs/`、`utils/` | 声明式配置及共享文件I/O、进程/日志等边缘工具 |
| `tests/` | 离线合成、因果、交易一致性、资源生命周期和身份测试 |
| `artifacts/`、`results/` | 当前及旧运行结果、模型与压缩源码证据；非运行中的展开源码副本已清理 |
| `ai/reporting.py`、`ai/report_server.py`、`ai/report_assets/` | GA/PPO共用的报告契约、单一服务与页面；从运行记录读取指标 |
| `dashboard/` | 数据覆盖率页面，与训练报告职责不同 |

`data/`和`offline_data/`、`factor_db/`和`factor/`尚处于明确的职责迁移中，不能把仍用的旧路径直接删除。旧`core/`目前只剩数据/缓存/结果，`trading/`只剩编译缓存；历史实验源码归档后不参与运行。目标迁移关系见 [PLAN.md](PLAN.md)。

代码精简记录、各模块审查、删除证据和验收结果见 [本轮报告](artifacts/code_cleanup_20260917/report.md)。

## 统一训练报告

```powershell
uv run python -m ai.report_server --config configs/training_reports.json --port 8766
```

一个只读服务显示多个GA/PPO运行，共用训练/验证/测试曲线、配置、权重、诊断和CSV导出。每项清单显式指定 `id`、`algorithm`、`output_dir`、`log_path`、`trace_dir`，相对路径以清单目录为基准；不再猜测root/run目录。

页面读取每个运行已保存的词表、预算、选择结果与曲线，不按当前默认值解释旧实验，不重新计算收益或调用历史源码。已生成逐日trace用统一controls结构展示；缺少trace时直接显示未保存。GA代数与PPO轮数不代表相同计算预算。

主代码只维护当前实现。旧模型、收益数据与源码压缩包作为审计证据保存；只有正在运行的训练保留其唯一冻结执行目录，避免spawn worker中途加载改变后的代码。旧源码归档没有自动解压/导入/回放入口。
