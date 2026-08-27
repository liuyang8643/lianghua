# WBR 目标架构与开发准则

同时参考 @CLAUDE.md。本文规定后续重构的目标架构和模块命名；`CLAUDE.md` 中关于离线回测、数据因果、T 日价格、回测/实盘一致性和性能的语义红线继续生效。路径归属冲突时以本文和 [PLAN.md](PLAN.md) 为准：每项职责迁移、验证并删除旧入口前由 `data/core/trading` 当前路径执行，完成后由 `offline_data/env/trade` 对应路径继承同一红线。迁移不扩大多进程/多线程权限，PPO 默认单进程运行。

## 1. 架构目标

统一决策链路：

```text
截至决策时刻 T 的全部有效数据
  -> Observation(T)
  -> GA / RL Policy
  -> DayConfig(T)
  -> env 生成 OrderPlan
  -> 模拟成交 / 实盘成交
  -> Fill
  -> 账户更新、收益与 Reward
```

- GA 和 RL 都是参数决策器，共用 `Policy.predict(Observation) -> DayConfig`。
- GA 输出搜索得到的静态 `DayConfig`；RL 根据时间窗口、市场和持仓状态动态输出 `DayConfig(T)`。
- 回测和实盘共享选股、合法性、调仓、账户状态转换等领域逻辑，只替换成交执行器。
- RL 可动态控制的范围由统一的 `DayConfigSchema` 决定。
- 训练、验证、测试和回测只读取 `offline_data` 生成的本地不可变数据快照。

## 2. 目标目录

```text
WBR/
├── ai/
│   ├── ga/
│   ├── rl/
│   ├── factor_discovery/
│   ├── policy.py
│   └── bundle.py
├── offline_data/
│   ├── contracts.py
│   ├── sources/
│   ├── update.py
│   ├── store.py
│   └── runtime.py
├── factor/
│   ├── base.py
│   ├── registry.py
│   └── library/
├── env/
│   ├── contracts.py
│   ├── action_schema.py
│   ├── observation.py
│   ├── encoder.py
│   ├── planner.py
│   ├── simulator.py
│   ├── fees.py
│   ├── gym_adapter.py
│   ├── backtest.py
│   └── metrics.py
├── trade/
│   ├── runtime.py
│   ├── executor.py
│   ├── broker/
│   └── journal.py
├── configs/
├── artifacts/
└── tests/
```

目录表示稳定职责边界，不作为文件数量目标。只有出现真实实现时才创建文件或子目录，不预建空基类、空目录和无调用方的抽象层。

## 3. 模块职责

| 模块 | 核心定位 | 主要职责 | 核心输入 | 核心输出 |
|---|---|---|---|---|
| `offline_data` | 统一数据底座 | 数据下载、更新、清洗、PIT 对齐、本地存储、runtime 矩阵和快照构建 | 外部数据源、本地历史数据 | `MarketSnapshot`、runtime、schema、manifest |
| `factor` | 因子计算层 | 因子定义、计算、注册、元数据和有效性 mask | `offline_data` 的只读数据视图 | `FactorBatch` |
| `env` | 唯一策略与回测领域核 | Observation、动作解码、评分、选股、合法性、调仓、账户转换、收益、Reward 和回测 | 数据快照、因子、账户、`DayConfig` | `Observation`、`OrderPlan`、`StepResult` |
| `ai` | 参数优化与动态决策 | GA 搜索、SB3 PPO 训练、评估、模型加载、动态推理和因子发现 | `Observation`、训练环境、参数空间 | GA/RL 模型、`DayConfig` |
| `trade` | 实盘适配与执行 | 实盘调度、券商连接、订单执行、成交回报、对账、持久化和运行报告 | 模型、账户、实时快照、`OrderPlan` | 真实 `Fill`、执行记录和审计日志 |

辅助目录：

| 目录 | 用途 |
|---|---|
| `configs` | 数据路径、训练范围、环境常量和部署选择等声明式配置 |
| `artifacts` | 模型、schema、normalizer、评估结果和版本记录 |
| `tests` | 单元、因果、路径一致性和实盘 replay 测试 |

## 5. 依赖与解耦

代码依赖保持单向 DAG。箭头表示左侧使用右侧的公开接口：

```text
trade -> ai + env + offline_data
ai    -> env；训练装配入口可读取 offline_data
ai/factor_discovery -> factor 的公开契约
env   -> factor + offline_data
factor -> offline_data 的只读数据契约
offline_data -> 基础库和外部数据源适配器
```

- `factor` 接收只读数据视图并返回因子结果，数据下载由 `offline_data` 统一完成。
- `env` 通过公开契约使用数据和因子，是策略语义的唯一所有者。
- `ai` 通过 env 契约训练和推理，不复制评分、选股、合法性、成交或 Reward 逻辑。
- `ai/factor_discovery` 仅通过 `factor` 的公开契约校验和注册候选因子。
- `trade` 调用 ai 推理和 env 领域核；可被回测复用的逻辑应沉淀到 `env`。
- 运行时允许 `Observation -> Policy -> DayConfig -> env` 的反馈循环，但不能由此形成循环 import。
- 根目录入口或 CLI 只负责对象装配和参数传递，不承载业务逻辑。
- 跨模块只使用公开契约，不读取其他模块的内部缓存、私有函数或临时表结构。

## 6. 公共契约

| 契约 | 所属模块 | 含义 |
|---|---|---|
| `MarketSnapshot` | `offline_data` | 截至指定决策时刻可使用的数据快照 |
| `FactorBatch` | `factor` | 因子矩阵、有效性 mask 和因子版本 |
| `Observation` | `env` | GA/RL 的标准输入 |
| `DayConfig` | `env` | 某个决策日采用的强类型策略配置 |
| `DayConfigSchema` | `env` | 参数名称、类型、范围、默认值和约束的唯一来源 |
| `AccountState` | `env` | 现金、持仓、成本、净值等账户状态 |
| `OrderPlan` | `env` | 与券商无关的目标订单计划 |
| `Fill` | `env` | 模拟执行器和券商适配器共用的标准成交结果 |
| `StepResult` | `env` | 新账户状态、收益、Reward 和诊断信息 |
| `Policy` | `env` 定义契约，`ai` 实现 | `predict(Observation) -> DayConfig` |
| `ExecutionPort` | `env` 定义契约 | `execute(OrderPlan) -> list[Fill]`；由模拟和实盘执行器实现 |
| `PolicyBundle` | `ai` | 可加载的模型运行单元 |

契约对象应保持明确、稳定、可序列化；跨模块数据变化通过 schema 版本表达，不依赖隐式全局状态。

## 7. GA 与 RL 的统一接口

```python
class Policy:
    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> DayConfig:
        ...
```

| 实现 | 行为 |
|---|---|
| `GAPolicy` | 返回搜索得到的静态 `DayConfig` |
| `RLPolicy` | 将 Observation 输入 PPO，每个决策时刻输出动态 `DayConfig` |
| `FixedPolicy` | 返回人工配置，用于基准回测和排错 |

`env` 和 `trade` 不根据策略类型分支。`ai/factor_discovery` 负责候选因子发现；候选因子通过因子契约和测试后再进入 `factor` 注册表。

## 8. RL 输入标准

`Observation` 表示截至决策时刻 T 已真实可获得、且在当前 `ObservationSchema` 注册的全部数据维度。

| 字段 | 推荐形状 | 内容 |
|---|---:|---|
| `stock_panel` | `[L, N, F]` | 股票历史行情、财务、资金和因子时序特征 |
| `market_panel` | `[L, M]` | 指数、市场宽度、风格、宏观和风险状态 |
| `position_panel` | `[N, H]` | 每只股票的持仓权重、成本、盈亏和可卖数量 |
| `portfolio` | `[P]` | 现金、总仓位、净值和回撤等组合状态 |
| `feature_mask` | `[L, N, F]` | 每个输入值是否真实存在，区分缺失值与真实零值 |
| `stock_mask` | `[L, N]` | 股票当日存在且具备有效开盘状态；不包含可动态关闭的软过滤条件 |
| `time_mask` | `[L]` | 历史长度不足时的时间 mask |
| `schema_version` | 标量 | Observation 结构版本 |

- 一个模型版本的 `L、N、F、M、H、P` 固定，历史不足使用 padding 和 mask。
- “全部数据”指 schema 明确登记的全部因果维度，不接受运行时列顺序或维度静默变化。
- 新增、删除或改变输入字段时升级 schema，并重新训练对应模型。
- Normalizer 仅使用训练集拟合，训练、回测和实盘使用同一版本。
- 决策输入严格遵守字段实际可用时间；决策后才产生的数据只进入后续结算。
- 固定维编码不得只保留每个字段的边际分布；至少保留因子与历史特征、因子之间以及持仓与因子暴露的联合关系。当前四因子主 schema 编码为 3306 维（静态市场 3222、动态账户 84），新增字段或改变联合编码必须升级 schema 并重训。

## 9. RL 输出标准

首版使用统一连续 `Box[-1, 1]^D` 动作空间，通过 `ActionSchema` 解码为强类型 `DayConfig`：

| 参数类型 | 解码方式 | 示例 |
|---|---|---|
| 连续 | 映射到声明范围 | 因子权重、目标仓位、现金比例 |
| 二值 | 阈值解码 | 因子启用、是否调仓、过滤器开关 |
| 有序离散 | 区间分桶 | `buy_n`、`sell_m`、持仓数量 |
| 枚举 | 在统一连续坐标上区间分桶 | 调仓模式、选股模式 |

典型动态配置包括：

```text
factor_weights
factor_enabled
target_exposure
buy_n
sell_m
rebalance_now
filter_flags
其他已注册且能被当日策略实际消费的参数
```

- `DayConfigSchema` 是动态参数名称、类型、范围、默认值、约束和解码规则的唯一来源。
- GA、PPO 训练、模型回测和实盘使用同一个 schema、校验器和动作解码器。
- 因子权重统一进行有效性处理和归一化；离散及二值输出最终转换为明确类型。
- 原则上，能够因市场状态变化而调整、具备因果输入且会影响当日决策的策略参数，都可注册为动态参数。
- 数据口径、PIT 股票池规则、手续费、滑点、公司行动、Reward 定义和训练测试边界属于固定环境语义。
- RL 学习的是 schema 暴露的策略自由度，不在运行时修改策略代码。新增可学习行为时先增加明确参数，再训练新模型。

## 10. env 执行模型

`env` 是 Observation、策略、订单、账户、收益和 Reward 的唯一领域核。组合入口向 runner 注入 `Policy` 和 `ExecutionPort`：

```text
build_observation(snapshot, account)
  -> policy.predict(observation)
  -> validate_day_config(config)
  -> plan_open(decision_snapshot, account, config)
  -> OrderPlan
  -> injected ExecutionPort
  -> Fill
  -> settle_next_open(next_open_snapshot, account, fills)
  -> StepResult
```

| 阶段 | 职责 |
|---|---|
| `build_observation` | 构造严格截至 T 的模型输入 |
| `plan_open` | 因子评分、选股、合法性、卖出、买入和仓位规划 |
| `ExecutionPort` | 接收 `OrderPlan` 并返回标准 `Fill` |
| `settle_next_open` | 在下一决策开盘处理公司行动、账户估值、收益和 Reward |
| `backtest` | 按交易日驱动同一领域核 |
| `gym_adapter` | 将领域核适配成 Gymnasium `reset/step` |

`env.SimExecutor` 和 `trade.BrokerExecutor` 都实现 `ExecutionPort`，由外层入口创建并注入；`env` 不 import、创建或判断 `trade` 实现。`plan_open` 的对象图只包含决策时刻可用数据。动作 T 的 Reward 默认按 T-open 到 T+1-open 结算；`close[T]` 不参与 T 日决策或该动作的估值，最早作为历史字段进入 T+1 Observation。Gym/SB3 类型停留在适配层，不渗入评分、选股、账户等领域函数。

当前离线 runtime 没有现金分红、送转、拆并股和配股明细，因此训练/回测采用版本化 `total_return_reinvested-v1` 合成账户：通过 `close[T] / preClose[T+1]` 保持下一开盘经济价值，并明确 `broker_exact=false`。该会计口径、费用、股票池、planner、Reward 和 ActionSchema 必须共同写入并校验 policy bundle 的 environment schema。合成股数不得直接 replay 为券商持仓；实盘迁移必须通过公司行动明细或券商日初股数/现金变动生成真实账户转换。

## 11. 标准数据流

| 场景 | 标准流程 |
|---|---|
| 数据更新 | 外部数据源 -> `offline_data.update` -> 清洗/PIT 对齐 -> 本地 store -> snapshot |
| GA 训练 | 本地快照 -> env 回测 -> GA 搜索静态 `DayConfig` -> `GAPolicy` |
| PPO 训练 | 本地快照 -> Gym adapter -> Observation -> PPO -> DayConfig -> Reward |
| 模型回测 | 加载模型 -> 确定性推理 -> 同一 env -> `SimExecutor` |
| 实盘 | 加载冻结模型 -> 决策前更新并冻结快照 -> Observation -> `deterministic=True` 推理 -> DayConfig -> OrderPlan -> 券商执行 |
| 盘后复盘 | 读取已记录的 Observation、动作、订单和成交进行 replay，不重新推理 |

数据更新入口和实盘券商适配器负责联网；因子、env、GA/PPO 训练和回测只读取本地数据。实盘数据先形成带版本的决策快照，再进入策略链路。实盘进程只加载冻结模型并确定性推理，不在线训练或更新模型参数。Bundle 只有在训练平台期、固定配置基线、验证集非劣性、有限值、最低仓位以及连续/二值/离散动态性门槛全部通过后才可加载；测试集仅在模型冻结且训练/验证门槛通过后打开一次用于最终报告。

## 14. 解耦与代码精简

- 股票池、因子、Observation、动作解码、评分、合法性、调仓、费用、账户转换、收益和 Reward 各只有一个权威实现。
- 核心计算优先使用确定性纯函数；网络、磁盘和券商副作用集中在边缘适配器。
- 新抽象应隔离真实外部依赖、服务至少两个调用方或实质消除重复，否则使用局部函数。
- 文件按共同变化的职责组织，不预建空层，不把领域逻辑堆入通用 `utils`。
- 默认值只定义在对应 schema；YAML 和 CLI 只覆盖，不复制默认值。
- 新路径验证后同步整理被替代的旧入口、旧参数、兼容分支、无引用文件和依赖，避免长期双轨。

## 15. 每次 AI 开发后的 Review

每次由 AI/Codex 完成功能开发或重构后，由独立 reviewer 检查：

1. 职责是否放在唯一正确模块，依赖是否仍为单向 DAG。
2. 是否产生第二套数据、Observation、策略、回测或实盘实现。
3. 旧类、旧函数、旧参数、重复默认值和无引用文件是否已整理。
4. 相同 snapshot/account/config 在训练、回测和实盘是否生成一致结果。
5. 是否满足时间因果、schema 版本、可复现性及本次变更所需测试。
6. 是否引入重复计算、逐股票循环、不必要复制或没有真实价值的抽象。

交付说明至少列出：`新增`、`复用`、`替换/删除`、`依赖变化`、`schema 变化`、`验证结果`、`冗余检查`和`独立 reviewer 结论`。功能可运行但仍保留两套同职责实现，不视为完成。
