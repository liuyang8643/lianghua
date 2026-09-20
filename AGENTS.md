# WBR 目标架构与开发准则

最新调仓覆盖（2026-09-19）：用户撤回关闭再平衡的要求，恢复唯一daily_equalize_then_cash_sweep实现及environment v36/planner v6，与当前冻结训练一致。超配减仓、低配补买、排名换股和现金sweep保留；单边滑点仍为0.0025。

最新费用覆盖（2026-09-19）：GA、PPO rollout、全部评估和当前静态基准统一单边slippage_rate=0.0025，唯一默认值由env.fees.FeeSchedule定义，PPO CLI引用该值。佣金、过户费、印花税不变；旧0.001运行保留审计，不沿旧费用身份续训。

最新用户覆盖（2026-09-19）：GA与PPO默认rollout worker统一20，由configs.training.DEFAULT_ROLLOUT_WORKERS定义。PPO默认episode_scope=full，账户从完整训练期起点连续推进到终点后才reset；n_steps=64保持，每个环境采64条后执行PPO更新，不要求先采完整训练期才更新。此次新训练沿用batch640、3epochs、固定3e-4及每50轮三段评估，使用新随机root并封存full采样身份。

当前PPO默认：100000轮、20env、n_steps=64、batch_size=640、n_epochs=3、固定learning_rate=3e-4、gamma=0.99、gae_lambda=0.95、target_kl=None、seed=None。其余算法设置继承SB3；标准DiagGaussianDistribution、Linear mean、log_std_init=0，环境动作裁剪到Box[-1,1]后由唯一ActionSchema映射到11个[0,1]权重与[0,0.2]换股比例。

PPO只保留固定周期评估：随机初始化、第50/100/150…轮及结束时，用同一checkpoint分别回放训练、验证、测试完整周期。每个split只准备一次并驻留只读共享内存，评估按顺序串行执行。仅按验证Calmar选模，测试仅诊断，不参与梯度、训练奖励或选模；静态配置仅作参考。没有训练成绩门槛、解封状态、一次测试限制或Calmar 1.5资格门槛。已反复观察的测试期不称为盲测。保留有限值、满仓、数据因果、schema/source/hash及同身份线性续训检查。

每个root在首个checkpoint前封存源码、runtime lineage、三段日期、schema、normalizer、网络与Reward语义；verified resume仅允许相同契约及canonical latest线性续训，重复诊断测试不阻止同身份续训。

最新用户覆盖（2026-09-18）：移除自定义 Beta 动作头，唯一 PPO 路径使用 SB3 标准 DiagGaussianDistribution / Linear mean / log_std_init=0；不再维护 Beta 浓度、Beta 分布或兼容加载。标准 SB3 对环境和公开 predict 输出裁剪到 Box[-1,1]，统一 ActionSchema 映射因子权重到[0,1]、换股比例到[0,0.2]；训练 buffer 保存原始未裁剪采样及对应高斯 log-prob。ActionSchema v19 明确全零权重合法：所有得分相同，复用现有稳定排序、合法性与满仓流程，不暗中启用因子。environment v36、action distribution v14、bundle v44及run identity v87拒绝旧身份续训。账户asinh、64天实际配置/成交历史输入、3epochs、64steps、batch640、20env和固定3e-4保留；仅修改实现与测试，不自动开启新训练。




## 最新覆盖：生产换股上限与更新时机（2026-09-18）

用户明确GA/PPO换股比例范围为[0,0.2]，固定buy_n=50下最多检查最差10只持仓。ActionSchema v18唯一codec通过显式turnover_maximum登记范围（生产默认0.2），已存在的固定100%换股研究入口显式声明1.0，仍复用同一领域核，不作为生产兼容分支。范围写入schema哈希，旧checkpoint不得重解释或续训。归一化Beta坐标仍为[0,1]，看板必须标明坐标与真实比例的区别。

用户确认保留20env×64steps=1280条采样、batch256、10epochs：每个minibatch立即执行一次optimizer.step，每个epoch通常5次、每轮最多50次，target-KL触发时可能提前停止；不改成每收到256条环境样本就打断采样。actor输入窗口仍64天，两个64含义不同。

## 最新覆盖：原始状态输入（2026-09-17）

用户要求只维护当前一套逻辑：生产代码无旧动作/输入/报告兼容分支；报告服务仅只读已保存结果，不执行历史源码。注册表明确output_dir/log_path/trace_dir。非运行中的冻结源码压缩归档，移除展开目录；当前训练只保留其正在使用的唯一冻结执行树，不能热改其source identity。历史checkpoint/报告/数据保留审计属性，旧源码无自动解压或执行入口。具体冗余删除与独立审查见artifacts/code_cleanup_20260917/report.md。

用户随后明确授权时间窗口从 504 缩为 64 个交易日；当前默认 DEFAULT_LOOKBACK=64，新 root 封存 lookback=64。它仅改变模型可见的历史长度，不缩短 env 的因子计算窗口或 IPO 生命周期计算。504 日候选在首次 rollout 前停止，保持独立记录。

用户授权 actor 删除全部因子排名与 soft-filter 信号，11 因子继续仅作为 DayConfig 输出及 env 评分依据。Observation/encoder v21、normalizer v13、raw network v3 使用 19 个历史字段（7 个已完成行情、9 个原始财务源字段、3 个报告季度）和 18 个当前已知字段（股本、发行资料、生命周期、开盘合法性、财报时效等）。保留完整股票轴、64 日窗口及原始账户和实际动作历史。财务使用同一公告事件回放，保留负值和源 YTD 口径，同日公告不进入当日开盘；旧 13 个因子财务面板不变。生命周期与因子复用唯一状态链；裁剪和 shared 路径从同一预计算静态窗口获取公开 Observation。

新输入必须新随机 root，禁止旧输入 checkpoint 兼容路径。每批默认 10 epochs，固定学习率 3e-4，与 SB3 这两个默认值一致，不启用衰减。显式启用学习率衰减时绝对 transition 预算写入 run identity，循环读取封存预算，不使用分段 learn 的局部进度；当前衰减模式只支持新 root，拒绝 resume。此前 1 epoch 与学习率衰减候选已停止，新 root 使用上述 10 epochs 与固定学习率。原始源快照、审查、性能证据见 artifacts/raw_state_only_20260917。

同时参考 @CLAUDE.md。本文规定后续重构的目标架构和模块命名；`CLAUDE.md` 中关于离线回测、数据因果、T 日价格、回测/实盘一致性和性能的语义红线继续生效。路径归属冲突时以本文和 [PLAN.md](PLAN.md) 为准；迁移不得越过本文明确的离线并行边界。

历史收敛实验记录保留在 `artifacts/ppo_convergence_20260918`，其旧训练配置不是当前默认。

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
- RL 可动态控制的范围由统一的 `ActionSchema` 决定。
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
| `ActionSchema` | `env` | 参数名称、类型、范围、默认值、约束和动作编解码的唯一来源 |
| `AccountState` | `env` | 现金、持仓、成本、净值等账户状态 |
| `PolicyMemory` | `env` | 上一期规范化配置、实际成交总换手率和总成本率组成的一步因果决策记忆 |
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
| `FixedPolicy` | 承载 GA 搜索结果或人工配置，返回静态 `DayConfig` |
| `RLPolicy` | 将 Observation 输入 PPO，每个决策时刻输出动态 `DayConfig` |

`env` 和 `trade` 不根据策略类型分支。`ai/factor_discovery` 负责候选因子发现；候选因子通过因子契约和测试后再进入 `factor` 注册表。

## 8. RL 输入标准

`Observation` 表示截至决策时刻 T 已真实可获得、且在当前 `ObservationSchema` 注册的全部数据维度。

| 字段 | 推荐形状 | 内容 |
|---|---:|---|
| `stock_panel` | `[L, N, F]` | 股票历史行情、已证明 PIT 的字段、资金、因子排名和两个 soft-filter 通过状态 |
| `market_panel` | `[L, M]` | 市场宽度、风格、宏观、风险，以及严格因果的过滤器压力与因子历史表现 |
| `position_panel` | `[N, H]` | 每只股票的持仓权重、成本、盈亏和可卖数量 |
| `portfolio` | `[P]` | 现金、总仓位、净值、回撤及上一期因果决策记忆 |
| `time_mask` | `[L]` | 历史长度不足时的时间 mask |
| `schema_version` | 标量 | Observation 结构版本 |

- 一个模型版本的 `L、N、F、M、H、P` 固定，`N` 必须等于 runtime 的完整股票轴；历史不足只允许使用时间 padding 和 `time_mask`。
- 模型输入不提供、不随机生成 `feature_mask` 或 `stock_mask`，股票张量始终保留固定完整轴。encoder 内部必须使用唯一的 PIT membership sidecar：`listing_age[T] >= 0 and not delisted_mask[T]`；停牌仍属于当日成员，持仓账户聚合使用 `PIT member or held`。所有股票截面、因子联合、持仓截面和 market-factor 参考分母都只在该时点成员集合内归一，sidecar 不进入 feature names 或 actor tensor，也不改变股票轴。局部缺失值只在当日 PIT 成员集合内按字段可用规则确定性 zero-fill；因子有效性、停牌和涨跌停等语义仍在领域层保留。追加未来上市股票列不得改变其上市前任一 raw/cached actor 编码或训练 normalizer 参数；normalizer 的账户尺度只能使用训练期逐日 PIT 成员数，不得使用 runtime 最终股票总数。两个 soft-filter 保留完整股票轴；小型 MLP 通过 20 维 pass/reject 压力消费它们，不机械扩张通用 moments。
- “全部数据”指 schema 明确登记的全部因果维度，不接受运行时列顺序或维度静默变化。
- 新增、删除或改变输入字段时升级 schema，并重新训练对应模型。
- Normalizer 仅使用训练集拟合，训练、回测和实盘使用同一版本。
- 决策输入严格遵守字段实际可用时间；决策后才产生的数据只进入后续结算。
- 固定维编码不得只保留每个字段的边际分布；至少保留因子与历史特征、因子之间以及持仓与因子暴露的联合关系。当前八因子 stationary 主 schema 编码为 2176 维（静态市场 2052、动态账户 124）：静态部分由 1944 维股票/联合时序关系和 108 维严格因果市场最新状态组成；股票行情、规模及因子状态必须先转换为截面 rank、相对比例、变化率或有界相对量等平稳语义，不向 actor 暴露可充当年份代理的绝对水平。当前财务源只有最新修订值，没有历史公告/修订版本，`bps/eps/roe/profit_yoy/revenue_yoy/operating_cf_ps/gross_margin` 只保留在离线快照中，不进入 actor；获得可验证的版本化 PIT 源后才能升级 schema 启用。动态账户保留 13 维 `PolicyMemory`（有效位、上一 `DayConfig` 的 10 维 canonical action、实际成交总换手率、总成本率）。108 维 `market.causal.*` 坐标的 normalizer 仅在训练集拟合，scale 使用 `max(population_std, 2/clip)` 下限；静态状态由父进程每个 sealed split 预计算一次并通过共享内存复用，决策记忆沿单条账户链串行更新。新增字段、改变平稳变换或联合编码必须升级 schema 并从随机初始化重训。
- `PolicyMemory` 必须由上一期实际 Fill 结算一次性生成：配置先 decode/validate 再由同一 `ActionSchema.encode` 规范化，总换手率为实际成交名义金额之和除以上一期调仓前 NAV，总成本率沿用 Fill 中已包含费用和滑点的成本除以同一 NAV。完整 episode 首期使用原子空记录；实盘连续推理必须从 journal 恢复完整记录，缺任一字段时不得伪造默认历史。该记忆只进入标准 Observation，帮助策略学习留仓惯性和切换成本，不额外进入 Reward/loss，也不替代 Calmar critic context。
- 路径依赖 Reward 的 episode 期限、进度、累计净收益、历史最大回撤和回撤惩罚系数只可作为训练期 critic context，必须在任何可学习层之前与 actor 硬隔离；actor、回测确定性推理和实盘始终只消费标准 Observation。Bundle 必须记录 critic context schema 和部署补零规则，并测试任意改变 context 都不会改变 actor logits、动作或 `DayConfig`。

## 9. RL 输出标准

首版使用统一连续 `Box[-1, 1]^D` 动作空间，通过 `ActionSchema` 解码为强类型 `DayConfig`：

PPO 内部必须按语义使用有界 Beta 和 Categorical typed heads，再编码回统一 Box 接口。八个因子权重分别使用独立 Beta 分布线性映射到 `[-1,1]`，负值表示反向使用该因子；Beta concentration 必须限定在 manifest 声明的有限区间。`buy_n/sell_m` 分别学习类别 logit，再对 `sell_m >= buy_n` 的合法组合构造并精确归一化联合 Categorical。所有实际动作组件都必须计入同一个精确 joint law 的 sample、mode、log-prob 和 entropy，禁止多个 policy 类别折叠到同一 `DayConfig`。

| 参数类型 | 解码方式 | 示例 |
|---|---|---|
| 连续 | 独立有界 Beta 映射 | 八个 `[-1,1]` 因子权重；负值反向使用，0 停用 |
| 有序离散 | 合法条件化联合类别 | `buy_n`、`sell_m` |

动态配置仅包括：

```text
factor_weights（每项范围 `[0,1]`，权重为 0 即关闭因子）
buy_n / sell_m
```

- `ActionSchema` 是动态参数名称、类型、范围、固定控制和动作编解码的唯一来源。当前八因子生产 schema 为 `day-config-v10-multistyle8-unit-weights`、`Box[-1,1]^10`：八个因子权重各自独立位于 `[0,1]`，通过线性 codec 映射到统一 Box 接口，另有联合合法的 `buy_n/sell_m`。`factor_enabled` 完全由权重是否不等于 0 派生，不是独立动作；过滤器、涨停保护、调仓带宽固定为配置值，`single_buy_pct` 固定为 `1 / buy_n`。
- `DayConfig` 不得包含目标仓位、现金比例、是否调仓、调仓模式或其他择时/空仓控制；planner 固定每日 equalize 并执行现金 sweep。
- `single_buy_pct` 不属于 PPO 动作，固定为 `1 / buy_n`，避免形成额外的隐蔽仓位控制。
- `prefilter_n` 不属于 PPO 动作：回测仅用 T-1 排名平滑截断 T 日候选池，实盘复用同一候选集加速 09:25 当日 K 线拉取。
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
| `plan_open` | 因子评分、选股、合法性、每日 equalize 和可执行现金 sweep |
| `ExecutionPort` | 接收 `OrderPlan` 并返回标准 `Fill` |
| `settle_next_open` | 在下一决策开盘处理公司行动、账户估值和已扣成本净收益 |
| `EpisodeSession` | 驱动唯一候选账户链，并生成训练 Reward |
| `backtest` | 按交易日驱动同一领域核 |
| `gym_adapter` | 将领域核适配成 Gymnasium `reset/step` |

`env.SimExecutor` 和 `trade.BrokerExecutor` 都实现 `ExecutionPort`，由外层入口创建并注入；`env` 不 import、创建或判断 `trade` 实现。`plan_open` 的对象图只包含决策时刻可用数据。动作 T 的 Reward 默认按 T-open 到 T+1-open 结算；`close[T]` 不参与 T 日决策或该动作的估值，最早作为历史字段进入 T+1 Observation。Gym/SB3 类型停留在适配层，不渗入评分、选股、账户等领域函数。

“始终满仓”定义为可实现满仓：T-open 先卖后买后，在现金非负、费用和滑点已计入、交易合法、单股集中度及交易所数量规则满足的前提下最大化持仓市值。`rebalance_band_pct` 只能减少普通再平衡，最终现金 sweep 必须忽略 band。只有不存在可按冻结价买入的合法最小增量，或所选目标的集中度容量已经耗尽时才允许残余现金；每日必须记录 `post_fill_cash`、`post_fill_exposure`、`cheapest_next_legal_buy_cost`、`residual_cash_reason` 和 `full_investment_contract_satisfied`。停牌/跌停锁仓和公司行动整数化现金属于约束残余，下一决策日仍须再次 sweep。

当前离线 runtime 没有现金分红、送转、拆并股和配股明细，因此训练/回测采用版本化 `total_return_reinvested-v1` 合成账户：通过 `close[T] / preClose[T+1]` 保持下一开盘经济价值，并明确 `broker_exact=false`。该会计口径、费用、股票池、planner、Reward 和 ActionSchema 必须共同写入并校验 policy bundle 的 environment schema。合成股数不得直接 replay 为券商持仓；实盘迁移必须通过公司行动明细或券商日初股数/现金变动生成真实账户转换。

## 11. 标准数据流

| 场景 | 标准流程 |
|---|---|
| 数据更新 | 外部数据源 -> `offline_data.update` -> 清洗/PIT 对齐 -> 本地 store -> snapshot |
| GA 训练 | 本地快照 -> env 回测 -> GA 搜索静态 `DayConfig` -> `FixedPolicy` |
| PPO 训练 | 完整训练快照 -> 随机连续训练窗口 -> Gym adapter -> Observation -> PPO -> DayConfig -> 年化净收益增量/新增最大回撤 Reward |
| 模型回测 | 加载模型 -> 确定性推理 -> 同一 env -> `SimExecutor` |
| 实盘 | 加载冻结模型 -> 决策前更新并冻结快照 -> Observation -> `deterministic=True` 推理 -> DayConfig -> OrderPlan -> 券商执行 |
| 盘后复盘 | 读取已记录的 Observation、动作、订单和成交进行 replay，不重新推理 |

数据更新入口和实盘券商适配器负责联网；因子、env、GA/PPO 训练和回测只读取本地数据。实盘数据先形成带版本的决策快照，再进入策略链路。实盘进程只加载冻结模型并确定性推理，不在线训练或更新模型参数。

- **离线并行边界**：GA 个体评估、彼此独立的 PPO rollout env，以及彼此独立且仅加载冻结模型的完整离线回测允许使用多进程；PPO 只允许一个 learner。每条账户时间链、三段评估编排和实盘路径必须串行。并行任务不得联网、不得跨 split 读取或拼接数据；每个 worker 的底层计算线程固定为 1，禁止嵌套并行。

当前使用最小非对称 Transformer actor：将 2176 维 actor Observation 按 32 维切为 token，使用单层、`d_model=16`、2 个 attention heads、FFN 32、dropout 0；critic 保持 MLP `[64,32]`。Stable-Baselines3 PPO 算法、typed 动作头和训练流程不变。训练固定 `learning_rate=1e-4`、`gamma=0.995`、`gae_lambda=0.95`、`ent_coef=0`；增量回撤奖励不再依赖整段 Monte Carlo 回传，约20个交易日的 GAE 信用窗口降低市场噪声，同时保留中期风格持续性。不加入 observation/action noise、dropout、gSDE、随机股票/特征 mask、股票子采样或静态配置先验。随机初始化、typed 分布采样和 PPO minibatch shuffle 属于算法内生随机性。learner 默认使用 `device=auto`（CUDA 可用时使用 GPU）；每个 rollout env 默认采集 64 个 transition 后执行一次 PPO 更新，root run 默认执行 2000 个 vector rollout；自动 batch 以 256 为目标选取可整分 vector rollout 的大小（20env 时为 256，小 worker 数自动适配）；rollout worker 默认按 `min(20, CPU核数)` 解析，与 GA 一样通过 spawn 多进程附着同一只读 shared episode，每个 worker 的底层线程固定为 1。

stationary schema 的默认数据协议为：train `2000-01-01～2022-12-31`、validation `2023-01-01～2024-12-31`、test `2025-01-01～2026-08-28`。边界必须封存在 run identity 中；不得复用旧 Reward、旧动作 schema、旧 lineage 或 normalizer。用户明确指定跨环境迁移时，只允许把验证集选中 checkpoint 的 policy 权重作为新 root 的 warm start：新训练集重新拟合 normalizer、重置 optimizer/timestep/Reward state，并记录来源 checkpoint SHA，不能伪装成 verified resume。


每个 rollout env 在 reset 后从训练集内均匀随机选择合法起点，再从最小有效长度到该起点剩余长度中均匀随机选择连续周期；每次 reset 重新采样。窗口始终使用完整股票轴，禁止股票截断、训练折和跨 split 拼接。`n_steps` 是每次 PPO 更新前各环境采集的固定短 rollout 长度，默认 64；episode 在到达自身随机终点时正常 reset，同一 rollout 可经历多个随机 episode。多个 env 使用独立 RNG 和账户链。

每个 rollout worker 只维护一条候选账户链。逐日 Reward 为当前随机 episode 年化净收益的增量，再在历史最大回撤创新高时立即扣除 `1.0 × 当日新增最大回撤`；两类增量分别望远镜求和，因此完整 episode 奖励和严格等于 `年化净收益 - 最大回撤`。这是 Calmar 在比率 1 附近的稳定分式规划 surrogate，避免直接使用回撤接近 0 时会爆炸的 Calmar 比值，并消除随机窗口长度不同造成的奖励尺度偏差。收益已经通过 NAV 扣除费用和滑点，造成回撤的动作可立即获得反馈。当前 rollout 训练执行、完整训练期回放和验证回放统一使用单边 0.1% 滑点，实际费用和滑点仍通过 NAV 扣除。两种费率必须同时封存在 run identity，planner 预算与 simulator 成交必须注入同一个 `FeeSchedule`。不得加入相对静态配置、Sharpe、波动率、换手、动作平滑或重复费用惩罚。完整训练期 Calmar 只用于训练诊断评估，不进入 rollout Reward。满仓、有限值、schema、identity 和 holdout 隔离是硬契约。


PPO 使用 `stable_baselines3.PPO` 标准 clipped update，只允许一个 learner；不得维护自定义三折 surrogate、update gate recovery、Pareto 选模或回滚 learner。每次有效评估及训练结束必须原子保存 canonical latest checkpoint 及 identity/SHA/counter sidecar；verified resume 只能从该 checkpoint 继续。

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

## 16. 多风格与缺失数据契约（2026-09-05）

- 八因子固定词表为原四因子，加 `CompletedReversal20`、`CompletedMomentum252Skip21`、`CompletedAmountImbalance20`、`CompletedCloseLocation20`。新四因子仅消费已完成日行情；动量精确使用 `[T-252,T-21)`，其余使用 `[T-20,T)`。不含隔夜跳空因子，不启用无版本化 PIT 的财务数据。
- 新因子完整窗口任一输入缺失则输出 NaN；不补造数据。各已合法打开 split 输出逐因子、逐年 PIT 成员覆盖率和整段全缺失日期；覆盖率不是 PIT 真实性证明。
- Actor 新增每因子当日及20日均值覆盖率，在 `market.causal.factor_coverage.*` 中使用训练期同一 normalizer；保留完整股票轴，不新增随机 mask。生产 Observation v11、encoder v11、factor v5、typed v5、run identity v39 必须随机初始化重新训练。
- env/scoring 为 planner 和 T-1 prefilter 唯一评分实现。完整有效行保留原始加权和；局部缺失按可用绝对权重对中心化 rank 归一化；全无信号股票位于稳定尾部并可作为合法满仓的最终补位，缺失不能成为现金开关。
- 静态基准新增四因子权重显式为0，保持原四因子权重和固定控制；PPO 八因子均使用独立 Beta 动作并解码为 `[0,1]` 权重。确定性推理使用精确 Beta 众数；初始对称分布的确定性权重为0.5。

## 17. 随机初始化 PPO 收敛优化实验（2026-09-06）


当前奖励候选为 `H/252 * (当日年化净收益增量 - 当日新增最大回撤)`：固定 H 的完整 episode 奖励和为 `H/252 * (年化净收益 - 最大回撤)`，保留该窗口内目标排序，但改变随机窗口间的训练权重。不得声称该缩放使不同 H 的梯度完全一致或精确优化完整期 Calmar。相同 307,200 transitions、12,000 Adam steps 的短训中，`n_steps=64`、240 轮达到 Calmar 0.864755，优于 `n_steps=256`、60 轮的 0.627817，进入长训检验；这不是最终收益验收。critic 零输出与 Beta 初始浓度 10 两项候选已因短训表现较差淘汰，没有合入主仓。


上一阶段 4000 个 n_steps=64 rollout 长训已正常结束：5,120,000 transitions，153.093 分钟，实际 39,973 个 SB3 update epoch / 199,846 次 Adam step（target-KL 可提前停止）。训练最佳 Calmar 1.035877，仅略超静态 1.035350，未达用户 1.2 目标；训练结束 Calmar 0.854525。唯一合法打开的验证候选 Calmar 0.383157，低于静态 0.504163；测试封存，无部署 bundle。未启动备选续训；当时提出的无状态 actor 消融已在下述后续授权中执行。不得将短训改善或这次训练峰值视为收益/泛化验收通过。完整证据见 artifacts/rl/reward_optimization_20260906/report.md 与 long_short64/independent_final_review.md。

用户后续已明确授权相关 PPO 优化实验全部自主运行，无需再次征求实验或 actor 消融的同意。无状态 actor、学习率、探索等对照先在隔离源码工作区随机初始化并用短训筛选；保留现有因子、标准 PPO、领域执行语义与 holdout 隔离，不启用静态权重先验或模仿损失。新的实验身份及结果记录在 artifacts/rl/autonomous_optimization_20260906；未通过实测的候选不得冒充已验收生产策略。

后续隔离实验：v45 constant actor 两组均完成 307,200 transitions；lr=1e-4 最佳/最终 Calmar 为 0.394524/0.287895，lr=1e-3 为 0.051224/0.010215，暂不采用。两组初始参数逐位相同，但训练仍逐日随机采样，不能解释成整段固定配置搜索。追加 v46 MLP `[64,32]` 对照，复用现有 typed policy，独立源码与协议封存在 artifacts/rl/mlp_architecture_20260907；相关 34 项测试通过，候选尚待真实短训检验。


主入口默认值已按本轮有效候选提升为 n_steps64、2000rollout、每50轮评估、训练/评估滑点均0.001；batch-size=0 自动选择不大于目标256且可整分 rollout 的最大 batch（不存在该范围内除数的 rollout 使用完整大小），显式 batch 仍必须整分。20env 的默认有效 batch 为256；保留原 Transformer、lr1e-4、ent0 与默认seed20260827，不合入失败的constant、MLP或加熵候选。run identity 升至v47，Reward/Observation/typed/领域schema不变，新主入口要求新随机root。已运行的v45模型及其续训继续使用原冻结v45源码；实验seed20260906与主入口默认seed须明确区分，不能把单种子收益声称为其他seed已验收。


## 18. 冻结评估可选重叠执行（2026-09-07）


正常排空的 overlap 运行可按同身份 verified resume；异常或中断且未排空的运行拒绝续训，保留未消费快照供审计，本实现不恢复未完成的异步评估。不得用旧 v45/v47 identity 冒充当前身份；既有模型须使用各自冻结源码和合法生命周期。

单次配对短训各 307,200 transitions、实际 12,000 Adam，blocking 19.92965 分钟、overlap 17.48609 分钟，总墙钟减少 12.260926%。独立审计确认四个完整训练评估点及动态配置相同，初始化/最终/训练最佳 policy 与 optimizer 逐位相同；验证、测试均封存。该结果是当前机器上的调度提速证据，不是样本效率、泛化或其他种子收益证明。源码与审计记录见 artifacts/rl/overlap_evaluation_20260907/report.md。

## 19. v45长续训与诊断测试结果（2026-09-07）

用户授权追加10000个rollout并随后明确要求分别打开测试。续训从canonical latest开始，新增12,800,000 transitions，最终累计17,920,000；最终训练Calmar 0.7864538066，本段最高1.0186458018，未刷新历史训练最佳1.2854324423或验证最佳0.5400281110。历史最佳模型未被覆盖。


## 20. 非负因子权重实验（2026-09-07）

用户授权将八个因子权重统一改为 `[0,1]` 并启动100000个vector rollout。ActionSchema升至`day-config-v10-multistyle8-unit-weights`，typed distribution升至v7，run identity升至v49；统一PPO接口仍为`Box[-1,1]^10`，其中因子坐标通过线性codec解码到`[0,1]`。该语义变化必须随机初始化新root，不得resume、warm start、动作迁移、使用静态配置先验或模仿损失。2025～2026测试期已经由上一实验查看，不得将本实验对该时段的结果称为新的盲测。

## 21. 十因子财务接入及时间协议（2026-09-12）

本节按用户最新授权覆盖前述生产因子数量、动作维度和默认 PPO 分期；历史实验继续使用各自冻结源码。当前七因子为 TrueMarketCap、VolumeCV、AmountBasedSmallCap、CompletedReversal20、CompletedMomentum252Skip21、BiliAdjustedIssueDiscount、BiliHighLifetimeRangeRatio，新增 PBBelowTwoROEAbove10Signal、LowCashOutflowProfitGrowthSpread、HighOperatingProfitRevenueGrowthSpread，总计十因子。

- PB<2 且 ROE>10% 是有效的二值评分：通过为1，不通过为0，财务缺失为NaN。它参与加权评分，不是硬过滤。另两个因子分别为净利润同比增速减现金流出同比增速、营业利润同比增速减营收同比增速；增速使用单季度同比、分母为去年同期绝对值，零分母无效。连续评分同值取平均排名，二值评分保留0/1。
- 十个独立 Beta 权重解码为[0,1]，加合法联合 buy_n/sell_m，共 Box[-1,1]^12；沿用 typed v7 分布，不新增开关动作。完整有效行加权求和，局部缺失继续复用 env/scoring 的可用权重校正，缺失不作有效0、不补未来值、不取完整股票交集。
- 财务原始字段只由 offline_data 对封存的公告版本按公告日严格早于T生成六个 float64 面板。股票轴、原行情及旧字段保持不变；新原始财务字段不直接进入actor，只通过因子及覆盖率进入因果Observation。数据哈希和有限原公告样本不能表述成全库原公告认证。
- 按逐年PIT成员覆盖率决定总区间2014-01-01～2026-08-28；默认训练2014～2021、验证2022～2024、测试2025～2026-08-28。2012年前严格长动量近乎不可用；2014起保留足够连续历史并允许局部缺失。只对覆盖率做全日期审计，不用新holdout收益选择分界。
- 生产 factor v7、Observation/encoder v13、ActionSchema v12、environment v27、run identity v52、bundle v31；新root必须随机初始化，训练期重新拟合normalizer。旧七因子checkpoint不能迁移为十因子续训。源文件、数据身份与分期在首checkpoint前封存。
- 用户授权替换原后台长训，本次十因子root沿用100000 rollout预算、20env、n_steps64、每50轮评估、batch自动256、原Transformer及0.001训练/评估滑点。该预算不改变CLI默认2000轮，也不代表收益已验收。

## 22. 三段诊断评估实验（2026-09-13）

用户最新明确授权当前十因子训练在每次完整训练评估时同时回放验证和测试，三条 Calmar 曲线同图展示；新阶段所有评估 checkpoint 仅按验证 Calmar 选择，不再用训练创新高预筛选。测试反复查看只作诊断，不进入奖励、loss、选模或收益资格，不能再宣称最终盲测。历史诊断不倒灌重选。


旧epoch5任务于第17394轮后的更新出现NaN并退出。最后完整checkpoint参数/optimizer经核验有限，从此恢复。已复现 float32 Beta样本到Box再逆解码舍入为端点、产生非有限log_prob的漏洞；typed v8只将写入Box坐标限制在 `[-1+eps/2, 1-eps]`（eps为动作dtype机器精度），保持可安全逆解码，浓度、entropy、RNG及内部样本不变。未保存原失败batch，不能把漏洞复现声称为捕获到了原故障样本。此数值语义变化记录在新身份/源码中，不修改旧冻结源码。

## 23. 长历史输入与统一回放提速（2026-09-14）

按本线程用户授权，主入口 Observation 历史默认由64增至504个交易日。行情复用原有平稳变换及固定维时序统计，统计窗口延长至504；不得表述为actor逐日接收未经压缩的504根原始K线。另增加同长度、逐日不聚合的真实历史配置输入：十因子权重及buy_n/sell_m的canonical action、实际总换手率、实际总成本率和有效位，共`[504,15]`。日期对齐T-504至T-1；只从本账户实际Fill链生成，未知日期为原子空记录，不伪造过去权重。新十因子actor为10217维（静态市场2519、动态账户7698），其中历史配置7560维；移除旧portfolio重复的一步记忆坐标。历史配置normalizer不对padding做去均值，费用scale固定0.01；市场normalizer仍只拟合训练期。

生产Observation/encoder v14、normalizer v6、environment v28、run identity v56、bundle v32、journal v4；动作schema v12、十因子及Reward不变。主入口同步已验收typed v8端点修复。504输入必须新随机PPO root，禁止将旧64日checkpoint冒充verified resume。journal存储完整日期历史，恢复/replay顺序重算实际Fill链并核对，不接受缺少历史记录的已初始化账户。`D:/coding/WBR-live`继续使用其自身冻结版本，不属于本次修改范围。

GA和PPO仍共用唯一EpisodeSession、planner、scoring、fees、quantity、simulator和Reward。通过只读日数据/合法性缓存、单次评分复用、稳定前缀排名、Numba串行数值内核、精确费用预算、不可变绩效缓存及可选不保留完整trace提速；禁止fastmath、嵌套线程或省略实际成交/满仓/费用/公司行动。GA采用无actor缓存的共享episode和RolloutSummary，同一账户链与完整trace数值一致；PPO继续保留标准输入与实际历史。Numba 0.63.1已锁定。GA主入口同步已有冻结实验的十个独立连续权重及合法买卖数量，清理旧独立single_buy_pct/band采样、0.1步长权重网格和重复YAML参数，搜索范围由ActionSchema唯一提供。

实测固定5544股、2002～2021共4852次转换：纯回测汇总静态2.629～2.637秒、逐日变权重2.973～3.025秒；保留完整trace分别约2.763秒和3.173～3.219秒。加载/因子准备约38秒、首次JIT另计，不能声称所有调用都严格≤3秒。与原始同口径完整trace的逐日订单、Fill、费用、配置、净值、Reward、现金、收益及满仓标记完全一致。该20年区间只用于性能等价验收，不改变正式2014～2021训练边界，也不证明早期因子覆盖或收益合格。

独立冻结个体实测20worker相对4worker吞吐约2倍；每条账户链仍串行、底层线程1。GA显式缓存迁移仅允许worker保持原值或本次已实测4→20；其他预算、评估间隔、选择/测试角色及未知协议字段保持冻结。保留完整代和已打开的诊断状态，育种RNG按声明seed重建，不伪称逐位续训。holdout至多保留一份带预加载的episode，已评估标量结果按split/config缓存，验证选模先于诊断测试。

完整证据见`artifacts/long_history_rollout_20260914/report.md`及独立review报告。504输入完成的是因果、数值和资源验收；没有运行新的PPO learner，不得宣称长历史已经改善泛化或换手。用户当前顺序为先GA、PPO保持停止；GA只能在约2～3秒纯回放验收及独立检查完成后恢复。

## 24. 高异常毛利润接入十一因子（2026-09-14）

按用户最新授权，在上述十因子末尾追加 `HighAbnormalGrossProfit`，供GA和PPO共用。唯一公式为 `(GP_q - GP_q_minus_4 * sales_cash_q / sales_cash_q_minus_4) / assets_q`；GP为单季营业收入减营业成本，sales_cash为单季销售商品、提供劳务收到现金，assets为相同季度末总资产。三张报表选截至T均可知的最新共同季度，各历史值使用当时已公告版本，公告日期严格早于T；流量由YTD还原单季，不作额外一天滞后。缺失、非正前期收现/资产、负当期收现输出NaN，不补造有效0。

生产财务数据在原6个float64面板后追加7个异常毛利润原始季度面板；字段清单由offline_data权威定义，研究和生产共用同一公告回放及因子公式。原始财务金额只供factor使用，不直接进入actor；因子rank、覆盖率和因果历史统计按既有Observation机制自动纳入。原10因子及原6财务面板数值保持不变。

十一项Beta权重独立解码到[0,1]，加合法buy_n/sell_m，动作13维；真实配置历史为[504,16]。actor10983维（静态2766、动态8217）。静态configs/config.json新因子显式权重0仅保持基准，GA和PPO均可学习非零权重。生产factor v8、ActionSchema v13、Observation/encoder v15、normalizer v7、environment v29、PPO run identity v57、bundle v33；typed v8、Reward、网络、费用及训练分期不变。数据events v3、financial snapshot/builder/replay v2、runtime/generation v7。

新封存快照为 `data/runtime/runtime_1990-12-19_2026-08-28_financial11.npz` 及同名manifest，旧六面板快照不得冒充新schema。旧十因子PPO checkpoint/normalizer及GA缓存不能作为十一因子续训；新PPO需要随机初始化，正常训练/验证/测试规则不变。本轮为接入与训练期机制验收，包括有限的GA小型搜索、PPO动作/梯度检查和合成数据训练契约测试，不代表已完成十一因子PPO收益训练或部署资格。既有冻结GA/PPO实验和WBR-live不作隐式迁移。证据及独立验收见 `artifacts/factor11_integration_20260914/report.md`。

## 25. 三段预计算常驻内存（2026-09-15）

用户明确要求将已合法打开的训练、验证、测试数据放在内存中复用，不再每次评估重新加载、计算因子和静态Observation编码。本次授权覆盖此前至多一份holdout驻留的内存限制；各split在首次评估前准备，共享数组只读，holdout评估与账户时间链仍串行。复用既有SharedPreparedEpisodeOwner/descriptor和ExitStack，运行退出统一释放，不新增第二套缓存、回测或评分实现。训练已转成共享内存后应释放原数组重复引用；持仓、现金、实际配置历史与Reward状态仍沿各自账户链逐日更新，不能缓存为固定输入。

本次旧v58十一因子诊断任务在完整三段第480轮后停下。仅允许经显式审计JSON绑定的v58 blocking到v59 resident-cache性能迁移：父canonical latest/identity/三段曲线/冻结源码及原总预算必须匹配，唯一允许的源码变化为本次已审查ai/rl/train.py缓存编排和迁移装配；领域、动作、Observation、Reward、normalizer、费用、网络及其他训练合同保持。保留policy、optimizer、累计计数、原验证选中和训练最佳ZIP及其历史SHA，不把迁移前评估改写成新身份。原总预算2000轮，已完成480轮，继续1520轮，评估沿累计轮数每80轮及结束执行。


实测v59三段共享数组为34.74GiB，首次准备引发明显内存/提交压力，因此在第560轮完整三段评估后停止，继续完成v60投影优化。GA/PPO共享入口统一调用env的 `PreparedEpisode.compact_for_replay()`：完整历史计算因子和静态市场编码后，再只保留本段及标准Observation/PolicyHistory所需前置行；504日输入保留前504行，无actor的静态回放保留T-1行。所有字段和股票轴保留，不重算listing_age或因子，不跨split拼接。offline_data的可选 `ReplayProjection` 保存原manifest、原行数、偏移和预计算schema证明；无投影时原manifest序列化保持，投影runtime禁止重新计算因子或因果市场状态。

v60允许已审计v58/v59诊断父节点的有限性能迁移；继承旧身份的曲线和选择文件须沿父链逐项核对，不改写历史。明确源码范围为ai/rl/train.py、env/backtest.py、env/shared_episode.py、env/observation.py、offline_data/contracts.py、factor/compute.py；每项变化绑定前后SHA，其余源码原样。contract只新增投影存储记录并允许shared_memory_bytes下降，因子/Observation/动作/Reward/数值运行时和数据lineage保持。v59停点560，最终继续原2000总预算的剩余1440轮。

## 26. 十一因子累计十万轮续训（2026-09-15）

用户明确要求已完成2000轮后继续到累计100000轮，并提供原2000轮可视化；本段新增98000轮。父v60已完成2560000 transitions、10000个SB3 update epoch，训练/验证/诊断测试各26点。续训保留当前learner的policy、optimizer、normalizer、累计计数、原始及训练最佳模型、验证选中模型和三段历史，不从验证模型回滚learner，不重新初始化参数。

复用既有显式审计恢复路径，新增 `wbr-ppo-diagnostic-budget-extension-v1`，仅接受正常完成的v60诊断父节点，生成v61子身份。核对原2000预算、training.json及三段曲线SHA、完整v60→v59→v58祖先历史和独占claim；源码仅允许经审查的ai/rl/train.py前后SHA变化，其他运行合同完全相同。最终canonical ZIP若因正常结束重新打包而SHA不同，必须与最后评估模型的全部ZIP成员名及未压缩内容逐项相同；不能只凭同计数或仅比较policy而跳过optimizer。canonical latest仍是恢复源，不重写旧评估SHA。


## 27. 延长验证与测试的共同分期（2026-09-15）

用户明确要求GA/PPO验证、测试各增加两年，PPO默认100000个vector rollout、GA默认10000代，并停止后台PPO后先跑GA。保留此前已验收的总范围2014-01-01～2026-08-28，默认重分为训练2014-01-01～2017-12-31、验证2018-01-01～2022-12-31、测试2023-01-01～2026-08-28；三段不重叠。验证从3年变5年，测试由约1.66年变约3.66年，代价训练由8年缩4年。延长评估不保证改善收益或泛化，旧已观察年份不得称为新的盲测；本次不为扩大训练而未经审计前推至2014以前。

共同默认分期唯一来源为configs/evaluation_splits.json，configs/training.py提供唯一解析校验器；PPO日期CLI仍可显式覆盖，GA显式evaluation-splits和ppo-reference仍可覆盖。默认GA模式启用共同三段诊断，每10代及结束评估训练冠军，仅验证Calmar选模，测试仅诊断；debug模式仅在显式要求时评估holdout。PPO默认预算改为100000，GA预算继续由strategy.yaml唯一声明10000。其余领域、因子、动作、Observation、Reward、费用、网络和评估节奏不变。

改变分期必须新随机root并封存新身份，不得从使用过2018～2021旧训练数据的PPO或GA承接模型、种群、缓存、normalizer或验证选择。旧运行文件保留，不重新贴新分期标签。PPO此次停止并核验保留3849轮checkpoint；只启动新的十一因子GA，PPO保持停止。本次证据目录为artifacts/ga/long_holdout_20260915。

## 28. 长动量允许内部缺失并前推训练起点（2026-09-15）

用户明确授权放宽长动量缺失规则，并将GA/PPO共同训练起点前推到最早可用历史。本节覆盖第16节该因子的完整窗口要求及第27节的2014训练起点；其他因子缺失语义不变。`CompletedMomentum252Skip21`仍严格使用[T-252,T-21)的231个已完成交易日行，首尾official close/preClose日收益必须有效，且至少185行有效（固定80%向上取整）。只跳过内部无效日收益，不回填原始价格、不延长窗口、不按有效数量外推收益；全部缺失和上市历史不足仍NaN。跳过相当于缺失日不累计log-return的中性约定，不声称还原了缺失日真实收益。

factor schema升至v9，长动量metadata为completed-official-return-v2-interior-skip-min185-endpoints。ActionSchema、Observation张量结构、Reward和领域执行不变；factor hash进入Observation/encoder/normalizer及run identity，旧模型、缓存、GA种群和normalizer不能作为新语义续训。反转因子实现位于同一文件，模块源码hash会随之变化，但其数值和严格缺失规则不变。

前缀实测长动量2010覆盖由0.82%升至78.42%，2012由1.42%升至82.22%。2003-04-21虽首次同股11项都有值，当时两个财务因子各只有1/1217股票有效；不能称为广覆盖起点。采用不依赖收益的可用性标准：11个因子各自均覆盖至少半数当时成员的首次日期2004-04-28，作为最早实用训练起点。该50%仅用于选择起点，不作逐日股票池过滤，也不承诺以后每天均超过50%；原有局部缺失评分继续生效。

默认分期唯一来源仍为configs/evaluation_splits.json：train 2004-04-28～2017-12-31，validation 2018-01-01～2022-12-31，test 2023-01-01～2026-08-28。PPO默认100000rollout、GA默认10000代，评估和选模规则保持。旧GA在完成656代后停止并保留，切换新随机GA root；PPO保持停止，之后新建任务自动使用新起点和因子。更长历史和覆盖提升不代表收益/泛化验收。核算及独立审查证据在artifacts/momentum_relaxed_20260915。

## 29. 固定50只与按最差持仓换股（2026-09-15）

用户明确授权GA/PPO把sell_m替换为turnover_rate，并固定buy_n=50。本节覆盖旧买卖数量动作定义：生产ActionSchema仅暴露11个独立[0,1]因子权重与1个换股类别，Box12维；single_buy_pct固定1/50。X=max(1,floor(50*turnover_rate))，5%～20%对应2～10只；唯一canonical codebook为(0.05,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20)，每种实际只数只学习一个类别。PPO使用原11个Beta及一个精确Categorical，所有组件同一joint law；GA采样、交叉、变异读取同一ActionSchema。

每个决策日先用唯一因子评分与完整PIT成员排序，取当前持仓最差X只作为固定检查集合。其中仍在全市场前50名者保留；只有不在前50名且可全部卖出的持仓退出。锁仓、T+1、部分可卖或涨停保护不向更好持仓递补检查；其余持仓保留。新买仍用既有T-1 prefilter、soft-filter偏好与买入合法性，填充目标50只的空位；冷启动不受X限制。保留排名不得由买入合法性或prefilter重新编号。组合target与buy-legal target分开，避免向涨停股补买。原每日等权和cash sweep保持，因此X是替换股票数量上限，不是当日名义资金换手率上限；合法股票不足或锁仓导致的现金残余继续如实记录。

复用唯一planner、scoring、PIT排序、合法性、费用、simulator与账户链；当日完整排名直接供次日候选池使用，避免重复评分/排序。研究小股票宇宙只能显式指定research ActionSchema固定持股数与无别名档位，不作为生产buy_n动作或旧sell_m兼容入口。历史报告可只读展示旧字段，训练/推理拒绝旧执行配置。

ActionSchema v14、typed v9、Observation/encoder v16、normalizer v8、environment v30、bundle v34、journal v5、正式/诊断PPO identity v62/v63；GA standalone/reference identity v6/v5。历史含有效位为[504,15]，actor10479维（静态2766、动态7713）。factor v9、Reward、费用、因果输入与共同日期不变，旧模型/种群/cache/normalizer不可续入，必须新随机root。停止旧早历史GA并保留240代，先启动新GA10000代，PPO保持停止。本次代码与机制验收不代表收益或泛化改善；证据在artifacts/turnover_action_20260915。


## 30. 换股范围0～1与GA审计续训（2026-09-15）

用户明确授权把换股搜索范围扩展至0～1，并接着训练当前GA。本节覆盖第29节九档边界与该次必须新随机root的GA限定：buy_n仍固定50，X=floor(50*turnover_rate)，0检查0只、1检查50只。整股边界用12位小数舍入消除二进制浮点误差；planner必须显式处理X=0，禁止[-0:]误选全部。0不关闭每日等权、现金sweep或冷启动建仓，只禁止主动排名换出。

生产51个唯一类别对应X=0～50；为保持旧GA配置字节、hash和执行，X=2仍用旧0.05，其余用X/50。GA和PPO读取同一ActionSchema，PPO仍11 Beta加1 Categorical，但后者扩大为51类；Box仍12维，actor10479维与504×15历史形状不变，历史canonical坐标含义变化必须升级身份并重训PPO。

版本为action v15、typed v10、environment v31、Observation/encoder v17、normalizer v9、bundle v35、journal v6、PPO正式/诊断identity v64/v65，GA standalone/reference v7/v6。因子、日期、费用、Reward及旧9档的完整领域轨迹不变。

旧GA停止于完整159代。显式ga-turnover-range-expansion-v1审计绑定parent identity、6个父状态文件、13个允许改动源码的before/after SHA和新action hash；仅允许v6固定50九档GA到v7全范围GA。完整核对其余合同、source集合、旧配置可解码与检查数量不变，复用原continue-from编排、训练缓存、159代历史及已合法验证选中结果，累计160继续且评估保持每10代，总预算10000。旧数据不重贴新配置标签，继承行标记historical_protocol；仅训练缓存与静态配置迁移，育种RNG按既有机制重建，不声称精确随机轨迹延续，不迁移PPO模型/normalizer/历史坐标。测试仍仅诊断，不参与选模；PPO保持停止。证据在artifacts/turnover_full_range_20260915。


## 31. 完整训练期 PPO 对照（2026-09-16）

用户明确要求停止 GA，PPO 不再随机 episode 周期。本次新随机 root 显式使用 `--episode-scope full`：每个环境从训练起点运行到训练终点再 reset；n_steps=64 仍是更新前采集长度，不是 episode 长度。复用 WBRGymEnv 已有完整 episode 路径（random_window_min_transitions=None），不增加领域实现。20 个环境账户独立、时间进度同步，动作仍按 PPO 分布采样；不声称同步采样更有泛化性。Reward H/252 缩放不变，此时 H 固定为完整训练期 transition 数。正式/诊断身份 v66/v67、rollout v5；动作、Observation、Reward 和领域 schema 不变，旧随机窗口模型不伪装同协议续训。默认入口保留 random 供明确对照，本次启动参数必须写 full 并封存。沿用100000轮、20env、64step、n_epochs5、每80轮三段诊断和验证选模；测试只诊断。GA 已保留至1051完整代并停止。


## 32. 连续动作、全局权重与唯一报告协议（2026-09-17）

本节按用户最新授权覆盖旧51类别换手、旧动作头及旧运行时兼容规定。生产唯一ActionSchema v16为11个[0,1]因子权重和一个[0,1]连续turnover_rate，统一Box12维；固定buy_n=50，领域replacement_count、PIT、每日等权、cash sweep、费用与Reward不变。不得保留类别表、离散换手解码、Categorical、温度参数或旧schema静默转换。GA采样连续比例，PPO使用一个12维独立Beta向量。

PPO actor只保留现有Transformer。动作头共享11个可训练基础权重β=sigmoid(b)，每日θ=tanh(状态头输出)，确定性权重位置w=β+θ×β（θ<0）或β+θ×(1-β)（θ≥0），因此θ=-1/0/1数学上对应0/β/1；同一checkpoint所有日期读取同一β。状态头为无bias的12维线性输出，前11维形成θ、末维sigmoid形成换手率位置，初始状态头为零；β随机初始化不使用GA或静态先验。12个全局可训练探索参数κ有界于[2,198]，初始30，Beta α=1+κw、β形状参数=1+κ(1-w)，初始总浓度32。实际随机采样保持固定[0,1]支持，基础参数更新不会让已记录动作掉出支持域；确定性推理取同一个分布众数。随机采样保留float32 Box端点安全处理；确定性Beta众数在形状参数为1时允许准确端点0/1，支持因子关闭与最多50只换出。基础+调整的确定性结构不等同于无采样波动。

typed v11、Observation/encoder v18、normalizer v10、environment v32、bundle v36、journal v7、PPO正式/诊断identity v70/v71。历史股票轴/输入维数不变，但历史换手坐标语义变化必须新随机root、重新拟合训练normalizer。生产只接受当前契约，旧checkpoint保留冻结源码，不添加跨版本fallback。静态完整期基准通过标准DayConfig序列化直接回放，与GA复用run_day_config_episode，避免float32 Box往返将0.1×50误变4。

用户确认下一次20env各自采完全部训练期才PPO更新；用--episode-scope full --n-steps 0解析为实际3326步，共66520 transitions/rollout。全采集允许batch256加最后216条，不自动退为40；每批更新5epochs。按原约1.28亿transitions预算折算1924轮，每2轮完整评估训练/验证/诊断测试；测试仅诊断、只按验证选模。不同轮预算必须报告transitions与Adam上界，不能把旧10万短rollout直接套成10万个全周期。随机窗口与固定短采集作为明确配置仍复用同一训练实现，非旧版本回退。

GA/PPO仅产出training-report-v2与training-diagnostics-v2，网站只消费当前契约。既有12个run的报告经artifacts/action_redesign_20260916一次性迁移并备份，缺失诊断明确标注未记录，不伪造数据、不修改checkpoint/identity/冻结源码。训练保存采样/确定性动作、真实换手/成本、探索宽度、基础权重/状态调整、PPO梯度与KL/value指标、GA种群多样性/缓存/耗时；详细样本固定抽取128条Observation并归档实际动作/Reward/returns等，记录身份与SHA；诊断不得改变优化器、参数或RNG。每次评估保留独立checkpoint归档，canonical latest仍为唯一正式续训入口。

历史缓存迁移、扩范围迁移及无调用旧入口移出生产；安全校验、异步成交对账、停牌估值和原子失败清理是实际领域与资源契约，不以消除fallback之名删去。多模块审查、测试和独立最终review证据记入artifacts/action_redesign_20260916。

## 33. GA/PPO 连续动作精度统一与暂停审计（2026-09-17）

用户要求停止 PPO，分析新版相对历史 PPO/GA 的训练退步，并保证两者使用相同动作空间。随机窗口、20env、n_steps64 的新版已停止于9918轮、12695040 transitions；canonical latest SHA 为0379cb20ab0a43f4bf5f5f9bcf293bdb28f67f47e2306c21a379409301ceeecb。历史模型、冻结源码、成绩及报告不迁移成新语义，未授权重新启动本次训练。

生产仍只有11个独立[0,1]因子权重及一个[0,1]连续换手率，固定buy_n=50、single_buy_pct=1/50。统一有限精度规则为unit float32经float32仿射映射形成canonical Box；GA生成、交叉、变异、导入配置及PPO动作解码均复用ActionSchema的唯一规范化。因子不额外归一化，不保留旧类别动作或epsilon容错。换出数量在env中按同一精度编码的k/buy_n物理边界确定，因此名义0.1/0.12分别对应5/6只；普通非边界比例仍保留连续值。相邻非canonical浮点数可能量化到同一配置，不声称任意实数/Box ULP无损。PPO确定性动作直接使用动作头给出的Beta众数参数，避免由alpha/beta重复除法造成额外舍入；随机采样、log-prob和entropy仍属于同一个Beta law。

ActionSchema v17、typed v12、environment v33、Observation/encoder v19、normalizer v11、bundle v37、journal v8、PPO正式/诊断identity v72/v73、GA三段identity v9绑定新的精度契约。张量维度、因子、Reward、费用及选模规则不变；动作历史数值语义变化要求新root与新normalizer，不接受旧checkpoint/种群的静默兼容续训。静态配置解析在完成声明式权重归一化后也进入同一canonical精度；FixedConfigProvider回放解析得到的DayConfig，不新增第二套转换或数值例外。

本次仅修复动作一致性，不把精度修复宣称为收敛改善。新版训练峰值1.141256（3840轮），最后完整评估0.696817（9840轮）；旧完整episode PPO峰值1.379643、旧随机窗口PPO1.503171、历史GA1.783426均来自训练集。全局/状态分支饱和、探索宽度及采样/确定性表现的证据、限制和独立review见artifacts/ppo_action_parity_20260917/report.md；不利用测试集反向改奖励或挑选本次诊断模型。

## 34. 直接单位动作头与状态输入审计（2026-09-17）

用户要求权重维持0～1并精简全局/残差参数化，同时检查输入重复及应提供但缺失的状态。本节覆盖第32节的全局β/θ动作头：生产唯一头为Linear(actor_latent,12,bias=True)后sigmoid，直接产生11个独立因子权重与连续turnover_rate的Beta众数m，不作softmax或总和归一化。删除独立global_base、tanh调整及分段映射，不保留旧类/兼容入口。保留同一个Beta law：alpha=1+κm、beta=1+κ(1-m)，12个共享可学习κ范围[2,198]、初始30；确定性使用m，训练仍随机采样。普通线性bias是网络参数，不再额外拆出全局β。小正交初始化gain0.01、bias0，不使用静态或GA先验。

typed v13、正式/诊断PPO identity v74/v75、bundle v38；ActionSchema v17、Observation/encoder v19、normalizer v11、env v33及因子/Reward/费用保持。头结构变更必须新随机root，不把冻结旧模型当新版续训；目前PPO/GA保持停止。诊断改存直接mode logits并保留通用动作/分布/梯度/实际成本字段；网站通过共有字段显示历史报告，不引入多套模型或报告分支。动作头参数215→216，简化的是参数化，不能声称参数量缩减或已解决饱和/收敛。

状态审计只读训练样本，未修改生产输入：actor10479=市场2766+账户153+历史7560，另有8维critic context硬隔离。源码确认至少309个重复坐标可合并；504天动作历史占72.14%，其必要长度尚待消融，样本常数不能直接视为可删字段。优先待验证的状态包括持仓top50边界/尾部、合法候选与持仓分数差、持仓价格规则锁仓和实际prefilter候选可买比例。现有sellable_ratio不等于完整卖出合法性；历史动作、实际成交换手/成本已有。若后续实现参考排名，必须用上一配置或单因子、复用env唯一评分/合法性，不用尚未输出的今日动作构造今日输入。输入变化另升schema、重拟训练normalizer并随机初始化。代码与输入独立审查及137项集成测试证据见artifacts/ppo_direct_actions_20260917。

## 35. 原始逐股输入与开盘交易状态（2026-09-17）

用户已明确授权替换旧手工汇总输入，并增加当时已知的交易合法性。本节覆盖旧stationary/moments/market.causal输入约定。唯一Observation v20保留完整股票轴和504日窗口，27个逐股字段为T-open的open/preClose、上一完整日OHLCVA/preClose/total_share、已知issue_price/ST、开盘price_buy_allowed/price_sell_allowed、两个soft-filter通过状态及11因子排名/信号。完整日字段严格lag1，上市前非成员数据不得进入上市首日的lag。沿用版本化因子，不添加财务原始金额。价格/量额不再预先转rank、截面分布或手工时序统计；因子仍保留领域实际使用的rank/二值信号。

账户保留完整逐股quantity/average_cost/sellable_quantity/last_mark_price、cash/nav/peak_nav/max_drawdown及504×15真实配置/成交换手/成本历史。可买/可卖标志复用DayMarketData.trade_legality及固定保护开关，与planner同一规则；它们表示市场价格规则允许，并非资金、数量与T+1均满足的订单保证。可卖数量独立表达账户约束。父进程预计算标志，spawn worker首次规划仍由同一函数建立完整合法性结果缓存；不得声称所有进程只检查一次。

删除旧market_panel、统计/关系汇总encoder和StaticMarketEncodingCache，包括此前确认的309个重复统计坐标。原始相邻决策行中的open与open_lag1存在自然重叠，不宣称原始数值彻底去重。成员局部缺失统一-1，与真实0区分；非成员和时间padding为0，通过唯一PIT与时间sidecar排除。Normalizer v12仅在训练决策日期的PIT成员上拟合字段RMS正尺度，不居中、不裁剪；缺失标志保持-1，因子/二值尺度为1，账户尺度由initial_cash与训练价格定义。原始绝对水平不再具备旧平稳变换承诺，泛化效果必须后续实测。

encoder v20保留既定float32数值进行传输，不额外统计压缩：RawMarketStore保存一次只读逐日原始表，PPO rollout保存不可学习row reference及原始动态账户。N=5544时public compact维数29741，另加8维critic context；这不是原始输入维数。reference在所有可学习层前移除。父进程/shared/live均使用显式绑定，模型文件不保存数据表；不保留旧模型fallback。唯一raw-panel-network-v1使用逐股21日可学习patch、d_model16/2head/FF32单层Transformer、逐股账户融合和4个学习查询聚合，历史动作单独按同样机制学习；actor不接触critic context。只允许缓存不可变原始数据到设备，同次forward可复用相同日期；学习embedding不得跨optimizer更新缓存。checkpoint重算捕获forward使用的store和scale，避免rebind改变反向语义。

ActionSchema v17、typed v13、Reward、费用、因子、调仓和GA/PPO的12维连续动作语义不变；environment v34、bundle v39、journal v9、正式/诊断PPO identity v76/v77绑定新输入与网络。必须新随机root、重拟normalizer；旧训练9918轮checkpoint及冻结源码保持不变。当前只完成工程验收，PPO和GA均未重启，未打开真实validation/test；完整股票轴标准小更新与20日回放不构成收益或收敛验收。详细输入、性能、测试和独立review见artifacts/raw_observation_20260917/report.md。

## 36. CUDA 与原始窗口计算复用（2026-09-17）

本节覆盖旧版 CPU 冻结评估和 device=auto 说明。PPO learner、采样、冻结回放、模型加载及实盘模型推理统一只允许 CUDA；显式 CPU/auto、CUDA 不可用或 CPU tensor 输入必须报错，不保留 CPU/fallback。账户领域核、GA、离线因子与只读数据路由仍在 CPU 执行，不属于 PPO 模型计算。

raw-panel-network-v2 保留全部原始字段、5544 股票轴、504 日窗口、可学习参数及同一 Transformer。不可变字段缩放与 PIT zero-fill 在绑定时一次完成；同一次可微 forward 内按日期/股票复用重叠 patch 投影，统一 gather 后 split 防止反向重复创建完整 patch 梯度表。冻结整期推理只按内存块流式组装同一组 token，数学编码实现唯一。保留有界分块和反向重算，不通过缩短窗口、减少股票、冻结 encoder 或减少 PPO 更新次数伪装提速。

参数不变的 rollout/评估期间可缓存整段 learned market embedding；每次优化更新后的首次使用必须重建，绑定变化或 train(True) 必须失效。缓存键包含 market 参数身份/地址/版本及 store 绑定版本，不能用于可微更新，不写入 checkpoint。确定性推理用 CUDA Graph 调用同一 actor；形状、dtype、device、参数及绑定变化必须重建，critic context 在捕获前剔除。图和设备缓存均不序列化，失败不回退旧模型或 CPU。

PPO identity v78/v79、bundle v40 绑定 CUDA 执行与网络 v2；Observation/encoder v20、normalizer v12、environment v34、ActionSchema v17、typed v13、Reward 与账户语义不变。GA 父代索引只增量处理新评估候选；共享因子中间量按实际字段 lag/raw view 分键，并沿用 float64 公式，保持研究定义与生产定义相同的输入契约。生产代码不保留旧实现，旧副本只存在性能证据目录。

性能必须分别报告首次加载/因子、参数变更后的缓存重建、完整账户链、采样、PPO 反向与 Adam、诊断及保存；不把热缓存命中称为首次因子计算，也不把单个 GA 个体称为一整代。上一 raw run 在第3轮/3840 transitions 停止保存；性能探针使用训练期，不读取 holdout，不视为正式训练或收益验收。完整证据与目标差距见 artifacts/performance_optimization_20260917/report.md。

后续等价性能优化复用唯一累计环形核计算 VolumeCV/Amount：环形存过去累计值，保留原 float32/float64 累计顺序与缺失语义；长动量只按股票列分块，不改变时间窗口求和核。env 复用同日合法价格、持仓市值、持仓 mask 和同一结算数值核，删除已由 concatenate/stack 保证独立所有权后的额外数组复制。CUDA patch 索引整批上传后分块取视图。确定性动作与浓度正值 flag 由同一 typed 数学核进入既有 CUDA Graph，图外仍检查 flag 并复制输出；该纯推理出口不承诺刷新内部瞬态 Beta 对象，取得分布必须调用 get_distribution/proba_distribution。训练 law、RNG、动作端点和所有权仍须独立对照，不能由局部微测推断整轮达标。

第三轮等价优化将 patch/sequence 共用的窗口成员索引在绑定时构造成唯一 CUDA CSR；只上传日期级索引，删除每批 CPU 前缀差分与大 nonzero 路径。CSR buffer 不序列化、不消耗 RNG，重绑用新 buffer 保留既有 forward 的旧反向引用；训练时仍重新计算全部可学习 embedding。新增 utils/stable_sort.py 仅承载两调用方共享的 uint64 稳定排序数值内核，factor/compute 和 env/prefilter 保留各自分数键、PIT、二值/平均 rank 及候选语义；排序内核源码必须进入 policy source identity。因子排序和 raw/validity/rank 写出统一成一个串行核，公开浮点16无损升浮点32、非本机字节序规范化，整数排序保留高于2**53的区别，不提供旧 NumPy fallback。版本化领域/schema/网络精度不变；完整输出/梯度/轨迹一致性和含首次准备的速度分别验收，见 round3_report.md。

第四轮在同一 Transformer 中按每次 forward 的固定 token 输入字节额度保留前缀分块的反向激活，超额分块继续使用原 checkpoint 重算。`retained_token_input_bytes=384MiB` 属于封存网络执行配置，不是总显存上限；禁止查询空闲显存后切策略或 OOM fallback，patch 投影重算与完整股票/历史保持。Bili 生命周期状态链和 styles 有限窗口前缀/后缀核编译为串行数值实现，保留源 float32/64 累加精度、缺失和端点语义；后者只接受 native float32/64，其他 dtype 显式拒绝。删除旧 styles NumPy 循环与列分块，不保留兼容别名；共享核通过完整 run source identity 绑定。planner 只把既有 int64/float64/bool 输入一次打包给原调仓核，再一次转换输出，不改领域数学、交易检查或诊断。参数、输入、Reward 和动作语义不变；工程性能与完整目标差距继续分开验收，见 round4_report.md。
