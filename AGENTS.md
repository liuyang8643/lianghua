参考@CLAUDE.md

## 唯一监督策略与文件执行契约（2026-11-08）

- 策略与回测环境来自研究分支 `codex/atrade-three-horizon-20261106` 的 `98075c2` 及该提交内冻结源码附件。模型直接复制，禁止重训或切换回旧因子加权/PPO入口。`core/` 和研究分支的本地 `trade/` 执行入口已移除，生产入口只有 `run.py → trading.planner_watchdog`。
- `strategy_release.runtime.FrozenStrategyRuntime` 是数据到预测与计划的共同装配；因子公式直接调用 `ai.supervised.feature_data.materialize_factor_values`，百分位归一化唯一实现在 `strategy_release.normalization`，TopK唯一算法在 `strategy_release.selection`。`PreparedModelPortfolio.policy/decision_at` 供完整回测与 `backtest_day` 共用；本地实盘只把单日回测计划转成JSON。股数规划与离线成交使用 `env.DayPlanner/DaySimulator`。默认40/40/3/1，MVO关闭，允许现金，不自动加仓已有股票。
- 完整截面归一化与预测，不采用旧prefilter。股票轴包含历史股票；T日只能使用开盘与官方前收，其他行情截至T-1；ST、财务、股本按T日时点重放，不复制昨日ST、不提前使用未来事件。盘前行情仅内存overlay，不下载或落盘全市场盘中历史。
- `configs/default_supervised_strategy.json` 固定模型包；`configs/config.json` 仅配置输入目录与明确不参与策略交易的资产。`158905.SZ` 原样保留，不占股票名额，其价值不计入可部署股票组合NAV，但完整账户对账仍保留。任何其他未知股票缺历史均阻断。
- `data/strategy_inputs/manifest.json` 绑定研究月度/行业/分红/财务事件输入和核验日期。过期或缺失资料不得用旧基础Runtime冒充完整输入；构建必须补全研究财务字段才可原子发布。初始迁移核验至2026-11-06，后续必须有真实采集证据才能推进核验日期。
- 真实持仓龄由账户绑定的连续日终JSON和逐笔成交重建；首次观察前的持仓显式标记保守日龄下界，不伪造买入日。股数变化无法由成交解释时须对账；新目标冻结模型摘要、容量、日龄和原始持仓。
- 客户端仍为Python3.6，无模型/因子/NumPy依赖。数量规则唯一实现是 `utils.stock.quantity`，研究端仅注册Numba编译；费用参数唯一来自 `utils.fees`，实际资金冻结不含模拟滑点。持仓容量检查计入在途及已成交未回报持仓的买单；未实际清仓不能提前释放名额，约束随journal恢复。
- `trading.preopen` 仅缓存已证明不依赖T-open/preClose的177个因子输出，其他因子仍调用同一实现。缓存绑定历史、代码、模型和输入身份；9:15后不补算缺失的完整历史缓存。发布阶段先预测，再读取新鲜账户分配；账户超过30秒不发布。原始/归一化因子、预测与报价来源保存在本地审计，不写入历史行情。
- 本地发布和客户端目标使用同一个 `FILE_EXECUTOR_PROTOCOL`。旧客户端未重载、未导出新协议标记时不得发布新策略目标。已开始的交易仅恢复原journal，不运行任何旧策略代码。模拟目标明确标记，生产客户端在导入真实SDK前拒绝模拟配置。
- 已移除旧 `trading.scheduler/watchdog/executor/trader/watcher` 及独立RPC交易入口；`qmt_bridge` 只保留客户端共用的SDK/回执适配，不再提供服务进程。唯一客户端准备入口是 `python -m trading.file_setup`，需要安装隔离标准库时显式加 `--install-runtime`。
- `python run.py --restart` 只替换命令行、工作目录、配置路径、Python镜像及启动时间均已核对的本仓库planner；忙于计算/更新时拒绝强停。它不终止券商客户端或数据采集进程，且先完成 `uv sync --locked`。默认重复启动仍不得解除制动。
- `python testback/run_backtest.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --output-dir <新目录>` 使用同一模型和原始因子入口。15:35依据当天冻结目标与真实账户，通过同一DaySimulator零磨损回放，不重选股、不回退空仓或旧策略。
- LightGBM4.6.0和Numba为运行依赖，uv.lock锁定；计算单线程。`python -m trading.simulate_app` 在券商Python3.6执行FakeSDK，禁止真实SDK和网络，不能充当真实权限验证。

## 实盘运行依赖

- 生产架构为纯文件交接：本地 Python 3.12 的 `trading.plan_publisher` 只计算策略并发布 `data/live_file/targets_YYYY-MM-DD.json` 绝对目标股数，不连接/启动/停止 QMT，不逐笔驱动交易 RPC。`python run.py` 与 `python -m trading.main` 均进入单实例 `trading.planner_watchdog`；历史RPC收据仅用于只读审计及基线迁移。
- 已登录客户端的注册策略 `mpython/WBR_EXEC_BRIDGE.py` 加载 `trading/app_executor.py`。客户端 Python 3.6 只读取目标文件、查询账户、计算机械持仓差额和执行交易，不导入因子、选股或数据框架。`trading.execution_engine` 是抽自原执行器的唯一买卖状态机；客户端与测试共同使用该状态机，不能另写两套资金等待、开盘限价、撤单换价规则。
- 客户端独立持有 `data/live_trades/trading_main.lock`，在 `data/live_file/journal/execution_YYYY-MM-DD.json` 中先原子保存原目标、文件哈希及委托标记再报单。首次接收目标须验证账户、日期、期限与格式；已开始执行按 journal 恢复，不依赖本地进程继续运行，也不因源文件到期/删除而重算。重复读取不重复加仓；未明确列出的持仓不清仓；未知委托禁止重发，外部同股活动须先对账，不取消手工单。
- 申万 SDK 实测 `order_stock` 返回 0 时也可能已真实报单/成交；所有非正返回只能按唯一委托标记核验真实正订单 ID，无法确认保持未知，禁止当作“未提交”重发。`not_submitted` 仅用于调用 SDK 前确定未调用的拒绝。错误旧记录须制动后用 `trading.repair_order_receipts` 对新鲜只读券商事实逐笔验证、备份并修正，保留原目标及旧错误审计，不重置 journal。
- 客户端导出账户、持仓、委托、真实成交和银证流水 JSON；本地仅用 `trading.file_account.FileAccount` 读文件。T-1 基线必须绑定相同账户并通过资产/持仓完整性校验；无绑定的旧 Parquet 不能自动当真实基线。`trading.file_setup --migrate-eod YYYY-MM-DD` 仅允许从同账户、收盘后、相同实例且时间接近的真实 RPC 收据迁移，保留原采集时间和请求 ID，不伪装成新查询。
- `data/live_file/config.json` 的 `execution_mode=file`，默认 `live_enabled=false`。只读模式不注册交易回调，不报撤单；启用实盘仍须券商父进程身份、已验证交易日历、连续竞价时段、全局单实例锁和紧急制动检查。不得用只读成功、模拟委托或无关手工成交宣称真实报单权限已验证。
- 每个策略 Python 进程仅创建一个 SDK 实例，失联在同实例上重新 connect/subscribe 并恢复 journal；禁止循环构造 SDK（原生资源可能累积）。初始化失败结束本策略，由受控策略冷启动恢复，不退出或重启用户已登录客户端。`stop.request` 在连接重试和执行 tick 中受控停止策略；制动须验证 PID、启动时间、命令行、当前仓库、配置及父客户端身份，然后先写 halt/stop 文件，保留 Qt 客户端。下一次人工 `python run.py` 清除本仓库制动文件；自动启动必须用 `python run.py --preserve-halt`，不得清除制动。
- 客户端回调排队后串行写 `data/live_file/execution.sqlite3`；账户/审计导出失败只告警，不中断已经开始的真实执行。15:35 本地文件适配器先将券商事实落到原 Parquet，再由 `trading.file_post_close` 复用原对账/本地 HTML 报告。迟到成交按自身交易日期筛选，未知/缺失证据不得假装空成交；出入金继续独立归因。
- 每日两端各归档一次不可覆盖 JSON：本地 `daily/planner_YYYY-MM-DD.json` 保存冻结目标与初始持仓，客户端 `daily/executor_YYYY-MM-DD.json` 在完成或15:35日终保存实际持仓、逐笔委托/成交、差额和完成状态。目录均在 `data/live_file/`。新目标保留 `planning_positions`，客户端首次执行保留 `initial_account`；旧记录补档须明确标注依据冻结计划重建股数，不伪造原始查询时间。日志失败不阻断真实执行。
- `trading.daily_notify` 由本地 planner 串行转发，开关 `data/live_file/notification_config.json` 的 `enabled`；归档不依赖飞书开关。复用原实盘日报五列表格，一天创建一次卡片，客户端结果更新同一消息；发送回执存在 `daily/feishu_YYYY-MM-DD.json`，未知回执禁止重复创建，成功审计可恢复message_id。飞书凭据和网络发送只在本地，客户端不导入飞书/策略框架。模拟日报必须标记并拒发，模拟目标不能发布到生产目录或被生产执行器接受。
- 文件日报仅转换数据，直接复用 `TradingDayBoard(auto_publish=False)` 原渲染器，禁止另写简化版卡片；原排序、合计、颜色、日志及盘后盈亏标题保持一致。15:35 文件对账验证并镜像同账户 T-1 日终 JSON 到原 Parquet（保留来源哈希，未知历史盈亏/成交数不补零），冻结目标适配原 plan；原 `PostCloseReport` 生成 HTML 和盈亏/滑点卡片，`daily/close_YYYY-MM-DD.json` 交给本地串行转发，更新同一主卡并各发送一次附件/明细，未知回执禁止重发。
- 运行前 `uv sync --locked`。客户端继续使用 `data/qmt_bridge_runtime/` 独立 Python 3.6 标准库、已锁定 XtQuant 与固定 psutil 5.9.8 cp36 官方 wheel；不得替换券商 exe/Python DLL 或绕过权限。`python -m trading.file_setup` 准备只读文件配置和已注册入口，入口更新只重载本策略，不退出客户端。
- `python -m trading.simulate_app` 在券商自带 Python 3.6 中运行 FakeSDK 和同一执行状态机。测试禁止导入真实 trader/data SDK 模块和网络连接，覆盖部分成交换价、卖出回款等待、拒单重试、丢确认后的 journal 恢复；报告明确真实连接/报撤单为零，不能充当真实权限证据。

- 09:25 盘前行情使用 `requests` 批量获取腾讯 T-open，仅形成内存 overlay，不写入 Parquet 或 NPZ。
- T-open overlay 的 ST 状态按截至各目标日已生效的最新已验证 ST 事件计算，和历史 Runtime 复用 `data.db.stock_name.build_st_mask`；不能复制 T-1 的 ST 状态，也不能提前应用未来事件。
- 纯文件 16:00 全量更新的日线只刷新已收盘当天（`history_days=1`），因为该路径从不落盘盘中日线；既有缺口仍由启动检查及单日备用源的历史完整性检查阻断。腾讯收盘备用源从 15:35 起可用，报价本身必须属于目标日且已到收盘。
- 巨潮 HTTPS 数据请求不使用环境代理；仅系统 DNS 失败时，`data.update_cninfo` 用阿里公共 HTTPS DNS 的有效 TTL 地址连接，并保留原域名 SNI、证书及主机名验证。依赖 dnspython~=2.8.0 锁定在 pyproject/uv.lock，仅本地数据端使用。不修改系统 DNS/VPN，不关闭 TLS 校验。更名历史原始响应按当日断点保存，失败不能写完成标记。
- 纯文件生产路径历史 K 线经 `data/update_kline.py` 显式使用 `allow_qmt=false`，股票池来自 mootdx，任何失败禁止回退 QMT。旧独立诊断路径仍保留 mootdx 失败后使用已登录 QMT 不复权日线和官方 preClose 的能力。QMT 必须主动下载并健康探测，按交易日索引转换为现有 Parquet 的无时区 15:00 时间，成交量保持手；禁止直接拼接 QMT 原始 time。`py-mini-racer` 用于 Windows 下 AkShare 兼容；requests、mootdx、py-mini-racer 必须记录在 `pyproject.toml` 和 `uv.lock`。
- QMT 补数逐股核验交易日覆盖；缺日仅按停复牌资料覆盖全天的时段豁免。资料存 `data/market_status/`，历史列表遗漏时可使用附公告来源和明确起止时间的 `configs/verified_suspensions.json`。发行资料确认待上市，或上市日晚于本次下载截止日，才允许无日线；已退市且本地存在的历史不由失去合约精度的 QMT 覆盖。
- 纯文件补数仅缺一个交易日且 mootdx 日线失败时，`data.update_tencent_close` 可使用有原始报文的当日收盘腾讯行情；核验日期、收盘时间、量额单位、前收、最近 60 个交易日覆盖及停牌/上市证据，全部验证后再写入。新股缺历史由 `data.update_sina_history` 读取新浪不复权历史并复用 mootdx 真实除权数据，末日与腾讯独立交叉核验。多日缺口不得伪装成单日补数，不得回退 QMT。

## 实盘启动约束

- 自动/手动实盘启动校验交易日历、Runtime 可读性、字段维度、最近 60 天交易日连续性及 T-1 有效行情；缺失时立即按缺口补下载并重建，重检通过才生成新计划。常驻跨日重检，失败每五分钟重试。已有执行记录先恢复委托，15:35 后启动先采集券商事实再补数。启动补数截止 T-1，使用独立 startup checkpoint，不占用 16:00 更新标记；`--skip` 时间点回放保留原阶段语义。
- 启动补数的 `stock_status` 步骤真实刷新完整 ST 事件和当前简称并验证活跃历史股票覆盖；策略与 Runtime 不依赖旧简称的展示历史，因此启动不逐股拉取 `p_stock2109`。16:00 `stock_name` 的必需项为完整 ST 事件及当前简称覆盖；历史更名 `p_stock2109` 仅用于展示，限时尽力更新，超时/失败记录独立 `name_history_status_日期.json=skipped` 并保留旧数据，不阻断 Runtime。不得把可选展示资料失败当作 ST 已验证，也不得将 skipped 标成历史资料完整成功。
- 根目录 `python run.py` 自动执行 `uv sync --locked`，失败不启动、不清除制动；成功后只用 `.venv/Scripts/python.exe` 后台启动单实例 watchdog，并把 stdout/stderr 写入 `configs/logs/watchdog-*.log`。
- 根目录 `run.py` 为人工启动入口，替代旧 PowerShell 脚本；仅取得 planner 单实例锁后才清除本仓库 halt/stop 标记，重复启动不得解除制动。用户另外在已登录客户端中点击运行 `WBR_EXEC_BRIDGE`；测试使用临时目录及进程替身，不得实际启动实盘。
- 本地 planner 24x7 跨日运行，不管理 QMT 进程；启动/跨日历史预检前移到发布窗口之外，按真实缺口补数并重检，16:00 复用 `run_update_all(allow_qmt=false)` 的独立 checkpoint 与完整下载/Runtime 校验。客户端内的 SDK 同实例重连后恢复原 journal；两端均不得退出用户已登录的客户端。
- 本地生产 CLI 仅提供文件 planner，不接受本地下单 `--confirm` 或旧实盘 `--skip` 阶段驱动；历史诊断走离线回测入口，执行回归走 `simulate_app`，均不得把模拟目标发布到客户端当日生产文件。
- dry-run/历史回放不得注册 QMT 交易回调，也不得覆盖 `trading_main.pid`；只读 Trader 必须禁止下单和撤单。同日 dry-run 仍查询真实账户，历史回放使用 T-1 账户基线，不以空仓代替查询。
- 可能下单的唯一进程为已登录客户端内的 file executor，持有原 `trading_main.lock`；本地 `trading.main` 只发布 JSON，不下单。旧 RPC/原生执行进程不得与该进程并行启用。
- 飞书紧急制动必须同时校验 PID 对应进程的命令行、当前仓库工作目录、实盘配置路径与启动时间，陈旧、被复用或属于其他仓库的 PID 不得终止；验证后先写 watchdog halt 标记，防止主进程被自动拉起，下一次人工运行 `python run.py` 才清除该标记。
- QMT 回调的 Parquet 事件写入必须跨进程串行；本地审计写入失败只能告警，不得中断已经开始的真实下单。
- watchdog 及子进程统一使用 UTF-8 输出。实盘下单前将原始目标、开盘限价截止时间与唯一委托标记保存到 `data/live_file/journal/execution_YYYY-MM-DD.json`，本地对账按账户/日期/执行 ID 核验后镜像到原审计目录；重启按券商委托恢复未覆盖余量。无法确认的委托禁止重发，已有委托却缺少执行记录时须先对账。完成状态必须验证目标覆盖，执行开始后的异常允许在 10:00 之后恢复，资金不足保留目标等待至收盘。

## 实盘日内时序

- 沿用 `liuyang-dev`：09:25:10 完成盘前计算并启动执行器，执行器到 09:30 连续竞价后才允许发单；15:35 运行盘后实盘/回测对账，16:00 运行全量数据更新并重建 Runtime。
- 15:35 必须先采集并落盘 QMT 资产、成交、持仓与日终账户基线，再执行 K 线下载和回测；后续分析失败不得丢失券商事实。
- 盘后执行对账固定当天原始计划（execution 日志优先，兼容 plan），使用昨日真实持仓与同一收盘估值做零磨损回放；差额拆为滑点、估算手续费、成交数量差损益和未解释残差。独立策略回测仅作策略诊断，账户出入金及其他现金变化单列，不能当执行偏差。个股盈亏只用市值变化与真实成交现金流，禁止用券商成本价反推成交；缺基线、成交证据或估值必须标不可验证，不得用累计持仓盈亏或账户资产变化补齐。
- 16:00 更新使用按日期隔离的步骤 checkpoint；任一下载失败时禁止构建 Runtime，仅在 Runtime 末日校验通过后写 final marker。Runtime 必须临时文件完整写入后再原子替换。
- 开盘前启动的订单先用 `open[T]` 限价等待至 09:31；盘中手动 `--confirm` 从启动时起保留开盘限价 60 秒。未成交余量随后改用对手价，并按 30 秒轮换。
- mootdx 必须从有限候选节点中探测出能实际返回日线的节点后再更新；全市场更新前还需做股票样本健康探测，运行中连续空数据必须熔断，禁止在数据源整体失效时串行等待完整股票池。
- 开盘前 Runtime 必须包含上一交易日；腾讯同日有效响应的 `open=0` 表示无可执行开盘价，保留零值交给合法性过滤，不算网络缺失；全部开盘价为零仍视为行情未就绪。
- 发行价读取巨潮上市资料的发行价格字段；已由公司公告核实的不发新股转板保留 NaN 并记录状态，不将开盘参考价当发行价。同花顺历史财务页面失效时使用新浪报告期指标备用源，两者 HTTP 请求均有 15 秒超时。解析/网络失败不得伪装成成功 checkpoint。
