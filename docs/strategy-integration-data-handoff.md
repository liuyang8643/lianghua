# 数据侧交接：98075c2 策略整合

用户已将数据更新交给另一个 agent。本策略整合任务停止修改和执行数据更新。

## 当前状态

- 本次保存合入研究提交 `98075c29b34ac4a8f8643082e4789d40008aa9e8` 及文件实盘接入代码；保存不代表数据更新链验收或实盘启用。
- 合入前源码备份：`results/integration_98075c2_20261108/before.zip`、`before_manifest.json`；Git stash 名称为 `before-98075c2-integration-20261108`。不能整体恢复 stash，否则会恢复已淘汰的旧策略。
- `strategy_release/new_model_topk_20261108/research_source.zip` 的冻结实现已展开到对应模块；模型未重训。
- 策略已通过 284 原始因子核验日重算、两段完整账户回放。数据更新器的新增适配尚未全部验收，不能据此声称生产更新链已就绪。

## 本任务已动过的数据侧文件（后续交由数据 agent）

- `data/build_runtime.py`：公开入口转交 `data.strategy_runtime.build_runtime`；ST 构建转为共享 `data.db.stock_name.build_st_mask`；原子保存返回路径；非新股转板仍保留上市日期而发行价可为空。
- `data/strategy_runtime.py`：新增保留冻结历史并扩展收盘日期的草稿实现，**尚未跑完整生产更新验收**。重点检查不可覆盖 seed、无新日期但新增股票、损坏 Runtime 恢复、股本固定基线及完整缺口检查。
- `data/strategy_inputs.py`：本地事件重放和内存 T-open overlay；提供 `with_open_overlay`、`replace_open_quotes`、`financial_panels`。输出数组必须只读；当前策略端依赖这几个接口。
- `data/update_strategy_financial.py`、`offline_data/financial_store.py`：新增只读公开源采集与不可变财务表存储。新浪为主、东方财富为备用/事件索引；未连接 QMT。已实际更新至 2026-11-07，响应与来源保存在 `data/strategy_inputs/public_financial_raw/`。
- `offline_data/financial_versions.py`、`offline_data/capital_history.py`：新增可选 `known_after_date`，防止后来采到的修订回写早先决策；不修改原始公告日期。需要数据侧审查并纳入测试。
- `data/db/stock_name.py`：ST 算法迁到此处；恢复当前名称显示入口。`data/update_all.py`：发行资料改用带摘要的原子保存入口。
- `data/kline_mootdx.py`：合入研究修复/证据逻辑，保留有限节点、股票列表独立健康检查、连接关闭、样本健康探测和连续失败熔断；默认严格失败。相关测试已过。
- `offline_data/runtime.py`：按文件内交易日末日选择最新 Runtime，避免两种文件名格式排序错误。
- `utils/eastmoney_request.py`：默认有界请求超时改为 15 秒。

## 已复制的数据资产

- `data/runtime/runtime_1990_20261106_reference_complete.npz`：研究已验证 Runtime。
- `data/strategy_inputs/`：绑定因子列表、行业、分红、月度信号与原始财务档案。`financial_manifest.json` 指向统一财务表；旧 `financial_0..2` 仅为迁移来源证据。
- `manifest.json` 有 `runtime_seed`、`runtime_seed_sha256`、`runtime_seed_end`、`runtime_seed_stocks`。历史 seed 不得被普通日更删除或改写。
- `data/live_file/position_history.json` 是本任务从真实日终与逐笔成交重建的持仓龄证据，不是模拟账户或交易目标。

## 策略所需数据接口

1. 只读 `RuntimeSlice`：稳定完整股票轴、完整历史（IPO累计与递归因子不能任意截断）、原始 OHLCVA/preClose、上市/退市/ST，以及冻结研究所需 PIT 财务字段。
2. `strategy_release.runtime.FeatureInputs.load(directory)`：读取并校验 `manifest.json` 中绑定的 `monthly.npz`、`industry.parquet`、`dividends.parquet`，以及 `bound_names`。
3. `offline_data.financial_store.read_tables(directory)`：返回 Income/Balance/CashFlow/PershareIndex/Capital 的标准化日期表和来源清单。旧历史值保留，不用新采集值重写过去。
4. T-open 只替换当天开盘价/官方前收；其他当日行情为 NaN；ST、公告与股本按当时已知且已生效事件计算。所有返回数组只读。
5. 发布前完整 active 股票报价覆盖，`open=0` 是真实无可执行价，不能按缺请求处理。缺响应则拒绝宣称完整截面。
6. 财务和事件覆盖必须包含周末/节假日；周一只有周五资料不能冒充周日已核验。月度信号沿研究 `monthly_pulse` 的下一开盘生效规则。

本任务接下来只维护 `strategy_release`、回测/实盘单日装配、JSON 与客户端执行及测试。数据更新、来源选择和数据侧完成标记由数据 agent 负责。
