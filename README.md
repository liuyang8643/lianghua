# WBR：唯一冻结策略的回测与文件实盘

部署策略来自研究提交 `98075c2`：284因子、523棵树，周期3/7/15/30/60按20/30/10/30/10%融合；TopK参数40/40/3/1，MVO关闭。直接使用冻结模型，不重训。

## 唯一计算链

`strategy_release.runtime` 计算因子与预测 → `PreparedModelPortfolio.decision_at` 分配 → `env.DayPlanner` 生成股数计划。

完整回测与本地实盘共用这条链。本地实盘调用 `PreparedModelPortfolio.backtest_day`，将当天计划转换成绝对目标股数JSON；历史回测用 `env.DaySimulator` 模拟成交及下一开盘结算。当天没有未来行情，因此单日入口只产生计划，不伪造未来收益。

券商内置Python3.6只读取JSON、查询账户、执行机械差额。唯一执行状态机为 `trading.execution_engine`；不在客户端安装模型或复制策略。交易数量规则共用 `utils.stock.quantity`，费用参数共用 `utils.fees`。

## 运行

本地Python3.12环境：

```powershell
uv sync --locked
python testback/run_backtest.py --start-date 2022-01-01 --end-date 2026-10-31 --output-dir results/new_model_check
```

离线计算需要已验证的Runtime和 `data/strategy_inputs/`，不在回测中联网补数据。`configs/default_supervised_strategy.json` 固定模型包；`configs/config.json` 只设置输入目录及明确不参与策略交易的资产。

生产人工入口为 `python run.py`；显式替换已核验的本地旧进程用 `python run.py --restart`，自动启动必须用 `python run.py --preserve-halt`。客户端另行重载已注册的 `WBR_EXEC_BRIDGE`，不会退出或重启已登录客户端。新协议握手会拒绝向未重载的旧客户端发布新目标。启动和目标发布前都必须满足 [AGENTS.md](AGENTS.md) 的数据、账户、时段、单实例及制动契约。

## 盘前与回放

- 完整股票截面归一化、预测，不使用旧prefilter。
- 只允许T-open及截至T-1的行情，事件按时点生效；不得盘前下载全市场历史。
- 已完成行情可确定的因子，在发布窗口前用同一实现计算并缓存；实际开盘价到达后计算剩余因子。缓存必须绑定同一数据、代码和模型，不能用旧缓存降级。
- 历史持仓龄根据账户绑定的日终与真实成交重建；首次观察前的持仓明确保留日龄下界，不伪造买入日。
- `158905.SZ` 不参与股票策略交易且不占股票名额，仍保留在完整账户对账中。策略可部署NAV为现金加受管理股票开盘市值。
- 15:35按当天冻结目标，通过同一 `DaySimulator` 做零磨损执行对账，不重新选股或回退旧策略。

## 执行验证

```powershell
python -m trading.simulate_app
python -m pytest tests/test_frozen_single_day_parity.py trading/test_strategy_release_integration.py
```

模拟使用券商自带Python3.6和FakeSDK；真实连接、报单、撤单均为零，不能用模拟结果宣称真实交易权限已验证。

数据更新由独立任务负责。当前接口、已改文件和待验事项见 [数据侧交接](docs/strategy-integration-data-handoff.md)。旧研究成果保留在Git及交付附件中，生产入口不兼容旧策略切换。
