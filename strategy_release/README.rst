默认策略交接：D模型 + TopK 缓冲
=================================

默认选择由 configs/default_supervised_strategy.json 指定，strategy_release.load_default() 实际读取它。
当前发布包 d_model_topk_20261111：拟合2005—2025，2026截至11月10日的成熟标签用于早停。
实际训练783轮，最佳583轮，278个因子；最多2000轮、连续200轮未改善早停。
周期3/7/15/30/60，各20%；使用完整横截面各周期百分位排名等权融合。
buy_n=40，sell_m=40，最短持仓3个交易日，每日最多主动换出2只；允许留现金，不自动补已有仓位。
MVO只保留研究归档，不作为默认。

拉取后先验证
------------
安装 strategy_release/requirements.txt 中的依赖，再在仓库根目录运行：

    python -m strategy_release

模型、元数据、因子顺序、归一化规则和校验摘要均随Git提交，不依赖本机D盘模型路径。
旧 configs/config.json / strategy.yaml 仍是研究静态基准，不能当作这套监督策略配置。

实盘仓库接入边界
----------------
这个包不下单。已有 trade.main 是PPO入口，不会因拉取分支自动变成LightGBM入口。
实盘仓库应调用 load_default()，将自己的数据/账户/成交接口接到下面三个步骤：

1. 按 features.json 的确切顺序计算278个因子；T日只允许open，其余行情使用此前已完成K线，事件使用当时已知版本。
   research_source.zip 是实际研究用因子实现和数据适配源码，配有逐文件摘要；供移植对照，不能用未来事件重算过去状态。
   数据下载、PIT公告/股本、月度数据、除权除息参考价和完整股票池仍由实盘数据提供者维护。
2. 对完整股票截面执行 normalize(raw, feature_names=..., eligible=...)，再 predict(..., codes=..., complete_universe=True)。
   不能只对持仓或候选前40只股票排名。eligible与可买状态不同；预测排名不能先按涨跌停可买性缩小股票池。
   保持稳定股票轴顺序用于同分排序。NaN仍为缺失，不补零。周期编码为0..4，不是3/7/15/30/60数值本身。
3. 用已确认账户调用 targets(scores, held=..., current_weights=..., buyable=..., sellable=..., holding_days=...)。
   current_weights以当前开盘估值NAV计算；holding_days是交易日龄；买卖合法性由实盘数据与账户共同决定。
   返回的是目标权重及退出索引，不是已经成交的订单。卖出失败不能释放持仓名额，买入须按真实可用现金、整手、费用执行。
   T+1不可卖仓位应标sellable=False；持仓超过40只时接口要求先明确迁移规则，不擅自清仓。

校验与限制
----------
包在完整历史截面上与已保存预测做一致性校验，TopK分配与研究原实现做随机场景一致性校验。
这不是券商接口上线认证，也不证明看见开盘价后能以相同价格成交；没有设置定时任务或发出订单。
研究回测使用修正后的preClose进行合法性、权益调整及缺失行情估值，并非纯原始价收益。
已有其他本地改动未混入本次提交；完整研究源码作为独立交接附件保存，不依赖未提交的工作树导入。

D版固化与复核
------------

本次只更新可移植监督策略默认包，不修改静态基准，也不执行券商订单。
40/40、最短持仓3个交易日、最多主动换出2只；新开仓预算为当前NAV/40。
初始建仓不受2只卖出上限限制；持有满3日只是允许退出，不要求第三日卖出。
回测默认单边滑点0.25%，其余费用见manifest.json；实际订单执行应遵守实盘账户费用与可成交性。

发布包包含冻结模型、因子顺序与归一化、训练来源、各周期因子IC和选定参数的年度回测摘要。
2026参与早停，不作为新样本外成绩；历史报告已经被观察。
2026早停采用统一60日成熟边界，最后可用样本日2026-08-10，标签结束于11月10日开盘。

研究机可用以下命令核对完整截面的预测：

    python scripts/verify_default_strategy.py --study artifacts/research/d_supervised_20261110/D --model-name D --output artifacts/research/d_release_verification.json

模型与归一化推理只依赖strategy_release/requirements.txt，不依赖本机artifacts路径。
research_source.zip保留因子实现和D训练程序；重新训练仍需要对应原始数据与输入清单。
旧new_model_topk_20261108和multistart_topk_20261109发布包保留，可用原manifest摘要另建配置回退。
