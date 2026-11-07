# 多周期研究与详细报告复现

本记录覆盖本线程的冻结研究、preClose修复和原版详细报告恢复。禁止以汇总HTML代替详细回测报告。

## 已完成且已验收

- 原研究：`artifacts/research/multihorizon_oof_ga_20261106_windowfix`。训练2005—2017，OOF2011—2017，外验2018—2021，原测试2022—2026-10-30。
- 已修数据：`artifacts/research/multihorizon_2026_explain_20261107_reference_complete`。仅preClose累计7429格修正，模型不变；其`completion.json`和`independent_review.json`绑定原始来源、代码、数据和结果。
- 全区间对照：`artifacts/research/multihorizon_all_periods_preclose_20261107`。14次原生回放；以`completion_receipt.json`及补充展示审查为准。
- 详细报告：`artifacts/research/multihorizon_detailed_reports_20261107`。14段详细模式与前述NAV、现金、仓位、日期严格一致，141540条实际成交。默认入口`reports/ml_test/single_report.html`，2026解释入口`reports/ml_ytd2026/single_report.html`。

## 唯一执行与呈现链

修正源及准入 → 因子/标签 → 折内筛选 → 年度模型 → 连续OOF组合参数搜索 → 固定轮数最终模型 → 原生账户回测 → `trace_to_report_result` → `generate_single_report`。

研究运行使用项目`.venv`、既有`qlib_open_rank_20261105/packages`和`run_research_guarded.py`。每阶段保存源码与输入SHA，保存失败实物，未完成模型不得盲目续跑；旧封存根不覆盖。较大数据和模型保存在本地artifacts，Git提交不是完整数据备份。

原报告保留月度、交易、当前持仓、已清仓、每日快照、退市六Tab和股票K线。汇总/解释为补充Tab或链接。只有摘要轨迹时必须补详细模式并核对财务一致；不得捏造成交。

## 当前限制和后续修复

已清仓Tab显示N/A并非按钮故障，而是原生报告适配器未提供逐持仓生命周期损益。应使用实际账户变化、真实成交、部分卖出、补仓、公司行动合成股数/现金残差和退市事件进行经济口径配对并对账；不能把简单买卖数量累计当作真实平仓。

保留同开盘决策/成交、原滑点与权益再投资近似；不等于券商逐笔账本或真实竞价成交能力。浏览器工具拒绝自动本地文件导航；已做离线数据、真实JS端口模拟和独立审核，不冒称浏览器视觉验收。WBR-live仅作只读参考。

## 扩大训练的新一轮

配置为`configs/research/multihorizon_pre_live_20261107.json`，状态明确为尚未启动。训练扩大至2005—2021；原测试段改为2022—2026-10-30外部验证，仅供观察。无新的独立测试集，不把既有已观察历史称为盲测。

须先移除旧setup/workflow中写死的训练年、旧runtime和旧测试输出假设，再在新根运行。旧`run_multihorizon_study.ps1`不能直接冒充新配置入口。新用户授权训练不包含向WBR-live部署或发送真实订单。
