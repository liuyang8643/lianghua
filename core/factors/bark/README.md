# bark — 冗余/已淘汰因子墓地

以下因子因与现有因子高度重复而被移入此目录，非丢失，可按需恢复。

| 文件 | 冗余原因 | 被谁替代 |
|------|----------|----------|
| RSI14Reversal.py | 与ShortTermReversal同概念（均值回复=超卖买入） | ShortTermReversal |
| WillR14.py | RSI≈100-WillR，数学等价 | RSI14Reversal/ShortTermReversal |
| ROC.py | 12日变动率，与CloseMom21D同构(仅窗口不同) | CloseMom21D |
| GapDown.py | -OvernightGap1D，仅符号相反 | OvernightGap1D |
| HistoricalGap.py | OvernightGap1D的20日均值基准版，GA从未使用 | OvernightGap1D |
| Valuation.py | EP+BP，EPValuation是EP | EPValuation |
| Growth.py | revenue_yoy，与ProfitYoy同属增长率 | ProfitYoy |
| Profitability.py | (ROE-15)/5+(GM-40)/10 固定参数版，GA从未使用 | ROE |
| TMC_GARP_Quality.py | TMC_GARP_Broad的真子集(仅少revenue_yoy) | TMC_GARP_Broad |
| SmallCapDailyMVMaskRoe2xBottom10.py | -MV+0.2*ROE，可在TMC_GARP_Broad框架表达 | TMC_GARP_Broad |
