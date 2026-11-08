研究归档，不是默认实盘策略。默认采用 configs/default_supervised_strategy.json。
这些脚本按原研究根目录布局执行，需要已有11折模型、预测和行情数据；Git不包含这些大型输入。请在 artifacts/research 下建立新的研究目录，保留原父研究目录关系；禁止在封存结果目录直接重跑。
原 mvo_adapter.py 与 run.py 保留搜索时字节。详细模式汇总接口的修正见 mvo_adapter_report.py；恢复记录及最终重放一致性见完成记录。
MVO结果不优于当前TopK，未设置为默认。
