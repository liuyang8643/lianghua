1、避免冗余输出：不要输出冗余md，不要过度回复和解释，给出精简结论即可。
2、代码精简：时刻注意代码精简，避免防御性编程、冗余容错try get等逻辑。
3、任务完成后，需要二次review项目简洁性。
4、所有输出和回复用中文。

## 核心规则

1. **验收规则**：完成实质性产出后，必须调用 verify agent 做独立验收，不得自行报告完成。
2. **调试规则**：先小量（10~50只/7~30天）→ 全量 → 长周期。>20s 无日志 → 卡死，立即 kill。运行用 `python -u`。
3. **代码精简**：避免防御性编程、冗余 try/except、无意义抽象。写完回头删废代码。
4. **执行优先级**：agent-team > subagent > main-agent + verify-agent。优先用 team 并行分发任务。
5. **GA 运行前强制清理**：`powershell Stop-Process -Name python -Force` 杀后台 → sleep 3 → 二次确认进程数=0 且空闲内存>30GB。不跳过，否则必报 PermissionError/WinError 5 拒绝访问。