from pathlib import Path

path = Path('AGENTS.md')
text = path.read_text(encoding='utf-8')
start = text.index('## 实盘运行依赖')
end = text.index('- 09:25 盘前行情', start)
replacement = '''## 实盘运行依赖

- 生产架构为纯文件交接：本地 Python 3.12 的 `trading.plan_publisher` 只计算策略并发布 `data/live_file/targets_YYYY-MM-DD.json` 绝对目标股数，不连接/启动/停止 QMT，不逐笔驱动交易 RPC。`run.ps1` 与 `python -m trading.main` 均进入单实例 `trading.planner_watchdog`；旧 RPC/原生适配器仅保留兼容诊断，不是生产执行入口。
- 已登录客户端的注册策略 `mpython/WBR_EXEC_BRIDGE.py` 加载 `trading/app_executor.py`。客户端 Python 3.6 只读取目标文件、查询账户、计算机械持仓差额和执行交易，不导入因子、选股或数据框架。`trading.execution_engine` 是抽自原执行器的唯一买卖状态机；`trading.executor` 是旧诊断适配器，不能另写两套资金等待、开盘限价、撤单换价规则。
- 客户端独立持有 `data/live_trades/trading_main.lock`，在 `data/live_file/journal/execution_YYYY-MM-DD.json` 中先原子保存原目标、文件哈希及委托标记再报单。首次接收目标须验证账户、日期、期限与格式；已开始执行按 journal 恢复，不依赖本地进程继续运行，也不因源文件到期/删除而重算。重复读取不重复加仓；未明确列出的持仓不清仓；未知委托禁止重发，外部同股活动须先对账，不取消手工单。
- 客户端导出账户、持仓、委托、真实成交和银证流水 JSON；本地仅用 `trading.file_account.FileAccount` 读文件。T-1 基线必须绑定相同账户并通过资产/持仓完整性校验；无绑定的旧 Parquet 不能自动当真实基线。`trading.file_setup --migrate-eod YYYY-MM-DD` 仅允许从同账户、收盘后、相同实例且时间接近的真实 RPC 收据迁移，保留原采集时间和请求 ID，不伪装成新查询。
- `data/live_file/config.json` 的 `execution_mode=file`，默认 `live_enabled=false`。只读模式不注册交易回调，不报撤单；启用实盘仍须券商父进程身份、已验证交易日历、连续竞价时段、全局单实例锁和紧急制动检查。不得用只读成功、模拟委托或无关手工成交宣称真实报单权限已验证。
- SDK 失联只在原策略 Python 进程内重建 SDK 会话并恢复 journal，不退出或重启用户已登录的客户端。`stop.request` 在执行 tick 中受控停止策略；制动须验证 PID、启动时间、命令行、当前仓库、配置及父客户端身份，然后先写 halt/stop 文件，保留 Qt 客户端。下一次人工 `run.ps1` 清除本仓库制动文件。
- 客户端回调排队后串行写 `data/live_file/execution.sqlite3`；账户/审计导出失败只告警，不中断已经开始的真实执行。15:35 本地文件适配器先将券商事实落到原 Parquet，再由 `trading.file_post_close` 复用原对账/本地 HTML 报告。迟到成交按自身交易日期筛选，未知/缺失证据不得假装空成交；出入金继续独立归因。
- 运行前 `uv sync --locked`。客户端继续使用 `data/qmt_bridge_runtime/` 独立 Python 3.6 标准库、已锁定 XtQuant 与固定 psutil 5.9.8 cp36 官方 wheel；不得替换券商 exe/Python DLL 或绕过权限。`python -m trading.file_setup` 准备只读文件配置和已注册入口，入口更新只重载本策略，不退出客户端。
- `python -m trading.simulate_app` 在券商自带 Python 3.6 中运行 FakeSDK 和同一执行状态机。测试禁止导入真实 trader/data SDK 模块和网络连接，覆盖部分成交换价、卖出回款等待、拒单重试、丢确认后的 journal 恢复；报告明确真实连接/报撤单为零，不能充当真实权限证据。

'''
text = text[:start] + replacement + text[end:]
text = text.replace('- 历史 K 线统一经 `data/update_kline.py`：优先 mootdx，失败后使用已登录 QMT（申万宏源/国金）的不复权日线和官方 preClose。',
                    '- 纯文件生产路径历史 K 线经 `data/update_kline.py` 显式使用 `allow_qmt=false`，股票池来自 mootdx，任何失败禁止回退 QMT。旧独立诊断路径仍保留 mootdx 失败后使用已登录 QMT 不复权日线和官方 preClose 的能力。')
text = text.replace('- watchdog 24x7 守护 scheduler 跨日运行，不在 16:00 强制退出；主进程失败时只重启主进程，终止旧主进程后必须确认其确已退出，否则保留旧引用并禁止启动新实例；启动前已存在的 QMT 只监控、不得终止。',
                    '- 本地 planner 24x7 跨日运行，不管理 QMT 进程；启动/跨日历史预检前移到发布窗口之外，按真实缺口补数并重检，16:00 复用 `run_update_all(allow_qmt=false)` 的独立 checkpoint 与完整下载/Runtime 校验。客户端内的 SDK 重建只恢复原 journal；两端均不得退出用户已登录的客户端。')
text = text.replace('- `--skip YYYYMMDDHHMM[SS]` 是只执行指定时间点阶段的模拟参数，必须配合 `--dry-run`；不会继续快进到盘后或 16:00 全量更新。',
                    '- 本地生产 CLI 仅提供文件 planner，不接受本地下单 `--confirm` 或旧实盘 `--skip` 阶段驱动；历史诊断走离线回测入口，执行回归走 `simulate_app`，均不得把模拟目标发布到客户端当日生产文件。')
text = text.replace('- 自动实盘和 `--confirm` 手动实盘共用单实例锁；同一时间只允许一个可能下单的 `trading.main` 进程。',
                    '- 可能下单的唯一进程为已登录客户端内的 file executor，持有原 `trading_main.lock`；本地 `trading.main` 只发布 JSON，不下单。旧 RPC/原生执行进程不得与该进程并行启用。')
text = text.replace('实盘下单前将原始目标、开盘限价截止时间与唯一委托标记保存到 `data/live_trades/execution_YYYY-MM-DD.json`；',
                    '实盘下单前将原始目标、开盘限价截止时间与唯一委托标记保存到 `data/live_file/journal/execution_YYYY-MM-DD.json`，本地对账按账户/日期/执行 ID 核验后镜像到原审计目录；')
path.write_text(text, encoding='utf-8')
