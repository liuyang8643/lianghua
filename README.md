# WBR 量化交易系统


## 环境要求

- Windows + Python 3.12 + [QMT 客户端]
- uv 包管理器

### 安装

```powershell
# 安装 uv
irm https://astral.sh/uv/install.ps1 | iex

# 安装依赖
uv sync --locked
```

## 实盘运行

已有文件执行配置的电脑，重启后只需两步：

1. 运行根目录 Python 启动脚本：

   ```powershell
   cd D:\coding\WBR-live
   python run.py
   ```

   自动执行 `uv sync --locked`，成功后用 `.venv/Scripts/python.exe` 后台启动单实例本地 planner。可关闭启动窗口，日志在 `configs/logs/watchdog-*.log`。启动器不连接或管理 QMT。

2. 登录券商客户端，在客户端内点击运行已注册的 `WBR_EXEC_BRIDGE` 策略。它加载 `trading/app_executor.py`，负责账户查询与交易执行。

策略配置在 `configs/config.json`，文件执行配置在 `data/live_file/config.json`。两端均须保持运行；`data/live_file/planner_status.json` 与 `status.json` 的更新时间分别用于检查两端状态。

人工 `python run.py` 在取得 planner 单实例锁后清除本仓库制动标记；自动启动使用 `python run.py --preserve-halt`，遇到制动标记拒绝启动。首次部署仍需先准备文件执行配置与注册策略入口。

## 开发指引

### 单回测

```bash
uv run python testback/run_backtest.py \
  --start-date 20240101 --end-date 20241231 \
  --individual-config configs/config.json
```

### GA 参数搜索

```bash
uv run python testback/run_ga.py --mode ga
```

### 添加新因子

在 `factor_db/factors/` 下创建新 `.py` 文件，定义类包含 `hist_days` 属性 + `calc_batch(self, panel)` 方法，`core/factors/registry.py` 自动发现注册。策略/GA 搜索模板统一维护在 `configs/strategy.yaml`，实盘最终配置统一使用 `configs/config.json`。

## 策略分支切换（2026-11-09）

完整静态版本：`codex/live-static-20261109`；完整模型版本：`codex/live-model-topk-20261109`。两个分支各自只有一个生效的生产策略入口，不需要恢复 stash 或手工拼文件。

```powershell
# 在 D:/coding/WBR-live 中执行，选择其中一个分支
git switch codex/live-static-20261109
# 或
git switch codex/live-model-topk-20261109
uv sync --locked
```

切换前先保存自己的源码修改；不要用 `--force`、`reset --hard` 或 `git clean` 清理运行目录。行情、账户、目标、journal 和本地 SQLite 登记库不随分支切换。模型所需研究快照仍按该分支的数据契约校验；静态分支恢复的是原来的静态数据接口，不能把两版数据格式视为可互换。

Git 只切磁盘代码，不切运行中的进程。真实运行前应先受控停止本地 planner 和客户端执行策略（保留已登录券商客户端），切分支并同步环境，再按该版本重载 `WBR_EXEC_BRIDGE` 和本地 planner。两端必须使用同一版本；不得在执行中的代码目录热切换。恢复点不代表新的实盘启动授权。

同一天已冻结的目标或已开始的 journal 必须继续恢复原计划；不得删除、改写它们来强制换策略。策略对比可分别输出离线报告；真实账户换策略应安排在既有执行完成后的交易日边界。

静态版本恢复自 `1b3c454` 的跟踪文件和 `1b3c454^3` 的配套源码：四因子权重0.4/0.9/0.1/0.6，buy_n20/sell_m25/prefilter_n300，现金预留15%，原滑点0.1%。模型版本为冻结模型TopK40/40/3/1，滑点0.25%。费用口径不同，对比收益时必须注明。静态恢复不包含此前仅用于验证模型的未接入模型包。
