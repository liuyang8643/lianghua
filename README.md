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
