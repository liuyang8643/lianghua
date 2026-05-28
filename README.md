# WBR 量化交易系统


## 环境要求

- Windows + Python 3.12 + [QMT 客户端]
- uv 包管理器

### 安装

```powershell
# 安装 uv
irm https://astral.sh/uv/install.ps1 | iex

# 安装依赖
uv sync
```

## 实盘运行

1. 配置 `configs/env.py`（QMT路径、账号等）
2. 登录 QMT，勾选极简模式
3. 运行 `.\run.ps1` 启动实盘交易

## 开发指引

### 单回测

```bash
uv run python testback/run_backtest.py \
  --start-date 20240101 --end-date 20241231 \
  --individual-config configs/single_tmc_pure.json
```

### GA 参数搜索

```bash
uv run python testback/run_ga.py --mode ga
```

### 添加新因子

在 `core/factors/` 下创建新 `.py` 文件，定义类包含 `hist_days` 属性 + `calc_batch(self, panel)` 方法，`registry.py` 自动发现注册。
