# WBR 量化交易系统

基于 QMT (`xtquant`) 的 Windows-only Python 量化交易项目，实盘 + GA 参数搜索 + 因子 benchmark。

## 项目架构

```
数据预下载(parquet) → runtime构建(np.savez) → np.load(秒级) → numpy向量化因子(毫秒) → 纯numpy回测(秒级)
```

回测与实盘路径严格分离。回测路径零中间对象——因子 `calc_batch(panel)` 一次算完全量，`_backtest_direct` 纯 numpy 循环执行交易。详见 CLAUDE.md。

## 环境要求

- Windows + Python 3.12 + [QMT 客户端](https://download.gjzq.com.cn/gjty/organ/gjzqqmt.rar)
- uv 包管理器

### 安装

```powershell
# 安装 uv
irm https://astral.sh/uv/install.ps1 | iex

# 安装依赖
uv sync

# TA-Lib（如安装失败用预编译whl）
uv pip install TA-Lib
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
