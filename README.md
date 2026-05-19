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

> 非首次运行可直接跳到第 5 步

1. 复制 `configs/env.template.py` 为 `env.py`，修改配置项
2. 复制 `_copy.ps1` 到 gjqmt\bin.x64 目录下执行
3. 登录 QMT，勾选极简模式
4. 检查 `_linkMini` 是否成功生成
5. 运行 `.\run.ps1` 启动实盘交易

## 开发指引

### 因子 benchmark

```powershell
python -m core.factors.benchmark.benchmark

# 可选环境变量
$env:BENCHMARK_START_DATE = '2024-01-01'
$env:BENCHMARK_GENERATE_HTML = '1'
```

详见 `core/factors/benchmark/README.md`。

### 因子可视化

```powershell
python -m core.factors.benchmark.web_chart
python -m core.factors.benchmark.web_chart --code 600000.SH --port 9090
```
