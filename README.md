# WBR 量化交易系统

基于 QMT (`xtquant`) 的 Windows-only Python 量化交易项目，包含实盘交易、GA 参数搜索和因子 benchmark 工具。

项目的详细工程说明集中在 [AGENTS.md](./AGENTS.md)。

## 环境要求

- Windows
- Python 3.12
- Windows 操作系统 (QMT 平台依赖) + [QMT 客户端](https://download.gjzq.com.cn/gjty/organ/gjzqqmt.rar)已安装
- uv 包管理器

### 安装 uv

方式一：使用官方安装脚本（推荐）

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

方式二：使用 pip 安装（作为 Python 包）
如果你更习惯使用 pip，也可以这样安装它（但安装后，uv 内部也会使用这个已存在的 Python 环境）：

```powershell
pip install uv
```

### 初始化并安装依赖

使用 uv 自动创建虚拟环境并安装所有依赖：

```powershell
# uv 会自动读取 pyproject.toml 并安装依赖
uv sync
# 或者手动激活虚拟环境
.\.venv\Scripts\Activate.ps1
```

#### TA-Lib 安装说明

Windows 下安装 TA-Lib 可能需要额外步骤：

```powershell
# 方法1：使用 uv 直接安装（推荐）
uv pip install TA-Lib

# 方法2：如果方法1失败，先下载预编译文件
# 从 https://github.com/cgohlke/talib-build/releases 下载对应版本
# 然后安装：
uv pip install TA_Lib-0.6.8-cp312-cp312-win_amd64.whl
```

## 实盘运行

> 非首次运行可直接跳到第 5 步

1. 复制 [env.template.py](configs/env.template.py) 为 `env.py`，并修改为实际的配置项
2. 复制 [_copy.ps1](_copy.ps1) 到 gjqmt\\bin.x64 目录下。
3. 执行 _copy.ps1 脚本以准备截取登录信息
4. 登录QMT，勾选极简模式
5. 检查 _linkMini 是否成功生成
6. 运行 `.\run.ps1` 启动实盘交易

## 开发指引

### 因子相关性 benchmark

```powershell
python -m core.factors.benchmark.benchmark
```

当前分支下，`benchmark.py` 默认会：

- 使用 `WMACross` 作为 benchmark 因子
- 分析 `2024-05-01` 到 `2025-05-01`
- 计算 `T+1/3/5/10/20/30/60`
- 总是写出 `.pkl` 报告到 `reports/`
- 在控制台打印汇总统计
- 仅在设置 `BENCHMARK_GENERATE_HTML=1/true/yes` 时生成 HTML 报告

可通过环境变量覆盖部分行为：

```powershell
$env:BENCHMARK_START_DATE = '2024-01-01'
$env:BENCHMARK_END_DATE = '2024-12-31'
$env:BENCHMARK_SAMPLE_STEP = '5'
$env:BENCHMARK_SAMPLE_SIZE = '1000'
$env:BENCHMARK_GENERATE_HTML = '1'
python -m core.factors.benchmark.benchmark
```

### 因子可视化 Web 页面

```powershell
python -m core.factors.benchmark.web_chart
python -m core.factors.benchmark.web_chart --port 9090
python -m core.factors.benchmark.web_chart --code 600000.SH
```

当前 `web_chart.py` 会展示 K 线、成交量、均线和因子得分；默认随机选取一只可交易股票，也可以用 `--code` 指定股票直接生成 HTML 并打开浏览器。当前实现只展示 `WMACross`，并把数据范围限制在 `2021-01-01` 到 `2022-12-31`。

更多 benchmark 说明见 [core/factors/benchmark/README.md](./core/factors/benchmark/README.md)。

## 项目架构

详见：https://ai.feishu.cn/wiki/NaSewvuQ8iRpQfkpzWMckZRSnmh

## 许可证

本项目采用私有许可，仅供内部使用。
