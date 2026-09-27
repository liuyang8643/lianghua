# WBR 量化策略研究与回测

GA 与 PPO 输出 `DayConfig`，回测和实盘共用 `env` 的选股、合法性、调仓、成交与账户结算。回测只读本地数据。

## 克隆后直接 review / debug

仓库包含完整当前源码和 **1990-12-19～2026-08-28、5544 只股票**的完整 runtime，包括行情、上市/退市状态、财报面板和原始财务状态。股票轴没有按今天的存续股票筛选。数据截至 2026-08-28，不代表实时数据。

完整快照约 **3.45 GiB**，以 48 MiB 分块随 Git 一起下载，不使用 Git LFS，不需要行情账号或额外下载市场数据。首次安装 Python 依赖仍需联网。建议浅克隆当前分支：

```sh
git clone --depth 1 --branch refactor/ai-rl-architecture https://github.com/liuyang8643/lianghua.git
cd lianghua
```

### 1. 安装轻量回测环境

Python **3.12**；支持 Windows / Linux。固定配置回测无需 QMT、PyTorch 或 CUDA。

```sh
python -m venv .venv-review
```

激活环境：Windows PowerShell 执行 `.venv-review/Scripts/Activate.ps1`；Linux / macOS 执行 `source .venv-review/bin/activate`。后续均使用此环境的 Python。

```sh
python -m pip install -r requirements-review.txt
```

若已经安装 uv，可用 `uv venv .venv-review --python 3.12` 和 `uv pip install --python .venv-review -r requirements-review.txt` 替代。不要为轻量回测运行 `uv sync`，它安装的是完整训练/数据更新环境。

### 2. 离线还原完整数据

```sh
python offline_data/portable_snapshot.py restore
```

还原到 `data/runtime/runtime_1990-12-19_2026-08-28_rawstate.npz`，同时还原其原始 `.manifest.json`。逐块 SHA-256 和最终文件 SHA-256 都会校验。再次执行只验证现有文件；遇到不同内容会报错，不覆盖本地数据。

分块、Git 对象及还原文件合计需要约 **11 GiB** 磁盘空间，另留依赖、计算内存和输出空间。原始快照保留全部历史，因子在本机重新计算；上市以来累计因子不能用简单日期截断替代历史。

完整因子重算建议 **32 GB 内存**：本机最近一年实测约 53～62 秒，进程树内存峰值约 22 GiB；首次运行还包含 Numba 编译，其他机器耗时可能不同。内存小于此规模时，缩短回测日期也不会免除上市以来因子的历史计算。

### 3. 回测最近一个完整可用年度

```sh
python -m testback.run_backtest --start-date 20250829 --end-date 20260828 --output-dir results/review-year
```

读取 `configs/config.json`，初始资金 100 万，固定 20 只目标持仓，单边滑点 0.0025。该静态配置是调试基准，不是 GA 冠军或已训练 PPO。

输出 `results/review-year/record.json`、日志及 HTML 报告，包含净值、收益、交易明细与个股图表。仅需机器可读结果时加 `--no-charts`。报告图表的前端库使用 CDN；回测计算和数据读取完全离线。

缩短调试日期，例如 `--start-date 20260803 --end-date 20260828`；也可在完整快照范围内选择其它历史时期。长历史因子仍需完整预计算，因此缩短回放主要减少账户步数。`--end-date` 是最后结算开盘日，不额外借用下一日数据。早期年份可能缺少部分财务因子。

调试入口依次为 `testback/run_backtest.py` → `testback/backtest.py` → `env/backtest.py`；因子计算见 `factor/compute.py`，权重和换股范围见 `env/action_schema.py`。修改配置直接用 `--individual-config your-config.json`。

## 目录与职责

| 目录 | 职责 |
|---|---|
| `offline_data/`、`data/*.py` | 不可变快照、PIT 财报、数据下载更新及 runtime 构建 |
| `factor/`、`factor_db/` | 生产因子、仍被引用的旧因子和动态研究候选 |
| `env/` | 唯一选股、合法性、调仓、成交、账户及 Reward 实现 |
| `ai/ga/`、`ai/rl/` | 静态 GA / 动态 PPO 的优化、评估与模型加载 |
| `trade/` | 券商适配、实盘执行、journal 与 replay |
| `testback/` | 固定配置回测及报告 |
| `configs/`、`tests/` | 声明式配置与契约/因果/一致性测试 |
| `snapshots/runtime/` | 本次分发的完整历史快照、分块清单和来源证明 |

本地 `artifacts/`、`results/`、日志、缓存、模型、历史实验源码副本不进入 Git；动态因子候选及仍被引用的代码保留。`AGENTS.md`、`CLAUDE.md`、`PLAN.md` 保留架构及研究历史约束。

本次交付的核对结果见 [分发验证记录](docs/review-delivery.md)。

## 完整训练与实盘环境

完整环境使用 `uv sync`，Windows QMT 用于实盘；PPO 的训练和模型推理要求 CUDA。当前生产词表是 **12 因子、20 持仓、换股范围 [0.05, 0.2]**。默认训练参数以 `ai/rl/train.py` 为准，运行 `python -m ai.rl.train --help` 查看，避免文档重复维护实验默认值。

```sh
python -m ai.ga.train --mode ga --runtime data/runtime/runtime_1990-12-19_2026-08-28_rawstate.npz
python -m ai.rl.train --runtime data/runtime/runtime_1990-12-19_2026-08-28_rawstate.npz --output artifacts/rl/new-run
```

这两个命令会开始正式搜索/训练，不是快速回测命令。固定配置回测无需下载训练模型。

`configs/training_reports.json` 是本地历史实验索引，模型和报告不随仓库分发；只读报告服务需要对应运行产物。实盘使用已校验 bundle 和 `run.ps1`，账号和凭据从环境变量注入。

## 更新分发快照

维护者从已有完整快照生成新的分块目录：

```sh
python offline_data/portable_snapshot.py pack data/runtime/runtime_1990-12-19_2026-08-28_rawstate.npz snapshots/new-runtime --sidecar data/runtime/runtime_1990-12-19_2026-08-28_rawstate.manifest.json
```

不修改浮点精度、股票轴、日期或 NPZ 字节。因子/数据因果规则没有新增兼容实现。大数据更新会增加 Git 历史体积；日常数据和实验输出继续只保留在本地。
