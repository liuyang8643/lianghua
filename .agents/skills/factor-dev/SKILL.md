---
name: factor-dev
description: A股量化因子迭代研发——agent-team 编写因子 → GA 500代 → 数据分析 → 淘汰无效 → 联网搜索迭代 → 循环至夏普>2.5
---

# 因子迭代研发

## 概述

在 `core4` GA profile 框架下做因子迭代研发。agent-team 并行编写纯 numpy 因子 → 单因子 smoke test → 注册到 profile → GA 搜索 500 代 → 统计分析权重/夏普 → 淘汰无效因子 → 联网搜索发掘新思路 → 循环，直到训练/验证/测试三段夏普率均 > 2.5。

## 终止条件

GA 跑出同时满足 **训练集、验证集、测试集夏普率均 > 2.5** 的个体，任务完成。

## 迭代循环

```
agent-team 编写因子 → smoke test → 杀旧GA → 注册因子 → 重启GA 500代
→ 数据分析（权重均值/方差/夏普率）→ 淘汰无效因子
→ 联网搜索 + 深度思考 → 发掘新因子 → 重复
```

### 1. agent-team 编写新因子

用 agent-team 并行分发任务。每个因子一个文件，放在 `core/factors/` 下：

```python
import numpy as np

class FactorName:
  """一句话描述因子逻辑"""
  hist_days = 0  # 按需调整：需N日滚动窗口就设为N

  def calc_batch(self, panel: dict) -> np.ndarray:
    open_prices = panel["open"]
    st_mask = panel["st_mask"]
    # ... 纯 numpy 向量化计算 ...
    valid = ~np.isnan(open_prices) & (open_prices >= 2.0) & ~st_mask
    return np.where(valid, result, np.nan)
```

**红线**：
- `calc_batch` 纯 numpy 向量化，禁止逐股票遍历
- 只能使用 panel 中的字段（open/high/low/close/volume/amount/st_mask/eps/roe/profit_yoy/revenue_yoy/operating_cf_ps/gross_margin/total_share/issue_price）
- 复用已有数据字段，不要新增数据源
- 禁止调 xtdata/mootdx/S3/CNINFO 等外部数据源

### 2. 单因子 Smoke Test

每个新因子写入后，先用 single 模式验证无报错：

```powershell
uv run python -u -m testback.single `
  --individual-config configs/single_<factor_name>.json `
  --start-date 20200101 --end-date 20201231 `
  > $env:TEMP\smoke_<factor_name>.log 2>&1
```

配置模板（复制 `configs/single_smallcap_g2a_config.json`，保留 `ga_profile: core4`，修改 `weights` 只保留新因子名）。

确认无报错后再进入注册步骤。

### 3. 杀旧 GA + 注册因子

```powershell
# 强制杀后台 Python
powershell Stop-Process -Name python -Force
Start-Sleep 3

# 二次确认进程数=0 且空闲内存>30GB（CLAUDE.md 安全红线）
$procs = (Get-Process python -ErrorAction SilentlyContinue).Count
if ($procs -gt 0) { Write-Warning "残留 $procs 个 Python 进程"; exit 1 }
$mem = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
if ($mem -lt 30) { Write-Warning "空闲内存不足: $mem GB"; exit 1 }
Write-Host "OK: 进程=0, 空闲内存=$([math]::Round($mem,1))GB"
```

**注册因子**：编辑 `configs/ga_profiles.yaml`，在 `_FACTOR_REGISTRY` 的 import 和列表中加入新因子类。

**更新 profile**：编辑 `configs/ga_profiles.yaml`，在 `core4` profile 的 `factor_classes` 中加入新因子名。

### 4. 重启 GA 500 代

先将 `ga_profiles.yaml` 的 `mode_configs.ga.generations` 临时改为 `500`（原值 10000）。GA 模式日期范围由 profile 的 `preload_start`/`preload_end` 决定（`--start-date`/`--end-date` 仅对 single 模式有效）。

```powershell
$logPath = Join-Path $env:TEMP ("ga-factor-dev-" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")
$outputDir = "results/ga_factor_dev_" + (Get-Date -Format 'yyyyMMdd_HHmmss')

uv run python -u -m testback.ga_run --profile core4 --output-dir $outputDir > $logPath 2>&1
```

### 5. 数据分析

读取 GA checkpoint，统计每个因子在所有历史个体中的表现：

```python
import pickle
from pathlib import Path
import numpy as np

ckpt = pickle.loads(Path('results/<latest>/checkpoint.pkl').read_bytes())
all_results = ckpt['all_results']  # list of {individual_config, sharpe, ...}

# 按因子统计权重均值、方差、平均夏普率
factor_stats = {}
for r in all_results:
    w = r['individual_config']['weights']
    s = r['sharpe']
    for fname, fw in w.items():
        if fname not in factor_stats:
            factor_stats[fname] = {'weights': [], 'sharpes': []}
        factor_stats[fname]['weights'].append(fw)
        factor_stats[fname]['sharpes'].append(s)

for fname, v in factor_stats.items():
    w_arr = np.array(v['weights'])
    s_arr = np.array(v['sharpes'])
    print(f"{fname}: 权重均值={w_arr.mean():.3f} 方差={w_arr.var():.3f} 平均夏普={s_arr.mean():.3f}")
```

**淘汰规则**：权重均值接近 0 且方差小、平均夏普率显著低于其他因子的 → 从 profile 的 `factor_classes` 中移除，同时从 `_FACTOR_REGISTRY` 的列表中移除。

### 6. 联网搜索 + 深度思考迭代

根据淘汰后剩余因子的收益特征（哪些风格有效、哪些失效），联网搜索新的因子思路，深度思考后提出新因子假设，回到步骤 1。

### 7. 终止判断

每次 GA 结束后检查 checkpoint 中最佳个体的三段夏普率：训练集、验证集、测试集是否同时 > 2.5。满足则停止迭代，将 `ga_profiles.yaml` 中 `mode_configs.ga.generations` 恢复为 `10000`。

## 注意事项

- 单次 GA 500 代约需 2-4 小时，耐心等待
- checkpoint 自动保存，中断后可用 `--resume` 恢复
- 因子总数控制在 4-8 个，过多会稀释搜索效率
- 恢复 `generations: 10000` 后再提交代码
