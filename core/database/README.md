# core.database Notes

## Overview

`core.database` 里的行情读取现在分成两层：

- `history.py`
  只负责读取和按需补齐 QMT 的 raw K 线。
- `data.py`
  只负责把 raw K 线过滤成可以安全暴露给上层的 reliable K 线。

这样做的目的有两个：

- 上层拿到的数据尽量稳定可靠，不混入停牌占位 bar。
- 只有在本地 raw 数据确实不够时才调用 `download_history_data2(...)`，避免慢路径反复触发。

## Raw vs Reliable

QMT 的日线停牌表现并不稳定，真实测到过三种情况：

- 停牌日完全缺 bar。
- 停牌日返回 `suspendFlag == 1`。
- 停牌日返回 `suspendFlag == 0`，但 OHLC 全 0 的占位 bar。

因此：

- raw bar
  指 QMT 原样返回的 bar。
- reliable bar
  指过滤掉以下情况之后可返回给上层的 bar：
  - `suspendFlag == 1`
  - `open/high/low/close` 全 0 的占位 bar

注意：

- `suspendFlag == -1` 表示当日起复牌，不应按停牌过滤。

## Download Policy

`history.get_history_data_after_download(...)` 的策略是：

1. 先读取本地 raw 数据。
2. 如果 reliable bar 数不够，优先扩大本地读取窗口。
3. 只有本地扩大后仍然不够，才触发 `download_history_data2(...)`。
4. 下载后如果数据签名没有变化，就停止继续尝试，避免死循环。

额外说明：

- `download_history_data2(..., incrementally=...)` 当前在本机 QMT 上不作为有效控制参数使用，代码里不再显式传递它。
- `for _ in range(6)` 只是防死循环的保险上限，不承载业务语义。

## API Semantics

### `get_market_data(...)`

严格接口。

- 目标时点必须存在“最新 reliable bar”。
- 若目标日当天停牌，接口应报错。
- 若目标日晚于停牌区间且已复牌，接口会跨过停牌段，返回前 `N` 条 reliable bar。

### `get_market_data_batch(...)`

批量接口，但语义需要和 `get_market_data(...)` 对齐。

- 单只股票成功/失败判定与 `get_market_data(...)` 一致。
- 失败时对应股票返回 `None`。
- 新股若上市以来可用交易日不足 `count`，按上市以来上限放行。

## Validated Cases

### Case 1: `000001.SZ`

文件路径：

- `os.path.dirname(xtdata.get_data_dir())/datadir/SZ/86400/000001.DAT`

验证方式：

- 在本地日线文件缺失的情况下调用  
  `get_market_data('000001.SZ', 5, datetime(2025, 5, 28, 15, 0, 0), '1d', dividend_type='none')`

结果：

- 第 1 次调用触发 1 次下载。
- 第 2 次同进程调用不再触发下载。
- 新进程再次调用也不再触发下载。

### Case 2: `603019.SH`

文件路径：

- `os.path.dirname(xtdata.get_data_dir())/datadir/SH/86400/603019.DAT`

验证方式：

- 在本地日线文件缺失的情况下调用  
  `get_market_data('603019.SH', 20, datetime(2025, 6, 20, 15, 0, 0), '1d', dividend_type='none')`

结果：

- 第 1 次调用触发 1 次下载，下载区间为 `20250522 -> 20250620`。
- 第 2 次同进程调用不再触发下载。
- 返回结果会自动跨过停牌区间，最终拿到 20 条 reliable bar。

### Case 3: `603019.SH @ 2025-05-28`

结果：

- `get_market_data(...)` 应失败。
- 原因不是补数失败，而是目标日当天停牌，最后一根 reliable bar 仍停留在 `2025-05-23`。

### Case 4: `603019.SH @ 2025-06-20`

结果：

- `get_market_data(...)` 成功。
- `get_market_data_batch(...)` 也成功，且返回结果与单只接口对齐。

## Tests

关键测试位于：

- [test_history.py](/C:/Users/Sleaf/PycharmProjects/WBR/core/database/test_history.py)
- [test_data_cache.py](/C:/Users/Sleaf/PycharmProjects/WBR/core/database/test_data_cache.py)

目前已覆盖：

- 新股短历史放行
- 全零占位 bar 过滤
- `suspendFlag == -1` 复牌 bar 保留
- 本地扩大读取窗口优先于二次下载
- batch/single 接口语义对齐
