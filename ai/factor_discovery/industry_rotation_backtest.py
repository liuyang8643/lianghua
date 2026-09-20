"""Offline CLI for video industry-index theoretical-account reproduction."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
from env.industry_rotation import run_industry_rotation


def align_industry_calendar(opens, closes, calendar, start, end):
    """Validate the signal interval plus ten real-session warmup dates."""
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing or calendar.hasnans:
        raise ValueError("calendar dates must be unique, finite and increasing")
    if len(calendar) == 0 or pd.Timestamp(end) > calendar[-1]:
        raise ValueError("calendar ends before requested end; use a calendar covering the requested boundary")
    first = int(calendar.searchsorted(pd.Timestamp(start)))
    stop = int(calendar.searchsorted(pd.Timestamp(end), side="right"))
    if first < 10 or stop <= first:
        raise ValueError("calendar cannot supply requested period and ten warmup sessions")
    expected = calendar[first - 10:stop]
    missing = expected.difference(opens.index)
    if len(missing):
        raise ValueError("industry panel entirely missing required trading dates: " + ", ".join(str(d.date()) for d in missing))
    return opens.reindex(expected), closes.reindex(expected), {
        "verified_start": str(expected[0].date()), "verified_end": str(expected[-1].date()),
        "verified_date_count": len(expected), "warmup_session_count": 10,
        "check": "all required calendar rows exist; individual industry validity remains explicit, no price fill"}


def write_report(output, summary, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for name, daily in curves.items():
        axes[0].plot(daily.date, daily.nav, label=name)
        axes[1].plot(daily.date, daily.nav / daily.nav.cummax() - 1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Theoretical NAV (log scale)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle("Industry momentum: separate theoretical index accounts")
    fig.tight_layout()
    fig.savefig(output / "nav.png", dpi=150)
    plt.close(fig)
    lines = ["# 行业轮动本地复现", "", "本报告为申万行业指数理论账户；指数不可直接交易，不代表 ETF、个股组合或 PPO 可实现收益。", "",
             f"请求信号起点 {summary['signal_start']}，请求结束 {summary['end']}。本地行情加载范围 {summary['source_first_date']} 至 {summary['source_last_date']}，共 {summary['source_union_date_count']} 个日期、{summary['index_count']} 个行业。", "",
             "|策略|年化收益|最大回撤|终值（初值1）|末日现金|末日未平仓市值|",
             "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in summary["results"].items():
        lines.append(f"|{name}|{metrics['annualized_return']:.2%}|{metrics['max_drawdown']:.2%}|{metrics['final_nav']:.6f}|{metrics['final_cash']:.6f}|{metrics['terminal_marked_holdings']:.6f}|")
    first = next(iter(summary["results"].values()))
    if summary["calendar_audit"] is not None:
        audit = summary["calendar_audit"]
        lines += ["", f"本地交易日历校验范围 {audit['verified_start']} 至 {audit['verified_end']}（含10个信号预热交易日），共{audit['verified_date_count']}日；全行业同时缺日将拒绝回测，不补价格。日历身份见summary.json。"]
    lines += ["", f"实际首次信号 {first['first_signal_date']}，首次入场 {first['first_entry_date']}，末次估值 {first['last_valuation_date']}；{first['effective_trading_day_count']} 个日收益区间。", "",
              "固定规则：S 收盘后使用 close[S]/close[S−10]−1 排名，要求连续11个收盘价有效；并列按行业代码排序。原始策略选第1名，S+1开盘入场，后续开盘换仓，同一行业续持。改进策略选前3名入场等权，两条初始各0.5资金的账户错开1交易日；每条在S+1开盘入场、S+2收盘按预定计划卖出，下一开盘前现金不计息。两条账户独立复利，不跨账户重新均分资金、不净额抵消交易。", "",
              "0bp 为无成本主复现；10bp 为每笔买入及卖出名义金额各收0.1%的综合成本敏感性，并非真实券商或ETF费率证明。买入预算包含成本；同标续持不重复收费。收盘退出是在入场时预定的，退出收盘价不用于入场决策。", "",
              "逐日按收盘盯市，包含实际持仓隔夜涨跌；起点收盘NAV为1。年化使用252交易日，最大回撤使用包含起始现金的逐日收盘NAV。期末尚未到退出日的持仓按末日收盘估值，不强制平仓且不预扣未来卖出成本。完整逐笔交易、信号、双链净值见同目录parquet。", "",
              "本研究未修改生产PPO的因子、动作或成交schema。生产股票账户采用开盘决策与开盘结算，不能把本研究的预定收盘退出直接当成同一执行环境。数据哈希证明可复现，不证明历史行业分类和回溯编制没有修订；行业数据缺失、目录版本与交易日完整性须同时查看数据审计。", "", "![净值与回撤](nav.png)", "", "## 分年收益", "",
              "年份按实际日收益所属日期归类，跨年首日包含前一估值日到该日收益；首尾年份可能不足全年。", "",
              "|年份|原始0bp|原始10bp|改进0bp|改进10bp|", "|---|---:|---:|---:|---:|"]
    years = sorted(set().union(*(values.keys() for values in summary["year_returns"].values())))
    for year in years:
        lines.append("|" + year + "|" + "|".join(f"{summary['year_returns'][name][year]['return']:.2%}" for name in curves) + "|")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/industry_rotation_20260914/data/indices.parquet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/industry_rotation_20260914/backtest"))
    parser.add_argument("--start", default="2012-01-04")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--calendar", type=Path, help="Local runtime NPZ; reads trade_dates only")
    args = parser.parse_args()
    frame = pd.read_parquet(args.input, filters=[("date", "<=", pd.Timestamp(args.end))])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["code"] = frame["code"].astype(str)
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("duplicate index/date rows")
    opens = frame.pivot(index="date", columns="code", values="open").sort_index().sort_index(axis=1)
    closes = frame.pivot(index="date", columns="code", values="close").reindex(index=opens.index, columns=opens.columns)
    calendar_audit = None
    if args.calendar is not None:
        with np.load(args.calendar, allow_pickle=False) as runtime:
            calendar = runtime["trade_dates"]
        opens, closes, calendar_audit = align_industry_calendar(opens, closes, calendar, args.start, args.end)
        with args.calendar.open("rb") as stream:
            calendar_audit["runtime_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        calendar_audit["runtime_path"] = str(args.calendar)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   signal_start=args.start, end=args.end, theoretical_index_account=True, broker_exact=False,
                   rules="10 completed close returns; next-open entry; top1 open switch with same-index continuation; top3 two separate half-capital chains, next-day scheduled close exit; entry equal weight; no chain netting; terminal close marking without liquidation",
                   annualization="252 trading days; daily close NAV including initial cash signal close",
                   source_first_date=str(opens.index[0].date()), source_last_date=str(opens.index[-1].date()),
                   source_union_date_count=len(opens), index_count=len(opens.columns), calendar_audit=calendar_audit,
                   source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  (Path(__file__), Path("env/industry_rotation.py"), Path("env/metrics.py"))}, results={}, year_returns={})
    curves = {}
    for strategy in ("original_top1", "staggered_top3"):
        for cost in (0.0, 0.001):
            name = f"{strategy}_{round(cost * 10000)}bp"
            result = run_industry_rotation(opens.index.to_numpy(), opens.columns.to_numpy(), opens.to_numpy(), closes.to_numpy(),
                                           signal_start=args.start, end=args.end, strategy=strategy, cost_rate=cost)
            daily = pd.DataFrame(dict(date=result.dates, nav=result.nav))
            for chain in range(result.chain_nav.shape[1]):
                daily[f"chain_{chain}"] = result.chain_nav[:, chain]
            daily.to_parquet(args.output / f"{name}_nav.parquet", index=False)
            curves[name] = daily
            pd.DataFrame(result.trades).to_parquet(args.output / f"{name}_trades.parquet", index=False)
            pd.DataFrame(result.signals).to_parquet(args.output / f"{name}_signals.parquet", index=False)
            summary["results"][name] = dict(result.metrics,
                                           first_signal_date=str(result.dates[0]),
                                           first_entry_date=result.signals[0]["entry_date"],
                                           last_valuation_date=str(result.dates[-1]),
                                           effective_trading_day_count=len(result.nav) - 1,
                                           trade_count=len(result.trades))
            returns = pd.Series(result.nav[1:] / result.nav[:-1], index=pd.DatetimeIndex(result.dates[1:]))
            summary["year_returns"][name] = {
                str(year): dict(return_=float(values.prod() - 1), days=len(values))
                for year, values in returns.groupby(returns.index.year)}
            for row in summary["year_returns"][name].values():
                row["return"] = row.pop("return_")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(args.output, summary, curves)
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
