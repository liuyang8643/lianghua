"""Fixed-config backtest CLI using the canonical env account session."""

from __future__ import annotations

import argparse
from env.observation import DEFAULT_LOOKBACK
from testback.backtest import run_single_mode
from utils.logger import configure_logger


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description="WBR canonical env backtest")
    parser.add_argument(
        "--individual-config",
        default="configs/config.json",
        help="static DayConfig wrapper JSON",
    )
    parser.add_argument("--runtime", dest="runtime_path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-date", default="20240101")
    parser.add_argument("--end-date", default="20241231")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="write record.json only",
    )
    args = parser.parse_args()

    configure_logger("INFO")
    return run_single_mode(
        args,
        {
            "desc": "canonical env fixed backtest",
            "log_level": "INFO",
            "save_charts": not args.no_charts,
        },
    )


if __name__ == "__main__":
    main()
