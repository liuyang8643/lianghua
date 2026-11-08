"""The sole production CLI: file planner, never a local broker executor."""
from trading.planner_watchdog import cli

if __name__ == '__main__':
    cli()
