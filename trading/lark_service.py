"""独立飞书交互服务。

交易主进程不再启动飞书 WebSocket 线程；需要菜单查询/紧急制动时单独运行本入口。
"""
from configs import TRADE_ACCOUNT
from trading.lark.receiver import create_lark_handler
from trading.logger import trading_logger
from trading.trader import Trader


def main():
    trading_logger.info("启动飞书交互服务")
    trader = Trader(TRADE_ACCOUNT)
    create_lark_handler(trader)


if __name__ == "__main__":
    main()
