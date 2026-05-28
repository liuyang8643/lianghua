import logging as _stdlib_logging
import threading
from datetime import timedelta

from trading.trader import Trader, get_position
from trading.logger import trading_logger


def _silence_lark_unknown_event_errors():
  """屏蔽 Lark SDK 对未注册事件类型的 'processor not found' ERROR。

  飞书后台开通的某些事件订阅（例如 im.chat.access_event.bot_p2p_chat_entered_v1
  P2P 进入聊天）我们并未在 EventDispatcherHandler 上注册处理器，
  SDK 收到时只会丢弃事件，无业务影响 — 但每次都打 ERROR 污染日志。
  """
  class _Filter(_stdlib_logging.Filter):
    def filter(self, record):
      return 'processor not found' not in record.getMessage()
  for name in ('lark_oapi', 'lark'):
    _stdlib_logging.getLogger(name).addFilter(_Filter())


_silence_lark_unknown_event_errors()

def create_lark_handler(trader: Trader):
  """ https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/server-side-sdk/python--sdk/handle-events """
  import lark_oapi as lark
  from configs import LARK_APP_SECRET, LARK_APP_ID
  from lark_oapi.event.callback.model.p2_card_action_trigger import P2CardActionTrigger, P2CardActionTriggerResponse
  from .sender import LarkMsgLevel, lark_sender

  def handle_action_query_positions():
    """ 查询持仓信息 """
    from data.db import get_stock_detail
    from utils.stock.format import get_stock_desc
    trading_logger.debug(f'开始查询持仓信息...')
    position_info = [
      f"{get_stock_desc(p['detail'])}\t\t {p['info'].market_value:.2f}元\t {p['info'].can_use_volume}/{p['info'].volume}股"
      for p in map(
        lambda x: {'detail': get_stock_detail(x.stock_code), 'info': x},
        get_position(trader, False)
      )
    ]
    trading_logger.debug(f'查询到 {len(position_info)} 条持仓信息')
    lark_sender.send_notification_card(
      level=LarkMsgLevel.Info,
      title="持仓信息",
      content="\n".join(position_info),
    )

  def handle_action_query_logs():
    """ 查询日志 """
    log_limit = 1000
    logs = []
    trading_logger.debug(f'开始读取日志，限制 {log_limit} 条...')
    if trading_logger.log_file_path:
      with open(trading_logger.log_file_path, 'r', encoding='utf-8') as file:
        for _ in range(log_limit):
          line = file.readline()
          if not line:
            break
          logs.append(line.strip())
    else:
      logs.append('日志文件不存在，请检查配置')
    trading_logger.debug(f'读取到 {len(logs)} 条日志')
    lark_sender.send_notification_card(
      level=LarkMsgLevel.Info,
      title=f"最近{log_limit}条日志",
      content="```\n" + '\n'.join(logs) + "\n```",
    )

  def handle_action_query_worth_buy():
    lark_sender.send_notification_card(
      level=LarkMsgLevel.Warning,
      title="维护中",
      content="该功能正在维护中，快去写代码实现一下！",
    )

  def handle_action_kill():
    """ 结束程序 """
    from utils.sys import terminate_process_tree
    trading_logger.debug(f"准备结束程序")
    lark_sender.send_notification_card(
      level=LarkMsgLevel.Info,
      title="紧急制动",
      content="程序已手动结束",
    )
    terminate_process_tree()

  handlers = {
    "query_positions": handle_action_query_positions,
    "query_logs": handle_action_query_logs,
    "query_worth_buy": handle_action_query_worth_buy,
    "process_kill": handle_action_kill
  }

  def handle_user_menu_trigger(data: lark.application.v6.p2_application_bot_menu_v6):
    from datetime import datetime

    event_key = data.event.event_key
    event_datetime = datetime.fromtimestamp(data.event.timestamp)
    trading_logger.info(f'接收到事件【{event_key}】@{event_datetime}')

    # 丢弃早于当前时间 1min 的事件
    if event_datetime < datetime.now() - timedelta(minutes=1):
      trading_logger.debug(f'丢弃过期事件【{event_key}】@{event_datetime}')
      return

    # 处理事件
    if event_key in handlers:
      threading.Thread(target=handlers[event_key], daemon=True).start()
    else:
      trading_logger.error(f"未知事件【{event_key}】")

    # 处理完成
    trading_logger.debug(f'已处理事件【{event_key}】@{event_datetime}')

  # 监听「卡片回传交互 card.action.trigger」事件 P2CardActionTrigger。
  def handle_card_action_trigger(data: P2CardActionTrigger) -> P2CardActionTriggerResponse:
    print(lark.JSON.marshal(data))
    # TODO 暂未调通
    resp = {
      "toast": {
        "type": "info",
        "content": "卡片回传成功 from python sdk"
      }
    }
    return P2CardActionTriggerResponse(resp)

  event_handler = lark.EventDispatcherHandler.builder("", "") \
    .register_p2_application_bot_menu_v6(handle_user_menu_trigger) \
    .register_p2_card_action_trigger(handle_card_action_trigger) \
    .build()
  cli = lark.ws.Client(LARK_APP_ID, LARK_APP_SECRET, event_handler=event_handler)
  try:
    cli.start()
  except Exception:
    trading_logger.warning('飞书 WebSocket 连接失败（app_id/app_secret 未配置）')

if __name__ == "__main__":
  from configs import TRADE_ACCOUNT

  td = Trader(TRADE_ACCOUNT)
  create_lark_handler(td)
