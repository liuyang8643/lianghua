import logging as _stdlib_logging
from datetime import timedelta
from pathlib import Path

import psutil

from trading.file_account import FileAccount as Trader, get_position
from trading.logger import trading_logger


_PID_START_TIME_TOLERANCE_SEC = 300
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_CONFIG_PATH = (_REPO_ROOT / 'configs' / 'config.json').resolve()
_WATCHDOG_HALT_PATH = (
  _REPO_ROOT
  / 'data' / 'live_trades' / 'trading_watchdog.halt'
)


def _arm_watchdog_halt(pid: int, halt_path: Path = _WATCHDOG_HALT_PATH):
  """先阻止 watchdog 重启，再由调用方终止已验证的主进程。"""
  halt_path.parent.mkdir(parents=True, exist_ok=True)
  halt_path.write_text(f'emergency_stop_pid={pid}\n', encoding='utf-8')


def _is_trading_main_command(process) -> bool:
  """仅接受本仓库 ``trading.main`` 的明确命令行。"""
  try:
    args = [str(arg).strip().strip('"') for arg in process.cmdline()]
  except (psutil.Error, OSError, AttributeError):
    return False

  for index, arg in enumerate(args[:-1]):
    if arg.casefold() == '-m' and args[index + 1].casefold() == 'trading.main':
      return True

  expected_script = (Path(__file__).resolve().parents[1] / 'main.py').resolve()
  try:
    process_cwd = Path(process.cwd())
  except (psutil.Error, OSError, AttributeError):
    process_cwd = None
  for arg in args:
    if not arg.replace('\\', '/').casefold().endswith('/trading/main.py'):
      continue
    candidate = Path(arg)
    if not candidate.is_absolute():
      if process_cwd is None:
        continue
      candidate = process_cwd / candidate
    try:
      if candidate.resolve() == expected_script:
        return True
    except OSError:
      continue
  return False


def _same_path(left: Path, right: Path) -> bool:
  return str(left.resolve()).casefold() == str(right.resolve()).casefold()


def _validate_repository_identity(process):
  """确认进程工作目录与显式配置都属于当前 WBR-live 仓库。"""
  try:
    process_cwd = Path(process.cwd()).resolve()
    args = [str(arg).strip().strip('"') for arg in process.cmdline()]
  except (psutil.Error, OSError, AttributeError) as exc:
    return f'无法读取进程仓库身份: {exc}'

  if not _same_path(process_cwd, _REPO_ROOT):
    return f'进程工作目录不是当前仓库: {process_cwd}'

  config_arg = None
  for index, arg in enumerate(args):
    if arg.casefold() == '--individual-config':
      if index + 1 >= len(args):
        return '--individual-config 缺少路径'
      config_arg = args[index + 1]
      break
    if arg.casefold().startswith('--individual-config='):
      config_arg = arg.split('=', 1)[1]
      break

  # 未显式传参时 main.py 默认读取当前仓库 configs/config.json。
  if config_arg is None:
    config_path = process_cwd / 'configs' / 'config.json'
  else:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
      config_path = process_cwd / config_path
  try:
    if not _same_path(config_path, _EXPECTED_CONFIG_PATH):
      return f'进程配置不是当前仓库实盘配置: {config_path.resolve()}'
  except OSError as exc:
    return f'无法解析进程配置路径: {exc}'
  return ''


def _resolve_trading_main_process(pid_path: Path, process_factory=psutil.Process):
  """从 PID 文件解析并验证实盘主进程，拒绝陈旧 PID 或 PID 复用。

  返回 ``(process, '')`` 或 ``(None, reason)``。命令行必须明确是
  ``trading.main``，工作目录与配置必须属于当前仓库，还要求进程启动时间与 PID
  文件写入时间接近；主进程会在获得实盘单实例锁后立即写 PID，因此这能识别旧
  PID 后被系统复用及同名其他仓库进程。
  """
  if not pid_path.exists():
    return None, f'未找到 {pid_path.name}'
  try:
    pid = int(pid_path.read_text(encoding='utf-8').strip())
  except (OSError, ValueError):
    return None, f'{pid_path.name} 内容无效'
  if pid <= 0:
    return None, f'{pid_path.name} 中 PID 非法: {pid}'

  try:
    process = process_factory(pid)
    if int(process.pid) != pid or not process.is_running():
      return None, f'PID={pid} 不存在或已退出'
    status = getattr(process, 'status', None)
    if callable(status) and status() == psutil.STATUS_ZOMBIE:
      return None, f'PID={pid} 已是僵尸进程'
    if not _is_trading_main_command(process):
      return None, f'PID={pid} 不是 trading.main，拒绝终止'
    repository_error = _validate_repository_identity(process)
    if repository_error:
      return None, f'PID={pid} {repository_error}，拒绝终止'

    written_at = pid_path.stat().st_mtime
    started_at = float(process.create_time())
  except (psutil.Error, OSError, ValueError) as exc:
    return None, f'PID={pid} 校验失败: {exc}'

  delta = abs(written_at - started_at)
  if delta > _PID_START_TIME_TOLERANCE_SEC:
    return None, (
      f'PID={pid} 启动时间与 PID 文件相差 {delta:.0f}s，疑似陈旧 PID，拒绝终止')
  return process, ''


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
    """校验 PID 身份后结束实盘主进程，禁止按陈旧 PID 误杀。"""
    from utils.sys import terminate_process_tree
    from trading.persistence import _TRADE_DIR
    if (_REPO_ROOT / 'data/live_file/identity.json').exists():
      from trading.file_control import request_file_stop
      try:
        pid = request_file_stop(_REPO_ROOT)
      except Exception as exc:
        trading_logger.warning(f'文件执行器制动未执行: {exc}')
        lark_sender.send_notification_card(level=LarkMsgLevel.Danger,
          title='紧急制动未执行', content=str(exc))
        return
      lark_sender.send_notification_card(level=LarkMsgLevel.Warning,
        title='紧急制动', content=f'已验证策略进程 PID={pid} 并写入制动/停止文件，保留券商客户端')
      return
    pid_path = _TRADE_DIR / "trading_main.pid"
    process, reason = _resolve_trading_main_process(pid_path)
    if process is None:
      trading_logger.warning(f"飞书紧急制动未执行: {reason}")
      lark_sender.send_notification_card(
        level=LarkMsgLevel.Danger,
        title="紧急制动未执行",
        content=reason,
      )
      return

    try:
      _arm_watchdog_halt(process.pid)
    except OSError as exc:
      reason = f'无法写入 watchdog 制动标记，拒绝只终止主进程: {exc}'
      trading_logger.warning(f"飞书紧急制动未执行: {reason}")
      lark_sender.send_notification_card(
        level=LarkMsgLevel.Danger,
        title="紧急制动未执行",
        content=reason,
      )
      return

    trading_logger.warning(f"飞书紧急制动: 已验证并终止交易主进程 PID={process.pid}")
    try:
      lark_sender.send_notification_card(
        level=LarkMsgLevel.Warning,
        title="紧急制动",
        content=f"已验证实盘主进程 PID={process.pid}，正在终止",
      )
    finally:
      # 通知失败也不能阻止已经验证过的紧急制动。
      terminate_process_tree(process)

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
      handlers[event_key]()
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
