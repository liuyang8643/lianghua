from typing import Dict, List


def _fmt_float(value: float) -> str:
  return f"{value:,.2f}"


def build_manual_confirmation_text(
    pending: Dict,
    execute_buy: bool,
    execute_sell: bool,
) -> str:
  signal_date = pending.get('signal_date')
  trade_date = pending.get('trade_date')
  sell_details: List[Dict] = pending.get('sell_details', [])
  buy_details: List[Dict] = pending.get('buy_details', [])

  lines: List[str] = []
  lines.append("=== 手动单次交易确认 ===")
  lines.append(f"signal_date: {signal_date}")
  lines.append(f"trade_date : {trade_date}")
  lines.append("")

  if execute_sell:
    lines.append(f"[卖出计划] {len(sell_details)} 笔")
    sell_total = 0.0
    for item in sell_details:
      code = item.get('code')
      name = item.get('name')
      board = item.get('board')
      volume = int(item.get('volume', 0))
      est_price = float(item.get('est_price', 0.0))
      est_amount = float(item.get('est_amount', 0.0))
      sell_total += est_amount
      parts = [code]
      if name:
        parts.append(name)
      if board:
        parts.append(f"[{board}]")
      label = ' '.join(parts)
      lines.append(f"- {label}: {volume} 股, 估算价格={_fmt_float(est_price)}, 估算金额={_fmt_float(est_amount)}")
    lines.append(f"卖出估算总金额: {_fmt_float(sell_total)}")
    lines.append("")

  if execute_buy:
    lines.append(f"[买入计划] {len(buy_details)} 笔")
    buy_total = 0.0
    for item in buy_details:
      code = item.get('code')
      name = item.get('name')
      board = item.get('board')
      shares = int(item.get('shares', 0))
      est_price = float(item.get('est_price', 0.0))
      est_amount = float(item.get('est_amount', 0.0))
      buy_total += est_amount
      parts = [code]
      if name:
        parts.append(name)
      if board:
        parts.append(f"[{board}]")
      label = ' '.join(parts)
      lines.append(f"- {label}: {shares} 股, 估算价格={_fmt_float(est_price)}, 估算金额={_fmt_float(est_amount)}")
    lines.append(f"买入估算总金额: {_fmt_float(buy_total)}")
    lines.append("")

  lines.append("输入 yes 确认执行，输入其他任意内容取消。")
  return "\n".join(lines)


def is_manual_confirmation_approved(user_input: str) -> bool:
  return (user_input or "").strip().lower() == "yes"
