from datetime import date
from xtquant import xtdata

from utils.stock.format import format_qmt_date
from core.logger import core_logger

# 用于记录最后一次下载板块数据的日期
_last_sector_data_download_date = None

def allow_buy_stock_code_list(base_date: date = None) -> list[str]:
  """ 获取可买股票列表 """
  global _last_sector_data_download_date

  input_date = base_date or date.today()

  # 检查今天是否已经下载过数据
  if _last_sector_data_download_date != date.today():
    # xtdata.download_sector_data()
    xtdata.download_history_contracts()
    sector_list = xtdata.get_sector_list()
    core_logger.debug(f'更新板块选择器成功：{'|'.join(sector_list)}')
    _last_sector_data_download_date = date.today()

  time_tag = format_qmt_date(input_date)
  source = xtdata.get_stock_list_in_sector("沪深A股", time_tag)
  kcb = xtdata.get_stock_list_in_sector("科创板", time_tag)
  st = xtdata.get_stock_list_in_sector("沪深风险警示", time_tag)
  sst = xtdata.get_stock_list_in_sector("沪深退市整理", time_tag)

  res = [item for item in source if item not in kcb and item not in st and item not in sst]
  if not res:
    core_logger.error(f'获取可买股票列表失败，返回空列表')
    return res

  core_logger.debug(f'已获取股票列表{len(res)}条')

  return res
