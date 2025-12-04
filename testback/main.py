from datetime import date, datetime, timedelta

from core import allow_buy_stock_code_list, TopN

if __name__ == "__main__":
  all_stocks = allow_buy_stock_code_list(date.today())
  sorted_stocks = TopN(all_stocks, datetime.now() - timedelta(2)).get_ordered_stocks()
  print(f"TopN选出的股票代码: {sorted_stocks}")
