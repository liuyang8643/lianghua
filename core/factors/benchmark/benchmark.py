import os
import pickle
import random
from datetime import date, datetime

from core.database import allow_buy_stock_code_list, init_stock_detail_cache
from core.factors import *
from core.factors.benchmark import calculate_factor_correlation, generate_html_report

if __name__ == '__main__':
  stock_list = allow_buy_stock_code_list()

  report = calculate_factor_correlation(
    factor_cls=RetailFlowMomentum,
    start_date=date(2024, 6, 1),
    end_date=date(2024, 7, 1),
    m_days=[1, 3, 5, 10, 20],
    stock_codes=stock_list
    # stock_codes=random.sample(stock_list, 1000),
  )

  # 保存Pickle报告
  report_dir = "./reports"
  os.makedirs(report_dir, exist_ok=True)
  pickle_filename = f"{report_dir}/factor-correlation-{report.factor_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

  with open(pickle_filename, 'wb') as f:
    pickle.dump(report, f, protocol=pickle.HIGHEST_PROTOCOL)

  print(f"Pickle报告已保存: {pickle_filename} ({os.path.getsize(pickle_filename) / 1024:.2f} KB)")

  # 生成HTML报告
  html_file = generate_html_report(report)
  print(f"HTML报告已生成: {html_file}")
