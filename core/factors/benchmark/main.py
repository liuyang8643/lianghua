import os
import pickle
import random

from core.database import allow_buy_stock_code_list
from core.factors.benchmark import calculate_factor_correlation, generate_html_report

if __name__ == '__main__':
  """
  因子相关性分析模块（日度截面分析）
  计算每日因子得分排名与T+M日收益率的相关性
  """
  stock_list = allow_buy_stock_code_list()

  from core.factors import *

  report = calculate_factor_correlation(
    factor_cls=KDJ,
    start_date=date(2024, 1, 1),
    end_date=date(2025, 1, 1),
    m_days=[1, 3, 5, 10, 20],  # 多个持有期对比：T+1, T+3, T+5, T+10, T+20日
    # stock_codes=stock_list
    stock_codes=random.sample(stock_list, 100),
  )

  # 保存Pickle报告（高效的Python原生格式）
  report_dir = "./reports"
  os.makedirs(report_dir, exist_ok=True)
  pickle_filename = f"{report_dir}/factor-correlation-{report.factor_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

  # 使用pickle直接序列化对象，无需转换
  with open(pickle_filename, 'wb') as f:
    pickle.dump(report, f, protocol=pickle.HIGHEST_PROTOCOL)

  print(f"Pickle报告已保存: {pickle_filename}")

  # 显示文件大小
  file_size = os.path.getsize(pickle_filename)
  print(f"文件大小: {file_size / 1024:.2f} KB")

  # 生成HTML报告
  generate_html_report(report)
