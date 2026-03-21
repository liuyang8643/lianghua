import os
import pickle
from datetime import date, datetime

from core.database import get_all_stock_code_list
from core.factors import *
from core.factors.benchmark import calculate_factor_correlation, generate_html_report


def print_report_summary(report) -> None:
  print(f"因子: {report.factor_name}")
  print(f"区间: {report.start_date} -> {report.end_date}")
  print(f"股票数: {report.total_stocks}")
  for ps in report.period_statistics:
    print(
      f"T+{ps.m_days}: avg_corr={ps.avg_correlation:.4f}, "
      f"rank_corr={ps.avg_rank_correlation:.4f}, IR={ps.ir:.2f}, "
      f"valid_days={ps.valid_days}"
    )


def parse_env_date(name: str, default: date) -> date:
  value = os.getenv(name)
  if not value:
    return default
  return datetime.strptime(value, '%Y-%m-%d').date()


def sample_stocks(stock_list: list[str]) -> list[str]:
  sample_step = int(os.getenv('BENCHMARK_SAMPLE_STEP', '1') or '1')
  if sample_step > 1:
    stock_list = stock_list[::sample_step]

  sample_size = int(os.getenv('BENCHMARK_SAMPLE_SIZE', '0') or '0')
  if sample_size > 0:
    stock_list = stock_list[:sample_size]

  return stock_list

if __name__ == '__main__':
  stock_list = sample_stocks(get_all_stock_code_list())

  report = calculate_factor_correlation(
    factor_cls=WMACross,
    start_date=parse_env_date('BENCHMARK_START_DATE', date(2024, 5, 1)),
    end_date=parse_env_date('BENCHMARK_END_DATE', date(2025, 5, 1)),
    m_days=[1, 3, 5, 10, 20, 30, 60],
    stock_codes=stock_list
  )

  # 保存Pickle报告
  report_dir = "./reports"
  os.makedirs(report_dir, exist_ok=True)
  pickle_filename = f"{report_dir}/factor-correlation-{report.factor_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

  with open(pickle_filename, 'wb') as f:
    pickle.dump(report, f, protocol=pickle.HIGHEST_PROTOCOL)

  print(f"Pickle报告已保存: {pickle_filename} ({os.path.getsize(pickle_filename) / 1024:.2f} KB)")
  print_report_summary(report)

  if os.getenv('BENCHMARK_GENERATE_HTML', '').lower() in {'1', 'true', 'yes'}:
    html_file = generate_html_report(report)
    print(f"HTML报告已生成: {html_file}")
