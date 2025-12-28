import os
import webbrowser
from datetime import datetime

import numpy as np

from core import core_logger
from core.factors.benchmark.calc_correlation import FactorCorrelationReport

def generate_html_report(report: FactorCorrelationReport) -> str:
  """生成HTML报告"""
  from jinja2 import Environment, FileSystemLoader, select_autoescape

  # 准备多持有期数据
  periods_data = []
  for ps in report.period_statistics:
    sorted_daily = sorted(ps.daily_correlations, key=lambda x: x.trade_date)

    periods_data.append({
      'm_days': ps.m_days,
      'daily_correlations': sorted_daily,
      'avg_correlation': ps.avg_correlation,
      'median_correlation': ps.median_correlation,
      'avg_rank_correlation': ps.avg_rank_correlation,
      'positive_count': ps.positive_samples,
      'negative_count': ps.negative_samples,
      'valid_days': ps.valid_days,
      'total_days': len(ps.daily_correlations),
      'total_samples': ps.total_samples,
      'positive_days': ps.positive_days,
      'negative_days': ps.negative_days,
      'ic_mean': ps.ic_mean,
      'ic_std': ps.ic_std,
      'ir': ps.ir,
      'ic_ir': ps.ic_ir,
    })

  # 计算平均有效股票数
  valid_stocks = 0
  if report.period_statistics and report.period_statistics[0].daily_correlations:
    valid_stocks = int(np.mean([dc.valid_stock_count
                                for dc in report.period_statistics[0].daily_correlations]))

  data = {
    'factor_name': report.factor_name,
    'start_date': report.start_date.strftime('%Y-%m-%d'),
    'end_date': report.end_date.strftime('%Y-%m-%d'),
    'm_days_list': report.m_days_list,
    'total_stocks': report.total_stocks,
    'valid_stocks': valid_stocks,
    'periods_data': periods_data,
    'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  }

  # 渲染模板
  report_dir = "./reports"
  os.makedirs(report_dir, exist_ok=True)

  template_dir = os.path.join(os.path.dirname(__file__), 'templates')
  env = Environment(loader=FileSystemLoader(template_dir),
                   autoescape=select_autoescape(['html', 'xml']))
  html_content = env.get_template('correlation_report.html').render(**data)

  # 保存HTML
  m_days_str = '_'.join(map(str, report.m_days_list))
  filename = f"{report_dir}/factor-correlation-{report.factor_name}-T+{m_days_str}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

  with open(filename, 'w', encoding='utf-8') as f:
    f.write(html_content)

  core_logger.info(f"报告已生成: {filename}")

  # 自动打开浏览器
  try:
    webbrowser.open(f'file:///{os.path.abspath(filename)}')
  except Exception as e:
    core_logger.warning(f"无法打开浏览器: {e}")

  return filename

