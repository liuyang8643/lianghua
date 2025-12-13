import os
import pickle
import webbrowser
from datetime import datetime

from core import core_logger
from core.factors.benchmark.calc_correlation import FactorCorrelationReport

def generate_html_report(report: FactorCorrelationReport) -> str:
  """
  生成HTML报告

  Args:
      report: 相关性报告

  Returns:
      str: HTML文件路径
  """
  from jinja2 import Environment, FileSystemLoader, select_autoescape

  # 准备模板数据
  # 按相关系数排序（绝对值从大到小）
  sorted_correlations = sorted(
    report.stock_correlations,
    key=lambda x: abs(x.correlation) if x.correlation is not None else -1,
    reverse=True
  )

  # 分类统计
  positive_corr = [r for r in report.stock_correlations if r.correlation is not None and r.correlation > 0]
  negative_corr = [r for r in report.stock_correlations if r.correlation is not None and r.correlation < 0]
  invalid_corr = [r for r in report.stock_correlations if r.correlation is None]

  # 准备数据
  data = {
    'factor_name': report.factor_name,
    'start_date': report.start_date.strftime('%Y-%m-%d'),
    'end_date': report.end_date.strftime('%Y-%m-%d'),
    'total_stocks': report.total_stocks,
    'valid_stocks': report.valid_stocks,
    'invalid_stocks': len(invalid_corr),
    'avg_correlation': report.avg_correlation,
    'median_correlation': report.median_correlation,
    'positive_count': len(positive_corr),
    'negative_count': len(negative_corr),
    'correlations': sorted_correlations,
    'generated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  }

  # 创建输出目录
  report_dir = "./reports"
  os.makedirs(report_dir, exist_ok=True)

  # 渲染模板
  template_dir = os.path.join(os.path.dirname(__file__), 'templates')
  env = Environment(
    loader=FileSystemLoader(template_dir),
    autoescape=select_autoescape(['html', 'xml'])
  )
  template = env.get_template('correlation_report.html')
  html_content = template.render(**data)

  # 保存HTML文件
  filename = f"{report_dir}/factor-correlation-{report.factor_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
  with open(filename, 'w', encoding='utf-8') as f:
    f.write(html_content)

  core_logger.info(f"相关性报告已生成: {filename}")

  # 自动在浏览器中打开
  try:
    abs_path = os.path.abspath(filename)
    webbrowser.open(f'file:///{abs_path}')
    core_logger.info(f"已在浏览器中打开报告")
  except Exception as e:
    core_logger.warning(f"无法自动打开浏览器: {e}")

  return filename

if __name__ == '__main__':
  """
  从Pickle文件读取报告并生成HTML
  用法: 修改下面的pickle_path变量为你想要读取的Pickle文件路径
  """
  # 指定要读取的Pickle文件路径
  # 例如: "./reports/factor-correlation-MACD-20251213_123456.pkl"
  pickle_path = "./reports/factor-correlation-MACD-20251213_123456.pkl"

  # 从Pickle加载报告
  print(f"正在读取Pickle报告: {pickle_path}")
  with open(pickle_path, 'rb') as f:
    report = pickle.load(f)
    print(f"报告加载成功: {report.factor_name} ({report.start_date} 至 {report.end_date})")
    print(f"  - 总股票数: {report.total_stocks}")
    print(f"  - 有效股票数: {report.valid_stocks}")
    print(f"  - 平均相关系数: {report.avg_correlation:.4f}")
    # 生成HTML报告
    html_path = generate_html_report(report)
    print(f"HTML报告已生成: {html_path}")
