"""
因子Benchmark模块
用于分析因子与股价的相关性
"""
from .calc_correlation import calculate_factor_correlation
from .report import generate_html_report

__all__ = ['calculate_factor_correlation', 'generate_html_report']
