"""
GA结果分析工具 - HTML报告生成器

功能：
1. 生成完整的HTML分析报告（Plotly交互式图表）
2. 包含：
   - Fitness随着代数迭代曲线（最小、平均、最大）
   - 各个因子值变化曲线
   - 所有重复个体的统计结果
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
import pickle

try:
    from core.logger import get_logger
    logger = get_logger(module="ga_analyzer")
except ImportError:
    import logging
    logger = logging.getLogger("ga_analyzer")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


class GAAnalyzer:
    """GA结果分析器 - HTML报告生成"""

    def __init__(self, result_dir: str):
        """
        Args:
            result_dir: GA结果目录路径（包含all_results.pkl和generation_results.pkl）
        """
        self.result_dir = Path(result_dir)
        self.data = self._load_data()
        self.all_individuals = []
        self.generation_stats = []
        self.factor_evolution = {}

        self._collect_all_individuals()
        self._compute_factor_evolution()

        logger.info(f"总代数: {len(self.generation_stats)}")
        logger.info(f"总个体数: {len(self.all_individuals)}")

    def _load_data(self) -> Dict:
        """加载GA结果数据"""
        all_individuals_file = self.result_dir / 'all_individuals.json'
        generation_results_file = self.result_dir / 'generation_results.pkl'

        data = {}

        # 优先从 generation_results.pkl 加载（更完整）
        with open(generation_results_file, 'rb') as f:
            data['generation_results'] = pickle.load(f)
        logger.info(f"加载结果文件: {generation_results_file}")

        # 从 pkl 中提取 all_individuals
        all_individuals = []
        for gen_stat in data['generation_results']:
            for ind in gen_stat.get('all_individuals', []):
                all_individuals.append(ind)
        data['all_individuals'] = all_individuals

        # 如果 JSON 文件存在，也加载（备用）
        if all_individuals_file.exists():
            with open(all_individuals_file, 'r', encoding='utf-8') as f:
                data['all_individuals_json'] = json.load(f)
            logger.info(f"加载结果文件: {all_individuals_file}")

        return data

    def _collect_all_individuals(self):
        """收集所有代数的所有个体"""
        # 从JSON文件加载所有个体
        for ind in self.data['all_individuals']:
            # 为旧结果添加默认值
            if 'target2_fitness' not in ind:
                ind['target2_fitness'] = 0
            self.all_individuals.append(ind)

        # 从generation_results加载统计信息
        for gen_stat in self.data['generation_results']:
            stats = {
                'generation': gen_stat['generation'],
                'generation_time': gen_stat['generation_time'],
                'max_fitness': gen_stat['max_fitness'],
                'mean_fitness': gen_stat['mean_fitness'],
                'min_fitness': gen_stat['min_fitness'],
                'population_size': gen_stat['population_size'],
                # 目标2（兼容旧结果）
                'max_target2_fitness': gen_stat.get('max_target2_fitness', 0),
                'mean_target2_fitness': gen_stat.get('mean_target2_fitness', 0),
                'min_target2_fitness': gen_stat.get('min_target2_fitness', 0),
            }
            self.generation_stats.append(stats)

        logger.info(f"收集到 {len(self.all_individuals)} 个个体，{len(self.generation_stats)} 代")

    def _compute_factor_evolution(self):
        """计算每一代所有因子的平均权重"""
        self.factor_evolution = {}

        # 按代数分组
        gen_groups = defaultdict(list)
        for ind in self.all_individuals:
            gen = ind['generation']
            gen_groups[gen].append(ind)

        # 计算每代的因子平均值
        for gen in sorted(gen_groups.keys()):
            inds = gen_groups[gen]
            weight_sums = defaultdict(float)
            weight_counts = defaultdict(int)

            for ind in inds:
                weights = ind['individual_config']['weights']
                for factor_key, weight_value in weights.items():
                    weight_sums[factor_key] += weight_value
                    weight_counts[factor_key] += 1

            factor_avgs = {}
            for factor_key in weight_sums:
                factor_avgs[factor_key] = weight_sums[factor_key] / weight_counts[factor_key]

            self.factor_evolution[gen] = factor_avgs

        logger.info(f"因子进化统计: {len(self.factor_evolution)}代")

    def _find_unique_individuals(self) -> Dict[str, List[Dict]]:
        """找出所有唯一的个体（基于Individual_config）"""
        unique_map = defaultdict(list)

        for ind in self.all_individuals:
            sig = json.dumps(ind['individual_config'], sort_keys=True)
            unique_map[sig].append(ind)

        return unique_map

    @staticmethod
    def smooth_curve(data: List[float], window: int = 5) -> List[float]:
        """使用移动平均平滑曲线

        Args:
            data: 原始数据
            window: 窗口大小（默认5）

        Returns:
            平滑后的数据
        """
        if len(data) < window:
            return data

        kernel = np.ones(window) / window
        smoothed = np.convolve(data, kernel, mode='valid')

        # 补齐头部（使用原始数据）
        pad_size = window - 1
        return list(data[:pad_size]) + list(smoothed)

    def generate_html_report(self, save_path: Optional[str] = None, smooth_window: int = 5):
        """生成完整的HTML分析报告"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("需要安装plotly: pip install plotly")
            return

        logger.info("=" * 80)
        logger.info("生成HTML分析报告...")
        logger.info("=" * 80)

        df_gen = pd.DataFrame(self.generation_stats)
        generations = df_gen['generation'].tolist()
        max_fitness = df_gen['max_fitness'].tolist()
        mean_fitness = df_gen['mean_fitness'].tolist()
        min_fitness = df_gen['min_fitness'].tolist()

        # 目标2（buy_n==sell_m）
        max_target2_fitness = df_gen['max_target2_fitness'].tolist()
        mean_target2_fitness = df_gen['mean_target2_fitness'].tolist()
        min_target2_fitness = df_gen['min_target2_fitness'].tolist()

        # 创建Plotly图表（4行：目标1、目标2、因子权重、重复个体统计）
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                '目标1 Fitness: 原始配置 (buy_n, sell_m)',
                '目标2 Fitness: 每天调仓 (buy_n==sell_m)',
                '各个因子值变化曲线',
                '重复个体统计'
            ),
            vertical_spacing=0.06,
            row_heights=[0.25, 0.25, 0.25, 0.25]
        )

        # === 目标1 Fitness进化曲线 ===
        max_fitness_smooth = self.smooth_curve(max_fitness, smooth_window)
        mean_fitness_smooth = self.smooth_curve(mean_fitness, smooth_window)
        min_fitness_smooth = self.smooth_curve(min_fitness, smooth_window)

        fig.add_trace(go.Scatter(
            x=generations, y=max_fitness_smooth,
            mode='lines', name='目标1-最大',
            line=dict(color='#2E7D32', width=2)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=generations, y=mean_fitness_smooth,
            mode='lines', name='目标1-平均',
            line=dict(color='#4CAF50', width=1, dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=generations, y=min_fitness_smooth,
            mode='lines', name='目标1-最小',
            line=dict(color='#81C784', width=1, dash='dot')
        ), row=1, col=1)

        # === 目标2 Fitness进化曲线 ===
        max_target2_smooth = self.smooth_curve(max_target2_fitness, smooth_window)
        mean_target2_smooth = self.smooth_curve(mean_target2_fitness, smooth_window)
        min_target2_smooth = self.smooth_curve(min_target2_fitness, smooth_window)

        fig.add_trace(go.Scatter(
            x=generations, y=max_target2_smooth,
            mode='lines', name='目标2-最大',
            line=dict(color='#1565C0', width=2)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=generations, y=mean_target2_smooth,
            mode='lines', name='目标2-平均',
            line=dict(color='#42A5F5', width=1, dash='dash')
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=generations, y=min_target2_smooth,
            mode='lines', name='目标2-最小',
            line=dict(color='#90CAF9', width=1, dash='dot')
        ), row=2, col=1)

        # === 因子权重进化 ===
        all_factor_keys = set()
        for gen_factors in self.factor_evolution.values():
            all_factor_keys.update(gen_factors.keys())

        plotly_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
            '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7'
        ]

        for idx, factor_key in enumerate(sorted(all_factor_keys)):
            factor_generations = []
            factor_values = []
            for gen in sorted(self.factor_evolution.keys()):
                if factor_key in self.factor_evolution[gen]:
                    factor_generations.append(gen)
                    factor_values.append(self.factor_evolution[gen][factor_key])

            factor_values_smooth = self.smooth_curve(factor_values, smooth_window)
            color = plotly_colors[idx % len(plotly_colors)]

            fig.add_trace(go.Scatter(
                x=factor_generations, y=factor_values_smooth,
                mode='lines', name=factor_key,
                line=dict(color=color, width=1.5)
            ), row=3, col=1)

        # === 重复个体统计 ===
        unique_map = self._find_unique_individuals()
        repeat_counts = [len(ind_list) for ind_list in unique_map.values()]
        if repeat_counts:
            # 统计重复次数分布
            count_distribution = Counter(repeat_counts)
            repeat_nums = sorted(count_distribution.keys())
            repeat_freqs = [count_distribution[n] for n in repeat_nums]

            fig.add_trace(go.Bar(
                x=repeat_nums, y=repeat_freqs,
                name='重复次数分布',
                marker=dict(color='#2196F3')
            ), row=4, col=1)

        # 设置坐标轴标签
        fig.update_xaxes(title_text="代数", row=1, col=1)
        fig.update_yaxes(title_text="Fitness", row=1, col=1)
        fig.update_xaxes(title_text="代数", row=2, col=1)
        fig.update_yaxes(title_text="Fitness", row=2, col=1)
        fig.update_xaxes(title_text="代数", row=3, col=1)
        fig.update_yaxes(title_text="因子平均权重", row=3, col=1)
        fig.update_xaxes(title_text="重复次数", row=4, col=1)
        fig.update_yaxes(title_text="个体数量", row=4, col=1)

        run_id = self.result_dir.name
        fig.update_layout(
            title=dict(text=f"GA进化分析 - {run_id}", x=0.5, font=dict(size=20)),
            height=1800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        unique_stats = []
        for sig_json, ind_list in unique_map.items():
            fitnesses = [ind['fitness'] for ind in ind_list]
            config = ind_list[0]['individual_config']

            unique_stats.append({
                'individual_config': config,
                'count': len(ind_list),
                'avg_fitness': np.mean(fitnesses),
                'max_fitness': max(fitnesses),
                'min_fitness': min(fitnesses),
                # 目标2数据
                'avg_target2_fitness': np.mean([ind.get('target2_fitness', 0) for ind in ind_list]),
                'max_target2_fitness': max([ind.get('target2_fitness', 0) for ind in ind_list]),
                'min_target2_fitness': min([ind.get('target2_fitness', 0) for ind in ind_list]),
                'individuals': ind_list
            })

        # 按平均fitness降序排序
        unique_stats.sort(key=lambda x: x['avg_fitness'], reverse=True)

        # 生成表格HTML
        table_rows_html = []
        all_details_data = []

        for idx, stat in enumerate(unique_stats):
            row_id = idx + 1
            config = stat['individual_config']
            weights = config['weights']
            sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
            weight_sig = ', '.join([f"{k}={v:.2f}" for k, v in sorted_weights[:5]])
            config_json = json.dumps(config).replace('"', '&quot;')

            detail_records = []
            sorted_individuals = sorted(stat['individuals'], key=lambda x: x['generation'])
            for ind in sorted_individuals:
                detail_records.append({
                    'gen': ind['generation'],
                    'fitness': round(ind['fitness'], 2),
                    'total_return': round(ind['total_return'], 2),
                    'buy_n': config['buy_n'],
                    'sell_m': config['sell_m'],
                })
            all_details_data.append(detail_records)

            row_html = f"""<tr data-row-id="{row_id}">
<td>{row_id}</td>
<td class="weight-cell" title="{', '.join([f'{k}={v:.4f}' for k, v in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)])}">{weight_sig}</td>
<td>{config['buy_n']}</td>
<td>{config['sell_m']}</td>
<td>{stat['count']}</td>
<td class="num-cell">{stat['avg_fitness']:.2f}</td>
<td class="num-cell">{stat['max_fitness']:.2f}</td>
<td class="num-cell">{stat['min_fitness']:.2f}</td>
<td class="num-cell" style="color:#1565C0">{stat['avg_target2_fitness']:.2f}</td>
<td class="num-cell" style="color:#1565C0">{stat['max_target2_fitness']:.2f}</td>
<td class="num-cell" style="color:#1565C0">{stat['min_target2_fitness']:.2f}</td>
<td>
<button class="detail-btn" data-row="{row_id}">详情</button>
<button class="config-btn" data-config="{config_json}" data-row="{row_id}">配置</button>
</td>
</tr>"""
            table_rows_html.append(row_html)

        table_html = '\n'.join(table_rows_html)
        all_details_json = json.dumps(all_details_data)

        # 生成完整HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GA运行结果分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 100%; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h2 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        #mainTable {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        #mainTable th {{ background: #e0f7fa; padding: 8px 5px; border: 1px solid #ddd; cursor: pointer; white-space: nowrap; }}
        #mainTable th:hover {{ background: #b2ebf2; }}
        #mainTable td {{ padding: 6px 5px; border: 1px solid #ddd; text-align: center; }}
        #mainTable tr:hover {{ background: #f0f8ff; }}
        .weight-cell {{ text-align: left !important; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .num-cell {{ font-family: monospace; }}
        .detail-btn {{ background: #2196F3; color: white; border: none; padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: 11px; margin-right: 3px; }}
        .detail-btn:hover {{ background: #1976D2; }}
        .config-btn {{ background: #4CAF50; color: white; border: none; padding: 3px 8px; border-radius: 3px; cursor: pointer; font-size: 11px; }}
        .config-btn:hover {{ background: #388E3C; }}
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; }}
        .modal.show {{ display: flex; justify-content: center; align-items: center; }}
        .modal-content {{ background: white; border-radius: 8px; max-width: 90%; max-height: 85%; overflow: auto; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .modal-header {{ padding: 15px 20px; background: #4CAF50; color: white; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; }}
        .modal-header h3 {{ margin: 0; }}
        .modal-close {{ background: none; border: none; color: white; font-size: 24px; cursor: pointer; }}
        .modal-body {{ padding: 20px; }}
        .modal-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .modal-table th {{ background: #f5f5f5; padding: 10px; border: 1px solid #ddd; }}
        .modal-table td {{ padding: 8px; border: 1px solid #ddd; text-align: center; }}
        .modal-table tr:nth-child(even) {{ background: #fafafa; }}
        .modal-table tr:hover {{ background: #e3f2fd; }}
    </style>
</head>
<body>
    <div class="container">
        <div>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>

        <h2>所有重复个体统计（共{len(unique_stats)}种唯一配置）</h2>
        <p style="color:#666;margin-bottom:10px;">点击"详情"查看该配置的所有历史记录。</p>

        <table id="mainTable">
            <thead><tr>
                <th>排名</th>
                <th>权重配置(前5)</th>
                <th>buy_n</th>
                <th>sell_m</th>
                <th>重复次数</th>
                <th>目标1-平均</th>
                <th>目标1-最大</th>
                <th>目标1-最小</th>
                <th>目标2-平均</th>
                <th>目标2-最大</th>
                <th>目标2-最小</th>
                <th>操作</th>
            </tr></thead>
            <tbody id="tableBody">{table_html}</tbody>
        </table>
    </div>

    <div id="detailModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modalTitle">详细记录</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <table class="modal-table">
                    <thead><tr>
                        <th>代数</th><th>Fitness</th><th>总收益%</th><th>buy_n</th><th>sell_m</th>
                    </tr></thead>
                    <tbody id="modalBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const allDetails = {all_details_json};

        document.addEventListener('DOMContentLoaded', function() {{
            document.addEventListener('click', function(e) {{
                if (e.target.classList.contains('detail-btn')) {{
                    const rowId = parseInt(e.target.dataset.row);
                    showDetail(rowId);
                }}
                if (e.target.classList.contains('config-btn')) {{
                    const config = JSON.parse(e.target.dataset.config.replace(/&quot;/g, '"'));
                    const rowNum = e.target.dataset.row;
                    downloadConfig(config, rowNum);
                }}
            }});

            document.getElementById('detailModal').addEventListener('click', function(e) {{
                if (e.target === this) closeModal();
            }});
        }});

        function showDetail(rowId) {{
            const details = allDetails[rowId - 1];
            const tbody = document.getElementById('modalBody');
            document.getElementById('modalTitle').textContent = `详细记录（共${{details.length}}条）`;

            tbody.innerHTML = details.map(d => {{
                return `<tr>
                <td>${{d.gen}}</td><td>${{d.fitness}}</td><td>${{d.total_return}}%</td>
                <td>${{d.buy_n}}</td><td>${{d.sell_m}}</td>
            </tr>`;
            }}).join('');

            document.getElementById('detailModal').classList.add('show');
        }}

        function closeModal() {{
            document.getElementById('detailModal').classList.remove('show');
        }}

        function downloadConfig(config, rowNum) {{
            const jsonStr = JSON.stringify(config, null, 2);
            const blob = new Blob([jsonStr], {{type: 'application/json'}});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `individual_config_row${{rowNum}}.json`;
            a.click();
        }}
    </script>
</body>
</html>"""

        if save_path is None:
            save_path = self.result_dir / 'analysis_report.html'
        else:
            save_path = Path(save_path)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"✅ HTML报告已生成: {save_path}")
        logger.info(f"   - 进化曲线: {len(generations)}代")
        logger.info(f"   - 因子进化: {len(all_factor_keys)}个因子")
        logger.info(f"   - 唯一配置: {len(unique_stats)}种")
        logger.info("=" * 80)

    def print_summary(self):
        """打印分析摘要"""
        logger.info("=" * 80)
        logger.info("GA优化分析摘要")
        logger.info("=" * 80)

        logger.info(f"总代数: {len(self.generation_stats)}")
        logger.info(f"总个体数: {len(self.all_individuals)}")

        df_gen = pd.DataFrame(self.generation_stats)
        logger.info(f"\n进化统计:")
        logger.info(f"  最终平均Fitness: {df_gen['mean_fitness'].iloc[-1]:.4f}")
        logger.info(f"  最终最大Fitness: {df_gen['max_fitness'].iloc[-1]:.4f}")

        unique_map = self._find_unique_individuals()
        logger.info(f"\n唯一配置: {len(unique_map)}种")
        logger.info("=" * 80)


def main():
    """命令行工具"""
    parser = argparse.ArgumentParser(description='GA结果分析工具 - HTML报告生成')

    parser.add_argument('result_dir', type=str, help='GA结果目录路径（包含all_results.pkl）')
    parser.add_argument('--output', type=str, default=None, help='输出HTML路径')
    parser.add_argument('--smooth-window', type=int, default=5, help='平滑窗口大小（默认5）')

    args = parser.parse_args()

    analyzer = GAAnalyzer(args.result_dir)
    analyzer.print_summary()
    analyzer.generate_html_report(save_path=args.output, smooth_window=args.smooth_window)


if __name__ == '__main__':
    main()

