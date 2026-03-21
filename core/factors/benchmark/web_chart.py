"""
多因子得分可视化 Web 页面 (ECharts)

用法:
  python web_chart.py                          # 随机选股，启动 Web 服务
  python web_chart.py --port 9090              # 指定端口
  python web_chart.py --code 600000.SH         # 直接查看指定股票（生成 HTML 并打开）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import pandas as pd

from core.database.data import get_full_market_data
from core.database.stock_list import allow_buy_stock_code_list
from core.factors.helpers.indicators import FactorCtx
from core.factors.helpers.interface import BaseFactor

def _get_factors() -> list[BaseFactor]:
  """延迟导入全部技术面因子"""
  from core.factors import (
    SmallCap, MACD, BBI, CCI, TRIXFactor, MOMFactor, ADXFactor, WMACross, KDJ,
  )
  return [WMACross()]

FACTOR_COLORS = [
  '#5470c6', '#91cc75', '#fac858', '#ee6666',
  '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc',
]

def _compute_one_factor(factor: BaseFactor, stock_code: str,
                        dates: pd.DatetimeIndex) -> tuple[str, list]:
  """计算单个因子在所有日期的得分（供线程池调用）"""
  name = factor.__class__.__name__
  scores: list[float | None] = []
  for d in dates:
    try:
      ctx = FactorCtx(stock_code, d.to_pydatetime())
      result = factor.calc(ctx)
      s = result['score'] if result['err'] is None else None
      scores.append(round(s, 6) if s is not None else None)
    except Exception:
      scores.append(None)
  print(f"  ✓ {name}")
  return name, scores

def _json_safe(v):
  """NaN / Inf → None，保证 JSON 可序列化"""
  if v is None:
    return None
  if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
    return None
  return v

def build_chart_data(stock_code: str) -> dict | None:
  """构建 K线 + 成交量 + 均线 + 因子得分所需的全部数据（仅 2020–2025）"""
  df = get_full_market_data(stock_code)
  if df is None or df.empty:
    return None

  dates = pd.to_datetime(df['time'], unit='ms')
  mask = (dates >= '2021-01-01') & (dates <= '2022-12-31')
  df = df.loc[mask].reset_index(drop=True)
  dates = dates.loc[mask].reset_index(drop=True)
  if df.empty:
    return None
  date_strs = dates.dt.strftime('%Y-%m-%d').tolist()

  # K线: ECharts candlestick 格式 [open, close, low, high]
  kline = [[_json_safe(round(r['open'], 3)),
            _json_safe(round(r['close'], 3)),
            _json_safe(round(r['low'], 3)),
            _json_safe(round(r['high'], 3))]
           for _, r in df.iterrows()]

  # 成交量 (附带涨跌标记 1/-1)
  volumes = []
  for _, row in df.iterrows():
    flag = 1 if row['close'] >= row['open'] else -1
    volumes.append([int(row['volume']), flag])

  # 均线
  ma5 = df['close'].rolling(5).mean().round(3)
  ma20 = df['close'].rolling(20).mean().round(3)
  ma5_list = [_json_safe(v) for v in ma5]
  ma20_list = [_json_safe(v) for v in ma20]

  # ── 因子得分（多线程并行计算） ──
  factors = _get_factors()
  print(f"  开始计算 {len(factors)} 个因子 × {len(dates)} 日 ...")
  factor_data: dict[str, list] = {}
  with ThreadPoolExecutor(max_workers=min(len(factors), os.cpu_count() or 4)) as pool:
    futures = [pool.submit(_compute_one_factor, f, stock_code, dates)
               for f in factors]
    for fut in futures:
      name, scores = fut.result()
      factor_data[name] = scores

  return {
    'dates': date_strs,
    'kline': kline,
    'volumes': volumes,
    'ma5': ma5_list,
    'ma20': ma20_list,
    'factors': factor_data,
  }

def build_html(stock_code: str, data: dict) -> str:
  """生成包含 ECharts 的完整 HTML 页面"""
  chart_json = json.dumps(data, ensure_ascii=False)
  factor_names = list(data['factors'].keys())

  # 因子 series 的 JS 片段
  factor_series_js_parts = []
  for i, name in enumerate(factor_names):
    color = FACTOR_COLORS[i % len(FACTOR_COLORS)]
    factor_series_js_parts.append(
      f"{{"
      f"name:'{name}',type:'line',xAxisIndex:2,yAxisIndex:2,"
      f"data:D.factors['{name}'],showSymbol:false,"
      f"lineStyle:{{width:1.5}},itemStyle:{{color:'{color}'}}"
      f"}}"
    )
  factor_series_js = ','.join(factor_series_js_parts)

  legend_data_js = json.dumps(['MA5', 'MA20'] + factor_names, ensure_ascii=False)

  # dataZoom 默认显示最近 120 个交易日
  total = len(data['dates'])
  zoom_start = max(0, round((1 - 120 / total) * 100)) if total > 120 else 0

  return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>多因子可视化 — {stock_code}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Microsoft YaHei",sans-serif;background:#1a1a2e;color:#eee}}
.bar{{display:flex;align-items:center;gap:8px;padding:10px 16px;background:#16213e}}
.bar input{{padding:6px 10px;font-size:14px;width:180px;border:1px solid #444;border-radius:4px;background:#0f3460;color:#eee}}
.bar button{{padding:6px 14px;font-size:14px;cursor:pointer;border:1px solid #444;border-radius:4px;background:#0f3460;color:#eee}}
.bar button:hover{{background:#1a4a8a}}
.bar .note{{color:#888;font-size:13px;margin-left:8px}}
#chart{{width:100%;height:calc(100vh - 44px)}}
</style>
</head>
<body>
<div class="bar">
  <input id="code" type="text" placeholder="股票代码 如 600000.SH" value="{stock_code}">
  <button onclick="go()">查看</button>
  <button onclick="location.href='/random'">🎲 随机</button>
  <span class="note">多因子得分可视化 · ECharts</span>
</div>
<div id="chart"></div>
<script>
const D={chart_json};
const chart=echarts.init(document.getElementById('chart'));
const upC='#ef5350',dnC='#26a69a';

const volData=D.volumes.map(v=>({{value:v[0],itemStyle:{{color:v[1]>0?'rgba(239,83,80,0.5)':'rgba(38,166,154,0.5)'}}}}));

chart.setOption({{
  animation:false,
  tooltip:{{
    trigger:'axis',axisPointer:{{type:'cross'}},
    backgroundColor:'rgba(22,33,62,0.92)',borderColor:'#444',
    textStyle:{{color:'#eee',fontSize:12}},
  }},
  legend:{{
    top:0,left:'center',
    textStyle:{{color:'#ccc',fontSize:11}},
    data:{legend_data_js},
  }},
  axisPointer:{{link:[{{xAxisIndex:'all'}}]}},
  grid:[
    {{left:60,right:60,top:40,height:'38%'}},
    {{left:60,right:60,top:'50%',height:'10%'}},
    {{left:60,right:60,top:'65%',height:'22%'}},
  ],
  xAxis:[
    {{type:'category',data:D.dates,gridIndex:0,boundaryGap:true,
      axisLine:{{lineStyle:{{color:'#444'}}}},axisLabel:{{show:false}}}},
    {{type:'category',data:D.dates,gridIndex:1,boundaryGap:true,
      axisLine:{{lineStyle:{{color:'#444'}}}},axisLabel:{{show:false}}}},
    {{type:'category',data:D.dates,gridIndex:2,boundaryGap:true,
      axisLine:{{lineStyle:{{color:'#444'}}}},axisLabel:{{color:'#888'}}}},
  ],
  yAxis:[
    {{scale:true,gridIndex:0,
      splitLine:{{lineStyle:{{color:'#2a2a4a'}}}},
      axisLine:{{lineStyle:{{color:'#444'}}}},axisLabel:{{color:'#888'}}}},
    {{scale:true,gridIndex:1,splitNumber:2,
      splitLine:{{lineStyle:{{color:'#2a2a4a'}}}},
      axisLine:{{lineStyle:{{color:'#444'}}}},axisLabel:{{color:'#888'}}}},
    {{scale:true,gridIndex:2,splitNumber:3,
      splitLine:{{lineStyle:{{color:'#2a2a4a'}}}},
      axisLine:{{lineStyle:{{color:'#444'}}}},axisLabel:{{color:'#888'}}}},
  ],
  dataZoom:[
    {{type:'inside',xAxisIndex:[0,1,2],start:{zoom_start},end:100}},
    {{type:'slider',xAxisIndex:[0,1,2],bottom:10,start:{zoom_start},end:100,
      borderColor:'#444',textStyle:{{color:'#888'}},
      dataBackground:{{lineStyle:{{color:'#5470c6'}},areaStyle:{{color:'rgba(84,112,198,0.2)'}}}},
    }},
  ],
  series:[
    {{name:'K线',type:'candlestick',xAxisIndex:0,yAxisIndex:0,data:D.kline,
      itemStyle:{{color:upC,color0:dnC,borderColor:upC,borderColor0:dnC}}}},
    {{name:'MA5',type:'line',xAxisIndex:0,yAxisIndex:0,data:D.ma5,
      smooth:true,showSymbol:false,lineStyle:{{width:1,color:'#ffa726'}},itemStyle:{{color:'#ffa726'}}}},
    {{name:'MA20',type:'line',xAxisIndex:0,yAxisIndex:0,data:D.ma20,
      smooth:true,showSymbol:false,lineStyle:{{width:1,color:'#42a5f5'}},itemStyle:{{color:'#42a5f5'}}}},
    {{name:'成交量',type:'bar',xAxisIndex:1,yAxisIndex:1,data:volData}},
    {factor_series_js}
  ],
}});

window.addEventListener('resize',()=>chart.resize());
function go(){{const c=document.getElementById('code').value.trim();if(c)location.href='/?code='+encodeURIComponent(c)}}
document.getElementById('code').addEventListener('keydown',e=>{{if(e.key==='Enter')go()}});
</script>
</body>
</html>"""

# ──────────────────────── HTTP 服务 ────────────────────────
class ChartHandler(BaseHTTPRequestHandler):
  stock_list: list[str] | None = None

  def do_GET(self):
    parsed = urlparse(self.path)
    params = parse_qs(parsed.query)

    if parsed.path == '/random':
      code = random.choice(self._get_stock_list())
      self.send_response(302)
      self.send_header('Location', f'/?code={code}')
      self.end_headers()
      return

    if parsed.path == '/favicon.ico':
      self.send_response(204)
      self.end_headers()
      return

    code = params.get('code', [None])[0]
    if not code:
      code = random.choice(self._get_stock_list())

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在计算 {code} ...")
    data = build_chart_data(code)
    if data is None:
      html = (f"<!DOCTYPE html><html><body style='background:#1a1a2e;"
              f"color:#ef5350;padding:40px;font-size:24px'>"
              f"无法获取 {code} 的数据</body></html>")
    else:
      html = build_html(code, data)

    self.send_response(200)
    self.send_header('Content-Type', 'text/html; charset=utf-8')
    self.end_headers()
    self.wfile.write(html.encode('utf-8'))

  def _get_stock_list(self) -> list[str]:
    if ChartHandler.stock_list is None:
      print("正在加载股票列表...")
      ChartHandler.stock_list = allow_buy_stock_code_list()
      print(f"共 {len(ChartHandler.stock_list)} 只可交易股票")
    return ChartHandler.stock_list

  def log_message(self, format, *args):
    pass

# ──────────────────────── 入口 ────────────────────────
def main():
  parser = argparse.ArgumentParser(description='多因子得分可视化')
  parser.add_argument('--port', type=int, default=8080, help='服务端口 (默认 8080)')
  parser.add_argument('--code', type=str, default=None,
                      help='直接查看指定股票（生成 HTML 并打开浏览器）')
  args = parser.parse_args()

  if args.code:
    import webbrowser
    import tempfile
    print(f"正在计算 {args.code} ...")
    data = build_chart_data(args.code)
    if data is None:
      print(f"无法获取 {args.code} 的数据")
      return
    html = build_html(args.code, data)
    path = os.path.join(tempfile.gettempdir(), 'factor_chart.html')
    with open(path, 'w', encoding='utf-8') as f:
      f.write(html)
    webbrowser.open(path)
    print(f"图表已保存至 {path}")
    return

  print(f"启动服务: http://localhost:{args.port}")
  print("Ctrl+C 停止\n")
  server = HTTPServer(('localhost', args.port), ChartHandler)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("\n已停止")
    server.server_close()

if __name__ == '__main__':
  main()
