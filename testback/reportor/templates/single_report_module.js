const REPORT_DATA = JSON.parse(document.getElementById('report-data').textContent);
const CHARTS = [];
const TOOLTIP_INSTANCES = [];
const TABLE_INSTANCES = [];
const COLORS = {
  strategy: '#2563EB', benchmark: '#B45309', positive: '#15803D',
  negative: '#C2413D', teal: '#0F766E', turnover: '#7C3AED', muted: '#647180', grid: '#E4E9EF',
};

function renderNoData(root, message) {
  if (!root) return;
  root.innerHTML = `<div class="no-data">${message || '暂无数据'}</div>`;
}

function makeChart(id) {
  const el = document.getElementById(id);
  if (!el) return null;
  const chart = echarts.init(el);
  CHARTS.push(chart);
  return chart;
}

function destroyTooltips() {
  while (TOOLTIP_INSTANCES.length) {
    const instance = TOOLTIP_INSTANCES.pop();
    try { instance.destroy(); } catch (_) {}
  }
}

function initTooltips(scope = document) {
  if (typeof window.tippy !== 'function') return;
  scope.querySelectorAll('[data-tippy-content]').forEach((node) => {
    const instance = window.tippy(node, {
      allowHTML: true,
      interactive: true,
      hideOnClick: false,
      maxWidth: 480,
      appendTo: () => document.body,
      theme: 'wbr',
    });
    TOOLTIP_INSTANCES.push(instance);
  });
}

function renderPerformanceCharts() {
  const payload = REPORT_DATA.charts.equity;
  const chart = makeChart('equity-chart');
  if (!chart || !payload?.trade_dates?.length) {
    renderNoData(document.getElementById('equity-chart'), '暂无净值数据');
    return;
  }

  const zoomStart = 0;
  const navValues = [...(payload.strategy_nav || []), ...(payload.benchmark_nav || [])]
    .map(Number).filter((value) => Number.isFinite(value) && value > 0);
  const navMin = Math.min(...navValues);
  const navMax = Math.max(...navValues);
  const useLogAxis = navValues.length > 0 && navMax / navMin > 20;

  const dateAxis = {
    type: 'category', data: payload.trade_dates, boundaryGap: false,
    axisLine: { lineStyle: { color: COLORS.grid } }, axisTick: { show: false },
    axisLabel: { hideOverlap: true, color: COLORS.muted, fontSize: 10 },
  };
  const valueAxis = {
    type: 'value', scale: true,
    splitLine: { lineStyle: { color: COLORS.grid } },
    axisLabel: { color: COLORS.muted, fontSize: 10 },
  };
  const maxDrawdownArea = REPORT_DATA.summary.max_drawdown_start && REPORT_DATA.summary.max_drawdown_end
    ? [[{ xAxis: REPORT_DATA.summary.max_drawdown_start }, { xAxis: REPORT_DATA.summary.max_drawdown_end }]]
    : [];

  chart.setOption({
    animation: false,
    color: [COLORS.strategy, COLORS.benchmark, COLORS.negative, COLORS.teal, COLORS.muted],
    legend: { top: 0, right: 8, textStyle: { color: COLORS.muted } },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'cross' },
      formatter: (params) => {
        const index = params[0]?.dataIndex ?? 0;
        const activity = payload.activity?.[index] || {};
        const dailyReturn = Number(payload.daily_returns_pct?.[index] || 0);
        const strategyNav = Number(payload.strategy_nav?.[index] || 0);
        const benchmarkNav = Number(payload.benchmark_nav?.[index] || 0);
        const drawdown = Number(payload.drawdown_pct?.[index] || 0);
        const exposure = Number(payload.exposure_pct?.[index] || 0);
        const rebalanceFunds = Number(payload.rebalance_funds_pct?.[index] || 0);
        const rows = [
          `<span style="color:${COLORS.strategy}">●</span> 策略净值: <b>${strategyNav.toFixed(4)}</b>`,
          `<span style="color:${COLORS.benchmark}">●</span> 沪深300: <b>${benchmarkNav.toFixed(4)}</b>`,
          `<span style="color:${COLORS.negative}">●</span> 回撤: <b>${drawdown.toFixed(2)}%</b>`,
          `<span style="color:${COLORS.teal}">●</span> 实际仓位: <b>${exposure.toFixed(1)}%</b>`,
          `<span style="color:${COLORS.turnover}">●</span> 当日调仓资金: <b>${rebalanceFunds.toFixed(1)}%</b>`,
        ];
        rows.push(`日收益: <b style="color:${dailyReturn >= 0 ? COLORS.positive : COLORS.negative}">${dailyReturn >= 0 ? '+' : ''}${dailyReturn.toFixed(2)}%</b>`);
        if ((activity.buys || 0) + (activity.sells || 0) > 0) rows.push(`成交: 买 ${activity.buys || 0} / 卖 ${activity.sells || 0}`);
        return `<b>${payload.trade_dates[index]}</b><br>${rows.join('<br>')}`;
      },
    },
    grid: { left: 60, right: 66, top: 48, bottom: 54 },
    xAxis: dateAxis,
    yAxis: [
      {
        ...valueAxis,
        type: useLogAxis ? 'log' : 'value',
        name: useLogAxis ? '净值（对数）' : '净值',
        logBase: 10,
        axisLabel: {
          ...valueAxis.axisLabel,
          formatter: (v) => {
            const value = Number(v);
            if (useLogAxis && value >= 10000) return `${(value / 10000).toFixed(0)}万`;
            if (useLogAxis && value >= 1000) return `${(value / 1000).toFixed(0)}k`;
            return value >= 10 ? value.toFixed(0) : value.toFixed(2);
          },
        },
      },
      {
        ...valueAxis,
        name: '仓位 / 回撤 / 日收益',
        position: 'right',
        min: (value) => Math.min(-10, Math.floor(value.min / 10) * 10),
        max: 100,
        axisLabel: { ...valueAxis.axisLabel, formatter: (v) => `${Number(v).toFixed(0)}%` },
      },
    ],
    dataZoom: [
      { type: 'inside', start: zoomStart, end: 100 },
      { type: 'slider', bottom: 4, height: 20, start: zoomStart, end: 100, borderColor: COLORS.grid },
    ],
    series: [
      {
        name: '策略净值', type: 'line', showSymbol: false, data: payload.strategy_nav,
        lineStyle: { width: 2.2 }, z: 5,
        markArea: { silent: true, itemStyle: { color: 'rgba(194,65,61,.07)' }, data: maxDrawdownArea },
      },
      { name: '沪深300', type: 'line', showSymbol: false, lineStyle: { type: 'dashed', width: 1.5 }, data: payload.benchmark_nav || [], z: 4 },
      { name: '回撤', type: 'line', yAxisIndex: 1, showSymbol: false, data: payload.drawdown_pct || [], lineStyle: { color: COLORS.negative, width: 1.4 }, z: 3 },
      {
        name: '实际仓位', type: 'bar', yAxisIndex: 1, data: payload.exposure_pct || [],
        barGap: '-100%', barMaxWidth: 9, z: 0,
        itemStyle: { color: COLORS.teal, opacity: .1 },
      },
      {
        name: '当日调仓资金占比', type: 'line', yAxisIndex: 1,
        showSymbol: false, data: payload.rebalance_funds_pct || [],
        lineStyle: { color: COLORS.turnover, width: 1.3 }, z: 3,
      },
      {
        name: '日收益率', type: 'bar', yAxisIndex: 1, data: payload.daily_returns_pct || [],
        barGap: '-100%', barMaxWidth: 7, z: 1,
        itemStyle: { color: (item) => item.value >= 0 ? COLORS.positive : COLORS.negative, opacity: .35 },
      },
    ],
  });
}

function renderMonthlyHeatmap() {
  const payload = REPORT_DATA.charts.monthly || [];
  const chart = makeChart('monthly-heatmap');
  if (!chart || !payload.length) {
    renderNoData(document.getElementById('monthly-heatmap'), '暂无月度收益数据');
    return;
  }
  const years = [...new Set(payload.map((item) => item.month.slice(0, 4)))].sort().reverse();
  const yearIndex = new Map(years.map((year, index) => [year, index]));
  const values = payload.map((item) => Math.abs(Number(item.monthly_return))).sort((a, b) => a - b);
  const cap = Math.max(1, values[Math.floor(values.length * 0.9)] || values[values.length - 1] || 1);
  const data = payload.map((item) => [Number(item.month.slice(5, 7)) - 1, yearIndex.get(item.month.slice(0, 4)), item.monthly_return]);
  chart.getDom().style.height = `${Math.max(270, years.length * 16 + 90)}px`;
  chart.resize();
  chart.setOption({
    animation: false,
    tooltip: { formatter: (item) => `${years[item.value[1]]}-${String(item.value[0] + 1).padStart(2, '0')}<br><b>${Number(item.value[2]) >= 0 ? '+' : ''}${Number(item.value[2]).toFixed(2)}%</b>` },
    grid: { left: 52, right: 20, top: 8, bottom: 44 },
    xAxis: { type: 'category', data: ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'], splitArea: { show: true }, axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: COLORS.muted, fontSize: 10 } },
    yAxis: { type: 'category', data: years, splitArea: { show: true }, axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: COLORS.muted, fontSize: 10 } },
    visualMap: { min: -cap, max: cap, calculable: false, orient: 'horizontal', left: 'center', bottom: 0, itemWidth: 12, itemHeight: 100, text: ['盈利', '亏损'], textStyle: { color: COLORS.muted, fontSize: 10 }, inRange: { color: ['#B94A48', '#F4F6F8', '#2F855A'] } },
    series: [{ type: 'heatmap', data, label: { show: true, fontSize: 9, formatter: (item) => `${Number(item.value[2]) > 0 ? '+' : ''}${Number(item.value[2]).toFixed(1)}` }, itemStyle: { borderColor: '#FFFFFF', borderWidth: 1 }, emphasis: { itemStyle: { borderColor: COLORS.strategy, borderWidth: 1 } } }],
  });
}

function renderFactorValidChart() {
  const payload = REPORT_DATA.charts.factor_valid;
  const chart = makeChart('factor-valid-chart');
  const series = Object.entries(payload?.series || {});
  if (!chart || !payload?.trade_dates?.length || !series.length) {
    renderNoData(document.getElementById('factor-valid-chart'), '暂无因子有效值数据');
    return;
  }

  const zoomStart = 0;

  chart.setOption({
    animation: false,
    color: ['#2563EB', '#C2413D', '#15803D', '#B45309', '#7C3AED', '#0F766E'],
    legend: { top: 0, type: 'scroll', textStyle: { color: COLORS.muted, fontSize: 10 } },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    grid: { left: 58, right: 22, top: 48, bottom: 48 },
    xAxis: { type: 'category', data: payload.trade_dates, boundaryGap: false, axisLabel: { hideOverlap: true, color: COLORS.muted, fontSize: 10 }, axisLine: { lineStyle: { color: COLORS.grid } } },
    yAxis: { type: 'value', name: '有效股票数', min: (value) => Math.max(0, Math.floor(value.min / 500) * 500), max: payload.stock_pool_size || null, axisLabel: { color: COLORS.muted, fontSize: 10 }, splitLine: { lineStyle: { color: COLORS.grid } } },
    dataZoom: [
      { type: 'inside', start: zoomStart, end: 100 },
      { type: 'slider', bottom: 4, height: 18, start: zoomStart, end: 100, borderColor: COLORS.grid },
    ],
    series: series.map(([name, data]) => ({
      name,
      type: 'line',
      showSymbol: false,
      data,
    })),
  });
}

function renderDistributionChart() {
  const payload = REPORT_DATA.charts.distribution;
  const chart = makeChart('distribution-chart');
  if (!chart || !payload || !payload.counts?.length) {
    renderNoData(document.getElementById('distribution-chart'), '暂无收益率分布数据');
    return;
  }

  chart.setOption({
    animation: false,
    tooltip: { trigger: 'axis' },
    title: { text: `均值 ${payload.mean.toFixed(3)}%   σ ${payload.std.toFixed(3)}%`, left: 8, top: 0, textStyle: { fontSize: 11, fontWeight: 500, color: COLORS.muted } },
    grid: { left: 44, right: 14, top: 38, bottom: 48 },
    xAxis: {
      type: 'category',
      data: payload.labels,
      axisLabel: {
        rotate: 35,
        fontSize: 10,
        interval: Math.max(0, Math.floor(payload.labels.length / 12)),
      },
    },
    yAxis: { type: 'value', name: '频次', axisLabel: { color: COLORS.muted, fontSize: 10 }, splitLine: { lineStyle: { color: COLORS.grid } } },
    series: [{ type: 'bar', data: payload.counts, itemStyle: { color: (item) => parseFloat(payload.labels[item.dataIndex]) >= 0 ? COLORS.positive : COLORS.negative, opacity: .82 }, barMaxWidth: 28 }],
  });
}

function renderWinLossChart() {
  const payload = REPORT_DATA.charts.winloss;
  const chart = makeChart('winloss-chart');
  if (!chart || !payload || (payload.wins || 0) + (payload.losses || 0) === 0) {
    renderNoData(document.getElementById('winloss-chart'), '暂无清仓盈亏数据');
    return;
  }

  const total = payload.wins + payload.losses;
  const winRate = total ? payload.wins / total * 100 : 0;
  chart.setOption({
    animation: false,
    tooltip: { trigger: 'item' },
    title: { text: `${winRate.toFixed(1)}%`, subtext: '清仓胜率', left: 'center', top: '37%', textStyle: { fontSize: 24, fontWeight: 700, color: COLORS.positive }, subtextStyle: { fontSize: 11, color: COLORS.muted } },
    legend: { bottom: 0, textStyle: { color: COLORS.muted } },
    series: [{
      type: 'pie',
      radius: ['48%', '72%'], center: ['50%', '45%'],
      label: { formatter: '{b}  {c}\n{d}%', color: COLORS.muted, fontSize: 11 },
      data: [
        { name: '盈利清仓', value: payload.wins, itemStyle: { color: COLORS.positive } },
        { name: '亏损清仓', value: payload.losses, itemStyle: { color: COLORS.negative } },
      ],
    }],
  });
}

let KLINE_CACHE = null;

async function loadKlineData() {
  if (KLINE_CACHE) return KLINE_CACHE;
  const encoded = REPORT_DATA.kline_b64;
  if (!encoded) return {};
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('当前浏览器不支持 K 线数据解压');
  }
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  KLINE_CACHE = JSON.parse(await new Response(stream).text());
  REPORT_DATA.kline_b64 = '';
  return KLINE_CACHE;
}

function movingAverage(values, windowSize) {
  let sum = 0;
  return values.map((value, index) => {
    sum += Number(value);
    if (index >= windowSize) sum -= Number(values[index - windowSize]);
    return index + 1 < windowSize ? null : Number((sum / windowSize).toFixed(4));
  });
}

function closeKline() {
  document.getElementById('klineOverlay')?.classList.remove('active');
  document.getElementById('klinePanel')?.classList.remove('active');
}

async function showKline(code, eventId) {
  const overlay = document.getElementById('klineOverlay');
  const panel = document.getElementById('klinePanel');
  const placeholder = document.getElementById('klinePlaceholder');
  const renderArea = document.getElementById('klineRenderArea');
  if (!overlay || !panel || !placeholder || !renderArea) return;
  overlay.classList.add('active');
  panel.classList.add('active');
  placeholder.style.display = 'flex';
  placeholder.textContent = '正在加载 K 线数据';
  renderArea.style.display = 'none';

  try {
    const allData = await loadKlineData();
    const stock = allData[code];
    if (!stock) throw new Error('该股票没有可用的本地日线数据');
    const selectedEvent = stock.events.find((event) => Number(event.id) === Number(eventId));
    if (!selectedEvent) throw new Error('未找到对应交易事件');
    const episode = stock.episodes.find((item) => Number(item.id) === Number(selectedEvent.episode));
    if (!episode) throw new Error('未找到对应持仓周期');

    const indices = [];
    stock.d.forEach((date, index) => {
      if (date >= episode.window_start && date <= episode.window_end) indices.push(index);
    });
    if (indices.length < 5) throw new Error('该持仓周期的 K 线数据不足');
    const pick = (values) => indices.map((index) => values[index]);
    const dates = pick(stock.d);
    const opens = pick(stock.o);
    const highs = pick(stock.h);
    const lows = pick(stock.l);
    const closes = pick(stock.c);
    const amounts = pick(stock.a);
    const episodeEvents = stock.events.filter((event) => Number(event.episode) === Number(episode.id));
    const buyEvents = episodeEvents.filter((event) => event.action === 'buy');
    const sellEvents = episodeEvents.filter((event) => event.action === 'sell');

    document.getElementById('klineTitle').textContent = `${code} ${stock.n || ''}`.trim();
    document.getElementById('klineMeta').textContent =
      `持仓周期 ${episode.start} → ${episode.open ? '仍持有' : episode.end} · K线 ${episode.window_start} → ${episode.window_end}`;
    placeholder.style.display = 'none';
    renderArea.style.display = 'block';
    if (window._klineChart) window._klineChart.dispose();
    const chart = echarts.init(renderArea);
    window._klineChart = chart;
    const selectedDate = selectedEvent.date;
    chart.setOption({
      animation: false,
      legend: { top: 0, right: 12, data: ['日K', 'MA20', 'MA60', '买入', '卖出'], textStyle: { color: COLORS.muted } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      axisPointer: { link: [{ xAxisIndex: [0, 1] }] },
      grid: [
        { left: 66, right: 24, top: 42, height: '66%' },
        { left: 66, right: 24, top: '76%', height: '13%' },
      ],
      xAxis: [
        { type: 'category', data: dates, boundaryGap: true, axisLabel: { show: false }, axisLine: { lineStyle: { color: COLORS.grid } } },
        { type: 'category', gridIndex: 1, data: dates, boundaryGap: true, axisLabel: { hideOverlap: true, color: COLORS.muted, fontSize: 10 }, axisLine: { lineStyle: { color: COLORS.grid } } },
      ],
      yAxis: [
        { type: 'value', scale: true, name: '价格', axisLabel: { color: COLORS.muted }, splitLine: { lineStyle: { color: COLORS.grid } } },
        { type: 'value', gridIndex: 1, name: '成交额', axisLabel: { color: COLORS.muted, formatter: (value) => value >= 1e8 ? `${(value / 1e8).toFixed(1)}亿` : `${(value / 1e4).toFixed(0)}万` }, splitLine: { show: false } },
      ],
      dataZoom: [
        { type: 'inside', xAxisIndex: [0, 1], start: 0, end: 100 },
        { type: 'slider', xAxisIndex: [0, 1], bottom: 4, height: 20, start: 0, end: 100 },
      ],
      series: [
        {
          name: '日K', type: 'candlestick', data: dates.map((_, index) => [opens[index], closes[index], lows[index], highs[index]]),
          itemStyle: { color: COLORS.negative, color0: COLORS.positive, borderColor: COLORS.negative, borderColor0: COLORS.positive },
          markLine: { silent: true, symbol: 'none', label: { formatter: '当前交易', color: COLORS.muted }, lineStyle: { color: COLORS.strategy, type: 'dashed' }, data: [{ xAxis: selectedDate }] },
        },
        { name: 'MA20', type: 'line', data: movingAverage(closes, 20), showSymbol: false, smooth: false, lineStyle: { width: 1.2, color: COLORS.benchmark } },
        { name: 'MA60', type: 'line', data: movingAverage(closes, 60), showSymbol: false, smooth: false, lineStyle: { width: 1.2, color: '#7C3AED' } },
        {
          name: '买入', type: 'scatter', symbol: 'triangle', symbolSize: (value, params) => Number(params.data.eventId) === Number(eventId) ? 18 : 13,
          data: buyEvents.map((event) => ({ value: [event.date, event.price], eventId: event.id, itemStyle: { color: COLORS.strategy, borderColor: '#FFFFFF', borderWidth: 1 } })),
          tooltip: { valueFormatter: (value) => Number(value).toFixed(4) }, z: 10,
        },
        {
          name: '卖出', type: 'scatter', symbol: 'triangle', symbolRotate: 180, symbolSize: (value, params) => Number(params.data.eventId) === Number(eventId) ? 18 : 13,
          data: sellEvents.map((event) => ({ value: [event.date, event.price], eventId: event.id, itemStyle: { color: '#7C3AED', borderColor: '#FFFFFF', borderWidth: 1 } })),
          tooltip: { valueFormatter: (value) => Number(value).toFixed(4) }, z: 10,
        },
        {
          name: '成交额', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: amounts,
          itemStyle: { color: (item) => closes[item.dataIndex] >= opens[item.dataIndex] ? COLORS.negative : COLORS.positive, opacity: .55 },
        },
      ],
    });
  } catch (error) {
    placeholder.style.display = 'flex';
    placeholder.textContent = error?.message || 'K 线加载失败';
  }
}

document.addEventListener('click', (event) => {
  const button = event.target.closest('[data-kline-code][data-kline-event]');
  if (button) showKline(button.dataset.klineCode, button.dataset.klineEvent);
});
document.getElementById('klineClose')?.addEventListener('click', closeKline);
document.getElementById('klineOverlay')?.addEventListener('click', closeKline);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeKline();
});

function parseSortValue(value, sortType) {
  if (sortType === 'number') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
  }
  if (sortType === 'date') {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
  }
  return (value || '').toString().toLowerCase();
}

async function loadTanStack() {
  const urls = [
    'https://cdn.jsdelivr.net/npm/@tanstack/table-core@8.21.3/+esm',
    'https://unpkg.com/@tanstack/table-core@8.21.3/build/lib/index.mjs',
  ];

  let lastError = null;
  for (const url of urls) {
    try {
      return await import(url);
    } catch (err) {
      lastError = err;
      console.error(`TanStack import failed: ${url}`, err);
    }
  }
  throw lastError || new Error('TanStack import failed');
}

class VirtualTable {
  constructor(root, payload, tanstack) {
    this.root = root;
    this.payload = payload;
    this.tanstack = tanstack;
    this.sorting = [];
    this.rowHeight = payload.row_height || 44;
    this.maxHeight = payload.max_height || 460;
    this.headers = payload.headers || [];
    this.totalSize = this.headers.reduce((sum, header) => sum + (header.size || 180), 0) || 1;
    this.data = (payload.rows || []).map((row) => {
      const out = {};
      this.headers.forEach((header, idx) => {
        out[header.id] = row[idx] || { html: '', sort: '', class: '', title: '' };
      });
      return out;
    });
    this.columns = this.headers.map((header) => ({
      id: header.id,
      header: header.label,
      accessorFn: (row) => parseSortValue(row[header.id]?.sort, header.sort_type),
      cell: (info) => info.row.original[header.id],
      enableSorting: header.sortable !== false,
      meta: { align: header.align || 'left' },
    }));

    this.buildDom();
    this.buildTable();
    this.renderHeader();
    this.renderBody();
    this.bodyWrap.addEventListener('scroll', () => this.renderBody());
    TABLE_INSTANCES.push(this);
  }

  buildDom() {
    this.root.innerHTML = '';
    this.shell = document.createElement('div');
    this.shell.className = 'tt-shell';

    this.headWrap = document.createElement('div');
    this.headWrap.className = 'tt-head-wrap';
    this.headTable = document.createElement('table');
    this.headTable.className = 'tt';
    this.headColgroup = document.createElement('colgroup');
    this.head = document.createElement('thead');
    this.headTable.append(this.headColgroup, this.head);
    this.headWrap.appendChild(this.headTable);

    this.bodyWrap = document.createElement('div');
    this.bodyWrap.className = 'tt-body-wrap';
    this.bodyWrap.style.maxHeight = `${this.maxHeight}px`;
    this.bodyTable = document.createElement('table');
    this.bodyTable.className = 'tt';
    this.bodyColgroup = document.createElement('colgroup');
    this.body = document.createElement('tbody');
    this.bodyTable.append(this.bodyColgroup, this.body);
    this.bodyWrap.appendChild(this.bodyTable);

    this.shell.append(this.headWrap, this.bodyWrap);
    this.root.appendChild(this.shell);
    this.renderColgroups();
  }

  buildTable() {
    const { createTable, getCoreRowModel, getSortedRowModel } = this.tanstack;
    this.table = createTable({
      data: this.data,
      columns: this.columns,
      state: {
        sorting: this.sorting,
        columnPinning: { left: [], right: [] },
      },
      onSortingChange: (updater) => {
        this.sorting = this.tanstack.functionalUpdate(updater, this.sorting);
        this.buildTable();
        this.renderHeader();
        this.renderBody(true);
      },
      getCoreRowModel: getCoreRowModel(),
      getSortedRowModel: getSortedRowModel(),
    });
  }

  renderColgroups() {
    const html = this.headers.map((header) => {
      const width = ((header.size || 180) / this.totalSize) * 100;
      return `<col style="width:${width}%">`;
    }).join('');
    this.headColgroup.innerHTML = html;
    this.bodyColgroup.innerHTML = html;
  }

  syncHeaderPadding() {
    const scrollbarWidth = Math.max(0, this.bodyWrap.offsetWidth - this.bodyWrap.clientWidth);
    this.headWrap.style.paddingRight = `${scrollbarWidth}px`;
  }

  renderHeader() {
    this.head.innerHTML = '';
    this.table.getHeaderGroups().forEach((headerGroup) => {
      const tr = document.createElement('tr');
      headerGroup.headers.forEach((header) => {
        const th = document.createElement('th');
        const align = header.column.columnDef.meta?.align;
        if (align === 'right') th.classList.add('align-right');
        if (align === 'center') th.classList.add('align-center');
        th.textContent = header.isPlaceholder ? '' : String(header.column.columnDef.header ?? '');
        if (header.column.getCanSort()) {
          th.classList.add('sortable');
          const sorted = header.column.getIsSorted();
          if (sorted === 'asc') th.classList.add('sort-asc');
          if (sorted === 'desc') th.classList.add('sort-desc');
          const toggle = header.column.getToggleSortingHandler?.()
            || (() => header.column.toggleSorting(header.column.getIsSorted() === 'asc'));
          th.addEventListener('click', toggle);
        }
        tr.appendChild(th);
      });
      this.head.appendChild(tr);
    });
    this.syncHeaderPadding();
  }

  renderBody(resetScroll = false) {
    if (resetScroll) this.bodyWrap.scrollTop = 0;

    const rows = this.table.getRowModel().rows;
    const visibleHeight = this.bodyWrap.clientHeight || this.maxHeight;
    const start = Math.max(0, Math.floor(this.bodyWrap.scrollTop / this.rowHeight) - 8);
    const end = Math.min(rows.length, Math.ceil((this.bodyWrap.scrollTop + visibleHeight) / this.rowHeight) + 8);
    const topPad = start * this.rowHeight;
    const bottomPad = Math.max(0, (rows.length - end) * this.rowHeight);

    this.body.innerHTML = '';
    if (topPad > 0) {
      const tr = document.createElement('tr');
      tr.className = 'spacer';
      const td = document.createElement('td');
      td.colSpan = this.headers.length;
      td.style.height = `${topPad}px`;
      tr.appendChild(td);
      this.body.appendChild(tr);
    }

    rows.slice(start, end).forEach((row) => {
      const tr = document.createElement('tr');
      tr.style.height = `${this.rowHeight}px`;
      row.getVisibleCells().forEach((cell) => {
        const td = document.createElement('td');
        const align = cell.column.columnDef.meta?.align;
        if (align === 'right') td.classList.add('align-right');
        if (align === 'center') td.classList.add('align-center');
        const payload = row.original[cell.column.id] || { html: '', class: '', title: '' };
        if (payload.class) td.className = `${td.className} ${payload.class}`.trim();
        const wrapper = document.createElement('div');
        wrapper.className = 'cell';
        wrapper.innerHTML = payload.html || '';
        td.appendChild(wrapper);
        tr.appendChild(td);
      });
      this.body.appendChild(tr);
    });

    if (bottomPad > 0) {
      const tr = document.createElement('tr');
      tr.className = 'spacer';
      const td = document.createElement('td');
      td.colSpan = this.headers.length;
      td.style.height = `${bottomPad}px`;
      tr.appendChild(td);
      this.body.appendChild(tr);
    }

    this.syncHeaderPadding();
    initTooltips(this.root);
  }
}

function mountTable(hostId, payload, tanstack) {
  const root = document.getElementById(hostId);
  if (!root) return;
  if (!payload || !payload.rows || payload.rows.length === 0) {
    renderNoData(root, payload?.empty_message || '暂无数据');
    return;
  }
  new VirtualTable(root, payload, tanstack);
}

async function initTables() {
  const mounts = [
    ['monthly-host', 'monthly'],
    ['trade-host', 'trades'],
    ['holdings-host', 'holdings'],
    ['cleared-host', 'cleared'],
    ['daily-host', 'daily'],
    ['delist-host', 'delist'],
  ];
  try {
    const tanstack = await loadTanStack();
    mounts.forEach(([hostId, key]) => mountTable(hostId, REPORT_DATA.tables[key], tanstack));
  } catch (err) {
    console.error('TanStack table bootstrap failed', err);
    mounts.forEach(([hostId]) => renderNoData(document.getElementById(hostId), 'TanStack Table CDN 加载失败'));
  }
}

function initTabs() {
  const buttons = [...document.querySelectorAll('.tab-button[data-tab]')];
  const panels = [...document.querySelectorAll('.tab-panel')];
  const activate = (button) => {
    const panelId = `panel-${button.dataset.tab}`;
    buttons.forEach((item) => {
      const selected = item === button;
      item.classList.toggle('is-active', selected);
      item.setAttribute('aria-selected', String(selected));
    });
    panels.forEach((panel) => {
      const selected = panel.id === panelId;
      panel.hidden = !selected;
      panel.classList.toggle('is-active', selected);
    });
    requestAnimationFrame(() => {
      TABLE_INSTANCES.forEach((table) => {
        if (!table.root.closest('[hidden]')) {
          table.syncHeaderPadding();
          table.renderBody();
        }
      });
    });
  };
  buttons.forEach((button, index) => {
    button.addEventListener('click', () => activate(button));
    button.addEventListener('keydown', (event) => {
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      const offset = event.key === 'ArrowRight' ? 1 : -1;
      const next = buttons[(index + offset + buttons.length) % buttons.length];
      next.focus();
      activate(next);
    });
  });
}

renderPerformanceCharts();
renderMonthlyHeatmap();
renderFactorValidChart();
renderDistributionChart();
renderWinLossChart();
initTabs();
initTables();
initTooltips(document);

window.addEventListener('resize', () => {
  CHARTS.forEach((chart) => chart.resize());
  TABLE_INSTANCES.forEach((table) => table.syncHeaderPadding());
  if (window._klineChart) window._klineChart.resize();
});
