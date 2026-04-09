const REPORT_DATA = JSON.parse(document.getElementById('report-data').textContent);
const CHARTS = [];
const TOOLTIP_INSTANCES = [];
const TABLE_INSTANCES = [];

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

function renderEquityChart() {
  const payload = REPORT_DATA.charts.equity;
  const chart = makeChart('equity-chart');
  if (!chart || !payload || !payload.trade_dates?.length) {
    renderNoData(document.getElementById('equity-chart'), '暂无净值数据');
    return;
  }

  const recentWindow = Math.min(payload.trade_dates.length, 240);
  const zoomStart = payload.trade_dates.length > recentWindow
    ? Math.max(0, (payload.trade_dates.length - recentWindow) / payload.trade_dates.length * 100)
    : 0;

  chart.setOption({
    animation: false,
    color: ['#1976D2', '#FF9800', '#42A5F5'],
    legend: { top: 0 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    axisPointer: { link: [{ xAxisIndex: [0, 1] }] },
    grid: [
      { left: 60, right: 32, top: 44, height: '58%' },
      { left: 60, right: 32, top: '76%', height: '14%' },
    ],
    xAxis: [
      { type: 'category', data: payload.trade_dates, boundaryGap: false, axisLabel: { hideOverlap: true } },
      { type: 'category', gridIndex: 1, data: payload.trade_dates, boundaryGap: false, axisLabel: { hideOverlap: true } },
    ],
    yAxis: [
      { type: 'value', name: '净值', scale: true, axisLabel: { formatter: (v) => v.toFixed(2) } },
      { type: 'value', gridIndex: 1, name: '日收益率 (%)', scale: true, axisLabel: { formatter: (v) => `${v.toFixed(2)}%` } },
    ],
    dataZoom: [
      { type: 'inside', xAxisIndex: [0, 1], start: zoomStart, end: 100 },
      { type: 'slider', xAxisIndex: [0, 1], top: '92%', start: zoomStart, end: 100 },
    ],
    series: [
      { name: '策略净值', type: 'line', showSymbol: false, data: payload.strategy_nav },
      { name: '沪深300净值', type: 'line', showSymbol: false, lineStyle: { type: 'dashed' }, data: payload.benchmark_nav || [] },
      {
        name: '日收益率',
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: payload.daily_returns_pct,
        tooltip: { valueFormatter: (v) => `${Number(v).toFixed(2)}%` },
        itemStyle: { color: (params) => params.value >= 0 ? '#2E7D32' : '#C62828' },
      },
    ],
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
    title: {
      text: `均值 ${payload.mean.toFixed(3)}% · σ ${payload.std.toFixed(3)}%`,
      left: 'center',
      top: 8,
      textStyle: { fontSize: 13, fontWeight: 500 },
    },
    grid: { left: 48, right: 18, top: 48, bottom: 54 },
    xAxis: {
      type: 'category',
      data: payload.labels,
      axisLabel: {
        rotate: 35,
        fontSize: 10,
        interval: Math.max(0, Math.floor(payload.labels.length / 12)),
      },
    },
    yAxis: { type: 'value', name: '频次' },
    series: [{ type: 'bar', data: payload.counts, itemStyle: { color: '#42A5F5' }, barMaxWidth: 28 }],
  });
}

function renderWinLossChart() {
  const payload = REPORT_DATA.charts.winloss;
  const chart = makeChart('winloss-chart');
  if (!chart || !payload || (payload.wins || 0) + (payload.losses || 0) === 0) {
    renderNoData(document.getElementById('winloss-chart'), '暂无清仓盈亏数据');
    return;
  }

  chart.setOption({
    animation: false,
    tooltip: { trigger: 'item' },
    legend: { bottom: 0 },
    series: [{
      type: 'pie',
      radius: ['42%', '68%'],
      label: { formatter: '{b}\n{d}%' },
      data: [
        { name: '盈利清仓', value: payload.wins, itemStyle: { color: '#2E7D32' } },
        { name: '亏损清仓', value: payload.losses, itemStyle: { color: '#C62828' } },
      ],
    }],
  });
}

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

renderEquityChart();
renderDistributionChart();
renderWinLossChart();
initTables();
initTooltips(document);

window.addEventListener('resize', () => {
  CHARTS.forEach((chart) => chart.resize());
  TABLE_INSTANCES.forEach((table) => table.syncHeaderPadding());
  if (window._klineChart) window._klineChart.resize();
});
