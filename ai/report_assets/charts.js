// Shared renderer for evaluations, diagnostics and daily weights.
// Missing observations are omitted; connecting saved points does not interpolate data.
export const palette = ['#285fcb', '#13806a', '#bb741f', '#8d57b6', '#d24f78', '#387f9c', '#756c28', '#966253', '#6b7fbc', '#488c55', '#bd7047', '#74668f'];
export const number = (value, digits = 3) => Number.isFinite(value)
  ? value.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—';
export const escapeHTML = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

export class LineChart {
  constructor(container) {
    this.hiddenSeries = new Set();
    this.series = [];
    this.references = [];
    this.options = {};
    this.el = document.createElement('article');
    this.el.className = 'panel chart';
    this.el.innerHTML = '<div class="chart-heading"><h3></h3><p></p></div><div class="legend"></div><div class="chart-wrap"><canvas></canvas><div class="chart-empty">暂无已保存的有限数据</div></div>';
    container.appendChild(this.el);
    this.canvas = this.el.querySelector('canvas');
    this.legend = this.el.querySelector('.legend');
    this.tooltip = document.getElementById('chart-tooltip');
    this.canvas.addEventListener('mousemove', event => this.hover(event));
    this.canvas.addEventListener('mouseleave', () => { this.tooltip.hidden = true; });
    this.legend.addEventListener('click', event => {
      const button = event.target.closest('button[data-series]');
      if (!button) return;
      const key = this.series[Number(button.dataset.series)].label;
      this.hiddenSeries.has(key) ? this.hiddenSeries.delete(key) : this.hiddenSeries.add(key);
      button.setAttribute('aria-pressed', String(!this.hiddenSeries.has(key)));
      this.draw();
    });
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(this.canvas.parentElement);
  }

  set(series, options = {}, references = []) {
    this.options = options;
    this.series = series.map(item => ({ ...item, points: item.points.filter(point => Number.isFinite(point[0]) && Number.isFinite(point[1])).sort((a, b) => a[0] - b[0]) }));
    this.references = references.filter(item => Number.isFinite(item.value));
    this.el.querySelector('h3').textContent = options.title;
    this.el.querySelector('.chart-heading p').textContent = options.subtitle || '';
    this.canvas.setAttribute('aria-label', options.title);
    this.legend.replaceChildren();
    this.series.forEach((item, index) => {
      const button = document.createElement('button');
      button.dataset.series = index;
      button.setAttribute('aria-pressed', String(!this.hiddenSeries.has(item.label)));
      const dot = document.createElement('i');
      dot.style.background = item.color;
      button.append(dot, document.createTextNode(item.label));
      this.legend.appendChild(button);
    });
    this.references.forEach(item => {
      const label = document.createElement('span');
      const dot = document.createElement('i');
      dot.className = 'dashed';
      dot.style.borderColor = item.color;
      label.append(dot, document.createTextNode(`${item.label} ${this.formatY(item.value)}`));
      this.legend.appendChild(label);
    });
    this.draw();
  }

  formatY(value) { return this.options.formatY ? this.options.formatY(value) : number(value, 3); }
  formatX(value) { return this.options.formatX ? this.options.formatX(value) : Math.round(value).toLocaleString('zh-CN'); }

  draw() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = rect.width * ratio;
    this.canvas.height = rect.height * ratio;
    const ctx = this.canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    const visible = this.series.filter(item => !this.hiddenSeries.has(item.label));
    const bounds = [Infinity, -Infinity, Infinity, -Infinity];
    for (const item of visible) for (const [x, y] of item.points) {
      bounds[0] = Math.min(bounds[0], x); bounds[1] = Math.max(bounds[1], x);
      bounds[2] = Math.min(bounds[2], y); bounds[3] = Math.max(bounds[3], y);
    }
    const empty = !Number.isFinite(bounds[0]);
    this.el.querySelector('.chart-empty').hidden = !empty;
    this.geometry = null;
    if (empty) return;
    let [xmin, xmax, ymin, ymax] = bounds;
    for (const item of this.references) { ymin = Math.min(ymin, item.value); ymax = Math.max(ymax, item.value); }
    if (xmin === xmax) { xmin -= 1; xmax += 1; }
    if (ymin === ymax) { ymin -= .5; ymax += .5; }
    const margin = (ymax - ymin) * .1;
    ymin -= margin; ymax += margin;
    const pad = { left: 66, right: 18, top: 16, bottom: 48 };
    const width = rect.width - pad.left - pad.right, height = rect.height - pad.top - pad.bottom;
    const sx = x => pad.left + (x - xmin) / (xmax - xmin) * width;
    const sy = y => pad.top + (ymax - y) / (ymax - ymin) * height;
    this.geometry = { xmin, xmax, pad, width, visible };
    ctx.font = '12px system-ui';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + height * i / 4;
      ctx.strokeStyle = '#e7ecf3'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + width, y); ctx.stroke();
      ctx.fillStyle = '#68768a'; ctx.textAlign = 'right';
      ctx.fillText(this.formatY(ymax - (ymax - ymin) * i / 4), pad.left - 9, y);
      ctx.textAlign = i === 0 ? 'left' : i === 4 ? 'right' : 'center';
      ctx.fillText(this.formatX(xmin + (xmax - xmin) * i / 4), pad.left + width * i / 4, pad.top + height + 19);
    }
    for (const item of this.references) {
      ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = item.color; ctx.globalAlpha = .65;
      ctx.beginPath(); ctx.moveTo(pad.left, sy(item.value)); ctx.lineTo(pad.left + width, sy(item.value)); ctx.stroke(); ctx.restore();
    }
    for (const item of visible) {
      ctx.strokeStyle = item.color; ctx.fillStyle = item.color; ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.beginPath();
      item.points.forEach(([x, y], index) => { index ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y)); });
      ctx.stroke();
      if (item.markers || item.points.length < 80) for (const [x, y] of item.points) {
        ctx.beginPath(); ctx.arc(sx(x), sy(y), 2.4, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.fillStyle = '#68768a'; ctx.textAlign = 'center';
    ctx.fillText(this.options.xLabel || '', pad.left + width / 2, rect.height - 6);
  }

  hover(event) {
    if (!this.geometry) return;
    const { xmin, xmax, pad, width, visible } = this.geometry;
    const x = event.clientX - this.canvas.getBoundingClientRect().left;
    if (x < pad.left || x > pad.left + width) { this.tooltip.hidden = true; return; }
    const target = xmin + (x - pad.left) / width * (xmax - xmin);
    this.tooltip.textContent = visible.filter(item => item.points.length).map(item => {
      const point = item.points.reduce((a, b) => Math.abs(b[0] - target) < Math.abs(a[0] - target) ? b : a);
      return `${item.label} · ${this.formatX(point[0])}${this.options.unit || ''}：${this.formatY(point[1])}`;
    }).join('\n');
    this.tooltip.hidden = !this.tooltip.textContent;
    this.tooltip.style.left = `${Math.max(8, Math.min(window.innerWidth - this.tooltip.offsetWidth - 12, event.clientX + 14))}px`;
    this.tooltip.style.top = `${Math.max(8, event.clientY - this.tooltip.offsetHeight - 12)}px`;
  }

  destroy() { this.observer.disconnect(); this.el.remove(); this.tooltip.hidden = true; }
}
