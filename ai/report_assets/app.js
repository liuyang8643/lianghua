import { LineChart, palette, number, escapeHTML as esc } from './charts.js';

const $ = id => document.getElementById(id);
const splits = { train: '训练', validation: '验证', test: '测试' };
const metrics = { calmar: 'Calmar', annualized_return: '年化净收益', max_drawdown: '最大回撤', sharpe: 'Sharpe' };
const labels = {
  selection: '正式选择规则', test_role: '测试用途', evaluation_interval: '评估间隔',
  train: '训练期', validation: '验证期', test: '测试期',
  weights: '因子权重', buy_n: '目标持股数', turnover_rate: '换股比例', sell_m: '旧版卖出排名阈值',
  single_buy_pct: '单股买入比例', prefilter_n: '候选池大小',
  learning_rate: '学习率', n_epochs: '每批重复更新次数', n_steps: '单环境采集步数',
  batch_size: '更新批量', workers: '并行环境数', population_size: '种群数量',
  evaluation_every_generations: '每隔多少代评估', complete_train_evaluation_every_rollouts: '完整训练期评估间隔',
  eval_every_generations: '每隔多少代评估', population: '种群数量', generations: '目标代数',
  objective: '训练目标', deployment_bundle: '已生成实盘模型', elapsed_scope: '计时范围',
  factor_enabled: '启用的因子', filter_factors: '固定过滤器', limit_up_protection: '涨停保护', rebalance_band_pct: '调仓带宽',
  seed: '随机种子', elapsed_seconds: '耗时（秒）', unique_candidates: '已评估不同配置数'
};
const roles = {
  formal: '正式评估', eligible: '可参与正式选择', selection: '正式选模', selected: '正式选中',
  diagnostic: '仅诊断', historical_diagnostic: '历史诊断', repeated_diagnostic: '重复诊断',
  training: '训练记录', train: '训练记录', validation: '验证评估', test: '测试评估',
  historical_evaluations: '历史评估记录', test_diagnostic: '测试诊断', diagnostic_only: '仅诊断',
  validation_calmar: '按验证期 Calmar', validation_only: '仅使用验证集',
  training_champion: '当代训练冠军', official_validation: '正式验证', historical: '历史记录',
  formal_validation: '正式验证', training_evaluation: '训练评估', generation_champion: '当代训练冠军', diagnostic_test: '测试诊断',
  historical_diagnostic_validation: '历史验证诊断', unlinked_final_evaluation: '最终评估（模型关联未确认）',
  historical_diagnostic_only_not_selection: '历史诊断', retrospective_validation_diagnostic_not_formal_selection: '补充验证诊断',
  diagnostic_only_not_blind_test: '测试诊断（已非盲测）', repeated_diagnostic_only: '反复诊断，不用于选择模型',
  repeated_diagnostic_only_never_selection: '反复诊断，不用于选择模型', maximum_validation_calmar: '按正式验证期 Calmar 选择',
  maximum_validation_calmar_of_generation_champions: '按规定评估点中训练冠军的验证 Calmar 选择',
  complete_train_calmar: '完整训练期 Calmar',
  recorded_collection_update_and_checkpoint_seconds_excludes_evaluation_and_startup: '累计已记录的采集、更新与保存时间；不含启动和评估'
};
const charts = new Map();
let snapshot = null, selectedId = new URLSearchParams(location.search).get('run'), activeTab = 'overview', paused = false, refreshing = false;
let evaluationLimit = 100, checkpointRun = null, checkpointState = null, traceLoading = false;

function currentRun() { return snapshot.runs.find(run => run.id === selectedId); }
function splitLabel(run, split) { return split === 'test' && String(run.protocol.test_role).includes('diagnostic') ? '测试 · 仅诊断' : splits[split]; }
function percent(value) { return Number.isFinite(value) ? `${number(value * 100, 2)}%` : '—'; }
function formatMetric(value, metric) { return ['annualized_return', 'max_drawdown'].includes(metric) ? percent(value) : number(value, 4); }
function progressLabel(run, step) { return Number.isFinite(step) ? `${number(step, 0)} ${run.progress.unit}` : '—'; }
function duration(seconds) {
  if (!Number.isFinite(seconds)) return '耗时未记录';
  return seconds < 60 ? `${number(seconds, 1)} 秒` : seconds < 3600 ? `${number(seconds / 60, 1)} 分钟` : `${number(seconds / 3600, 2)} 小时`;
}
function valueText(value) {
  if (value == null) return '—';
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (typeof value === 'object') return JSON.stringify(value, null, 2);
  return roles[value] || String(value);
}
function keyValues(value) {
  const entries = Object.entries(value || {});
  return entries.length ? `<dl class="key-values">${entries.map(([key, item]) => `<dt>${esc(labels[key] || key)}</dt><dd>${esc(valueText(item))}</dd>`).join('')}</dl>` : '<p class="muted">尚无已保存记录。</p>';
}
function chart(key, container, series, options, references = []) {
  if (!charts.has(key)) charts.set(key, new LineChart($(container)));
  charts.get(key).set(series, options, references);
}
function pruneCharts(prefix, keys) {
  for (const [key, item] of charts) if (key.startsWith(prefix) && !keys.includes(key)) { item.destroy(); charts.delete(key); }
}
function resetTrace() {
  checkpointRun = null;
  checkpointState = null;
  $('checkpoint').replaceChildren();
  $('trace-status').textContent = '';
  pruneCharts('trace:', []);
}

function renderCards() {
  $('run-cards').innerHTML = snapshot.runs.map(run => {
    const p = run.progress, selection = run.selection;
    const elapsedLabel = Number.isFinite(p.wall_elapsed_seconds) ? '本次累计运行耗时（含准备与评估）' : run.protocol.elapsed_scope === 'recorded_collection_update_and_checkpoint_seconds_excludes_evaluation_and_startup'
      ? '采集、更新与保存（不含评估）' : run.protocol.elapsed_scope ? '已记录耗时' : '累计运行耗时';
    const rate = Number.isFinite(p.current) && Number.isFinite(p.total) && p.total > 0 ? Math.max(0, Math.min(100, 100 * p.current / p.total)) : null;
    return `<article class="run-card ${run.id === selectedId ? 'selected' : ''}">
      <div class="run-title"><h2>${esc(run.algorithm)}</h2><span class="state">${esc(run.state_label)}</span></div>
      <p class="run-label">${esc(run.label)}</p>
      <div class="run-progress">${number(p.current, 0)} <small>/ ${number(p.total, 0)} ${esc(p.unit)}</small></div>
      <div class="progress" ${rate == null ? 'hidden' : ''}><div style="width:${rate ?? 0}%"></div></div>
      <div class="run-meta"><span>${esc(elapsedLabel)}：${esc(duration(p.wall_elapsed_seconds ?? p.elapsed_seconds))}</span><span>${Number.isFinite(p.timesteps) ? `${number(p.timesteps, 0)} 个采集步` : ''}</span></div>
      <div class="run-metrics">${Object.keys(splits).map(split => `<div><span>选中记录 · ${splits[split]} Calmar</span><strong>${number(selection?.metrics?.[split]?.calmar, 4)}</strong></div>`).join('')}</div>
      <p class="run-selection">${selection ? `正式选中：${esc(progressLabel(run, selection.step))}` : '尚无正式选中记录'}</p>
    </article>`;
  }).join('');
  const signature = JSON.stringify(snapshot.runs.map(run => [run.id, run.label, run.algorithm]));
  if ($('run-select').dataset.signature !== signature) {
    $('run-select').innerHTML = snapshot.runs.map(run => `<option value="${esc(run.id)}">${esc(run.algorithm)} · ${esc(run.label)}</option>`).join('');
    $('run-select').dataset.signature = signature;
  }
  $('run-select').value = selectedId;
}

function renderOverview(run) {
  const metric = $('metric').value;
  const keys = [];
  for (const item of snapshot.runs) {
    const key = `performance:${item.id}`;
    keys.push(key);
    chart(key, 'performance-charts', Object.keys(splits).map((split, index) => ({
      label: splitLabel(item, split), color: palette[index], markers: true,
      points: item.evaluations.filter(record => record.split === split).map(record => [record.step, record.metrics[metric]])
    })), {
      title: `${item.algorithm} · ${metrics[metric]}`, subtitle: item.label,
      xLabel: `训练进度（${item.progress.unit}）`, unit: item.progress.unit,
      formatY: value => formatMetric(value, metric)
    }, Object.keys(splits).map((split, index) => ({ label: `${splits[split]}静态基准`, value: item.baseline?.[split]?.[metric], color: palette[index] })));
  }
  pruneCharts('performance:', keys);
  const selection = run.selection;
  $('selection').innerHTML = selection
    ? `<div class="selection-title"><h3>${esc(run.algorithm)} · ${esc(progressLabel(run, selection.step))}</h3><p>${esc(selection.artifact_id || '模型标识未记录')}</p></div><div class="selection-grid">${Object.keys(splits).map(split => {
      const value = selection.metrics?.[split]?.calmar, base = run.baseline?.[split]?.calmar;
      const delta = Number.isFinite(value) && Number.isFinite(base) ? value - base : null;
      const note = !Number.isFinite(value) ? '尚无同一模型指纹的对应评估' : delta == null ? '没有可比较的静态基准' : `较静态基准 ${delta >= 0 ? '+' : ''}${number(delta, 4)}`;
      return `<div><span>${splitLabel(run, split)} Calmar</span><strong>${number(value, 4)}</strong><small>${note}</small></div>`;
    }).join('')}</div>`
    : '<p class="muted">该训练流程尚未保存正式选择结果。请结合下方评估约定查看进度。</p>';
  const protocolKeys = ['selection', 'test_role', 'objective', 'evaluation_interval', 'eval_every_generations', 'deployment_bundle', 'elapsed_scope'];
  const protocol = Object.fromEntries(protocolKeys.filter(key => key in run.protocol).map(key => [key, run.protocol[key]]));
  $('protocol').innerHTML = `<div class="split-list">${Object.entries(run.splits).map(([split, dates]) => `<span>${esc(splits[split] || split)}：${dates == null ? '未记录' : esc(dates.join(' — '))}</span>`).join('')}</div>${keyValues(protocol)}`;
  const evaluations = run.evaluations.slice().sort((a, b) => b.step - a.step);
  $('evaluation-count').textContent = `${run.label} · 共 ${evaluations.length} 条，当前显示 ${Math.min(evaluationLimit, evaluations.length)} 条。各记录保留自身模型标识与用途。`;
  $('evaluations').innerHTML = evaluations.slice(0, evaluationLimit).map(record => {
    const role = roles[record.role] || record.role || '用途未记录';
    const eligible = record.eligible === true ? ' · 可参与选择' : record.eligible === false ? ' · 不参与选择' : '';
    const artifact = record.artifact_id || '未记录';
    return `<tr><td>${esc(progressLabel(run, record.step))}</td><td><span class="tag ${record.split}">${splits[record.split]}</span></td><td>${formatMetric(record.metrics.calmar, 'calmar')}</td><td>${percent(record.metrics.annualized_return)}</td><td>${percent(record.metrics.max_drawdown)}</td><td>${esc(role + eligible)}</td><td><code title="${esc(artifact)}">${esc(artifact.length > 28 ? artifact.slice(0, 25) + '…' : artifact)}</code></td></tr>`;
  }).join('') || '<tr><td class="empty-cell" colspan="7">尚无已保存评估。</td></tr>';
  $('more-evaluations').hidden = evaluations.length <= evaluationLimit;
  $('download').href = `/api/evaluations.csv?run=${encodeURIComponent(run.id)}`;
}

function renderActions(run) {
  const actions = run.actions.slice().sort((a, b) => a.step - b.step), latest = actions.at(-1);
  const config = run.selection?.config, selectedWeights = config?.weights || {};
  const names = [...new Set([...run.factors, ...Object.keys(selectedWeights), ...Object.keys(latest?.weights || {})])];
  $('action-note').textContent = latest
    ? `${run.label} · 最新动作记录来自 ${progressLabel(run, latest.step)} 的评估；静态配置的最低、平均与最高权重相同。`
    : `${run.label} · 尚无回放动作统计；正式选中权重按原配置展示。`;
  $('weights').innerHTML = names.map(name => {
    const values = latest?.weights[name];
    return `<tr><td>${esc(name)}</td><td>${number(selectedWeights[name], 4)}</td><td>${number(values?.min, 4)}</td><td>${number(values?.mean, 4)}</td><td>${number(values?.max, 4)}</td></tr>`;
  }).join('') || '<tr><td class="empty-cell" colspan="5">尚无因子记录。</td></tr>';
  $('config').innerHTML = `${keyValues(config)}${latest ? `<h3>最新回放的调仓设置</h3>${keyValues(latest.controls)}` : ''}`;
  const series = names.map((name, index) => ({ label: name, color: palette[index % palette.length], points: actions.map(action => [action.step, action.weights[name]?.mean]) }));
  if (actions.length) chart('weights:history', 'weight-history', series, { title: '各次回放的平均因子权重', subtitle: '每个点为该次评估策略的权重均值，静态配置使用其固定值；不是相邻交易日权重。', xLabel: `训练进度（${run.progress.unit}）`, unit: run.progress.unit });
  else pruneCharts('weights:', []);
  if (checkpointRun !== run.id) loadCheckpoints(run);
}

function renderDiagnostics(run) {
  $('diagnostic-note').textContent = run.diagnostic_status === 'historical_summary_only_detail_not_recorded'
    ? '历史运行只展示当时已记录的诊断；新增的分布、梯度与成本明细未记录，不补造。'
    : run.protocol.diagnostic_sampling === 'bucket_minmax_max1000_points_per_series_evaluations_not_sampled'
    ? '诊断曲线保留首尾与分段极值，每条最多 1000 个采样点，不做平滑或补点。训练、验证与测试的评估结果全部保留。'
    : '按当前训练记录提供的诊断项展示，不做平滑或补点。';
  const keys = [];
  const group = $('diagnostic-group').value;
  const belongs = key => key.startsWith('execution/') || ['mean_gross_turnover_ratio', 'mean_total_cost_ratio', 'total_fees'].includes(key)
    ? 'execution' : key.startsWith('gradient/') ? 'gradient' : /^(exploration|deterministic|diversity)\//.test(key) ? 'actions' : 'learning';
  for (const [key, item] of Object.entries(run.diagnostics)) {
    if (belongs(key) !== group) continue;
    const id = `diagnostic:${key}`;
    keys.push(id);
    chart(id, 'diagnostic-charts', [{ label: item.label, color: palette[0], points: item.points }], { title: item.label, xLabel: `训练进度（${run.progress.unit}）`, unit: run.progress.unit });
  }
  pruneCharts('diagnostic:', keys);
  $('diagnostic-empty').hidden = keys.length > 0;
  $('parameters').innerHTML = keyValues(run.parameters);
  const law = run.latest_distribution;
  $('distribution-details').innerHTML = law == null ? '<p class="muted">未记录此项。GA 使用静态配置，权重见“策略动作”。</p>'
    : `<p>来自 ${esc(progressLabel(run, law.step))} 更新前的 ${law.sample_count} 个训练状态。确定性值为 0～1 归一化动作坐标，不是实际换股比例；须按对应运行动作范围换算，例如范围 0～0.2 时，0.5 对应实际比例 0.1。状态间差异与随机采样波动分别列出。采样统计坐标：${esc(law.sampling_scale)}。</p><div class="table-wrap"><table><thead><tr><th>动作</th><th>确定性值均值</th><th>确定性值状态间标准差</th><th>采样均值</th><th>采样标准差</th></tr></thead><tbody>${law.action_names.map((name, index) => `<tr><td>${esc(name)}</td><td>${number(law.deterministic_action_mean[index], 4)}</td><td>${number(law.deterministic_action_std[index], 4)}</td><td>${number(law.sampling_mean[index], 4)}</td><td>${number(law.sampling_std[index], 4)}</td></tr>`).join('')}</tbody></table></div>`;
}

function render() {
  const run = currentRun();
  renderCards();
  $('run-issues').textContent = run.issues.join('\n');
  $('run-issues').hidden = !run.issues.length;
  if (activeTab === 'overview') renderOverview(run);
  if (activeTab === 'actions') renderActions(run);
  if (activeTab === 'diagnostics') renderDiagnostics(run);
  if (activeTab === 'logs') {
    $('log-label').textContent = run.label;
    $('event-log').textContent = run.logs.length ? run.logs.join('\n') : '暂无运行日志。';
    $('identity').textContent = JSON.stringify(run.identity, null, 2);
  }
}

async function request(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `读取失败（${response.status}）`);
  return data;
}
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    // Only the selected run carries its heavy detail sections; other runs arrive card/curve-sized.
    let data = await request(`/api/status${selectedId ? `?detail=${encodeURIComponent(selectedId)}` : ''}`);
    if (data.schema_version !== 'training-report-v2') throw new Error('报告数据版本不匹配');
    if (data.runs.length && !data.runs.some(run => run.id === selectedId)) {
      selectedId = data.runs[0].id; resetTrace();
      data = await request(`/api/status?detail=${encodeURIComponent(selectedId)}`);
    }
    snapshot = data;
    $('empty-runs').hidden = data.runs.length > 0;
    $('report').hidden = !data.runs.length;
    if (data.runs.length) render();
    $('connection-error').hidden = true;
    $('connection').textContent = paused ? '已暂停自动刷新' : '已连接 · 每 5 秒刷新';
    $('updated').textContent = `${new Date().toLocaleString('zh-CN')} 更新`;
  } catch (error) {
    $('connection').textContent = '读取失败';
    $('connection-error').hidden = false;
    $('connection-error').textContent = `无法更新报告：${error.message}。已有显示保留。`;
  } finally { refreshing = false; }
}

function activateTab(id) {
  const tab = [...document.querySelectorAll('[data-tab]')].find(button => button.dataset.tab === id);
  activeTab = tab ? id : 'overview';
  document.querySelectorAll('[data-tab]').forEach(button => { button.classList.toggle('active', button.dataset.tab === activeTab); button.setAttribute('aria-selected', String(button.dataset.tab === activeTab)); });
  document.querySelectorAll('.tab-panel').forEach(panel => { panel.hidden = panel.id !== activeTab; });
  history.replaceState(null, '', `#${activeTab}`);
  if (snapshot?.runs.length) render();
}

async function loadCheckpoints(run, preserve = false) {
  checkpointRun = run.id;
  $('trace-status').textContent = '正在读取可用模型记录…';
  $('load-trace').disabled = true;
  try {
    const data = await request(`/api/checkpoints?run=${encodeURIComponent(run.id)}`);
    if (selectedId !== run.id) return;
    const selected = preserve ? $('checkpoint').value : null;
    checkpointState = data;
    $('checkpoint').innerHTML = data.checkpoints.map(item => `<option value="${esc(item.id)}">${esc(progressLabel(run, item.step))} · ${esc(item.labels.join(' / ') || item.id)}</option>`).join('');
    if (selected && data.checkpoints.some(item => item.id === selected)) $('checkpoint').value = selected;
    $('load-trace').disabled = !data.checkpoints.length;
    traceStatus(run);
  } catch (error) {
    if (selectedId === run.id) $('trace-status').textContent = `无法读取模型列表：${error.message}`;
  }
}
function traceStatus(run) {
  const item = checkpointState.checkpoints.find(record => record.id === $('checkpoint').value);
  const state = item?.job?.state;
  $('trace-status').textContent = !item
    ? (run.algorithm === 'GA' ? 'GA 使用静态配置；没有逐日记录时，请查看上方正式选中权重。' : '尚无已保存的逐日模型记录。')
    : state === 'ready' ? '选择模型后读取已有逐日记录。'
      : '该历史模型没有已保存的逐日记录；报告服务不重新执行历史模型。';
}
async function loadTrace() {
  const run = currentRun(), id = $('checkpoint').value;
  if (!id || traceLoading) return;
  traceLoading = true;
  $('load-trace').disabled = true;
  $('trace-status').textContent = '正在读取逐日记录…';
  try {
    const trace = await request(`/api/trace?run=${encodeURIComponent(run.id)}&id=${encodeURIComponent(id)}`);
    if (selectedId !== run.id || $('checkpoint').value !== id) return;
    const date = value => trace.dates[Math.max(0, Math.min(trace.dates.length - 1, Math.round(value)))] || '—';
    const options = { xLabel: '交易日', formatX: date };
    chart('trace:weights', 'trace-charts', Object.entries(trace.weights).map(([name, values], index) => ({ label: name, color: palette[index % palette.length], points: values.map((value, index) => [index, value]) })), { ...options, title: '逐日因子权重', subtitle: '所选模型的训练期确定性回放' });
    const keys = ['trace:weights'];
    if (Object.keys(trace.controls).length) {
      keys.push('trace:quantities');
      chart('trace:quantities', 'trace-charts', Object.values(trace.controls).map(({ label, values }, index) => ({ label, color: palette[index % palette.length], points: values.map((value, index) => [index, value]) })), { ...options, title: '逐日持股与换股设置' });
    }
    if (Array.isArray(trace.market?.level)) {
      keys.push('trace:market');
      chart('trace:market', 'trace-charts', [{ label: trace.market.label, color: palette[1], points: trace.market.level.map((value, index) => [index, value]) }], { ...options, title: '同期市场参考', subtitle: trace.market.method || '' });
    }
    pruneCharts('trace:', keys);
    $('trace-status').textContent = `已读取 ${trace.dates.length} 个交易日：${trace.dates[0] || '—'} 至 ${trace.dates.at(-1) || '—'}。`;
  } catch (error) {
    if (selectedId === run.id && $('checkpoint').value === id) { pruneCharts('trace:', []); $('trace-status').textContent = `没有可用的逐日记录：${error.message}`; }
  } finally { traceLoading = false; $('load-trace').disabled = !$('checkpoint').value; }
}
document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => activateTab(button.dataset.tab)));
$('run-select').addEventListener('change', async event => { selectedId = event.target.value; evaluationLimit = 100; resetTrace(); render(); await refresh(); });
$('metric').addEventListener('change', () => renderOverview(currentRun()));
$('diagnostic-group').addEventListener('change', () => renderDiagnostics(currentRun()));
$('more-evaluations').addEventListener('click', () => { evaluationLimit += 100; renderOverview(currentRun()); });
$('refresh').addEventListener('click', async () => { await refresh(); if (activeTab === 'actions' && snapshot?.runs.length) loadCheckpoints(currentRun(), true); });
$('pause').addEventListener('click', () => { paused = !paused; $('pause').textContent = paused ? '继续刷新' : '暂停刷新'; $('pause').setAttribute('aria-pressed', String(paused)); $('connection').textContent = paused ? '已暂停自动刷新' : '正在读取'; if (!paused) refresh(); });
$('checkpoint').addEventListener('change', () => { pruneCharts('trace:', []); traceStatus(currentRun()); });
$('load-trace').addEventListener('click', loadTrace);
activateTab(location.hash.slice(1));
refresh();
setInterval(() => { if (!paused && !document.hidden) refresh(); }, 5000);
