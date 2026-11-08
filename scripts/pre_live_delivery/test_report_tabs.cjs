// Pure JavaScript unit test with synthetic data and mocked document/chart ports.
// Does not launch or control a browser, render HTML, or access a network.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const path='testback/reportor/templates/single_report_module.js';const source=fs.readFileSync(path,'utf8');
const begin=source.indexOf('function renderExplanations() {'),end=source.indexOf('\nrenderPerformanceCharts();',begin);
assert(begin>=0 && end>begin);
class Element {
  constructor(){this.innerHTML='';this.textContent='';this.value='';this.dataset={};this.children=[];this.events={};}
  appendChild(x){this.children.push(x);if(!this.value)this.value=x.value;}
  replaceChildren(){this.children=[];this.value='';}
  addEventListener(name,fn){this.events[name]=fn;}
}
const nodes=new Map(),selectors={};const get=(id)=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
const document={getElementById:get,createElement:()=>new Element(),querySelectorAll:(selector)=>{
  const attr=selector.slice(1,-1);const key=attr.slice(5).replace(/-([a-z])/g,(_,x)=>x.toUpperCase());const result=[];
  for(const element of nodes.values())for(const match of element.innerHTML.matchAll(new RegExp(attr+'="([^"]+)"','g'))){const item=new Element();item.dataset[key]=match[1];result.push(item);}
  selectors[selector]=result;return result;
}};
const charts={},klineCalls=[];
const makeChart=(id)=>charts[id]={events:{},option:null,cleared:false,setOption(x){this.option=x;this.cleared=false;},clear(){this.cleared=true;},on(name,fn){this.events[name]=fn;}};
const stock=(code)=>({code,open_weight:.5,final_score:.8,head_ranks:[.8,.7,.9,.4,.3],raw_blended_score:.63,baseline:.5,horizon_contribution:.02,other_factor_contribution:.03,nav_contribution_pct:.2,top_factors:[{name:'X<script>',contribution:.1,normalized_value:.7,raw_value:null},{name:'Y',contribution:-.02,normalized_value:null,raw_value:null}]});
const position=(code,pnl)=>({code,open_weight:.5,net_pnl:pnl,nav_contribution_pct:pnl/10000,corporate_action:{}});
const d={dates:['2026-01-05','2026-01-06'],settlement_dates:['2026-01-06','2026-01-07'],horizons:[3,7,15,30,60],factor_names:['X<script>','Y'],heatmap:[[0,0,.1],[0,1,-.02],[1,0,.2],[1,1,-.01]],importance:[{name:'X<script>',mean_abs_weighted_contribution:.15},{name:'Y',mean_abs_weighted_contribution:.02}],feature_info:{'X<script>':{description:'test <description>',formula:'Ref(close, 1)'}},daily_stocks:[[stock('600001.SH')],[stock('600002.SH')]],daily_pnl:[{decision_date:'2026-01-05',settlement_date:'2026-01-06',return_pct:-1,fees:1,reconciliation_error:0,stocks:[position('600001.SH',-100),position('600003.SH',0)]},{decision_date:'2026-01-06',settlement_date:'2026-01-07',return_pct:2,fees:2,reconciliation_error:0,stocks:[position('600002.SH',200)]}],coverage_open_weight:[.5,.5],daily_top_factors:[{positive:[{name:'X<script>',contribution:.1}],negative:[{name:'Y',contribution:-.02}]},{positive:[{name:'X<script>',contribution:.2}],negative:[{name:'Y',contribution:-.01}]}]};
const context={document,makeChart,COLORS:{strategy:'blue',negative:'red'},REPORT_DATA:{explainability:d,audit_summary:{state:'pass',counts:{invalid_open:0},max_pnl_reconciliation_error:0}},showKline:(code)=>klineCalls.push(code)};
get('panel-explanation').hidden=true;
vm.runInNewContext(source.slice(begin,end)+'\nrenderExplanations();',context,{timeout:2000});
assert.equal(Object.keys(charts).length,0);
get('panel-explanation').hidden=false;
vm.runInNewContext(source.slice(begin,end)+'\nrenderExplanations();',context,{timeout:2000});
const firstOptions=get('explain-day').children.length;
vm.runInNewContext(source.slice(begin,end)+'\nrenderExplanations();',context,{timeout:2000});
assert.equal(get('explain-day').children.length,firstOptions);
assert.equal(get('explain-day').value,'1');assert(get('explain-day-summary').innerHTML.includes('2026-01-06'));
assert.equal(charts['explain-heatmap'].option.series[0].data.length,4);
assert(charts['explain-heatmap'].option.tooltip.formatter({data:[0,0,.1]}).includes('X&lt;script&gt;'));
charts['explain-heatmap'].events.click({data:[0,0,.1]});assert.equal(get('explain-day').value,'0');
assert(get('explain-top-factors').innerHTML.includes('X&lt;script&gt;'));assert(!get('explain-top-factors').innerHTML.includes('X<script>'));
get('explain-stock').value='600001.SH';get('explain-stock').events.change();
const rawMissingTooltip=charts['explain-stock-chart'].option.tooltip.formatter([{dataIndex:0}]);assert(rawMissingTooltip.includes('模型输入值 0.7') && rawMissingTooltip.includes('原始因子值 —'));
const value=charts['explain-stock-chart'].option.series[0].data.reduce((s,x)=>s+x.value,0)+.5;assert(Math.abs(value-.63)<1e-12);
selectors['[data-explain-kline]'][0].events.click();assert.equal(klineCalls[0],'600001.SH');
selectors['[data-explain-code]'].find(x=>x.dataset.explainCode==='600003.SH').events.click();assert(charts['explain-stock-chart'].cleared);assert(get('explain-stock-summary').textContent.includes('没有可用'));
get('explain-day').value='1';get('explain-day').events.change();assert(!charts['explain-stock-chart'].cleared);assert.equal(get('explain-stock').value,'600002.SH');
const result={state:'pass',cases:['default_maximum_gain_day','heatmap_day_selection','date_selector','stock_selector_and_score_parts','native_kline_action_routing','missing_prediction_state','HTML_text_escaping'],script_sha256:crypto.createHash('sha256').update(source).digest('hex'),test_script_sha256:crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),browser_used:false,network_used:false,rendered_pixels:false};

// Exercise the real tab controller against isolated DOM/table/chart ports.
const ids=['monthly','trades','holdings','cleared','daily','delist','comparison','explanation'];
const tabButtons=ids.map(id=>{const e=new Element();e.dataset.tab=id;e.attributes={};e.classList={toggle:(k,v)=>{e[k]=v;}};e.setAttribute=(k,v)=>e.attributes[k]=v;e.focus=()=>{e.focused=true;};return e;});
const tabPanels=ids.map((id,i)=>({id:'panel-'+id,hidden:i!==0,classList:{toggle(){}}}));
let visibleTableRefresh=0,hiddenTableRefresh=0,chartResize=0,explanationRequests=0;
const tabContext={document:{querySelectorAll:s=>s.includes('tab-button')?tabButtons:tabPanels},requestAnimationFrame:fn=>fn(),renderExplanations:()=>explanationRequests++,TABLE_INSTANCES:[{root:{closest:()=>null},syncHeaderPadding(){},renderBody:()=>visibleTableRefresh++},{root:{closest:()=>({hidden:true})},syncHeaderPadding(){},renderBody:()=>hiddenTableRefresh++}],CHARTS:[{getDom:()=>({closest:()=>null}),resize:()=>chartResize++},{getDom:()=>({closest:()=>tabPanels[7].hidden?{hidden:true}:null}),resize:()=>chartResize++}]};
const tabStart=source.indexOf('function initTabs() {');assert(tabStart>=0 && begin>tabStart);
vm.runInNewContext(source.slice(tabStart,begin)+'\ninitTabs();',tabContext,{timeout:2000});
tabButtons[3].events.click();assert(!tabPanels[3].hidden && tabPanels.filter(p=>!p.hidden).length===1);
tabButtons[1].events.click();assert.equal(tabPanels[1].hidden,false);assert.equal(tabPanels[0].hidden,true);assert.equal(tabButtons[1].attributes['aria-selected'],'true');assert.equal(tabButtons[0].attributes['aria-selected'],'false');
let prevented=false;tabButtons[1].events.keydown({key:'ArrowRight',preventDefault(){prevented=true;}});assert(prevented && tabButtons[2].focused && !tabPanels[2].hidden);
tabButtons[7].events.click();assert.equal(explanationRequests,1);assert.equal(tabPanels[7].hidden,false);assert.equal(tabPanels.filter(p=>!p.hidden).length,1);
tabButtons[7].events.keydown({key:'ArrowRight',preventDefault(){}});assert(!tabPanels[0].hidden && tabButtons[0].focused);assert.equal(hiddenTableRefresh,0);assert(visibleTableRefresh>0 && chartResize>0);
result.cases.push('explanation_lazy_while_hidden','explanation_initialized_once','six_original_plus_two_optional_tabs','tab_click_visibility_and_ARIA','tab_keyboard_navigation_and_wrap','visible_charts_resize_after_tab_change','only_visible_tables_refresh');
let equityOption;
const chartContext={REPORT_DATA:{summary:{},charts:{equity:{trade_dates:['2015-07-10'],execution_dates:['2015-07-09'],strategy_nav:[1.02],benchmark_nav:[],daily_returns_pct:[2],drawdown_pct:[0],exposure_pct:[99],rebalance_funds_pct:[103.494],activity:[{buys:1,sells:1}]}}},makeChart:()=>({setOption:o=>equityOption=o}),COLORS:{strategy:'blue',benchmark:'orange',negative:'red',teal:'green',turnover:'purple',muted:'gray',grid:'gray'},document:{getElementById:()=>({})},renderNoData:()=>{throw Error('unexpected missing chart');}};
const chartBegin=source.indexOf('function renderPerformanceCharts() {'),chartEnd=source.indexOf('function renderMonthlyHeatmap()',chartBegin);
assert(chartBegin>=0 && chartEnd>chartBegin);vm.runInNewContext(source.slice(chartBegin,chartEnd)+'\nrenderPerformanceCharts();',chartContext,{timeout:2000});
assert.equal(equityOption.yAxis[1].max({max:103.494}),110);assert.equal(equityOption.yAxis[1].max({max:99}),100);
const tip=equityOption.tooltip.formatter([{dataIndex:0}]);assert(tip.includes('估值日 2015-07-10') && tip.includes('执行日 2015-07-09'));assert(tip.includes('103.5%') && !tip.includes('沪深300'));assert(tip.includes('估值仓位'));
result.cases.push('gross_turnover_above100pct_not_clipped','execution_and_next_valuation_dates_distinguished','missing_benchmark_not_invented_as_zero','valuation_exposure_label');
(async()=>{
  const kNodes=new Map();const kn=id=>{if(!kNodes.has(id))kNodes.set(id,{style:{},classList:{add(){},remove(){}},textContent:''});return kNodes.get(id);};
  const kStock={n:'样本',d:['2026-01-01','2026-01-02','2026-01-03','2026-01-04','2026-01-05','2026-01-06','2026-01-07'],o:[10,10,10,10,10,10,10],h:[11,11,11,11,11,11,11],l:[9,9,9,9,9,9,9],c:[10,10,10,10,10,10,10],a:[100,100,100,100,100,100,100],events:[{id:1,episode:0,date:'2026-01-02',action:'buy',price:10},{id:2,episode:0,date:'2026-01-06',action:'sell',price:10}],episodes:[{id:0,start:'2026-01-02',end:'2026-01-07',open:true,window_start:'2026-01-01',window_end:'2026-01-07'}]};
  let option;const kContext={REPORT_DATA:{summary:{position_lot_analytics_available:false}},document:{getElementById:kn},window:{},loadKlineData:async()=>({'600001.SH':kStock}),echarts:{init:()=>({setOption:o=>option=o,dispose(){}})},COLORS:{},movingAverage:()=>[]};
  const kb=source.indexOf('async function showKline('),ke=source.indexOf("document.addEventListener('click'",kb);assert(kb>=0&&ke>kb);vm.runInNewContext(source.slice(kb,ke),kContext,{timeout:2000});
  await kContext.showKline('600001.SH',1);assert(kn('klineMeta').textContent.includes('成交窗口')&&!kn('klineMeta').textContent.includes('仍持有'));assert.equal(option.series[0].markLine.data[0].xAxis,'2026-01-02');
  await kContext.showKline('600001.SH');assert.equal(kn('klinePlaceholder').style.display,'none');assert.equal(option.series[0].markLine.data[0].xAxis,'2026-01-06');
  await kContext.showKline('600001.SH',undefined,'2026-01-03');assert.equal(option.series[0].markLine.data[0].xAxis,'2026-01-02');
  kContext.REPORT_DATA.summary.position_lot_analytics_available=true;await kContext.showKline('600001.SH',1);assert(kn('klineMeta').textContent.includes('仍持有'));
  result.cases.push('Kline_trade_window_not_invented_holding_status','Kline_explicit_event_preserved','Kline_code_only_opens_latest_event','Kline_explanation_date_selects_nearest_actual_trade','Kline_supported_lot_contract_retains_original_label');
  kContext.REPORT_DATA.summary.kline_position_state_available=false;await kContext.showKline('600001.SH',1);assert(kn('klineMeta').textContent.includes('成交窗口'));
  const notes=[],mounted=[],cleared={rows:[[{html:'实际已清仓',sort:1}]],accounting_note:'经济口径 <说明只作文本>'};
  const host={parentElement:{insertBefore(p,h){assert.equal(h,host);notes.push(p.textContent);}}};
  const tableContext={REPORT_DATA:{tables:{cleared}},document:{getElementById:()=>host,createElement:()=>({})},loadTanStack:async()=>({fixture:true}),mountTable:(id,payload)=>mounted.push([id,payload]),renderNoData:()=>{throw Error('Unexpected bootstrap fallback');},console};
  const tablesBegin=source.indexOf('async function initTables() {');vm.runInNewContext(source.slice(tablesBegin,tabStart),tableContext,{timeout:2000});await tableContext.initTables();
  assert.deepEqual(notes,[cleared.accounting_note]);assert.equal(mounted.length,6);assert.equal(mounted.find(x=>x[0]==='cleared-host')[1],cleared);
  result.cases.push('normalized_input_available_raw_unavailable_not_zero','cleared_tab_click_visibility','economic_episode_Kline_state_flag','cleared_accounting_note_and_six_mounts');
  const receipt=require('node:path').join(__dirname,'ui_logic_test.json');fs.writeFileSync(receipt+'.tmp',JSON.stringify(result,null,2));fs.renameSync(receipt+'.tmp',receipt);console.log(result);
})().catch(error=>{console.error(error);process.exitCode=1;});

