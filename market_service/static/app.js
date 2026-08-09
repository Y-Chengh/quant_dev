const $=id=>document.getElementById(id);let period='1d',page=1,pageSize=100,total=0;const periodLabels={'5m':'5分钟','15m':'15分钟','30m':'30分钟','60m':'60分钟','1d':'日K','1w':'周K','1mo':'月K'};
const queryConfigStorageKey='marketService.queryConfig.v1';
const chartFieldIds=['chartCode','chartStart','chartEnd'];
const rawFieldIds=['rawCode','rawDate','startTime','endTime','minClose','maxClose','minVolume','maxVolume','minAmount','maxAmount'];
/**
 * 从页面输入框收集可持久化的查询值。
 * @param {string[]} fieldIds 查询表单输入框ID列表，仅收集这些白名单字段。
 * @returns {object} 以输入框ID为键、当前字符串值为值的配置对象。
 */
function collectFieldValues(fieldIds){return Object.fromEntries(fieldIds.map(id=>[id,$(id).value]))}
/**
 * 读取当前浏览器保存的上次查询配置，无法解析或旧版本配置按不存在处理。
 * @returns {object} 通过版本和外层对象结构校验的配置；不可读取时返回空对象。
 */
function readQueryConfig(){try{const config=JSON.parse(window.localStorage.getItem(queryConfigStorageKey));return config&&typeof config==='object'&&config.version===1?config:{}}catch{return{}}}
/**
 * 激活指定查询页签。
 * @param {string} panelId 目标面板ID，只接受页面中已有页签声明的面板。
 * @returns {boolean} 成功切换时返回true，面板无效时返回false。
 */
function activateTab(panelId){const button=[...document.querySelectorAll('.tab')].find(item=>item.dataset.tab===panelId),panel=$(panelId);if(!button||!panel)return false;document.querySelectorAll('.tab,.panel').forEach(item=>item.classList.remove('active'));button.classList.add('active');panel.classList.add('active');if(panelId==='chartPanel')setTimeout(()=>chart.resize(),0);return true}
/**
 * 选择K线周期并同步周期按钮状态。
 * @param {string} value K线周期，仅接受页面支持的周期枚举。
 * @returns {boolean} 周期有效并完成切换时返回true，否则返回false。
 */
function selectPeriod(value){if(!Object.prototype.hasOwnProperty.call(periodLabels,value))return false;const button=[...document.querySelectorAll('.periods button')].find(item=>item.dataset.period===value);if(!button)return false;document.querySelectorAll('.periods button').forEach(item=>item.classList.remove('selected'));button.classList.add('selected');period=value;return true}
let savedQueryConfig=readQueryConfig();
/**
 * 把指定查询区域的最新配置保存到当前浏览器。
 * @param {'chart'|'raw'|'tab'} section 本次发生变化的配置区域；未变化区域沿用原保存值。
 * @returns {void} 不返回结果；浏览器禁用本地存储时静默保留当前页面功能。
 */
function persistQueryConfig(section){const activeTab=document.querySelector('.tab.active')?.dataset.tab;const next={...savedQueryConfig,version:1};if(activeTab)next.activeTab=activeTab;if(section==='chart')next.chart={...collectFieldValues(chartFieldIds),period};if(section==='raw')next.raw=collectFieldValues(rawFieldIds);try{window.localStorage.setItem(queryConfigStorageKey,JSON.stringify(next));savedQueryConfig=next}catch{}}
/**
 * 将上次查询配置恢复到表单、周期和页签，并记录已恢复的日期字段。
 * @param {object} config 从浏览器本地存储读取并完成版本校验的配置对象。
 * @returns {Set<string>} 已恢复的输入框ID集合，供元数据默认值避免覆盖用户配置。
 */
function restoreQueryConfig(config){const restored=new Set(),requiredDateIds=new Set(['chartStart','chartEnd','rawDate']);for(const [section,fieldIds] of [['chart',chartFieldIds],['raw',rawFieldIds]]){const values=config[section];if(!values||typeof values!=='object')continue;for(const id of fieldIds){if(typeof values[id]!=='string')continue;const input=$(id);input.value=values[id];if(input.value===values[id]&&(!requiredDateIds.has(id)||input.value))restored.add(id)}}if(config.chart&&typeof config.chart.period==='string')selectPeriod(config.chart.period);if(typeof config.activeTab==='string')activateTab(config.activeTab);return restored}
const chart=echarts.init($('chart'),null,{renderer:'canvas'});
const restoredFieldIds=restoreQueryConfig(savedQueryConfig);
function toast(message,error=false){const el=$('toast');el.textContent=message;el.className=error?'show error':'show';setTimeout(()=>el.className='',3500)}
async function api(url){const r=await fetch(url);if(!r.ok){let x;try{x=await r.json()}catch{x={detail:r.statusText}}throw new Error(x.detail||r.statusText)}return r.json()}
document.querySelectorAll('.tab').forEach(btn=>btn.onclick=()=>{if(activateTab(btn.dataset.tab))persistQueryConfig('tab')});
document.querySelectorAll('.periods button').forEach(btn=>btn.onclick=()=>{if(selectPeriod(btn.dataset.period)){++chartRequestVersion;chart.hideLoading()}});
/**
 * 加载行情元数据，并只为未恢复的日期输入框设置数据库范围默认值。
 * @returns {Promise<void>} 元数据加载完成后结束；失败信息直接展示在页面状态区。
 */
async function loadMeta(){try{const m=await api('/api/meta');$('datasetStatus').textContent=`${m.symbols.toLocaleString()}只证券 · ${m.rows.toLocaleString()}行 · ${m.first_time.slice(0,10)} 至 ${m.last_time.slice(0,10)}`;$('datasetStatus').classList.add('ok');if(!restoredFieldIds.has('chartStart'))$('chartStart').value=m.first_time.slice(0,4)==='2010'?'2025-01-01':m.first_time.slice(0,10);if(!restoredFieldIds.has('chartEnd'))$('chartEnd').value=m.last_time.slice(0,10);if(!restoredFieldIds.has('rawDate'))$('rawDate').value=m.last_time.slice(0,10)}catch(e){$('datasetStatus').textContent=e.message;toast(e.message,true)}}
/**
 * 格式化后端按本根收盘价和上一根K线收盘价计算的标准涨跌幅。
 * @param {object} item 当前周期的K线行情，价格单位与数据源一致。
 * @returns {string} 带正负号、保留两位小数的百分比文本；涨跌幅无效时返回破折号。
 */
function formatPctChange(item){if(item.pct_change==null||item.pct_change==='')return'—';const raw=Number(item.pct_change);if(!Number.isFinite(raw))return'—';const value=Math.abs(raw)<.005?0:raw;return`${value>=0?'+':''}${value.toFixed(2)}%`}
/**
 * 格式化本根K线从开盘到收盘的日内涨跌幅。
 * @param {object} item 当前周期的K线行情，日内涨跌幅由后端统一计算。
 * @returns {string} 带正负号、保留两位小数的百分比文本；数值无效时返回破折号。
 */
function formatIntradayPctChange(item){if(item.intraday_pct_change==null||item.intraday_pct_change==='')return'—';const raw=Number(item.intraday_pct_change);if(!Number.isFinite(raw))return'—';const value=Math.abs(raw)<.005?0:raw;return`${value>=0?'+':''}${value.toFixed(2)}%`}
/**
 * 格式化有限数值，避免把空值静默转换为零。
 * @param {*} value 待格式化的行情数值。
 * @param {object} options Intl.NumberFormat 使用的小数位配置；缺省为整数格式。
 * @returns {string} 本地化数值文本；空值或非有限值返回破折号。
 */
function formatNumber(value,options={}){if(value==null||value==='')return'—';const number=Number(value);return Number.isFinite(number)?number.toLocaleString('zh-CN',options):'—'}
/**
 * 转义提示框中的文本，防止数据字段被解释为HTML。
 * @param {*} value 待显示的文本值。
 * @returns {string} 完成HTML实体转义的安全文本。
 */
function escapeHtml(value){return String(value??'—').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]))}
/**
 * 生成K线悬停内容，集中展示价格、涨跌幅和成交量。
 * @param {object[]} items 当前查询返回的全部K线行情。
 * @param {object[]} params ECharts 当前横轴位置命中的系列参数。
 * @returns {string} 可直接交给 ECharts tooltip 渲染的HTML文本。
 */
function klineTooltip(items,params){if(!Array.isArray(params)||!params.length)return'';const point=params.find(x=>x.seriesType==='candlestick')||params[0],item=point&&items[point.dataIndex];if(!item)return'';const priceOptions={minimumFractionDigits:2,maximumFractionDigits:4},pct=formatPctChange(item),intradayPct=formatIntradayPctChange(item),pctColor=pct==='—'?'#8291a5':pct.startsWith('-')?'#29c782':'#ef5b70',intradayPctColor=intradayPct==='—'?'#8291a5':intradayPct.startsWith('-')?'#29c782':'#ef5b70';return`<strong>${escapeHtml(item.time)}</strong><br>开盘：${formatNumber(item.open,priceOptions)}<br>最高：${formatNumber(item.high,priceOptions)}<br>最低：${formatNumber(item.low,priceOptions)}<br>收盘：${formatNumber(item.close,priceOptions)}<br>涨跌幅：<span style="color:${pctColor};font-weight:700">${pct}</span><br>日内涨跌幅：<span style="color:${intradayPctColor};font-weight:700">${intradayPct}</span><br>成交量：${formatNumber(item.volume)}`}
/**
 * 构造价格、成交量、缩放和悬停提示所需的完整ECharts配置。
 * @param {object[]} items 当前查询返回的全部K线行情。
 * @returns {object} 可传给 ECharts setOption 的图表配置。
 */
function chartOption(items){const times=items.map(x=>x.time),ohlc=items.map(x=>[x.open,x.close,x.low,x.high]),vol=items.map(x=>({value:x.volume,itemStyle:{color:x.close>=x.open?'#ef5b70':'#29c782'}}));return{animation:false,backgroundColor:'#111821',axisPointer:{link:[{xAxisIndex:'all'}]},tooltip:{trigger:'axis',axisPointer:{type:'cross'},backgroundColor:'#111923',borderColor:'#42566c',textStyle:{color:'#dce6f2'},formatter:params=>klineTooltip(items,params)},grid:[{left:66,right:24,top:28,height:'67%'},{left:66,right:24,top:'78%',height:'12%'}],xAxis:[{type:'category',data:times,boundaryGap:true,axisLine:{lineStyle:{color:'#334254'}},axisLabel:{color:'#8291a5'},min:'dataMin',max:'dataMax'},{type:'category',gridIndex:1,data:times,boundaryGap:true,axisLabel:{show:false},axisLine:{lineStyle:{color:'#334254'}},min:'dataMin',max:'dataMax'}],yAxis:[{scale:true,splitLine:{lineStyle:{color:'#1e2936'}},axisLabel:{color:'#8291a5'}},{scale:true,gridIndex:1,splitNumber:2,splitLine:{show:false},axisLabel:{color:'#8291a5',formatter:v=>v>=1e8?(v/1e8).toFixed(1)+'亿':v>=1e4?(v/1e4).toFixed(0)+'万':v}}],dataZoom:[{type:'inside',xAxisIndex:[0,1],start:0,end:100,zoomOnMouseWheel:true,moveOnMouseMove:true},{type:'slider',xAxisIndex:[0,1],bottom:8,height:20,start:0,end:100,borderColor:'#253140',backgroundColor:'#0d131b',fillerColor:'rgba(75,216,200,.12)',handleStyle:{color:'#4bd8c8'}}],series:[{name:'OHLC',type:'candlestick',data:ohlc,itemStyle:{color:'#ef5b70',color0:'#29c782',borderColor:'#ef5b70',borderColor0:'#29c782'}},{name:'成交量',type:'bar',xAxisIndex:1,yAxisIndex:1,data:vol}]}}
let chartRequestVersion=0;
/**
 * 按当前代码、日期和周期加载K线，并忽略被后续自动查询取代的旧响应。
 * @returns {Promise<void>} 最新一次图表请求完成后结束，查询错误通过页面提示反馈。
 */
async function loadChart(){const requestVersion=++chartRequestVersion,requestPeriod=period,code=$('chartCode').value.trim().toUpperCase(),start=$('chartStart').value,end=$('chartEnd').value;if(!code||!start||!end){chart.hideLoading();return toast('请完整填写代码和日期',true)}$('chartCode').value=code;persistQueryConfig('chart');chart.showLoading({color:'#4bd8c8',textColor:'#8291a5',maskColor:'rgba(11,15,20,.8)'});try{const q=new URLSearchParams({code,start:start+'T00:00:00',end:end+'T23:59:59',period:requestPeriod});const r=await api('/api/kline?'+q);if(requestVersion!==chartRequestVersion)return;chart.setOption(chartOption(r.items),true);$('chartTitle').textContent=`${r.code} · ${periodLabels[requestPeriod]}`;$('chartCount').textContent=`${r.count.toLocaleString()} 根K线`;if(!r.count)toast('该条件下没有数据',true)}catch(e){if(requestVersion===chartRequestVersion)toast(e.message,true)}finally{if(requestVersion===chartRequestVersion)chart.hideLoading()}}
let chartDateTimer;
/**
 * 日期选择变化后防抖刷新K线，避免连续选择两个日期时重复请求。
 * @returns {void} 不返回结果；查询必填项不完整时仅取消待执行请求。
 */
function scheduleChartLoad(){clearTimeout(chartDateTimer);++chartRequestVersion;if(!$('chartCode').value.trim()||!$('chartStart').value||!$('chartEnd').value){chart.hideLoading();return}chartDateTimer=setTimeout(loadChart,300)}
/**
 * 标记用户已经编辑过的日期，防止较晚返回的元数据默认值覆盖输入。
 * @param {Event} event 日期输入框触发的浏览器输入事件。
 * @returns {void} 不返回结果；仅更新本次页面生命周期内的默认值保护集合。
 */
function protectDateFromMetadataDefault(event){restoredFieldIds.add(event.target.id)}
$('chartForm').onsubmit=e=>{e.preventDefault();clearTimeout(chartDateTimer);loadChart()};['chartStart','chartEnd','rawDate'].forEach(id=>$(id).addEventListener('input',protectDateFromMetadataDefault));['chartStart','chartEnd'].forEach(id=>$(id).addEventListener('change',scheduleChartLoad));window.addEventListener('resize',()=>chart.resize());
function rawParams(includePage=true){const map={code:'rawCode',trade_date:'rawDate',start_time:'startTime',end_time:'endTime',min_close:'minClose',max_close:'maxClose',min_volume:'minVolume',max_volume:'maxVolume',min_amount:'minAmount',max_amount:'maxAmount'},q=new URLSearchParams();for(const [k,id] of Object.entries(map)){const v=$(id).value.trim();if(v)q.set(k,v)}if(includePage){q.set('page',page);q.set('page_size',pageSize)}return q}
const n=v=>v==null?'—':Number(v).toLocaleString('zh-CN',{maximumFractionDigits:4});
async function loadRaw(){try{const r=await api('/api/raw?'+rawParams());total=r.total;$('rawBody').innerHTML=r.items.map(x=>`<tr><td>${x.code}</td><td>${x.trade_time}</td><td>${n(x.open)}</td><td>${n(x.high)}</td><td>${n(x.low)}</td><td>${n(x.close)}</td><td>${n(x.volume)}</td><td>${n(x.amount)}</td><td>${n(x.pre_close)}</td><td>${n(x.change)}</td><td>${n(x.pct_change)}</td></tr>`).join('');$('rawSummary').textContent=`共 ${total.toLocaleString()} 条记录`;$('pageInfo').textContent=`第 ${page} / ${Math.max(1,Math.ceil(total/pageSize))} 页`;$('prevPage').disabled=page<=1;$('nextPage').disabled=page*pageSize>=total}catch(e){toast(e.message,true)}}
$('rawForm').onsubmit=e=>{e.preventDefault();$('rawCode').value=$('rawCode').value.trim().toUpperCase();persistQueryConfig('raw');page=1;loadRaw()};$('prevPage').onclick=()=>{if(page>1){page--;loadRaw()}};$('nextPage').onclick=()=>{if(page*pageSize<total){page++;loadRaw()}};$('resetFilters').onclick=()=>{['startTime','endTime','minClose','maxClose','minVolume','maxVolume','minAmount','maxAmount'].forEach(id=>$(id).value='');persistQueryConfig('raw');page=1;loadRaw()};$('downloadCsv').onclick=()=>{persistQueryConfig('raw');window.location='/api/raw.csv?'+rawParams(false)};
let symbolTimer;['chartCode','rawCode'].forEach(id=>$(id).addEventListener('input',e=>{clearTimeout(symbolTimer);symbolTimer=setTimeout(async()=>{try{const r=await api('/api/symbols?q='+encodeURIComponent(e.target.value));$('symbols').innerHTML=r.items.map(x=>`<option value="${x}">`).join('')}catch{}},200)}));
loadMeta().then(loadChart);
