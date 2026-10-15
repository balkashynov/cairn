'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number(value).toLocaleString('en-US', {maximumFractionDigits: 1});
const percent = value => `${(100 * value).toFixed(1)}%`;
const ms = value => `${Number(value).toFixed(value < 1 ? 3 : 2)} ms`;
const names = {exhaustive:'Exhaustive reference', certified:'Cairn · bound-guided exact', binary_1x:'Binary · 10 candidates', binary_4x:'Binary · 40 candidates', binary_16x:'Binary · 160 candidates', int8_1x:'Int8 · 10 candidates', int8_4x:'Int8 · 40 candidates', int8_16x:'Int8 · 160 candidates'};
const datasetNames = {scifact:'SciFact', 'cqadupstack-programmers':'Programmers', fiqa:'FiQA', 'cohere-wiki-100k':'Wikipedia 100k', 'cohere-wiki-1m':'Wikipedia 1m'};
let toyCertified = false;
function renderToy() {
  const ties = $('near-ties').checked;
  const estimates = ties ? [.810,.808,.806,.804,.802,.800] : [.920,.880,.820,.800,.650,.310];
  const scores = ties ? [.812,.807,.801,.827,.815,.825] : [.923,.876,.801,.827,.645,.315];
  const errors = ties ? [.025,.025,.025,.025,.025,.025] : [.010,.010,.025,.030,.015,.015];
  const rows = estimates.map((estimate,i) => ({id:'ABCDEF'[i],estimate,score:scores[i],max:estimate+errors[i],scored:i<3}));
  let selected = rows.slice(0,3).sort((a,b)=>b.score-a.score);
  if (toyCertified) {
    for (const row of rows.slice(3)) {
      if (row.max >= selected.at(-1).score) {
        row.scored = true;
        selected = [...selected,row].sort((a,b)=>b.score-a.score).slice(0,3);
      }
    }
  }
  const expected = [...rows].sort((a,b)=>b.score-a.score).slice(0,3);
  const gold = new Set(expected.map(row=>row.id));
  const chosen = new Set(selected.map(row=>row.id));
  const missing = expected.filter(row=>!chosen.has(row.id));
  $('toy-shortlist').setAttribute('aria-pressed', String(!toyCertified));
  $('toy-cairn').setAttribute('aria-pressed', String(toyCertified));
  $('toy-rows').innerHTML = rows.map((row,i) => {
    const recovered = row.scored && i>=3;
    const missed = !row.scored && gold.has(row.id);
    const status = recovered ? 'Check original' : row.scored ? 'Seed checked' : toyCertified ? 'Safe to skip' : 'Outside shortlist';
    return `<div class="toy-row ${recovered?'recovered':missed?'missed':''}" role="row"><span class="vector-name" role="cell">${row.id}</span><div role="cell"><div class="score-track" aria-hidden="true"><span class="estimate" style="width:${row.estimate*100}%"></span><span class="uncertainty" style="left:${(row.estimate-errors[i])*100}%;width:${2*errors[i]*100}%"></span>${row.scored?`<span class="exact-dot" style="left:${row.score*100}%"></span>`:''}</div><div class="score-caption">${row.estimate.toFixed(3)} → max ${row.max.toFixed(3)}</div></div><span class="decision" role="cell">${status}<small>${row.scored?'Full score '+row.score.toFixed(3):'Original not scored'}</small></span></div>`;
  }).join('');
  $('toy-label').textContent = toyCertified ? 'Bound-guided result' : 'Fixed-shortlist result';
  $('toy-answer').innerHTML = selected.map(row=>`<span class="${gold.has(row.id)?'found':''}">${row.id}</span>`).join('');
  $('toy-count').textContent = `${rows.filter(row=>row.scored).length} / 6`;
  $('toy-verdict').textContent = missing.length ? `${missing.map(row=>row.id).join(', ')} ${missing.length===1?'was':'were'} left out.` : ties ? 'Ambiguous? Check all six.' : 'D gets its place back.';
  $('toy-verdict').className = missing.length ? 'bad' : 'good';
  $('toy-explanation').textContent = toyCertified
    ? ties ? 'Every remaining upper bound could beat the current cutoff. Cairn checks every original and returns D, F, E. The fallback costs work, but preserves the ranking.' : 'D could score as high as 0.830, above the seed cutoff of 0.801. Checking its original gives 0.827. E and F cannot compete, so their originals stay unread.'
    : ties ? 'Choosing A, B, C by compressed score misses all three full-score winners. Reordering those same three candidates cannot bring D, F, E back.' : 'The compressed scores pick A, B, C. After full rescoring, C is only 0.801. D would score 0.827, but a fixed shortlist never checks it.';
}
$('toy-shortlist').addEventListener('click',()=>{toyCertified=false;renderToy();});
$('toy-cairn').addEventListener('click',()=>{toyCertified=true;renderToy();});
$('near-ties').addEventListener('change',renderToy);
renderToy();

let measurements;
function renderHits(target,hits,gold,missing) {
  $(target).innerHTML = hits.map((hit,rank)=>{
    const doc = measurements.documents[String(hit.id)] || {};
    const isMissing = missing.has(hit.id);
    const tag = isMissing ? '<span class="miss-tag">MISSED BY SELECTED PATH</span>' : !gold.has(hit.id) ? '<span class="miss-tag">OUTSIDE EXACT TOP 10</span>' : '';
    return `<details class="hit ${isMissing?'lost':''}"><summary><span class="rank">${rank+1}</span><span><span class="hit-title">${esc(doc.title||'Document '+hit.id)}${tag}</span><span class="hit-meta">row ${hit.id} · source ${esc(doc._id||'unknown')} · score ${hit.score.toFixed(6)} <span aria-hidden="true">↴</span></span></span></summary><p class="hit-text">${esc(doc.text||'No excerpt is included for this document.')}<br><span class="hit-meta">Shortened dataset excerpt. Numerical rank does not establish that this document supports the query’s claim.</span></p></details>`;
  }).join('');
}
function syncExampleUrl() {
  const url = new URL(location.href);
  url.searchParams.set('query', $('queries').value);
  url.searchParams.set('method', $('mode').value);
  history.replaceState(null,'',url);
}
function renderExample(updateUrl=false) {
  const query = measurements.queries.find(q=>String(q.query_id)===$('queries').value) || measurements.queries[0];
  const methodKey = $('mode').value;
  const method = query.methods[methodKey];
  const reference = query.methods.exhaustive;
  const missed = new Set(method.missed_ids);
  const expected = new Set(reference.hits.map(hit=>hit.id));
  const kept = reference.hits.length-method.missed_ids.length;
  $('query-text').textContent = query.query.text;
  $('kept').textContent = `${kept} / ${reference.hits.length}`;
  $('kept').className = missed.size?'bad':'good';
  $('scored').textContent = `${number(method.stats.exact_scored)} / ${number(method.stats.eligible)}`;
  const hasRelevance = method.ndcg_at_k!=null && reference.ndcg_at_k!=null;
  $('relevance').textContent = hasRelevance ? `${method.ndcg_at_k.toFixed(3)} vs ${reference.ndcg_at_k.toFixed(3)} exact` : 'Not available';
  let statement = methodKey==='certified'
    ? `Cairn scanned all ${number(method.stats.eligible)} compressed vectors and fully scored ${number(method.stats.exact_scored)} originals. Its returned order matches the exhaustive reference for this recorded query.`
    : missed.size
    ? `This path misses ${missed.size} of the exact top-10 neighbors after rescoring its fixed candidate pool. The highlighted documents were not returned.`
    : 'This fixed candidate budget recovers every exact top-10 neighbor for this query. Approximate search can work well; this result alone is not a guarantee for other queries.';
  if (hasRelevance) {
    const delta = method.ndcg_at_k-reference.ndcg_at_k;
    statement += Math.abs(delta)<1e-9 ? ' The relevance score is unchanged here. A neighbor miss does not automatically mean a worse answer.' : delta>0 ? ' The selected path actually has a higher relevance score here. The embedding’s exact ranking is not the same thing as human relevance.' : ' The selected path has a lower relevance score here, based on the dataset’s separate relevance judgments.';
  }
  $('explanation').textContent = statement;
  renderHits('expected',reference.hits,expected,missed);
  renderHits('returned',method.hits,expected,new Set());
  const anyMiss = measurements.queries.some(q=>q.methods[methodKey].missed_ids.length);
  $('next-miss').disabled = !anyMiss;
  $('next-miss').textContent = anyMiss ? 'Next miss →' : 'No misses in these 12 examples';
  $('copy-status').textContent = '';
  if (updateUrl) syncExampleUrl();
}
function initializeExamples(data) {
  measurements=data;
  $('queries').innerHTML = data.queries.map(q=>`<option value="${q.query_id}">${q.query_id} · ${esc(q.query.text)}</option>`).join('');
  $('mode').innerHTML = Object.keys(data.summary).filter(key=>key!=='exhaustive').map(key=>`<option value="${key}">${esc(names[key]||key)}</option>`).join('');
  const params=new URLSearchParams(location.search);
  $('queries').value=data.queries.some(q=>String(q.query_id)===params.get('query'))?params.get('query'):'29';
  if (!$('queries').value) $('queries').selectedIndex=0;
  const mode=params.get('method');
  $('mode').value=mode && data.summary[mode] && mode!=='exhaustive'?mode:'binary_4x';
  $('queries').addEventListener('change',()=>renderExample(true));
  $('mode').addEventListener('change',()=>renderExample(true));
  $('next-miss').addEventListener('click',()=>{
    const start=data.queries.findIndex(q=>String(q.query_id)===$('queries').value);
    for(let offset=1;offset<=data.queries.length;offset++) {
      const query=data.queries[(start+offset)%data.queries.length];
      if(query.methods[$('mode').value].missed_ids.length) {
        $('queries').value=String(query.query_id);renderExample(true);break;
      }
    }
  });
  $('copy-example').addEventListener('click',async()=>{
    const url=new URL(location.href);url.hash='example';url.searchParams.set('query',$('queries').value);url.searchParams.set('method',$('mode').value);
    try {await navigator.clipboard.writeText(url.href);$('copy-status').textContent='Link copied.';}
    catch {$('copy-status').textContent=`Copy from the address bar: ${url.href}`;history.replaceState(null,'',url);}
  });
  $('path-table').tBodies[0].innerHTML=Object.entries(data.summary).map(([key,s])=>`<tr class="${key==='certified'?'selected':''}"><td>${esc(names[key]||key)}</td><td>${percent(s.recall_at_k)}</td><td>${s.queries_with_misses} / ${data.protocol.queries}</td><td>${ms(s.p50_ms)}</td><td>${number(s.mean_exact_vectors)}</td><td>${s.ndcg_at_k==null?'n/a':s.ndcg_at_k.toFixed(4)}</td></tr>`).join('');
  $('explorer').setAttribute('aria-busy','false');
  renderExample();
}
let benchmarks;
function renderBenchmark() {
  const key=$('benchmark-dataset').value;
  const data=benchmarks[key], certified=data.summary.certified, faiss=data.faiss;
  const view=$('timing-view').value;
  let cairn=certified.p50_ms,baseline=faiss.p50_ms,firstLabel='Cairn · native',secondLabel='FAISS · one query';
  if(view==='roundtrip'){cairn=certified.roundtrip_p50_ms;firstLabel='Cairn · Python + IPC';}
  if(view==='p95'){cairn=certified.p95_ms;baseline=faiss.p95_ms;firstLabel='Cairn · native p95';secondLabel='FAISS · p95';}
  if(view==='batch'){baseline=faiss.all_queries_batch_per_query_ms;firstLabel='Cairn · single query';secondLabel='FAISS · batch / query';}
  const maximum=Math.max(cairn,baseline);
  $('timing-chart').innerHTML=[[firstLabel,cairn,''],[secondLabel,baseline,'faiss']].map(([label,time,cls])=>`<div class="timing-row ${cls}"><span class="timing-name">${label}</span><div class="timing-track" aria-hidden="true"><div class="timing-fill" style="width:${time/maximum*100}%"></div></div><strong>${ms(time)}</strong></div>`).join('');
  $('benchmark-context').textContent=`${datasetNames[key]} · ${number(data.index.layout.rows)} × ${data.index.layout.dimensions} · ${data.protocol.queries} queries · warm cache`;
  let takeaway;
  if(view==='batch') {
    takeaway=baseline<cairn ? 'FAISS wins this throughput comparison. It processes the query set as a batch, while Cairn processes one query at a time. If you need bulk ground truth, use a tuned batched baseline before choosing Cairn.' : 'This million-vector batch was unexpectedly slow in the recorded run. It does not establish a bulk-throughput advantage over tuned FAISS. Batching wins clearly on the four smaller datasets.';
  } else if(view==='p95' && key==='cohere-wiki-1m') {
    takeaway='The million-vector p95 is effectively tied. Cairn’s p99 was 198.65 ms, with substantial variability. The median improvement does not establish better tail latency.';
  } else if(cairn>baseline) {
    takeaway=`FAISS is faster in this view. ${view==='roundtrip'?'Python serialization and IPC erase Cairn’s native advantage at this dataset size.':'This measurement favors the exhaustive FAISS baseline.'} Both methods matched the reference neighbors in this run.`;
  } else {
    takeaway=`Cairn takes ${(100*cairn/baseline).toFixed(0)}% of the FAISS time in this view. This is one laptop, warm cache, one query at a time and one FAISS CPU thread. It is not a claim about batched, GPU or production workloads.`;
  }
  $('benchmark-takeaway').textContent=takeaway;
  $('bench-agreement').textContent=`${data.ordered_exact_queries} / ${data.protocol.queries}`;
  $('bench-work').textContent=`${number(certified.mean_exact_vectors)} (${(100*certified.mean_exact_vectors/data.index.layout.rows).toFixed(3)}%)`;
  $('bench-size').textContent=`${(data.index.layout.file_bytes/1024**2).toFixed(1)} MiB`;
}
function initializeBenchmarks(data) {
  benchmarks=data;
  const order=['scifact','cqadupstack-programmers','fiqa','cohere-wiki-100k','cohere-wiki-1m'];
  $('benchmark-dataset').innerHTML=order.map(key=>`<option value="${key}">${datasetNames[key]} · ${number(data[key].index.layout.rows)} vectors</option>`).join('');
  $('benchmark-dataset').value='cohere-wiki-1m';
  $('benchmark-dataset').addEventListener('change',renderBenchmark);
  $('timing-view').addEventListener('change',renderBenchmark);
  $('bench-table').tBodies[0].innerHTML=order.map(key=>{const b=data[key],c=b.summary.certified,f=b.faiss;return `<tr><td>${datasetNames[key]}</td><td>${number(b.index.layout.rows)}</td><td>${b.protocol.queries}</td><td>${ms(c.p50_ms)} / ${ms(c.p95_ms)}</td><td>${ms(c.roundtrip_p50_ms)}</td><td>${ms(f.p50_ms)} / ${ms(f.p95_ms)}</td><td>${ms(f.all_queries_batch_per_query_ms)}</td></tr>`;}).join('');
  renderBenchmark();
}
async function loadData(path,initialize) {
  try {const response=await fetch(path);if(!response.ok)throw new Error(`HTTP ${response.status}`);initialize(await response.json());}
  catch(error){const host=$('data-error');host.hidden=false;host.textContent='The recorded data could not be loaded. Reload this page, or use the linked full benchmark documentation. '+error.message;}
}
loadData('scifact.json',initializeExamples);
loadData('benchmarks.json',initializeBenchmarks);
