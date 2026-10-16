'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number(value).toLocaleString('en-US', {maximumFractionDigits: 1});
const percent = value => `${(100 * value).toFixed(1)}%`;
const ms = value => `${Number(value).toFixed(value < 1 ? 3 : 2)} ms`;
const names = {exhaustive:'Exhaustive reference', certified:'Cairn · exact reference', binary_1x:'Binary · 10 candidates', binary_4x:'Binary · 40 candidates', binary_16x:'Binary · 160 candidates', int8_1x:'Int8 · 10 candidates', int8_4x:'Int8 · 40 candidates', int8_16x:'Int8 · 160 candidates'};
const datasetNames = {scifact:'SciFact', 'cqadupstack-programmers':'Programmers', fiqa:'FiQA', 'cohere-wiki-100k':'Wikipedia 100k', 'cohere-wiki-1m':'Wikipedia 1m'};
function stepper(prefix, labels, nextLabels, render, finish) {
  let step = 0;
  const nav = $(`${prefix}-steps`);
  nav.innerHTML = labels.map((label, i) => `<button type="button" data-step="${i}" aria-label="Step ${i + 1}: ${esc(label)}"><span>${i + 1}</span>${esc(label)}</button>`).join('');
  function show(next, interacted = false) {
    step = Math.max(0, Math.min(labels.length - 1, next));
    $(`${prefix}-progress`).textContent = `Step ${step + 1} of ${labels.length}`;
    [...nav.children].forEach((button, i) => {
      if (i === step) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
    $(`${prefix}-back`).disabled = step === 0;
    $(`${prefix}-restart`).disabled = step === 0;
    $(`${prefix}-next`).disabled = false;
    $(`${prefix}-next`).hidden = step === labels.length - 1 && !finish;
    $(`${prefix}-next`).textContent = nextLabels[step] || 'Next →';
    render(step);
    if (interacted) {
      const title = $(`${prefix}-title`);
      title.tabIndex = -1;
      title.focus({ preventScroll: true });
      title.scrollIntoView({ block: 'nearest' });
    }
  }
  nav.addEventListener('click', event => {
    const button = event.target.closest('button[data-step]');
    if (button) show(Number(button.dataset.step), true);
  });
  $(`${prefix}-back`).addEventListener('click', () => show(step - 1, true));
  $(`${prefix}-restart`).addEventListener('click', () => show(0, true));
  $(`${prefix}-next`).addEventListener('click', () => {
    if (step === labels.length - 1 && finish) finish();
    else show(step + 1, true);
  });
  show(0);
  return { refresh: () => show(step) };
}

function initializeAuditLesson(data) {
  const query = data.queries.find(q => q.query_id === 29);
  if (!query) throw new Error('The guided query is missing from the recorded data.');
  const fast = query.methods.binary_4x, exact = query.methods.certified;
  const reference = query.methods.exhaustive, int8 = query.methods.int8_4x;
  const total = reference.hits.length, missing = new Set(fast.missed_ids);
  const kept = total - missing.size;
  const firstMissing = reference.hits.find(hit => missing.has(hit.id));
  const missedDocument = data.documents[String(firstMissing.id)];
  const missedRank = reference.hits.findIndex(hit => hit.id === firstMissing.id) + 1;
  const referenceRibbon = () => `<div class="rank-ribbon" aria-label="Exact ranking with missing documents highlighted">${reference.hits.map((hit, i) => `<div class="rank-tile ${missing.has(hit.id) ? 'absent' : 'present'}"><b>${i + 1}</b><span>${missing.has(hit.id) ? 'Missing' : 'Kept'}</span></div>`).join('')}</div>`;
  const comparison = (name, method) => `<div class="comparison-row"><span>${name}</span><strong class="${method.missed_ids.length ? 'bad' : 'good'}">${total - method.missed_ids.length} / ${total}</strong><small>${method.stats.exact_scored} originals scored</small></div>`;
  const titles = ['Start with one search', 'The fast search returns ten papers', 'Cairn finds four missing matches', 'Check whether the misses matter', 'Use the audit to choose a search setting'];
  const copy = [
    'We want the ten closest papers about a topic. Search compares embeddings: lists of numbers that represent the query and each document. Higher similarity scores rank first.',
    `This method scans tiny binary versions of the embeddings, picks ${fast.stats.exact_scored} candidates, then scores their original vectors to return ${fast.hits.length} papers. A candidate is simply a document on that shortlist.`,
    `Cairn computes an exact reference from the same embeddings. Only ${kept} of its top ${total} appear in the fast result. The four highlighted positions below are missing from the fast search.`,
    'Cairn returns the complete exact top ten. But a closer vector is not automatically a more useful paper. The dataset has separate relevance judgments we can use to check that.',
    'Try a less aggressive shortcut before replacing your search engine. Int8 keeps more detail than binary compression. Here its shortlist recovers every exact match. Cairn gives you the reference to make that comparison.'
  ];
  const takeaways = [
    'The question: did a faster search change which documents we get?',
    'Ten returned results can still leave out some of the closest ten.',
    'A reranker can reorder a shortlist. It cannot recover a document that never made that shortlist.',
    'This example establishes missing vector neighbors. It does not establish better answers or less full-vector work.',
    'Your workflow: export embeddings and results → audit the differences → decide whether to tune the search.'
  ];
  stepper('audit', ['Task', 'Search', 'Audit', 'Impact', 'Decision'], ['See the fast search →', 'Check it with Cairn →', 'Do the misses matter? →', 'Choose a next step →', 'Explore other settings ↓'], step => {
    $('audit-title').textContent = titles[step];
    $('audit-copy').textContent = copy[step];
    $('audit-takeaway').textContent = takeaways[step];
    const scenes = [
      `<div class="task-card"><p class="eyebrow">Search topic, in plain language</p><p class="topic">Papers about combining classroom and web-based collaborative learning</p><div class="task-facts"><span><b>${number(exact.stats.eligible)}</b> papers in the collection</span><span><b>${total}</b> closest matches wanted</span></div><details><summary>See the original benchmark query</summary><p>${esc(query.query.text)}</p><p class="small muted">This claim is a search input from SciFact, not a statement that it is true. Query 29 in the saved audit.</p></details></div>`,
      `<div class="task-card"><p class="eyebrow">Fast search / returned result positions</p><div class="rank-ribbon">${fast.hits.map((hit, i) => `<div class="rank-tile"><b>${i + 1}</b><span>Returned</span></div>`).join('')}</div><p class="scene-note">The list looks complete. We have not compared it with the exact ranking yet.</p><details><summary>Inspect the first returned paper</summary><p>${esc(data.documents[String(fast.hits[0].id)].title)}</p><p class="small muted">This paper is present in both results. The missed matches are farther down the ranking.</p></details></div>`,
      `<div class="task-card"><p class="eyebrow">Exact reference / positions 1 to ${total}</p>${referenceRibbon()}<div class="missing-document"><span class="miss-tag">One of four missing matches</span><h4>The paper at exact rank #${missedRank} is absent.</h4><p>Its full-vector score puts it in the closest ten, but the fast search leaves it out.</p><details><summary>Inspect this missing paper</summary><p>${esc(missedDocument.title)}</p><p class="small muted">Full-vector score: ${firstMissing.score.toFixed(6)}. Its title may look off-topic: a close vector match is not a relevance judgment. We check that next.</p></details></div></div>`,
      `<div class="comparison-card"><p class="eyebrow">Exact top-ten matches recovered</p>${comparison('Binary compressed search', fast)}${comparison('Cairn exact search', exact)}<div class="relevance-note"><strong>Judged relevance is unchanged.</strong><p>Both score ${fast.ndcg_at_k.toFixed(3)} on this query’s relevance metric (nDCG@10). Higher is better. Recovering these four neighbors did not improve that metric.</p></div></div>`,
      `<div class="comparison-card"><p class="eyebrow">Closest ten recovered / same query</p>${comparison('Binary · 40 candidates', fast)}${comparison('Int8 · 40 candidates', int8)}${comparison('Cairn · exact reference', exact)}<p class="scene-note">The fixed int8 setting also recovers every exact neighbor across all ${data.protocol.queries} queries in this dataset. Cairn’s adaptive checking matters when that fixed shortlist is not enough, illustrated separately below.</p></div>`
    ];
    $('audit-scene').innerHTML = scenes[step];
  }, () => {
    $('explore-details').open = true;
    $('explore-details').querySelector('summary').focus();
    $('explore-details').scrollIntoView({ block: 'start' });
  });
  $('audit-lesson').setAttribute('aria-busy', 'false');
}

let toyStep = 0;
function renderToy(step = toyStep) {
  toyStep = step;
  if (step !== 4) $('near-ties').checked = false;
  const ties = $('near-ties').checked;
  const estimates = ties ? [.810,.808,.806,.804,.802,.800] : [.920,.880,.820,.800,.650,.310];
  const scores = ties ? [.812,.807,.801,.827,.815,.825] : [.923,.876,.801,.827,.645,.315];
  const errors = ties ? [.025,.025,.025,.025,.025,.025] : [.010,.010,.025,.030,.015,.015];
  const rows = estimates.map((estimate, i) => ({id:'ABCDEF'[i], estimate, score:scores[i], max:estimate + errors[i], scored: ties || (step >= 1 && i < 3) || (step >= 3 && i === 3)}));
  const selected = step === 0 ? rows.slice(0, 3) : rows.filter(r => r.scored).sort((a, b) => b.score - a.score).slice(0, 3);
  const chosen = new Set(selected.map(r => r.id));
  const titles = ['Start with cheap estimates', 'Check the first three originals', 'Could D beat our weakest result?', 'Check D. It really does beat C.', 'Skip only what cannot change the answer'];
  const copy = [
    'We want the best three of six documents. Compression makes scoring cheaper but introduces error. The estimates put A, B and C first. We have not checked their full scores yet.',
    'The full scores are 0.923, 0.876 and 0.801. C is the weakest of our three results. A fixed shortlist would stop here, even if another document could beat C.',
    'D’s estimate is only 0.800, but its error allowance is 0.030. Its full score could be as high as 0.830. That is above C’s 0.801, so skipping D would be unsafe.',
    'D’s full score is 0.827. It replaces C, which scored 0.801. Checking one extra original changed the answer from A, B, C to A, B, D.',
    'E can score at most 0.665 and F at most 0.325. Neither can beat D’s 0.827. Cairn can skip their originals and still return the exact best three.'
  ];
  $('toy-title').textContent = ties ? 'If all bounds overlap, check all six.' : titles[step];
  $('toy-copy').textContent = ties ? 'Now every remaining document could beat the current third-place score. Cairn checks all six originals and returns D, F, E. Preserving the exact result can require exhaustive work.' : copy[step];
  $('toy-rows').innerHTML = rows.map((row, i) => {
    let decision = 'Not checked';
    if (row.scored) decision = `${row.score.toFixed(3)}${step >= 3 && !chosen.has(row.id) ? ' · outside top 3' : ''}`;
    else if (step >= 4) decision = 'Safe to skip';
    else if (step === 2 && i === 3) decision = 'Could beat C';
    const showBound = (step === 2 && i === 3) || step >= 4;
    return `<div class="toy-row ${step >= 2 && i === 3 ? 'recovered' : ''}" role="row"><span class="vector-name" role="cell">${row.id}</span><div role="cell"><span class="estimate-value">${row.estimate.toFixed(3)}</span>${showBound ? `<small class="bound-caption">At most ${row.max.toFixed(3)}</small>` : ''}</div><span class="decision" role="cell">${decision}</span></div>`;
  }).join('');
  $('toy-label').textContent = step === 0 ? 'Initial shortlist, by estimate' : 'Best full scores checked so far';
  $('toy-answer').innerHTML = selected.map(row => `<span class="${step >= 3 ? 'found' : ''}">${row.id}</span>`).join('');
  $('toy-count').textContent = rows.filter(row => row.scored).length;
  $('toy-explanation').textContent = ties ? 'Exact result: D, F, E. No full-vector work saved in this case.' : [
    'A shortlist is a starting point. Its order may change after full scoring.',
    'Current cutoff: 0.801. A new candidate must beat C to enter the top three.',
    'D’s maximum: 0.830 > 0.801. We must check its original.',
    'New cutoff: 0.827. D has replaced C.',
    'Exact result: A, B, D. Four full scores instead of six.'
  ][step];
  $('tie-option').hidden = step !== 4;
  $('tie-explanation').textContent = ties ? 'All six originals were checked. The algorithm preserves the ranking, not a promise to always do less work.' : 'When the bounds cannot rule anything out, exactness costs more work.';
}
const toyGuide = stepper('toy', ['Estimate', 'Score', 'Check bound', 'Replace', 'Finish'], ['Score the first three →', 'Check D’s possibility →', 'Score D’s original →', 'Can we skip E and F? →'], renderToy);
$('near-ties').addEventListener('change', () => toyGuide.refresh());


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
  initializeAuditLesson(data);
  $('queries').innerHTML = data.queries.map(q=>`<option value="${q.query_id}">${q.query_id} · ${esc(q.query.text)}</option>`).join('');
  $('mode').innerHTML = Object.keys(data.summary).filter(key=>key!=='exhaustive').map(key=>`<option value="${key}">${esc(names[key]||key)}</option>`).join('');
  const params=new URLSearchParams(location.search);
  $('queries').value=data.queries.some(q=>String(q.query_id)===params.get('query'))?params.get('query'):'29';
  if (!$('queries').value) $('queries').selectedIndex=0;
  const mode=params.get('method');
  $('mode').value=mode && Object.hasOwn(data.summary, mode) && mode!=='exhaustive'?mode:'binary_4x';
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
    const url=new URL(location.href);url.hash='explorer';url.searchParams.set('query',$('queries').value);url.searchParams.set('method',$('mode').value);
    try {await navigator.clipboard.writeText(url.href);$('copy-status').textContent='Link copied.';}
    catch {$('copy-status').textContent=`Copy from the address bar: ${url.href}`;history.replaceState(null,'',url);}
  });
  $('path-table').tBodies[0].innerHTML=Object.entries(data.summary).map(([key,s])=>`<tr class="${key==='certified'?'selected':''}"><td>${esc(names[key]||key)}</td><td>${percent(s.recall_at_k)}</td><td>${s.queries_with_misses} / ${data.protocol.queries}</td><td>${ms(s.p50_ms)}</td><td>${number(s.mean_exact_vectors)}</td><td>${s.ndcg_at_k==null?'n/a':s.ndcg_at_k.toFixed(4)}</td></tr>`).join('');
  $('explorer').setAttribute('aria-busy','false');
  renderExample();
  if (params.has('query') || params.has('method') || location.hash === '#explorer') {
    $('explore-details').open = true;
    if (location.hash === '#example' || location.hash === '#explorer') {
      requestAnimationFrame(() => $('explorer').scrollIntoView({ block: 'start' }));
    }
  }
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
