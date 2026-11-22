"""Freeze flow for the demo's AI part: paste 1, run, tweak, copy results, paste 2, benchmarks."""
import json, sys
SP = sys.argv[1]
FREEZE = open('/Users/arnaudbarbier/Quant reaserch/Multi-strategy-backtest-engine/tools/site/freeze.js').read()
a1, a2 = open(f'{SP}/ai/answer1.md').read(), open(f'{SP}/ai/answer2.md').read()
NAME = 'Mon portefeuille'
HELP = r"""
window.__h = {
  leaf: (t, root=document) => [...root.querySelectorAll('*')].filter(e => e.offsetParent && [...e.childNodes].some(n => n.nodeType === 3 && n.textContent.trim() === t)).pop(),
  up: (el, ok) => { while (el && !ok(el)) el = el.parentElement; return el; },
  slider: name => __h.up(__h.leaf(name, document.getElementById('settings')), e => e.querySelector('input[type=range]')).querySelector('input[type=range]'),
  mark: (el, name) => { if (el) el.setAttribute('data-freeze', name); return !!el; },
  layout: () => { if (window.__sec) { reportSections = structuredClone(__sec); reportPage = structuredClone(__page); renderDashboard(); } },
  clock: () => { const c = document.getElementById('use-strategy-clock'); if (!c.checked) c.click(); return c.checked; },
  equity: () => __h.up(__h.leaf('Equity'), e => e.getBoundingClientRect().height > 300),
  tiles: () => [...document.querySelectorAll('[class*=tile]')].map(e => e.innerText.replace(/\n/g, ' ')).filter(x => x.length < 70 && x.length > 3).slice(0, 12).join(' | '),
  sideTo: (el, top=70) => { const s = document.getElementById('settings'); s.scrollTop += el.getBoundingClientRect().top - top; return s.scrollTop; },
  commit: async () => { const n = %s; const d = await (await fetch('/api/editor/draft?name=' + encodeURIComponent(n))).json();
    const r = await (await fetch('/api/editor/save', {method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({name: n, source: d.source, settings: d.settings, contracts: d.contracts})})).json();
    return r.version || r.error || JSON.stringify(r).slice(0, 300); },
};
1""" % json.dumps(NAME)

S = []
def ev(js): S.append(['eval', js])
def wait(s): S.append(['wait', s])
def freeze(name, targets):
    ev('scrollTo(0,0); window.__freezeTargets=' + json.dumps(targets) + '; 1'); wait(1)
    S.append(['save', FREEZE, f'{SP}/flow/{name}.json']); S.append(['shot', f'{SP}/flow/z_{name}.png'])
ALLIND = """(async()=>{const s=r=>new Promise(f=>setTimeout(f,r));
  const t=(__h.leaf('Exécution · USA500IDXUSD')||__h.leaf('Execution · USA500IDXUSD')); if(!t) return 'no panel'; const cell=__h.up(t, e=>e.querySelector('.report-execution-gear'));
  cell.querySelector('.report-execution-gear').click(); await s(700);
  const vis=e=>e.getClientRects().length>0; const open=[...document.querySelectorAll('.report-indicators-open')].find(vis); if(!open) return 'no indicators button'; open.click(); await s(700);
  const page=[...document.querySelectorAll('.report-indicators-page')].find(vis);
  const none=[...page.querySelectorAll('button')].find(b=>/^(None|Aucun)$/.test(b.textContent.trim())); none.click(); await s(1500);
  for (const r of [...page.querySelectorAll('.report-indicator-pane-toggle')]) { if(/^(Prix|Price|individual_sharpe)$/.test(r.textContent.trim())){ const c=r.querySelector('input'); if(!c.checked){ c.click(); await s(1500);} } }
  await s(2000);
  const res=[...page.querySelectorAll('.report-indicator-pane-toggle')].map(r=>r.textContent+':'+r.querySelector('input').checked).join(' | ');
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true})); await s(300); document.body.click(); await s(1500); return res})()"""
def run(sec=14):
    ev('(()=>{runNow(true);return 1})()'); wait(sec)
    ev('(()=>{__h.layout(); return document.getElementById("state").textContent})()'); wait(6)
    ev(ALLIND); wait(2)
    ev('__h.tiles()')

ev("localStorage.setItem('podlab_lang','fr'); localStorage.setItem('podlab.theme.terminal.v1',JSON.stringify({accent:'#29c5ff',updown:'theme'})); localStorage.setItem('podlab.theme.editor.v1',JSON.stringify({accent:'#29c5ff',updown:'theme'})); location.reload(); 1")
wait(6)
ev("(()=>{const l=document.getElementById('landing'); if(l) l.remove(); scrollTo(0,0); runNow(true); return document.getElementById('template').value})()")
wait(25)
ev("window.__sec=structuredClone(reportSections); window.__page=structuredClone(reportPage); 1")
ev(HELP)
ev("(()=>{const t=document.getElementById('template'); t.value='editor:%s'; t.dispatchEvent(new Event('change',{bubbles:true})); return 1})()" % NAME)
wait(8)
run(14)                                                  # warm: the 1/N white portfolio as the visitor left it
# ---- paste 1 (frozen before Apply: Apply empties the field) ----
ev("(()=>{const ai=document.querySelector('#settings .settings-ai'); ai.open=true; const f=document.getElementById('aipaste-text'); f.value=%s; f.dispatchEvent(new Event('input',{bubbles:true})); f.scrollTop=0; __h.sideTo(document.querySelector('#settings .settings-ai')); return 1})()" % json.dumps(a1))
wait(1)
freeze('p1', {'paste': '#aipaste-text', 'apply': '#aipaste-apply'})
ev("(()=>{document.getElementById('aipaste-apply').click(); return 1})()"); wait(8)
ev("document.getElementById('aipaste-status').textContent")
ev("(async()=>await __h.commit())()"); wait(2)
ev("(()=>{const t=document.getElementById('template'); t.dispatchEvent(new Event('change',{bubbles:true})); return 1})()"); wait(6)
ev("__h.clock()")
run(30); run(10)                                        # second run: warm time on screen
ev("(()=>JSON.stringify({editor: current.settings?.editor?.version, period: current.period || null, keys: Object.keys(current).slice(0,60)}))()")
# ---- a1: strategy settings in the panel, equity chart (OOS), drag frame 0 ----
ev("(()=>{const ai=document.querySelector('#settings .settings-ai'); ai.open=false; const p=__h.leaf('Paramètres', document.getElementById('settings')); __h.mark(p,'params'); __h.mark(__h.up(__h.leaf('sma slow'), e=>e.querySelector('input[type=range]')),'params_end'); __h.sideTo(p, 64); __h.mark(__h.slider('max weight'),'mw'); __h.mark(__h.equity(),'equity'); return 1})()")
wait(1)
freeze('a1', {'params': '[data-freeze=params]', 'params_end': '[data-freeze=params_end]', 'mw': '[data-freeze=mw]', 'equity': '[data-freeze=equity]', 'copy': '#copy-results'})
for k, v in enumerate((.30, .25, .20), 1):
    ev("(()=>{const r=__h.slider('max weight'); r.value=%s; r.dispatchEvent(new Event('input',{bubbles:true})); r.dispatchEvent(new Event('change',{bubbles:true})); return r.value})()" % v)
    run(12)
    if k == 3:   # Pod Lab draws a view when it comes into view: pass through the whole report first
        ev("(async()=>{const s=r=>new Promise(f=>setTimeout(f,r));for(let y=0;y<document.documentElement.scrollHeight;y+=450){scrollTo(0,y);await s(700);}scrollTo(0,0);await s(500);return [...document.querySelectorAll('body *')].filter(e=>e.children.length===0&&/se charge/.test(e.textContent)).length})()")
        wait(3)
    freeze(f'm{k}', {'mw': '[data-freeze=mw]', 'copy': '#copy-results', 'equity': '[data-freeze=equity]'})
ev("(()=>{const b=document.getElementById('keep-run'); const h=b.hidden; b.click(); return h})()"); wait(4)
ev("document.getElementById('keep-run').textContent")
ev("(()=>[...document.querySelectorAll('*')].filter(e=>e.scrollHeight>e.clientHeight+50&&getComputedStyle(e).overflowY!=='visible'||e===document.scrollingElement).map(e=>(e.id||e.className||e.tagName)+':'+e.scrollHeight+'/'+e.clientHeight).slice(0,12))()")
# ---- paste 2 ----
ev("(()=>{const ai=document.querySelector('#settings .settings-ai'); ai.open=true; const f=document.getElementById('aipaste-text'); f.value=%s; f.dispatchEvent(new Event('input',{bubbles:true})); f.scrollTop=0; __h.sideTo(ai); return 1})()" % json.dumps(a2))
wait(1)
freeze('p2', {'paste': '#aipaste-text', 'apply': '#aipaste-apply'})
ev("(()=>{document.getElementById('aipaste-apply').click(); return 1})()"); wait(8)
ev("document.getElementById('aipaste-status').textContent")
ev("(async()=>await __h.commit())()"); wait(2)
ev("(()=>{const t=document.getElementById('template'); t.dispatchEvent(new Event('change',{bubbles:true})); return 1})()"); wait(6)
ev("__h.clock()")
run(30); run(10)
ev("(()=>{const ai=document.querySelector('#settings .settings-ai'); ai.open=false; const p=__h.leaf('Paramètres', document.getElementById('settings')); __h.sideTo(p, 64); __h.mark(__h.equity(),'equity'); return 1})()")
wait(1)
freeze('a2', {'tiles': '#tiles', 'equity': '[data-freeze=equity]', 'compare': '#compare-stored'})
# ---- benchmarks: 1/N and iteration 1 (its last settings: the kept run) ----
ev("(()=>{document.getElementById('compare-stored').click(); return 1})()"); wait(5)
ev("(()=>[...document.querySelectorAll('#reference-options label')].map(l=>l.textContent))()")
ev("(()=>{const ls=[...document.querySelectorAll('#reference-options label')]; const want=l=>/eqw|Equal weight/.test(l.textContent)||/@v2 · r00[0-9] · .*kept/.test(l.textContent); ls.forEach(l=>{const c=l.querySelector('input'); if(c.checked!==want(l)) c.click();}); return ls.filter(l=>l.querySelector('input').checked).map(l=>l.textContent)})()")
wait(10)
ev("(()=>{__h.layout(); return 1})()"); wait(6)
ev(ALLIND); wait(2)
ev("(()=>{__h.mark(__h.equity(),'equity'); return 1})()")
freeze('bench', {'equity': '[data-freeze=equity]', 'options': '#reference-options', 'tiles': '#tiles'})

json.dump(S, open(f'{SP}/flow/flow.json', 'w'))
print(len(S),'steps')
