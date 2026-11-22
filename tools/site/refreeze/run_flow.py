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
def run(sec=14):
    ev('(()=>{runNow(true);return 1})()'); wait(sec)
    ev('(()=>{__h.layout(); return document.getElementById("state").textContent})()'); wait(6)
    ev('__h.tiles()')


ev("localStorage.setItem('podlab_lang','fr'); localStorage.setItem('podlab.theme.terminal.v1',JSON.stringify({accent:'#29c5ff',updown:'theme'})); localStorage.setItem('podlab.theme.editor.v1',JSON.stringify({accent:'#29c5ff',updown:'theme'})); location.reload(); 1")
wait(6)
ev("(()=>{const l=document.getElementById('landing'); if(l) l.remove(); scrollTo(0,0); runNow(true); return document.getElementById('template').value})()")
wait(25)
ev("window.__sec=structuredClone(reportSections); window.__page=structuredClone(reportPage); 1")
ev(HELP)
ev("(()=>{const t=document.getElementById('template'); t.value='max_sharpe'; t.dispatchEvent(new Event('change',{bubbles:true})); return t.value})()")
wait(6)
ev("(()=>{const l=__h.leaf('DEUIDXEUR'); const box=__h.up(l, e=>e.querySelector('input[type=checkbox]')).querySelector('input[type=checkbox]'); if(box.checked) box.click(); return box.checked})()")
wait(4)
run(25); run(10)
ev("""(async()=>{const s=r=>new Promise(f=>setTimeout(f,r));
  const t=(__h.leaf('Exécution · USA500IDXUSD')||__h.leaf('Execution · USA500IDXUSD')); const cell=__h.up(t, e=>e.querySelector('.report-execution-gear')); cell.querySelector('.report-execution-gear').click(); await s(800);
  const open=[...document.querySelectorAll('.report-indicators-open')].find(b=>b.offsetParent); open.click(); await s(800);
  const rows=[...document.querySelectorAll('.report-indicator-pane-toggle')].filter(r=>r.offsetParent);
  const row=rows.find(r=>/individual_sharpe/.test(r.textContent)); const box=row.querySelector('input'); if(!box.checked) box.click(); await s(2500);
  document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true})); document.body.click(); await s(800);
  return rows.map(r=>r.textContent+':'+r.querySelector('input').checked).join(' | ')})()""")
wait(4)
ev("(()=>{const ai=document.querySelector('#settings .settings-ai'); if(ai) ai.open=false; document.getElementById('settings').scrollTop=0; __h.mark(__h.leaf('Créer une stratégie'),'create'); return 1})()")
ev("(async()=>{const s=r=>new Promise(f=>setTimeout(f,r));for(let y=0;y<document.documentElement.scrollHeight;y+=450){scrollTo(0,y);await s(700);}scrollTo(0,0);await s(800);return 1})()")
wait(3)
freeze('run', {'create': '[data-freeze=create]'})
json.dump(S, open(f'{SP}/flow/run_flow.json', 'w'))
