// Freeze the live Pod Lab page into one static HTML document (no scripts, styles inlined, canvases as images).
// Returns JSON: {html, rects}. `window.__freezeTargets` = {name: cssSelector | "text:Button label"} gives target rects.
(() => {
  const live = document.documentElement;
  const rectOf = el => { const r = el.getBoundingClientRect(); return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; };
  const find = spec => spec.startsWith('text:')
    ? [...document.querySelectorAll('button,a,label,summary,div,span,h3,b')].filter(e => e.offsetParent && e.textContent.trim() === spec.slice(5)).pop()
    : document.querySelector(spec);
  const rects = {};
  for (const [name, spec] of Object.entries(window.__freezeTargets || {})) { const el = find(spec); if (el) rects[name] = rectOf(el); }

  // Form state lives in properties, not attributes: copy it before cloning.
  live.querySelectorAll('input').forEach(i => {
    if (i.type === 'checkbox' || i.type === 'radio') i.toggleAttribute('checked', i.checked); else i.setAttribute('value', i.value);
  });
  live.querySelectorAll('textarea').forEach(t => { t.textContent = t.value; });
  live.querySelectorAll('select').forEach(s => [...s.options].forEach(o => o.toggleAttribute('selected', o.selected)));
  const canvases = [...live.querySelectorAll('canvas')].map(c => { try { return c.width && c.height ? c.toDataURL('image/png') : null; } catch (_) { return null; } });
  const scrolls = [...live.querySelectorAll('*')].filter(e => e.scrollTop || e.scrollLeft).map(e => [e, e.scrollTop, e.scrollLeft]);
  scrolls.forEach(([e], k) => e.setAttribute('data-freeze-scroll', k));

  const clone = live.cloneNode(true);
  scrolls.forEach(([e]) => e.removeAttribute('data-freeze-scroll'));
  clone.querySelectorAll('canvas').forEach((c, k) => {
    const src = live.querySelectorAll('canvas')[k];
    if (!canvases[k]) { c.remove(); return; }
    const img = document.createElement('img'); img.src = canvases[k];
    img.setAttribute('style', (src.getAttribute('style') || '') + `;width:${src.clientWidth}px;height:${src.clientHeight}px;display:block`);
    img.className = src.className; c.replaceWith(img);
  });
  clone.querySelectorAll('script,noscript,#landing,link[rel=preload],link[rel=modulepreload]').forEach(e => e.remove());

  // All styles, including rules inserted through the CSSOM (Plotly, theme), as one inline sheet.
  let css = '';
  for (const sheet of document.styleSheets) {
    try { css += [...sheet.cssRules].map(r => r.cssText).join('\n') + '\n'; }
    catch (_) { /* cross-origin (Google Fonts): keep its <link> */ }
  }
  clone.querySelectorAll('style,link[rel=stylesheet]').forEach(e => { if (!/fonts\.googleapis/.test(e.href || '')) e.remove(); });
  const style = document.createElement('style');
  style.textContent = css + '\nhtml,body{overflow:hidden!important;scroll-behavior:auto!important}*{pointer-events:none!important;caret-color:transparent!important;animation-play-state:paused!important;transition:none!important}';
  clone.querySelector('head').append(style);
  // Restore inner scroll positions once the frozen page loads (the only script kept, inline and tiny).
  const keep = document.createElement('script');
  keep.textContent = `addEventListener('load',()=>{const s=${JSON.stringify(scrolls.map(([, t, l]) => [t, l]))};document.querySelectorAll('[data-freeze-scroll]').forEach(e=>{const v=s[+e.dataset.freezeScroll];e.scrollTop=v[0];e.scrollLeft=v[1];});scrollTo(0,0)})`;
  clone.querySelector('body').append(keep);
  return JSON.stringify({html: '<!doctype html>\n' + clone.outerHTML, rects, scrollY: Math.round(scrollY)});
})()
