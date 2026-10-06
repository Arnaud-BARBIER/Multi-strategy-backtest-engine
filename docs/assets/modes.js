(function () {
  'use strict';
  var page = document.getElementById('page');
  var buttons = document.querySelectorAll('[data-set]');
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var hasIO = 'IntersectionObserver' in window;

  /* langue */
  function setLanguage(lang, keepSection) {
    if (lang !== 'fr' && lang !== 'en') return;
    var hash = window.location.hash;
    var targetHash = hash.replace(/-(fr|en)-/, '-' + lang + '-');
    page.dataset.lang = lang;
    document.documentElement.lang = lang;
    document.querySelectorAll('main > article[lang]').forEach(function (article) {
      article.hidden = article.lang !== lang;
    });
    buttons.forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.dataset.set === lang));
    });
    document.title = page.dataset['title' + (lang === 'fr' ? 'Fr' : 'En')];
    try { localStorage.setItem('ab-lang', lang); } catch (_) {}
    if (targetHash !== hash && document.getElementById(targetHash.slice(1))) {
      history.replaceState(null, '', targetHash);
      if (keepSection) document.getElementById(targetHash.slice(1)).scrollIntoView();
    }
  }
  buttons.forEach(function (button) {
    button.addEventListener('click', function () { setLanguage(button.dataset.set, true); });
  });
  var hashLanguage = location.hash.match(/-(fr|en)-/);
  var saved = 'fr';
  try { saved = localStorage.getItem('ab-lang') || 'fr'; } catch (_) {}
  setLanguage(hashLanguage ? hashLanguage[1] : saved, false);
  window.addEventListener('hashchange', function () {
    var match = location.hash.match(/-(fr|en)-/);
    if (match && match[1] !== page.dataset.lang) setLanguage(match[1], true);
  });

  /* titre : apparition mot par mot */
  document.querySelectorAll('.mast h1').forEach(function (h) {
    if (reduce) { h.classList.add('on'); return; }
    var words = h.textContent.trim().split(/\s+/);
    h.setAttribute('aria-label', h.textContent.trim());
    h.innerHTML = words.map(function (w, k) {
      return '<span class="w" aria-hidden="true" style="transition-delay:' + (0.05 + k * 0.045).toFixed(3) + 's">' + w + '</span>';
    }).join(' ');
    requestAnimationFrame(function () { requestAnimationFrame(function () { h.classList.add('on'); }); });
  });

  /* apparition au défilement */
  var targets = document.querySelectorAll('.figs > div, .apercu, .reading-map li, main section > *:not(.shead), .flow li');
  if (hasIO && !reduce) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -6% 0px' });
    targets.forEach(function (el) {
      if (el.closest('.flow') && !el.matches('.flow li')) return;
      el.classList.add('reveal'); io.observe(el);
    });
  }

  /* compteurs */
  document.querySelectorAll('[data-count]').forEach(function (el) {
    var to = Number(el.getAttribute('data-count'));
    var lang = el.closest('[lang]') ? el.closest('[lang]').lang : 'fr';
    function fmt(v) { return Math.round(v).toLocaleString(lang === 'en' ? 'en-GB' : 'fr-FR'); }
    if (!hasIO || reduce || !isFinite(to)) return;
    var co = new IntersectionObserver(function (es) {
      if (!es[0].isIntersecting) return; co.disconnect();
      var t0 = performance.now();
      (function tick(t) {
        var q = Math.min(1, Math.max(0, (t - t0) / 1300));
        el.textContent = fmt(to * (1 - Math.pow(1 - q, 3)));
        if (q < 1) requestAnimationFrame(tick);
      })(t0);
    }, { threshold: 0.6 });
    el.textContent = fmt(0); co.observe(el);
  });

  /* aperçu : défilement automatique, pause au survol, reprise à la demande */
  document.querySelectorAll('.apercu').forEach(function (ap) {
    var tabs = Array.prototype.slice.call(ap.querySelectorAll('.ap-tabs button'));
    var slides = Array.prototype.slice.call(ap.querySelectorAll('.ap-slide'));
    var pause = ap.querySelector('.ap-pause');
    var cur = 0;
    function show(i) {
      cur = (i + slides.length) % slides.length;
      tabs.forEach(function (t, k) {
        t.setAttribute('aria-selected', String(k === cur));
        t.tabIndex = k === cur ? 0 : -1;
        var bar = t.querySelector('.bar');
        if (bar) { bar.style.animation = 'none'; void bar.offsetWidth; bar.style.animation = ''; }
      });
      slides.forEach(function (s, k) { s.classList.toggle('is-on', k === cur); s.setAttribute('aria-hidden', String(k !== cur)); });
    }
    function setAuto(on) {
      ap.classList.toggle('auto', on);
      if (pause) pause.textContent = on ? pause.dataset.pause : pause.dataset.play;
    }
    tabs.forEach(function (t, k) {
      t.addEventListener('click', function () { setAuto(false); show(k); });
      t.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          e.preventDefault(); setAuto(false); show(cur + (e.key === 'ArrowRight' ? 1 : -1)); tabs[cur].focus();
        }
      });
      var bar = t.querySelector('.bar');
      if (bar) bar.addEventListener('animationend', function () { if (ap.classList.contains('auto')) show(cur + 1); });
    });
    if (pause) pause.addEventListener('click', function () { setAuto(!ap.classList.contains('auto')); show(cur); });
    ap.addEventListener('mouseenter', function () { ap.classList.add('hold'); });
    ap.addEventListener('mouseleave', function () { ap.classList.remove('hold'); });
    if (hasIO) new IntersectionObserver(function (es) { ap.classList.toggle('off', !es[0].isIntersecting); }, { threshold: 0.3 }).observe(ap);
    setAuto(!reduce);
    show(0);
  });

  /* agrandir une capture */
  var lb = document.createElement('dialog');
  lb.className = 'lb';
  lb.innerHTML = '<img alt=""><button type="button"></button>';
  document.body.appendChild(lb);
  var lbImg = lb.querySelector('img'), lbBtn = lb.querySelector('button');
  function closeLb() { if (lb.close) lb.close(); else lb.removeAttribute('open'); }
  lbBtn.addEventListener('click', closeLb);
  lb.addEventListener('click', function (e) { if (e.target === lb) closeLb(); });
  document.querySelectorAll('.shot img, .ap-frame img').forEach(function (img) {
    img.tabIndex = 0;
    function open() {
      lbImg.src = img.currentSrc || img.src; lbImg.alt = img.alt;
      lbBtn.textContent = page.dataset.lang === 'en' ? 'Close' : 'Fermer';
      var ap = img.closest('.apercu'); if (ap) ap.classList.add('hold');
      if (lb.showModal) lb.showModal(); else lb.setAttribute('open', '');
    }
    img.addEventListener('click', open);
    img.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  });
  lb.addEventListener('close', function () { document.querySelectorAll('.apercu.hold').forEach(function (a) { if (!a.matches(':hover')) a.classList.remove('hold'); }); });
})();
