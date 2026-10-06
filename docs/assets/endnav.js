/* bas de page commun : entrée en cascade à l'arrivée */
(function () {
  'use strict';
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduce || !('IntersectionObserver' in window)) return;
  var io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
  }, { rootMargin: '0px 0px -12% 0px' });
  document.querySelectorAll('.endnav').forEach(function (nav) {
    nav.classList.add('en-anim');
    nav.querySelectorAll('li').forEach(function (li, k) { li.style.transitionDelay = (0.25 + k * 0.07).toFixed(2) + 's'; });
    io.observe(nav);
  });
})();
