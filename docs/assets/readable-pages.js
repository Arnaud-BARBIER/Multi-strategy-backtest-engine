(() => {
  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !("IntersectionObserver" in window)) return;

  const targets = document.querySelectorAll(
    ".site-readable .mast, .site-readable section, .site-readable figure, .site-readable .note, .site-readable .subject"
  );
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -5% 0px" });

  targets.forEach((element) => {
    element.classList.add("scroll-reveal");
    observer.observe(element);
  });

  window.addEventListener("page-language-change", () => {
    targets.forEach((element) => {
      if (!element.classList.contains("is-visible")) observer.observe(element);
    });
  });
})();
