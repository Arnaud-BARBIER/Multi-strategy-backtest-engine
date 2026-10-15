(() => {
  // Keep the reading guide beside the page introduction on the long-form mode pages.
  // A few pages place it after their visual carousel in the source; move it into the
  // masthead so desktop can present it as a right-hand reading rail.
  document.querySelectorAll("article").forEach((article) => {
    const mast = article.querySelector(".mast");
    const map = article.querySelector(":scope > .reading-map");
    if (mast && map && !mast.contains(map)) mast.insertBefore(map, mast.querySelector(".figs") || null);
  });

  document.querySelectorAll(".mast").forEach((mast) => {
    const mastMap = mast.querySelector(":scope > .reading-map");
    if (!mastMap || mast.querySelector(":scope > .mast-copy")) return;

    const copy = document.createElement("div");
    copy.className = "mast-copy";
    mast.insertBefore(copy, mastMap);
    mast.querySelectorAll(":scope > h1, :scope > .lede, :scope > .lede2, :scope > .lede-note, :scope > .small")
      .forEach((element) => copy.appendChild(element));
  });

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
