# FORGE English version + main site section (Codex session K6)

Two repos, both local, both live on GitHub Pages. One commit per item, do not push (the director checks, then pushes).
No browser needed; say what you could not check.

## K6-1 — English version of the FORGE page
Repo `Multi-strategy-backtest-engine`. Source `docs/next/index.html` (French). The home `docs/index.html` is generated
from it by `tools/site/make_home.py`.
- Create `docs/next/en/index.html`: the same page, every visible text in English — hero, ticker, schema, tour bubbles
  (`ALL_STEPS` / `TOUR` `n`/`h`/`p`), AI chat texts (`CHATS`), validation / editor / tests windows, join label, « next »
  prompts and the arrow-key notice, engine block, costs column (keep the terms already in English), 12 tests list,
  benchmark (numbers unchanged, decimal point « . », thin-space thousands « 1,480 » style), end of page and cards,
  `aria-label`s, `<title>` (« FORGE · Portfolio allocation lab »), `lang="en"`. Brand stays FORGE.
- Paths from `docs/next/en/`: frozen states `../states/…`, `../hero.jpg`, old pages `../../research/` etc., old home
  `../../ancien.html`. The frozen terminal captures stay in French: add one discreet line under the demo intro
  « Screens shown in French; the interface also runs in English. ».
- Language switch in the top bar of both pages: « FR · EN », current one highlighted (accent), linking to the other
  (`docs/next/index.html` ↔ `docs/next/en/index.html`).
- Extend `tools/site/make_home.py` so it also writes `docs/en/index.html` from the English page (paths rewritten for
  `docs/en/`: states `../next/states/`, hero `../next/hero.jpg`, old pages `../research/`, old home `../ancien.html`,
  language switch to `../`), and the French home's switch points to `en/`. Run it.
- Tone: plain, professional English for asset managers; no marketing superlatives; same meaning as the French.

## K6-2 — main site: replace the notebook / Backtest Studio presentation by FORGE's front end
Repo `Arnaud-BARBIER.github.io` (cloned at `~/Quant reaserch/Arnaud-BARBIER.github.io`), file `index.html`.
New images already in `img/`: `forge-terminal.jpg` (the terminal), `forge-editor.jpg` (the code editor with the
allocator box), `forge-ide.jpg` (editor and terminal on the same screen). Commit them.
- Project 3 (`<section class="proj theme-glacier" id="p-backtest">`, article `#c2`): rewrite it around FORGE, keeping
  the page's existing structure and classes (proj-head, proj-hero, keynote, body, tags, figs, more):
  - kicker « Projet personnel 3 sur 3 », title « FORGE, laboratoire d'allocation de portefeuille », sub-title: one
    sentence (build and test multi-asset allocations, with an AI or in code, on a compiled simulation engine);
  - hero image `forge-ide.jpg`, caption « L'éditeur et le terminal sur le même écran : le code à gauche, ses résultats
    à droite. »;
  - keynote: what the front end does — terminal (compose a portfolio, move a weight and every view is recomputed,
    compare to benchmarks, 12 validation tests with a sealed out-of-sample period) and editor (the strategy as a tree
    of code boxes: indicators, allocator, parameters, charts; code from your AI or your own, simulated immediately);
  - replace the four « Backtest Studio en quatre étapes » videos by three figures in the same `.sv` style but with
    images (no video): `forge-terminal.jpg` « Le terminal », `forge-editor.jpg` « L'éditeur », `forge-ide.jpg »
    « Les deux ensemble », one sentence each;
  - figs: keep the three existing numbers (they are still true);
  - links: « Voir FORGE » → https://arnaud-barbier.github.io/Multi-strategy-backtest-engine/ , « Code sur GitHub » unchanged.
- The summary card of project 3 (around the `sum-card` linking to `#c2`): image `forge-ide.jpg`, title « FORGE,
  laboratoire d'allocation de portefeuille », remove the « Refonte en cours » flag; same for any other « refonte »
  mention about this project and the `#c2` link texts (« Voir Backtest Studio » → « Voir FORGE »).
- The top showcase carousel (`vit-win`, ~line 1013) item that plays `studio-1-creer-notebook.mp4`: show
  `forge-ide.jpg` instead, name/description updated to FORGE.
- The page has an FR→EN dictionary in its script: add English entries for every new French string, remove the
  entries of strings you deleted.
- Forbidden: touching the other projects, the contact block, styles of other sections; deleting the old videos from
  `video/` (just stop referencing them).

## Report
Files changed, anything left unchecked without a browser.
