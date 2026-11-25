# FORGE site — sealed period faster, new terminal beside the editor (Codex session K5)

Repo: `Multi-strategy-backtest-engine` (branch `main`, the live site). Page source: `docs/next/index.html`; the home
`docs/index.html` is generated from it with `python3 tools/site/make_home.py` (run it after every change).
Site text in French, brand FORGE. One commit per item, do not push (the director checks, then pushes).

**K5-2 needs a headless browser on 127.0.0.1** (Pod Lab test server + Brave via `tools/site/refreeze/drive.py`). If
your sandbox refuses localhost or `ps`, do K5-1 only and say so: the director will run K5-2.

## K5-1 — sealed period, much faster (`startVal()` in `docs/next/index.html`)
- Keep the current principle exactly: the window and the text above it NEVER grow or move; the camera zooms INSIDE
  the capture only (`cam()` on `#vwcam`), on the locked card and its « Ouvrir la période scellée » button; the cursor
  clicks; the capture switches to the opened results; the camera zooms back out to the whole capture = the normal
  view (window + title + result line visible).
- Make it quick: zoom in ≈ 0.4 s after the window appears, cursor move ≈ 0.5 s, click, open, zoom out ≈ 0.4 s after
  the click; « continuer » prompt (`nextOn('val')`) right after the zoom-out. `.vw-cam` transition ≈ 0.45 s.
  Update `stepTime()` for `s.val` to match (about 4.5 s).
- Do not call `popVal()` anywhere (remove it if unused). Reduced motion: opened capture shown at once.

## K5-2 — the terminal beside the editor (frozen state `term23`)
Arnaud arranged the report in his running Pod Lab: at the bottom, « Allocation » section with **Impact des frais
d'allocation**, **Positions investies par actif — devise principale**, **Registre des flux de trésorerie (D)**
(no second stacked weights chart). The joined screen (editor 1/3 + terminal 2/3) must show this arrangement.
1. Find the report arrangement of his running server (its `SHOWCASE_STATE`, the strategy « Mon portefeuille »
   `report.toml`, or the global one he saved). Do not modify his state.
2. Test copy: start from the existing test state (`SHOWCASE_STATE` of the server on port 8767, read it with
   `ps eww`), copy his `report.toml` into it. Stop that test server and restart it from `Github/` with
   `../.venv/bin/python -m Backtest_Git.showcase_app.server` (`SHOWCASE_STATE=<copy>`, `SHOWCASE_PORT=8767`), so it runs
   the current code.
3. Refreeze `term23` with `tools/site/refreeze/term23.json` (replace the old scratchpad paths by a temp dir), viewport
   1333 x 1250, then place it with `tools/site/refreeze/finish.py <tmp> term23` (it brands FORGE and hides the stale
   banner). The terminal must be in the same state as before (strategy « Mon portefeuille », same run as the current
   `term23`: 17,6 % / 16,1 % / Sharpe 1,09), French, accent `#29c5ff`, only the report arrangement changes.
4. Check by screenshot of `docs/next/states/term23.html` at 1333 x 1250 that the three new charts are drawn (not empty
   placeholders) and that « Pod Lab » appears nowhere. Run `make_home.py`.
- Forbidden: touching any other frozen state, Arnaud's own state, Pod Lab code, or the kernel.

## Report
What changed, the screenshot path for K5-2, and anything you could not do.
