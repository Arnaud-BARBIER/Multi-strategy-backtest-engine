# Site FORGE — term23 with indicators on the candle chart, then the link preview card (Codex)

Context: `prompts/SITE_FORGE_DIRECTOR.md` (Next, item 2). The frozen terminal `docs/next/states/term23.html`
(terminal beside the editor, demo steps « ide » and the link preview card) shows the USA500IDXUSD execution chart
WITHOUT indicators. `docs/next/states/term_custom.html` shows what Arnaud wants on it: envelope bands, moving
averages and the `individual_sharpe` pane under the candles.

## Item 1 — refreeze term23 with indicators
- Result: `term23.html` = same layout as today (Capital, Execution USA500IDXUSD large top right, Allocation row,
  journal), but the execution chart carries the same indicators as in `term_custom.html` (bands, averages,
  `individual_sharpe` pane), readable at the size it has in the demo. Bottom « Allocation » row = Impact des frais
  d'allocation, Positions investies par actif, Registre des flux de trésorerie (Arnaud's arrangement).
- How: rebuild the demo test state « Mon portefeuille » (lost at reboot), start the Pod Lab test server on current
  code (port 8767), replay `tools/site/refreeze/term23.json` (fix old scratchpad paths), enable the indicators on the
  execution panel the way `term_custom` was produced (Max Sharpe strategy, `# bt:chart` boxes), freeze.
- Check: open `http://localhost:8780/next/` (serve `docs/`), go through the « ide » steps at 1288 x 723: no layout
  shift versus before, candles ≥ 40 % of the cell.

## Item 2 — regenerate the link preview card
- Serve `docs/` on 8780. Copy `tools/site/og/compose.html` into `docs/next/states/_og.html`, capture it at
  1600 x 1000, DPR 1.8 (`tools/site/refreeze/drive.py`, SNAP_W/SNAP_H/SNAP_DPR) → JPEG `tools/site/og/ide.jpg`;
  delete `_og.html`. Capture `tools/site/og/card.html` (file://) at 1200 x 627, DPR 2 → resize to 1200 px JPEG q88
  → `docs/og5.jpg`; remove `docs/og4.jpg`; replace `/og4.jpg"` by `/og5.jpg"` in `docs/next/index.html` and
  `docs/next/en/index.html`; run `python3 tools/site/make_home.py`.
- Check: the editor tree (Mon portefeuille → Allocateur → …) and the indicators on the candles are visible in the card.

## Constraints / forbidden
- Do not touch the `<head>` lines for Umami, og:* and the absence of `noindex` (check with
  `grep -c "umami\|og:image" docs/index.html docs/en/index.html` = 1 each, `grep -c noindex` = 0).
- No engine/kernel edit, no compilation. Commit; do NOT push (Arnaud says « push »).
- End with « For the director » (≤ 10 lines) and add one line to `SITE_FORGE_DIRECTOR.md` Next.

## Item 1 execution — 25 November 2026
- Reused the verified main-demo restoration in a new isolated `forge_demo_recovery/2026-11-25/term23_indicators_work/state`; original recovered/user states untouched. Current Pod Lab code on 8767, `NUMBA_DISABLE_JIT=1`; no engine edit or compilation.
- Updated `term23.json` to run the captured main-demo source/settings, explicitly select price + individual-Sharpe panes and write to `/tmp/forge-term23/{flow,chk}` (create these directories before replay).
- Refroze only `term23.html`, branded FORGE. All six report cells drawn; bottom row remains cost impact / invested positions / cash ledger.
- Exact metrics: annual return .17603038755922906, volatility .1611225113321582, Sharpe 1.0874035216788729, drawdown -.266643949899779.
- Browser QA: 1333×1250 capture, joined demo forward/back at 1288×723; all six cell rectangles unchanged, no horizontal overflow. Candles 231/476px = 48.5% of execution cell.
- Evidence: `/tmp/forge-term23/{capture.log,qa.log,geometry_before.json,geometry_after.json,chk/ide.png}`; levers: not concerned.

## Item 2 execution — 25 November 2026
- Captured the unchanged `compose.html` at 1600×1000 DPR1.8 → `tools/site/og/ide.jpg` (2880×1800); removed temporary `_og.html`.
- Captured `card.html` via file:// at 1200×627 DPR2, fonts loaded; resized with Lanczos to 1200×627 JPEG quality88 → `docs/og5.jpg`; removed `og4.jpg`.
- Verified visually: editor tree (Mon portefeuille → Allocateur and branches), moving averages, bands and individual-Sharpe pane visible in the final card.
- Updated FR/EN preview sources and regenerated both homes. All four HTML files differ only by the requested og4→og5 path; Umami/other OG metadata preserved, no noindex.
- Checks: exact JPEG dimensions/format, temporary-file cleanup, geometry and branding assertions, `git diff --check` all pass. No push.
- Persistent QA evidence: `forge_demo_recovery/2026-11-25/term23_indicators_work/`; levers: not concerned.

For the director
- Items1–2 complete: real term23 indicators, unchanged report cells, candle budget48.5%, updated og5 card.
- Current code, isolated restored state, JIT off; no engine edit or compilation.
- Desktop1288×723 forward/back QA and FR/EN metadata checks pass.
- Cross-asset recording and other director tasks were outside this session. No push.
