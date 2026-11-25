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
