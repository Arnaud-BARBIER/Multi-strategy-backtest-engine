# FORGE site — engine schema, round 2 (Codex session K3)

File: `Multi-strategy-backtest-engine/docs/next/index.html`, block `#moteur` only (its HTML, the `.eng` CSS and the
« under the hood 1 » IIFE). Page 2 (`#carnet`) was removed: do not bring it back. Site text in French, brand FORGE,
no red. You cannot open a browser: the director checks in a real browser afterwards. Commit on `site-redesign`
(one commit), do not push.

Arnaud's verdict on K1: the structure is right, the look is « un peu moche ». Keep the same structure and style
family (terminal panels, mono labels, `--accent` for allocator flows, `#d9bf5a` for watcher flows), and fix this:

## K3 — items
1. **Active box = whole outline.** When a flow reaches a box, colour its entire border (header band included) in the
   flow's colour, not only the body's lower half. Same for the engine.
2. **Nicer boxes.** Compact panels (no large empty body): header band with a small coloured square on the left
   (accent = allocator, yellow = watchers, white = engine), then one mono line with the signature
   (`max_sharpe(ctx, mem, out)`, `watch_portfolio(ctx, book)`, `watch_orders(ctx, book)`; engine: `barre 1 586` and
   the date `01/03/2023`), then one muted line (the current descriptions). Equal widths, aligned.
3. **Clearer flows.** Connectors: one route per call, no parallel duplicate lines, no line crossing another or a box.
   At rest 1px `#3a4558`; while a pulse travels, the segment it is on lights up in the flow colour (2px), and fades
   after. Pulses visible (6-8 px squares or short dashes), not tiny dots.
4. **Time strip, more stylish and granular.** A real time axis: a tick per bar, day numbers every 5 bars in small mono,
   month names (« mars 2023 ») as section labels with a thin rule, year shown on January. Scheduled rebalances = accent
   squares with the date; watcher events leave yellow squares labelled « watcher » (no wrapped text). The current-bar
   line level with the engine stays, thin accent, no label.
5. **More events.** Scheduled rebalance about every 10 bars; a watcher event about one bar in 5 (both watchers, at
   random), so the schema is always busy.
6. **Slow then very fast.** Start at about 700 ms per bar for the first ~8 bars (so one can follow a rebalance and a
   watcher call), then accelerate smoothly over ~4 s to about 40 ms per bar and keep that speed (pulses shorten with
   the bar duration). Restart slow when the block comes back into view.
7. **Right column: transaction costs.** In the free right third of the panel, a title and a short description, then
   the list of costs applied to each order, in two groups:
   - « Ordre au marché » : écart achat/vente, glissement, commission, taxes, impact de marché (selon le volume moyen) ;
   - « Ordre limite » : exécuté à son prix limite seulement s'il est atteint ; commission et taxes, sans écart ni
     glissement ;
   - plus a line « Sur la position » : financement du levier et des positions courtes, conversion de devises.
   Styled like `.bench` / `.tlist` (mono labels, thin rules). Static (no animation).
- Constraints: runs only while visible; `prefers-reduced-motion` = static final state; at 375 px the right column goes
  under the schema and nothing overflows horizontally.
- Forbidden: rounded corners, glow, gradients, icons, new colours, touching anything outside `#moteur`.

## Report
Selectors changed, and what you could not check without a browser.
