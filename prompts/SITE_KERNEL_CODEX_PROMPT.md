# FORGE site — « Sous le capot », two pages (Codex session K)

File: `Multi-strategy-backtest-engine/docs/next/index.html` (one page: HTML, CSS and JS inline; branch `site-redesign`).
Touch only the blocks `#moteur` (page 1) and `#carnet` (page 2), their CSS and their two IIFEs at the end of the
script. Site text in French; brand written FORGE; no red anywhere (sells / losses = `--muted` or `#c3ccdb`).
You cannot open a browser here: the director checks the result in a real browser after the session.

## Style to match (the problem today: page 1 and page 2 look like generic diagrams, not like FORGE)
- Look at the existing `.pane` / `.pane-h` (terminal panel: square corners, 1px `#243044` border, header band
  `#10161f` with an uppercase mono label), `.bench`, `.log` and `.tlist`: that is the visual language.
- Fonts: `var(--mono)` for labels, values, code; `var(--sans)` only for sentences. Colours: `--accent` #29c5ff
  (allocator flows and active states), watcher flows `#d9bf5a`, strokes `#3a4558`, text `--text` / `#c3ccdb` / `--muted`.
- Square corners everywhere, no rounded boxes, no glow, no drop-shadow, no gradients except the existing fades.
  Connectors are orthogonal (horizontal / vertical segments with right-angle elbows), 1px, grey at rest, coloured
  only while a pulse travels on them.

## K1 — page 1: the engine and its calls (structure fixed by Arnaud, restyle it)
- Result: same structure as today, restyled:
  - left column, three small terminal-style panels: `ALLOCATEUR` (body: `max_sharpe(ctx, mem, out)` in mono,
    one muted line « appelé aux dates prévues »), below it `WATCHER · PORTEFEUILLE` and `WATCHER · OPÉRATIONS`
    (same format, one line each: « peut demander un rebalancement », « ajuste les ordres en cours »);
  - centre: `MOTEUR COMPILÉ` panel, body « barre 1 532 » counter (mono, tabular figures);
  - right of the engine: the vertical time strip, styled like a terminal time axis (thin ticks per bar, a longer tick
    every 5, month labels in mono at the scheduled rebalances drawn as small accent squares); time flows downwards;
    a thin accent line level with the engine marks the current bar (no « MAINTENANT » label);
  - flows: every bar a small yellow pulse engine → fork → both watchers; at a scheduled rebalance an accent pulse
    strip → engine → allocator, then allocator → engine (« nouvelles pondérations » tag in mono); about 1 bar in 14,
    one watcher reacts: top one sends a yellow pulse to the allocator and the allocator answers the engine in
    YELLOW (the allocator's output takes the colour of whoever called it), and a yellow square stays on the strip
    (« hors calendrier »); bottom one sends a yellow pulse back to the engine (« ordre ajusté »).
  - Keep the right third of the panel free (equity + stacked weights chart comes later).
- Constraints: SVG or HTML+SVG; runs only while visible (IntersectionObserver); `prefers-reduced-motion` = static
  final state; works at 375 px (stack vertically or scale; no horizontal page scroll).
- Forbidden: rounded corners, glow, decorative icons, new colours.

## K2 — page 2: M2 book → emission on the order book → execution (NO candles)
- Result, inside the `#carnet` panel (header band « MOTEUR COMPILÉ · REBALANCEMENT DU 01/03/2023 », right label
  « exemple chiffré »):
  - left: `CARNET M2` — the strategy's book as a list (asset, side, quantity, origin), the watcher line in yellow;
  - right: one ORDER BOOK per asset (XAUUSD, USA500IDXUSD), drawn like a real exchange order book: a price column,
    a size column on each side (sells above, buys below), the spread row in the middle (best bid / best ask, spread
    value). There are NO other orders: the only size shown is ours. No candles, no OHLC.
  - sequence, in slow motion with one caption per stage under the panel (keep the four existing captions, adapt
    wording to « carnet d'ordres »): (1) M2 lines appear; (2) the XAUUSD line leaves M2 and is EMITTED onto its book
    (it lands as a size bar at the ask side for a buy / bid side for a sell, quantity shown); (3) execution: the
    size bar drains to 0, the fill price appears (« exécuté 1 836,67 = vendeur 1 836,55 + glissement 0,12 »), then
    the cost lines: écart, glissement, commission, taxes, coût total; (4) same for USA500IDXUSD (sell at bid −
    slippage); (5) the BRENT watcher line is ticked (« 4 autres actifs sautés »); then (6) « Vitesse réelle »: the
    same loop very fast for ~3 s (bar counter racing, lines flashing in turn), ending on the final state.
  - Keep the existing example figures (quantities, prices, spreads, slippage, costs, taxes 0 / 0,77) exactly.
- Constraints: plays once when 40 % visible; reduced motion = final state; readable at 375 px (books stacked).
- Forbidden: candles, other market participants' orders, invented depth, red.

## Report
- What changed (selectors), and anything you could not do without a browser. Commit on `site-redesign`, one commit
  per item, message ending with the usual co-author line. Do not push.
