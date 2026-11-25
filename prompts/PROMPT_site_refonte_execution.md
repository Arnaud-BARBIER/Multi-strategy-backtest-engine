# Task: add the "execution core rebuild" block to the homepage

FOLDER: `/Users/arnaudbarbier/Quant reaserch/Multi-strategy-backtest-engine`
FILE: `docs/index.html` only (GitHub Pages, branch `main`, folder `/docs`).
The page is bilingual: every change goes into the French block (`lang-fr`, around lines 442-760)
**and** the English block (`lang-en`, from line ~760). Same structure, same classes.

## Writing rules (non-negotiable, same as `prompts/PROMPT_site_corrections.md`)

- Never an em dash (—), in either language.
- No monospace for figures or words in running text.
- Professional written register; no aphorisms; lists introduced by « Elle permet de : » / "It lets you:".
- Locked style: Swiss grid, Archivo + IBM Plex Mono, white / black / red #C8102E. Reuse existing
  classes; add CSS only if no existing class fits, inside the existing `<style>` block.
- Copy the French texts below **verbatim**. The English texts below are the reference
  translations; copy them verbatim too.

## Change 1: the modes heading (line ~513 FR, ~782 EN)

`.modes-heading p` is limited to 43ch: keep it short.

- FR, replace `Un noyau partagé pour concevoir, tester et suivre une stratégie.` with:
  > Quatre modes sur un noyau Numba compilé, en cours d'unification sur une base d'exécution commune.
- EN, replace `One shared core for designing, testing and monitoring a strategy.` with:
  > Four modes on a compiled Numba core, being unified on one common execution base.

If the live text differs from the line quoted above (Arnaud may have edited it to « Une base
commune à tous les modes : ordres, exécutions, coûts et comptabilité sont gérés au même
endroit. Moins de mécanismes en double, et des résultats comparables d'un mode à l'autre. »),
replace whatever sentence sits in that `<p>`.

## Change 2: the scope sentence of "Repères techniques" (line ~582 FR, ~851 EN)

- FR, replace `Trois modes d'exécution partagent ce kernel` with `Trois modes d'exécution reposent sur ce noyau`
  (rest of the sentence unchanged).
- EN, replace `Three execution modes share that kernel` with `Three execution modes run on that core`
  (rest unchanged).

## Change 3: new block in "Repères techniques"

Inside `.tech-banner`, after `.tech-scope` and before `.tech-note`, in both languages. Use a
small label in the style of the existing `.lbl` (or the `.tech-scope` paragraph style) for the
title, the `.tech-scope` paragraph style for the text, a plain `<ul>` styled like the page's
other lists on a dark background (white text, readable spacing), and a closing sentence in the
`.tech-scope` style.

FR:

> **Refonte en cours : un moteur d'exécution commun**
>
> Chaque mode repose aujourd'hui sur sa propre version du noyau, adaptée à ses besoins. La refonte en cours les remplace par une base unique, inspirée de Nautilus Trader : un carnet d'ordres, un compte et une seule passe d'exécution par barre, pour tous les modes. L'allocation devient le mode portefeuille unique : elle reprend le suivi d'un compte réel et accueille des stratégies algorithmiques comme lignes, en spot, CFD ou futures.
>
> Elle permet de :
> - exécuter un ordre de la même façon dans tous les modes, avec le même prix, les mêmes coûts et le même journal ;
> - placer une stratégie dans un portefeuille sans la réécrire, et comparer ses résultats seule ou en portefeuille ;
> - faire lire à une stratégie l'état des autres lignes du portefeuille, par exemple pour une couverture ;
> - relier chaque position à l'ordre qui l'a ouverte et à la stratégie qui la porte ;
> - ne compiler que ce qu'une simulation utilise, pour garder la vitesse sur des centaines de milliers d'ordres.
>
> Les démonstrations de cette page tournent sur les noyaux actuels.

EN:

> **Rebuild in progress: one common execution engine**
>
> Each mode currently runs on its own version of the core, tailored to its needs. The rebuild in progress replaces them with a single base, modelled on Nautilus Trader: one order book, one account and a single execution pass per bar, for every mode. Allocation becomes the single portfolio mode: it takes over real-account tracking and accepts algorithmic strategies as lines, in spot, CFDs or futures.
>
> It lets you:
> - execute an order the same way in every mode, with the same price, the same costs and the same journal;
> - put a strategy into a portfolio without rewriting it, and compare its results alone and in the portfolio;
> - let a strategy read the state of the portfolio's other lines, for example to hedge it;
> - trace every position back to the order that opened it and the strategy that holds it;
> - compile only what a simulation uses, to keep the speed on hundreds of thousands of orders.
>
> The demonstrations on this page run on the current cores.

## Do not change

Everything else on the page: mode cards, diagram, figures (`.figs`), entries, roadmap, other
pages. If, while reading the whole homepage, you find another sentence that now contradicts the
new block (for example one stating that all modes already share one execution layer), **list it
in your report with its line, do not edit it**.

## Checks before committing

1. `grep -n "—" docs/index.html` returns no new occurrence.
2. Open the page locally (`python3 -m http.server` in `docs/`), check FR and EN with the language
   switch, desktop width and 390 px phone width: no horizontal scroll, the new list is readable on
   the dark banner, the reveal animation still applies (`.tech-banner` is already in the list).
3. One commit: `Homepage: execution core rebuild block`. Do not push. Report the diff summary,
   the two screenshots' observations, and the list of contradicting sentences if any.

Note: the claude.ai artifact draft of the page is updated separately by the director; ignore it.
