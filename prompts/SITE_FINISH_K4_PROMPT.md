# FORGE site — last text and end-of-page pass (Codex session K4)

File: `Multi-strategy-backtest-engine/docs/next/index.html` only (branch `site-redesign`). Site text in French, brand
FORGE, no red, keep the current style (terminal panels, mono labels, `--accent` #29c5ff). You cannot open a browser:
the director checks afterwards. One commit per item, do not push, do not merge.

## K4-1 — engine block (`#moteur`)
- Cap « Sous le capot » → « Moteur ».
- Paragraph `.sub`, exact text (bold as marked):
  « Le moteur compilé (**Numba**) avance **barre par barre**. Aux dates prévues, il appelle **votre fonction
  d'allocation**, qui renvoie les nouvelles pondérations. À chaque barre, il appelle des **watchers d'événements**,
  qui peuvent intervenir sur les positions en cours ou déclencher une nouvelle pondération en appelant la fonction
  d'allocation. »
- Costs column (`.eng-costs`), the trade terms in English, the rest stays French:
  - « Ordre au marché » : `Bid/ask spread`, `Slippage`, `Commission`, `Taxes`, `Market impact` (selon le volume moyen).
  - « Ordre limite » : exécuté à son prix limite seulement s'il est atteint ; Commission et Taxes ; « Sans spread ni
    slippage ».
  - « Sur la position » : unchanged.

## K4-2 — benchmark block (`#mesure`)
- Row titles: « Stratégie classique » → « Stratégie algorithmique classique »; « Gestion de position avancée » →
  « Stratégie algorithmique avec gestion de position avancée ». Allocation and grid titles unchanged.
- Add the user code each tool needs, as a second small line in each engine cell (same style as the memory line):
  classic 31 / 11 / 109 lignes (FORGE / vectorbt / NautilusTrader); advanced 64 / 78 / 146; allocation 49 / 56 / —;
  grid: no line count. Under the table, add to the note: « Lignes : code à écrire dans le notebook pour la stratégie,
  hors imports et chargement des données (l'optimiseur de poids commun aux trois, 175 lignes, n'est pas compté). »
- Grid row: keep the measured 25,5 s / 13,7 s (one process). Add below the table one line: « Depuis, sur la même
  machine avec 7 cœurs : 1 480 réglages en 14 s dans FORGE (mesure non comparée aux autres outils). »
- Nothing else in the numbers changes.

## K4-3 — end of page (`#fin`), modelled on the old site's footer, in the new style
Replace the current `#fin` content with, in this order:
1. Label « CONTACT » (mono, accent), the sentence « Une question, un retour, ou l'envie de voir FORGE tourner en
   direct ? Écrivez-moi. », then the e-mail as a large link `arnaud.barbier@grenoble-em.com →`
   (mailto, subject FORGE) with an accent underline, then « Une démonstration est possible sur place ou en visio. »
   and the button « Découvrir mon parcours ↗ » (https://arnaud-barbier.github.io/).
2. Label « CONTINUER LA LECTURE », and one muted line « Ces pages gardent pour l'instant l'ancienne présentation. »
3. A 4-column grid of cards (2 rows; 2 columns under 1100 px, 1 column at 375 px), each: category label (mono,
   uppercase), title, one sentence, « Lire → » link. The first card has an accent top border.
   - Suite conseillée · Le signal — Mode Research — « Développer un indicateur et mesurer ses signaux, sans
     compilation. » — `../research/`
   - La stratégie — Mode Algo — « Du signal aux trades gérés, avec frais et sizing. » — `../algo/`
   - Le portefeuille — Mode allocation — « Quarante lignes pour déclarer un portefeuille, et la sortie réelle du
     moteur. » — `../engine/`
   - La mesure — Validation · Max DDR — « Trois tests de validation appliqués à une optimisation de la durée de
     drawdown. » — `../validation/`
   - La mesure — Validation · Sharpe — « Dix contrôles, du hors échantillon jusqu'à 80 univers inédits. » —
     `../validation-2/`
   - Le rapprochement — Les coûts — « Chaque coût suivi de sa déclaration jusqu'à son débit dans la NAV. » —
     `../costs/`
   - La mécanique — Replay & resume — « Comment un kernel compilé décide en fonction de son propre passé. » —
     `../stateful/`
   - Le code — Le dépôt — « Source publique du moteur, sous licence MIT. » — « Voir sur GitHub → »
     https://github.com/Arnaud-BARBIER/Multi-strategy-backtest-engine
4. A full-width card « ← Retour sur mon site principal » / « arnaud-barbier.github.io · projets, parcours et contact »
   linking to https://arnaud-barbier.github.io/.
- Constraints: square corners, 1px `#243044` borders, card hover = accent border; the block keeps working with the
  block-by-block paging (it may be taller than the screen: the paging reads it in screenfuls); no horizontal
  overflow at 375 px.

## Report
Selectors changed and anything left unchecked without a browser.
