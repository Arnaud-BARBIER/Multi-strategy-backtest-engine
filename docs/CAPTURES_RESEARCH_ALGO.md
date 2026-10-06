# Démonstrations pour les pages Research et Algo

Les pages `research/index.html` et `algo/index.html` sont complètes sans captures : leurs schémas décrivent les parcours, et les extraits sont explicitement illustratifs. Les images ci-dessous doivent provenir de vrais runs. Aucun résultat chiffré ne doit être inventé pour remplir la page.

## Première sélection : cinq captures et un tableau intégré

| Fichier proposé dans `docs/img/` | Écran à capturer | Message de l’image | Emplacement |
| --- | --- | --- | --- |
| `research-workspace.png` | Graphique local : bougies, indicateur, signaux, un panneau secondaire et navigation START / END | Le résultat d’une étude se lit dans un espace graphique relié au notebook | Research §05 |
| Tableau HTML intégré — réalisé | Export fourni de `research.report().predictive` : 9 lignes, synthèse et 25 colonnes dépliables | Le signal se mesure contre un point de comparaison explicite ; aucune capture nécessaire | Research §02 |
| `research-comparison.png` | Comparaison avant / après modification du même indicateur : « vs previous » et « vs baseline », avec signaux conservés / ajoutés / retirés | Chaque itération retrouve à la fois sa précédente version et sa référence | Research §03 |
| `research-learning.png` | Campagne : phase, décision à l’aveugle et panneau de réservation / Check sizes | L’entraînement distingue ce qui a été vu de ce qui reste réservé | Research §06 |
| `algo-trade-chart.png` | Un run Algo dans le serveur Research : entrée, sortie et indicateur, zoomés sur un trade | Les trades du moteur rejoignent le même graphique | Algo §07 et Research §08 |
| `algo-trade-audit.png` | `metrics["trades_df"]` filtré sur un `trade_uid` ayant des sorties partielles, avec raisons, prix et coûts | Le résultat se décompose en événements de gestion inspectables | Algo §05 |

La troisième démonstration montre la différence entre deux versions de l’indicateur, et non une autre vue du tableau prédictif. Exemple : A = version de référence, B = premier changement, C = deuxième changement. Le rapport de C montre C contre B (« previous ») puis C contre A (« baseline »). Les cohortes disent quels signaux sont conservés, ajoutés ou retirés.

Pour afficher les comparaisons déjà disponibles sans relancer une étude :

```python
from IPython.display import display

report = research.report()
comparisons = report.shown_comparisons()
if not comparisons:
    print("Aucune comparaison disponible dans cette série pour ce run.")
for name, comparison in comparisons.items():
    print(f"── vs {name} ──")
    display(comparison.cohorts.to_frame())
    display(comparison.predictive)
```

Si nécessaire, réaliser trois variantes dans la même série DEV : A devient la référence, B la précédente, C la courante. Au deuxième run seulement, précédente et référence sont identiques et le rapport n’est affiché qu’une fois. Conserver le même dataset, la même période et la même évaluation.

## Compléments Research

| Fichier proposé | Préparation / source | Ce qui doit rester visible |
| --- | --- | --- |
| `research-study.png` | Cellule d’une étude réelle ou fichier `.py` avec calcul, signaux et évaluation | Un extrait assez court pour lire le contrat ; éviter le notebook entier |
| `research-timing.png` | Activer `MovementJudge`, comparer deux variantes, sélectionner un mouvement apparié | Même mouvement, entrée avant / après ; mention « diagnostic a posteriori » |
| `research-history.png` | Journal `<notebook>_history.md` ou `research.history.variants()` | Référence, identifiants des variantes et persistance de l’historique |
| `research-zones.png` | Sélectionner une zone en DEV, Send to notebook, afficher la requête correspondante | Même intervalle dans le graphique et dans le notebook ; sélection discrétionnaire identifiée |
| `research-replay.png` | Replay simple en pause, juste avant un signal | Frontière du futur masqué, choix Signal / Candle et commandes de lecture |
| `research-learning-feedback.png` | Phase 2 ou 3 : décision enregistrée, puis feedback | Niveaux et résultat dévoilés après la décision ; préciser la phase |
| `research-learning-report.png` | `research.evaluate_learning()` sur une campagne réellement terminée | Effectifs par phase, sélection aléatoire comparable et intervalles ; ne pas mélanger annotations rétrospectives et résultats aveugles |
| `research-freeze-oos.png` | Artefact de freeze et sorties OOS / cross-asset existantes | Configuration figée, périmètre et identité du run |

## Compléments Algo

| Fichier proposé | Préparation / source | Ce qui doit rester visible |
| --- | --- | --- |
| `algo-htf-alignment.png` | Un extrait de données et du graphique autour de 11:00 | H1 horodatée 10:00, disponibilité à 11:00, valeurs M15 avant/après ; préciser la convention de timestamp |
| `algo-setup-regime.png` | Inspection d’un run existant avec plusieurs setups et régime | Scores concurrents, setup retenu et profil de sortie sélectionné |
| `algo-profile-phases.png` | Déclaration réelle d’un profil et règles, accompagnée de lignes du trade correspondant | Déclencheur, action et changement de phase ; vérifier le timing avant de présenter le comportement comme validé |
| `algo-costs.png` | Coûts à la racine d’`ExecutionCostSpec` et extrait du même run | Prix prévu / exécuté, spread, commission et slippage ; paramètres de démonstration identifiés |
| `algo-sizing.png` | Deux résultats de sizing issus de la même tape, avec specs | Entrées/sorties identiques, exposition différente ; le sizing ne recalcule pas la logique de gestion |
| `algo-random-management.png` | Rapport existant `random_entry_keep_exit_logic` | Nombre de simulations, gestion conservée, distribution de référence et stratégie ; ne pas utiliser une référence à durée fixe pour illustrer ce test |
| `algo-compile-mode.png` | Ligne de sélection du noyau et temps d’un premier run puis d’un run chaud | Même environnement et même interpréteur ; dates et configuration, sans généraliser le temps mesuré |

## Deux courtes vidéos possibles

1. **La boucle Research, 20–30 secondes.** Modifier un paramètre en DEV, voir le rapport de comparaison, retrouver le mouvement dans le graphique. Enchaîner sur une zone envoyée au notebook.
2. **Le passage vers Algo, 20–30 secondes.** Montrer le signal, la déclaration du profil, puis le run existant dans le graphique Research. Ouvrir les lignes du trade correspondant. Ne pas faire croire que le simple partage de la vue convertit automatiquement l’étude en stratégie.

## Prise de vue et intégration

- Choisir un run DEV déjà disponible. L’ouverture du graphique ne recalcule pas le moteur. Ne pas lancer une compilation ou une campagne statistique uniquement pour une image sans nécessité.
- Réutiliser les artefacts OOS existants : ne pas consommer de nouvelles observations réservées pour illustrer le site.
- Capturer en PNG à résolution native, idéalement en Retina ; contrôler la lisibilité à environ 960 px de large et sur mobile. Pour les tableaux, réduire les colonnes plutôt que la police.
- Garder l’actif, le timeframe, la période et la version du candidat dans la légende. Éviter les chemins personnels, les jetons et les données de compte.
- Préférer une idée par image. Ne pas agrandir un PNG au-delà de sa résolution native. Ne pas fabriquer d’écran de résultat avec un générateur d’images.
- Une fois disponible, ajouter chaque capture avec `<figure>`, un `alt` descriptif et une légende bilingue. Aucun emplacement vide ni fichier d’image absent n’est référencé dans les pages actuelles.
- Pour le code, conserver les blocs HTML sélectionnables déjà présents ; une capture n’est utile que si elle montre aussi une sortie réelle.

## Sources éditoriales

- `Github/Backtest_Git/docs/RESEARCH_MODE_GUIDE.md`, daté du 6 octobre 2026.
- `Github/Backtest_Git/docs/ALGO_MODE_GUIDE.md`, daté du 6 octobre 2026.

Les instructions destinées aux agents dans ces guides restent hors des pages de présentation. Les limites fonctionnelles utiles au lecteur sont indiquées dans les pages.

Le tableau prédictif fourni est intégré sans supposer son actif, sa période ou son unité. Ces métadonnées ne figurent pas dans le collage HTML. Les 9 × 25 valeurs sont conservées dans le tableau complet.
