# Captures utilisées par docs/engine/index.html

Composées depuis `~/Desktop/SS4html/Moteur` et `~/Desktop/SS4html/Validation`. Les
captures d'un même bloc sont empilées verticalement.

**Aucune image n'est rééchantillonnée.** Quand un empilement mélange du 955 et du 965,
la plus étroite est complétée par du fond, jamais redimensionnée : un redimensionnement
de 1 % suffit à rendre du texte de code flou. La page les affiche à `width:auto`,
plafonné à 965 px, donc jamais agrandies non plus.

| Fichier | Contenu | Source |
|---|---|---|
| `01-alloc-spec.png` | Bloc `AllocationSpec` entier | `Alloc_spec1` + `Alloc_spec2` |
| `02-stateful-driver.png` | `StatefulSpec` avec son `ResumeWatchSpec` | `Statefulspec` |
| `03-event-bindings.png` | Les trois dictionnaires de binding | `Event_binding` |
| `04-watch-scanner.png` | `watch_two_actions_numba` | `watch_two_action` |
| `07-borrow-timeline.png` | Chronologie des tranches | `Borrow_timeline_Audit_comptale1` (haut) |
| `08-accounting-audit.png` | Bilan comptable, 4 panneaux, `4.44e-16` | `..._comptale1` (bas) + `..._comptale2` (haut) |
| `09-cost-layers.png` | Ancienne cascade, conservée comme archive mais retirée de la page moteur | `Cost_cascade` + `..._comptale2` (bas) |
| `10-validation-calls.png` | Baseline naïve, purchasing power, thresholds | `validation_baseline1/_2/3` |
| `11-performance-carrier.png` | Ancien résumé du run, conservé comme archive mais retiré de la page moteur | `Run function + alloc_audit` + `...2` |
| `12-engine-run.png` | `NJITEngine` avec `simple_oos(0.30)` | `engine_run_OOS` |
| `13-null-sizing.png` | Null de sizing, attribution, N_eff | `Null_Sizing1/2/_3` |
| `14-null-leverage.png` | Null de levier, frontière atteignable | `validation_null_levreage` (1337 px natifs) |
| `16-overnight-rate.png` | Courbe SOFR/DFF lue par le kernel | `courbe_des_taux` |
| `costs-cascade.png` | Cascade du run coûts : exécution, carry/roll, frais de gestion et financement séparés | `SS4html/costs/cascades.png` |
| `costs-allocation-cost-drag.png` | Débits cumulés et annuels du run coûts | `SS4html/costs/Allocation_cost_drag.png` |
| `costs-slippage-spec.png` | Politique de slippage dynamique | `SS4html/costs/execspec = slipage_spec.png` |
| `costs-execution-assets.png` | Contrats, spreads, commissions, taxe et carry par actif | `SS4html/costs/execspec = contract spec, carry, tax, .png` |
| `engine-cost-cascade.png` | Cascade du run moteur avec watcher | `SS4html/maj_moteur/nouvelle cascade.png` |
| `engine-cost-drag.png` | Débits cumulés et annuels du run moteur | `SS4html/maj_moteur/nouveau_cost_drag.png` |
| `engine-verdict-1.png` à `engine-verdict-4.png` | Run, résumé, verdict et trajectoire actualisés | `SS4html/maj_moteur/new_verdict_1.png`, `nv_2.png`, `nv3.png`, `nv_4.png` |

## Encore à capturer

| Fichier | Contenu | Où le prendre |
|---|---|---|
| `05-watch-events.png` | Journal du watcher, lignes `fired` + `cooldown` | `metrics_stateful["iterative_allocation_watch_events_df"]`, filtré |
| `06-borrow-ledger.png` | `borrow_ledger_df`, les 11 tranches | sortie du run avec borrow |

La page tourne sans elles : les deux `<figure>` correspondantes ont été retirées et le
texte a été réécrit pour ne plus en dépendre. Les rajouter demandera de réinsérer une
figure dans la section 02 et une dans la section 04.

## Non utilisé

`15-alloc-equity.png`, la courbe d'equity du run sans emprunt. Disponible si la note de
validation passe un jour en illustré.

## Qualité des captures

Les sources sont en 72 dpi, soit 1 pixel physique par pixel logique. Sur un écran
retina le navigateur doit donc les doubler, et c'est de là que vient le flou. Deux
façons de gagner de la netteté, par ordre d'efficacité :

1. **Pour les figures Plotly**, ne pas capturer du tout :
   `pip install kaleido` puis `fig.write_image("nom.png", width=1400, scale=2)`.
   Rendu vectoriel, 2800 px réels, net partout.
2. **Pour le code et les tableaux**, zoomer le notebook à 150 % (`Cmd +` deux ou trois
   fois) avant de capturer. Même contenu, une fois et demie plus de pixels.

Capture avec `Cmd+Shift+4` (pas `Cmd+Ctrl+Shift+4`, qui ne fait que copier dans le
presse-papiers sans écrire de fichier).
