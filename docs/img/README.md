# Captures utilisées par docs/engine/index.html

Composées depuis `~/Desktop/Moteur` et `~/Desktop/Validation`, normalisées à 955 px
de large. Les captures d'un même bloc ont été empilées verticalement.

| Fichier | Contenu | Source |
|---|---|---|
| `01-alloc-spec.png` | Bloc `AllocationSpec` entier | `Alloc_spec1` + `Alloc_spec2` |
| `02-stateful-driver.png` | `StatefulSpec` avec son `ResumeWatchSpec` | `Statefulspec` |
| `03-event-bindings.png` | Les trois dictionnaires de binding | `Event_binding` |
| `04-watch-scanner.png` | `watch_two_actions_numba` | `watch_two_action` |
| `07-borrow-timeline.png` | Chronologie des tranches | `Borrow_timeline_Audit_comptale1` (haut) |
| `08-accounting-audit.png` | Bilan comptable, 4 panneaux, `4.44e-16` | `..._comptale1` (bas) + `..._comptale2` (haut) |
| `09-cost-layers.png` | Coûts avec et sans + cost drag cumulé | `Cost_cascade` + `..._comptale2` (bas) |
| `10-validation-calls.png` | Baseline naïve, purchasing power, thresholds | `validation_baseline1/_2/3` |
| `11-performance-carrier.png` | Run + résumé alloc, verdict `BELOW_RISK_FREE` | `Run function + alloc_audit` + `...2` |
| `12-engine-run.png` | `NJITEngine` avec `simple_oos(0.30)` | `engine_run_OOS` |
| `13-null-sizing.png` | Null de sizing, attribution, N_eff | `Null_Sizing1/2/_3` |
| `14-null-leverage.png` | Null de levier, frontière atteignable | `validation_null_levreage` |

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

Capture avec `Cmd+Shift+4` (pas `Cmd+Ctrl+Shift+4`, qui ne fait que copier dans le
presse-papiers sans écrire de fichier).
