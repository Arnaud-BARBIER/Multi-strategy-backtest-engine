# Tâche — réécrire le CV en affirmations vérifiables, pas en conclusions

## Le registre demandé, et pourquoi

Un recruteur qui lit « j'ai construit une batterie de validation out-of-sample » se dit
« ah, d'accord » et passe à la ligne suivante. Une conclusion ne provoque rien : elle est
déjà digérée, il n'y a plus rien à faire avec.

Ce qui accroche, c'est une affirmation **assez spécifique et assez inattendue pour donner
envie de vérifier**. Presque grotesque, jamais grotesque. Le lecteur doit se dire « attends,
c'est vrai, ça ? » et avoir envie de poser la question en entretien.

La différence tient en un mot : une conclusion **résume un travail**, une affirmation
**expose un fait**.

```
CONCLUSION (à bannir)      J'ai développé une batterie de tests de robustesse.
                           Mon outil détecte le surapprentissage.
                           Expertise en validation out-of-sample.

AFFIRMATION (recherchée)   Redécouper la grille de rebalance ne crée aucune
                           significativité : le t est strictement invariant. C'est
                           démontré dans le code, et le moteur est construit pour que
                           ce ne soit pas jouable.

                           Un watcher censé protéger le portefeuille lui coûte 8 points
                           de drawdown supplémentaires dans le krach qu'il devait amortir.
                           Mesuré par rejeu apparié, pas supposé.

                           Localiser le seuil de rupture d'une stratégie demande douze
                           backtests et aucune probabilité.
```

## 🔴 La contrainte qui prime sur tout

**Chaque affirmation doit être vraie et traçable à un chiffre vérifié.** « Presque
grotesque » ne veut pas dire exagéré : ça veut dire **surprenant et exact**. Une affirmation
invérifiable détruit tout le positionnement, dont l'argument central est précisément la
rigueur. En entretien, chaque ligne doit pouvoir être ouverte et montrée.

Avant d'écrire une ligne, demander : *où est le chiffre, dans quel run, quel notebook ?*
Si la réponse n'existe pas, la ligne ne s'écrit pas.

---

## Matériau vérifié disponible

### Reverse stress déterministe (run in-sample, 2 300 barres, 25 décisions)

- Seuil « equity 50 % sous son plus-haut » franchi à partir d'une baisse de **14,25 %**
  étalée sur 40 barres, machinerie complète active. Publié comme **frontière de
  franchissement**, pas comme minimum prouvé : la réponse est non monotone, quatre
  renversements.
- **Une grille de balayage plus fine a déplacé la réponse de 23,3 % à 14,25 %.** Neuf points
  d'écart. Le diagnostic de non-monotonicité a payé son coût dès son premier usage.
- **Latence de réaction : capacité d'une barre, effective de 105 à 209 barres.** Les
  déclencheurs du watcher demandent 20 % de drawdown ou 100 barres sous l'eau. Il ne peut
  structurellement pas réagir pendant un krach de 40 barres.
- Le délai de réaction est une fonction **monotone** du choc (209 barres à 2 %, 105 barres
  à 16 %) alors que le résultat, lui, ne l'est pas.
- Neutraliser une seule action du watcher, l'overlay de borrow, **améliore le drawdown de
  8 points**, à nombre de déclenchements identique.
- Une autre action n'a **pas pu être mesurée** : la neutraliser envoie le watcher dans une
  boucle de tir (1 028 déclenchements contre 22). Le rapport la marque contaminée et
  l'exclut de la somme au lieu de publier son chiffre.
- Chaque rapport est précédé d'un **run identité à choc nul** qui doit reproduire l'equity
  de référence à 1e-9, sinon rien n'est publié.

### Out-of-sample scellé

- Edge in-sample **-0,61 %/an**, out-of-sample **+3,26 %/an**. Verdict rendu par l'outil :
  `POSITIVE_OOS_WITHOUT_IS_EDGE`, « emergence rather than retention: the IS denominator was
  not positive ».
- L'outil refuse de publier un ratio de rétention quand le dénominateur in-sample est nul
  ou négatif.

### Positions méthodologiques démontrées dans le code

- Redécouper la grille de décision ne crée aucune puissance : passer de n à 2n blocs divise
  la moyenne par 2 et l'écart-type par racine de 2, le t est invariant.
- Le pouvoir d'acquisition d'un investisseur **est** le facteur d'escompte stochastique au
  cas gamma = 1.
- La frontière atteignable en dé-levant un benchmark n'est **pas** une droite : elle bombe
  au-dessus, parce qu'un mélange dé-levé subit moins de volatility drag. La juger sur la
  droite flatterait la stratégie.
- Compter les **épisodes** de drawdown, pas les barres sous l'eau : 929 « creux » sur
  1 500 barres est un non-sens.

### Run A (notebook `PIA_Validation.ipynb`, univers de 5 actifs, **46 décisions**)

- Baseline naïve : `ADDS_NOTHING`, -1,86 point de CAGR.
- Null de levier : `DOMINATED`, -0,74 à -1,42 point.
- Décomposition : pondération -0,84 point (t = -0,63) ; panier et trésorerie -1,02 point
  (t = -1,37, environ 99 décisions seraient nécessaires pour trancher).
- Le résultat reproduit DeMiguel, Garlappi et Uppal (2009).

---

## ⚠️ Ce qui n'est PAS vérifié et ne doit pas être écrit

- **Le null de levier sur l'out-of-sample n'a pas été lancé.** La volatilité
  out-of-sample est supérieure de 3,11 points à celle du benchmark, donc le +3,26 % pourrait
  n'être que la rémunération mécanique de ce risque. Tant que le test n'est pas passé,
  **aucune ligne du CV ne doit s'appuyer sur ce +3,26 %**.
- **Le Monte-Carlo block bootstrap n'est pas écrit.** Rien sur les intervalles, la VaR ou la
  CVaR.
- **Ne jamais mélanger le run A (46 décisions) et le run courant (25 décisions).** Ce sont
  deux runs différents, sur des périodes et des univers qui ne coïncident pas. Si une ligne
  cite un chiffre, elle doit citer le run dont il vient, ou n'en citer aucun.

---

## Méthode de construction

Pour chaque ligne candidate, dérouler :

1. **Le fait brut**, avec son chiffre.
2. **Ce qu'il contredit** dans l'intuition courante. S'il ne contredit rien, la ligne
   n'accroche pas et se coupe.
3. **La reformulation la plus courte** qui conserve les deux.

Exemple :

```
fait          neutraliser l'overlay de borrow du watcher améliore le drawdown de 8 points
contredit     un mécanisme de protection est censé réduire le risque, pas l'augmenter
ligne         Un mécanisme de protection qui aggrave de 8 points le drawdown qu'il devait
              amortir. Mesuré par rejeu apparié.
```

## Règles de forme

- **Pas de tiret cadratin.** Cohérent avec le site.
- Pas de vocabulaire de compétence (« expertise en », « maîtrise de », « capable de »). Le
  fait remplace l'adjectif.
- Pas de superlatif, pas d'auto-évaluation, aucune mention d'un outil d'IA.
- Une ligne = un fait. Pas d'énumération de technologies.
- Le jargon n'est admis que s'il est **immédiatement** désambiguïsé par le chiffre qui
  l'accompagne : beaucoup de recruteurs en gestion d'actifs ne sont pas quantitatifs.
- Longueur : le CV reste un CV. Viser **cinq à sept affirmations** au total, pas quinze.
  Une liste trop longue redevient un résumé.

## Ce qu'il faut rendre

1. Dix à douze lignes candidates, chacune accompagnée de **sa source vérifiable** (run,
   notebook, fichier).
2. Un classement par pouvoir d'accroche, avec en une phrase la raison du classement.
3. La sélection finale de cinq à sept, prête à coller.
4. La liste des lignes **écartées parce qu'invérifiables**, avec ce qu'il faudrait lancer
   pour pouvoir les écrire un jour. Cette liste est utile : elle devient la file d'attente
   des prochains tests.
