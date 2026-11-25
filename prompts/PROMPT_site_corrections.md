# Tâche — trois corrections sur le site vitrine

DOSSIER : `/Users/arnaudbarbier/Quant reaserch/Multi-strategy-backtest-engine`
Site publié : https://arnaud-barbier.github.io/Multi-strategy-backtest-engine/
(GitHub Pages, branche main, dossier `/docs`)

## 🔴 DEUX COPIES À MAINTENIR

Chaque page existe en deux exemplaires : le fichier dans `docs/` et un brouillon artifact
sur claude.ai. **Toute modification doit aller dans les deux.** Les URL des brouillons sont
dans la mémoire projet `showcase-recruiter-vitrine`. Ne jamais envoyer une URL claude.ai à
un recruteur : le domaine annonce l'IA avant le contenu.

## Règles de rédaction imposées, non négociables

- **JAMAIS de tiret cadratin.** Seule exception : les blocs de sortie console bruts, qui
  sont des pièces à conviction et se recopient tels quels.
- **Pas de monospace pour les chiffres** dans le texte courant. Une factsheet de fonds n'en
  utilise pas. Le monospace est réservé aux blocs de code et de console.
- **Registre écrit professionnel**, pas d'oral. « Ce n'est pas une idée de moi » a été
  rejeté explicitement.
- **Exemple chiffré AVANT l'abstraction**, systématiquement.
- Style verrouillé : grille suisse, Archivo + IBM Plex Mono, blanc / noir / rouge #C8102E.
  Les blocs de sortie console brute sont le matériau principal, pas une décoration.
- Tout nouveau schéma : **SVG inline** avec `currentColor` et les tokens du thème. Les PNG
  drawio sont bloqués par la CSP, c'est un problème déjà rencontré et déjà résolu ainsi.

---

## Correction 1 — le chapô de la page stateful

FICHIER : `docs/stateful/index.html`, ligne 172 environ, la balise `<p class="lede">`.

**Texte actuel :**

> Un kernel Numba ne peut pas s'arrêter à chaque barre pour demander l'avis d'une fonction
> Python. Et cette fonction doit rester du Python ordinaire, puisque c'est là que
> l'utilisateur écrit sa stratégie. Trois mécanismes tiennent les deux à la fois : le replay,
> le resume, et le watcher qui relie les deux.

**Ce qui ne va pas.** Il présente une **limitation technique** comme point de départ. Un
lecteur y comprend « Numba m'empêche de faire ce que je veux ». C'est l'inverse de
l'intention.

**L'intention à restituer.** L'objectif du moteur est d'accueillir et de tester **toutes**
les stratégies, y compris celles dont la décision dépend d'un état accumulé. L'utilisateur
doit pouvoir tester n'importe laquelle de ses idées, écrite en Python ordinaire, sans avoir
à la traduire. Le problème n'est pas que Numba refuse : c'est qu'un aller-retour vers une
fonction Python **utilisateur** à chaque barre est un mécanisme trop lourd, en particulier
dans un moteur hybride Python/Numba où la frontière entre les deux est franchie des
milliers de fois. Les trois mécanismes existent pour tenir l'objectif sans payer ce coût.

**Contraintes de réécriture :**

- Dire « une fonction Python **utilisateur** » ou « la fonction où l'utilisateur écrit sa
  stratégie ». Le mot **utilisateur** doit apparaître : c'est le sujet de la phrase.
- Le premier temps est l'**ambition** (accueillir toutes les stratégies), le second est le
  **coût** (un appel par barre est trop lourd en hybride), le troisième la **réponse**
  (replay, resume, watcher).
- Ne pas allonger : quatre phrases maximum, comme aujourd'hui.
- Ne pas toucher au `<h1>` ni au bloc `.tickers` qui suit.
- La page est en français seul, sans bascule EN, contrairement à `index.html` et `engine/`.
  Ne pas introduire de bascule ici.

---

## Correction 2 — l'ouverture de la page d'accueil

FICHIER : `docs/index.html`, le chapô d'ouverture (page **bilingue FR/EN**, défaut FR :
les deux versions doivent être modifiées, et elles doivent dire la même chose).

**Ce qu'il faut ajouter.** Que l'outil sait évaluer une stratégie **qui a déjà fait ses
preuves dans le passé**, pour déterminer si cette performance a des chances de tenir à
l'avenir. C'est la proposition de valeur pour un family office ou un petit gérant
indépendant : il arrive avec une stratégie qui gagne, et la question qu'il se pose n'est pas
« est-ce que ça marche », c'est « est-ce que ça continuera ».

**⚠️ Ce n'est PAS une contradiction avec le positionnement verrouillé.** La règle « ne
jamais mettre en avant une stratégie qui gagne » porte sur **ses propres** résultats : pas
de courbe d'equity triomphante de l'auteur. Évaluer la stratégie **d'un tiers**, déjà
rentable, est un cas d'usage, pas une vitrine. Le ton reste celui d'un outil qui cherche à
invalider, jamais à célébrer.

**Contraintes :**

- Rester factuel sur ce qui existe **aujourd'hui** : rétention d'edge in-sample vers
  out-of-sample, comparaison à une référence 1/N payant les mêmes frais, null de levier,
  reverse stress déterministe. Ne rien promettre qui ne soit pas codé.
- Ne pas employer « prédire ». L'outil ne prédit pas : il mesure ce qui est reproductible
  et localise les conditions où la stratégie cesse de tenir.
- Deux ou trois phrases. L'accueil doit rester lisible en 90 secondes par un recruteur
  souvent non-Python.

---

## Correction 3 — la suite du programme, à annoncer comme telle

À placer en fin de page d'accueil, dans une section clairement identifiée comme
**travaux à venir**, jamais mélangée à ce qui existe.

Deux axes, dans cet ordre :

1. **Anticiper qu'une stratégie a cessé d'être bonne**, pour la retirer au bon moment
   plutôt qu'après coup. Autrement dit : détecter une dégradation d'edge pendant qu'elle se
   produit, pas trois ans plus tard dans un rapport annuel.
2. **Déterminer quand injecter une mesure d'urgence** (par exemple un `force flat all`)
   pour esquiver un événement qui tuerait la stratégie, et surtout à quel seuil cette mesure
   coûte plus qu'elle ne protège.

**Contrainte de formulation.** Ces deux points sont des intentions, pas des résultats. Ils
doivent être écrits au futur ou sous un intitulé « en cours » / « prochaines étapes », et
aucun chiffre ne doit leur être attaché. Le reste du site tire sa crédibilité de sa
prudence : une promesse chiffrée non tenue la détruirait.

Un lien honnête existe avec le reverse stress déjà en ligne : il mesure déjà ce qu'une
mesure de protection **coûte** quand elle se déclenche. Le dire en une phrase donne de la
continuité au programme sans rien promettre.

---

## Vérifications avant de rendre

- [ ] Aucun tiret cadratin ajouté hors bloc console
- [ ] Aucun chiffre en monospace dans le texte courant
- [ ] `docs/index.html` : versions FR **et** EN modifiées, contenu équivalent
- [ ] Les deux copies (fichier `docs/` et brouillon artifact) sont à jour
- [ ] Le mot « utilisateur » apparaît bien dans le nouveau chapô stateful
- [ ] La section « prochaines étapes » ne porte aucun chiffre et est visuellement séparée
- [ ] Rendu vérifié en clair et en sombre si la page gère les deux thèmes

Rendre : le diff, et les deux chapôs réécrits en clair dans la réponse pour relecture avant
publication.
