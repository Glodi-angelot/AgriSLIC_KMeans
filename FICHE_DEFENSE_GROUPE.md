# Fiche d'étude commune — Reconnaissance de formes (AgriSLIC, variante A)

**Objectif** : donner aux dix membres un vocabulaire commun et un déroulé
fiable pour la défense du 23 octobre 2026. Cette fiche résume le rapport fourni,
et distingue explicitement les limites observées.

## Le projet en une minute
Nous cherchons à déterminer si les images Sentinel-2 permettent de former
spontanément des groupes de régions agricoles similaires, sans enseigner
à K-means les noms des cultures. Nous retenons 200 tuiles sur 2 650, chacune
observée à trois périodes (4 bandes : bleu B02, vert B03, rouge B04, proche
infrarouge B08). SLIC segmente les tuiles en régions, et les contrôles
expérimentaux conservent 8 945 superpixels sur 39 200. Pour chaque région,
64 caractéristiques sont extraites ; K-means utilise 24 moyennes et teste K=2
à K=12. Les métriques internes recommandent K=4, mais le rapprochement avec
les neuf classes reste faible, même à K=9.

## Chapitres essentiels (rappel)
1. **Introduction** : explorer des régions agricoles similaires sans noms de
   cultures dans l'apprentissage de K-means.
2. **Concepts** : pixel ≠ superpixel ≠ parcelle ≠ cluster ; SLIC forme des
   régions contiguës, K-means rassemble ensuite des régions semblables.
3. **Données** : South Africa Crop Type Competition, 200 tuiles, 3 périodes,
   4 bandes, 9 cultures.
4. **SLIC** : N=200 pour les métriques finales ; compacité=10, sigma=1.
   Les démonstrations visuelles N=500, 1000, 2000 ne sont pas les métriques finales.
5. **Caractéristiques** : 64 extraites = 48 statistiques spectrales et
   végétatives (moyenne/écart-type) + 16 indices de variation ; 24 moyennes
   finalement entrées dans K-means. Pas de PCA dans la variante A finale.
6. **K-means** : 7 156 superpixels train, 1 789 test ; split par parcelle
   (1 559 vs 390 parcelles). StandardScaler ajusté sur train.
7. **Évaluation** : silhouette et Davies-Bouldin mesurent la qualité interne.
   ARI/NMI comparent ensuite aux classes ; correspondance après étiquetage
   majoritaire ne représente pas une précision supervisée ordinaire.
8. **Résultats** : K=4 (silhouette .290, Davies-Bouldin 1.014), K=9
   (silhouette .243, ARI .119, NMI .196, correspondance indicative 35.7%).
9. **Application** : accueil, segmentation SLIC, groupes K-means, résultats.
10. **Variante B** : Random Forest apprend les classes étiquetées ; ce n'est
    pas le même objectif que le clustering non supervisé.
11. **Limites** : peu de tuiles, classes déséquilibrées, groupes mêlant
    plusieurs cultures, sensibilité au choix K et aux conditions spatiales.
12. **Conclusion** : quatre grands profils spectraux apparaissent ; cela
    n'implique pas qu'ils correspondent aux neuf cultures.

## Trois formulations à éviter et à remplacer
- Éviter « SLIC retrouve parfaitement les contours des parcelles ».
  Dire « SLIC regroupe des pixels voisins et semblables ; un superpixel peut
  traverser une frontière et le filtrage par parcelle réduit certains cas ».
- Éviter « 35,7 % de cultures reconnues avec K-means ».
  Dire « 35,7 % de correspondance indicative au niveau superpixel pour K=9,
  après association postérieure des groupes aux classes majoritaires ».
- Éviter « aucune étiquette n'a jamais été utilisée dans le projet ».
  Dire « K-means ne reçoit pas de labels pendant l'apprentissage ; le
  filtrage et certaines évaluations emploient toutefois les annotations ».

## Déroulé suggéré de la démonstration (environ 3–4 minutes)
1. Accueil : rappeler 200 tuiles, 64 extraites et 24 retenues.
2. Segmentation : changer N=500, 1000, 2000 sur la même tuile d'exemple.
3. Groupement : choisir K=4, puis K=9 ; afficher les superpixels de test et
   expliquer le rôle des centres et des correspondances indicatives.
4. Résultats : exposer le compromis entre structure naturelle (K=4) et
   proximité avec les neuf cultures (K=9).

**Avertissement pour le ZIP fourni** : la segmentation dispose de deux tuiles
réelles d'exemple, insuffisantes pour couvrir le corpus de 200 tuiles. Les cartes des
tuiles de test nécessitent le dossier original `dataset_lite`.

## Questions fréquentes : réponses courtes
- **Pourquoi SLIC ?** Réduire le nombre d'unités et exploiter la proximité
  spatiale, tout en gardant des régions spectralement cohérentes.
- **Pourquoi 24 au lieu de 64 ?** Les indicateurs internes observés sont plus
  favorables sur 24 moyennes ; le protocole de sélection doit encore être
  renforcé pour ne pas dépendre du test final.
- **Pourquoi K=4 si neuf cultures ?** Le nombre de groupes naturels n'est pas
  nécessairement égal au nombre de classes annotées.
- **K-means est-il supervisé ?** Non ; le calcul des centres n'utilise pas les
  étiquettes. Attention : les filtres expérimentaux, eux, en utilisent.
- **Quelle est la faiblesse majeure ?** Les signatures se chevauchent entre
  cultures, et les classes peuvent varier selon les dates et les conditions.
- **Pourquoi pas Random Forest ?** C'est la variante B et elle répond au but
  distinct de prédire des cultures connues avec des étiquettes fiables.

## À maîtriser par TOUS les membres
La chaîne entière, la différence SLIC/K-means, les unités, le choix des
24 moyennes, la signification de K=4/K=9, la lecture des métriques et les
limites. Ne distribuez pas uniquement des diapositives sans étude commune.
