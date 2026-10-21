# Notes de version — Kit de préparation V1.1

Ce dossier partageable vise à faire gagner du temps avant l'oral.
Il ne constitue **pas encore** une révision méthodologique complète de
l'expérience sur 200 tuiles.

## Vérifié
- 8 945 superpixels extraits pour 1 949 parcelles ; séparation des 1 559
  parcelles d'apprentissage et des 390 de test.
- Reproduction exacte des groupes enregistrés sur 1 789 superpixels de test,
  pour les modèles K=4 et K=9, avec scikit-learn 1.8.0.
- Lecture et segmentation SLIC sur une tuile d'exemple pour N=200, 500,
  1 000 et 2 000 (nombre effectif de régions approximatif).
- Syntaxe des scripts Python vérifiée.

## Modifié sans changer les résultats scientifiques
- `app.py` : mode de démonstration SLIC à partir des deux tuiles incluses si
  `dataset_lite` est absent ; absence de cartes de test signalée plutôt que
  tentative de chargement d'images inexistantes.
- `requirements.txt` : ajout `boto3` et `tqdm` pour la récupération du dataset.
- `verifier_projet.py` : contrôle des fichiers, des données et des modèles.
- `GUIDE_VARIANTE_A.md` : lancement sous Windows et deux modes de démo.
- `FICHE_DEFENSE_GROUPE.md` et PDF correspondant : révision collective.
- PowerPoint V1.1 : deux clarifications sur les labels, raccourcissement du
  texte débordant sur la diapositive 10.

## Non vérifié / réservé à la version finale
- L'interface Streamlit elle-même n'a pas été testée en fonctionnement :
  Streamlit n'était pas installé ici et l'accès réseau pour l'installer a échoué.
  La tester sur un PC du groupe avec `python -m streamlit run app.py`.
- Les 200 tuiles originales n'ont pas été réexécutées ; seules deux tuiles
  d'exemple sont jointes aux images brutes de ce kit.
- La sélection des 24 variables devrait être justifiée sur entraînement ou
  validation, et non à partir du score du test final.
- Les annotations réelles sont utilisées au filtrage initial des régions,
  même si K-means n'utilise aucun label pour ajuster ses centres.
- La comparaison Random Forest n'est pas une confrontation expérimentale sur
  le même jeu de test, seulement une comparaison méthodologique.

## Avant de diffuser la version scientifique définitive
Tester l'application sur Windows, confirmer les captures de démonstration,
puis décider si le protocole expérimental doit être corrigé et rerun.
Adapter alors le rapport et les diapositives aux nouveaux résultats éventuels.
