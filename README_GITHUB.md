# AgriSLIC — Sujet 07, Variante A

Projet de reconnaissance de formes, Master 1 IA & Data Science, Universite de Kinshasa.

**Objectif :** regrouper des regions agricoles similaires par segmentation SLIC et clustering K-means sur des bandes Sentinel-2.

## Reproduction
- Installer Python 3.12 64 bits et les dependances de `requirements.txt`.
- Executer `python verifier_projet.py`.
- Pour une extraction de test, voir `GUIDE_TESTS_VSCODE.md`.
- La demonstration d'examen Streamlit V2.1 reside dans un projet distinct.

## Principaux resultats du corpus (precalcules)
- 8 945 superpixels conserves issus de 1 949 parcelles.
- 24 moyennes sur 64 caracteristiques extraites pour K-means.
- K=4 : silhouette de 0,290 sur test; K=9 : rapprochement indicatif 35,7 % avec les cultures.

Les annotations de cultures sont utilisees pour le filtrage et l'evaluation mais **pas** pour ajuster les centres K-means. La selection des 24 variables utilise en partie le jeu de test (limite methodologique).

## Provenance des donnees
Western Cape Department of Agriculture / Radiant Earth Foundation, *South Africa Crop Type Competition* (2021). Source: https://source.coop/radiantearth/south-africa-crops-competition (lien a verifier avant publication). Licence presentee dans le rapport: CC BY 4.0. Les donnees originales ne sont pas fournies integralement.

## A noter
- Le dossier `.venv/` et le corpus complet `dataset_lite/` ne doivent pas etre commits.
- Le reentrainement cree de nouveaux resultats ; sauvegarder les artefacts originaux avant de le tenter.
- Charger les fichiers `.joblib` uniquement si le depot est de confiance.
