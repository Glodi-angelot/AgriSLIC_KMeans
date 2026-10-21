# AgriSLIC — Segmentation SLIC et regroupement K-means

Projet de **reconnaissance de formes** — **Sujet 07, Variante A**, Master 1 Intelligence artificielle et Data Science, Université de Kinshasa (UNIKIN), année académique 2025–2026.

## Objectif

Étudier si les caractéristiques spectrales de régions agricoles issues de Sentinel-2 permettent de découvrir des groupes similaires **sans donner les noms des cultures à K-means pendant son entraînement**.

Chaîne du projet : **images Sentinel-2 → prétraitement → segmentation SLIC → extraction des caractéristiques → standardisation → K-means → évaluation**.

> **Important :** K-means est non supervisé, mais les annotations du corpus interviennent dans certains filtres de qualité, dans l'organisation de la séparation et dans l'évaluation a posteriori. Les clusters ne sont pas des noms de cultures prédits.

## Résultats de référence

| Indicateur | Résultat |
| --- | ---: |
| Tuiles sélectionnées dans l'expérience | 200 |
| Superpixels conservés | 8 945 |
| Parcelles représentées | 1 949 |
| Caractéristiques extraites | 64 |
| Moyennes utilisées par K-means | 24 |
| Superpixels d'apprentissage / de test | 7 156 / 1 789 |
| Silhouette K = 4 | 0,290 |
| Silhouette K = 9 | 0,243 |
| Correspondance indicative K = 9 | 35,7 % |

Le modèle **K = 4** donne une meilleure séparation interne ; **K = 9** présente une meilleure correspondance *indicative* avec les neuf cultures annotées, mais cette correspondance reste faible. Ce n'est **pas une précision de classification supervisée**.

## Installation sur Windows (VS Code)

Pré-requis : **Python 3.12 64 bits**, Git et une connexion Internet pour installer les dépendances. Ouvrez le dossier cloné dans VS Code puis, dans PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

### Vérifier les données et les modèles

```powershell
.\.venv\Scripts\python.exe verifier_projet.py
```

Le script doit confirmer la présence des artefacts, **8 945 superpixels**, **1 949 parcelles**, la séparation train/test et la reproduction des affectations enregistrées pour K = 4 et K = 9.

### Refaire une petite extraction SLIC

```powershell
New-Item -ItemType Directory -Force verification | Out-Null
.\.venv\Scripts\python.exe extraire_caracteristiques_slic.py --dataset test_dataset --sortie verification/extraction_test.csv --limite-tuiles 1
```

Pour la tuile d'exemple 2461, le résultat de référence est **196 régions SLIC générées, 31 superpixels conservés, 64 caractéristiques**. Le dossier `verification/` est ignoré par Git.

### Réentraîner les modèles sans écraser les artefacts officiels

```powershell
.\.venv\Scripts\python.exe entrainer_kmeans.py --donnees resultats\caracteristiques_superpixels.csv --split-reference resultats\split_test_reference.csv --caracteristiques moyennes --k-min 4 --k-max 4 --k-final 4 --sortie verification\entrainement_k4
.\.venv\Scripts\python.exe entrainer_kmeans.py --donnees resultats\caracteristiques_superpixels.csv --split-reference resultats\split_test_reference.csv --caracteristiques moyennes --k-min 9 --k-max 9 --k-final 9 --sortie verification\entrainement_k9
```

Sur le PC de développement, les résultats obtenus étaient respectivement une silhouette de **0,2901** (K = 4) et de **0,2428** (K = 9). Les tests utilisent les caractéristiques déjà sauvegardées ; ils **ne refont pas toute l'acquisition des 200 tuiles**.

## Fichiers principaux

| Fichier ou répertoire | Rôle |
| --- | --- |
| `telecharger_dataset_lite.py` | Préparation / téléchargement d'un sous-ensemble des tuiles source. |
| `extraire_caracteristiques_slic.py` | Segmentation, indices de végétation, 64 caractéristiques, contrôles qualité. |
| `entrainer_kmeans.py` | Séparation, standardisation, entraînement, métriques et export des modèles. |
| `verifier_projet.py` | Contrôles de cohérence et de reproductibilité. |
| `resultats/` | Caractéristiques, modèles `.joblib`, scores et prédictions de référence. |
| `test_dataset/` | Petites tuiles Sentinel-2 et masques fournis pour l'exemple. |
| `app.py` | Première interface exploratoire Streamlit du projet scientifique. |
| `GUIDE_TESTS_VSCODE.md` | Procédure de test détaillée. |

**Pour la démonstration orale officielle**, utilisez plutôt le dépôt séparé : [AgriSLIC-Streamlit-Demo](https://github.com/Glodi-angelot/AgriSLIC-Streamlit-Demo) **une fois celui-ci publié**.

## Provenance des données et limites

Données d'origine : **Western Cape Department of Agriculture / Radiant Earth Foundation**, *South Africa Crop Type Competition* (2021) : https://source.coop/radiantearth/south-africa-crops-competition — licence signalée dans le rapport : **CC BY 4.0**. Attribution à conserver pour les extraits redistribués. Ce dépôt **ne contient pas le dataset complet**.

Limites : sous-ensemble expérimental, classes déséquilibrées, dépendance du filtrage aux annotations, choix des caractéristiques en partie éclairé par le jeu de test, et correspondance faible entre clusters et classes agricoles. Les résultats ne sont pas directement généralisables aux cultures de RDC.

**Sécurité :** n'ajoutez jamais `.venv`, `.env`, secrets ou identifiants à Git. Ne chargez des fichiers `.joblib` que depuis une source de confiance.
