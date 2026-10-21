# AgriSLIC — Variante A K-means — Guide de lancement V1.1

## Objet
Ce kit permet de réviser la chaîne Sentinel-2 → SLIC → 64 caractéristiques →
24 moyennes standardisées → K-means K=4 / K=9 → interprétation et évaluation.
Les résultats scientifiques originaux n'ont PAS été modifiés.

## Deux modes de démonstration
- **Sans téléchargement supplémentaire (ZIP fourni)** : `test_dataset` contient
  deux tuiles complètes pour la page « Segmentation SLIC ». Les CSV et modèles
  fournis permettent les autres pages, notamment le recalcul des groupes à
  partir des caractéristiques des tuiles de test, **sans carte satellitaire**
  de ces tuiles. Les deux tuiles de démonstration ne prouvent pas les métriques
  obtenues sur 200 tuiles.
- **Démonstration complète avec cartes** : copier au même niveau que `app.py`
  le dossier original `dataset_lite` contenant les 200 tuiles et `manifest.csv`.
  Cette opération est nécessaire pour montrer les cartes des tuiles de test.

## Installation — Windows / PowerShell
À partir du dossier `AgriSLIC_KMeans_Examen` :
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python verifier_projet.py
python -m streamlit run app.py
```
Si PowerShell refuse `Activate.ps1`, utiliser sans activation :
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe verifier_projet.py
.venv\Scripts\python.exe -m streamlit run app.py
```
Ne jamais transférer le dossier `.venv` à ses collègues.

## Vérification des modèles
`python verifier_projet.py` vérifie l'accès aux fichiers, la séparation des
parcelles et la reproduction des prédictions de K=4 et K=9.

## Rejouer l'apprentissage (facultatif, calcul plus long)
```powershell
python entrainer_kmeans.py --k-final 4 --sortie resultats/kmeans_k4
python entrainer_kmeans.py --k-final 9 --sortie resultats/kmeans_k9
```
Ne relancer que si les artefacts du modèle doivent être régénérés. Conserver
un exemplaire des résultats fournis pour garantir la comparaison.

## Télécharger toutes les images (facultatif, connexion requise)
```powershell
python telecharger_dataset_lite.py --nombre-tuiles 200 --racine dataset_lite
```
Cette récupération peut demander du temps et de l'espace disque. Les tuiles
retéléchargées peuvent différer du jeu expérimental initial : pour reproduire
rigoureusement les cartes, il faut **récupérer le dossier original**.

## Résultats annoncés sur les 1 789 superpixels du test
| Config. | Silhouette | Davies-Bouldin | ARI | NMI | Correspondance indicative |
| --- | ---: | ---: | ---: | ---: | ---: |
| K=4 | 0,290 | 1,014 | 0,079 | 0,131 | 30,4 % |
| K=9 | 0,243 | 1,066 | 0,119 | 0,196 | 35,7 % |

## Précautions scientifiques à expliquer
1. K-means n'apprend pas à partir des noms des cultures ; le prétraitement
   expérimental emploie néanmoins des annotations pour filtrer les superpixels
   et organiser le split.
2. Le choix des 24 variables a été justifié en partie par le test ; une
   révision méthodologique devrait sélectionner les variables à l'aide d'une
   validation distincte et réserver le test final.
3. Le résultat K=4 est une structure spectrale, **pas** quatre cultures.
4. Les groupes K=9 ne représentent pas neuf cultures distinguées : plusieurs
   groupes reçoivent une même étiquette majoritaire.
5. Aucune comparaison expérimentale complète avec Random Forest sur le même
   split n'est incluse dans ce kit ; la comparaison est méthodologique.

Voir `FICHE_DEFENSE_GROUPE.md` pour préparer l'oral.
