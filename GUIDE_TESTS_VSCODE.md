# AgriSLIC - installation et premiers tests VS Code

Ce dossier est le **projet scientifique**, distinct de l'application de demonstration Streamlit V2.1.
Les resultats/modeles sauvegardes sont conserves; les scripts ne doivent pas etre reentraines avant la verification.

## Windows 64 bits / PowerShell dans VS Code

1. Extraire l'archive et ouvrir le dossier `AgriSLIC_KMeans_Examen` dans VS Code.
2. Verifier `py -0p` puis `py -3.12 --version`.
3. `py -3.12 -m venv .venv`
4. `./.venv/Scripts/python.exe -m pip install --upgrade pip`
5. `./.venv/Scripts/python.exe -m pip install --only-binary=:all: -r requirements.txt`
6. `./.venv/Scripts/python.exe -m pip check`
7. `./.venv/Scripts/python.exe verifier_projet.py`
8. `./.venv/Scripts/python.exe -m py_compile telecharger_dataset_lite.py extraire_caracteristiques_slic.py entrainer_kmeans.py app.py verifier_projet.py`
9. `./.venv/Scripts/python.exe extraire_caracteristiques_slic.py --dataset test_dataset --sortie verification/extraction_test.csv --limite-tuiles 1`

Dans PowerShell, le prefixe exact de l'interpreteur est `.\.venv\Scripts\python.exe`.
On peut aussi double-cliquer sur `LANCER_TESTS_VSCODE.bat` apres avoir installe les dependances.

## Resultats attendus

- 8945 superpixels dans les resultats enregistres; 1949 parcelles; 1789 superpixels de test; K=4 et K=9 reproduits a 100 % (concordance des clusters, **pas** precision de cultures).
- Extraction de la tuile 2461 avec les parametres d'origine; attendus: environ 196 regions SLIC, 31 conservees; verification en fonction du socle scientifique.
- Le fichier nouveau est `verification/extraction_test.csv` et ne modifie pas `resultats/`.
- `python -m streamlit run app.py` permet de consulter l'ancienne interface, mais pour l'examen utiliser l'application V2.1 **separee**.

## Avant GitHub public

Ne pas pousser `.venv`, `dataset_lite`, fichiers `.env`, identifiants, comptes de cloud ni temoins secrets.
Les extraits de donnees Sentinel-2 appartiennent a leurs auteurs: verifier et citer la source et sa licence CC BY 4.0 (comme dans le rapport) avant republication.
Les modeles .joblib ne doivent etre charges qu'a partir d'une archive de confiance, car le chargement Python peut executer du code.

Le depot pourra inclure `resultats/`, les tuiles d'exemple si leur licence le permet, et les instructions de reproductibilite.
