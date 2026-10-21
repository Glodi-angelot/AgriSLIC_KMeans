"""Contrôles rapides AgriSLIC — sans entraînement ni Streamlit."""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
RESULTATS = R / 'resultats'

def verifier(test, message):
    print(('OK  ' if test else 'ECHEC  ') + message)
    if not test:
        raise AssertionError(message)

def main():
    fichiers = [RESULTATS / 'caracteristiques_superpixels.csv',
                RESULTATS / 'caracteristiques_superpixels_resume.json']
    for k in (4, 9):
        fichiers += [RESULTATS/f'kmeans_k{k}'/'modele_kmeans.joblib',
                     RESULTATS/f'kmeans_k{k}'/'predictions_superpixels_test.csv']
    for f in fichiers:
        verifier(f.is_file(), f'Fichier présent : {f.relative_to(R)}')
    df = pd.read_csv(fichiers[0])
    verifier(len(df)==8945, '8 945 superpixels extraits dans le CSV')
    verifier(df['field_id'].nunique()==1949, '1 949 parcelles représentées')
    for k in (4,9):
        base=RESULTATS/f'kmeans_k{k}'
        out=pd.read_csv(base/'predictions_superpixels_test.csv')
        verifier(len(out)==1789,f'K={k}: 1 789 superpixels de test')
        trainfields=set(df['field_id'].astype(int))-set(out['field_id'].astype(int))
        testfields=set(out['field_id'].astype(int))
        verifier(len(trainfields)==1559 and len(testfields)==390,
                f'K={k}: parcelles train/test distinctes (1 559 / 390)')
        a=joblib.load(base/'modele_kmeans.joblib')
        merged=out.merge(df, on=['tile_id','superpixel_id'], suffixes=('_pred',''),validate='one_to_one')
        features=a['colonnes_features']
        verifier(len(features)==24, f'K={k}: 24 caractéristiques dans le modèle')
        predict=a['kmeans'].predict(a['scaler'].transform(merged[features].astype(float)))
        verifier(np.array_equal(predict,merged['groupe_kmeans'].to_numpy()),
                f'K={k}: 100 % des clusters de test reproduits')
    demo=(R/'test_dataset'/'manifest.csv').is_file()
    full=(R/'dataset_lite'/'manifest.csv').is_file()
    verifier(demo or full,'Des images Sentinel-2 sont disponibles pour SLIC')
    print('\nCONCLUSION : contrôles des artefacts réussis.')
    print('Mode : '+('dataset original complet' if full else '2 tuiles exemple ; cartes de test non fournies'))
    return 0
if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as e:
        print('ÉCHEC DE VÉRIFICATION :',e)
        sys.exit(1)
