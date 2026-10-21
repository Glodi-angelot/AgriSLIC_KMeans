"""Compare a re-downloaded Sentinel-2 tile with the original extracted CSV.

Usage (from project folder):
 python verifier_tuile_test.py --tuile 276 --dataset dataset_lite
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from extraire_caracteristiques_slic import traiter_tuile, noms_colonnes_features


def main() -> int:
    parser = argparse.ArgumentParser(description="Vérifier l'alignement de la tuile avec le corpus officiel")
    parser.add_argument('--tuile', default='276')
    parser.add_argument('--dataset', type=Path, default=Path('dataset_lite'))
    parser.add_argument('--reference', type=Path, default=Path('resultats/caracteristiques_superpixels.csv'))
    args = parser.parse_args()

    chemin_manifest = args.dataset / 'manifest.csv'
    if not chemin_manifest.is_file() or not args.reference.is_file():
        print('ERREUR : manifest.csv ou fichier de référence introuvable.')
        return 2
    manifeste = pd.read_csv(chemin_manifest)
    groupe = manifeste[manifeste['tile_id'].astype(str) == str(args.tuile)]
    groupe = groupe.copy()
    groupe['dossier'] = groupe['dossier'].astype(str).str.replace('\\', '/', regex=False)
    if len(groupe) != 3:
        print(f'ERREUR : la tuile {args.tuile} doit avoir 3 dates dans le manifest (trouvé : {len(groupe)}).')
        return 2

    config = SimpleNamespace(n_segments=200, compactness=10.0, sigma=1.0, surface_min=20,
                             purete_classe_min=0.80, purete_parcelle_min=0.70,
                             couverture_label_min=0.80, pixels_clairs_min=0.70)
    try:
        lignes, nombre_slic = traiter_tuile(args.dataset.resolve(), groupe, config)
    except Exception as exc:
        print(f'ERREUR : impossible de recalculer SLIC : {exc}')
        return 2

    nouveau = pd.DataFrame(lignes).sort_values('superpixel_id').reset_index(drop=True)
    original = pd.read_csv(args.reference)
    original = original[original['tile_id'].astype(str) == str(args.tuile)]
    original = original.sort_values('superpixel_id').reset_index(drop=True)

    print(f'Tuile {args.tuile} : {nombre_slic} régions SLIC recalculées')
    print(f'Régions fiables : {len(nouveau)} recalculées / {len(original)} dans le CSV de référence')
    if len(nouveau) != len(original) or nouveau.empty:
        print('ECHEC : la liste de régions ne correspond pas. Ne pas utiliser ces images pour cartographier les groupes sauvegardés.')
        return 1

    for colonne in ['superpixel_id', 'field_id', 'class_code', 'surface_pixels']:
        if not np.array_equal(nouveau[colonne].to_numpy(), original[colonne].to_numpy()):
            print(f'ECHEC : désaccord sur {colonne}. Ne pas afficher une fausse carte.')
            return 1

    features = noms_colonnes_features()
    if not np.allclose(nouveau[features].to_numpy(dtype=float),
                       original[features].to_numpy(dtype=float), rtol=1e-5, atol=1e-6):
        plus_grand_ecart = np.max(np.abs(nouveau[features].to_numpy(dtype=float) -
                                         original[features].to_numpy(dtype=float)))
        print(f'ECHEC : caractéristiques non conformes (écart maximal {plus_grand_ecart:.7g}).')
        return 1

    print(f'OK : {len(features)} caractéristiques et toutes les régions concordent avec le CSV officiel.')
    print('La carte peut être affichée avec les modèles déjà entraînés.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
