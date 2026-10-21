"""Extraction classique de superpixels SLIC et de caractéristiques spectrales.

Entrée attendue
---------------
Le dossier ``dataset_lite`` produit par ``telecharger_dataset_lite.py``.

Sorties
-------
Un fichier CSV contenant une ligne par superpixel suffisamment pur, ainsi
qu'un résumé JSON. Aucune méthode de Deep Learning n'est utilisée.

Exemples
--------
Test sur deux tuiles :
    python extraire_caracteristiques_slic.py --limite-tuiles 2 \
        --sortie resultats/test_caracteristiques.csv

Traitement complet :
    python extraire_caracteristiques_slic.py \
        --sortie resultats/caracteristiques_superpixels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from skimage.segmentation import slic
from tqdm import tqdm


BANDES = ("B02", "B03", "B04", "B08")
PERIODES = ("debut_saison", "milieu_saison", "fin_saison")
INDICES = ("ndvi", "gndvi", "savi", "evi")

CLASSES = {
    1: "Lucerne/Medics",
    2: "Planted pastures (perennial)",
    3: "Fallow",
    4: "Wine grapes",
    5: "Weeds",
    6: "Small grain grazing",
    7: "Wheat",
    8: "Canola",
    9: "Rooibos",
}


def arguments():
    parser = argparse.ArgumentParser(
        description="Extraire les superpixels SLIC et leurs caractéristiques spectrales."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset_lite"),
        help="Dossier du dataset allégé (défaut : dataset_lite).",
    )
    parser.add_argument(
        "--sortie",
        type=Path,
        default=Path("resultats/caracteristiques_superpixels.csv"),
        help="Fichier CSV de sortie.",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=200,
        help="Nombre SLIC demandé par tuile (défaut : 200).",
    )
    parser.add_argument(
        "--compactness",
        type=float,
        default=10.0,
        help="Compacité SLIC (défaut : 10).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Lissage préalable SLIC (défaut : 1).",
    )
    parser.add_argument(
        "--purete-classe-min",
        type=float,
        default=0.80,
        help="Pureté minimale de l'étiquette du superpixel (défaut : 0.80).",
    )
    parser.add_argument(
        "--purete-parcelle-min",
        type=float,
        default=0.70,
        help="Pureté minimale de l'identifiant de parcelle (défaut : 0.70).",
    )
    parser.add_argument(
        "--couverture-label-min",
        type=float,
        default=0.80,
        help="Part minimale du superpixel couverte par les labels (défaut : 0.80).",
    )
    parser.add_argument(
        "--pixels-clairs-min",
        type=float,
        default=0.70,
        help="Part minimale de pixels clairs par date (défaut : 0.70).",
    )
    parser.add_argument(
        "--surface-min",
        type=int,
        default=20,
        help="Surface minimale d'un superpixel en pixels (défaut : 20).",
    )
    parser.add_argument(
        "--limite-tuiles",
        type=int,
        default=0,
        help="Limiter le nombre de tuiles pour un test ; 0 traite tout.",
    )
    return parser.parse_args()


def identifiant_texte(valeur) -> str:
    """Évite des identifiants tels que '1.0' après lecture du CSV."""
    texte = str(valeur)
    if texte.endswith(".0"):
        texte = texte[:-2]
    return texte


def lire_raster(chemin: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(chemin) as src:
        tableau = src.read(1)
        meta = {
            "shape": tableau.shape,
            "transform": src.transform,
            "crs": src.crs,
            "dtype": tableau.dtype,
        }
    return tableau, meta


def convertir_reflectance(tableau: np.ndarray) -> np.ndarray:
    """Convertit les bandes uint8 du dataset dans l'intervalle [0, 1]."""
    resultat = tableau.astype(np.float32)
    maximum = float(np.nanmax(resultat)) if resultat.size else 0.0
    if maximum > 1.0:
        facteur = 255.0 if maximum <= 255.0 else 10000.0
        resultat /= facteur
    return np.clip(resultat, 0.0, 1.0)


def charger_observation(dataset: Path, ligne: pd.Series):
    tuile = identifiant_texte(ligne["tile_id"])
    date = identifiant_texte(ligne["date"])
    # Le manifest fourni utilise des antislashs Windows : accepter aussi Linux/macOS.
    dossier = dataset / Path(str(ligne["dossier"]).replace("\\", "/"))

    couches = []
    meta_reference = None
    for bande in BANDES:
        chemin = dossier / f"{tuile}_{date}_{bande}_10m.tif"
        tableau, meta = lire_raster(chemin)
        if meta_reference is None:
            meta_reference = meta
        elif tableau.shape != meta_reference["shape"]:
            raise ValueError(f"Dimensions incompatibles dans {dossier}")
        couches.append(convertir_reflectance(tableau))

    clm, _ = lire_raster(dossier / f"{tuile}_{date}_CLM_10m.tif")
    clair = clm == 0
    cube = np.stack(couches, axis=-1)
    return cube, clair, meta_reference


def calculer_indices(cube: np.ndarray) -> dict[str, np.ndarray]:
    bleu = cube[..., 0]
    vert = cube[..., 1]
    rouge = cube[..., 2]
    pir = cube[..., 3]
    eps = np.float32(1e-6)

    ndvi = (pir - rouge) / (pir + rouge + eps)
    gndvi = (pir - vert) / (pir + vert + eps)
    savi = 1.5 * (pir - rouge) / (pir + rouge + 0.5 + eps)
    evi = 2.5 * (pir - rouge) / (pir + 6.0 * rouge - 7.5 * bleu + 1.0 + eps)

    return {
        "ndvi": np.clip(ndvi, -1.0, 1.0),
        "gndvi": np.clip(gndvi, -1.0, 1.0),
        "savi": np.clip(savi, -1.0, 1.0),
        "evi": np.clip(evi, -2.0, 2.0),
    }


def composite_pour_slic(cubes: list[np.ndarray], clairs: list[np.ndarray]):
    pile = np.stack(cubes, axis=0).astype(np.float32)
    validite = np.stack(clairs, axis=0)
    pile = np.where(validite[..., None], pile, np.nan)

    with np.errstate(invalid="ignore"):
        composite = np.nanmedian(pile, axis=0)

    for indice_bande in range(composite.shape[-1]):
        bande = composite[..., indice_bande]
        valides = np.isfinite(bande)
        remplissage = float(np.nanmedian(bande)) if np.any(valides) else 0.0
        bande[~valides] = remplissage

        valeurs = bande[valides]
        if valeurs.size:
            bas, haut = np.percentile(valeurs, [2, 98])
        else:
            bas, haut = 0.0, 1.0
        if haut <= bas:
            haut = bas + 1e-6
        composite[..., indice_bande] = np.clip((bande - bas) / (haut - bas), 0, 1)

    return composite


def statistiques_par_segment(
    segments: np.ndarray,
    valeurs: np.ndarray,
    masque_valide: np.ndarray,
    nombre_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = segments[masque_valide].ravel()
    x = valeurs[masque_valide].astype(np.float64).ravel()

    compte = np.bincount(ids, minlength=nombre_segments).astype(np.float64)
    somme = np.bincount(ids, weights=x, minlength=nombre_segments)
    somme_carres = np.bincount(ids, weights=x * x, minlength=nombre_segments)

    moyenne = np.full(nombre_segments, np.nan, dtype=np.float64)
    ecart_type = np.full(nombre_segments, np.nan, dtype=np.float64)
    non_vide = compte > 0
    moyenne[non_vide] = somme[non_vide] / compte[non_vide]
    variance = np.zeros(nombre_segments, dtype=np.float64)
    variance[non_vide] = (
        somme_carres[non_vide] / compte[non_vide] - moyenne[non_vide] ** 2
    )
    ecart_type[non_vide] = np.sqrt(np.maximum(variance[non_vide], 0.0))
    return moyenne, ecart_type, compte


def etiquettes_majoritaires(
    segments: np.ndarray,
    labels: np.ndarray,
    field_ids: np.ndarray,
    nombre_segments: int,
):
    surface = np.bincount(segments.ravel(), minlength=nombre_segments).astype(np.int64)

    comptes_classes = np.zeros((10, nombre_segments), dtype=np.int64)
    for code in range(1, 10):
        comptes_classes[code] = np.bincount(
            segments[labels == code], minlength=nombre_segments
        )

    pixels_labellises = comptes_classes[1:].sum(axis=0)
    classes = np.argmax(comptes_classes, axis=0)
    majoritaires = comptes_classes[classes, np.arange(nombre_segments)]
    purete_classe = np.divide(
        majoritaires,
        pixels_labellises,
        out=np.zeros(nombre_segments, dtype=float),
        where=pixels_labellises > 0,
    )
    couverture = np.divide(
        pixels_labellises,
        surface,
        out=np.zeros(nombre_segments, dtype=float),
        where=surface > 0,
    )

    meilleur_field = np.zeros(nombre_segments, dtype=np.int64)
    meilleur_compte = np.zeros(nombre_segments, dtype=np.int64)
    masque_field = field_ids > 0
    if np.any(masque_field):
        base = int(field_ids[masque_field].max()) + 1
        cles = (
            segments[masque_field].astype(np.int64) * base
            + field_ids[masque_field].astype(np.int64)
        )
        cles_uniques, comptes = np.unique(cles, return_counts=True)
        for cle, compte in zip(cles_uniques, comptes):
            segment_id = int(cle // base)
            field_id = int(cle % base)
            if compte > meilleur_compte[segment_id]:
                meilleur_compte[segment_id] = int(compte)
                meilleur_field[segment_id] = field_id

    pixels_field = np.bincount(
        segments[masque_field], minlength=nombre_segments
    ).astype(np.int64)
    purete_field = np.divide(
        meilleur_compte,
        pixels_field,
        out=np.zeros(nombre_segments, dtype=float),
        where=pixels_field > 0,
    )

    return {
        "surface": surface,
        "classe": classes,
        "purete_classe": purete_classe,
        "couverture": couverture,
        "field_id": meilleur_field,
        "purete_field": purete_field,
    }


def noms_colonnes_features() -> list[str]:
    colonnes = []
    for periode in PERIODES:
        for bande in BANDES:
            colonnes.extend((f"{periode}_{bande}_moy", f"{periode}_{bande}_std"))
        for indice in INDICES:
            colonnes.extend((f"{periode}_{indice}_moy", f"{periode}_{indice}_std"))

    for variable in (*BANDES, *INDICES):
        colonnes.extend((f"temp_{variable}_amplitude", f"temp_{variable}_delta"))
    return colonnes


def traiter_tuile(dataset: Path, groupe: pd.DataFrame, args):
    tuile = identifiant_texte(groupe.iloc[0]["tile_id"])
    groupe = groupe.set_index("periode")
    if any(periode not in groupe.index for periode in PERIODES):
        raise ValueError(f"Les trois périodes sont absentes pour la tuile {tuile}")

    cubes = []
    clairs = []
    for periode in PERIODES:
        cube, clair, _ = charger_observation(dataset, groupe.loc[periode])
        cubes.append(cube)
        clairs.append(clair)

    shape = cubes[0].shape[:2]
    if any(cube.shape[:2] != shape for cube in cubes):
        raise ValueError(f"Dimensions temporelles incompatibles pour la tuile {tuile}")

    composite = composite_pour_slic(cubes, clairs)
    segments = slic(
        composite,
        n_segments=args.n_segments,
        compactness=args.compactness,
        sigma=args.sigma,
        start_label=0,
        channel_axis=-1,
        convert2lab=False,
        enforce_connectivity=True,
    ).astype(np.int32)
    nombre_segments = int(segments.max()) + 1

    labels, _ = lire_raster(dataset / "masques" / f"{tuile}.tif")
    field_ids, _ = lire_raster(dataset / "masques" / f"{tuile}_field_ids.tif")
    if labels.shape != shape or field_ids.shape != shape:
        raise ValueError(f"Masques non alignés pour la tuile {tuile}")

    etiquettes = etiquettes_majoritaires(
        segments, labels.astype(np.int64), field_ids.astype(np.int64), nombre_segments
    )

    valeurs_features: dict[str, np.ndarray] = {}
    clartes_segments = []
    moyennes_temporelles: dict[str, list[np.ndarray]] = {
        nom: [] for nom in (*BANDES, *INDICES)
    }

    for periode, cube, clair in zip(PERIODES, cubes, clairs):
        total_segment = np.bincount(segments.ravel(), minlength=nombre_segments)
        compte_clair = np.bincount(segments[clair], minlength=nombre_segments)
        taux_clair = np.divide(
            compte_clair,
            total_segment,
            out=np.zeros(nombre_segments, dtype=float),
            where=total_segment > 0,
        )
        clartes_segments.append(taux_clair)

        for indice_bande, bande in enumerate(BANDES):
            moyenne, std, _ = statistiques_par_segment(
                segments, cube[..., indice_bande], clair, nombre_segments
            )
            valeurs_features[f"{periode}_{bande}_moy"] = moyenne
            valeurs_features[f"{periode}_{bande}_std"] = std
            moyennes_temporelles[bande].append(moyenne)

        indices = calculer_indices(cube)
        for nom_indice in INDICES:
            moyenne, std, _ = statistiques_par_segment(
                segments, indices[nom_indice], clair, nombre_segments
            )
            valeurs_features[f"{periode}_{nom_indice}_moy"] = moyenne
            valeurs_features[f"{periode}_{nom_indice}_std"] = std
            moyennes_temporelles[nom_indice].append(moyenne)

    for variable, liste_moyennes in moyennes_temporelles.items():
        pile = np.stack(liste_moyennes, axis=0)
        valeurs_features[f"temp_{variable}_amplitude"] = np.nanmax(pile, axis=0) - np.nanmin(
            pile, axis=0
        )
        valeurs_features[f"temp_{variable}_delta"] = pile[-1] - pile[0]

    clartes = np.stack(clartes_segments, axis=0)
    valide = (
        (etiquettes["surface"] >= args.surface_min)
        & (etiquettes["classe"] >= 1)
        & (etiquettes["purete_classe"] >= args.purete_classe_min)
        & (etiquettes["couverture"] >= args.couverture_label_min)
        & (etiquettes["field_id"] > 0)
        & (etiquettes["purete_field"] >= args.purete_parcelle_min)
        & np.all(clartes >= args.pixels_clairs_min, axis=0)
    )

    colonnes_features = noms_colonnes_features()
    for colonne in colonnes_features:
        valide &= np.isfinite(valeurs_features[colonne])

    lignes = []
    for segment_id in np.flatnonzero(valide):
        code = int(etiquettes["classe"][segment_id])
        ligne = {
            "tile_id": tuile,
            "superpixel_id": int(segment_id),
            "field_id": int(etiquettes["field_id"][segment_id]),
            "class_code": code,
            "class_name": CLASSES.get(code, f"Classe {code}"),
            "surface_pixels": int(etiquettes["surface"][segment_id]),
            "purete_classe": round(float(etiquettes["purete_classe"][segment_id]), 6),
            "purete_parcelle": round(float(etiquettes["purete_field"][segment_id]), 6),
            "couverture_label": round(float(etiquettes["couverture"][segment_id]), 6),
            "clair_debut": round(float(clartes[0, segment_id]), 6),
            "clair_milieu": round(float(clartes[1, segment_id]), 6),
            "clair_fin": round(float(clartes[2, segment_id]), 6),
        }
        for colonne in colonnes_features:
            ligne[colonne] = round(float(valeurs_features[colonne][segment_id]), 8)
        lignes.append(ligne)

    return lignes, nombre_segments


def main() -> int:
    args = arguments()
    dataset = args.dataset.resolve()
    sortie = args.sortie.resolve()

    manifeste_path = dataset / "manifest.csv"
    if not manifeste_path.exists():
        print(f"ERREUR : fichier introuvable : {manifeste_path}")
        return 1

    manifeste = pd.read_csv(manifeste_path)
    groupes = list(manifeste.groupby("tile_id", sort=True))
    if args.limite_tuiles > 0:
        groupes = groupes[: args.limite_tuiles]

    sortie.parent.mkdir(parents=True, exist_ok=True)
    colonnes_meta = [
        "tile_id",
        "superpixel_id",
        "field_id",
        "class_code",
        "class_name",
        "surface_pixels",
        "purete_classe",
        "purete_parcelle",
        "couverture_label",
        "clair_debut",
        "clair_milieu",
        "clair_fin",
    ]
    colonnes = colonnes_meta + noms_colonnes_features()

    compteur_classes: Counter = Counter()
    total_slic = 0
    total_retenu = 0
    erreurs = []

    with sortie.open("w", newline="", encoding="utf-8-sig") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=colonnes)
        writer.writeheader()

        for tile_id, groupe in tqdm(groupes, desc="Extraction SLIC", unit="tuile"):
            try:
                lignes, nombre_segments = traiter_tuile(dataset, groupe, args)
                writer.writerows(lignes)
                total_slic += nombre_segments
                total_retenu += len(lignes)
                compteur_classes.update(ligne["class_code"] for ligne in lignes)
            except Exception as exc:
                erreurs.append({"tile_id": identifiant_texte(tile_id), "erreur": str(exc)})
                tqdm.write(f"Tuile {tile_id} ignorée : {exc}")

    resume = {
        "dataset": str(dataset),
        "fichier_sortie": str(sortie),
        "tuiles_demandees": len(groupes),
        "tuiles_en_erreur": len(erreurs),
        "superpixels_slic_total": total_slic,
        "superpixels_retenus": total_retenu,
        "taux_conservation_pct": round(100 * total_retenu / max(total_slic, 1), 2),
        "nombre_caracteristiques_spectrales": len(noms_colonnes_features()),
        "distribution_superpixels": {
            str(code): {
                "classe": CLASSES[code],
                "nombre": compteur_classes.get(code, 0),
            }
            for code in range(1, 10)
        },
        "parametres_slic": {
            "n_segments": args.n_segments,
            "compactness": args.compactness,
            "sigma": args.sigma,
        },
        "seuils": {
            "purete_classe_min": args.purete_classe_min,
            "purete_parcelle_min": args.purete_parcelle_min,
            "couverture_label_min": args.couverture_label_min,
            "pixels_clairs_min": args.pixels_clairs_min,
            "surface_min": args.surface_min,
        },
        "erreurs": erreurs,
    }

    resume_path = sortie.with_name(sortie.stem + "_resume.json")
    resume_path.write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nEXTRACTION TERMINÉE")
    print(f"Tuiles traitées              : {len(groupes) - len(erreurs)}/{len(groupes)}")
    print(f"Superpixels SLIC générés     : {total_slic}")
    print(f"Superpixels retenus          : {total_retenu}")
    print(f"Caractéristiques spectrales  : {len(noms_colonnes_features())}")
    print(f"Fichier CSV                  : {sortie}")
    print(f"Résumé JSON                  : {resume_path}")

    if erreurs:
        print(f"ATTENTION : {len(erreurs)} tuile(s) ont provoqué une erreur.")
        return 1
    if total_retenu == 0:
        print("ERREUR : aucun superpixel n'a satisfait les critères de qualité.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nExtraction interrompue par l'utilisateur.")
        raise SystemExit(130)
