"""Télécharge un sous-ensemble reproductible du jeu South Africa Crop Type.

Le script utilise uniquement l'accès public de Source Cooperative. Il choisit
trois observations Sentinel-2 par tuile (début, milieu et fin de saison), en
privilégiant les dates dont le masque CLM contient le plus de pixels clairs.

Exemples
--------
Test rapide sur deux tuiles :
    python telecharger_dataset_lite.py --nombre-tuiles 2 --racine test_dataset

Constitution du dataset de travail :
    python telecharger_dataset_lite.py --nombre-tuiles 200 --racine dataset_lite
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import boto3
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from rasterio.io import MemoryFile
from tqdm import tqdm


ENDPOINT = "https://data.source.coop"
BUCKET = "radiantearth"
BASE = "south-africa-crops-competition"
PREFIX_S2 = f"{BASE}/train/imagery/s2"
PREFIX_LABELS = f"{BASE}/train/labels"

BANDES = ("B02", "B03", "B04", "B08")

# Trois périodes de la saison agricole. La première date de chaque liste est
# testée en priorité ; les suivantes servent de solutions de remplacement.
PERIODES = {
    "debut_saison": ("2017_05_21", "2017_05_31", "2017_05_11"),
    "milieu_saison": ("2017_07_03", "2017_07_15", "2017_07_28"),
    "fin_saison": ("2017_09_01", "2017_09_18", "2017_10_01"),
}

CLASSES = {
    0: "No Data",
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

SEUIL_CLAIR_MINIMUM = 0.70
SEUIL_CLAIR_EXCELLENT = 0.95
GRAINE_ALEATOIRE = 42


class LimiteVolumeAtteinte(RuntimeError):
    """Signale que le prochain fichier dépasserait la limite autorisée."""


def construire_client_s3():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name="us-east-1",
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 6, "mode": "standard"},
            connect_timeout=30,
            read_timeout=120,
        ),
    )


def lister_tuiles(s3) -> list[str]:
    """Retourne tous les identifiants de tuiles d'entraînement."""
    prefix = f"{PREFIX_S2}/"
    paginator = s3.get_paginator("list_objects_v2")
    tuiles: list[str] = []

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        for element in page.get("CommonPrefixes", []):
            identifiant = element["Prefix"].rstrip("/").split("/")[-1]
            if identifiant.isdigit():
                tuiles.append(identifiant)

    return sorted(tuiles, key=int)


def lire_clm_distant(s3, cle: str) -> tuple[float, bytes]:
    """Lit un petit masque CLM distant et calcule la proportion CLM == 0."""
    reponse = s3.get_object(Bucket=BUCKET, Key=cle)
    contenu = reponse["Body"].read()
    with MemoryFile(contenu) as memfile:
        with memfile.open() as src:
            masque = src.read(1)
    return float(np.mean(masque == 0)), contenu


def choisir_date(s3, tuile: str, dates: tuple[str, ...]):
    """Choisit la date la plus claire parmi les dates proposées."""
    meilleur = None

    for date in dates:
        cle = f"{PREFIX_S2}/{tuile}/{date}/{tuile}_{date}_CLM_10m.tif"
        try:
            ratio, contenu = lire_clm_distant(s3, cle)
        except (ClientError, BotoCoreError, rasterio.errors.RasterioError):
            continue

        if meilleur is None or ratio > meilleur[1]:
            meilleur = (date, ratio, cle, contenu)

        if ratio >= SEUIL_CLAIR_EXCELLENT:
            break

    if meilleur is None or meilleur[1] < SEUIL_CLAIR_MINIMUM:
        return None
    return meilleur


def taille_repertoire(racine: Path) -> int:
    return sum(p.stat().st_size for p in racine.rglob("*") if p.is_file())


def enregistrer_octets(
    contenu: bytes,
    destination: Path,
    total_octets: int,
    limite_octets: int,
) -> int:
    """Enregistre des octets en respectant la limite globale."""
    if destination.exists() and destination.stat().st_size > 0:
        return total_octets
    if total_octets + len(contenu) > limite_octets:
        raise LimiteVolumeAtteinte

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporaire = destination.with_suffix(destination.suffix + ".part")
    temporaire.write_bytes(contenu)
    temporaire.replace(destination)
    return total_octets + len(contenu)


def telecharger_objet(
    s3,
    cle: str,
    destination: Path,
    total_octets: int,
    limite_octets: int,
) -> int:
    """Télécharge un objet public avec reprise simple et écriture atomique."""
    if destination.exists() and destination.stat().st_size > 0:
        return total_octets

    reponse = s3.get_object(Bucket=BUCKET, Key=cle)
    taille = int(reponse.get("ContentLength", 0))
    if total_octets + taille > limite_octets:
        reponse["Body"].close()
        raise LimiteVolumeAtteinte

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporaire = destination.with_suffix(destination.suffix + ".part")
    with temporaire.open("wb") as fichier:
        while True:
            bloc = reponse["Body"].read(1024 * 1024)
            if not bloc:
                break
            fichier.write(bloc)
    temporaire.replace(destination)
    return total_octets + destination.stat().st_size


def lire_classes_masque(chemin: Path) -> tuple[list[int], Counter]:
    with rasterio.open(chemin) as src:
        masque = src.read(1)
    valeurs, effectifs = np.unique(masque, return_counts=True)
    compteur = Counter(
        {int(code): int(n) for code, n in zip(valeurs, effectifs) if int(code) != 0}
    )
    return sorted(compteur), compteur


def ecrire_csv(chemin: Path, lignes: list[dict], colonnes: list[str]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", newline="", encoding="utf-8-sig") as fichier:
        writer = csv.DictWriter(fichier, fieldnames=colonnes)
        writer.writeheader()
        writer.writerows(lignes)


def analyser_arguments():
    parser = argparse.ArgumentParser(
        description="Télécharger un dataset Sentinel-2 allégé et reproductible."
    )
    parser.add_argument(
        "--nombre-tuiles",
        type=int,
        default=200,
        help="Nombre de tuiles complètes à conserver (défaut : 200).",
    )
    parser.add_argument(
        "--racine",
        type=Path,
        default=Path("dataset_lite"),
        help="Dossier de destination (défaut : dataset_lite).",
    )
    parser.add_argument(
        "--tuiles", nargs="+", default=None,
        help="IDs précis à télécharger, ex. --tuiles 276 (sans les 200 autres).",
    )
    parser.add_argument(
        "--max-go",
        type=float,
        default=0.8,
        help="Volume maximal du dossier en Gio (défaut : 0.8).",
    )
    return parser.parse_args()


def main() -> int:
    args = analyser_arguments()
    if args.nombre_tuiles < 1:
        print("ERREUR : --nombre-tuiles doit être supérieur à zéro.")
        return 2
    if args.max_go <= 0:
        print("ERREUR : --max-go doit être supérieur à zéro.")
        return 2

    racine = args.racine.resolve()
    racine.mkdir(parents=True, exist_ok=True)
    limite_octets = int(args.max_go * 1024**3)
    total_octets = taille_repertoire(racine)

    print("Connexion au dépôt public Source Cooperative...")
    s3 = construire_client_s3()
    if args.tuiles:
        if any(not identifiant.isdecimal() for identifiant in args.tuiles):
            print("ERREUR : --tuiles doit contenir des identifiants numériques.")
            return 2
        tuiles = sorted(set(args.tuiles), key=int)
        args.nombre_tuiles = len(tuiles)
        print("Tuiles explicitement demandées : " + ", ".join(tuiles))
        manifeste_existant = racine / "manifest.csv"
        if manifeste_existant.exists():
            import pandas as pd
            precedentes = set(pd.read_csv(manifeste_existant)["tile_id"].astype(str))
            autres = precedentes - set(tuiles)
            if autres:
                print("ERREUR : dataset_lite possède déjà d'autres tuiles ; "
                      "utilisez un autre --racine pour ne pas écraser son manifeste.")
                return 2
    else:
        try:
            tuiles = lister_tuiles(s3)
        except (ClientError, BotoCoreError) as exc:
            print(f"ERREUR DE CONNEXION : {exc}")
            return 1

        if not tuiles:
            print("ERREUR : aucune tuile n'a été trouvée.")
            return 1

        print(f"{len(tuiles)} tuiles d'entraînement disponibles.")
        random.Random(GRAINE_ALEATOIRE).shuffle(tuiles)

    manifeste: list[dict] = []
    pixels_par_classe: Counter = Counter()
    tuiles_par_classe: defaultdict[int, set[str]] = defaultdict(set)
    tuiles_acceptees = 0
    tuiles_ignorees = 0
    volume_bloque = False

    barre = tqdm(total=args.nombre_tuiles, desc="Tuiles complètes", unit="tuile")

    for tuile in tuiles:
        if tuiles_acceptees >= args.nombre_tuiles:
            break

        try:
            choix_dates = {}
            for periode, dates in PERIODES.items():
                choix = choisir_date(s3, tuile, dates)
                if choix is None:
                    choix_dates = {}
                    break
                choix_dates[periode] = choix

            if len(choix_dates) != len(PERIODES):
                tuiles_ignorees += 1
                continue

            dossier_masques = racine / "masques"
            masque_classes = dossier_masques / f"{tuile}.tif"
            masque_champs = dossier_masques / f"{tuile}_field_ids.tif"

            total_octets = telecharger_objet(
                s3,
                f"{PREFIX_LABELS}/{tuile}.tif",
                masque_classes,
                total_octets,
                limite_octets,
            )
            total_octets = telecharger_objet(
                s3,
                f"{PREFIX_LABELS}/{tuile}_field_ids.tif",
                masque_champs,
                total_octets,
                limite_octets,
            )

            classes_presentes, compteurs = lire_classes_masque(masque_classes)
            if not classes_presentes:
                tuiles_ignorees += 1
                continue

            lignes_tuile = []
            for periode, (date, ratio, cle_clm, contenu_clm) in choix_dates.items():
                dossier_date = racine / f"tuile_{tuile}" / date

                destination_clm = dossier_date / f"{tuile}_{date}_CLM_10m.tif"
                total_octets = enregistrer_octets(
                    contenu_clm, destination_clm, total_octets, limite_octets
                )

                for bande in BANDES:
                    nom = f"{tuile}_{date}_{bande}_10m.tif"
                    cle = f"{PREFIX_S2}/{tuile}/{date}/{nom}"
                    total_octets = telecharger_objet(
                        s3,
                        cle,
                        dossier_date / nom,
                        total_octets,
                        limite_octets,
                    )

                lignes_tuile.append(
                    {
                        "tile_id": tuile,
                        "periode": periode,
                        "date": date,
                        "pixels_clairs_pct": f"{100 * ratio:.2f}",
                        "classes_codes": ";".join(map(str, classes_presentes)),
                        "classes_noms": ";".join(
                            CLASSES.get(code, f"Classe {code}")
                            for code in classes_presentes
                        ),
                        "dossier": str(dossier_date.relative_to(racine)),
                    }
                )

            manifeste.extend(lignes_tuile)
            pixels_par_classe.update(compteurs)
            for code in classes_presentes:
                tuiles_par_classe[code].add(tuile)

            tuiles_acceptees += 1
            barre.update(1)
            barre.set_postfix(
                volume_Mo=f"{total_octets / 1024**2:.1f}", ignorees=tuiles_ignorees
            )

        except LimiteVolumeAtteinte:
            volume_bloque = True
            print("\nLa limite de volume a été atteinte proprement.")
            break
        except (ClientError, BotoCoreError, OSError, rasterio.errors.RasterioError) as exc:
            tuiles_ignorees += 1
            tqdm.write(f"Tuile {tuile} ignorée après une erreur : {exc}")

    barre.close()

    colonnes_manifeste = [
        "tile_id",
        "periode",
        "date",
        "pixels_clairs_pct",
        "classes_codes",
        "classes_noms",
        "dossier",
    ]
    ecrire_csv(racine / "manifest.csv", manifeste, colonnes_manifeste)

    resume_classes = []
    for code in range(1, 10):
        resume_classes.append(
            {
                "code": code,
                "classe": CLASSES[code],
                "nombre_pixels": pixels_par_classe.get(code, 0),
                "nombre_tuiles": len(tuiles_par_classe.get(code, set())),
            }
        )
    ecrire_csv(
        racine / "resume_classes.csv",
        resume_classes,
        ["code", "classe", "nombre_pixels", "nombre_tuiles"],
    )

    resume = {
        "tuiles_demandees": args.nombre_tuiles,
        "tuiles_completes": tuiles_acceptees,
        "tuiles_ignorees": tuiles_ignorees,
        "dates_par_tuile": len(PERIODES),
        "bandes": list(BANDES),
        "volume_octets": total_octets,
        "volume_mo": round(total_octets / 1024**2, 2),
        "limite_atteinte": volume_bloque,
        "graine_aleatoire": GRAINE_ALEATOIRE,
    }
    (racine / "resume_telechargement.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nTÉLÉCHARGEMENT TERMINÉ")
    print(f"Tuiles complètes : {tuiles_acceptees}/{args.nombre_tuiles}")
    print(f"Tuiles ignorées  : {tuiles_ignorees}")
    print(f"Volume obtenu    : {total_octets / 1024**2:.2f} Mo")
    print(f"Dossier          : {racine}")
    print("Rapports         : manifest.csv et resume_classes.csv")

    classes_absentes = [
        f"{code} ({CLASSES[code]})"
        for code in range(1, 10)
        if pixels_par_classe.get(code, 0) == 0
    ]
    if classes_absentes:
        print("Classes absentes  : " + ", ".join(classes_absentes))
        print("Nous ajouterons ensuite des tuiles ciblées pour ces classes.")
    else:
        print("Couverture        : les 9 classes agricoles sont représentées.")

    if tuiles_acceptees < args.nombre_tuiles and not volume_bloque:
        print("ATTENTION : toutes les tuiles demandées n'ont pas pu être constituées.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nTéléchargement interrompu. Relancez la même commande pour reprendre.")
        raise SystemExit(130)
