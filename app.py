"""Application Streamlit de la variante A : SLIC + caractéristiques + K-means.

Lancement sous Windows (depuis la racine du projet) :
    python -m streamlit run app.py

L'application ne réentraîne pas K-means. Elle charge les modèles K=4 et K=9
produits par ``entrainer_kmeans.py`` et recalcule les groupes d'une tuile à la
demande. Les classes réelles sont affichées seulement pour l'évaluation.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    import rasterio
    from skimage.segmentation import mark_boundaries, slic
except ImportError as exc:  # Message plus clair qu'une longue trace Python.
    st.error(
        "Une bibliothèque d'imagerie manque. Activez `.venv`, puis exécutez "
        "`python -m pip install -r requirements.txt`."
    )
    st.exception(exc)
    st.stop()


st.set_page_config(
    page_title="AgriSLIC — Classification agricole",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


RACINE = Path(__file__).resolve().parent
DATASET_COMPLET = RACINE / "dataset_lite"
DATASET_EXEMPLE = RACINE / "test_dataset"
# Le ZIP partageable ne contient pas les 200 tuiles brutes. En leur absence,
# les deux tuiles d'exemple suffisent pour demonstrer SLIC, mais elles ne
# doivent pas etre confondues avec les images du protocole d'evaluation.
MODE_EXEMPLE = not (DATASET_COMPLET / "manifest.csv").is_file()
DATASET = DATASET_EXEMPLE if MODE_EXEMPLE else DATASET_COMPLET
RESULTATS = RACINE / "resultats"
MODELES_KMEANS = {
    4: RESULTATS / "kmeans_k4",
    9: RESULTATS / "kmeans_k9",
}

BANDES = ("B02", "B03", "B04", "B08")
PERIODES = ("debut_saison", "milieu_saison", "fin_saison")
PERIODES_FR = {
    "debut_saison": "Début de saison",
    "milieu_saison": "Milieu de saison",
    "fin_saison": "Fin de saison",
}

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

COULEURS_CLASSES = [
    "#111827",  # 0 : zone non affichée
    "#5DA5DA",
    "#60BD68",
    "#F17CB0",
    "#B2912F",
    "#B276B2",
    "#F15854",
    "#4D4D4D",
    "#FAA43A",
    "#2CA02C",
]
CMAP_CLASSES = ListedColormap(COULEURS_CLASSES)
CMAP_VALIDITE = ListedColormap(["#111827", "#DC2626", "#16A34A"])
COULEURS_GROUPES = [
    "#2563EB",
    "#16A34A",
    "#DC2626",
    "#9333EA",
    "#EA580C",
    "#0891B2",
    "#CA8A04",
    "#DB2777",
    "#4B5563",
    "#65A30D",
    "#7C3AED",
    "#0F766E",
]

META_COLONNES = {
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
}


CSS = """
<style>
    :root {
        --fond: #f3f3f0;
        --papier: #ffffff;
        --texte: #1f2924;
        --secondaire: #667069;
        --vert: #1f6b4b;
        --vert-clair: #e7f0eb;
        --ligne: #d7dcd8;
        --ocre: #b9852b;
    }
    .stApp { background: var(--fond); color: var(--texte); }
    [data-testid="stHeader"] { background: rgba(243, 243, 240, .92); }
    .block-container {
        max-width: 1320px;
        padding-top: 1.35rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: #fafaf8;
        border-right: 1px solid var(--ligne);
    }
    [data-testid="stSidebar"] * { color: var(--texte); }
    [data-testid="stSidebar"] .stRadio label {
        padding: .38rem .15rem;
        font-weight: 600;
    }
    [data-testid="stSidebar"] hr { border-color: var(--ligne); }
    .sidebar-brand {
        padding: .35rem 0 1rem 0;
        border-bottom: 2px solid var(--vert);
        margin-bottom: 1.1rem;
    }
    .sidebar-brand strong {
        display: block;
        color: var(--vert);
        font-size: 1.45rem;
        letter-spacing: -.02em;
    }
    .sidebar-brand span {
        display: block;
        color: var(--secondaire);
        font-size: .83rem;
        margin-top: .15rem;
    }
    .sidebar-status {
        padding: .7rem .8rem;
        background: var(--vert-clair);
        border-left: 3px solid var(--vert);
        font-size: .86rem;
        margin: .9rem 0 1rem 0;
    }
    .page-head {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 0 0 .85rem 0;
        border-bottom: 2px solid var(--vert);
        margin-bottom: 1rem;
    }
    .page-head h1 {
        margin: .05rem 0 .15rem 0;
        color: var(--texte);
        font-size: 1.85rem;
        letter-spacing: -.025em;
    }
    .page-head p {
        margin: 0;
        color: var(--secondaire);
        font-size: .98rem;
    }
    .eyebrow {
        color: var(--vert);
        font-size: .73rem;
        font-weight: 800;
        letter-spacing: .11em;
        text-transform: uppercase;
    }
    .method-tag {
        flex: 0 0 auto;
        color: var(--vert);
        border: 1px solid #a9bdb1;
        background: var(--papier);
        padding: .38rem .65rem;
        font-size: .78rem;
        font-weight: 700;
    }
    .question-panel {
        background: var(--papier);
        border: 1px solid var(--ligne);
        border-left: 4px solid var(--vert);
        padding: 1rem 1.1rem;
        margin-bottom: .9rem;
    }
    .question-panel h3 { margin: 0 0 .45rem 0; font-size: 1.08rem; }
    .question-panel p { margin: 0; color: #3f4943; line-height: 1.55; }
    .process-list {
        background: var(--papier);
        border: 1px solid var(--ligne);
        margin-top: .45rem;
    }
    .process-row {
        display: grid;
        grid-template-columns: 2.2rem 8.2rem 1fr;
        gap: .65rem;
        align-items: center;
        padding: .58rem .75rem;
        border-bottom: 1px solid #e6e9e6;
    }
    .process-row:last-child { border-bottom: 0; }
    .process-number { color: var(--vert); font-weight: 800; }
    .process-title { font-weight: 700; }
    .process-text { color: var(--secondaire); font-size: .86rem; }
    .figure-caption {
        color: var(--secondaire);
        font-size: .8rem;
        line-height: 1.4;
        margin-top: .3rem;
    }
    .note {
        padding: .75rem .9rem;
        border-left: 4px solid var(--ocre);
        background: #fffaf0;
        border-radius: 0;
        font-size: .92rem;
    }
    div[data-testid="stMetric"] {
        background: var(--papier);
        border: 1px solid var(--ligne);
        border-top: 3px solid var(--vert);
        border-radius: 3px;
        padding: .62rem .75rem;
        box-shadow: none;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--secondaire);
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--texte);
        font-size: 1.55rem;
    }
    [data-testid="stForm"] {
        background: var(--papier);
        border: 1px solid var(--ligne);
        border-radius: 3px;
        padding: .8rem;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 3px;
        font-weight: 700;
    }
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: var(--vert);
        border-color: var(--vert);
    }
    [data-testid="stDataFrame"], [data-testid="stImage"] {
        background: var(--papier);
    }
    h2, h3 { color: var(--texte); letter-spacing: -.015em; }
    .footer {
        color: var(--secondaire);
        font-size: .78rem;
        margin-top: 2rem;
        padding-top: .7rem;
        border-top: 1px solid var(--ligne);
    }
    @media (max-width: 800px) {
        .page-head { align-items: flex-start; }
        .method-tag { display: none; }
        .process-row { grid-template-columns: 2rem 1fr; }
        .process-text { grid-column: 2; }
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


class DonneesApplicationError(RuntimeError):
    """Erreur compréhensible liée aux fichiers de démonstration."""


def id_texte(valeur) -> str:
    texte = str(valeur)
    return texte[:-2] if texte.endswith(".0") else texte


def tri_identifiants(valeurs) -> list[str]:
    def cle(valeur: str):
        return (0, int(valeur)) if valeur.isdigit() else (1, valeur)

    return sorted({id_texte(v) for v in valeurs}, key=cle)


def exiger_fichiers(fichiers: list[Path]) -> None:
    absents = [str(p.relative_to(RACINE)) for p in fichiers if not p.exists()]
    if absents:
        raise DonneesApplicationError(
            "Fichiers nécessaires absents : " + ", ".join(absents)
        )


@st.cache_data(show_spinner=False)
def lire_csv(chemin: str) -> pd.DataFrame:
    path = Path(chemin)
    if not path.exists():
        raise DonneesApplicationError(f"Fichier CSV introuvable : {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DonneesApplicationError(f"CSV invalide : {path.name} ({exc})") from exc


@st.cache_data(show_spinner=False)
def lire_json(chemin: str) -> dict:
    path = Path(chemin)
    if not path.exists():
        raise DonneesApplicationError(f"Fichier JSON introuvable : {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DonneesApplicationError(f"JSON invalide : {path.name} ({exc})") from exc


@st.cache_data(show_spinner=False)
def lire_raster(chemin: str) -> np.ndarray:
    path = Path(chemin)
    if not path.exists():
        raise DonneesApplicationError(f"GeoTIFF introuvable : {path}")
    try:
        with rasterio.open(path) as src:
            tableau = src.read(1)
    except Exception as exc:
        raise DonneesApplicationError(
            f"Impossible de lire le GeoTIFF {path.name} : {exc}"
        ) from exc
    if tableau.ndim != 2 or tableau.size == 0:
        raise DonneesApplicationError(f"Dimensions invalides dans {path.name}")
    return tableau


@st.cache_resource(show_spinner=False)
def charger_modele(k: int):
    chemin = MODELES_KMEANS[int(k)] / "modele_kmeans.joblib"
    if not chemin.exists():
        raise DonneesApplicationError(f"Modèle introuvable : {chemin.name}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            return joblib.load(chemin)
    except Exception as exc:
        raise DonneesApplicationError(
            f"Le modèle K={k} ne peut pas être chargé. Vérifiez la version de "
            f"scikit-learn : {exc}"
        ) from exc


def convertir_reflectance(tableau: np.ndarray) -> np.ndarray:
    resultat = tableau.astype(np.float32)
    valides = np.isfinite(resultat)
    maximum = float(np.nanmax(resultat)) if np.any(valides) else 0.0
    if maximum > 1.0:
        resultat /= 255.0 if maximum <= 255.0 else 10000.0
    resultat[~valides] = np.nan
    return np.clip(resultat, 0.0, 1.0)


@st.cache_data(show_spinner=False)
def charger_observations(tuile: str):
    manifeste = lire_csv(str(DATASET / "manifest.csv")).copy()
    manifeste["_tile"] = manifeste["tile_id"].map(id_texte)
    lignes = manifeste[manifeste["_tile"] == id_texte(tuile)].set_index("periode")
    if any(p not in lignes.index for p in PERIODES):
        raise DonneesApplicationError(
            f"La tuile {tuile} ne possède pas les trois périodes attendues."
        )

    cubes, clairs, dates = [], [], []
    shape_reference = None
    for periode in PERIODES:
        ligne = lignes.loc[periode]
        # Le manifeste a été créé sous Windows et contient donc parfois des
        # antislashs. La conversion rend l'application portable Windows/Linux.
        dossier_relatif = str(ligne["dossier"]).replace("\\", "/")
        dossier = DATASET / dossier_relatif
        date = id_texte(ligne["date"])
        couches = []
        for bande in BANDES:
            chemin = dossier / f"{tuile}_{date}_{bande}_10m.tif"
            couches.append(convertir_reflectance(lire_raster(str(chemin))))
        cube = np.stack(couches, axis=-1)
        clm = lire_raster(str(dossier / f"{tuile}_{date}_CLM_10m.tif"))
        clair = clm == 0
        clair &= np.all(np.isfinite(cube), axis=-1)

        if shape_reference is None:
            shape_reference = cube.shape
        elif cube.shape != shape_reference:
            raise DonneesApplicationError(
                f"Dimensions temporelles incompatibles pour la tuile {tuile}."
            )
        cubes.append(cube)
        clairs.append(clair)
        dates.append(date)
    return cubes, clairs, dates


def normaliser_rgb(cube: np.ndarray, masque: np.ndarray) -> np.ndarray:
    rgb = cube[..., [2, 1, 0]].astype(np.float32).copy()
    for canal in range(3):
        bande = rgb[..., canal]
        valeurs = bande[masque & np.isfinite(bande)]
        if valeurs.size:
            bas, haut = np.percentile(valeurs, [2, 98])
        else:
            bas, haut = 0.0, 1.0
        if haut <= bas:
            haut = bas + 1e-6
        rgb[..., canal] = np.clip((bande - bas) / (haut - bas), 0, 1)
    rgb[~masque] = 0
    return rgb


def composite_pour_slic(cubes: list[np.ndarray], clairs: list[np.ndarray]) -> np.ndarray:
    pile = np.stack(cubes, axis=0).astype(np.float32)
    validite = np.stack(clairs, axis=0)
    pile = np.where(validite[..., None], pile, np.nan)
    with np.errstate(invalid="ignore"):
        composite = np.nanmedian(pile, axis=0)

    for indice in range(composite.shape[-1]):
        bande = composite[..., indice]
        valides = np.isfinite(bande)
        remplissage = float(np.nanmedian(bande)) if np.any(valides) else 0.0
        bande[~valides] = remplissage
        valeurs = bande[valides]
        bas, haut = np.percentile(valeurs, [2, 98]) if valeurs.size else (0.0, 1.0)
        if haut <= bas:
            haut = bas + 1e-6
        composite[..., indice] = np.clip((bande - bas) / (haut - bas), 0, 1)
    return composite


@st.cache_data(show_spinner=False)
def segmenter_tuile(
    tuile: str, n_segments: int, compactness: float = 10.0, sigma: float = 1.0
):
    cubes, clairs, dates = charger_observations(tuile)
    composite = composite_pour_slic(cubes, clairs)
    segments = slic(
        composite,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        channel_axis=-1,
        convert2lab=False,
        enforce_connectivity=True,
    ).astype(np.int32)
    return segments, cubes, clairs, dates


def calculer_ndvi(cube: np.ndarray) -> np.ndarray:
    rouge = cube[..., 2]
    pir = cube[..., 3]
    return np.clip((pir - rouge) / (pir + rouge + 1e-6), -1, 1)


def mode_stable(valeurs) -> int:
    compte = Counter(int(v) for v in valeurs)
    maximum = max(compte.values())
    return min(code for code, n in compte.items() if n == maximum)


def entete(titre: str, sous_titre: str) -> None:
    st.markdown(
        '<div class="page-head"><div>'
        '<div class="eyebrow">Projet de classification agricole</div>'
        f'<h1>{titre}</h1><p>{sous_titre}</p></div>'
        '<div class="method-tag">SENTINEL-2 · SLIC · K-MEANS</div></div>',
        unsafe_allow_html=True,
    )


def legende_classes(figure) -> None:
    figure.legend(
        handles=[
            Patch(facecolor=COULEURS_CLASSES[code], label=f"{code} — {CLASSES[code]}")
            for code in range(1, 10)
        ],
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.03),
    )


def diagnostic_projet() -> tuple[list[Path], list[Path]]:
    requis = [
        RESULTATS / "caracteristiques_superpixels.csv",
        RESULTATS / "caracteristiques_superpixels_resume.json",
        MODELES_KMEANS[4] / "rapport_kmeans.json",
        MODELES_KMEANS[4] / "modele_kmeans.joblib",
        MODELES_KMEANS[9] / "rapport_kmeans.json",
        MODELES_KMEANS[9] / "modele_kmeans.joblib",
    ]
    absents = [p for p in requis if not p.exists()]
    return requis, absents


def page_accueil() -> None:
    entete(
        "AgriSLIC — regroupement non supervisé",
        "Découvrir des groupes de parcelles similaires sans fournir les cultures à K-means.",
    )

    resume_slic = lire_json(str(RESULTATS / "caracteristiques_superpixels_resume.json"))
    rapport = lire_json(str(MODELES_KMEANS[9] / "rapport_kmeans.json"))
    caracteristiques = lire_csv(str(RESULTATS / "caracteristiques_superpixels.csv"))

    cols = st.columns(5)
    cols[0].metric("Tuiles analysées", f"{resume_slic['tuiles_demandees']}")
    cols[1].metric("Périodes", "3")
    cols[2].metric("Superpixels retenus", f"{resume_slic['superpixels_retenus']:,}".replace(",", " "))
    cols[3].metric(
        "Caractéristiques utilisées",
        f"{rapport['nombre_caracteristiques_utilisees']}/64",
    )
    cols[4].metric("Groupes étudiés", "K=4 et K=9")

    gauche, droite = st.columns([.9, 1.1], gap="large")
    titres = [
        ("1", "Données", "Bandes B02, B03, B04, B08 et masque nuageux CLM."),
        ("2", "Prétraitement", "Contrôle des dimensions, pixels invalides et normalisation."),
        ("3", "SLIC", "Regroupement des pixels voisins en régions homogènes."),
        ("4", "Caractéristiques", "24 moyennes spectrales et de végétation sur 3 dates."),
        ("5", "K-means", "Standardisation puis regroupement sans utiliser les étiquettes."),
    ]
    with gauche:
        st.markdown(
            '<div class="question-panel"><h3>Question étudiée</h3><p>'
            "Peut-on découvrir des groupes de parcelles possédant des signatures "
            "spectrales similaires, sans donner à K-means les noms des cultures ?"
            "</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Chaîne de traitement")
        lignes = "".join(
            f'<div class="process-row"><span class="process-number">{numero}</span>'
            f'<span class="process-title">{titre}</span>'
            f'<span class="process-text">{texte}</span></div>'
            for numero, titre, texte in titres
        )
        st.markdown(f'<div class="process-list">{lignes}</div>', unsafe_allow_html=True)
    with droite:
        illustration = MODELES_KMEANS[9] / "groupes_espace_spectral.png"
        st.image(
            str(illustration),
            caption="Groupes K-means observés dans deux caractéristiques spectrales.",
            width="stretch",
        )

    st.subheader("Jeu de données et couverture des classes")
    classes = (
        caracteristiques.groupby(["class_code", "class_name"])
        .agg(superpixels=("superpixel_id", "size"), parcelles=("field_id", "nunique"))
        .reset_index()
    )
    gauche, droite = st.columns([1.15, 1], gap="large")
    with gauche:
        st.markdown(
            "Le sous-ensemble provient du **South Africa Crop Type Competition**. "
            "Les noms des cultures sont conservés uniquement pour vérifier après coup "
            "si les groupes découverts correspondent aux cultures réelles."
        )
        st.dataframe(
            classes.rename(
                columns={
                    "class_code": "Code",
                    "class_name": "Classe",
                    "superpixels": "Superpixels",
                    "parcelles": "Parcelles",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with droite:
        graphique = classes.set_index("class_name")["superpixels"]
        st.bar_chart(graphique, color="#16845c", horizontal=True)

    st.markdown(
        '<div class="note"><b>Point à défendre :</b> K-means ne reçoit pas les '
        "noms des cultures pendant son apprentissage. Toutefois, les annotations "
        "ont été utilisées au préalable pour filtrer le corpus et organiser "
        "la séparation des données, puis après apprentissage pour interpréter "
        "et évaluer les groupes.</div>",
        unsafe_allow_html=True,
    )


def page_segmentation() -> None:
    entete(
        "Segmentation SLIC",
        "Observer l'effet du nombre de superpixels et de la compacité sur une image multispectrale.",
    )

    if MODE_EXEMPLE:
        st.info(
            "Mode démonstration : deux tuiles d'exemple incluses dans le ZIP "
            "servent à illustrer la segmentation SLIC. Les métriques K-means "
            "restent celles de l'expérience distincte sur 200 tuiles."
        )
    manifeste = lire_csv(str(DATASET / "manifest.csv"))
    tuiles = tri_identifiants(manifeste["tile_id"])
    index_pilote = tuiles.index("1") if "1" in tuiles else 0

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    tuile = c1.selectbox("Tuile Sentinel-2", tuiles, index=index_pilote)
    n_segments = c2.selectbox("Superpixels demandés", [500, 1000, 2000], index=1)
    compactness = c3.slider("Compacité", 1.0, 30.0, 10.0, 1.0)
    periode = c4.selectbox(
        "Image affichée", PERIODES, index=1, format_func=lambda x: PERIODES_FR[x]
    )

    with st.spinner("Segmentation SLIC en cours…"):
        segments, cubes, clairs, dates = segmenter_tuile(
            tuile, n_segments, compactness, 1.0
        )
    indice_periode = PERIODES.index(periode)
    rgb = normaliser_rgb(cubes[indice_periode], clairs[indice_periode])
    image_slic = mark_boundaries(rgb, segments, color=(1, 0, 0))
    nombre_reel = int(np.unique(segments).size)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Segments demandés", n_segments)
    m2.metric("Segments réellement produits", nombre_reel)
    m3.metric("Surface moyenne", f"{segments.size / nombre_reel:.1f} pixels")
    m4.metric("Pixels clairs", f"{100 * clairs[indice_periode].mean():.2f} %")

    gauche, droite = st.columns(2)
    figure_originale, axe = plt.subplots(figsize=(6, 6))
    axe.imshow(rgb)
    axe.set_title(f"Image originale — {dates[indice_periode].replace('_', '-')}")
    axe.axis("off")
    gauche.pyplot(figure_originale, width="stretch")
    plt.close(figure_originale)

    figure_slic, axe = plt.subplots(figsize=(6, 6))
    axe.imshow(image_slic)
    axe.set_title(f"SLIC — N={n_segments}, compacité={compactness:g}")
    axe.axis("off")
    droite.pyplot(figure_slic, width="stretch")
    plt.close(figure_slic)

    with st.expander("Comparer automatiquement 500, 1 000 et 2 000 superpixels"):
        if st.button("Produire la chaîne comparative", type="primary"):
            with st.spinner("Calcul des trois segmentations…"):
                figure, axes = plt.subplots(1, 4, figsize=(18, 5))
                axes[0].imshow(rgb)
                axes[0].set_title("Image originale")
                axes[0].axis("off")
                for axe, n in zip(axes[1:], (500, 1000, 2000)):
                    seg, _, _, _ = segmenter_tuile(tuile, n, compactness, 1.0)
                    axe.imshow(mark_boundaries(rgb, seg, color=(1, 0, 0)))
                    axe.set_title(f"N={n}\n{np.unique(seg).size} régions obtenues")
                    axe.axis("off")
                figure.suptitle(
                    f"Comparaison SLIC — tuile {tuile}", fontsize=15, fontweight="bold"
                )
                figure.tight_layout()
                st.pyplot(figure, width="stretch")
                plt.close(figure)

    st.markdown(
        '<div class="note"><b>Important :</b> SLIC cherche un nombre approximatif de '
        "régions. La compacité règle le compromis entre similarité spectrale et proximité "
        "spatiale. Le modèle final du prototype a été entraîné avec <b>N=200</b> ; les "
        "valeurs 500/1 000/2 000 sont ici comparées conformément à la tâche du groupe 2, "
        "sans modifier le modèle déjà entraîné.</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def executer_regroupement(tuile: str, k: int) -> dict:
    caracteristiques_path = RESULTATS / "caracteristiques_superpixels.csv"
    dossier_modele = MODELES_KMEANS[int(k)]
    predictions_path = dossier_modele / "predictions_superpixels_test.csv"
    rapport_path = dossier_modele / "rapport_kmeans.json"
    resume_slic_path = RESULTATS / "caracteristiques_superpixels_resume.json"
    exiger_fichiers(
        [caracteristiques_path, predictions_path, rapport_path, resume_slic_path]
    )

    caracteristiques = lire_csv(str(caracteristiques_path)).copy()
    predictions_test = lire_csv(str(predictions_path)).copy()
    rapport = lire_json(str(rapport_path))
    resume_slic = lire_json(str(resume_slic_path))

    caracteristiques["_tile"] = caracteristiques["tile_id"].map(id_texte)
    predictions_test["_tile"] = predictions_test["tile_id"].map(id_texte)
    test_tuile = predictions_test[predictions_test["_tile"] == id_texte(tuile)]
    donnees = caracteristiques[
        caracteristiques["_tile"].eq(id_texte(tuile))
        & caracteristiques["superpixel_id"].isin(test_tuile["superpixel_id"])
    ].copy()
    donnees = donnees.sort_values("superpixel_id")
    if donnees.empty:
        raise DonneesApplicationError(
            f"Aucun superpixel de test disponible pour la tuile {tuile}."
        )

    artefact = charger_modele(int(k))
    colonnes_features = artefact["colonnes_features"]
    manquantes = [c for c in colonnes_features if c not in donnees.columns]
    if manquantes:
        raise DonneesApplicationError(
            "Caractéristiques manquantes dans le CSV : " + ", ".join(manquantes[:5])
        )
    X = donnees[colonnes_features].astype(np.float64)
    if X.isna().any().any():
        raise DonneesApplicationError("Les caractéristiques contiennent des valeurs NaN.")

    X_standardise = artefact["scaler"].transform(X)
    groupes_predits = artefact["kmeans"].predict(X_standardise).astype(int)
    correspondance = {
        int(groupe): int(classe)
        for groupe, classe in artefact["correspondance_groupes_classes"].items()
    }
    y_pred = np.array([correspondance[int(g)] for g in groupes_predits], dtype=int)
    donnees["groupe_kmeans"] = groupes_predits
    donnees["prediction_live"] = y_pred

    parametres = resume_slic["parametres_slic"]
    cartes = None
    # Ne dessiner la carte que si les images de la tuile de test sont
    # réellement présentes. Le mini-dataset joint ne couvre pas le test.
    tuiles_disponibles = set()
    if (DATASET / "manifest.csv").is_file():
        manifeste_images = lire_csv(str(DATASET / "manifest.csv"))
        tuiles_disponibles = set(manifeste_images["tile_id"].map(id_texte))
    if id_texte(tuile) in tuiles_disponibles:
        segments, cubes, clairs, dates = segmenter_tuile(
            id_texte(tuile),
            int(parametres["n_segments"]),
            float(parametres["compactness"]),
            float(parametres["sigma"]),
        )
        carte_reelle = np.zeros(segments.shape, dtype=np.int16)
        carte_groupes = np.zeros(segments.shape, dtype=np.int16)
        carte_predite = np.zeros(segments.shape, dtype=np.int16)
        carte_validite = np.zeros(segments.shape, dtype=np.int16)
        for ligne in donnees.itertuples(index=False):
            masque = segments == int(ligne.superpixel_id)
            vrai = int(ligne.class_code)
            predit = int(ligne.prediction_live)
            carte_reelle[masque] = vrai
            carte_groupes[masque] = int(ligne.groupe_kmeans) + 1
            carte_predite[masque] = predit
            carte_validite[masque] = 2 if vrai == predit else 1
        milieu = PERIODES.index("milieu_saison")
        cartes = {
            "segments": segments,
            "rgb": normaliser_rgb(cubes[milieu], clairs[milieu]),
            "ndvi": calculer_ndvi(cubes[milieu]),
            "carte_reelle": carte_reelle,
            "carte_groupes": carte_groupes,
            "carte_predite": carte_predite,
            "carte_validite": carte_validite,
            "date": dates[milieu],
        }

    exactitude_sp = float(
        np.mean(donnees["class_code"].astype(int).to_numpy() == y_pred)
    )
    votes = []
    for _, parcelle in donnees.groupby("field_id"):
        votes.append(
            mode_stable(parcelle["class_code"])
            == mode_stable(parcelle["prediction_live"])
        )
    exactitude_parcelles = float(np.mean(votes)) if votes else 0.0

    concordance_stockee = None
    if "groupe_kmeans" in test_tuile.columns:
        attendu = test_tuile.set_index("superpixel_id")["groupe_kmeans"]
        aligne = donnees["superpixel_id"].map(attendu).astype(int)
        concordance_stockee = float(
            np.mean(aligne.to_numpy() == groupes_predits)
        )

    return {
        "donnees": donnees.drop(columns=["_tile"]),
        "cartes": cartes,
        "exactitude_sp": exactitude_sp,
        "exactitude_parcelles": exactitude_parcelles,
        "concordance_stockee": concordance_stockee,
        "nombre_parcelles": int(donnees["field_id"].nunique()),
        "nombre_groupes": int(donnees["groupe_kmeans"].nunique()),
        "parametres_slic": parametres,
        "rapport": rapport,
    }


def page_regroupement() -> None:
    entete(
        "Regroupement K-means",
        "Former des groupes spectraux sans fournir à l'algorithme les noms des cultures.",
    )

    with st.form("formulaire_kmeans"):
        c1, c2 = st.columns(2)
        k = c1.selectbox(
            "Nombre de groupes",
            [4, 9],
            index=1,
            format_func=lambda valeur: (
                "K=4 — structure naturelle recommandée"
                if valeur == 4
                else "K=9 — comparaison avec les neuf cultures"
            ),
        )
        predictions = lire_csv(
            str(MODELES_KMEANS[int(k)] / "predictions_superpixels_test.csv")
        ).copy()
        predictions["_tile"] = predictions["tile_id"].map(id_texte)
        tuiles = tri_identifiants(predictions["_tile"])
        index_demo = tuiles.index("276") if "276" in tuiles else 0
        tuile = c2.selectbox("Tuile appartenant au jeu de test", tuiles, index=index_demo)
        lance = st.form_submit_button("Lancer le regroupement", type="primary")

    if lance:
        st.session_state["dernier_regroupement"] = (tuile, int(k))

    demande = st.session_state.get("dernier_regroupement")
    if demande is None:
        st.info(
            "Choisissez K et une tuile, puis cliquez sur « Lancer le regroupement ». "
            "K=4 décrit la structure la plus nette; K=9 facilite la comparaison avec les cultures."
        )
        return

    tuile_calculee, k_calcule = demande
    if (tuile_calculee, k_calcule) != (tuile, int(k)):
        st.caption(
            f"Résultat affiché : tuile {tuile_calculee}, K={k_calcule}. "
            "Cliquez sur le bouton pour appliquer la nouvelle sélection."
        )

    try:
        with st.spinner(f"Affectation aux groupes K-means, K={k_calcule}…"):
            resultat = executer_regroupement(tuile_calculee, k_calcule)
    except DonneesApplicationError as exc:
        st.error(str(exc))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Superpixels test", len(resultat["donnees"]))
    c2.metric("Parcelles", resultat["nombre_parcelles"])
    c3.metric("Groupes présents", resultat["nombre_groupes"])
    c4.metric("Correspondance indicative", f"{100 * resultat['exactitude_sp']:.1f} %")

    if resultat["cartes"] is not None:
        cartes = resultat["cartes"]
        cmap_groupes = ListedColormap(["#111827"] + COULEURS_GROUPES[:k_calcule])
        figure, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes[0, 0].imshow(
            mark_boundaries(cartes["rgb"], cartes["segments"], color=(1, 0, 0))
        )
        axes[0, 0].set_title("Image Sentinel-2 + SLIC")
        image_ndvi = axes[0, 1].imshow(cartes["ndvi"], cmap="RdYlGn", vmin=-1, vmax=1)
        axes[0, 1].set_title("NDVI — milieu de saison")
        figure.colorbar(image_ndvi, ax=axes[0, 1], fraction=0.046, pad=0.04)
        axes[0, 2].imshow(
            cartes["carte_groupes"], cmap=cmap_groupes, vmin=0, vmax=k_calcule
        )
        axes[0, 2].set_title(f"Groupes découverts — K={k_calcule}")
        axes[1, 0].imshow(cartes["carte_reelle"], cmap=CMAP_CLASSES, vmin=0, vmax=9)
        axes[1, 0].set_title("Cultures réelles — évaluation seulement")
        axes[1, 1].imshow(cartes["carte_predite"], cmap=CMAP_CLASSES, vmin=0, vmax=9)
        axes[1, 1].set_title("Culture majoritaire associée au groupe")
        axes[1, 2].imshow(cartes["carte_validite"], cmap=CMAP_VALIDITE, vmin=0, vmax=2)
        axes[1, 2].set_title("Vert : correspondance — Rouge : différence")
        for axe in axes.flat:
            axe.axis("off")
        figure.suptitle(
            f"Regroupement de la tuile {tuile_calculee} — K-means K={k_calcule}",
            fontsize=16,
            fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    else:
        st.warning(
            "Les images brutes de cette tuile de test ne sont pas fournies dans "
            "le ZIP. Les affectations sont recalculées avec le modèle enregistré "
            "et présentées dans le tableau ci-dessous. Pour afficher la carte "
            "géographique complète, ajouter le dossier original `dataset_lite`."
        )

    st.subheader("Superpixels et groupes obtenus")
    affiche = resultat["donnees"][
        [
            "tile_id",
            "superpixel_id",
            "field_id",
            "groupe_kmeans",
            "class_name",
            "prediction_live",
        ]
    ].copy()
    affiche["culture_associée"] = affiche["prediction_live"].map(CLASSES)
    st.dataframe(
        affiche.drop(columns=["prediction_live"]).rename(
            columns={
                "tile_id": "Tuile",
                "superpixel_id": "Superpixel",
                "field_id": "Parcelle",
                "groupe_kmeans": "Groupe K-means",
                "class_name": "Culture réelle (évaluation)",
                "culture_associée": "Culture majoritaire du groupe",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    if resultat["concordance_stockee"] is not None:
        st.success(
            "Contrôle de reproductibilité : "
            f"{100 * resultat['concordance_stockee']:.1f} % des groupes recalculés "
            "correspondent aux groupes enregistrés."
        )

    st.download_button(
        "Télécharger les groupes de cette tuile",
        data=resultat["donnees"].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"groupes_k{k_calcule}_tuile_{tuile_calculee}.csv",
        mime="text/csv",
    )
    st.caption(
        "Un numéro de groupe n’est pas un nom de culture. L’association à une culture "
        "est calculée après l’apprentissage, uniquement pour interpréter les résultats."
    )


def page_resultats() -> None:
    entete(
        "Résultats du regroupement",
        "Choisir K, mesurer la séparation des groupes et comparer après coup avec les cultures.",
    )

    rapports = {
        k: lire_json(str(dossier / "rapport_kmeans.json"))
        for k, dossier in MODELES_KMEANS.items()
    }
    rapport4, rapport9 = rapports[4], rapports[9]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("K recommandé en interne", "4")
    c2.metric("Silhouette K=4", f"{rapport4['metriques_internes_test']['silhouette']:.3f}")
    c3.metric("NMI K=9", f"{rapport9['metriques_externes_test']['nmi']:.3f}")
    c4.metric(
        "Correspondance K=9",
        f"{100 * rapport9['correspondance_superpixels_test']['exactitude']:.1f} %",
    )

    st.subheader("Comment le nombre de groupes a été choisi")
    gauche, droite = st.columns([1.35, 1], gap="large")
    gauche.image(
        str(MODELES_KMEANS[9] / "comparaison_k.png"),
        caption="Comparaison de K=2 à K=12 sur les données d’apprentissage.",
        width="stretch",
    )
    comparaison = pd.DataFrame(
        [
            {
                "Configuration": "K=4 — structure naturelle",
                "Silhouette": rapport4["metriques_internes_test"]["silhouette"],
                "Davies-Bouldin": rapport4["metriques_internes_test"]["davies_bouldin"],
                "ARI": rapport4["metriques_externes_test"]["ari"],
                "NMI": rapport4["metriques_externes_test"]["nmi"],
                "Exactitude indicative": rapport4["correspondance_superpixels_test"]["exactitude"],
                "Macro-F1 indicatif": rapport4["correspondance_superpixels_test"]["f1_macro"],
            },
            {
                "Configuration": "K=9 — neuf cultures",
                "Silhouette": rapport9["metriques_internes_test"]["silhouette"],
                "Davies-Bouldin": rapport9["metriques_internes_test"]["davies_bouldin"],
                "ARI": rapport9["metriques_externes_test"]["ari"],
                "NMI": rapport9["metriques_externes_test"]["nmi"],
                "Exactitude indicative": rapport9["correspondance_superpixels_test"]["exactitude"],
                "Macro-F1 indicatif": rapport9["correspondance_superpixels_test"]["f1_macro"],
            },
        ]
    )
    droite.dataframe(
        comparaison.style.format(
            {
                "Silhouette": "{:.3f}",
                "Davies-Bouldin": "{:.3f}",
                "ARI": "{:.3f}",
                "NMI": "{:.3f}",
                "Exactitude indicative": "{:.1%}",
                "Macro-F1 indicatif": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Composition des groupes et correspondance avec les cultures")
    k = st.selectbox("Configuration à analyser", [4, 9], index=1, format_func=lambda x: f"K={x}")
    dossier = MODELES_KMEANS[int(k)]
    gauche, droite = st.columns(2, gap="large")
    gauche.image(
        str(dossier / "groupes_vs_cultures.png"),
        caption="Chaque ligne indique les cultures présentes dans un groupe.",
        width="stretch",
    )
    droite.image(
        str(dossier / "matrice_superpixels.png"),
        caption="Correspondance après association de chaque groupe à sa culture majoritaire.",
        width="stretch",
    )

    st.markdown(
        '<div class="note"><b>Interprétation :</b> K=4 forme les groupes les plus '
        "compacts et les mieux séparés. K=9 correspond mieux au nombre attendu de cultures, "
        "mais les groupes restent mélangés. Cela montre que la ressemblance spectrale seule "
        "ne suffit pas à reconstruire exactement les neuf étiquettes agricoles.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Limites par rapport à la variante Random Forest")
    st.table(
        pd.DataFrame(
            [
                ["Étiquettes pendant l'apprentissage", "Non", "Oui"],
                ["Sortie directe", "Groupes numérotés", "Noms des cultures"],
                ["Besoin d'annotations", "Faible", "Important"],
                ["Cultures spectralement proches", "Souvent mélangées", "Mieux séparées"],
                ["Usage principal", "Exploration", "Prédiction"],
            ],
            columns=["Critère", "K-means — variante A", "Random Forest — variante B"],
        )
    )


def afficher_sidebar() -> str:
    st.sidebar.markdown(
        '<div class="sidebar-brand"><strong>AgriSLIC</strong>'
        '<span>Démonstrateur scientifique · Groupe 5</span></div>',
        unsafe_allow_html=True,
    )
    libelles = {
        "Accueil": "01  Vue d’ensemble",
        "Segmentation SLIC": "02  Segmentation SLIC",
        "Regroupement K-means": "03  Regroupement K-means",
        "Résultats": "04  Résultats",
    }
    page = st.sidebar.radio(
        "Parcours de démonstration",
        ["Accueil", "Segmentation SLIC", "Regroupement K-means", "Résultats"],
        format_func=lambda valeur: libelles[valeur],
    )
    _, absents = diagnostic_projet()
    if absents:
        st.sidebar.error(f"{len(absents)} fichier(s) requis absent(s)")
        with st.sidebar.expander("Voir le diagnostic"):
            for path in absents:
                st.write(f"❌ `{path.relative_to(RACINE)}`")
    else:
        st.sidebar.markdown(
            '<div class="sidebar-status"><b>État du projet</b><br>'
            + ("Modèles et exemples SLIC disponibles. Les cartes des tuiles de test "
             "requièrent les images brutes complètes.</div>"
             if MODE_EXEMPLE else "Données et modèles disponibles. Démonstration prête.</div>"),
            unsafe_allow_html=True,
        )
    st.sidebar.caption(
        "Les modèles K-means sont déjà entraînés. Pendant la présentation, "
        "l'affectation des superpixels de la tuile aux groupes est recalculée."
    )
    return page


def main() -> None:
    page = afficher_sidebar()
    try:
        if page == "Accueil":
            page_accueil()
        elif page == "Segmentation SLIC":
            page_segmentation()
        elif page == "Regroupement K-means":
            page_regroupement()
        else:
            page_resultats()
    except DonneesApplicationError as exc:
        st.error(str(exc))
        st.info(
            "Vérifiez que `app.py` est resté à la racine du projet et que les dossiers "
            "`dataset_lite` et `resultats` n'ont pas été déplacés."
        )
    except Exception as exc:
        st.error("Une erreur inattendue a interrompu cette page.")
        st.exception(exc)

    st.markdown(
        '<div class="footer">Projet académique · Classification de parcelles agricoles '
        "par superpixels SLIC et caractéristiques spectrales · Variante A : K-means non supervisé.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
