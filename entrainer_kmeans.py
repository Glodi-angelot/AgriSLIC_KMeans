"""Regroupement non supervise des superpixels agricoles avec K-means.

Cette version correspond a la variante A du sujet d'examen :
SLIC + caracteristiques spectrales + k-moyennes (non supervise).

Les etiquettes de cultures ne sont jamais transmises a K-means. Elles servent
uniquement a construire un jeu de test equitable, a donner un nom indicatif
aux groupes apres l'apprentissage et a evaluer les regroupements obtenus.

Exemple
-------
    python entrainer_kmeans.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    completeness_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


GRAINE = 42
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tester plusieurs valeurs de K et entrainer K-means sur les superpixels."
    )
    parser.add_argument(
        "--donnees",
        type=Path,
        default=Path("resultats/caracteristiques_superpixels.csv"),
        help="CSV contenant les metadonnees et les 64 caracteristiques.",
    )
    parser.add_argument(
        "--sortie",
        type=Path,
        default=Path("resultats/kmeans"),
        help="Dossier dans lequel enregistrer les resultats.",
    )
    parser.add_argument(
        "--split-reference",
        type=Path,
        default=Path("resultats/split_test_reference.csv"),
        help=(
            "Anciennes predictions contenant les parcelles de test. Leur reutilisation "
            "permet une comparaison directe avec LDA, KNN et SVM."
        ),
    )
    parser.add_argument(
        "--caracteristiques",
        choices=("moyennes", "toutes"),
        default="moyennes",
        help=(
            "moyennes utilise les 24 signatures moyennes des trois saisons; "
            "toutes utilise les 64 variables extraites."
        ),
    )
    parser.add_argument(
        "--k-min", type=int, default=2, help="Premiere valeur de K a comparer."
    )
    parser.add_argument(
        "--k-max", type=int, default=12, help="Derniere valeur de K a comparer."
    )
    parser.add_argument(
        "--k-final",
        type=int,
        default=9,
        help="Nombre de groupes du modele final (9 cultures dans le corpus).",
    )
    return parser.parse_args()


def choisir_split_test(
    X: pd.DataFrame, y: pd.Series, groupes: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Produit un test d'environ 20 %, separe par parcelle.

    Les classes servent ici seulement a equilibrer l'evaluation. Elles ne sont
    pas utilisees par StandardScaler ni par K-means.
    """

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=GRAINE)
    classes_attendues = set(np.unique(y))
    candidats = []
    for train_idx, test_idx in splitter.split(X, y, groupes):
        classes_test = set(np.unique(y.iloc[test_idx]))
        couverture = len(classes_test & classes_attendues)
        ecart_taille = abs(len(test_idx) / len(y) - 0.20)
        candidats.append((couverture, -ecart_taille, train_idx, test_idx))
    candidats.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidats[0][2], candidats[0][3]


def calculer_purete(y_vrai: np.ndarray, groupes: np.ndarray) -> float:
    """Part des observations appartenant a la classe majoritaire de leur groupe."""

    total_majoritaires = 0
    for groupe in np.unique(groupes):
        valeurs = y_vrai[groupes == groupe]
        total_majoritaires += Counter(valeurs).most_common(1)[0][1]
    return total_majoritaires / len(y_vrai)


def construire_correspondance(
    y_train: np.ndarray, groupes_train: np.ndarray
) -> tuple[dict[int, int], pd.DataFrame]:
    """Associe chaque groupe a sa culture majoritaire apres le clustering."""

    correspondance: dict[int, int] = {}
    lignes = []
    for groupe in sorted(np.unique(groupes_train)):
        classes_groupe = y_train[groupes_train == groupe]
        compte = Counter(int(x) for x in classes_groupe)
        classe, nombre = compte.most_common(1)[0]
        correspondance[int(groupe)] = int(classe)
        lignes.append(
            {
                "groupe_kmeans": int(groupe),
                "classe_majoritaire_code": int(classe),
                "classe_majoritaire": CLASSES[int(classe)],
                "effectif_apprentissage": int(len(classes_groupe)),
                "purete_groupe": float(nombre / len(classes_groupe)),
            }
        )
    return correspondance, pd.DataFrame(lignes)


def mode_deterministe(valeurs: pd.Series) -> int:
    compte = valeurs.value_counts()
    maximum = compte.max()
    return int(min(compte[compte == maximum].index))


def agreger_par_parcelle(predictions: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    for field_id, groupe in predictions.groupby("field_id", sort=True):
        lignes.append(
            {
                "field_id": int(field_id),
                "tile_id": int(groupe["tile_id"].mode().iloc[0]),
                "classe_reelle_code": mode_deterministe(groupe["classe_reelle_code"]),
                "classe_predite_code": mode_deterministe(groupe["classe_predite_code"]),
                "nombre_superpixels": int(len(groupe)),
            }
        )
    resultat = pd.DataFrame(lignes)
    resultat["classe_reelle"] = resultat["classe_reelle_code"].map(CLASSES)
    resultat["classe_predite"] = resultat["classe_predite_code"].map(CLASSES)
    resultat["correct"] = (
        resultat["classe_reelle_code"] == resultat["classe_predite_code"]
    )
    return resultat


def metriques_classification(y_vrai: np.ndarray, y_predit: np.ndarray) -> dict:
    return {
        "exactitude": float(accuracy_score(y_vrai, y_predit)),
        "exactitude_equilibree": float(balanced_accuracy_score(y_vrai, y_predit)),
        "f1_macro": float(f1_score(y_vrai, y_predit, average="macro", zero_division=0)),
        "f1_pondere": float(
            f1_score(y_vrai, y_predit, average="weighted", zero_division=0)
        ),
    }


def sauvegarder_comparaison_k(comparaison: pd.DataFrame, k_final: int, sortie: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(comparaison["k"], comparaison["inertie"], marker="o", color="#1F6B4B")
    axes[0].set_title("Methode du coude")
    axes[0].set_ylabel("Inertie (plus faible = groupes compacts)")

    axes[1].plot(
        comparaison["k"], comparaison["silhouette"], marker="o", color="#2878B5"
    )
    axes[1].set_title("Score de silhouette")
    axes[1].set_ylabel("Plus eleve = meilleure separation")

    axes[2].plot(
        comparaison["k"],
        comparaison["davies_bouldin"],
        marker="o",
        color="#D1495B",
    )
    axes[2].set_title("Indice de Davies-Bouldin")
    axes[2].set_ylabel("Plus faible = meilleure separation")

    for axe in axes:
        axe.axvline(k_final, color="#B9852B", linestyle="--", label=f"K final = {k_final}")
        axe.set_xlabel("Nombre de groupes K")
        axe.set_xticks(comparaison["k"])
        axe.grid(alpha=0.2)
        axe.legend()

    figure.tight_layout()
    figure.savefig(sortie / "comparaison_k.png", dpi=180)
    plt.close(figure)


def sauvegarder_table_groupes_classes(
    y_vrai: np.ndarray, groupes: np.ndarray, sortie: Path
) -> None:
    tableau = pd.crosstab(
        pd.Series(groupes, name="Groupe K-means"),
        pd.Series([CLASSES[int(x)] for x in y_vrai], name="Culture reelle"),
        normalize="index",
    )
    tableau.to_csv(sortie / "proportions_classes_par_groupe.csv", encoding="utf-8-sig")
    figure, axe = plt.subplots(figsize=(13, 7))
    sns.heatmap(tableau, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=axe)
    axe.set_title("Composition des groupes K-means sur le jeu de test")
    axe.set_xlabel("Culture reelle (utilisee seulement pour l'interpretation)")
    axe.set_ylabel("Groupe trouve sans etiquette")
    plt.xticks(rotation=40, ha="right")
    figure.tight_layout()
    figure.savefig(sortie / "groupes_vs_cultures.png", dpi=180)
    plt.close(figure)


def sauvegarder_matrice(
    y_vrai: np.ndarray, y_predit: np.ndarray, nom: str, sortie: Path
) -> None:
    codes = sorted(CLASSES)
    matrice = confusion_matrix(y_vrai, y_predit, labels=codes, normalize="true")
    etiquettes = [f"{code}: {CLASSES[code]}" for code in codes]
    figure, axe = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        matrice,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0,
        vmax=1,
        xticklabels=etiquettes,
        yticklabels=etiquettes,
        ax=axe,
    )
    axe.set_title(f"Correspondance groupes-cultures - {nom}")
    axe.set_xlabel("Culture associee au groupe")
    axe.set_ylabel("Culture reelle")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    figure.tight_layout()
    figure.savefig(sortie / f"matrice_{nom}.png", dpi=180)
    plt.close(figure)


def sauvegarder_projection_spectrale(predictions: pd.DataFrame, sortie: Path) -> None:
    """Visualisation interpretable, sans utiliser PCA dans le modele."""

    x_colonne = "milieu_saison_ndvi_moy"
    y_colonne = "milieu_saison_B08_moy"
    figure, axe = plt.subplots(figsize=(10, 7))
    nuage = axe.scatter(
        predictions[x_colonne],
        predictions[y_colonne],
        c=predictions["groupe_kmeans"],
        cmap="tab10",
        s=20,
        alpha=0.7,
    )
    axe.set_title("Groupes K-means dans deux caracteristiques spectrales")
    axe.set_xlabel("NDVI moyen - milieu de saison")
    axe.set_ylabel("B08 moyenne - milieu de saison")
    axe.grid(alpha=0.2)
    legende = axe.legend(*nuage.legend_elements(), title="Groupe", loc="best")
    axe.add_artist(legende)
    figure.tight_layout()
    figure.savefig(sortie / "groupes_espace_spectral.png", dpi=180)
    plt.close(figure)


def principal() -> None:
    args = arguments()
    if args.k_min < 2 or args.k_max < args.k_min:
        raise ValueError("La plage de K est invalide.")
    if not args.k_min <= args.k_final <= args.k_max:
        raise ValueError("K final doit appartenir a la plage comparee.")

    donnees = pd.read_csv(args.donnees)
    toutes_colonnes_features = [c for c in donnees.columns if c not in META_COLONNES]
    if len(toutes_colonnes_features) != 64:
        raise ValueError(
            f"64 caracteristiques attendues, {len(toutes_colonnes_features)} trouvees."
        )
    if args.caracteristiques == "moyennes":
        colonnes_features = [c for c in toutes_colonnes_features if c.endswith("_moy")]
    else:
        colonnes_features = toutes_colonnes_features

    X = donnees[colonnes_features].astype(np.float64)
    y = donnees["class_code"].astype(int)
    groupes_parcelles = donnees["field_id"].astype(int)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Le CSV contient des valeurs manquantes ou infinies.")

    if args.split_reference.exists():
        reference = pd.read_csv(args.split_reference)
        champs_test = set(reference["field_id"].astype(int))
        masque_test = groupes_parcelles.isin(champs_test).to_numpy()
        train_idx = np.flatnonzero(~masque_test)
        test_idx = np.flatnonzero(masque_test)
        source_split = str(args.split_reference)
    else:
        train_idx, test_idx = choisir_split_test(X, y, groupes_parcelles)
        source_split = "StratifiedGroupKFold genere par le script"
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx].to_numpy()
    y_test = y.iloc[test_idx].to_numpy()

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    args.sortie.mkdir(parents=True, exist_ok=True)
    lignes_comparaison = []
    modeles: dict[int, KMeans] = {}

    for k in range(args.k_min, args.k_max + 1):
        modele = KMeans(n_clusters=k, n_init=30, random_state=GRAINE)
        groupes_train = modele.fit_predict(X_train_std)
        modeles[k] = modele
        taille_echantillon = min(5000, len(X_train_std))
        lignes_comparaison.append(
            {
                "k": int(k),
                "inertie": float(modele.inertia_),
                "silhouette": float(
                    silhouette_score(
                        X_train_std,
                        groupes_train,
                        sample_size=taille_echantillon,
                        random_state=GRAINE,
                    )
                ),
                "davies_bouldin": float(
                    davies_bouldin_score(X_train_std, groupes_train)
                ),
                "calinski_harabasz": float(
                    calinski_harabasz_score(X_train_std, groupes_train)
                ),
            }
        )

    comparaison = pd.DataFrame(lignes_comparaison)
    comparaison.to_csv(args.sortie / "comparaison_k.csv", index=False, encoding="utf-8-sig")
    sauvegarder_comparaison_k(comparaison, args.k_final, args.sortie)

    k_recommande_silhouette = int(
        comparaison.loc[comparaison["silhouette"].idxmax(), "k"]
    )
    k_recommande_davies = int(
        comparaison.loc[comparaison["davies_bouldin"].idxmin(), "k"]
    )

    modele_final = modeles[args.k_final]
    groupes_train = modele_final.labels_.astype(int)
    groupes_test = modele_final.predict(X_test_std).astype(int)
    correspondance, table_correspondance = construire_correspondance(
        y_train, groupes_train
    )
    y_predit = np.array([correspondance[int(g)] for g in groupes_test], dtype=int)

    predictions = donnees.iloc[test_idx][
        ["tile_id", "superpixel_id", "field_id", "class_code", "class_name"]
        + ["milieu_saison_ndvi_moy", "milieu_saison_B08_moy"]
    ].copy()
    predictions = predictions.rename(
        columns={"class_code": "classe_reelle_code", "class_name": "classe_reelle"}
    )
    predictions["groupe_kmeans"] = groupes_test
    predictions["classe_predite_code"] = y_predit
    predictions["classe_predite"] = predictions["classe_predite_code"].map(CLASSES)
    predictions["correct"] = (
        predictions["classe_reelle_code"] == predictions["classe_predite_code"]
    )
    predictions.to_csv(
        args.sortie / "predictions_superpixels_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    predictions_parcelles = agreger_par_parcelle(predictions)
    predictions_parcelles.to_csv(
        args.sortie / "predictions_parcelles_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    table_correspondance["effectif_test"] = table_correspondance["groupe_kmeans"].map(
        pd.Series(groupes_test).value_counts()
    ).fillna(0).astype(int)
    table_correspondance.to_csv(
        args.sortie / "correspondance_groupes_cultures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metriques_internes_test = {
        "silhouette": float(silhouette_score(X_test_std, groupes_test)),
        "davies_bouldin": float(davies_bouldin_score(X_test_std, groupes_test)),
        "calinski_harabasz": float(
            calinski_harabasz_score(X_test_std, groupes_test)
        ),
    }
    metriques_externes_test = {
        "ari": float(adjusted_rand_score(y_test, groupes_test)),
        "nmi": float(normalized_mutual_info_score(y_test, groupes_test)),
        "homogeneite": float(homogeneity_score(y_test, groupes_test)),
        "completude": float(completeness_score(y_test, groupes_test)),
        "v_measure": float(v_measure_score(y_test, groupes_test)),
        "purete": float(calculer_purete(y_test, groupes_test)),
    }
    metriques_superpixels = metriques_classification(y_test, y_predit)
    metriques_parcelles = metriques_classification(
        predictions_parcelles["classe_reelle_code"].to_numpy(),
        predictions_parcelles["classe_predite_code"].to_numpy(),
    )

    rapport = {
        "methode": "StandardScaler + KMeans",
        "apprentissage_non_supervise": True,
        "usage_des_etiquettes": (
            "Uniquement pour equilibrer le split, nommer les groupes apres "
            "apprentissage et evaluer; jamais pour calculer les centroides."
        ),
        "graine": GRAINE,
        "nombre_lignes": int(len(donnees)),
        "nombre_caracteristiques_extraites": int(len(toutes_colonnes_features)),
        "famille_caracteristiques_utilisee": args.caracteristiques,
        "nombre_caracteristiques_utilisees": int(len(colonnes_features)),
        "nombre_parcelles": int(donnees["field_id"].nunique()),
        "source_split_test": source_split,
        "train_superpixels": int(len(train_idx)),
        "test_superpixels": int(len(test_idx)),
        "train_parcelles": int(donnees.iloc[train_idx]["field_id"].nunique()),
        "test_parcelles": int(donnees.iloc[test_idx]["field_id"].nunique()),
        "k_final": int(args.k_final),
        "k_recommande_silhouette": k_recommande_silhouette,
        "k_recommande_davies_bouldin": k_recommande_davies,
        "metriques_internes_test": metriques_internes_test,
        "metriques_externes_test": metriques_externes_test,
        "correspondance_superpixels_test": metriques_superpixels,
        "correspondance_parcelles_test": metriques_parcelles,
        "colonnes_features": colonnes_features,
        "correspondance_groupes_classes": {
            str(k): int(v) for k, v in correspondance.items()
        },
    }
    (args.sortie / "rapport_kmeans.json").write_text(
        json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": modele_final,
            "correspondance_groupes_classes": correspondance,
            "colonnes_features": colonnes_features,
            "classes": CLASSES,
            "k_final": args.k_final,
            "graine": GRAINE,
        },
        args.sortie / "modele_kmeans.joblib",
    )

    sauvegarder_table_groupes_classes(y_test, groupes_test, args.sortie)
    sauvegarder_matrice(y_test, y_predit, "superpixels", args.sortie)
    sauvegarder_matrice(
        predictions_parcelles["classe_reelle_code"].to_numpy(),
        predictions_parcelles["classe_predite_code"].to_numpy(),
        "parcelles",
        args.sortie,
    )
    sauvegarder_projection_spectrale(predictions, args.sortie)

    print("Experience K-means terminee.")
    print(
        f"Superpixels : {len(donnees)} | Caracteristiques utilisees : "
        f"{len(colonnes_features)}/64 ({args.caracteristiques})"
    )
    print(f"Train/Test : {len(train_idx)}/{len(test_idx)} superpixels")
    print(f"K final : {args.k_final}")
    print(
        "Silhouette / Davies-Bouldin (test) : "
        f"{metriques_internes_test['silhouette']:.4f} / "
        f"{metriques_internes_test['davies_bouldin']:.4f}"
    )
    print(
        "ARI / NMI (test) : "
        f"{metriques_externes_test['ari']:.4f} / "
        f"{metriques_externes_test['nmi']:.4f}"
    )
    print(
        "Correspondance superpixels - exactitude / macro-F1 : "
        f"{metriques_superpixels['exactitude']:.4f} / "
        f"{metriques_superpixels['f1_macro']:.4f}"
    )
    print(
        "Correspondance parcelles - exactitude / macro-F1 : "
        f"{metriques_parcelles['exactitude']:.4f} / "
        f"{metriques_parcelles['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    principal()
