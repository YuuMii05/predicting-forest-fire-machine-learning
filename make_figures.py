"""Build every figure from the saved results (reports/results -> reports/figures).

    python make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.data import ALL_FEATURES, TARGET, ZONES, load_clean
from src.models import MODEL_COLORS

ROOT = Path(__file__).parent
RES = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- Style -------------------------------------------------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
FIRE, NOFIRE = "#eb6834", "#2a78d6"          # orange = fire, blue = no fire
SCALER_COLORS = {"aucune": "#8a8984", "standard": "#2a78d6", "minmax": "#1baf7a"}
FS_COLORS = {"meteo+FWI": "#2a78d6", "meteo seule": "#eb6834"}
FS_TITLES = {"meteo+FWI": "Météo + indices FWI (10 variables)", "meteo seule": "Météo du jour seule (4 variables)"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f1ee", "#2a78d6"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "text.color": INK,
})

MODEL_ORDER = list(MODEL_COLORS)
ZONE_ORDER = [z[0] for z in ZONES]


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  ->", name)


def short(name: str) -> str:
    return {"Baseline (au hasard)": "Hasard", "Régression logistique": "Rég. logistique",
            "SVM (noyau RBF)": "SVM (RBF)", "Indice FWI officiel seul": "Indice FWI seul"}.get(name, name)


def wrap(name: str) -> str:
    return {"Arbre de décision": "Arbre de\ndécision", "Random Forest": "Random\nForest",
            "Gradient Boosting": "Gradient\nBoosting", "Régression logistique": "Rég.\nlogistique",
            "SVM (noyau RBF)": "SVM\n(RBF)", "Indice FWI officiel seul": "Indice\nFWI seul"}.get(name, short(name))


# --- 1. Exploration ----------------------------------------------------------
def fig_fires_by_zone_year(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    z = df.groupby("zone")[TARGET].agg(["sum", "mean"]).loc[ZONE_ORDER[::-1]]
    ax = axes[0]
    gov = {a[0]: a[1] for a in ZONES}
    ax.barh([zz if zz == gov[zz] else f"{zz} ({gov[zz]})" for zz in z.index], z["sum"], color=FIRE, height=0.6)
    for i, (s, m) in enumerate(zip(z["sum"], z["mean"])):
        ax.text(s + 3, i, f"{s} jours ({m:.0%})", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, z["sum"].max() * 1.35)
    ax.set_title("Jours avec feu détecté, par zone (2013–2025)")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Nombre de jours (juin–septembre)")

    ax = axes[1]
    piv = df.pivot_table(index="year", columns="month", values=TARGET, aggfunc="sum")
    bottom = np.zeros(len(piv))
    shades = {6: "#f6c7b3", 7: "#f09a72", 8: "#eb6834", 9: "#b8441d"}
    names = {6: "juin", 7: "juillet", 8: "août", 9: "septembre"}
    for m in [6, 7, 8, 9]:
        ax.bar(piv.index, piv[m], bottom=bottom, color=shades[m], label=names[m], width=0.7,
               edgecolor="white", linewidth=1)
        bottom += piv[m].values
    ax.set_title("Jours-zones avec feu, par été")
    ax.set_xticks(piv.index, [f"’{str(y)[2:]}" for y in piv.index])
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "01_feux_par_zone_et_annee.png")


def fig_distributions(df):
    fig, axes = plt.subplots(2, 5, figsize=(13, 5))
    for ax, f in zip(axes.ravel(), ALL_FEATURES):
        data = [df.loc[df[TARGET] == 0, f], df.loc[df[TARGET] == 1, f]]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", linewidth=1.6))
        for patch, c in zip(bp["boxes"], [NOFIRE, FIRE]):
            patch.set(facecolor=c, edgecolor=c)
        for part in ["whiskers", "caps"]:
            for line in bp[part]:
                line.set(color=MUTED)
        ax.set_xticks([1, 2], ["Pas de feu", "Feu"])
        ax.set_title(f, fontsize=11)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Distribution de chaque variable selon la classe (valeurs extrêmes masquées)", x=0.01,
                 ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "02_distributions_par_classe.png")


def fig_scales(df):
    lo, hi = df[ALL_FEATURES].quantile(0.01), df[ALL_FEATURES].quantile(0.99)
    order = (hi - lo).sort_values().index
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(order))
    ax.hlines(y, lo[order], hi[order], color=NOFIRE, linewidth=6, alpha=0.85)
    for i, f in enumerate(order):
        ax.text(hi[f] + 15, i, f"{lo[f]:.0f} → {hi[f]:.0f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, order)
    ax.set_xlim(-20, hi.max() * 1.2)
    ax.set_title("Les variables n'ont pas du tout la même échelle")
    ax.set_xlabel("Plage de valeurs (1er → 99e centile)")
    ax.grid(axis="y", visible=False)
    save(fig, "03_echelles_des_variables.png")


def fig_correlation(df):
    cols = ALL_FEATURES + [TARGET]
    corr = df[cols].corr()
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ee", "#eb6834"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    labels = ALL_FEATURES + ["feu"]
    ax.set_xticks(range(len(cols)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Corrélation de Pearson")
    ax.set_title("Corrélations entre variables (et avec la cible « feu »)")
    save(fig, "04_correlations.png")


# --- 2. Main comparison ------------------------------------------------------
def fig_cv_scores(folds, features, fname, title):
    sub = folds[folds.features.isin([features, "FWI seul"] if features == "meteo+FWI" else [features])]
    order = sub.groupby("model").average_precision.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        s = sub[sub.model == m].average_precision.values
        c = MODEL_COLORS.get(m, MUTED)
        ax.scatter(s, i + rng.uniform(-0.18, 0.18, len(s)), s=14, color=c, alpha=0.45, edgecolor="none", zorder=2)
        ax.errorbar(s.mean(), i, xerr=s.std(), fmt="o", color=INK, markersize=7, markerfacecolor=c,
                    markeredgecolor="white", capsize=0, elinewidth=1.6, zorder=3)
        ax.text(1.005, i, f"{s.mean():.3f} ± {s.std():.3f}", va="center", fontsize=9, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(order)), [short(m) for m in order])
    ax.set_xlabel("Précision moyenne (aire sous la courbe précision-rappel) sur 15 plis de test, groupés par année")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    save(fig, fname)


def fig_feature_sets(summary):
    s = summary[summary.features.isin(["meteo+FWI", "meteo seule"])]
    piv = s.pivot(index="model", columns="features", values="average_precision_mean")
    piv = piv.loc[[m for m in MODEL_ORDER if m in piv.index]].iloc[::-1]
    fwi = summary[summary.features == "FWI seul"].average_precision_mean.iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(piv))
    ax.axvline(fwi, color=INK2, linestyle="--", linewidth=1)
    ax.text(fwi, len(piv) - 0.4, " indice FWI seul", color=INK2, fontsize=9, va="bottom")
    ax.hlines(y, piv["meteo seule"], piv["meteo+FWI"], color=GRID, linewidth=3, zorder=1)
    for fs, lab in [("meteo seule", "Météo du jour seule (4 variables)"),
                    ("meteo+FWI", "Météo + indices FWI (10 variables)")]:
        ax.scatter(piv[fs], y, s=70, color=FS_COLORS[fs], zorder=2, label=lab, edgecolor="white", linewidth=1.5)
    ax.set_yticks(y, [short(m) for m in piv.index])
    ax.set_xlabel("Précision moyenne (validation croisée)")
    ax.set_title("Les indices FWI apportent-ils quelque chose ?")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, "06_meteo_seule_vs_fwi.png")


def fig_scaling(scaling):
    models = [m for m in MODEL_ORDER if m in scaling.model.unique()]
    labels = {"aucune": "Aucune", "standard": "Standardisation (z-score)", "minmax": "Min-Max [0, 1]"}
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.6), sharex=True)
    for ax, fs in zip(axes, ["meteo+FWI", "meteo seule"]):
        piv = scaling[scaling.features == fs].pivot(index="model", columns="scaler", values="ap_mean").loc[models]
        x = np.arange(len(models))
        w = 0.26
        for k, sc in enumerate(["aucune", "standard", "minmax"]):
            ax.bar(x + (k - 1) * w, piv[sc], width=w - 0.03, color=SCALER_COLORS[sc], label=labels[sc])
        ax.set_ylim(max(0, piv.values.min() - 0.03), piv.values.max() + 0.015)
        ax.set_ylabel("Précision moyenne")
        ax.set_title(FS_TITLES[fs], fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x, [wrap(m) for m in models])
    fig.suptitle("Effet de la mise à l'échelle selon le modèle", x=0.01, ha="left", fontweight="bold", fontsize=13)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    save(fig, "07_effet_mise_a_echelle.png")


def fig_knn_curve(curve):
    labels = {"aucune": "Sans mise à l'échelle", "standard": "Avec standardisation"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for ax, fs in zip(axes, ["meteo+FWI", "meteo seule"]):
        for sc, g in curve[curve.features == fs].groupby("scaler"):
            c = SCALER_COLORS[sc]
            ax.plot(g.k, g["mean"], color=c, linewidth=2, label=labels[sc], marker="o", markersize=3)
            best = g.loc[g["mean"].idxmax()]
            ax.scatter([best.k], [best["mean"]], s=60, color=c, edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(f"k = {int(best.k)} : {best['mean']:.3f}", (best.k, best["mean"]),
                        xytext=(6, 7), textcoords="offset points", fontsize=9, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100, 200], ["1", "3", "10", "30", "100", "200"])
        ax.set_xlabel("Nombre de voisins k (échelle log)")
        ax.set_title(FS_TITLES[fs], fontsize=11)
    axes[0].set_ylabel("Précision moyenne")
    axes[0].legend(loc="lower right")
    fig.suptitle("k-NN : choix de k par validation croisée, avec et sans mise à l'échelle",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, "08_knn_choix_de_k.png")


def fig_zone_out(loz):
    sub = loz[loz.features == "meteo+FWI"]
    models = [m for m in MODEL_ORDER if m in sub.model.unique() and not m.startswith("Baseline")]
    piv = sub.pivot(index="model", columns="test_zone", values="roc_auc").loc[models, ZONE_ORDER]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    cmap = LinearSegmentedColormap.from_list("rg", ["#f6c7b3", "#f2f1ee", "#2a78d6"])
    im = ax.imshow(piv.values, cmap=cmap, vmin=0.4, vmax=0.8, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(ZONE_ORDER)), ZONE_ORDER)
    ax.set_yticks(range(len(models)), [short(m) for m in models])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="ROC-AUC (0,5 = hasard)")
    ax.set_title("Entraîné sur 6 zones, testé sur la 7e (zone jamais vue) — météo + FWI")
    save(fig, "09_generalisation_zone_inconnue.png")


def fig_temporal(temporal, summary):
    fig, ax = plt.subplots(figsize=(9, 4))
    models = [m for m in MODEL_ORDER if m in temporal.model.unique()]
    x = np.arange(len(models))
    w = 0.36
    for k, fs in enumerate(["meteo seule", "meteo+FWI"]):
        sub = temporal[temporal.features == fs].set_index("model")
        vals = [sub.loc[m, "average_precision"] if m in sub.index else np.nan for m in models]
        ax.bar(x + (k - 0.5) * w, vals, width=w - 0.04, color=FS_COLORS[fs], label=FS_TITLES[fs])
    rate = temporal.fire_rate.iloc[0]
    ax.set_ylim(rate * 0.8, temporal.average_precision.max() * 1.08)
    ax.axhline(rate, color=INK2, linestyle="--", linewidth=1)
    ax.text(len(models) - 0.5, rate, f" hasard ({rate:.2f})", color=INK2, fontsize=9, va="bottom", ha="right")
    ax.set_xticks(x, [wrap(m) for m in models])
    ax.set_ylabel("Précision moyenne")
    ax.set_title("Entraîné sur 2013–2021, testé sur les étés 2022–2025")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "09b_generalisation_temporelle.png")


def fig_leakage(leak, summary):
    s = summary[summary.features == "meteo+FWI"].set_index("model")
    models = [m for m in MODEL_ORDER if m in leak.model.values and not m.startswith("Baseline")][::-1]
    leak = leak.set_index("model")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(models))
    grp = [s.loc[m, "average_precision_mean"] for m in models]
    shf = [leak.loc[m, "ap_shuffled"] for m in models]
    ax.hlines(y, grp, shf, color=GRID, linewidth=3, zorder=1)
    ax.scatter(grp, y, s=70, color="#2a78d6", edgecolor="white", linewidth=1.5, zorder=2,
               label="Plis groupés par année (honnête)")
    ax.scatter(shf, y, s=70, color="#eb6834", edgecolor="white", linewidth=1.5, zorder=2,
               label="Plis mélangés (fuite temporelle)")
    for yi, a, b in zip(y, grp, shf):
        ax.text(max(a, b) + 0.004, yi, f"{(b - a) / a:+.0%}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [short(m) for m in models])
    ax.set_xlabel("Précision moyenne — météo + FWI")
    ax.set_title("Mélanger les jours gonfle les scores")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "09c_fuite_temporelle.png")


def fig_confusions(cms, features, fname, title):
    keys = [f"{features}|{m}" for m in MODEL_ORDER if f"{features}|{m}" in cms]
    n, cols = len(keys), 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.9 * rows))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        cm = np.array(cms[k])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=SEQ, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if share > 0.6 else INK)
        ax.set_xticks([0, 1], ["Pas de feu", "Feu"])
        ax.set_yticks([0, 1], ["Pas de feu", "Feu"])
        ax.set_xlabel("Prédit", fontsize=9)
        ax.set_ylabel("Réel", fontsize=9)
        ax.grid(False)
        tp, fn, fp = cm[1, 1], cm[1, 0], cm[0, 1]
        ax.set_title(f"{short(k.split('|')[1])}\n{tp} feux vus · {fn} manqués · {fp} fausses alertes", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, fname)


def fig_importance(imp, fname, title):
    imp = imp.sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(imp) + 1.2))
    ax.barh(imp.feature, imp.importance, xerr=imp["std"], color="#2a78d6", height=0.6,
            error_kw=dict(ecolor=MUTED, elinewidth=1, capsize=0))
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("Baisse de la précision moyenne quand la variable est mélangée")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    save(fig, fname)


def fig_tradeoff(summary):
    s = summary[summary.features == "meteo+FWI"]
    s = s[~s.model.str.startswith("Baseline")]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for _, r in s.iterrows():
        c = MODEL_COLORS.get(r.model, MUTED)
        ax.scatter(r.fit_time_mean, r.average_precision_mean, s=80, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(short(r.model), (r.fit_time_mean, r.average_precision_mean), xytext=(7, -3),
                    textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Temps d'entraînement par pli, réglage inclus (s, échelle log)")
    ax.set_ylabel("Précision moyenne")
    ax.set_title("Performance vs coût de calcul")
    save(fig, "12_performance_vs_temps.png")


def main():
    for old in FIG.glob("*.png"):
        old.unlink()
    df = load_clean(save=False)
    folds = pd.read_csv(RES / "cv_folds.csv")
    summary = pd.read_csv(RES / "cv_summary.csv")
    print("Figures :")
    fig_fires_by_zone_year(df)
    fig_distributions(df)
    fig_scales(df)
    fig_correlation(df)
    fig_cv_scores(folds, "meteo+FWI", "05_comparaison_modeles.png", "Comparaison des modèles — météo + indices FWI")
    fig_cv_scores(folds, "meteo seule", "05b_comparaison_modeles_meteo_seule.png",
                  "Comparaison des modèles — météo du jour seule")
    fig_feature_sets(summary)
    fig_scaling(pd.read_csv(RES / "scaling.csv"))
    fig_knn_curve(pd.read_csv(RES / "knn_k_curve.csv"))
    fig_zone_out(pd.read_csv(RES / "leave_one_zone_out.csv"))
    fig_temporal(pd.read_csv(RES / "temporal_split.csv"), summary)
    fig_leakage(pd.read_csv(RES / "leakage_shuffled_cv.csv"), summary)
    cms = json.loads((RES / "confusion_matrices.json").read_text())
    fig_confusions(cms, "meteo+FWI", "10_matrices_confusion.png",
                   "Matrices de confusion (prédictions hors-pli, seuil 0,5) — météo + FWI")
    fig_importance(pd.read_csv(RES / "importance_all.csv"), "11_importance_variables.png",
                   "Random Forest : variables les plus utiles (météo + FWI)")
    fig_importance(pd.read_csv(RES / "importance_weather.csv"), "11b_importance_meteo_seule.png",
                   "Random Forest : variables les plus utiles (météo seule)")
    fig_tradeoff(summary)


if __name__ == "__main__":
    main()
