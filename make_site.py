"""Build the results website in docs/ (served by GitHub Pages).

    python make_site.py
    python make_site.py --repo https://github.com/<user>/predicting-forest-fire-machine-learning

Reads the dataset and the saved results, injects them (and the key numbers
quoted in the text) into site/template.html and copies the figures next to
the page. On GitHub Pages the GitHub links are found automatically from the
address; --repo only sets the fallback used when the page is opened locally.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.data import TARGET, ZONES, load_clean

ROOT = Path(__file__).parent
RES = ROOT / "reports" / "results"
DOCS = ROOT / "docs"
FIGURES = [
    "01_feux_par_zone_et_annee.png", "02_distributions_par_classe.png", "05_comparaison_modeles.png",
    "05b_comparaison_modeles_meteo_seule.png", "07_effet_mise_a_echelle.png", "08_knn_choix_de_k.png",
    "09_generalisation_zone_inconnue.png", "09b_generalisation_temporelle.png", "09c_fuite_temporelle.png",
    "10_matrices_confusion.png", "11_importance_variables.png", "11b_importance_meteo_seule.png",
]


def fr(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def calendar_data(df: pd.DataFrame) -> dict:
    zi = {z[0]: i for i, z in enumerate(ZONES)}
    start = pd.to_datetime(df["year"].astype(str) + "-06-01")
    rows = [
        [zi[r.zone], int(r.year), int(d), round(float(r.Temperature), 1), int(round(r.RH)), round(float(r.Ws), 1),
         round(float(r.Rain), 1), round(float(r.FWI), 1), int(r.n_detections)]
        for r, d in zip(df.itertuples(), (df["date"] - start).dt.days)
    ]
    return {"zones": [z[0] for z in ZONES], "governorates": [z[1] for z in ZONES],
            "years": sorted(df["year"].unique().tolist()), "rows": rows}


def results_data() -> dict:
    s = pd.read_csv(RES / "cv_summary.csv")
    tests = pd.read_csv(RES / "significance_tests.csv")
    rows = []
    for r in s.itertuples():
        fs_for_test = "meteo+FWI" if r.features == "FWI seul" else r.features
        t = tests[(tests.features == fs_for_test) & (tests.other == r.model)]
        is_best = bool(((tests.features == r.features) & (tests.best == r.model)).any())
        ref = "baseline" if r.model.startswith("Baseline") else ("rule" if r.features == "FWI seul" else None)
        rows.append({
            "features": r.features, "model": r.model, "family": r.family,
            "ap": r.average_precision_mean, "std": r.average_precision_std, "roc": r.roc_auc_mean,
            "rec": r.recall_mean, "best": is_best, "ref": ref,
            "p": float(t.p_value.iloc[0]) if len(t) else None,
            "delta": float(t.delta.iloc[0]) if len(t) else None,
        })
    return {"rows": rows}


def key_numbers(df: pd.DataFrame) -> dict:
    s = pd.read_csv(RES / "cv_summary.csv")
    tests = pd.read_csv(RES / "significance_tests.csv")
    leak = pd.read_csv(RES / "leakage_shuffled_cv.csv").set_index("model")
    temporal = pd.read_csv(RES / "temporal_split.csv")
    scaling = pd.read_csv(RES / "scaling.csv")
    imp_all, imp_w = pd.read_csv(RES / "importance_all.csv"), pd.read_csv(RES / "importance_weather.csv")

    allf = s[s.features == "meteo+FWI"].set_index("model")
    best = tests[tests.features == "meteo+FWI"].best.iloc[0]
    rand = allf.loc["Baseline (au hasard)", "average_precision_mean"]
    ap_best = allf.loc[best, "average_precision_mean"]
    ap_fwi = s[s.features == "FWI seul"].average_precision_mean.iloc[0]

    ties = tests[(tests.features == "meteo+FWI") & (tests.p_value > 0.05)
                 & ~tests.other.str.startswith("Baseline")].other.tolist()
    if ties:
        note = (f"Précision moyenne {fr(ap_best, 3)}, mais à égalité statistique avec "
                + ", ".join(ties[:-1]) + (" et " if len(ties) > 1 else "") + ties[-1] + ".")
    else:
        note = f"Précision moyenne {fr(ap_best, 3)}, significativement devant tous les autres."

    knn_grp = allf.loc["k-NN", "average_precision_mean"]
    t = temporal[(temporal.features == "meteo+FWI") & ~temporal.model.str.startswith("Baseline")]
    sc = scaling[(scaling.features == "meteo+FWI") & (scaling.model == "k-NN")].set_index("scaler").ap_mean
    gain = (sc["standard"] - sc["aucune"]) / sc["aucune"]
    return {
        "N_ROWS": f"{len(df):,}".replace(",", " "),
        "N_FIRE": f"{int(df[TARGET].sum()):,}".replace(",", " "),
        "FIRE_RATE": f"{df[TARGET].mean():.0%}".replace("%", " %"),
        "BEST_ALL": best,
        "AP_BEST": fr(ap_best),
        "AP_RAND": fr(rand),
        "AP_FWI": fr(ap_fwi),
        "RATIO_BEST": fr(ap_best / rand, 1),
        "LEAK_KNN": f"{(leak.loc['k-NN', 'ap_shuffled'] - knn_grp) / knn_grp:+.0%}".replace("%", " %"),
        "ROC_TEMP": fr(t.roc_auc.max()),
        "KNN_SCALE": f"{gain:+.0%}".replace("%", " %"),
        "TOP_ALL": " et ".join(imp_all.feature.iloc[:2]),
        "TOP_W": " et ".join(imp_w.feature.iloc[:2]),
        "BEST_NOTE": note,
    }


def main(repo: str):
    DOCS.mkdir(exist_ok=True)
    figdir = DOCS / "figures"
    if figdir.exists():
        shutil.rmtree(figdir)
    figdir.mkdir()
    for f in FIGURES:
        shutil.copy(ROOT / "reports" / "figures" / f, figdir / f)

    df = load_clean(save=False)
    cal = calendar_data(df)
    default_year = int(df.groupby("year")[TARGET].sum().idxmax())

    html = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
    for k, v in key_numbers(df).items():
        html = html.replace("{{" + k + "}}", v)
    html = (html.replace("__DAYS__", json.dumps(cal, ensure_ascii=False, separators=(",", ":")))
                .replace("__RESULTS__", json.dumps(results_data(), ensure_ascii=False))
                .replace("__DEFAULT_YEAR__", str(default_year))
                .replace("__REPO_FALLBACK__", json.dumps(repo)))
    assert "{{" not in html, "unfilled placeholder in template"
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    print(f"Site écrit dans docs/index.html ({len(html) // 1024} Ko)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/YuuMii05/predicting-forest-fire-machine-learning")
    main(ap.parse_args().repo)
