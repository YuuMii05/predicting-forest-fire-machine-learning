"""Run every experiment on the Tunisian dataset and save the raw results to reports/results/.

    python run_experiments.py            # everything (~20-30 min on a laptop)
    python run_experiments.py --quick    # fewer CV repetitions, for a fast check

Then:  python make_figures.py  and  python make_site.py
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd

from src.data import ALL_FEATURES, TARGET, WEATHER_FEATURES, load_clean
from src.evaluation import (N_SPLITS, Timer, corrected_ttest, cv_permutation_importance, knn_k_curve,
                            leave_one_zone_out, nested_cv, oof_confusion, summarize, temporal_split)
from src.models import FWI_ALONE, get_models

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "reports" / "results"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_SETS = {
    "meteo+FWI": ALL_FEATURES,         # weather + the 6 Fire Weather Index components
    "meteo seule": WEATHER_FEATURES,   # only the 4 weather readings of the day
}
FIRST_TEST_YEAR = 2022                 # temporal test: train 2013-2021, test 2022-2025


def log(msg):
    print(msg, flush=True)


def main(quick: bool = False):
    n_rep = 1 if quick else 3
    df = load_clean()
    y = df[TARGET]
    years = df["year"]
    models = get_models()
    log(f"{len(df)} jours-zones, {y.sum()} avec feu ({y.mean():.1%})\n")

    # 1 + 2. Main comparison (nested CV) on both feature sets
    all_folds = []
    for fs_name, feats in FEATURE_SETS.items():
        X = df[feats]
        for spec in models:
            with Timer() as t:
                f = nested_cv(spec, X, y, years, n_repeats=n_rep)
            f["features"] = fs_name
            all_folds.append(f)
            log(f"[{fs_name:11}] {spec.name:28} AP = {f.average_precision.mean():.3f} ± "
                f"{f.average_precision.std():.3f}   ROC-AUC = {f.roc_auc.mean():.3f}  ({t.elapsed:.0f}s)")
    ref = nested_cv(FWI_ALONE, df[ALL_FEATURES], y, years, n_repeats=n_rep)
    ref["features"] = "FWI seul"
    all_folds.append(ref)
    log(f"[FWI seul   ] {FWI_ALONE.name:28} AP = {ref.average_precision.mean():.3f}   "
        f"ROC-AUC = {ref.roc_auc.mean():.3f}")

    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(OUT / "cv_folds.csv", index=False)
    summarize(folds, ["features", "model", "family"]).to_csv(OUT / "cv_summary.csv", index=False)
    (folds.groupby(["features", "model"])["best_params"]
          .agg(lambda s: s.value_counts().index[0]).reset_index()
          .to_csv(OUT / "best_params.csv", index=False))

    # 3. Is the winner significantly better? (corrected resampled t-test on average precision)
    n = len(df)
    n_test = n // N_SPLITS
    tests = []
    for fs_name in FEATURE_SETS:
        sub = folds[folds.features == fs_name]
        means = sub.groupby("model").average_precision.mean().sort_values(ascending=False)
        best = means.index[0]
        a = sub[sub.model == best].sort_values("fold").average_precision.values
        others = list(means.index[1:])
        for other in others:
            b = sub[sub.model == other].sort_values("fold").average_precision.values
            t, p = corrected_ttest(a, b, n - n_test, n_test)
            tests.append({"features": fs_name, "best": best, "other": other,
                          "delta": a.mean() - b.mean(), "t": t, "p_value": p})
        # the official index alone, compared with the best model
        b = ref.sort_values("fold").average_precision.values
        t, p = corrected_ttest(a, b, n - n_test, n_test)
        tests.append({"features": fs_name, "best": best, "other": FWI_ALONE.name,
                      "delta": a.mean() - b.mean(), "t": t, "p_value": p})
    pd.DataFrame(tests).to_csv(OUT / "significance_tests.csv", index=False)
    log("\nTests de significativité enregistrés.")

    # 3b. Temporal leakage: same models, classic SHUFFLED folds instead of year-grouped folds
    rows = []
    for spec in models:
        f = nested_cv(spec, df[ALL_FEATURES], y, years, n_repeats=1, shuffled=True)
        rows.append({"model": spec.name, "ap_shuffled": f.average_precision.mean(),
                     "roc_auc_shuffled": f.roc_auc.mean()})
    pd.DataFrame(rows).to_csv(OUT / "leakage_shuffled_cv.csv", index=False)
    log("[fuite] CV mélangée enregistrée")

    # 4. Does scaling matter? Every model x every scaler
    rows = []
    for fs_name, feats in FEATURE_SETS.items():
        for spec in models[1:]:
            for scaler in ["aucune", "standard", "minmax"]:
                f = nested_cv(spec, df[feats], y, years, scaler=scaler, n_repeats=1)
                rows.append({"features": fs_name, "model": spec.name, "scaler": scaler,
                             "ap_mean": f.average_precision.mean(), "ap_std": f.average_precision.std(),
                             "roc_auc_mean": f.roc_auc.mean()})
            log(f"[mise à l'échelle] [{fs_name}] {spec.name}")
    pd.DataFrame(rows).to_csv(OUT / "scaling.csv", index=False)

    # 5. k-NN: score as a function of k, with and without scaling
    curves = []
    for fs_name, feats in FEATURE_SETS.items():
        c = knn_k_curve(df[feats], y, years)
        c.insert(0, "features", fs_name)
        curves.append(c)
    pd.concat(curves).to_csv(OUT / "knn_k_curve.csv", index=False)
    log("[k-NN] courbe en fonction de k enregistrée")

    # 6. Geographic generalisation: leave one forest zone out
    rows = []
    for fs_name, feats in FEATURE_SETS.items():
        for spec in models + [FWI_ALONE]:
            if spec is FWI_ALONE and fs_name != "meteo+FWI":
                continue
            for r in leave_one_zone_out(spec, df, feats, TARGET):
                rows.append({"features": fs_name, **r})
        log(f"[zones] {fs_name} terminé")
    pd.DataFrame(rows).to_csv(OUT / "leave_one_zone_out.csv", index=False)

    # 7. Temporal generalisation: train on the past, test on recent summers
    rows = []
    for fs_name, feats in FEATURE_SETS.items():
        for spec in models + [FWI_ALONE]:
            if spec is FWI_ALONE and fs_name != "meteo+FWI":
                continue
            rows.append({"features": fs_name, **temporal_split(spec, df, feats, TARGET, FIRST_TEST_YEAR)})
    pd.DataFrame(rows).to_csv(OUT / "temporal_split.csv", index=False)
    log("[temps] entraînement 2013-2021, test 2022-2025 enregistré")

    # 8. Confusion matrices (out-of-fold predictions, 0.5 threshold)
    cms = {}
    for fs_name, feats in FEATURE_SETS.items():
        for spec in models:
            cms[f"{fs_name}|{spec.name}"] = oof_confusion(spec, df[feats], y, years).tolist()
    (OUT / "confusion_matrices.json").write_text(json.dumps(cms, ensure_ascii=False, indent=1))
    log("[confusion] matrices enregistrées")

    # 9. Which variables matter? (Random Forest, permutation importance on held-out folds)
    rf = next(m for m in models if m.name == "Random Forest")
    for fs_name, feats in FEATURE_SETS.items():
        imp = cv_permutation_importance(rf, df[feats], y, years)
        imp.to_csv(OUT / f"importance_{'all' if fs_name == 'meteo+FWI' else 'weather'}.csv", index=False)
    log("[importance] enregistrée\n\nTerminé.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
