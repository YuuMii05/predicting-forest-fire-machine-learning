"""Evaluation protocol.

Nested cross-validation, with folds grouped BY YEAR:
  - the OUTER loop (5 folds x 3 repetitions = 15 test folds) measures
    performance; all the days of a given summer are in the same fold;
  - the INNER loop (3 folds, also grouped by year, inside each outer training
    set) chooses the hyper-parameters with GridSearchCV.
Hyper-parameters are never chosen by looking at the test fold.

Why group by year? A large fire burns for several days and the drought codes
change very little from one day to the next. With shuffled folds, the day
after a fire can be in the test set while the fire day itself is in the
training set: the model "recognises" it instead of predicting it (temporal
leakage). Grouping by year makes every test summer completely unseen.

Main metric: average precision (area under the precision-recall curve).
Fire days are only ~10 % of the data, so accuracy is misleading (always
answering "no fire" scores 90 %). Average precision measures how well a model
ranks the risky days first, whatever the decision threshold; a random model
scores the fire rate (~0.10).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold, RepeatedStratifiedKFold, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict, cross_validate)

from .models import RANDOM_STATE, ModelSpec

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
}
MAIN_METRIC = "average_precision"
N_SPLITS, N_REPEATS = 5, 3


def outer_splits(y, groups, n_repeats: int = N_REPEATS):
    """Year-grouped folds, repeated with different shuffles of the years."""
    splits = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
        splits += list(cv.split(np.zeros(len(y)), y, groups))
    return splits


def shuffled_splits(y, n_repeats: int = 1):
    """Classic shuffled folds (NOT grouped) - only used to show the leakage."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=RANDOM_STATE)
    return list(cv.split(np.zeros(len(y)), y))


def tuned(spec: ModelSpec, scaler: str | None = None):
    """Model wrapped in its inner hyper-parameter search (if it has any).
    The inner folds are grouped by year too (groups passed at fit time)."""
    pipe = spec.pipeline(scaler)
    grid = spec.param_grid()
    if not grid:
        return pipe
    return GridSearchCV(pipe, grid, scoring=MAIN_METRIC, cv=GroupKFold(n_splits=3), n_jobs=-1)


def _fit_params(spec: ModelSpec, groups):
    return {"groups": groups} if spec.param_grid() else {}


def _X(spec: ModelSpec, X: pd.DataFrame) -> pd.DataFrame:
    return X[spec.features] if spec.features else X


def nested_cv(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, groups: pd.Series, scaler: str | None = None,
              n_repeats: int = N_REPEATS, shuffled: bool = False) -> pd.DataFrame:
    """One row per outer fold: every metric + fit time + chosen hyper-parameters.
    groups = the year of each row (folds never split a summer)."""
    cv = shuffled_splits(y, n_repeats) if shuffled else outer_splits(y, groups, n_repeats)
    res = cross_validate(tuned(spec, scaler), _X(spec, X), y, cv=cv, scoring=SCORING,
                         return_estimator=True, n_jobs=1, params=_fit_params(spec, groups))
    rows = pd.DataFrame({m: res[f"test_{m}"] for m in SCORING})
    rows["fit_time"] = res["fit_time"]
    rows["fold"] = range(len(rows))
    rows["best_params"] = [
        str({k.replace("clf__", ""): v for k, v in est.best_params_.items()})
        if hasattr(est, "best_params_") else "-"
        for est in res["estimator"]
    ]
    rows.insert(0, "model", spec.name)
    rows.insert(1, "family", spec.family)
    return rows


def summarize(folds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by or ["model", "family"]
    agg = folds.groupby(by, sort=False)[list(SCORING) + ["fit_time"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def corrected_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int):
    """Nadeau & Bengio (2003) corrected resampled t-test.

    The folds of a repeated k-fold share training data, so their scores are not
    independent and a plain t-test is too optimistic. The correction inflates
    the variance by n_test / n_train.
    """
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def _score_values(model, X):
    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except AttributeError:
            pass
    return model.predict_proba(X)[:, 1]


def scores(y_true, y_pred, y_score) -> dict:
    return {
        "average_precision": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "fire_rate": float(np.mean(y_true)),
    }


def fit_and_test(spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame, features, target) -> dict:
    """Tune on `train` (inner CV inside it), evaluate once on `test`."""
    model = clone(tuned(spec)).fit(_X(spec, train[features]), train[target], **_fit_params(spec, train["year"]))
    Xt = _X(spec, test[features])
    return scores(test[target], model.predict(Xt), _score_values(model, Xt))


def leave_one_zone_out(spec: ModelSpec, df: pd.DataFrame, features, target) -> list[dict]:
    """Train on 6 forest zones, test on the 7th one, for each zone in turn."""
    rows = []
    for zone in df["zone"].unique():
        sc = fit_and_test(spec, df[df.zone != zone], df[df.zone == zone], features, target)
        rows.append({"model": spec.name, "test_zone": zone, **sc})
    return rows


def temporal_split(spec: ModelSpec, df: pd.DataFrame, features, target, first_test_year: int) -> dict:
    """Train on the past, test on the most recent summers."""
    train, test = df[df.year < first_test_year], df[df.year >= first_test_year]
    return {"model": spec.name, **fit_and_test(spec, train, test, features, target)}


def oof_confusion(spec: ModelSpec, X, y, groups) -> np.ndarray:
    """Confusion matrix of out-of-fold predictions (default 0.5 threshold)."""
    cv = outer_splits(y, groups, n_repeats=1)
    pred = cross_val_predict(tuned(spec), _X(spec, X), y, cv=cv, n_jobs=1, params=_fit_params(spec, groups))
    return confusion_matrix(y, pred, labels=[0, 1])


def knn_k_curve(X, y, groups, ks=(1, 3, 5, 9, 15, 25, 41, 61, 81, 101, 151, 201),
                scalers=("aucune", "standard")) -> pd.DataFrame:
    """CV score of k-NN as a function of k, with and without scaling."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from .models import SCALERS

    rows = []
    cv = outer_splits(y, groups, n_repeats=1)
    for sc in scalers:
        for k in ks:
            step = SCALERS[sc] if SCALERS[sc] == "passthrough" else clone(SCALERS[sc])
            pipe = Pipeline([("scaler", step), ("clf", KNeighborsClassifier(n_neighbors=k))])
            s = cross_validate(pipe, X, y, cv=cv, scoring=MAIN_METRIC, n_jobs=-1)["test_score"]
            rows.append({"scaler": sc, "k": k, "mean": s.mean(), "std": s.std()})
    return pd.DataFrame(rows)


def cv_permutation_importance(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                              n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance measured on held-out folds, averaged over folds."""
    imps = []
    for tr, te in outer_splits(y, groups, n_repeats=1):
        model = clone(tuned(spec)).fit(X.iloc[tr], y.iloc[tr], **_fit_params(spec, groups.iloc[tr]))
        r = permutation_importance(model, X.iloc[te], y.iloc[te], scoring=MAIN_METRIC,
                                   n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1)
        imps.append(r.importances_mean)
    imps = np.array(imps)
    return (pd.DataFrame({"feature": X.columns, "importance": imps.mean(0), "std": imps.std(0)})
            .sort_values("importance", ascending=False).reset_index(drop=True))


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t
