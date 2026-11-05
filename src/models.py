"""The competing models: two references + six algorithm families.

Every model is a scikit-learn Pipeline, so preprocessing (scaling) is learned
on the training folds only, never on the test fold: no data leakage.
Each model comes with the hyper-parameter grid searched by the inner
cross-validation.

Fire days are rare (about 10 %), so models that support it use
class_weight="balanced": errors on the rare class (fire) weigh more.
k-NN has no such option; that is part of the comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

SCALERS = {
    "aucune": "passthrough",
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
}


@dataclass
class ModelSpec:
    name: str                 # display name used in tables and figures
    family: str               # algorithm family
    estimator: object         # the bare scikit-learn estimator
    grid: dict = field(default_factory=dict)   # hyper-parameters (without the "clf__" prefix)
    needs_scaling: bool = True                 # does the model rely on distances / magnitudes?
    features: list | None = None               # restrict to these columns (None = all given)

    def pipeline(self, scaler: str | None = None) -> Pipeline:
        """scaler -> classifier. By default, scale only the models that need it."""
        if scaler is None:
            scaler = "standard" if self.needs_scaling else "aucune"
        step = SCALERS[scaler]
        if step != "passthrough":
            step = clone(step)
        return Pipeline([("scaler", step), ("clf", clone(self.estimator))])

    def param_grid(self) -> dict:
        return {f"clf__{k}": v for k, v in self.grid.items()}


def get_models() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="Baseline (au hasard)",
            family="Référence",
            estimator=DummyClassifier(strategy="prior"),
            needs_scaling=False,
        ),
        ModelSpec(
            name="Régression logistique",
            family="Linéaire",
            estimator=LogisticRegression(max_iter=5000, class_weight="balanced"),
            grid={"C": [0.01, 0.1, 1, 10]},
        ),
        ModelSpec(
            name="k-NN",
            family="Basé sur les instances",
            estimator=KNeighborsClassifier(),
            grid={"n_neighbors": [5, 15, 31, 61, 101], "weights": ["uniform", "distance"]},
        ),
        ModelSpec(
            name="Arbre de décision",
            family="Arbres",
            estimator=DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            grid={"max_depth": [3, 5, 8], "min_samples_leaf": [10, 50]},
            needs_scaling=False,
        ),
        ModelSpec(
            name="Random Forest",
            family="Ensemble (bagging)",
            estimator=RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                             random_state=RANDOM_STATE, n_jobs=-1),
            grid={"max_depth": [6, 12], "min_samples_leaf": [5, 20]},
            needs_scaling=False,
        ),
        ModelSpec(
            name="Gradient Boosting",
            family="Ensemble (boosting)",
            estimator=HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            grid={"learning_rate": [0.03, 0.1], "max_depth": [3, None]},
            needs_scaling=False,
        ),
        ModelSpec(
            name="SVM (noyau RBF)",
            family="Méthodes à noyau",
            estimator=SVC(kernel="rbf", class_weight="balanced"),
            grid={"C": [1, 10], "gamma": ["scale", 0.3]},
        ),
    ]


# The official index used alone as a risk score (a monotonic transform of FWI,
# so its ranking of days is exactly the FWI ranking)
FWI_ALONE = ModelSpec(
    name="Indice FWI officiel seul",
    family="Référence",
    estimator=LogisticRegression(class_weight="balanced"),
    features=["FWI"],
)


def get_model(name: str) -> ModelSpec:
    return next(m for m in get_models() if m.name == name)


# Fixed colour per model (colour follows the entity, never its rank)
MODEL_COLORS = {
    "Baseline (au hasard)": "#8a8984",
    "Indice FWI officiel seul": "#52514e",
    "Régression logistique": "#2a78d6",
    "k-NN": "#eb6834",
    "Arbre de décision": "#1baf7a",
    "Random Forest": "#eda100",
    "Gradient Boosting": "#e87ba4",
    "SVM (noyau RBF)": "#6250d6",
}
