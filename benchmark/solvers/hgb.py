"""Sample submission: gradient boosting on the given features plus the
season, the only feature engineering being calendar features from ``date``.

To submit it, copy this file as ``submission.py`` in a folder and zip the
folder (``tools/create_bundle.py`` ships it as the sample solution).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from benchmark_utils.base_solver import CompetSolver


def add_calendar(X):
    X = X.copy()
    date = X.pop("date")
    doy = date.dt.dayofyear.to_numpy()
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["weekday"] = date.dt.weekday.to_numpy()
    return X


class Solver(CompetSolver):

    name = "HGB"

    requirements = ["scikit-learn"]

    def get_model(self, meta):
        return make_pipeline(
            FunctionTransformer(add_calendar),
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                early_stopping=True, random_state=0),
        )
