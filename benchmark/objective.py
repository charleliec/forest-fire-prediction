"""Objective: what a submission is evaluated on, and how.

Task: every day, rank the 25 km cells of metropolitan France by their risk
of a wildfire starting that day, to pre-position firefighting resources.
One row is a (cell, day); the label ``fire`` says whether at least one fire
alert was recorded there that day (BDIFF). Scored by the metrics of
``benchmark_utils/metric.py`` on the model's scores:
``model.predict_proba(X)[:, 1]``, or ``model.predict(X)`` for a model
without ``predict_proba`` (a regressor of the fire count, for instance).

The objective cross-validates the solver on the training years, one fold per
repetition (run with ``n_repetitions`` equal to ``N_SPLITS``), with whole
years held out (``GroupKFold`` on the year): days of the same season are
correlated, so a random split would leak. The solver trains on the fold's
training years; ``evaluate_result`` scores it on the held-out years
(``objective_<metric>_cv``) and on the public test year
(``objective_<metric>_public``).

The private test year comes without labels: ``evaluate_result`` returns the
fold model's scores on it (``objective_private_pred``), and the scoring
program scores them against the labels of the phase ``reference_data``. The
scoring program averages every score over the folds.
"""

import numpy as np
from benchopt import BaseObjective
from sklearn.model_selection import GroupKFold

from benchmark_utils.metric import METRICS

N_SPLITS = 5


def predict_score(model, X):
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = model.predict(X)
    score = np.asarray(score, dtype="float64").ravel()
    if score.shape != (len(X),):
        raise ValueError(f"expected {len(X)} scores, got {score.shape}")
    return score


class Objective(BaseObjective):

    name = "wildfire-daily-risk"
    url = "https://github.com/charleliec/forest-fire-prediction"

    requirements = ["scikit-learn", "pandas"]

    min_benchopt_version = "1.10.0"

    # Each solver trains once per fold (no convergence curve).
    sampling_strategy = "run_once"

    def set_data(self, X, y, X_public, y_public, X_private):
        self.X, self.y = X, y
        self.X_public, self.y_public = X_public, y_public
        self.X_private = X_private
        # Hold out whole years: no shuffling, deterministic folds.
        self.cv = GroupKFold(n_splits=N_SPLITS)
        self.cv_metadata = {"groups": X["date"].dt.year.to_numpy()}

    def split(self, cv_fold, X, y):
        train, val = cv_fold
        return X.iloc[train], X.iloc[val], y.iloc[train], y.iloc[val]

    def get_objective(self):
        # The fold follows the repetition index.
        X_train, self.X_val, y_train, self.y_val = self.get_split(self.X,
                                                                  self.y)
        return dict(X_train=X_train, y_train=y_train)

    def _scores(self, model, X, y, suffix):
        s = predict_score(model, X)
        return {f"{name}_{suffix}": metric(y, s, X["date"])
                for name, metric in METRICS.items()}

    def evaluate_result(self, model):
        return {
            **self._scores(model, self.X_val, self.y_val, "cv"),
            **self._scores(model, self.X_public, self.y_public, "public"),
            # Scored by the scoring program against the hidden labels.
            "private_pred": predict_score(model, self.X_private),
        }

    def get_one_result(self):
        # A trained trivial model, to validate the metric in ``benchopt test``.
        from sklearn.dummy import DummyClassifier
        return dict(model=DummyClassifier().fit(self.X, self.y))
