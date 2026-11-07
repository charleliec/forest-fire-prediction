"""Objective: what a submission is evaluated on, and how.

Task: every day, rank the 25 km cells of metropolitan France by their risk
of a wildfire starting that day, to pre-position firefighting resources.
One row is a (cell, day); the label ``fire`` says whether at least one fire
alert was recorded there that day (BDIFF). Scored by the metrics of
``benchmark_utils/metric.py`` on the model's scores:
``model.predict_proba(X)[:, 1]``, or ``model.predict(X)`` for a model
without ``predict_proba`` (a regressor of the fire count, for instance).

By default (``n_splits=0``, what the platform runs) the solver trains once
on all the training years; ``evaluate_result`` scores it on the public test
year (``objective_<metric>_public``). The private test year comes without
labels: ``evaluate_result`` returns the model's scores on it
(``objective_private_pred``), and the scoring program scores them against
the labels of the phase ``reference_data``.

Optional cross-validation, for local model selection only: with
``n_splits=k`` the solver trains on the training years minus one fold per
repetition, whole years held out (``GroupKFold`` on the year, since days of
the same season are correlated), and is also scored on the held-out years
(``objective_<metric>_cv``). Run it with as many repetitions as folds::

    benchopt run benchmark --config local_config.yml \\
        -o "wildfire-daily-risk[n_splits=5]" -r 5
"""

import numpy as np
from benchopt import BaseObjective
from sklearn.model_selection import GroupKFold

from benchmark_utils.metric import METRICS


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

    # 0: train once on all training years (the platform). k > 0: k-fold CV
    # over the years, one fold per repetition (local use, run with -r k).
    parameters = {"n_splits": [0]}

    # Each solver trains once per run (no convergence curve).
    sampling_strategy = "run_once"

    def set_data(self, X, y, X_public, y_public, X_private, outcomes,
                 outcomes_public):
        # ``outcomes*``: fire count and burned area of each row (n_fires,
        # area_m2). Unused by the current metrics; kept (and split with y
        # under CV) so that a new objective (count regression, area
        # weighting) can use them.
        self.X, self.y, self.outcomes = X, y, outcomes
        self.X_public, self.y_public = X_public, y_public
        self.outcomes_public = outcomes_public
        self.X_private = X_private
        if self.n_splits:
            # Hold out whole years: no shuffling, deterministic folds.
            self.cv = GroupKFold(n_splits=self.n_splits)
            self.cv_metadata = {"groups": X["date"].dt.year.to_numpy()}

    def split(self, cv_fold, X, y):
        train, val = cv_fold
        return X.iloc[train], X.iloc[val], y.iloc[train], y.iloc[val]

    def get_objective(self):
        if not self.n_splits:
            self.X_val = None
            return dict(X_train=self.X, y_train=self.y)
        # The fold follows the repetition index. y and the outcomes are
        # split together, as one label frame.
        target = self.y.name
        labels = self.outcomes.assign(**{target: self.y})
        X_train, self.X_val, labels_train, labels_val = self.get_split(
            self.X, labels)
        self.y_val = labels_val.pop(target)
        self.outcomes_val = labels_val
        return dict(X_train=X_train, y_train=labels_train[target])

    def _scores(self, model, X, y, suffix):
        s = predict_score(model, X)
        return {f"{name}_{suffix}": metric(y, s, X["date"])
                for name, metric in METRICS.items()}

    def evaluate_result(self, model):
        result = {}
        if self.X_val is not None:
            result.update(self._scores(model, self.X_val, self.y_val, "cv"))
        return {
            **result,
            **self._scores(model, self.X_public, self.y_public, "public"),
            # Scored by the scoring program against the hidden labels.
            "private_pred": predict_score(model, self.X_private),
        }

    def get_one_result(self):
        # A trained trivial model, to validate the metric in ``benchopt test``.
        from sklearn.dummy import DummyClassifier
        return dict(model=DummyClassifier().fit(self.X, self.y))
