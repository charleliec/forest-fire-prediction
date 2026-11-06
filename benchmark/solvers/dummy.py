"""Floor baseline: the same risk for every cell and every day.

Its daily top-K capture is K / n_cells, its average precision the fire
rate. Any real model should beat it. Keep one such floor in every
competition.
"""

from sklearn.dummy import DummyClassifier

from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):

    name = "Dummy"

    requirements = ["scikit-learn"]

    def get_model(self, meta):
        return DummyClassifier(strategy="prior")
