"""The competition metrics, the single source of truth for the objective and
the scoring program (``tools/create_bundle.py`` ships ``benchmark_utils``
with the scoring program).

The use case is the daily pre-positioning of firefighting resources: every
day, a fixed number ``K`` of 25 km cells can be reinforced, and the best
model is the one whose daily top-``K`` cells contain the most fires.

- ``capture`` (ranking metric): the share of fire (cell, day) rows that fall
  in the ``K`` highest-scored cells of their day. Ties are broken at random
  in expectation, so a constant prediction scores ``K / n_cells`` rather
  than whatever the row order gives. Higher is better.
- ``ap``: average precision over all (cell, day) rows, a threshold-free
  summary of how well fire rows are ranked across days and cells. Higher is
  better.

Both take ``(y_true, y_score, day)``: the 0/1 fire label, the predicted
score (any monotone transform of a probability works) and the day of each
row.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Cells that can be reinforced each day (out of 977, about 5.1% of the
# territory).
K = 50


def capture(y_true, y_score, day, k=K):
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score),
                       "day": np.asarray(day)})
    if df["s"].isna().any():
        raise ValueError("predicted scores contain NaN")
    n_pos = df["y"].sum()
    if n_pos == 0:
        raise ValueError("no fire in the evaluated rows")
    pos = df["y"] == 1
    # Rows of the same day scored strictly higher, and equally.
    n_greater = df.groupby("day")["s"].rank(
        method="min", ascending=False) - 1
    n_equal = df.groupby(["day", "s"])["s"].transform("size")
    share = ((k - n_greater) / n_equal).clip(0, 1)
    return float(share[pos].sum() / n_pos)


def ap(y_true, y_score, day=None):
    return float(average_precision_score(np.asarray(y_true),
                                         np.asarray(y_score)))


METRICS = {"capture": capture, "ap": ap}
