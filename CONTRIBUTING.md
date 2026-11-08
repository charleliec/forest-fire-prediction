# Working on the model

The goal: **each day, rank the 977 cells of France by wildfire risk.** A
model is scored on how many fires fall in its daily top 50 cells
(`capture`), and on average precision (`ap`). See [DATA.md](DATA.md) for the
columns.

## Setup

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

### 2. Data

The data is not in git. It is published as a GitHub Release:
**[data-v1](https://github.com/charleliec/forest-fire-prediction/releases/tag/data-v1)**,
one file `firegrid.parquet` (281 MB, all years 2015-2025). Download it into
`data/processed/`:

```bash
mkdir -p data/processed    # PowerShell: mkdir data/processed
curl.exe -L -o data/processed/firegrid.parquet https://github.com/charleliec/forest-fire-prediction/releases/download/data-v1/firegrid.parquet
```

(`curl.exe` works in PowerShell, cmd and bash; in Windows PowerShell, plain
`curl` is a different command.) Check the download:

```bash
sha256sum data/processed/firegrid.parquet               # bash
Get-FileHash data/processed/firegrid.parquet            # PowerShell
# b6f8b259487096d8ac407970b8a45fefcbe30bceaf893f191175c9f5596a9abc
```

Then build the splits:

```bash
benchopt prepare benchmark --config local_config.yml --force
```

This writes `benchmark/data/real/{train,public,private}.parquet` (train
2015-2017 + 2019-2024, public 2018, private 2025 without labels).

**The file holds the 2025 labels.** Never train or tune on 2025, and don't
share the file outside the team.

**New data versions** are new releases (`data-v2`, ...), listed on the
[Releases page](https://github.com/charleliec/forest-fire-prediction/releases).
When one comes out, download it the same way (replace `data-v1` in the
URL), then re-run `prepare` with `--force`. Write the data version in your
PR next to your scores.

## Where things are

| Path | What you do with it |
|---|---|
| `benchmark/solvers/` | **Your models go here.** `hgb.py` is the baseline to beat, `dummy.py` the floor |
| `benchmark/benchmark_utils/metric.py` | The metrics (read, don't edit) |
| `benchmark/objective.py` | training, optional CV and scoring (read, don't edit) |
| `benchmark/datasets/real.py` | Train / public / private split |
| `pipeline/` | Builds the data table from raw sources. You don't need to run it |
| `template_starting_kit.ipynb` | Notebook for participants (EDA and baseline) |

## Writing a model

### 1. Create the file

Create `benchmark/solvers/<your_model>.py` and paste the template below.
Give it a unique `name`: `-s` selects it by this name, and the results
show it. A model is one file, holding the `Solver` class and every helper
it needs.

### 2. The template

The model is a plain Python class with two methods, `fit` and
`predict_proba`. Both build the features with the same `features` method,
then call the classifier: the chain is written out, step by step.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from benchmark_utils.base_solver import CompetSolver


class Model:

    def __init__(self):
        self.clf = HistGradientBoostingClassifier(random_state=0)

    def features(self, X):
        """Raw X -> table the classifier sees. Same code at train and test."""
        X = X.copy()

        # Calendar, from the row only. Then drop `date`: the
        # classifier can't use a datetime column.
        doy = X["date"].dt.day_of_year.to_numpy()
        X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        X = X.drop(columns="date")

        # Per-cell fire rate, learned in `fit` (training labels only).
        X["cell_rate"] = X["cell_id"].map(self.cell_rate).fillna(self.base_rate)
        return X

    def fit(self, X, y):
        # 1. Learn what the features need from the training data.
        self.cell_rate = y.groupby(X["cell_id"]).mean()
        self.base_rate = y.mean()
        # 2. Build the features, then train the classifier on them.
        self.clf.fit(self.features(X), y)

    def predict_proba(self, X):
        # Same features, then the trained classifier.
        return self.clf.predict_proba(self.features(X))


class Solver(CompetSolver):
    name = "MyModel"
    requirements = ["scikit-learn"]

    def get_model(self, meta):
        return Model()   # untrained, one per run
```

What happens on the platform (once, on all the training years):

1. `Solver.get_model(meta)` creates a fresh, untrained `Model()`.
2. `Model.fit(X_train, y_train)` learns the per-cell rates, builds the
   features and trains the classifier.
3. `Model.predict_proba(X)` is called on the public year and the private
   year (and on the held-out years under local CV). Column 1 is the risk
   score.

Notes:

- Anything learned from `y` (per-cell rates, target encoding) must be
  computed in `fit` and stored on `self`, as `cell_rate` is. `fit` only sees
  the training years (the training fold under CV), so it does not leak.
- Sample weights, negative subsampling and torch training loops go in
  `Model.fit` too. For example, `self.clf.fit(F, y, sample_weight=w)`.
- `meta["submission_dir"]` is the folder of the file, for loading extra
  non-code files (weights, configs) in `get_model`.
- When you run locally, benchopt prints a joblib `Unable to cache to disk`
  warning for classes defined in the file. It does no harm: ignore it.
- `hgb.py` builds the same kind of model with sklearn's `make_pipeline`.
  Both styles work, use whichever you find clearer.

### 3. The contract

- **Input.** `X` is the raw DataFrame of [DATA.md](DATA.md), `date`
  included (datetime). At test time the model gets raw `X` only, so **all
  feature engineering lives inside the model**, never in a preprocessing
  step outside it.
- **Output.** The trained model exposes `predict_proba(X)[:, 1]` or
  `predict(X)`, one finite score per row (a NaN score fails the run). Any
  monotone score works: `capture` only compares cells of the same day.
- **One file.** The platform loads it by path: helpers must be defined in
  it, sibling `.py` files can't be imported.
- **Columns.** Only use columns that exist in your `train.parquet`. The
  columns under "Being added" in [DATA.md](DATA.md) are in the `Simulated`
  data but not yet in every real file.
- **Packages.** Only `requirements.txt` (benchopt, pandas, scikit-learn,
  plus torch in the image). Want LightGBM / XGBoost? Ask first, it means
  changing the Docker image.
- **Time.** The platform trains the model once (~3.2 M rows) and predicts,
  within **1 h in total** on CPU. Check the `train_time` of your local run.
- **Reproducibility.** Set `random_state` / seeds: the organizers may re-run
  a submission.

## Running it

```bash
# quick smoke test on synthetic data (seconds)
benchopt run benchmark --config local_config.yml -d Simulated -s MyModel

# what the platform runs: one training on all training years, compared to
# the baselines
benchopt run benchmark --config local_config.yml -s Dummy -s HGB -s MyModel

# model selection: 5-fold CV over the training years (5 trainings)
benchopt run benchmark --config local_config.yml -s Dummy -s HGB -s MyModel     -o "wildfire-daily-risk[n_splits=5]" -r 5
```

Results open in the browser and are saved in `benchmark/outputs/`. Read the
columns:

- `objective_capture_cv` (CV run only): **the one to optimize.** 5-fold CV,
  whole years held out, averaged over the folds.
- `objective_capture_public`: the 2018 test year. A single year, so noisy.
  Don't tune on it.
- `objective_ap_*`: secondary metric.

The floor (`Dummy`) scores `capture` ≈ 0.051 (50 / 977).

For faster iteration, work in a notebook and import the metric directly:

```python
import sys; sys.path.insert(0, "benchmark")
import pandas as pd
from benchmark_utils.metric import capture, ap

train = pd.read_parquet("benchmark/data/real/train.parquet")
# hold out a whole year, e.g. 2022
val = train["date"].dt.year == 2022
...
print(capture(y_val, scores, X_val["date"]))
```

## Avoiding leakage

- **Never split rows at random.** Days of the same season are correlated.
  Always hold out whole years (as the objective does).
- **No features from fire labels of the predicted period.** For example,
  "fires in the cell yesterday" is not available on the 2025 test year.
  Statistics learned from the *training* labels inside `fit` (per-cell fire
  rate, target encoding) are fine.
- Same-day features from other cells (neighbour weather, spatial
  smoothing) are fine. Future weather (tomorrow's rain) is not.

## Ideas worth trying

- Calendar: day of year (sin/cos), weekday, holidays, summer flag.
- Per-cell history: fire rate of the cell in the training data, overall and
  per month.
- Spatial context: weather of neighbouring cells, spatially smoothed risk.
- Fire-weather indices built from the raw weather (e.g. dryness ×
  temperature × wind, a simplified Canadian FWI).
- Imbalance (0.6% positives): class weights, negative subsampling of winter
  days.
- Ranking-aware objectives: `capture` only compares cells of the same day.

## Git

- One branch per idea (`model/<name>`), small PRs to `main`.
- Never commit data (`data/`, `benchmark/data/`, `outputs/` are ignored).
- In the PR, paste the `capture_cv` / `capture_public` of your model next
  to `HGB`.
