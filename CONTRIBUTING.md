# Working on the model

The goal: **each day, rank the 977 cells of France by wildfire risk.** A
model is scored on how many fires fall in its daily top 50 cells
(`capture`), and on average precision (`ap`). See [DATA.md](DATA.md) for the
columns.

## Where things are

| Path | What you do with it |
|---|---|
| `benchmark/solvers/` | **Your models go here.** `hgb.py` is the baseline to beat, `dummy.py` the floor |
| `benchmark/benchmark_utils/metric.py` | The metrics (read, don't edit) |
| `benchmark/objective.py` | CV scheme and scoring (read, don't edit) |
| `benchmark/datasets/real.py` | Train / public / private split |
| `pipeline/` | Builds the data table from raw sources. You don't need to run it |
| `template_starting_kit.ipynb` | Notebook for participants (EDA and baseline) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Data is not in git. Ask Charlelie for `firegrid.parquet`, put it in
`data/processed/`, then build the splits:

```bash
benchopt prepare benchmark --config local_config.yml --force
```

This writes `benchmark/data/real/{train,public,private}.parquet`. Re-run it
(with `--force`) whenever you get a new `firegrid.parquet`.

## Writing a model

Copy `benchmark/solvers/hgb.py` to `benchmark/solvers/<your_model>.py` and
change `name` and `get_model`:

```python
from benchmark_utils.base_solver import CompetSolver

class Solver(CompetSolver):
    name = "MyModel"
    requirements = ["scikit-learn"]

    def get_model(self, meta):
        return make_pipeline(...)   # untrained model

    # optional, default is model.fit(X_train, y_train)
    def fit(self, model, X_train, y_train):
        model.fit(X_train, y_train)
```

Rules of the contract:

- `X` is a pandas DataFrame with the raw columns of [DATA.md](DATA.md),
  including `date` (datetime). **All feature engineering must happen inside
  the model** (e.g. a `FunctionTransformer` in a pipeline). At test time it
  receives raw `X` only.
- The trained model must have `predict_proba(X)[:, 1]` or `predict(X)`.
  Any score works. Only the ranking of cells within a day matters for
  `capture`.
- Everything must be in **one file**: the platform cannot import sibling
  `.py` files.
- Only packages from `requirements.txt` are available (benchopt, pandas,
  scikit-learn, plus torch in the image). Want LightGBM / XGBoost? Ask first,
  it means changing the Docker image.

## Running it

```bash
# quick smoke test on synthetic data (seconds)
benchopt run benchmark --config local_config.yml -d Simulated -s MyModel

# real evaluation: 5 CV folds, compared to the baselines
benchopt run benchmark --config local_config.yml -s Dummy -s HGB -s MyModel
```

Results open in the browser and are saved in `benchmark/outputs/`. Read the
columns:

- `objective_capture_cv`: **the one to optimize.** 5-fold CV, whole years
  held out.
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
