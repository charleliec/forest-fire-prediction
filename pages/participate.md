# How to participate

Each section below is a short summary. The
[starting kit](TODO-link-to-the-starting-kit-notebook) gives the full argument,
with the data analysis behind each choice.

## The task

Every summer, French fire services pre-position crews, trucks and aircraft
where they expect fires to start. Your model helps decide where: **every day,
rank the 1,104 cells of 25 km × 25 km covering metropolitan France by their
risk of a wildfire starting that day.**

One row is a (cell, day). The label `fire` is 1 when at least one wildfire
alert was recorded in the cell that day in the French forest-fire database
[BDIFF](https://bdiff.agriculture.gouv.fr/) (fires of 10 m² or more, overseas
territories excluded), about 0.6% of the rows. The features are:

- the cell: `cell_id`, its centre `x`, `y` (Lambert-93, km) and `lon`, `lat`;
- the day: `date`;
- the weather of the day, from the Météo-France SIM2 reanalysis averaged over
  the cell: temperatures (`t_min`, `t_mean`, `t_max`), `rain`, `snow`,
  `wind`, humidity (`rh`, `q`), radiation, evapotranspiration (`etp`,
  `evap`), soil wetness (`swi`, `wg_root`), `snow_depth`;
- the weather of the previous days (never the day itself): rain over 3, 7
  and 30 days, `days_since_rain`, 7-day means of `t_max` and `rh`,
  `swi_prev`;
- the vegetation: the last MODIS MOD13Q1 NDVI composite (250 m, 16 days)
  completed before the day, averaged over the cell (`ndvi`), its anomaly
  with respect to the cell's usual NDVI at that time of year (`ndvi_anom`),
  and its age in days (`ndvi_age`).

## The metric

**Fires captured in the daily top 50** (higher is better): each day, the 50
cells your model scores highest (about 4.5% of the territory) are the ones
reinforced; the metric is the share of all fire (cell, day) rows of the test
year that fall in their day's top 50. Only the ranking of the cells *within
a day* matters. Ties are counted as broken at random: a constant prediction
scores 50 / 1,104 ≈ 0.045.

**Average precision** (higher is better) is shown as a second column: it
ranks all the (cell, day) rows of the year together, so it also rewards
telling quiet days from busy ones.

## The split

Whole years are held out, because days of the same season share their
weather, their vegetation and their fires.

- **Training data**: 2015-2017 and 2019-2024.
- **CV column**: 5-fold cross-validation on the training data, each fold
  holding out whole years (`GroupKFold` on the year).
- **Public column** (live feedback): the year 2018, a mild fire year.
- **Private / final ranking** (hidden): the year 2025.

Each fold's model predicts both test years, and every score is averaged over
the 5 folds. Select your models on the CV score rather than on the public
one: a single year is a noisy benchmark.

## What you submit

A single **`submission.py`** defining `class Solver(CompetSolver)`. All your
Python code must live in this one file: the platform loads it on its own, so
it cannot import other `.py` files you ship. You may add non-code files next
to it (model weights, configs) and read them from `meta["submission_dir"]`.

```python
from benchmark_utils.base_solver import CompetSolver


class Solver(CompetSolver):
    name = "submission"

    def get_model(self, meta):
        # build your (untrained) model
        ...

    def fit(self, model, X_train, y_train):
        # optional: train it (default: model.fit(X_train, y_train))
        ...
```

`X_train` is a pandas DataFrame with the columns above (`date` is a
datetime), `y_train` the 0/1 `fire` label. Your trained model must expose
`predict_proba(X)` (the second column is used as the risk score), or
`predict(X) -> ndarray (n_samples,)` returning a risk score, for instance a
predicted number of fires. The platform **trains your code**, once per fold,
then evaluates it, within the execution time limit. Copy
`benchmark/solvers/hgb.py`, a working gradient-boosting baseline, to
`my_submission/submission.py`.

Use only the columns you are given: the rows of the test years come
without their label, and any external record of their fires is forbidden
(see the terms).

## Run it locally, same code as the server

`local_config.yml` sets the dataset, the solvers, the seed and the 5 folds
of a local run. Add your `my_submission/submission.py` to its `solver` list,
then:

```bash
pip install -r requirements.txt
benchopt run benchmark --config local_config.yml
```

with the prepared `train.parquet`, `public.parquet` and `private.parquet`
(features only) of the starting kit in `benchmark/data/real/`. A command-line option overrides the file, e.g.
`-s my_submission/submission.py` runs your submission alone, and
`-d Simulated` runs on a small synthetic dataset.

The cross-validation and public scores you get locally (`objective_capture_cv`,
`objective_capture_public`, `objective_ap_cv`, `objective_ap_public`, averaged
over the 5 repetitions) are the CV and public columns of the leaderboard. The
benchmark you run *is* the benchmark the server runs.
