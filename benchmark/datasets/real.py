"""Real dataset: ``prepare`` builds the splits, ``get_data`` only loads them.

    benchopt prepare benchmark -d Real

One row per (25 km cell, day) over metropolitan France, 2015-2025; label
``fire``: at least one wildfire alert in the cell that day (BDIFF). The
features are the cell position, the date, the weather of the day and of the
previous days (Météo-France SIM2) and the last MODIS NDVI composite
completed before the day. See ``pipeline/`` for how the table is built.

Split by year: ``train`` = 2015-2017 and 2019-2024, ``public`` = 2018
(features + label, scored on the public leaderboard), ``private`` = 2025
(features only, ranks the competition). The private labels (and the day of
each row, needed by the metric) go to ``get_data_path("reference")``, which
the bundle ships in the phase ``reference_data``, read only by the scoring
program.

``prepare`` reads the private pipeline output ``firegrid.parquet``
(``$FIREGRID_SOURCE``, default ``data/processed/firegrid.parquet`` at the
repository root). It is not distributed: participants get the prepared
``train`` and ``public`` files.
"""

import os
from pathlib import Path

import pandas as pd
from benchopt import BaseDataset
from benchopt.config import get_data_path

PUBLIC_YEAR, PRIVATE_YEAR = 2018, 2025
TARGET = "fire"
# Outcomes of the fire, not available before it: never features.
OUTCOMES = ["n_fires", "area_m2"]


def source_path():
    default = (Path(__file__).resolve().parents[2]
               / "data" / "processed" / "firegrid.parquet")
    return Path(os.environ.get("FIREGRID_SOURCE", default))


class Dataset(BaseDataset):

    name = "Real"

    requirements = ["pandas", "pyarrow"]

    def prepare(self):
        src = source_path()
        if not src.exists():
            raise SystemExit(f"[Real] no source table at {src}: build it "
                             "with pipeline/ or set FIREGRID_SOURCE.")
        df = pd.read_parquet(src).drop(columns=OUTCOMES)
        year = df["date"].dt.year
        train = df[~year.isin([PUBLIC_YEAR, PRIVATE_YEAR])]
        public = df[year == PUBLIC_YEAR]
        private = df[year == PRIVATE_YEAR]
        # The split must hold out whole years.
        assert not set(train["date"]) & set(public["date"])
        assert not set(train["date"]) & set(private["date"])

        root, ref = get_data_path("real"), get_data_path("reference")
        root.mkdir(parents=True, exist_ok=True)
        ref.mkdir(parents=True, exist_ok=True)
        train.to_parquet(root / "train.parquet", index=False)
        public.to_parquet(root / "public.parquet", index=False)
        private.drop(columns=TARGET).to_parquet(root / "private.parquet",
                                                index=False)
        (private[[TARGET, "date"]].rename(columns={TARGET: "target"})
         .to_csv(ref / "private.csv", index=False))
        print(f"[Real] train {len(train)}, public {len(public)}, "
              f"private {len(private)} rows")

    def get_data(self):
        root = get_data_path("real")
        if not (root / "train.parquet").exists():
            # Fail if the data has not been prepared. Use --force to bypass
            # an eventual cache.
            raise SystemExit(f"[Real] no data at {root}. Run "
                             "`benchopt prepare benchmark -d Real --force`.")

        train = pd.read_parquet(root / "train.parquet")
        public = pd.read_parquet(root / "public.parquet")
        return dict(
            X=train.drop(columns=TARGET), y=train[TARGET],
            X_public=public.drop(columns=TARGET), y_public=public[TARGET],
            X_private=pd.read_parquet(root / "private.parquet"),
        )
