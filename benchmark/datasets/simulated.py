"""Synthetic dataset: no data needed, runs anywhere (smoke tests and CI).

It returns the same keys, column names and dtypes and the same kind of
split as the real dataset: (cell, day) rows, whole years held out for the
public and private test sets. Fire risk rises with heat, drought, wind and
a per-cell propensity, so a real model beats the constant floor.
"""

import numpy as np
import pandas as pd
from benchopt import BaseDataset

WEATHER = ["t_mean", "t_min", "t_max", "rain", "snow", "wind", "rh", "q",
           "sw_rad", "lw_rad", "etp", "evap", "swi", "wg_root",
           "snow_depth", "rain_3d", "rain_7d", "rain_30d",
           "days_since_rain", "t_max_7d", "rh_7d", "swi_prev"]
# Vegetation features of the last MODIS composite: shares / percentiles in
# [0, 1] (dry_gap in [-1, 1]), then the integer columns.
VEGETATION = ["ndvi_pct", "ndvi_pct_p10", "ndvi_extreme", "nbr_pct",
              "nbr_pct_p10", "nbr_extreme", "dry_gap", "burnt_1y",
              "burnt_5y", "vi_valid_share", "vi_prev_share"]
# Static per-cell features (constant over days).
ROADS = ["road_km", "motorway_km", "road_traffic", "road_dist"]
POPULATION = ["pop_density", "inhabited_share", "sparse_share"]
LANDCOVER = ["land_share", "lc_urban", "lc_crops", "lc_vine_orchard",
             "lc_pasture", "lc_agri_natural", "lc_forest", "lc_grassland",
             "lc_shrub", "lc_maquis", "lc_sparse", "lc_water"]
FOREST = ["forest_broadleaf", "forest_conifer", "forest_mixed",
          "forest_open", "heath", "sp_pine_maritime", "sp_pine_aleppo",
          "sp_pine_scots", "sp_pine_black", "sp_pine_mountain",
          "sp_pine_other", "sp_pine_mixed", "sp_fir_spruce", "sp_douglas",
          "sp_larch", "sp_conifer_mixed", "sp_oak_evergreen",
          "sp_oak_deciduous", "sp_beech", "sp_chestnut", "sp_robinia",
          "sp_poplar", "sp_broadleaf_mixed", "sp_mixed"]
FUEL_CLASSES = ["fuel_broadleaf_decid", "fuel_broadleaf_med",
                "fuel_oak_evergreen", "fuel_pine_maritime",
                "fuel_pine_aleppo", "fuel_pine_mountain",
                "fuel_fir_spruce_douglas", "fuel_larch",
                "fuel_conifer_mixed", "fuel_mixed", "fuel_open_forest",
                "fuel_heath_med", "fuel_heath_other", "fuel_grass"]
SHARES = LANDCOVER + FOREST + FUEL_CLASSES
# Calendar features: shares in [0, 1], then 0/1 flags.
CALENDAR_SHARES = ["holiday_pop_share"]
CALENDAR_FLAGS = ["holiday_zone_a", "holiday_zone_b", "holiday_zone_c",
                  "summer_holiday", "short_holiday", "public_holiday",
                  "bridge_day", "long_weekend"]


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = ["pandas"]

    parameters = {"n_cells": [120], "n_days": [40]}
    test_config = {"n_cells": 60, "n_days": 20}

    def get_data(self):
        rng = np.random.default_rng(self.get_seed())
        # 7 train years (>= the 5 CV folds), then public and private years.
        years = list(range(2001, 2008)) + [2018, 2025]
        days = pd.DatetimeIndex(np.concatenate([
            pd.date_range(f"{y}-07-01", periods=self.n_days) for y in years
        ]))
        cells = np.arange(self.n_cells)
        x, y = rng.uniform(100, 1200, (2, self.n_cells))
        propensity = rng.normal(-1, 1, self.n_cells)

        n = len(days) * self.n_cells
        df = pd.DataFrame({
            "cell_id": np.tile(cells, len(days)),
            "date": np.repeat(days, self.n_cells),
            "x": np.tile(x, len(days)), "y": np.tile(y, len(days)),
        })
        df["lon"] = -5 + df["x"] / 80
        df["lat"] = 41 + df["y"] / 110
        roads = rng.exponential(1, (self.n_cells, len(ROADS)))
        for j, col in enumerate(ROADS):
            df[col] = roads[df["cell_id"], j]
        pop = rng.uniform(size=(self.n_cells, len(POPULATION)))
        for j, col in enumerate(POPULATION):
            df[col] = pop[df["cell_id"], j]
        shares = rng.dirichlet(np.ones(len(SHARES)), self.n_cells)
        for j, col in enumerate(SHARES):
            df[col] = shares[df["cell_id"], j]
        fuel_px = rng.integers(0, 3000, self.n_cells)
        df["fuel_px"] = fuel_px[df["cell_id"]].astype("int32")
        df["fuel_pure_share"] = rng.uniform(size=self.n_cells)[df["cell_id"]]
        for col in WEATHER:
            df[col] = rng.normal(size=n)
        for col in VEGETATION:
            df[col] = rng.uniform(size=n)
        df["dry_gap"] = df["nbr_pct"] - df["ndvi_pct"]
        df["vi_px"] = (df["vi_valid_share"] * df["fuel_px"]).astype("int32")
        df["vi_age"] = rng.integers(1, 17, n).astype("int16")
        calendar = {col: rng.uniform(size=n) for col in CALENDAR_SHARES}
        calendar.update({col: (rng.uniform(size=n) < 0.3).astype("int8")
                         for col in CALENDAR_FLAGS})
        df = pd.concat([df, pd.DataFrame(calendar, index=df.index)], axis=1)
        logit = (-5 + 1.2 * df["t_max"] - 0.8 * df["rain_30d"]
                 + 0.6 * df["wind"] + propensity[df["cell_id"]])
        fire = pd.Series(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)),
                         index=df.index, name="fire").astype("int8")
        # Fire outcomes, as in the real data: count >= 1 and burned area
        # (m2, heavy-tailed) on fire rows, 0 elsewhere.
        n_fires = fire * (1 + rng.poisson(0.3, n))
        outcomes = pd.DataFrame({
            "n_fires": n_fires.astype("int16"),
            "area_m2": (n_fires * rng.lognormal(7, 2, n)).astype("float32"),
        }, index=df.index)
        df = df.astype({c: "float32" for c in WEATHER + VEGETATION + ROADS
                        + POPULATION + SHARES + CALENDAR_SHARES
                        + ["fuel_pure_share", "x", "y", "lon", "lat"]})

        year = df["date"].dt.year
        train, public = year < 2018, year == 2018
        private = year == 2025
        return dict(X=df[train], y=fire[train], X_public=df[public],
                    y_public=fire[public], X_private=df[private],
                    outcomes=outcomes[train],
                    outcomes_public=outcomes[public])
