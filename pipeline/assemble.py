"""Assemble the (cell, day) table: features + fire label.

Writes ``data/processed/firegrid.parquet``, one row per (cell_id, date) for
every cell of the grid and every day of FIRST_YEAR..LAST_YEAR:

- cell: ``cell_id``, ``x``, ``y`` (Lambert-93 centre, km), ``lon``, ``lat``;
- roads (static, see build_roads.py): national network length, motorway
  length and traffic in the cell, distance to the nearest national road;
- population (static, see build_population.py): density, share of the
  cell inhabited, share with scattered housing (INSEE 1 km grid);
- land cover (static, see build_landcover.py): share of the cell in each
  Corine Land Cover 2018 group (forest, maquis, crops, urban...);
- forest type (static, see build_forest.py): share of the cell in each
  BD Forêt v2 composition (broadleaf, conifer...) and dominant species;
- calendar: ``date``;
- weather of the day (SIM2, see build_weather.VARIABLES);
- antecedent weather, from the days *before* the day only: rain sums over
  3 / 7 / 30 days, days since the last rain >= 1 mm, mean t_max / rh over
  7 days, swi of the previous day;
- vegetation: the last MOD13Q1 NDVI composite *completed before* the day
  (a composite covers 16 days from its start date; using one that overlaps
  the day would show the burn scar), its age in days, and its anomaly with
  respect to the cell's mean for that composite period over the training
  years;
- label: ``fire`` (at least one BDIFF fire alert in the cell that day),
  plus ``n_fires`` and ``area_m2`` (burned area), which are not features.

    python pipeline/assemble.py [--no-ndvi]
"""

import argparse

import numpy as np
import pandas as pd

from common import FIRST_YEAR, INTERIM, LAST_YEAR, PROCESSED, TEST_YEARS

COMPOSITE_DAYS = 16
# Static per-cell tables, data/interim/<name>_cell.parquet.
STATIC = ["roads", "population", "landcover", "forest"]


def wide(df, col, cells, days):
    """(n_days, n_cells) array of ``col``."""
    return (df.pivot(index="date", columns="cell_id", values=col)
            .reindex(index=days, columns=cells).to_numpy(dtype="float64"))


def past_sum(a, window):
    """Sum over the ``window`` days strictly before each day."""
    c = np.vstack([np.zeros((1, a.shape[1])), np.nancumsum(a, axis=0)])
    out = np.full_like(a, np.nan)
    out[window:] = c[window:-1] - c[:-window - 1]
    return out


def days_since(mask):
    """Days since ``mask`` was last true, strictly before each day."""
    out = np.full(mask.shape, np.nan)
    last = np.full(mask.shape[1], np.nan)
    for t in range(mask.shape[0]):
        out[t] = t - last
        last = np.where(mask[t], t, last)
    return out


def antecedent(weather, cells):
    days = pd.date_range(weather["date"].min(), weather["date"].max())
    rain = wide(weather, "rain", cells, days)
    feats = {f"rain_{w}d": past_sum(rain, w) for w in (3, 7, 30)}
    feats["days_since_rain"] = days_since(rain >= 1.)
    feats["t_max_7d"] = past_sum(wide(weather, "t_max", cells, days), 7) / 7
    feats["rh_7d"] = past_sum(wide(weather, "rh", cells, days), 7) / 7
    swi = wide(weather, "swi", cells, days)
    feats["swi_prev"] = np.vstack([np.full((1, swi.shape[1]), np.nan),
                                   swi[:-1]])
    idx = pd.MultiIndex.from_product([days, cells],
                                     names=["date", "cell_id"])
    return pd.DataFrame({k: v.ravel().astype("float32")
                         for k, v in feats.items()}, index=idx).reset_index()


def add_ndvi(df):
    ndvi = pd.read_parquet(INTERIM / "ndvi_cell.parquet")
    ndvi = ndvi.rename(columns={"date": "composite_start"})
    ndvi["available"] = ndvi["composite_start"] + pd.Timedelta(
        days=COMPOSITE_DAYS)
    # Climatology per cell and composite period, training years only.
    ndvi["period"] = ndvi["composite_start"].dt.dayofyear
    train = ~ndvi["composite_start"].dt.year.isin(TEST_YEARS)
    clim = (ndvi[train].groupby(["cell_id", "period"])["ndvi"].mean()
            .rename("ndvi_clim"))
    ndvi = ndvi.join(clim, on=["cell_id", "period"])
    ndvi["ndvi_anom"] = ndvi["ndvi"] - ndvi["ndvi_clim"]
    ndvi = ndvi.sort_values("available")[
        ["cell_id", "available", "ndvi", "ndvi_anom"]]

    df = df.sort_values("date")
    df = pd.merge_asof(df, ndvi, left_on="date", right_on="available",
                       by="cell_id", direction="backward",
                       allow_exact_matches=True)
    df["ndvi_age"] = (df["date"] - df.pop("available")).dt.days
    return df


def main(no_ndvi):
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    weather = pd.read_parquet(INTERIM / "weather_cell_day.parquet")
    fires = pd.read_parquet(INTERIM / "fires_cell_day.parquet")
    static = [pd.read_parquet(INTERIM / f"{name}_cell.parquet")
              for name in STATIC]
    cells = grid["cell_id"].to_numpy()

    df = weather.merge(antecedent(weather, cells), on=["date", "cell_id"])
    df = df[df["date"].dt.year.between(FIRST_YEAR, LAST_YEAR)]
    expected = len(cells) * len(pd.date_range(f"{FIRST_YEAR}-01-01",
                                              f"{LAST_YEAR}-12-31"))
    if len(df) != expected:
        raise SystemExit(f"{len(df)} rows, expected {expected}: missing "
                         "weather for some (cell, day)")

    if not no_ndvi:
        df = add_ndvi(df)

    df = df.merge(grid[["cell_id", "x", "y", "lon", "lat"]], on="cell_id")
    df["x"] = (df["x"] / 1000).astype("float32")
    df["y"] = (df["y"] / 1000).astype("float32")
    for table in static:
        df = df.merge(table, on="cell_id", how="left")
    df = df.merge(fires[["cell_id", "date", "n_fires", "area_m2"]],
                  on=["cell_id", "date"], how="left")
    df["n_fires"] = df["n_fires"].fillna(0).astype("int16")
    df["area_m2"] = df["area_m2"].fillna(0).astype("float32")
    df["fire"] = (df["n_fires"] > 0).astype("int8")
    if df["n_fires"].sum() != fires.loc[
            fires["date"].dt.year.between(FIRST_YEAR, LAST_YEAR),
            "n_fires"].sum():
        raise SystemExit("some fires were lost in the merge")

    first = ["cell_id", "date", "x", "y", "lon", "lat",
             *(c for t in static for c in t.columns.drop("cell_id"))]
    last = ["fire", "n_fires", "area_m2"]
    df = df[first + [c for c in df.columns if c not in first + last] + last]
    df = df.sort_values(["date", "cell_id"]).reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "firegrid.parquet", index=False)
    print(f"{len(df)} rows x {df.shape[1]} columns, "
          f"{df['fire'].mean():.3%} fire (cell, day); NaN share: "
          f"{df.isna().mean()[lambda s: s > 0].round(4).to_dict()}")
    print(df.groupby(df["date"].dt.year)["fire"].agg(["sum", "mean"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ndvi", action="store_true",
                        help="skip NDVI (development only)")
    main(parser.parse_args().no_ndvi)
