"""BDIFF fires -> fires per (25 km cell, day).

Each fire is located at the centroid of its commune (geo.api.gouv.fr, current
communes plus merged ones that remain as delegated / associated communes),
then assigned to the grid cell containing it, or to the nearest cell when the
centroid falls outside the grid (small islands, coastline).

Writes ``data/interim/fires.parquet`` (one row per located fire, for EDA) and
``data/interim/fires_cell_day.parquet`` (n_fires and burned area per
(cell_id, date)).
"""

import json

import numpy as np
import pandas as pd
from pyproj import Transformer

from common import (BDIFF_CSV, CRS_L93, GEO_DIR, INTERIM, cell_index)

COLUMNS = {
    "Année": "year", "Numéro": "bdiff_id", "Département": "dept",
    "Code INSEE": "insee", "Nom de la commune": "commune",
    "Date de première alerte": "alert", "Surface parcourue (m2)": "area_m2",
}


def load_bdiff():
    df = pd.read_csv(BDIFF_CSV, sep=";", skiprows=3, encoding="utf-8",
                     usecols=list(COLUMNS),
                     dtype={"Département": str, "Code INSEE": str})
    df = df.rename(columns=COLUMNS)
    n = len(df)
    df = df[~df["dept"].str.startswith("97")].copy()  # metropolitan only
    print(f"BDIFF: {n} fires, {len(df)} in metropolitan France")
    df["alert"] = pd.to_datetime(df["alert"])
    df["date"] = df["alert"].dt.normalize()
    return df


def load_centroids():
    rows = []
    for name in ("communes_all.json", "communes_deleguees.json"):
        rows += json.loads((GEO_DIR / name).read_text(encoding="utf-8"))
    c = pd.DataFrame(rows).dropna(subset=["centre"])
    # A current commune wins over a delegated one with the same code.
    c = c.drop_duplicates("code", keep="first")
    lonlat = np.array([p["coordinates"] for p in c["centre"]])
    return pd.DataFrame({"insee": c["code"].to_numpy(),
                         "lon": lonlat[:, 0], "lat": lonlat[:, 1]})


def main():
    fires = load_bdiff().merge(load_centroids(), on="insee", how="left")
    missing = fires["lon"].isna()
    print(f"dropping {missing.sum()} fires ({missing.mean():.2%}) whose "
          f"commune could not be located: "
          f"{sorted(fires.loc[missing, 'insee'].unique())}")
    fires = fires[~missing].copy()

    to_l93 = Transformer.from_crs("EPSG:4326", CRS_L93, always_xy=True)
    fires["x"], fires["y"] = to_l93.transform(fires["lon"], fires["lat"])
    fires["ix"], fires["iy"] = cell_index(fires["x"], fires["y"])

    grid = pd.read_parquet(INTERIM / "grid.parquet")
    fires = fires.merge(grid[["ix", "iy", "cell_id"]], on=["ix", "iy"],
                        how="left")
    out = fires["cell_id"].isna()
    if out.any():
        gx, gy = grid["x"].to_numpy(), grid["y"].to_numpy()
        fx, fy = fires.loc[out, "x"].to_numpy(), fires.loc[out, "y"].to_numpy()
        d2 = (fx[:, None] - gx) ** 2 + (fy[:, None] - gy) ** 2
        fires.loc[out, "cell_id"] = grid["cell_id"].to_numpy()[d2.argmin(1)]
        print(f"{out.sum()} fires outside the grid -> nearest cell "
              f"(max distance {np.sqrt(d2.min(1)).max() / 1000:.1f} km)")
    fires["cell_id"] = fires["cell_id"].astype(int)

    INTERIM.mkdir(parents=True, exist_ok=True)
    fires.drop(columns=["ix", "iy"]).to_parquet(INTERIM / "fires.parquet",
                                                 index=False)
    per_day = (fires.groupby(["cell_id", "date"])
               .agg(n_fires=("bdiff_id", "size"),
                    area_m2=("area_m2", "sum"),
                    max_area_m2=("area_m2", "max"))
               .reset_index())
    per_day.to_parquet(INTERIM / "fires_cell_day.parquet", index=False)
    print(f"{len(fires)} fires -> {len(per_day)} fire (cell, day) pairs in "
          f"{per_day['cell_id'].nunique()} cells")


if __name__ == "__main__":
    main()
