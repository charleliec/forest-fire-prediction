"""Build the 25 km grid from the SIM2 8 km points.

A cell is kept when at least one SIM2 point and at least one French commune
centroid fall in it. SIM2 alone also covers land beyond the eastern border
(Switzerland, Germany, Italy...), where BDIFF records no fire; and since a
fire is located at its commune centroid, a cell without any centroid could
never have a fire. The grid thus covers metropolitan France (Corsica
included). ``cell_id`` numbers all the SIM2 cells, before this filter, so it
has gaps but does not depend on it.

Writes ``data/interim/grid.parquet`` (one row per cell) and
``data/interim/sim2_points.parquet`` (SIM2 point -> cell).
"""

import numpy as np
import pandas as pd
from pyproj import Transformer

from build_fires import load_centroids
from common import (CELL_SIZE, CRS_L93, CRS_SIM2, FIRST_YEAR, INTERIM,
                    cell_index, sim2_file)


def main():
    pts = pd.read_csv(sim2_file(FIRST_YEAR), sep=";",
                      usecols=["LAMBX", "LAMBY"]).drop_duplicates()
    to_l93 = Transformer.from_crs(CRS_SIM2, CRS_L93, always_xy=True)
    x, y = to_l93.transform(pts["LAMBX"].to_numpy() * 100.,
                            pts["LAMBY"].to_numpy() * 100.)
    pts["ix"], pts["iy"] = cell_index(x, y)

    grid = (pts.groupby(["ix", "iy"]).size().rename("n_sim2_points")
            .reset_index().sort_values(["iy", "ix"], ascending=[False, True])
            .reset_index(drop=True))
    grid.insert(0, "cell_id", np.arange(len(grid)))

    com = load_centroids()
    to_l93_wgs = Transformer.from_crs("EPSG:4326", CRS_L93, always_xy=True)
    cx, cy = to_l93_wgs.transform(com["lon"].to_numpy(),
                                  com["lat"].to_numpy())
    com["ix"], com["iy"] = cell_index(cx, cy)
    n_com = com.groupby(["ix", "iy"]).size().rename("n_communes")
    grid = grid.join(n_com, on=["ix", "iy"])
    foreign = grid["n_communes"].isna()
    print(f"dropping {foreign.sum()} cells without any commune centroid "
          f"({grid.loc[foreign, 'n_sim2_points'].sum()} SIM2 points)")
    grid = grid[~foreign].astype({"n_communes": int}).reset_index(drop=True)
    grid["x"] = (grid["ix"] + .5) * CELL_SIZE
    grid["y"] = (grid["iy"] + .5) * CELL_SIZE
    to_wgs = Transformer.from_crs(CRS_L93, "EPSG:4326", always_xy=True)
    grid["lon"], grid["lat"] = to_wgs.transform(grid["x"], grid["y"])

    pts = pts.merge(grid[["ix", "iy", "cell_id"]], on=["ix", "iy"])
    INTERIM.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(INTERIM / "grid.parquet", index=False)
    pts[["LAMBX", "LAMBY", "cell_id"]].to_parquet(
        INTERIM / "sim2_points.parquet", index=False)
    print(f"{len(pts)} SIM2 points -> {len(grid)} cells of "
          f"{CELL_SIZE // 1000} km; points per cell: "
          f"{grid['n_sim2_points'].describe().round(1).to_dict()}")
    print(f"lon {grid['lon'].min():.2f}..{grid['lon'].max():.2f}, "
          f"lat {grid['lat'].min():.2f}..{grid['lat'].max():.2f}")


if __name__ == "__main__":
    main()
