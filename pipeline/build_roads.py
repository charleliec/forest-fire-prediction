"""National road network and its traffic -> static road features per cell.

Source: "Trafic moyen journalier annuel sur le réseau routier national"
(DGITM, Ministère de la Transition écologique, Licence Ouverte,
https://www.data.gouv.fr/fr/datasets/trafic-moyen-journalier-annuel-sur-le-reseau-routier-national/):
the motorways and national roads as polylines (Lambert-93), each section with
its annual average daily traffic (TMJA, vehicles per day, both directions).

The features are static, from the 2019 file: it is the last year that covers
the whole national network (2020 and 2021 are not published, and from 2022
only the conceded motorways are). Sections without a count (TMJA = 0) get the
median TMJA of their road, else of their class (motorway / other).

Corsica is not covered (its national roads belong to the Collectivité de
Corse), so its cells get missing values rather than "no road".

Per cell:
- ``road_km``: length of national network in the cell (km);
- ``motorway_km``: of which motorways (km);
- ``road_traffic``: traffic on it, thousands of vehicle-km per day
  (sum of TMJA x length);
- ``road_dist``: distance from the cell centre to the nearest national road
  or motorway (km).

    python pipeline/build_roads.py

Writes ``data/interim/roads_cell.parquet``.
"""

import io
import zipfile

import numpy as np
import pandas as pd
import requests
import shapefile
from scipy.spatial import cKDTree

from common import INTERIM, ROADS_DIR, cell_index

URL = ("https://static.data.gouv.fr/resources/trafic-moyen-journalier-annuel"
       "-sur-le-reseau-routier-national/20211222-170254/tmja2019-shp.zip")
SHP = ROADS_DIR / "tmja2019" / "TMJA2019.shp"
STEP = 100.  # m, the polylines are cut into pieces this long at most
# Corsica: south-east of the mainland coast (lon 8.5-9.6, lat 41.3-43.1).
CORSICA_LON, CORSICA_LAT = 8., 43.5


def download():
    if SHP.exists():
        return
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(SHP.parent)


def load_sections():
    """One row per section, with its polyline (Lambert-93, m)."""
    reader = shapefile.Reader(str(SHP), encoding="utf-8")
    df = pd.DataFrame([rec.as_dict() for rec in reader.iterRecords()])
    df = df[["route", "longueur", "TMJA"]].rename(
        columns={"longueur": "length_m", "TMJA": "tmja"})
    df["shape"] = [s for s in reader.iterShapes()]
    df["motorway"] = df["route"].str.startswith("A")

    counted = df["tmja"] > 0
    by_route = df[counted].groupby("route")["tmja"].median()
    by_class = df[counted].groupby("motorway")["tmja"].median()
    df["tmja"] = df["tmja"].astype("float64")
    df.loc[~counted, "tmja"] = df.loc[~counted, "route"].map(by_route)
    still = df["tmja"].isna()
    df.loc[still, "tmja"] = df.loc[still, "motorway"].map(by_class)
    print(f"{len(df)} sections, {df['length_m'].sum() / 1e3:.0f} km; "
          f"without count: {(~counted).sum()} sections "
          f"({df.loc[~counted, 'length_m'].sum() / 1e3:.0f} km), "
          f"{still.sum()} filled with the class median")
    return df


def pieces(sections):
    """Cut the polylines into pieces of at most STEP m: midpoint, length."""
    rows = []
    for shape, tmja, motorway in zip(sections["shape"], sections["tmja"],
                                     sections["motorway"]):
        pts = np.asarray(shape.points)
        bounds = list(shape.parts) + [len(pts)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 2:
                continue
            p0, p1 = pts[a:b - 1], pts[a + 1:b]
            length = np.hypot(*(p1 - p0).T)
            n = np.maximum(np.ceil(length / STEP), 1).astype(int)
            seg = np.repeat(np.arange(len(n)), n)
            k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
            t = ((k + .5) / n[seg])[:, None]
            mid = p0[seg] + t * (p1[seg] - p0[seg])
            rows.append(pd.DataFrame({
                "x": mid[:, 0], "y": mid[:, 1],
                "km": length[seg] / n[seg] / 1e3,
                "tmja": tmja, "motorway": motorway}))
    return pd.concat(rows, ignore_index=True)


def main():
    download()
    p = pieces(load_sections())
    grid = pd.read_parquet(INTERIM / "grid.parquet")

    p["ix"], p["iy"] = cell_index(p["x"], p["y"])
    p = p.merge(grid[["ix", "iy", "cell_id"]], on=["ix", "iy"], how="left")
    out = p["cell_id"].isna()
    print(f"{p['km'].sum():.0f} km cut into {len(p)} pieces; outside the "
          f"grid: {p.loc[out, 'km'].sum():.1f} km (dropped)")
    p["motorway_km"] = p["km"] * p["motorway"]
    p["road_traffic"] = p["km"] * p["tmja"] / 1e3
    per_cell = (p[~out].groupby("cell_id")
                [["km", "motorway_km", "road_traffic"]].sum()
                .rename(columns={"km": "road_km"}))

    roads = grid[["cell_id", "x", "y", "lon", "lat"]].join(
        per_cell, on="cell_id")
    roads[per_cell.columns] = roads[per_cell.columns].fillna(0.)
    dist, _ = cKDTree(p[["x", "y"]].to_numpy()).query(
        roads[["x", "y"]].to_numpy())
    roads["road_dist"] = dist / 1e3
    features = ["road_km", "motorway_km", "road_traffic", "road_dist"]
    corsica = ((roads["lon"] > CORSICA_LON)
               & (roads["lat"] < CORSICA_LAT))
    roads.loc[corsica, features] = np.nan
    roads = roads[["cell_id", *features]].astype(
        {f: "float32" for f in features})

    INTERIM.mkdir(parents=True, exist_ok=True)
    roads.to_parquet(INTERIM / "roads_cell.parquet", index=False)
    print(f"{len(roads)} cells, {corsica.sum()} in Corsica (NaN); cells with "
          f"a national road: {(roads['road_km'] > 0).sum()}, with a "
          f"motorway: {(roads['motorway_km'] > 0).sum()}")
    print(roads[features].describe().round(1))


if __name__ == "__main__":
    main()
