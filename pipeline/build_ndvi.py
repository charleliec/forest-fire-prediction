"""MODIS MOD13Q1.061 NDVI (250 m, 16-day composites) -> NDVI per cell.

Uses NASA AppEEARS (https://appeears.earthdatacloud.nasa.gov/api/). AppEEARS
caps the number of files of a request, and writes one raster per polygon,
date and layer, so per-cell polygons are out of reach. Instead, one *area*
task per year clips the NDVI to a single polygon, the outline of the grid;
the rasters are downloaded one at a time, averaged over the 25 km cells
locally, and deleted.

Credentials (NASA Earthdata Login), never stored in the repository:
``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` environment variables, or a
``~/.netrc`` entry for ``urs.earthdata.nasa.gov``.

    python pipeline/build_ndvi.py submit     # one task per year -> tasks.json
    python pipeline/build_ndvi.py status
    python pipeline/build_ndvi.py fetch      # download + aggregate, resumable

Writes ``data/interim/ndvi_cell.parquet``: cell_id, composite start date,
NDVI mean / std and number of valid 250 m pixels.
"""

import argparse
import json
import netrc
import os
import re
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

from common import (CELL_SIZE, CRS_L93, INTERIM, LAST_YEAR, NDVI_DIR,
                    WARMUP_YEAR, cell_index)

API = "https://appeears.earthdatacloud.nasa.gov/api"
PRODUCT = "MOD13Q1.061"
LAYER = "_250m_16_days_NDVI"
TASKS = NDVI_DIR / "tasks.json"
PARTS = NDVI_DIR / "parts"


def credentials():
    user = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if user and password:
        return user, password
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None
    if auth is None:
        raise SystemExit("Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD or add "
                         "urs.earthdata.nasa.gov to ~/.netrc")
    return auth[0], auth[2]


def login():
    r = requests.post(f"{API}/login", auth=credentials(), timeout=60)
    r.raise_for_status()
    return {"Authorization": f"Bearer {r.json()['token']}"}


def grid_outline():
    """GeoJSON of the union of the cells, in lon/lat, slightly buffered."""
    from shapely.geometry import box, mapping
    from shapely.ops import transform, unary_union

    cells = pd.read_parquet(INTERIM / "grid.parquet")
    union = unary_union([
        box(ix * CELL_SIZE, iy * CELL_SIZE, (ix + 1) * CELL_SIZE,
            (iy + 1) * CELL_SIZE) for ix, iy in zip(cells["ix"], cells["iy"])
    ]).buffer(1000).simplify(500).segmentize(5000)
    to_wgs = Transformer.from_crs(CRS_L93, "EPSG:4326", always_xy=True)
    geom = transform(to_wgs.transform, union)
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(geom)}]}


def submit():
    headers, geo = login(), grid_outline()
    tasks = json.loads(TASKS.read_text()) if TASKS.exists() else {}
    for year in range(WARMUP_YEAR, LAST_YEAR + 1):
        if str(year) in tasks:
            continue
        # The warm-up year only needs the composites before January.
        start = f"11-01-{year}" if year == WARMUP_YEAR else f"01-01-{year}"
        task = {
            "task_type": "area",
            "task_name": f"firegrid_ndvi_{year}",
            "params": {
                "dates": [{"startDate": start, "endDate": f"12-31-{year}"}],
                "layers": [{"product": PRODUCT, "layer": LAYER}],
                "geo": geo,
                "output": {"format": {"type": "geotiff"},
                           "projection": "native"},
            },
        }
        r = requests.post(f"{API}/task", json=task, headers=headers,
                          timeout=300)
        if not r.ok:
            raise SystemExit(f"{year}: {r.status_code} {r.text}")
        tasks[str(year)] = r.json()["task_id"]
        print(year, tasks[str(year)])
        NDVI_DIR.mkdir(parents=True, exist_ok=True)
        TASKS.write_text(json.dumps(tasks, indent=1))


def status():
    headers = login()
    for year, task_id in json.loads(TASKS.read_text()).items():
        info = requests.get(f"{API}/task/{task_id}", headers=headers,
                            timeout=60).json()
        print(year, task_id, info["status"])


class CellLookup:
    """Pixel -> cell_id for a raster grid, computed once per grid."""

    def __init__(self):
        grid = pd.read_parquet(INTERIM / "grid.parquet")
        self.ix0, self.iy0 = grid["ix"].min(), grid["iy"].min()
        self.table = np.full((grid["ix"].max() - self.ix0 + 1,
                              grid["iy"].max() - self.iy0 + 1), -1)
        self.table[grid["ix"] - self.ix0, grid["iy"] - self.iy0] = \
            grid["cell_id"]
        self.cache = {}

    def __call__(self, src):
        key = (src.crs.to_string(), tuple(src.transform), src.shape)
        if key not in self.cache:
            rows, cols = np.indices(src.shape)
            xs, ys = src.transform * (cols + .5, rows + .5)
            to_l93 = Transformer.from_crs(src.crs, CRS_L93, always_xy=True)
            x, y = to_l93.transform(xs.ravel(), ys.ravel())
            ix, iy = cell_index(x, y)
            ix, iy = ix - self.ix0, iy - self.iy0
            ok = ((ix >= 0) & (ix < self.table.shape[0])
                  & (iy >= 0) & (iy < self.table.shape[1]))
            cell = np.full(ix.shape, -1)
            cell[ok] = self.table[ix[ok], iy[ok]]
            self.cache[key] = cell
        return self.cache[key]


def aggregate(path, lookup):
    import rasterio

    with rasterio.open(path) as src:
        a = src.read(1).ravel().astype("float64")
        nodata, scale = src.nodata, src.scales[0]
        cell = lookup(src)
    valid = cell >= 0
    if nodata is not None:
        valid &= a != nodata
    if np.issubdtype(src.dtypes[0], np.integer):
        # Raw MOD13Q1 values: scale 0.0001, valid range -2000..10000.
        valid &= (a >= -2000) & (a <= 10000)
        a = a * (scale if scale != 1 else 1e-4)
    n_cells = lookup.table.max() + 1
    count = np.bincount(cell[valid], minlength=n_cells)
    s1 = np.bincount(cell[valid], a[valid], minlength=n_cells)
    s2 = np.bincount(cell[valid], a[valid] ** 2, minlength=n_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / count
        std = np.sqrt(np.maximum(s2 / count - mean ** 2, 0))
    doy = re.search(r"doy(\d{7})", path.name).group(1)
    date = datetime(int(doy[:4]), 1, 1) + timedelta(days=int(doy[4:]) - 1)
    return pd.DataFrame({"cell_id": np.arange(n_cells), "date": date,
                         "ndvi": mean.astype("float32"),
                         "ndvi_std": std.astype("float32"),
                         "ndvi_count": count.astype("int32")})


def fetch():
    tasks = json.loads(TASKS.read_text())
    lookup = CellLookup()
    PARTS.mkdir(parents=True, exist_ok=True)
    tmp = NDVI_DIR / "tmp"
    tmp.mkdir(exist_ok=True)
    for year, task_id in tasks.items():
        headers = login()
        while (st := requests.get(f"{API}/task/{task_id}", headers=headers,
                                  timeout=60).json()["status"]) != "done":
            if st == "error":
                raise SystemExit(f"{year}: task {task_id} failed")
            print(f"{year}: {st}, waiting...", flush=True)
            time.sleep(600)
            headers = login()
        files = requests.get(f"{API}/bundle/{task_id}", headers=headers,
                             timeout=60).json()["files"]
        tifs = [f for f in files if f["file_name"].endswith(".tif")
                and "_NDVI_doy" in f["file_name"]]
        print(f"{year}: {len(tifs)} NDVI rasters", flush=True)
        for f in tifs:
            name = f["file_name"].split("/")[-1]
            part = PARTS / name.replace(".tif", ".parquet")
            if part.exists():
                continue
            path = tmp / name
            with requests.get(f"{API}/bundle/{task_id}/{f['file_id']}",
                              headers=headers, stream=True,
                              timeout=600) as r:
                r.raise_for_status()
                with open(path, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            df = aggregate(path, lookup)
            df.to_parquet(part, index=False)
            path.unlink()
            print(f"  {name}: {f['file_size'] / 1e6:.0f} MB, mean NDVI "
                  f"{np.nanmean(df['ndvi']):.3f}, cells without data "
                  f"{(df['ndvi_count'] == 0).sum()}", flush=True)
    combine()


def combine():
    ndvi = pd.concat([pd.read_parquet(p) for p in sorted(PARTS.glob("*"))],
                     ignore_index=True)
    ndvi = ndvi.sort_values(["cell_id", "date"]).drop_duplicates(
        ["cell_id", "date"])
    ndvi.to_parquet(INTERIM / "ndvi_cell.parquet", index=False)
    print(f"NDVI: {len(ndvi)} rows, {ndvi['cell_id'].nunique()} cells, "
          f"{ndvi['date'].nunique()} composites "
          f"({ndvi['date'].min():%Y-%m-%d} .. {ndvi['date'].max():%Y-%m-%d})"
          f"; (cell, composite) without valid pixel: "
          f"{(ndvi['ndvi_count'] == 0).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action",
                        choices=["submit", "status", "fetch", "combine"])
    action = parser.parse_args().action
    {"submit": submit, "status": status, "fetch": fetch,
     "combine": combine}[action]()
