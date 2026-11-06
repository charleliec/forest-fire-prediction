"""Corine Land Cover 2018 -> static land-cover shares per cell.

Source: CORINE Land Cover 2018 (Copernicus Land Monitoring Service, produced
for France by SDES and IGN), raster version V2020_20u1, 100 m, metropolitan
France, distributed by IGN on the Géoplateforme
(https://data.geopf.fr/telechargement/resource/CORINELANDCOVER), Licence
Ouverte. The raster comes in WGS84; it is reprojected (nearest neighbour)
onto a 100 m Lambert-93 grid aligned on the 25 km cells, 250 x 250 pixels
per cell.

Per cell, the share of the *whole* 25 km cell (625 km2) covered by each
group of CLC classes (level-3 codes):

- ``land_share``: any class but sea and ocean (523), i.e. French land and
  inland / coastal water; it is below 1 on the coast and the borders;
- ``lc_urban``: artificial surfaces (1xx);
- ``lc_crops``: arable land and annual crops (211-213, 241, 242);
- ``lc_vine_orchard``: vineyards, orchards, olive groves (221-223);
- ``lc_pasture``: pastures (231);
- ``lc_agri_natural``: agriculture mixed with natural vegetation,
  agro-forestry (243, 244);
- ``lc_forest``: broad-leaved, coniferous and mixed forest (311-313);
- ``lc_grassland``: natural grassland (321);
- ``lc_shrub``: moors, heathland, transitional woodland-shrub (322, 324);
- ``lc_maquis``: sclerophyllous vegetation, i.e. maquis and garrigue (323);
- ``lc_sparse``: beaches, rocks, sparse vegetation, burnt areas, glaciers
  (331-335);
- ``lc_water``: wetlands and inland / coastal water (4xx, 511-522).

    python pipeline/build_landcover.py

Writes ``data/interim/landcover_cell.parquet``.
"""

import time

import numpy as np
import pandas as pd
import py7zr
import rasterio
import requests
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from common import CELL_SIZE, CRS_L93, INTERIM, LANDCOVER_DIR

NAME = "CLC_1-0__TIF_WGS84G_FXX_2018-01-01"
URL = ("https://data.geopf.fr/telechargement/download/CORINELANDCOVER/"
       f"{NAME}/{NAME}.7z")
TIF = (LANDCOVER_DIR / NAME / "U2018_CLC2018_V2020_20u1"
       / "U2018_CLC2018_V2020_20u1.tif")
RES = 100  # m, the CLC resolution
NODATA = -128

# Raster value v (1..44) is the CLC level-3 code CODES[v - 1].
CODES = [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142,
         211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244,
         311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335,
         411, 412, 421, 422, 423, 511, 512, 521, 522, 523]
GROUPS = {
    "lc_urban": range(100, 200),
    "lc_crops": [211, 212, 213, 241, 242],
    "lc_vine_orchard": [221, 222, 223],
    "lc_pasture": [231],
    "lc_agri_natural": [243, 244],
    "lc_forest": [311, 312, 313],
    "lc_grassland": [321],
    "lc_shrub": [322, 324],
    "lc_maquis": [323],
    "lc_sparse": range(331, 336),
    "lc_water": [*range(400, 500), 511, 512, 521, 522],
}
SEA = 523


def get(url, tries=8, **kwargs):
    """``requests.get``, retried with backoff when the Géoplateforme
    rate-limits (429) or fails (5xx)."""
    for k in range(tries):
        r = requests.get(url, timeout=300, **kwargs)
        if r.status_code != 429 and r.status_code < 500:
            break
        r.close()
        time.sleep(float(r.headers.get("Retry-After", 2 ** k)))
    r.raise_for_status()
    return r


def download(url, path):
    """Download ``url`` to ``path`` (streamed) unless it exists."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with get(url, stream=True) as r, open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.replace(path)


def grid_frame(grid, res):
    """Top-left corner (m) and shape of a ``res`` m raster over the grid."""
    x0 = grid["ix"].min() * CELL_SIZE
    ytop = (grid["iy"].max() + 1) * CELL_SIZE
    per = CELL_SIZE // res
    shape = ((grid["iy"].max() - grid["iy"].min() + 1) * per,
             (grid["ix"].max() - grid["ix"].min() + 1) * per)
    return x0, ytop, shape


def cell_counts(a, x0, ytop, res, nvals):
    """Pixel counts per (cell, value) of a raster aligned on the cells.

    ``a`` holds integers in 0..nvals-1, its top-left corner is (x0, ytop) in
    Lambert-93 (multiples of CELL_SIZE) and its pixels are ``res`` m; its
    shape is a multiple of CELL_SIZE / res. Returns ``ix``, ``iy``,
    ``value``, ``n`` (non-zero counts only).
    """
    per = CELL_SIZE // res
    ncy, ncx = a.shape[0] // per, a.shape[1] // per
    col_cell = np.repeat(np.arange(ncx), per)[None, :] * nvals
    n = np.empty((ncy, ncx, nvals), dtype=np.int64)
    for k in range(ncy):
        key = col_cell + a[k * per:(k + 1) * per].astype(np.int64)
        n[k] = np.bincount(key.ravel(),
                           minlength=ncx * nvals).reshape(ncx, nvals)
    k, j, v = np.nonzero(n)
    return pd.DataFrame({"ix": x0 // CELL_SIZE + j,
                         "iy": ytop // CELL_SIZE - 1 - k,
                         "value": v, "n": n[k, j, v]})


def main():
    archive = LANDCOVER_DIR / f"{NAME}.7z"
    download(URL, archive)
    if not TIF.exists():
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(LANDCOVER_DIR)

    grid = pd.read_parquet(INTERIM / "grid.parquet")
    x0, ytop, shape = grid_frame(grid, RES)
    dst = np.full(shape, NODATA, dtype=np.int8)
    with rasterio.open(TIF) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_nodata=NODATA, dst_nodata=NODATA,
                  dst_transform=from_origin(x0, ytop, RES, RES),
                  dst_crs=CRS_L93, resampling=Resampling.nearest)
    dst[dst == NODATA] = 0  # value 0: outside the CLC coverage
    counts = cell_counts(dst, x0, ytop, RES, len(CODES) + 1)
    counts = counts.merge(grid[["ix", "iy", "cell_id"]], on=["ix", "iy"])
    counts["code"] = np.r_[0, CODES][counts["value"]]

    per_code = counts.pivot_table(index="cell_id", columns="code",
                                  values="n", aggfunc="sum", fill_value=0)
    share = per_code / (CELL_SIZE // RES) ** 2
    lc = pd.DataFrame(index=share.index)
    land = share.columns.difference([0, SEA])
    lc["land_share"] = share[land].sum(axis=1)
    for name, codes in GROUPS.items():
        lc[name] = share[share.columns.intersection(codes)].sum(axis=1)
    lc = grid[["cell_id"]].join(lc, on="cell_id").fillna(0.)
    features = lc.columns.drop("cell_id")
    lc = lc.astype({f: "float32" for f in features})

    INTERIM.mkdir(parents=True, exist_ok=True)
    lc.to_parquet(INTERIM / "landcover_cell.parquet", index=False)
    print(f"{len(lc)} cells; mean share of the cell (%):")
    print((lc[features].mean() * 100).round(2).to_string())


if __name__ == "__main__":
    main()
