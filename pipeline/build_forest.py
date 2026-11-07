"""IGN BD Forêt v2 -> static forest-type shares per cell.

Source: BD Forêt® version 2.0 (IGN, Licence Ouverte), one shapefile per
département (96, Corsica included) on the Géoplateforme
(https://data.geopf.fr/telechargement/resource/BDFORET). Polygons of forest
and semi-natural vegetation of 0.5 ha or more, photo-interpreted between
2007 and 2018, each with its vegetation type (``TFV_G11``, 11 groups: closed
or open forest of broadleaves / conifers / mixed, poplar plantations, heath,
grassland...) and its dominant species (``ESSENCE``). Corine Land Cover
(build_landcover.py) says how much of the cell is forest; this says which.

Each département is rasterised at 50 m on a Lambert-93 grid aligned on the
25 km cells; the (type, species) pixel counts per cell are cached in
``data/interim/bdforet/``. Per cell, the share of the *whole* 25 km cell
(625 km2) covered by:

- ``forest_broadleaf``, ``forest_conifer``, ``forest_mixed``: closed or
  open forest of that composition (poplar plantations count as broadleaf);
- ``forest_open``: open forest (canopy cover 10-40 %), any composition;
- ``heath``: heathland (lande);
- by dominant species: ``pine_maritime`` (maritime pine), ``pine_aleppo``
  (Aleppo pine), ``pine_other`` (other and mixed pines), ``fir_spruce``
  (fir, spruce, Douglas fir, larch), ``oak_evergreen`` (holm and cork oak).

    python pipeline/build_forest.py

Downloads about 4 GB (kept in ``data/raw/bdforet/``). Writes
``data/interim/forest_cell.parquet``.
"""

import re
import shutil
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
import shapefile
from rasterio.features import rasterize
from rasterio.transform import from_origin

from build_landcover import cell_counts, download, get
from common import BDFORET_DIR, CELL_SIZE, INTERIM

FEED = "https://data.geopf.fr/telechargement/resource/BDFORET"
DOWNLOAD = "https://data.geopf.fr/telechargement/download/BDFORET"
NAME = re.compile(r"BDFORET_2-0__SHP_LAMB93_D\w{3}_\d{4}-\d{2}-\d{2}")
RES = 50  # m
CACHE = INTERIM / "bdforet"

# Normalised TFV_G11 / ESSENCE (see norm) -> feature.
COMPOSITION = {"feuillus": "forest_broadleaf", "coniferes": "forest_conifer",
               "mixte": "forest_mixed"}
SPECIES = {
    "pin maritime": "pine_maritime",
    "pin d'alep": "pine_aleppo",
    "pin sylvestre": "pine_other",
    "pin laricio, pin noir": "pine_other",
    "pin a crochets, pin cembro": "pine_other",
    "pin autre": "pine_other",
    "pins melanges": "pine_other",
    "sapin, epicea": "fir_spruce",
    "douglas": "fir_spruce",
    "meleze": "fir_spruce",
    "chenes sempervirents": "oak_evergreen",
}
FEATURES = ["forest_broadleaf", "forest_conifer", "forest_mixed",
            "forest_open", "heath", "pine_maritime", "pine_aleppo",
            "pine_other", "fir_spruce", "oak_evergreen"]


def norm(s):
    """Lower case, no accents, single spaces."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def releases():
    """Names of the BD Forêt v2 département releases, from the Atom feed."""
    names, page, n_pages = set(), 1, 1
    while page <= n_pages:
        r = get(FEED, params={"page": page})
        n_pages = int(re.search(r'pagecount="(\d+)"', r.text).group(1))
        names |= set(NAME.findall(r.text))
        page += 1
    return sorted(names)


def fetch(name):
    path = BDFORET_DIR / f"{name}.7z"
    download(f"{DOWNLOAD}/{name}/{name}.7z", path)
    return path


def counts(archive):
    """Pixel counts per (cell, TFV_G11, ESSENCE) of one département."""
    with tempfile.TemporaryDirectory() as tmp:
        with py7zr.SevenZipFile(archive) as z:
            z.extract(tmp, targets=[n for n in z.getnames()
                                    if "FORMATION_VEGETALE" in n])
        shp = next(Path(tmp).rglob("FORMATION_VEGETALE.shp"))
        reader = shapefile.Reader(str(shp), encoding="utf-8")
        recs = pd.DataFrame([r.as_dict() for r in reader.iterRecords()])
        pairs = recs[["TFV_G11", "ESSENCE"]].drop_duplicates()
        pairs = pairs.reset_index(drop=True)
        pairs["value"] = np.arange(1, len(pairs) + 1)
        value = recs.merge(pairs, on=["TFV_G11", "ESSENCE"],
                           how="left")["value"].to_numpy()

        xmin, ymin, xmax, ymax = reader.bbox
        x0 = int(np.floor(xmin / CELL_SIZE)) * CELL_SIZE
        ytop = int(np.ceil(ymax / CELL_SIZE)) * CELL_SIZE
        per = CELL_SIZE // RES
        shape = ((ytop - int(np.floor(ymin / CELL_SIZE)) * CELL_SIZE) // RES,
                 (int(np.ceil(xmax / CELL_SIZE)) * CELL_SIZE - x0) // RES)
        assert shape[0] % per == 0 and shape[1] % per == 0
        a = rasterize(
            ((s.__geo_interface__, int(v))
             for s, v in zip(reader.iterShapes(), value)),
            out_shape=shape, transform=from_origin(x0, ytop, RES, RES),
            fill=0, dtype="uint16")
        reader.close()
    c = cell_counts(a, x0, ytop, RES, len(pairs) + 1)
    c = c[c["value"] > 0].merge(pairs, on="value").drop(columns="value")
    return c.rename(columns={"TFV_G11": "tfv", "ESSENCE": "essence"})


def features(c, grid):
    c = c.merge(grid[["ix", "iy", "cell_id"]], on=["ix", "iy"])
    c["share"] = c["n"] / (CELL_SIZE // RES) ** 2
    tfv, essence = c["tfv"].map(norm), c["essence"].map(norm)
    forest = tfv.str.startswith("foret")
    cols = {name: forest & tfv.str.endswith(key)
            for key, name in COMPOSITION.items()}
    cols["forest_broadleaf"] |= tfv.eq("peupleraie")
    cols["forest_open"] = tfv.str.startswith("foret ouverte")
    cols["heath"] = tfv.eq("lande")
    for name in set(SPECIES.values()):
        cols[name] = essence.map(SPECIES).eq(name)
    out = pd.DataFrame({name: c["share"].where(m, 0.)
                        for name, m in cols.items()})
    out["cell_id"] = c["cell_id"]
    out = out.groupby("cell_id").sum()
    out = grid[["cell_id"]].join(out, on="cell_id").fillna(0.)
    return out[["cell_id", *FEATURES]].astype({f: "float32"
                                               for f in FEATURES})


def main():
    names = releases()
    print(f"{len(names)} département releases", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (CACHE / f"{n}.parquet").exists()]
    # The Géoplateforme throttles each connection: download in parallel.
    failed = []
    with ThreadPoolExecutor(8) as pool:
        jobs = {pool.submit(fetch, n): n for n in todo}
        for i, job in enumerate(as_completed(jobs), 1):
            if job.exception() is not None:
                failed.append(jobs[job])
                print(f"  {i}/{len(todo)} {jobs[job]} FAILED: "
                      f"{job.exception()}", flush=True)
                continue
            archive = job.result()
            counts(archive).to_parquet(CACHE / f"{archive.stem}.parquet",
                                       index=False)
            print(f"  {i}/{len(todo)} {archive.stem}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} downloads failed, run again to "
                         f"resume: {failed}")

    c = pd.concat([pd.read_parquet(CACHE / f"{n}.parquet") for n in names])
    seen = (c.groupby([c["tfv"].map(norm), c["essence"].map(norm)])["n"]
            .sum() * RES ** 2 / 1e6).round(0)
    print("area per type and species (km2):")
    print(seen.to_string())
    unknown = set(seen.index.get_level_values(1)) - set(SPECIES) - {
        "nc", "nr", "feuillus", "coniferes", "mixte", "chenes decidus",
        "hetre", "chataignier", "robinier", "peuplier"}
    if unknown:
        print(f"species left out of the species features: {unknown}")

    grid = pd.read_parquet(INTERIM / "grid.parquet")
    out = features(c, grid)
    out.to_parquet(INTERIM / "forest_cell.parquet", index=False)
    print(f"{len(out)} cells; mean share of the cell (%):")
    print((out[FEATURES].mean() * 100).round(2).to_string())


if __name__ == "__main__":
    shutil.rmtree(BDFORET_DIR / "tmp", ignore_errors=True)
    main()
