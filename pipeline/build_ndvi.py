"""MODIS MOD13Q1.061 (250 m, 16-day composites) -> rasters on disk.

Uses NASA AppEEARS (https://appeears.earthdatacloud.nasa.gov/api/). AppEEARS
caps the number of files of a request, and writes one raster per polygon,
date and layer, so per-cell polygons are out of reach. Instead, one *area*
task per year clips the layers to a single polygon, the outline of the
grid. Two tasks per year: NDVI + VI Quality (``submit``), and NIR + MIR
(band 7, 2.1 um) reflectance for the NBR (``submit-nbr``, same dates and
polygon).

Credentials (NASA Earthdata Login), never stored in the repository:
``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` environment variables, or a
``~/.netrc`` entry for ``urs.earthdata.nasa.gov``.

    python pipeline/build_ndvi.py submit      # NDVI tasks -> tasks.json
    python pipeline/build_ndvi.py submit-nbr  # NIR + MIR -> tasks_nbr.json
    python pipeline/build_ndvi.py status
    python pipeline/build_ndvi.py download    # resumable

Keeps every raster (about 25 GB) under ``data/raw/ndvi/rasters/<year>/``;
``build_vegetation.py`` turns them into the vegetation features.
"""

import argparse
import json
import netrc
import os
import time

import pandas as pd
import requests
from pyproj import Transformer
from urllib3.util.retry import Retry

from common import (CELL_SIZE, CRS_L93, INTERIM, LAST_YEAR, NDVI_DIR,
                    WARMUP_YEAR)

API = "https://appeears.earthdatacloud.nasa.gov/api"
PRODUCT = "MOD13Q1.061"
LAYER = "_250m_16_days_NDVI"
TASKS = NDVI_DIR / "tasks.json"
# NIR and MIR (MODIS band 7, 2.1 um) reflectance: NBR = (NIR-MIR)/(NIR+MIR).
NBR_LAYERS = ["_250m_16_days_NIR_reflectance",
              "_250m_16_days_MIR_reflectance"]
TASKS_NBR = NDVI_DIR / "tasks_nbr.json"
RASTERS = NDVI_DIR / "rasters"

# AppEEARS drops connections now and then: retry every call with backoff
# (about 10 min in all) rather than abort a run of several hours.
http = requests.Session()
http.mount("https://", requests.adapters.HTTPAdapter(max_retries=Retry(
    total=8, backoff_factor=5, allowed_methods=None,
    status_forcelist=(429, 500, 502, 503, 504))))


def save(url, headers, dest, size, tries=5):
    """Stream ``url`` to ``dest`` through ``*.part``, renamed once complete,
    so an interrupted run never leaves a truncated file behind."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, tries + 1):
        try:
            with http.get(url, headers=headers, stream=True,
                          timeout=600) as r:
                r.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if part.stat().st_size != size:
                raise IOError(f"{dest.name}: incomplete download")
            part.replace(dest)
            return
        except (requests.RequestException, IOError) as e:
            if attempt == tries:
                raise
            print(f"  {dest.name}: {e}, retry {attempt}/{tries - 1}",
                  flush=True)
            time.sleep(60 * attempt)


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
    r = http.post(f"{API}/login", auth=credentials(), timeout=60)
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
        r = http.post(f"{API}/task", json=task, headers=headers,
                      timeout=300)
        if not r.ok:
            raise SystemExit(f"{year}: {r.status_code} {r.text}")
        tasks[str(year)] = r.json()["task_id"]
        print(year, tasks[str(year)])
        NDVI_DIR.mkdir(parents=True, exist_ok=True)
        TASKS.write_text(json.dumps(tasks, indent=1))


def submit_nbr():
    """One NIR + MIR task per year, on the area and dates of the NDVI task
    of that year, so that the rasters of both line up pixel for pixel."""
    headers = login()
    tasks = json.loads(TASKS.read_text())
    nbr = json.loads(TASKS_NBR.read_text()) if TASKS_NBR.exists() else {}
    for year, task_id in tasks.items():
        if year in nbr:
            continue
        params = http.get(f"{API}/task/{task_id}", headers=headers,
                          timeout=60).json()["params"]
        params["layers"] = [{"product": PRODUCT, "layer": layer}
                            for layer in NBR_LAYERS]
        task = {"task_type": "area", "task_name": f"firegrid_nbr_{year}",
                "params": params}
        r = http.post(f"{API}/task", json=task, headers=headers,
                      timeout=300)
        if not r.ok:
            raise SystemExit(f"{year}: {r.status_code} {r.text}")
        nbr[year] = r.json()["task_id"]
        print(year, nbr[year])
        TASKS_NBR.write_text(json.dumps(nbr, indent=1))


def status():
    headers = login()
    for path in (TASKS, TASKS_NBR):
        if not path.exists():
            continue
        for year, task_id in json.loads(path.read_text()).items():
            info = http.get(f"{API}/task/{task_id}", headers=headers,
                            timeout=60).json()
            print(path.stem, year, task_id, info["status"])


def wait_done(year, task_id):
    """Block until the task is done; returns fresh auth headers."""
    headers = login()
    while (st := http.get(f"{API}/task/{task_id}", headers=headers,
                          timeout=60).json()["status"]) != "done":
        if st == "error":
            raise SystemExit(f"{year}: task {task_id} failed")
        print(f"{year}: {st}, waiting...", flush=True)
        time.sleep(600)
        headers = login()
    return headers


def download():
    """Download every raster of the wanted layers and keep it on disk.

    Existing rasters of the expected size are skipped, so an interrupted
    run resumes where it stopped.
    """
    jobs = [(TASKS, [LAYER, "_250m_16_days_VI_Quality"])]
    if TASKS_NBR.exists():
        jobs.append((TASKS_NBR, NBR_LAYERS))
    for path, layers in jobs:
        for year, task_id in json.loads(path.read_text()).items():
            headers = wait_done(year, task_id)
            files = http.get(f"{API}/bundle/{task_id}", headers=headers,
                             timeout=60).json()["files"]
            tifs = [f for f in files if f["file_name"].endswith(".tif")
                    and any(f"{layer}_doy" in f["file_name"]
                            for layer in layers)]
            out = RASTERS / year
            out.mkdir(parents=True, exist_ok=True)
            todo = [f for f in tifs
                    if not ((out / f["file_name"].split("/")[-1]).exists()
                            and (out / f["file_name"].split("/")[-1])
                            .stat().st_size == f["file_size"])]
            print(f"{path.stem} {year}: {len(tifs)} rasters, "
                  f"{len(todo)} to download "
                  f"({sum(f['file_size'] for f in todo) / 1e9:.2f} GB)",
                  flush=True)
            for f in todo:
                save(f"{API}/bundle/{task_id}/{f['file_id']}", headers,
                     out / f["file_name"].split("/")[-1], f["file_size"])
    total = sum(p.stat().st_size for p in RASTERS.rglob("*.tif"))
    print(f"done: {total / 1e9:.1f} GB in {RASTERS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    actions = {"submit": submit, "submit-nbr": submit_nbr, "status": status,
               "download": download}
    parser.add_argument("action", choices=list(actions))
    actions[parser.parse_args().action]()
