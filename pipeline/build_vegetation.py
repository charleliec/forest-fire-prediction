"""MODIS rasters + BD Forêt fuel map -> vegetation features per (cell, composite).

Inputs:

- the MOD13Q1.061 rasters kept on disk by ``build_ndvi.py download``
  (250 m, 16-day composites, sinusoidal grid): NDVI, VI Quality, NIR and
  MIR (band 7, 2.1 um) reflectance;
- the BD Forêt v2 archives downloaded by ``build_forest.py``;
- the IGN sylvoecoregions (SER, IGN Inventaire forestier,
  https://inventaire-forestier.ign.fr/spip.php?article773), downloaded here:
  the Mediterranean zone is the union of the regions of the grandes regions
  ecologiques J (Mediterranee) and K (Corse).

Steps (each cached under ``data/interim/vegetation/``):

1. Fuel map. Each departement of BD Forêt is rasterised at 25 m in
   Lambert-93, each 25 m pixel gets a fuel class (CLASSES, from the
   vegetation type TFV_G11, the species ESSENCE and the Mediterranean zone,
   see fuel_class), and the 25 m pixels are counted per MODIS pixel (by
   their centre). A MODIS pixel is a *fuel pixel* when one class covers at
   least PURITY of its area; it belongs to the 25 km cell holding its
   centre.
2. Extraction. For every composite, the NDVI and the NBR = (NIR - MIR) /
   (NIR + MIR) of the fuel pixels, set missing where the VI Quality flags
   are bad (quality_ok) or a layer has no data.
2b. Mixture correction (mixture.py): the NDVI and NBR the main class of
   each fuel pixel would have without the rest of the pixel (other fuel
   classes, fields, roads...), from local reference reflectances.
3. Features. Burns are detected on the observed NBR series of each pixel
   (burns); on the corrected values, the reference distribution of each class is the histogram of all the
   usable values of its pixels over the reference composites (training
   years only); each value becomes its percentile in that distribution,
   and the percentiles are summarised per cell (features). A fuel pixel
   with no usable value in a composite (cloud, snow...) takes its value
   observed in the previous composite, if any.

    python pipeline/build_vegetation.py

Writes ``data/interim/fuel_cell.parquet`` (static: cell_id, the share of
the cell covered by each class ``fuel_<class>``, fuel_px,
fuel_pure_share) and ``data/interim/vegetation_cell.parquet`` (cell_id,
composite_start and the features of DYNAMIC). DATA.md describes every
column.
"""

import re
import tempfile
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
import rasterio
import requests
import shapefile
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin

from build_forest import norm
from common import (BDFORET_DIR, CELL_SIZE, CRS_L93, FIRST_YEAR, GEO_DIR,
                    INTERIM, LAST_YEAR, NDVI_DIR, TEST_YEARS, cell_index)

RASTERS = NDVI_DIR / "rasters"
WORK = INTERIM / "vegetation"
FUEL_CACHE = WORK / "fuel_dept"
COMPOSITES = WORK / "composites"
SNOW = WORK / "snow"
SER_URL = "https://inventaire-forestier.ign.fr/IMG/zip/ser_l93.zip"
SER_SHP = GEO_DIR / "ser" / "ser_l93.shp"
MED_GRECO = ("J", "K")

RES = 25            # m, BD Forêt rasterisation
PURITY = 0.75       # share of a MODIS pixel one class must cover
MIN_PX = 10         # usable fuel pixels below which the features are NaN
EXTREME = 0.1       # percentile below which a pixel is "extreme"
DROP = 2700         # NBR drop (x 1e-4) flagging a possible burn
DROP_NDVI = 1000    # NDVI drop (x 1e-4) a burn must come with
# Day of year of the composite start: burns are searched from 18 Feb to the
# composite starting 30 Sep (31 Aug in leap years 29 Sep); with a low sun,
# the NBR of mountain forests falls far below 0 every autumn and winter.
BURN_DOY = (49, 273)
# Snow. A pixel is snowy in a composite when its VI Quality has the snow /
# ice bit, or says cloudy (MODLAND: MODIS often takes snow for cloud) while
# SIM2 has at least SNOW_DEPTH m of snow on average over its cell in the
# composite. For POST_SNOW composites after its last snowy one, a burn
# candidate of the pixel can be flagged but not confirmed: after the melt,
# flattened, still-brown vegetation looks burnt for a few weeks.
SNOW_DEPTH = 0.01   # m
POST_SNOW = 4
BASELINE = 4        # usable composites in the NBR baseline of a pixel
YEAR_COMPOSITES = 23  # MOD13Q1 composites per year (start days 1, 17...)
BURNT_DAYS = {"burnt_1y": 365, "burnt_5y": 1826}
COMPOSITE_DAYS = 16
MISSING = -32768    # int16 code of a missing value in the caches
OFFSET, NBINS = 10000, 20001  # histogram of values -1..1 (x 1e-4)
WORKERS = 6

CLASSES = ["broadleaf_decid", "broadleaf_med", "oak_evergreen",
           "pine_maritime", "pine_aleppo", "pine_mountain",
           "fir_spruce_douglas", "larch", "conifer_mixed", "mixed",
           "open_forest", "heath_med", "heath_other", "grass"]
CODE = {name: k for k, name in enumerate(CLASSES, 1)}  # 0 = not fuel
CLASS_COLUMNS = [f"fuel_{name}" for name in CLASSES]
CONIFERS = {
    "pin maritime": "pine_maritime", "pin d'alep": "pine_aleppo",
    "pin sylvestre": "pine_mountain", "pin laricio, pin noir": "pine_mountain",
    "pin a crochets, pin cembro": "pine_mountain",
    "pins melanges": "pine_mountain", "pin autre": "pine_mountain",
    "sapin, epicea": "fir_spruce_douglas", "douglas": "fir_spruce_douglas",
    "meleze": "larch", "coniferes": "conifer_mixed",
}
LAYERS = {"NDVI": "ndvi", "VI_Quality": "qa", "NIR_reflectance": "nir",
          "MIR_reflectance": "mir"}
DYNAMIC = ["ndvi_pct", "ndvi_pct_p10", "ndvi_extreme", "nbr_pct",
           "nbr_pct_p10", "nbr_extreme", "dry_gap", *BURNT_DAYS, "vi_px",
           "vi_valid_share", "vi_prev_share"]
REF_YEARS = [y for y in range(FIRST_YEAR, LAST_YEAR + 1)
             if y not in TEST_YEARS]


def fuel_class(tfv, essence, med):
    """Fuel class code of a BD Forêt (TFV_G11, ESSENCE), both normalised by
    ``norm``, inside (``med``) or outside the Mediterranean zone."""
    if tfv.startswith("foret ouverte") and "sans couvert" not in tfv:
        name = "open_forest"
    elif tfv == "foret fermee feuillus":
        if essence == "chenes sempervirents":
            name = "oak_evergreen"
        elif essence == "feuillus" and med:
            name = "broadleaf_med"
        else:
            name = "broadleaf_decid"
    elif tfv == "foret fermee coniferes":
        name = CONIFERS.get(essence, "conifer_mixed")
    elif tfv == "foret fermee mixte":
        name = "mixed"
    elif tfv == "lande":
        name = "heath_med" if med else "heath_other"
    elif tfv == "formation herbacee":
        name = "grass"
    else:  # poplar plantations, forest without tree cover
        return 0
    return CODE[name]


# --- MODIS grid ----------------------------------------------------------

def composites():
    """{composite start date: {layer: path}}, NDVI and VI Quality present.

    A year's folder also holds the last composite of the previous year:
    each composite is kept once."""
    found = {}
    for path in RASTERS.rglob("*.tif"):
        m = re.search(r"_250m_16_days_(\w+?)_doy(\d{4})(\d{3})", path.name)
        if m is None or m.group(1) not in LAYERS:
            continue
        date = (datetime(int(m.group(2)), 1, 1)
                + timedelta(days=int(m.group(3)) - 1))
        found.setdefault(date, {}).setdefault(LAYERS[m.group(1)], path)
    return {d: found[d] for d in sorted(found)
            if {"ndvi", "qa"} <= set(found[d])}


def modis_grid():
    """CRS, transform and shape of the MODIS rasters (all on one grid)."""
    first = next(iter(composites().values()))["ndvi"]
    with rasterio.open(first) as src:
        return src.crs.to_wkt(), src.transform, src.shape


def pixel_cells(pix, transform, width, crs):
    """cell_id of the 25 km cell holding each MODIS pixel centre (-1: none)."""
    rows, cols = np.divmod(pix, width)
    xs, ys = transform * (cols + .5, rows + .5)
    x, y = Transformer.from_crs(crs, CRS_L93, always_xy=True).transform(xs, ys)
    ix, iy = cell_index(x, y)
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    lookup = pd.Series(grid["cell_id"].to_numpy(),
                       index=pd.MultiIndex.from_arrays([grid["ix"],
                                                        grid["iy"]]))
    cell = lookup.reindex(pd.MultiIndex.from_arrays([ix, iy]))
    return cell.fillna(-1).to_numpy(dtype="int64")


# --- 1. fuel map ---------------------------------------------------------

def med_shapes():
    """GeoJSON geometries of the Mediterranean sylvoecoregions (Lambert-93)."""
    if not SER_SHP.exists():
        SER_SHP.parent.mkdir(parents=True, exist_ok=True)
        archive = SER_SHP.parent / "ser_l93.zip"
        r = requests.get(SER_URL, timeout=300)
        r.raise_for_status()
        archive.write_bytes(r.content)
        zipfile.ZipFile(archive).extractall(SER_SHP.parent)
    reader = shapefile.Reader(str(SER_SHP), encoding="latin-1")
    return [s.__geo_interface__ for s, rec in
            zip(reader.iterShapes(), reader.iterRecords())
            if rec["codeser"].startswith(MED_GRECO)]


def dept_counts(args):
    """25 m pixel counts per (MODIS pixel, fuel class) of one departement."""
    archive, med, (crs, transform, shape) = args
    out = FUEL_CACHE / f"{archive.stem}.parquet"
    if out.exists():
        return
    with tempfile.TemporaryDirectory() as tmp:
        with py7zr.SevenZipFile(archive) as z:
            z.extract(tmp, targets=[n for n in z.getnames()
                                    if "FORMATION_VEGETALE" in n])
        shp = next(Path(tmp).rglob("FORMATION_VEGETALE.shp"))
        reader = shapefile.Reader(str(shp), encoding="utf-8")
        recs = pd.DataFrame([r.as_dict() for r in reader.iterRecords()])
        recs = recs[["TFV_G11", "ESSENCE"]].fillna("").astype(str)
        pairs = recs.drop_duplicates().reset_index(drop=True)
        pairs["value"] = np.arange(1, len(pairs) + 1)
        value = recs.merge(pairs, on=["TFV_G11", "ESSENCE"],
                           how="left")["value"].to_numpy()
        xmin, ymin, xmax, ymax = reader.bbox
        x0 = int(np.floor(xmin / RES)) * RES
        ytop = int(np.ceil(ymax / RES)) * RES
        out_shape = (int(np.ceil((ytop - ymin) / RES)),
                     int(np.ceil((xmax - x0) / RES)))
        tr = from_origin(x0, ytop, RES, RES)
        a = rasterize(((s.__geo_interface__, int(v)) for s, v in
                       zip(reader.iterShapes(), value)),
                      out_shape=out_shape, transform=tr, fill=0,
                      dtype="uint16")
        reader.close()
    in_med = rasterize(((g, 1) for g in med), out_shape=out_shape,
                       transform=tr, fill=0, dtype="uint8")
    lut = np.zeros((len(pairs) + 1, 2), dtype="int8")
    for t, e, v in pairs.itertuples(index=False):
        for m in (0, 1):
            lut[v, m] = fuel_class(norm(t), norm(e), bool(m))
    cls = lut[a, in_med]
    del a, in_med
    rows, cols = np.nonzero(cls)
    c = cls[rows, cols]
    xs, ys = Transformer.from_crs(CRS_L93, crs, always_xy=True).transform(
        x0 + (cols + .5) * RES, ytop - (rows + .5) * RES)
    col = np.floor((xs - transform.c) / transform.a).astype("int64")
    row = np.floor((ys - transform.f) / transform.e).astype("int64")
    ok = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    key = (row[ok] * shape[1] + col[ok]) * 16 + c[ok]
    key, n = np.unique(key, return_counts=True)
    pd.DataFrame({"pix": key // 16, "cls": (key % 16).astype("int8"),
                  "n": n.astype("int32")}).to_parquet(out, index=False)
    print(f"  {archive.stem}", flush=True)


def fuel_map(grid):
    """Fuel pixels (pix, cls, cell_id) and the static per-cell table."""
    crs, transform, shape = grid
    archives = sorted(BDFORET_DIR.glob("BDFORET_*.7z"))
    if not archives:
        raise SystemExit("no BD Forêt archive: run build_forest.py first")
    FUEL_CACHE.mkdir(parents=True, exist_ok=True)
    med = med_shapes()
    with ProcessPoolExecutor(WORKERS) as pool:
        list(pool.map(dept_counts, [(a, med, grid) for a in archives]))

    # A MODIS pixel on a departement border gets counts from both.
    c = pd.concat([pd.read_parquet(FUEL_CACHE / f"{a.stem}.parquet")
                   for a in archives])
    c = c.groupby(["pix", "cls"], as_index=False)["n"].sum()
    fuel_n = c.groupby("pix")["n"].sum()
    top = c.sort_values("n").drop_duplicates("pix", keep="last")
    top = top.set_index("pix").join(fuel_n.rename("fuel_n"))
    # 25 m pixels in a MODIS pixel (both grids are equal-area enough).
    full = abs(transform.a * transform.e) / RES ** 2
    top["pure"] = top["n"] >= PURITY * full
    top["cell_id"] = pixel_cells(top.index.to_numpy(), transform, shape[1],
                                 crs)
    top = top[top["cell_id"] >= 0]

    pixels = top[top["pure"]].reset_index()[["pix", "cls", "cell_id"]]
    pixels = pixels.astype({"cls": "int8", "cell_id": "int32"})
    cells = pd.read_parquet(INTERIM / "grid.parquet")[["cell_id"]]
    top["pure_n"] = top["fuel_n"].where(top["pure"], 0)
    per_cell = top.groupby("cell_id")[["pure", "pure_n", "fuel_n"]].sum()
    per_cell["fuel_px"] = per_cell["pure"]
    per_cell["fuel_pure_share"] = per_cell["pure_n"] / per_cell["fuel_n"]
    # Share of the whole cell covered by each class (all the 25 m squares,
    # pure pixels or not, counted in the cell of their MODIS pixel).
    c["cell_id"] = top["cell_id"].reindex(c["pix"]).to_numpy()
    c = c[c["cell_id"] >= 0]
    shares = (c.groupby(["cell_id", "cls"])["n"].sum().unstack(fill_value=0)
              .reindex(columns=range(1, len(CLASSES) + 1), fill_value=0)
              * RES ** 2 / CELL_SIZE ** 2)
    shares.columns = CLASS_COLUMNS
    static = (cells.join(shares, on="cell_id")
              .join(per_cell[["fuel_px", "fuel_pure_share"]], on="cell_id"))
    static[CLASS_COLUMNS] = static[CLASS_COLUMNS].fillna(0.)
    static["fuel_px"] = static["fuel_px"].fillna(0).astype("int32")
    static = static.astype({c: "float32" for c in
                            CLASS_COLUMNS + ["fuel_pure_share"]})
    pixels.to_parquet(WORK / "fuel_pixels.parquet", index=False)
    static.to_parquet(INTERIM / "fuel_cell.parquet", index=False)

    print(f"{len(pixels)} fuel pixels; per class:")
    print(pixels["cls"].map(dict(enumerate(CLASSES, 1))).value_counts()
          .to_string())
    print(f"fuel pixels per cell: "
          f"{static['fuel_px'].quantile([0, .1, .5, .9]).to_dict()}; "
          f"cells with < {MIN_PX}: {(static['fuel_px'] < MIN_PX).sum()}")
    return pixels


# --- 2. extraction -------------------------------------------------------

def quality_ok(qa):
    """Usable pixels from the MOD13Q1 VI Quality bits."""
    qa = qa.astype("uint32")
    return (((qa & 0b11) <= 1)                 # MODLAND: VI produced
            & (((qa >> 2) & 0b1111) <= 0b1011)  # usefulness not the lowest
            & (((qa >> 6) & 0b11) != 0b11)     # aerosol not high
            & ((qa >> 10) & 1 == 0)            # no mixed clouds
            & ((qa >> 14) & 1 == 0)            # no snow / ice
            & ((qa >> 15) & 1 == 0))           # no shadow


def read(path, pix, shape):
    with rasterio.open(path) as src:
        if src.shape != shape:
            raise ValueError(f"{path.name}: not on the common grid")
        return src.read(1).ravel()[pix], src.nodata


def extract_one(args):
    """Masked NDVI and NBR of the fuel pixels for one composite."""
    date, layers, pix, shape = args
    out = COMPOSITES / f"{date:%Y-%m-%d}.npz"
    nbr_ready = {"nir", "mir"} <= set(layers)
    if out.exists():
        with np.load(out) as z:
            if (len(z["ndvi"]) == len(pix) and int(z["checksum"])
                    == int(pix.sum()) and (z["has_nbr"] or not nbr_ready)):
                return
    ndvi, nodata = read(layers["ndvi"], pix, shape)
    qa, _ = read(layers["qa"], pix, shape)
    usable = quality_ok(qa) & (ndvi != nodata)
    nbr = np.full(len(pix), MISSING, dtype="int16")
    if nbr_ready:
        nir, nir_nodata = read(layers["nir"], pix, shape)
        mir, mir_nodata = read(layers["mir"], pix, shape)
        nir, mir = nir.astype("float64"), mir.astype("float64")
        usable &= (nir != nir_nodata) & (mir != mir_nodata) & (nir + mir > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.round((nir - mir) / (nir + mir) * 1e4)
        nbr = np.where(usable, v, MISSING).astype("int16")
    ndvi = np.where(usable, np.clip(ndvi, -OFFSET, OFFSET),
                    MISSING).astype("int16")
    np.savez(out, ndvi=ndvi, nbr=nbr, has_nbr=nbr_ready,
             checksum=pix.sum())


def extract(dates, pix, shape):
    COMPOSITES.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(WORKERS) as pool:
        list(pool.map(extract_one, [(d, layers, pix, shape)
                                    for d, layers in dates.items()]))
    no_nbr = [d for d, layers in dates.items()
              if not {"nir", "mir"} <= set(layers)]
    if no_nbr:
        print(f"WARNING: {len(no_nbr)} composites without NIR / MIR "
              f"({no_nbr[0]:%Y-%m-%d} .. {no_nbr[-1]:%Y-%m-%d}): NBR "
              "features NaN and burns not detected there; rerun once "
              "build_ndvi.py download is done", flush=True)


def snow_one(args):
    """VI Quality snow and cloudy flags of the fuel pixels, one composite."""
    date, path, pix, shape = args
    out = SNOW / f"{date:%Y-%m-%d}.npz"
    if out.exists():
        with np.load(out) as z:
            if int(z["n"]) == len(pix) and int(z["checksum"]) == int(
                    pix.sum()):
                return
    qa, nodata = read(path, pix, shape)
    ok = qa != nodata
    qa = qa.astype("uint32")
    np.savez(out, snow=np.packbits(ok & ((qa >> 14) & 1 == 1)),
             cloudy=np.packbits(ok & ((qa & 0b11) >= 0b10)), n=len(pix),
             checksum=pix.sum())


def extract_snow(dates, pix, shape):
    SNOW.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(WORKERS) as pool:
        list(pool.map(snow_one, [(d, layers["qa"], pix, shape)
                                 for d, layers in dates.items()]))


def load_snow(date):
    """Snow and cloudy VI Quality flags of the fuel pixels."""
    with np.load(SNOW / f"{date:%Y-%m-%d}.npz") as z:
        n = int(z["n"])
        return (np.unpackbits(z["snow"])[:n].astype(bool),
                np.unpackbits(z["cloudy"])[:n].astype(bool))


def cell_snow(dates, cells):
    """SIM2 snow depth (m) per (composite, cell): the mean over the 16 days
    of the composite of the cell's daily snow depth."""
    w = pd.read_parquet(INTERIM / "weather_cell_day.parquet",
                        columns=["cell_id", "date", "snow_depth"])
    out = np.zeros((len(dates), len(cells)), dtype="float32")
    cols = pd.Index(cells)
    for t, date in enumerate(dates):
        d = w[(w["date"] >= date) & (w["date"] < date + timedelta(
            days=COMPOSITE_DAYS))]
        m = d.groupby("cell_id")["snow_depth"].mean()
        out[t] = m.reindex(cols).fillna(0).to_numpy()
    return out


def load(date, corrected=False):
    """NDVI, NBR of the fuel pixels (x 1e4, MISSING where unusable) and
    whether the NBR exists; ``corrected``: after the mixture correction
    (mixture.py), else as observed."""
    with np.load(COMPOSITES / f"{date:%Y-%m-%d}.npz") as z:
        ndvi, nbr, has_nbr = z["ndvi"], z["nbr"], bool(z["has_nbr"])
    if corrected:
        with np.load(WORK / "corrected" / f"{date:%Y-%m-%d}.npz") as z:
            ndvi, nbr = z["ndvi"], z["nbr"]
    return ndvi, nbr, has_nbr


# --- 3. features ---------------------------------------------------------

def cell_excess(drop, ci, nc):
    """``drop`` minus the median drop of its cell, when positive."""
    ok = ~np.isnan(drop)
    med = group_quantile(ci[ok], drop[ok], .5, nc)
    med[np.bincount(ci[ok], minlength=nc) < MIN_PX] = 0.
    return drop - np.clip(np.nan_to_num(med), 0, None)[ci]


def burns(dates, ci, cells):
    """Burn detection on the NBR and NDVI series of each fuel pixel.

    Baselines per pixel and composite:

    - recent: the median of its last BASELINE usable values (at least 2)
      that were not candidates, for the NBR and for the NDVI;
    - seasonal: the lowest of its usable NBR values at the same time of
      year (same composite period, or one before / after) in the previous
      two years (``YEAR_COMPOSITES`` composites per year). Leaf fall,
      curing of grass and other yearly cycles drop the NBR as fast as a
      fire, but every year: this baseline rules them out.

    The NBR drop of a pixel below each baseline is taken *in excess of* the
    median drop of the usable fuel pixels of its cell (when that median is
    positive and rests on at least MIN_PX pixels): a fire is local, while
    haze, low winter sun or an early leaf fall lower a whole region.

    A pixel is a *candidate* in a composite where its NBR is DROP below
    both baselines (in excess of the cell) and its NDVI DROP_NDVI below its
    recent NDVI baseline (a burn removes green leaves; winter artefacts
    lower the NBR of still-green canopies). The burn is *confirmed* by the
    pixel's next usable composite holding another observation (the last
    composite of a year and the first of the next share 3 days, so they
    can hold the same observation, then their values are equal) if that
    composite passes the same tests against the candidate's recent
    baselines and its own seasonal baseline; otherwise it was a false
    alarm (e.g. a residual cloud). Candidate values never enter the recent
    baselines; after a confirmed burn they restart from post-fire values.
    With no seasonal baseline (the first year of data), nothing is
    detected.

    Only the composites starting within BURN_DOY are used (as candidates,
    confirmations and baselines): in autumn and winter, with a low sun,
    the NBR of mountain forests falls far below 0 while their NDVI stays
    high. A burn of that season is detected at the first composite of
    BURN_DOY where its scar still shows. A pending candidate waits across
    the winter for its confirmation.

    After snow (snowy pixel, see POST_SNOW), a candidate waits until the
    pixel's POST_SNOW composites after its last snowy one have passed: only
    a composite past them can confirm it (or dismiss it), and the values
    the pixel shows while it waits do not enter its recent baselines. The
    browning after the melt is gone by then; a burn scar remains.

    ``cells``: cell_id of each value of ``ci``. Returns the candidates of
    each composite (pixel positions) and the burns, (pixel position, start
    of the burn composite, start of the confirming composite).
    """
    n, nc = len(ci), ci.max() + 1
    buf = {k: np.full((BASELINE, n), np.nan, dtype="float32")
           for k in ("nbr", "ndvi")}
    # NBR of the last two years of composites, by composite index.
    past = np.full((2 * YEAR_COMPOSITES + 2, n), MISSING, dtype="int16")
    lags = [k * YEAR_COMPOSITES + d for k in (1, 2) for d in (-1, 0, 1)]
    pending = np.zeros(n, dtype=bool)
    pend = {k: np.full(n, np.nan, dtype="float32")
            for k in ("base", "nbase", "nbr", "ndvi")}
    pend_t = np.zeros(n, dtype="int32")
    days = list(dates)
    candidates, events = {}, []
    snow_depth = cell_snow(days, cells)
    last_snow = np.full(n, -POST_SNOW - 1, dtype="int32")

    def push(mask, vals):
        for k, v in vals.items():
            buf[k][:-1, mask] = buf[k][1:, mask]
            buf[k][-1, mask] = v[mask]

    def median(b):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(b, axis=0)
        m[np.isnan(b).sum(axis=0) > BASELINE - 2] = np.nan
        return m

    for t, date in enumerate(days):
        ndvi, nbr, has_nbr = load(date)
        past[t % len(past)] = nbr
        snow, cloudy = load_snow(date)
        last_snow[snow | (cloudy & (snow_depth[t][ci] >= SNOW_DEPTH))] = t
        if not has_nbr or not (BURN_DOY[0] <= date.timetuple().tm_yday
                               <= BURN_DOY[1]):
            candidates[date] = np.array([], dtype="int64")
            continue
        v = np.where(nbr == MISSING, np.nan, nbr).astype("float32")
        nv = np.where(ndvi == MISSING, np.nan, ndvi).astype("float32")
        same_obs = pending & (v == pend["nbr"]) & (nv == pend["ndvi"])
        ok = ~np.isnan(v) & ~same_obs
        base, nbase = median(buf["nbr"]), median(buf["ndvi"])
        same_time = np.stack([past[(t - lag) % len(past)] for lag in lags
                              if t - lag >= 0] or [np.full(n, MISSING)])
        same_time = np.where(same_time == MISSING, np.nan, same_time)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            seasonal = np.nanmin(same_time, axis=0)
        below_season = ok & (cell_excess(seasonal - v, ci, nc) >= DROP)
        context = cell_excess(base - v, ci, nc) - (base - v)
        drop = (below_season & (base - v + context >= DROP)
                & (nbase - nv >= DROP_NDVI))
        # Just after snow: a pending candidate waits (no confirmation, no
        # dismissal, its values stay out of the baselines).
        waiting = pending & (t - last_snow <= POST_SNOW)
        confirm = (pending & ~waiting & below_season
                   & (pend["base"] - v + context >= DROP)
                   & (pend["nbase"] - nv >= DROP_NDVI))
        new = drop & ~pending
        for i in np.flatnonzero(confirm):
            events.append((i, days[pend_t[i]], date))
        candidates[date] = np.flatnonzero(drop | confirm | same_obs)
        pending = (pending & ~(ok & ~waiting)) | new
        for k, val in (("base", base), ("nbase", nbase), ("nbr", v),
                       ("ndvi", nv)):
            pend[k][new] = val[new]
        pend_t[new] = t
        vals = {"nbr": v, "ndvi": nv}
        push(ok & ~new & ~confirm & ~waiting, vals)
        for b in buf.values():
            b[:, confirm] = np.nan
        push(confirm, vals)
    events = pd.DataFrame(events, columns=["pos", "burn_start",
                                           "known_start"])
    return candidates, events


def excluded(date, candidates, events, n):
    """Pixels left out of the features of composite ``date``: candidate
    burns, and burns confirmed by then within the last 5 years."""
    out = np.zeros(n, dtype=bool)
    out[candidates[date]] = True
    end = date + timedelta(days=COMPOSITE_DAYS - 1)
    recent = events[(events["known_start"] <= date) & (
        events["burn_start"] > end - timedelta(days=BURNT_DAYS["burnt_5y"]))]
    out[recent["pos"].to_numpy()] = True
    return out


def is_reference(date):
    end = date + timedelta(days=COMPOSITE_DAYS - 1)
    return date.year in REF_YEARS and end.year in REF_YEARS


def reference(dates, cls, candidates, events):
    """Class histograms of NDVI and NBR over the reference composites, and
    the mid-rank percentile tables F[class, value + OFFSET]."""
    n = len(cls)
    hist = {"ndvi": np.zeros((len(CLASSES) + 1) * NBINS, dtype="int64"),
            "nbr": np.zeros((len(CLASSES) + 1) * NBINS, dtype="int64")}
    for date in dates:
        if not is_reference(date):
            continue
        ndvi, nbr, _ = load(date, corrected=True)
        keep = ~excluded(date, candidates, events, n)
        for name, v in (("ndvi", ndvi), ("nbr", nbr)):
            m = keep & (v != MISSING)
            hist[name] += np.bincount(
                cls[m].astype("int64") * NBINS + v[m] + OFFSET,
                minlength=len(hist[name]))
    tables = {}
    for name, h in hist.items():
        h = h.reshape(len(CLASSES) + 1, NBINS).astype("float64")
        total = h.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            tables[name] = ((np.cumsum(h, axis=1) - h / 2)
                            / total).astype("float32")
    np.savez(WORK / "reference.npz", classes=np.array(CLASSES),
             ndvi=hist["ndvi"].reshape(len(CLASSES) + 1, NBINS),
             nbr=hist["nbr"].reshape(len(CLASSES) + 1, NBINS))
    counts = pd.DataFrame({k: v.reshape(len(CLASSES) + 1, NBINS)[1:]
                           .sum(axis=1) for k, v in hist.items()},
                          index=CLASSES)
    print("reference observations per class:")
    print(counts.to_string())
    return tables


def group_quantile(g, v, q, n):
    """Per-group ``q`` quantile (linear interpolation) of ``v``."""
    order = np.lexsort((v, g))
    g, v = g[order], v[order]
    counts = np.bincount(g, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out = np.full(n, np.nan)
    has = counts > 0
    pos = q * (counts[has] - 1)
    lo = np.floor(pos).astype("int64")
    i = starts[has] + lo
    j = np.minimum(i + 1, starts[has] + counts[has] - 1)
    out[has] = v[i] + (pos - lo) * (v[j] - v[i])
    return out


def features(dates, pixels, tables, candidates, events):
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    cells = grid["cell_id"].to_numpy()
    nc = len(cells)
    ci = pd.Index(cells).get_indexer(pixels["cell_id"])
    cls = pixels["cls"].to_numpy().astype("int64")
    n = len(cls)
    fuel_px = np.bincount(ci, minlength=nc)
    parts = []
    prev = None  # (observed, ndvi, nbr) of the previous composite
    for date in dates:
        ndvi, nbr, has_nbr = load(date, corrected=True)
        excl = excluded(date, candidates, events, n)
        observed = (ndvi != MISSING) & ~excl
        # No usable value (cloud, snow, quality, missing band): the value
        # observed in the previous composite, if any. Burnt pixels stay out.
        current = (observed, ndvi, nbr)
        from_prev = np.zeros(n, dtype=bool)
        if prev is not None:
            from_prev = (ndvi == MISSING) & ~excl & prev[0]
            ndvi = np.where(from_prev, prev[1], ndvi)
            nbr = np.where(from_prev, prev[2], nbr)
        # Only observed values carry over: never twice in a row.
        prev = current
        use = observed | from_prev
        vi_px = np.bincount(ci[use], minlength=nc)
        row = {"vi_px": vi_px}
        with np.errstate(invalid="ignore", divide="ignore"):
            row["vi_valid_share"] = np.where(fuel_px > 0, vi_px / fuel_px, 0.)
            row["vi_prev_share"] = np.where(
                vi_px > 0, np.bincount(ci[from_prev], minlength=nc) / vi_px,
                0.)
            for name, v in (("ndvi", ndvi), ("nbr", nbr)):
                if name == "nbr" and not has_nbr:
                    for s in ("pct", "pct_p10", "extreme"):
                        row[f"nbr_{s}"] = np.full(nc, np.nan)
                    continue
                pct = tables[name][cls[use], v[use].astype("int64") + OFFSET]
                row[f"{name}_pct"] = np.bincount(
                    ci[use], weights=pct, minlength=nc) / vi_px
                row[f"{name}_pct_p10"] = group_quantile(ci[use], pct, .1, nc)
                row[f"{name}_extreme"] = np.bincount(
                    ci[use], weights=pct < EXTREME, minlength=nc) / vi_px
            row["dry_gap"] = row["nbr_pct"] - row["ndvi_pct"]
            end = date + timedelta(days=COMPOSITE_DAYS - 1)
            known = events[events["known_start"] <= date]
            for col, span in BURNT_DAYS.items():
                burnt = np.zeros(n, dtype=bool)
                burnt[known.loc[known["burn_start"] > end - timedelta(
                    days=span), "pos"].to_numpy()] = True
                row[col] = np.where(fuel_px > 0, np.bincount(
                    ci[burnt], minlength=nc) / fuel_px, np.nan)
        df = pd.DataFrame(row)
        thin = vi_px < MIN_PX
        for col in ("ndvi_pct", "ndvi_pct_p10", "ndvi_extreme", "nbr_pct",
                    "nbr_pct_p10", "nbr_extreme", "dry_gap"):
            df.loc[thin, col] = np.nan
        df.insert(0, "composite_start", date)
        df.insert(0, "cell_id", cells)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.astype({c: "float32" for c in DYNAMIC
                       if c != "vi_px"} | {"vi_px": "int32"})


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    dates = composites()
    gaps = np.diff(np.array(list(dates), dtype="datetime64[D]")).astype(int)
    if (gaps > COMPOSITE_DAYS).any():
        print(f"WARNING: gaps in the composites: {gaps.max()} days")
    print(f"{len(dates)} composites, {min(dates):%Y-%m-%d} .. "
          f"{max(dates):%Y-%m-%d}", flush=True)
    grid = modis_grid()
    pixels = fuel_map(grid)
    pix = pixels["pix"].to_numpy()
    extract(dates, pix, grid[2])
    extract_snow(dates, pix, grid[2])
    import mixture  # step 2b; imports this module
    mixture.run(dates, pixels, grid)
    cells = pd.read_parquet(INTERIM / "grid.parquet")["cell_id"].to_numpy()
    ci = pd.Index(cells).get_indexer(pixels["cell_id"])
    candidates, events = burns(dates, ci, cells)
    print(f"{len(events)} burnt fuel pixels detected "
          f"({len(events) * abs(grid[1].a * grid[1].e) / 1e6:.0f} km2); "
          "per year of the burn:")
    print(events.groupby(events["burn_start"].dt.year).size().to_string())
    events.assign(pix=pix[events["pos"]]).to_parquet(
        WORK / "burns.parquet", index=False)
    cls = pixels["cls"].to_numpy()
    tables = reference(dates, cls, candidates, events)
    out = features(dates, pixels, tables, candidates, events)
    out.to_parquet(INTERIM / "vegetation_cell.parquet", index=False)
    print(f"vegetation: {len(out)} rows; NaN share: "
          f"{out.isna().mean().round(3).to_dict()}")


if __name__ == "__main__":
    main()
