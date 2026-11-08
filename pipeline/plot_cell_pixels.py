"""Pixel-level view of one 25 km cell, to judge the vegetation data by eye.

For a (day, cell), the composite the rows of that day use (the last one
ended by then) and 12 maps of the cell:

1. Sentinel-2 L2A true colour (10 m, shown at 20 m; Earth Search, AWS open
   data): the clearest scene over the cell within the composite period
   +- SEARCH_DAYS;
2. Corine Land Cover 2018 (100 m), grouped;
3. BD Forêt v2 vegetation type (TFV_G11) and 4. dominant species
   (ESSENCE), rasterised at 25 m;
5. fuel class at 25 m (build_vegetation.fuel_class) and 6. fuel class of
   the fuel pixels (250 m MODIS pixels with one class >= 75 %);
7. MODIS VI Quality of the composite: why each pixel is (un)usable;
8. status of each 250 m pixel in the features (not fuel, mixed, burnt,
   no value, carried over from the previous composite, corrected...);
9. observed MODIS NDVI of every pixel;
10. mixture correction of the NDVI of the fuel pixels;
11-12. NDVI and NBR class percentiles of the fuel pixels, as the features
   use them.

By default the summer and winter example days of plot_vegetation.py, each
with its cell with most fires among the cells with at least MIN_FUEL_PX fuel
pixels (ties: most fuel pixels).

    python pipeline/plot_cell_pixels.py [--seed 0] [--view 2023-08-10:749 ...]

Writes ``data/processed/cell_pixels_<day>_<cell>.png``; caches under
``data/interim/vegetation/cellview/`` (Sentinel-2 image, BD Forêt rasters).
"""

import argparse
import json
import tempfile
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py7zr
import rasterio
import requests
import shapefile
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (LinearSegmentedColormap, Normalize,
                               TwoSlopeNorm, to_rgba)
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, reproject

import build_vegetation as bv
from build_forest import norm
from build_landcover import CODES as CLC_CODES
from build_landcover import NODATA as CLC_NODATA
from build_landcover import TIF as CLC_TIF
from common import CELL_SIZE, CRS_L93, INTERIM, PROCESSED
from plot_map import INK, ramp, style_colorbar
from plot_vegetation import (CATEGORICAL, DRY, LABELS, MUTED, NAN_C,
                             OTHER_C, example_days)

CACHE = bv.WORK / "cellview"
STAC = "https://earth-search.aws.element84.com/v1/search"
SEARCH_DAYS = 10    # days searched for Sentinel-2 around the composite
S2_RES = 20         # m, Sentinel-2 display resolution
FINE = 25           # m, BD Forêt / Corine display resolution
MODIS_RES = 50      # m, display resolution of the 250 m pixels
MIN_FUEL_PX = 1000  # fuel pixels of a default cell
CELL_LINE = "#ffffff"
NONE_C = "#000000"      # nothing mapped / not fuel
LOW_C = "#55544f"       # some fuel, under 75 %
PALE_C = "#d9d7cf"      # BD Forêt polygon without species
BLUE, ORANGE, GREEN, YELLOW, PINK, DGREEN, VIOLET = CATEGORICAL


def lighter(c, k=.55):
    r, g, b, _ = to_rgba(c)
    return ((1 - k) * r + k, (1 - k) * g + k, (1 - k) * b + k, 1.)


# Corine level-3 codes -> displayed groups (fixed colours).
CLC_GROUPS = [
    ("Urban, roads, industry", PINK, lambda c: c < 200),
    ("Arable land", YELLOW, lambda c: c in (211, 212, 213, 241, 242)),
    ("Vineyards, orchards, olives", VIOLET, lambda c: 221 <= c <= 223),
    ("Pastures, farm mosaics", ORANGE, lambda c: c in (231, 243, 244)),
    ("Forest", DGREEN, lambda c: 311 <= c <= 313),
    ("Grassland, heath, maquis", GREEN, lambda c: 321 <= c <= 324),
    ("Rocks, sand, sparse", OTHER_C, lambda c: 331 <= c <= 335),
    ("Wetlands, water", BLUE, lambda c: c >= 400),
]
# BD Forêt TFV_G11 (normalised) -> displayed groups (fixed colours).
TFV_GROUPS = [
    ("Closed broadleaf forest", GREEN, ("foret fermee feuillus",)),
    ("Closed conifer forest", DGREEN, ("foret fermee coniferes",)),
    ("Closed mixed forest", BLUE, ("foret fermee mixte",)),
    ("Open forest (10-40 % trees)", YELLOW,
     ("foret ouverte feuillus", "foret ouverte coniferes",
      "foret ouverte mixte")),
    ("Forest without trees (cut, burnt)", ORANGE,
     ("foret fermee sans couvert arbore", "foret ouverte sans couvert arbore")),
    ("Heath (lande)", PINK, ("lande",)),
    ("Herbaceous", VIOLET, ("formation herbacee",)),
    ("Poplar plantation, other", OTHER_C, ()),
]
SPECIES = {
    "pin maritime": "Maritime pine", "pin d'alep": "Aleppo pine",
    "pin sylvestre": "Scots pine", "pin laricio, pin noir":
    "Laricio, black pine", "pin a crochets, pin cembro":
    "Mountain, stone pine", "pin autre": "Other pine", "pins melanges":
    "Mixed pines", "sapin, epicea": "Fir, spruce", "douglas": "Douglas fir",
    "meleze": "Larch", "coniferes": "Conifers, none dominant",
    "chenes sempervirents": "Holm, cork oak", "chenes decidus":
    "Deciduous oaks", "hetre": "Beech", "chataignier": "Chestnut",
    "robinier": "Black locust", "peuplier": "Poplar",
    "feuillus": "Broadleaves, none dominant", "mixte":
    "Broadleaves and conifers",
}

# Status of a 250 m pixel in the features.
STATUS = [
    ("No BD Forêt fuel", NONE_C),
    ("Fuel under 75 % of the pixel", LOW_C),
    ("75 % fuel, no class at 75 %: excluded", BLUE),
    ("Burnt (5 y) or burn candidate: excluded", ORANGE),
    ("No usable value here or in the previous composite", NAN_C),
    ("Value of the previous composite (cloud...)", lighter(GREEN)),
    ("Usable, mixture-corrected", GREEN),
    ("Usable, observed value kept (no correction)", YELLOW),
]
NO_FUEL, LOW_FUEL, MIXED, BURNT, NO_VALUE, PREVIOUS, CORRECTED, KEPT = \
    range(len(STATUS))
# MODIS VI Quality: first reason a pixel is unusable (bv.quality_ok).
QUALITY = [
    ("Usable", lighter(GREEN, .3)),
    ("Not produced / band missing", NONE_C),
    ("Cloudy (MODLAND)", BLUE),
    ("Mixed clouds", lighter(BLUE)),
    ("Snow / ice", VIOLET),
    ("Cloud shadow", LOW_C),
    ("High aerosol", ORANGE),
    ("Lowest usefulness", YELLOW),
]
DIVERGE = LinearSegmentedColormap.from_list(
    "diverge", [ORANGE, "#e6e4dd", BLUE])


# --- geometry --------------------------------------------------------------

def frame_of(bounds, res):
    xmin, ymin, xmax, ymax = bounds
    return (from_origin(xmin, ymax, res, res),
            (int(round((ymax - ymin) / res)), int(round((xmax - xmin) / res))))


def modis_window(src, bounds, pad=2):
    """Row / column slices of the MODIS grid covering ``bounds``."""
    crs, transform, shape = src
    xmin, ymin, xmax, ymax = bounds
    xs = np.linspace(xmin, xmax, 21)
    ys = np.linspace(ymin, ymax, 21)
    gx, gy = np.meshgrid(xs, ys)
    sx, sy = Transformer.from_crs(CRS_L93, crs, always_xy=True).transform(
        gx.ravel(), gy.ravel())
    cols = (sx - transform.c) / transform.a
    rows = (sy - transform.f) / transform.e
    r0 = max(int(rows.min()) - pad, 0)
    r1 = min(int(rows.max()) + pad + 1, shape[0])
    c0 = max(int(cols.min()) - pad, 0)
    c1 = min(int(cols.max()) + pad + 1, shape[1])
    return slice(r0, r1), slice(c0, c1)


class Crop:
    """A crop of the MODIS grid and its projection on the cell."""

    def __init__(self, src, bounds):
        self.crs, transform, self.full = src
        self.rows, self.cols = modis_window(src, bounds)
        self.transform = transform * Affine.translation(self.cols.start,
                                                        self.rows.start)
        self.shape = (self.rows.stop - self.rows.start,
                      self.cols.stop - self.cols.start)
        self.bounds = bounds
        r, c = np.mgrid[self.rows, self.cols]
        self.pix = (r * self.full[1] + c).ravel()

    def local(self, pix):
        """Position in the crop of MODIS pixel ``pix`` (-1: outside)."""
        r, c = np.divmod(pix, self.full[1])
        r, c = r - self.rows.start, c - self.cols.start
        ok = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
        return np.where(ok, r * self.shape[1] + c, -1)

    def grid(self, pix, values, fill=np.nan):
        a = np.full(self.shape[0] * self.shape[1], fill, dtype="float32")
        pos = self.local(pix)
        a[pos[pos >= 0]] = np.asarray(values)[pos >= 0]
        return a.reshape(self.shape)

    def to_cell(self, a):
        """Crop of the MODIS grid -> Lambert-93 cell at MODIS_RES."""
        tr, shape = frame_of(self.bounds, MODIS_RES)
        out = np.full(shape, np.nan, dtype="float32")
        reproject(a.astype("float32"), out, src_transform=self.transform,
                  src_crs=self.crs, dst_transform=tr, dst_crs=CRS_L93,
                  resampling=Resampling.nearest, src_nodata=np.nan,
                  dst_nodata=np.nan)
        return out


# --- Sentinel-2 ------------------------------------------------------------

def s2_search(bounds, start, end):
    to_ll = Transformer.from_crs(CRS_L93, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform([bounds[0], bounds[2], bounds[0], bounds[2]],
                               [bounds[1], bounds[1], bounds[3], bounds[3]])
    r = requests.post(STAC, json={
        "collections": ["sentinel-2-l2a"],
        "bbox": [min(lon), min(lat), max(lon), max(lat)],
        "datetime": f"{start:%Y-%m-%d}T00:00:00Z/{end:%Y-%m-%d}T23:59:59Z",
        "limit": 200}, timeout=120)
    r.raise_for_status()
    items = pd.DataFrame([{
        "id": f["id"], "day": f["properties"]["datetime"][:10],
        "cloud": f["properties"].get("eo:cloud_cover", 100.),
        "visual": f["assets"]["visual"]["href"],
        "scl": f["assets"]["scl"]["href"]} for f in r.json()["features"]])
    # Reprocessed duplicates (_1_L2A): keep one item per (tile, day).
    items["tile"] = items["id"].str.split("_").str[1]
    return items.drop_duplicates(["tile", "day"])


def warp(href, bounds, res, resampling, count):
    tr, shape = frame_of(bounds, res)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src, WarpedVRT(
                src, crs=CRS_L93, transform=tr, width=shape[1],
                height=shape[0], resampling=resampling, nodata=0) as vrt:
            return vrt.read(list(range(1, count + 1)))


def sentinel2(bounds, composite, key):
    """True-colour image of the cell, its day and cloud share."""
    out = CACHE / f"s2_{key}.npz"
    if out.exists():
        with np.load(out) as z:
            return z["rgb"], str(z["day"]), float(z["cloud"])
    mid = composite + timedelta(days=bv.COMPOSITE_DAYS / 2)
    items = s2_search(bounds, composite - timedelta(days=SEARCH_DAYS),
                      composite + timedelta(days=bv.COMPOSITE_DAYS - 1
                                            + SEARCH_DAYS))
    days = items.groupby("day")["cloud"].mean().reset_index()
    days["gap"] = (pd.to_datetime(days["day"]) - mid).abs().dt.days
    days = days.sort_values(["cloud", "gap"]).head(8)
    best = None
    for day in days["day"]:
        # Scene classification at 100 m: coverage and clouds over the cell.
        scl = np.zeros(frame_of(bounds, 100)[1], dtype="uint8")
        for href in items.loc[items["day"] == day, "scl"]:
            a = warp(href, bounds, 100, Resampling.nearest, 1)[0]
            scl = np.where(scl == 0, a, scl)
        cover = (scl > 0).mean()
        cloud = np.isin(scl, (3, 8, 9, 10)).sum() / max((scl > 0).sum(), 1)
        print(f"  Sentinel-2 {day}: cover {cover:.0%}, cloud {cloud:.0%}",
              flush=True)
        if cover > .95 and (best is None or cloud < best[1]):
            best = (day, cloud)
        if best is not None and best[1] < .02:
            break
    if best is None:
        raise SystemExit("no Sentinel-2 scene covers the cell")
    day, cloud = best
    rgb = np.zeros((3, *frame_of(bounds, S2_RES)[1]), dtype="uint8")
    for href in items.loc[items["day"] == day, "visual"]:
        a = warp(href, bounds, S2_RES, Resampling.average, 3)
        rgb = np.where((rgb == 0).all(axis=0), a, rgb)
    rgb = np.moveaxis(rgb, 0, -1)
    np.savez_compressed(out, rgb=rgb, day=day, cloud=cloud)
    return rgb, day, cloud


# --- BD Forêt and Corine ---------------------------------------------------

def departments(win):
    """BD Forêt archives with fuel in the window."""
    keep = []
    pix = set(win.pix.tolist())
    for f in sorted(bv.FUEL_CACHE.glob("*.parquet")):
        if pd.read_parquet(f, columns=["pix"])["pix"].isin(pix).any():
            keep.append(bv.BDFORET_DIR / f"{f.stem}.7z")
    return keep


def bdforet(win, key):
    """25 m rasters over the cell: TFV_G11 and ESSENCE codes, and the
    normalised values behind the codes (0: no polygon)."""
    out = CACHE / f"bdforet_{key}.npz"
    if out.exists():
        with np.load(out) as z:
            return (z["tfv"], z["ess"], z["med"], json.loads(str(z["tfvs"])),
                    json.loads(str(z["esss"])))
    tr, shape = frame_of(win.bounds, FINE)
    xmin, ymin, xmax, ymax = win.bounds
    tfv = np.zeros(shape, dtype="int16")
    ess = np.zeros(shape, dtype="int16")
    tfvs, esss = [""], [""]
    for archive in departments(win):
        print(f"  BD Forêt {archive.stem}", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            with py7zr.SevenZipFile(archive) as z:
                z.extract(tmp, targets=[n for n in z.getnames()
                                        if "FORMATION_VEGETALE" in n])
            shp = next(Path(tmp).rglob("FORMATION_VEGETALE.shp"))
            reader = shapefile.Reader(str(shp), encoding="utf-8")
            shapes = {"tfv": [], "ess": []}
            for s, rec in zip(reader.iterShapes(), reader.iterRecords()):
                b = s.bbox
                if b[2] < xmin or b[0] > xmax or b[3] < ymin or b[1] > ymax:
                    continue
                t, e = norm(str(rec["TFV_G11"] or "")), norm(
                    str(rec["ESSENCE"] or ""))
                for name, v, vals in (("tfv", t, tfvs), ("ess", e, esss)):
                    if v not in vals:
                        vals.append(v)
                    shapes[name].append((s.__geo_interface__, vals.index(v)))
            reader.close()
        for name, a in (("tfv", tfv), ("ess", ess)):
            if shapes[name]:
                r = rasterize(shapes[name], out_shape=shape, transform=tr,
                              fill=0, dtype="int16")
                a[r > 0] = r[r > 0]
    med = rasterize(((g, 1) for g in bv.med_shapes()), out_shape=shape,
                    transform=tr, fill=0, dtype="uint8")
    np.savez_compressed(out, tfv=tfv, ess=ess, med=med,
                        tfvs=json.dumps(tfvs), esss=json.dumps(esss))
    return tfv, ess, med, tfvs, esss


def corine(bounds):
    tr, shape = frame_of(bounds, FINE)
    out = np.full(shape, CLC_NODATA, dtype="int16")
    with rasterio.open(CLC_TIF) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=tr,
                  dst_crs=CRS_L93, resampling=Resampling.nearest,
                  dst_nodata=CLC_NODATA)
    group = np.full(256, -1, dtype="int8")
    for v, code in enumerate(CLC_CODES, 1):
        group[v] = next(k for k, (_, _, f) in enumerate(CLC_GROUPS)
                        if f(code))
    return np.where(out == CLC_NODATA, len(CLC_GROUPS),
                    group[out.astype("int64") % 256])


# --- MODIS -----------------------------------------------------------------

def modis_layers(win, composite):
    """Observed NDVI (x 1e-4, NaN where unusable) and the quality reason of
    every pixel of the window."""
    layers = bv.composites()[composite]
    a = {}
    for name in ("ndvi", "qa", "nir", "mir"):
        with rasterio.open(layers[name]) as src:
            a[name] = src.read(1, window=((win.rows.start, win.rows.stop),
                                          (win.cols.start, win.cols.stop)))
            a[name + "_nd"] = src.nodata
    qa = a["qa"].astype("uint32")
    missing = ((a["ndvi"] == a["ndvi_nd"]) | (a["nir"] == a["nir_nd"])
               | (a["mir"] == a["mir_nd"]) | ((qa & 0b11) == 0b11))
    reasons = [missing, (qa & 0b11) == 0b10, (qa >> 10) & 1 == 1,
               (qa >> 14) & 1 == 1, (qa >> 15) & 1 == 1,
               ((qa >> 6) & 0b11) == 0b11, ((qa >> 2) & 0b1111) > 0b1011]
    reason = np.zeros(qa.shape, dtype="float32")
    for k in range(len(reasons) - 1, -1, -1):   # first reason wins
        reason[reasons[k]] = k + 1
    ndvi = np.where(reason == 0, a["ndvi"], np.nan).astype("float32")
    return ndvi, reason


def fuel_shares(win):
    """Fuel share and largest class share of every pixel of the window."""
    pix = set(win.pix.tolist())
    counts = pd.concat([d[d["pix"].isin(pix)] for d in
                        (pd.read_parquet(f) for f in
                         bv.FUEL_CACHE.glob("*.parquet"))])
    counts = counts.groupby(["pix", "cls"])["n"].sum()
    full = abs(win.transform.a * win.transform.e) / bv.RES ** 2
    fuel = counts.groupby("pix").sum() / full
    top = counts.groupby("pix").max() / full
    return fuel, top


def burns_near(cell_ids, dates, pixels):
    """Burn candidates and burns of the fuel pixels of ``cell_ids``, by
    bv.burns on those pixels only (identical to the full run: a pixel's
    detection depends on itself and the pixels of its cell)."""
    sub = np.flatnonzero(pixels["cell_id"].isin(cell_ids).to_numpy())
    ci = pd.Index(cell_ids).get_indexer(pixels["cell_id"].to_numpy()[sub])
    orig = bv.load

    def load(date, corrected=False):
        ndvi, nbr, has = orig(date, corrected)
        return ndvi[sub], nbr[sub], has

    bv.load = load
    try:
        candidates, events = bv.burns(dates, ci)
    finally:
        bv.load = orig
    candidates = {d: sub[c] for d, c in candidates.items()}
    events = events.assign(pos=sub[events["pos"].to_numpy()])
    return candidates, events


def pixel_status(win, composite, pixels, cell_ids):
    """Status of every pixel of the window, NDVI / NBR used by the features
    (x 1e-4, fuel pixels), and the mixture correction of the NDVI."""
    dates = list(bv.composites())
    prev = dates[dates.index(composite) - 1]
    candidates, events = burns_near(cell_ids, dates, pixels)
    n = len(pixels)
    ndvi, nbr, _ = bv.load(composite, corrected=True)
    o_ndvi, o_nbr, _ = bv.load(composite)
    p_ndvi, p_nbr, _ = bv.load(prev, corrected=True)
    excl = bv.excluded(composite, candidates, events, n)
    p_excl = bv.excluded(prev, candidates, events, n)
    observed = (ndvi != bv.MISSING) & ~excl
    previous = (ndvi == bv.MISSING) & ~excl & (p_ndvi != bv.MISSING) & ~p_excl
    kept = observed & (ndvi == o_ndvi) & (nbr == o_nbr)
    st = np.full(n, NO_VALUE, dtype="float32")
    st[excl] = BURNT
    st[previous] = PREVIOUS
    st[observed] = CORRECTED
    st[kept] = KEPT

    fuel, top = fuel_shares(win)
    status = np.full(len(win.pix), NO_FUEL, dtype="float32")
    f = fuel.reindex(win.pix).fillna(0).to_numpy()
    t = top.reindex(win.pix).fillna(0).to_numpy()
    status[f > 0] = LOW_FUEL
    status[(f >= bv.PURITY) & (t < bv.PURITY)] = MIXED
    status = status.reshape(win.shape)
    pix = pixels["pix"].to_numpy()
    pos = win.local(pix)
    inside = pos >= 0
    status.ravel()[pos[inside]] = st[inside]
    values = {k: np.where(observed, v, np.where(previous, pv, bv.MISSING))
              for k, v, pv in (("ndvi", ndvi, p_ndvi), ("nbr", nbr, p_nbr))}
    delta = np.where(observed, (ndvi.astype("float32") - o_ndvi) / 1e4,
                     np.nan)
    return status, values, delta, inside, events


# --- figure ----------------------------------------------------------------

def categorical(ax, img, entries, title, note=None, ncol=2):
    """``img``: category index per pixel (-1 / NaN: background)."""
    lut = np.array([to_rgba(c) for _, c in entries])
    rgba = np.zeros(img.shape + (4,))
    ok = ~np.isnan(img) & (img >= 0)
    rgba[ok] = lut[img[ok].astype(int)]
    present = np.unique(img[ok].astype(int))
    show(ax, rgba, title, note)
    handles = [plt.Rectangle((0, 0), 1, 1, color=entries[k][1],
                             label=entries[k][0]) for k in present]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -.005),
              ncol=ncol, frameon=False, fontsize=7.5, labelcolor=INK,
              handlelength=1.2, columnspacing=1., borderaxespad=0)


def continuous(ax, fig, rgba, title, cmap, norm, label, note=None,
               fires=False):
    show(ax, rgba, title, note, fires)
    cb = fig.colorbar(ScalarMappable(norm, cmap),
                      cax=ax.inset_axes([.05, -.075, .9, .035]),
                      orientation="horizontal")
    style_colorbar(cb, label)


EXTENT = None   # set per cell: (xmin, xmax, ymin, ymax)
FIRES = None


def show(ax, img, title, note=None, fires=False):
    ax.imshow(img, extent=EXTENT, interpolation="nearest")
    if fires and FIRES is not None and len(FIRES):
        for lw, c in ((2.6, "black"), (1.2, "white")):
            ax.scatter(FIRES["x"], FIRES["y"], s=40, facecolor="none",
                       edgecolor=c, linewidths=lw, zorder=3)
    ax.set_xlim(EXTENT[:2])
    ax.set_ylim(EXTENT[2:])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c8c5bb")
    ax.set_title(title, color=INK, fontsize=10.5, loc="left",
                 pad=14 if note else 5)
    if note:
        ax.text(0, 1.008, note, transform=ax.transAxes, color=MUTED,
                fontsize=7.5, va="bottom")


def top_entries(img, labels, palette=CATEGORICAL, none_label=None):
    """Legend entries of the codes of ``img`` (into ``labels``) and the
    code -> entry map: by area, the n largest get CATEGORICAL colours, the
    rest Other; code 0 -> background (last entry)."""
    n = len(palette)
    codes, counts = np.unique(img[img > 0], return_counts=True)
    order = codes[np.argsort(-counts)]
    if none_label is not None:   # no species: its own pale colour
        order = [c for c in order if c != none_label]
    shown = list(order[:n])
    entries = [(labels[c], palette[k]) for k, c in enumerate(shown)]
    remap = np.full(len(labels), -1, dtype="float32")
    remap[list(shown)] = range(len(shown))
    if len(order) > n:
        entries.append(("Other", OTHER_C))
        remap[list(order[n:])] = len(entries) - 1
    if none_label is not None:
        entries.append(("No species (heath, open forest...)", PALE_C))
        remap[none_label] = len(entries) - 1
    entries.append(("Not in BD Forêt", NONE_C))
    remap[0] = len(entries) - 1
    return remap, entries


def percentile_rgba(status, pct, cmap):
    rgba = np.zeros(status.shape + (4,))
    rgba[:] = to_rgba(NONE_C)
    ok = ~np.isnan(status)
    rgba[ok & (status >= MIXED)] = to_rgba(NAN_C)
    used = np.isin(status, (PREVIOUS, CORRECTED, KEPT)) & ~np.isnan(pct)
    col = cmap(Normalize(0, 1)(np.nan_to_num(pct)))
    rgba[used] = col[used]
    light = used & (status == PREVIOUS)
    rgba[light, :3] = .45 * col[light, :3] + .55
    return rgba


def cell_view(day, cell, grid, pixels, src):
    global EXTENT, FIRES
    CACHE.mkdir(parents=True, exist_ok=True)
    dates = list(bv.composites())
    composite = max(d for d in dates
                    if d + timedelta(days=bv.COMPOSITE_DAYS) <= day)
    end = composite + timedelta(days=bv.COMPOSITE_DAYS - 1)
    g = grid.set_index("cell_id")
    ix, iy = g.loc[cell, "ix"], g.loc[cell, "iy"]
    bounds = (ix * CELL_SIZE, iy * CELL_SIZE, (ix + 1) * CELL_SIZE,
              (iy + 1) * CELL_SIZE)
    EXTENT = (bounds[0], bounds[2], bounds[1], bounds[3])
    near = grid.loc[grid["ix"].between(ix - 1, ix + 1)
                    & grid["iy"].between(iy - 1, iy + 1), "cell_id"].tolist()
    fires = pd.read_parquet(INTERIM / "fires.parquet")
    FIRES = fires[(fires["date"] == day) & (fires["cell_id"] == cell)]
    key = f"{cell}_{composite:%Y-%m-%d}"
    print(f"cell {cell}, {day:%Y-%m-%d}: composite {composite:%Y-%m-%d}",
          flush=True)

    win = Crop(src, bounds)
    rgb, s2_day, s2_cloud = sentinel2(bounds, composite, key)
    tfv, ess, med, tfvs, esss = bdforet(win, str(cell))
    clc = corine(bounds)
    ndvi_obs, reason = modis_layers(win, composite)
    status, values, delta, inside, events = pixel_status(
        win, composite, pixels, near)

    # 25 m fuel class, as fuel_class.
    lut = np.zeros((len(tfvs), len(esss), 2), dtype="int16")
    for i, t in enumerate(tfvs):
        for j, e in enumerate(esss):
            for m in (0, 1):
                lut[i, j, m] = bv.fuel_class(t, e, bool(m)) if t else 0
    fuel25 = lut[tfv, ess, med]
    cls_labels = [""] + [LABELS[c] for c in bv.CLASSES]
    # Blue is kept for the mixed pixels of the 250 m panel.
    remap, fuel_entries = top_entries(fuel25, cls_labels, CATEGORICAL[1:])
    fuel_entries[-1] = ("Not fuel", NONE_C)
    fuel_img = remap[fuel25]
    # Same colours for the 250 m fuel pixels (a class absent at 25 m in the
    # cell: Other); the other pixels: not fuel, or mixed.
    cls_entries = fuel_entries + [("75 % fuel, no class at 75 %", BLUE),
                                  ("Other", OTHER_C)]
    remap[remap < 0] = len(cls_entries) - 1
    cls_px = win.grid(pixels["pix"].to_numpy(),
                      remap[pixels["cls"].to_numpy()])
    cls_px = np.where(np.isnan(cls_px), np.where(
        status == MIXED, len(fuel_entries), len(fuel_entries) - 1), cls_px)

    tfv_group = np.zeros(len(tfvs), dtype="float32")
    for i, t in enumerate(tfvs):
        tfv_group[i] = next((k for k, (_, _, vals) in enumerate(TFV_GROUPS)
                             if t in vals), len(TFV_GROUPS) - 1)
    tfv_group[0] = len(TFV_GROUPS)
    tfv_entries = [(n, c) for n, c, _ in TFV_GROUPS] + [("Not in BD Forêt",
                                                         NONE_C)]
    none = [i for i, e in enumerate(esss) if i and e in ("nc", "nr", "")]
    ess_labels = [SPECIES.get(e, e) for e in esss]
    ess_codes = ess.copy()
    if none:
        ess_codes[np.isin(ess, none)] = none[0]
    ess_map, ess_entries = top_entries(ess_codes, ess_labels,
                                       none_label=none[0] if none else None)
    ess_img = ess_map[ess_codes]

    ref = np.load(bv.WORK / "reference.npz")
    pct = {}
    pix, cls = pixels["pix"].to_numpy(), pixels["cls"].to_numpy()
    for name in ("ndvi", "nbr"):
        h = ref[name].astype("float64")
        with np.errstate(invalid="ignore"):
            table = (np.cumsum(h, axis=1) - h / 2) / h.sum(axis=1,
                                                           keepdims=True)
        v = values[name]
        ok = (v != bv.MISSING) & inside
        p = np.full(len(pix), np.nan, dtype="float32")
        p[ok] = table[cls[ok], v[ok].astype(int) + bv.OFFSET]
        pct[name] = win.to_cell(win.grid(pix, p))

    st = win.to_cell(status)
    in_cell = pixels["cell_id"].to_numpy() == cell
    burnt = events[(events["known_start"] <= composite)
                   & in_cell[events["pos"].to_numpy()]]
    cell_st = status.ravel()[win.local(pix[in_cell])]
    n_st = np.bincount(cell_st.astype(int), minlength=len(STATUS))

    fig, axes = plt.subplots(3, 4, figsize=(22, 20.5), dpi=120,
                             gridspec_kw={"wspace": .04, "hspace": .42})
    ax = axes.flat
    # Display stretch, the same for the 3 bands (keeps the colours):
    # the 1st-99th percentiles of the cell over 0-1, gamma 1.4.
    lo, hi = np.percentile(rgb[(rgb > 0).all(axis=-1)], (1, 99))
    show(ax[0], np.clip((rgb - lo) / (hi - lo), 0, 1) ** (1 / 1.4),
         "Sentinel-2 true colour (10 m, contrast stretched)",
         f"{s2_day}, {s2_cloud:.0%} cloud or shadow over the cell; "
         "circles: fires of the day", fires=True)
    categorical(ax[1], clc.astype("float32"),
                [(n, c) for n, c, _ in CLC_GROUPS]
                + [("No Corine data (sea, estuary)", PALE_C)],
                "Corine Land Cover 2018 (100 m)")
    categorical(ax[2], tfv_group[tfv], tfv_entries,
                "BD Forêt vegetation type (TFV_G11)",
                "photo-interpreted 2007-2018, polygons of 0.5 ha or more")
    categorical(ax[3], ess_img, ess_entries,
                "BD Forêt dominant species (ESSENCE)")
    categorical(ax[4], fuel_img, fuel_entries, "Fuel class (25 m)",
                "TFV_G11 x ESSENCE x Mediterranean zone")
    categorical(ax[5], win.to_cell(cls_px), cls_entries,
                "Fuel class of the fuel pixels (250 m)",
                "MODIS pixels with one class on at least 75 %")
    categorical(ax[6], win.to_cell(reason), QUALITY,
                "MODIS VI Quality of the composite (all pixels)",
                "first reason a pixel is unusable")
    categorical(ax[7], st, STATUS, "Status of each 250 m pixel",
                f"cell's fuel pixels: {n_st[CORRECTED] + n_st[KEPT]:,} "
                f"usable, {n_st[PREVIOUS]:,} from the previous composite, "
                f"{n_st[NO_VALUE]:,} none, {n_st[BURNT]:,} burnt",
                ncol=1)
    green = ramp("Greens")
    obs = win.to_cell(ndvi_obs) / 1e4
    rgba = green(Normalize(0, 1)(np.nan_to_num(obs)))
    rgba[np.isnan(obs)] = to_rgba(NAN_C)
    continuous(ax[8], fig, rgba, "Observed MODIS NDVI (all pixels)", green,
               Normalize(0, 1), "NDVI (grey: unusable)")
    d = win.to_cell(win.grid(pix, delta))
    norm_d = TwoSlopeNorm(0, -.05, .05)
    rgba = DIVERGE(norm_d(np.nan_to_num(np.clip(d, -.05, .05))))
    rgba[np.isnan(d)] = to_rgba(NONE_C)
    continuous(ax[9], fig, rgba, "Mixture correction of the NDVI", DIVERGE,
               norm_d, "corrected − observed NDVI",
               "black: no usable value; blue: corrected above observed")
    for k, name in ((10, "NDVI"), (11, "NBR")):
        continuous(ax[k], fig, percentile_rgba(st, pct[name.lower()], DRY),
                   f"{name} class percentile (as the features use it)", DRY,
                   Normalize(0, 1), "0 = lowest value of the class; "
                   "lighter: previous composite",
                   "black: not fuel; grey: fuel pixel left out (see status)",
                   fires=True)
    lon, lat = g.loc[cell, "lon"], g.loc[cell, "lat"]
    fig.suptitle(f"Cell {cell} ({lat:.2f}° N, {abs(lon):.2f}° "
                 f"{'E' if lon >= 0 else 'W'}) on {day:%d %B %Y}"
                 f": MODIS composite {composite:%d %b} – {end:%d %b %Y}; "
                 f"{len(FIRES)} fire(s) that day; fuel pixels of the cell burnt "
                 f"since 2016: {len(burnt)}", color=INK, fontsize=14, x=.125, ha="left",
                 y=.925)
    out = PROCESSED / f"cell_pixels_{day:%Y-%m-%d}_{cell}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}; cell fuel pixels by status: "
          f"{dict(zip([s for s, _ in STATUS], n_st.tolist()))}", flush=True)


def main(seed, views):
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    pixels = pd.read_parquet(bv.WORK / "fuel_pixels.parquet")
    src = bv.modis_grid()
    if not views:
        df = pd.read_parquet(PROCESSED / "firegrid.parquet", columns=["date"])
        fires = pd.read_parquet(INTERIM / "fires.parquet")
        fuel_px = pixels["cell_id"].value_counts()
        for day in example_days(df, seed):
            c = fires.loc[fires["date"] == day, "cell_id"].value_counts(
                ).rename("fires").to_frame()
            c["fuel_px"] = fuel_px.reindex(c.index).fillna(0)
            c = c[c["fuel_px"] >= MIN_FUEL_PX].sort_values(
                ["fires", "fuel_px"], ascending=False)
            views.append((day, int(c.index[0])))
    else:
        views = [(pd.Timestamp(v.split(":")[0]), int(v.split(":")[1]))
                 for v in views]
    for day, cell in views:
        cell_view(day, cell, grid, pixels, src)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", action="append", default=[],
                   help="DAY:CELL, e.g. 2023-08-10:749 (repeatable)")
    a = p.parse_args()
    main(a.seed, a.view)
