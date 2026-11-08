"""Mixture correction of the fuel pixels (step 2b of build_vegetation.py).

A fuel pixel is covered at least PURITY by one fuel class (its *main*
class); the rest of it is other fuel classes and non-fuel land (crops,
roads...), whose own reflectance, e.g. a field harvested in July, biases the
pixel. The correction removes it, band by band, on the reflectances (they
mix linearly within a pixel, NDVI and NBR do not):

    main = (observed - sum_k share_k * reference_k) / (1 - sum_k share_k)

over the *contaminants* k of the pixel, then NDVI and NBR are recomputed
from the corrected red, NIR and SWIR (MIR) reflectances. The red comes from
the NDVI and the NIR: red = NIR (1 - NDVI) / (1 + NDVI).

- Shares. The 25 m BD Forêt squares give the share of each fuel class in
  the pixel; the rest is non-fuel, split between the non-fuel TYPES in
  proportion to Corine Land Cover in the pixel. Corine (100 m, 25 ha
  minimum) often maps small fields or roads inside a forest as forest: the
  part of the non-fuel share that Corine's non-fuel groups do not cover is
  "other".
- References, per 25 km cell and composite: for another fuel class, the
  median reflectance of its usable fuel pixels at least REF_PURITY pure in
  the cell; for a non-fuel type, of the pixels of the cell with less than
  MAX_FUEL fuel and at least PURITY of that Corine group (at most REF_MAX
  of them, drawn once); "other": of all the pixels with less than MAX_FUEL
  fuel. Each needs MIN_REF usable pixels, else the median of the
  references of the 3 x 3 cells around, else the national median.
- A contaminant with no reference at all is assumed to look like the main
  class (the denominator keeps its share). A correction giving a
  reflectance outside 0-1 is dropped: the pixel keeps its observed values.

Caches under ``data/interim/vegetation/``: ``clc_modis.parquet`` (Corine
group counts per MODIS pixel), ``mixture.npz`` (shares, reference pixels),
``reflect/<date>.npz`` (red, NIR, SWIR of the fuel and reference pixels)
and ``corrected/<date>.npz`` (corrected NDVI, NBR of the fuel pixels, and
the references).
"""

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

import build_vegetation as bv
from build_landcover import CODES as CLC_CODES
from build_landcover import NODATA as CLC_NODATA
from build_landcover import TIF as CLC_TIF
from common import INTERIM

TYPES = ["crops", "vine_orchard", "pasture", "agri_natural", "urban",
         "wetland", "water", "sparse", "other"]
NTYPE = len(TYPES)
NKIND = len(bv.CLASSES) + NTYPE   # contaminant kinds: fuel classes, types
REF_PURITY = .95    # purity of the fuel pixels used as class references
MAX_FUEL = .10      # fuel share below which a pixel is non-fuel reference
REF_MAX = 200       # reference pixels drawn per (cell, non-fuel type)
MIN_REF = 10        # usable pixels a reference needs
BANDS = ["red", "nir", "swir"]
REFLECT = bv.WORK / "reflect"
CORRECTED = bv.WORK / "corrected"


def clc_group(code):
    """TYPES index of a Corine level-3 code; natural vegetation -> other."""
    if code < 200:
        return 4
    if code in (211, 212, 213, 241, 242):
        return 0
    if 221 <= code <= 223:
        return 1
    if code == 231:
        return 2
    if code in (243, 244):
        return 3
    if 331 <= code <= 335:
        return 7
    if 400 <= code < 500:
        return 5
    if code >= 500:
        return 6
    return 8  # 311-324: forest, grassland, heath, maquis


def clc_counts(grid):
    """Corine 100 m pixel counts per (MODIS pixel, TYPES group)."""
    out = bv.WORK / "clc_modis.parquet"
    if out.exists():
        return pd.read_parquet(out)
    crs, transform, shape = grid
    lut = np.full(128, -1, dtype="int8")  # Corine raster value -> group
    for v, code in enumerate(CLC_CODES, 1):
        lut[v] = clc_group(code)
    to_sin = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    parts = []
    with rasterio.open(CLC_TIF) as src:
        tr, (h, w) = src.transform, src.shape
        for r0 in range(0, h, 400):
            a = src.read(1, window=Window(0, r0, w, min(400, h - r0)))
            rows, cols = np.nonzero(a != CLC_NODATA)
            g = lut[a[rows, cols].astype("int64")]
            lon, lat = tr * (cols + .5, rows + r0 + .5)
            xs, ys = to_sin.transform(lon, lat)
            col = np.floor((xs - transform.c) / transform.a).astype("int64")
            row = np.floor((ys - transform.f) / transform.e).astype("int64")
            ok = ((g >= 0) & (row >= 0) & (row < shape[0]) & (col >= 0)
                  & (col < shape[1]))
            key, n = np.unique((row[ok] * shape[1] + col[ok]) * NTYPE
                               + g[ok], return_counts=True)
            parts.append(pd.DataFrame({"key": key, "n": n}))
    d = pd.concat(parts).groupby("key", as_index=False)["n"].sum()
    d = pd.DataFrame({"pix": d["key"] // NTYPE,
                      "grp": (d["key"] % NTYPE).astype("int8"),
                      "n": d["n"].astype("int32")})
    d.to_parquet(out, index=False)
    print(f"Corine counts on the MODIS grid: {d['pix'].nunique()} pixels",
          flush=True)
    return d


def setup(pixels, grid):
    """Shares of the contaminants of each fuel pixel, and the reference
    pixels; cached in ``mixture.npz``."""
    out = bv.WORK / "mixture.npz"
    crs, transform, shape = grid
    pix = pixels["pix"].to_numpy()
    if out.exists():
        with np.load(out) as z:
            if len(z["main"]) == len(pix) and int(z["checksum"]) == int(
                    pix.sum()):
                return
    full = abs(transform.a * transform.e) / bv.RES ** 2
    counts = pd.concat([pd.read_parquet(p)
                        for p in bv.FUEL_CACHE.glob("*.parquet")])
    counts = counts.groupby(["pix", "cls"], as_index=False)["n"].sum()
    clc = clc_counts(grid)

    # Fuel class shares of the fuel pixels (aliasing can count a little
    # more than a pixel: the shares are then scaled to sum to 1).
    pos_of = pd.Series(np.arange(len(pix)), index=pix)
    c = counts[counts["pix"].isin(pos_of.index)].copy()
    c["pos"] = pos_of.reindex(c["pix"]).to_numpy()
    tot = c.groupby("pos")["n"].sum()
    c["s"] = c["n"] / np.maximum(full, tot.reindex(c["pos"]).to_numpy())
    main_cls = pixels["cls"].to_numpy()
    is_main = c["cls"].to_numpy() == main_cls[c["pos"].to_numpy()]
    main = np.zeros(len(pix))
    main[c.loc[is_main, "pos"].to_numpy()] = c.loc[is_main, "s"].to_numpy()
    rows = [pd.DataFrame({"pos": c.loc[~is_main, "pos"].to_numpy(),
                          "kind": c.loc[~is_main, "cls"].to_numpy() - 1,
                          "share": c.loc[~is_main, "s"].to_numpy()})]
    nonfuel = (1 - c.groupby("pos")["s"].sum()).clip(lower=0)
    nonfuel = nonfuel.reindex(range(len(pix)), fill_value=1.).to_numpy()

    # Non-fuel share split by Corine groups of the pixel.
    q = clc[clc["pix"].isin(pos_of.index)].copy()
    q["pos"] = pos_of.reindex(q["pix"]).to_numpy()
    q["q"] = q["n"] / q.groupby("pos")["n"].transform("sum")
    q = q[q["grp"] < NTYPE - 1]  # Corine's non-fuel groups
    avail = q.groupby("pos")["q"].sum().reindex(range(len(pix)),
                                                fill_value=0.).to_numpy()
    scale = np.where(avail > nonfuel, nonfuel / np.maximum(avail, 1e-9), 1.)
    rows.append(pd.DataFrame({
        "pos": q["pos"].to_numpy(),
        "kind": len(bv.CLASSES) + q["grp"].to_numpy().astype("int64"),
        "share": q["q"].to_numpy() * scale[q["pos"].to_numpy()]}))
    other = nonfuel - np.minimum(avail, nonfuel)
    rows.append(pd.DataFrame({"pos": np.arange(len(pix)),
                              "kind": len(bv.CLASSES) + NTYPE - 1,
                              "share": other}))
    contam = pd.concat(rows)
    contam = contam[contam["share"] > 1e-4]

    # Reference pixels of the non-fuel types: less than MAX_FUEL fuel.
    fuel_share = (counts.groupby("pix")["n"].sum() / full).clip(upper=1)
    r = clc.copy()
    r["q"] = r["n"] / r.groupby("pix")["n"].transform("sum")
    r = r[(fuel_share.reindex(r["pix"]).fillna(0).to_numpy() < MAX_FUEL)]
    cand = r["pix"].unique()
    cell = pd.Series(bv.pixel_cells(cand, transform, shape[1], crs),
                     index=cand)
    r["cell_id"] = cell.reindex(r["pix"]).to_numpy()
    r = r[r["cell_id"] >= 0]
    typed = r[(r["q"] >= bv.PURITY) & (r["grp"] < NTYPE - 1)]
    typed = typed[["pix", "grp", "cell_id"]]
    anyp = r.drop_duplicates("pix")[["pix", "cell_id"]].assign(
        grp=NTYPE - 1)
    refs = pd.concat([typed, anyp])
    refs = (refs.sample(frac=1, random_state=0)
            .groupby(["cell_id", "grp"]).head(REF_MAX)
            .sort_values("pix"))

    ci = pd.Index(pd.read_parquet(INTERIM / "grid.parquet")["cell_id"])
    purity = main
    np.savez(out, checksum=pix.sum(), main=main,
             c_pos=contam["pos"].to_numpy().astype("int64"),
             c_kind=contam["kind"].to_numpy().astype("int64"),
             c_share=contam["share"].to_numpy().astype("float32"),
             ref_pix=refs["pix"].to_numpy().astype("int64"),
             ref_kind=(len(bv.CLASSES) + refs["grp"].to_numpy()).astype(
                 "int64"),
             ref_ci=ci.get_indexer(refs["cell_id"]).astype("int64"),
             fuel_ci=ci.get_indexer(pixels["cell_id"]).astype("int64"),
             fuel_kind=(main_cls - 1).astype("int64"),
             fuel_ref=purity >= REF_PURITY)
    contaminated = 1 - main
    print(f"mixture: {len(contam)} contaminant shares; fuel pixels' main "
          f"share median {np.median(main):.2f}, contaminants mean "
          f"{contaminated.mean():.3f} (non-fuel {nonfuel.mean():.3f}); "
          f"{len(refs)} non-fuel reference pixels", flush=True)
    print(refs.groupby("grp").size().rename(index=dict(enumerate(TYPES)))
          .to_string(), flush=True)


def static():
    with np.load(bv.WORK / "mixture.npz") as z:
        return {k: z[k] for k in z.files}


# --- reflectances ----------------------------------------------------------

def reflect_one(args):
    """Red, NIR and SWIR of the fuel and reference pixels (x 1e4, MISSING
    where unusable)."""
    date, layers, pix, shape = args
    out = REFLECT / f"{date:%Y-%m-%d}.npz"
    if out.exists():
        with np.load(out) as z:
            if len(z["nir"]) == len(pix) and int(z["checksum"]) == int(
                    pix.sum()):
                return
    if not {"nir", "mir"} <= set(layers):
        raise SystemExit(f"{date:%Y-%m-%d}: no NIR / MIR raster")
    ndvi, nd_nodata = bv.read(layers["ndvi"], pix, shape)
    qa, _ = bv.read(layers["qa"], pix, shape)
    nir, nir_nodata = bv.read(layers["nir"], pix, shape)
    mir, mir_nodata = bv.read(layers["mir"], pix, shape)
    usable = (bv.quality_ok(qa) & (ndvi != nd_nodata) & (nir != nir_nodata)
              & (mir != mir_nodata) & (nir + mir.astype("int32") > 0)
              & (ndvi > -10000))
    v = ndvi.astype("float64") * 1e-4
    with np.errstate(divide="ignore", invalid="ignore"):
        red = nir * (1 - v) / (1 + v)
    usable &= (red >= -100) & (red <= 16000)

    def i16(a):
        return np.where(usable, np.round(np.clip(a, -32000, 32000)),
                        bv.MISSING).astype("int16")

    np.savez(out, red=i16(red), nir=i16(nir), swir=i16(mir),
             checksum=pix.sum())


def extract(dates, pix, ref_pix, shape):
    REFLECT.mkdir(parents=True, exist_ok=True)
    allpix = np.concatenate([pix, ref_pix])
    with ProcessPoolExecutor(bv.WORKERS) as pool:
        list(pool.map(reflect_one, [(d, layers, allpix, shape)
                                    for d, layers in dates.items()]))


# --- correction --------------------------------------------------------------

_S = {}


def _init(cells_ixy):
    _S.update(static())
    _S["cells_ixy"] = cells_ixy


def references(bands, nf):
    """(n_cells, NKIND, 3) median reflectance per cell and kind, with the
    3 x 3 neighbourhood then the national fallbacks; and the source of each
    (0 cell, 1 neighbours, 2 national, 3 none)."""
    s = _S
    nc = len(s["cells_ixy"][0])
    fuel = s["fuel_ref"] & (bands[0][:nf] != bv.MISSING)
    ref = bands[0][nf:] != bv.MISSING
    cell = np.concatenate([s["fuel_ci"][fuel], s["ref_ci"][ref]])
    kind = np.concatenate([s["fuel_kind"][fuel], s["ref_kind"][ref]])
    vals = np.stack([np.concatenate([b[:nf][fuel], b[nf:][ref]])
                     for b in bands], axis=1).astype("float32")
    df = pd.DataFrame(vals, columns=BANDS)
    df["key"] = cell * NKIND + kind
    med = df.groupby("key")[BANDS].median()
    size = df.groupby("key").size()
    med = med[size >= MIN_REF]
    R = np.full((nc * NKIND, 3), np.nan, dtype="float32")
    R[med.index.to_numpy()] = med.to_numpy()
    R = R.reshape(nc, NKIND, 3)
    src = np.where(np.isnan(R[..., 0]), 3, 0)

    # 3 x 3 neighbourhood: median of the neighbours' references.
    ix, iy = s["cells_ixy"]
    gx, gy = ix - ix.min() + 1, iy - iy.min() + 1
    G = np.full((gx.max() + 2, gy.max() + 2, NKIND, 3), np.nan, "float32")
    G[gx, gy] = R
    stack = np.stack([G[gx + dx, gy + dy] for dx in (-1, 0, 1)
                      for dy in (-1, 0, 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        neigh = np.nanmedian(stack, axis=0)
    fill = np.isnan(R[..., 0]) & ~np.isnan(neigh[..., 0])
    R[fill] = neigh[fill]
    src[fill] = 1
    nat = df.assign(kind=df["key"] % NKIND).groupby("kind")[BANDS].median()
    for k, row in nat.iterrows():
        miss = np.isnan(R[:, k, 0])
        R[miss, k] = row.to_numpy()
        src[miss, k] = 2
    return R, src


def correct_one(date):
    out = CORRECTED / f"{date:%Y-%m-%d}.npz"
    s = _S
    nf = len(s["main"])
    with np.load(REFLECT / f"{date:%Y-%m-%d}.npz") as z:
        bands = [z[b] for b in BANDS]
    R, src = references(bands, nf)
    obs = np.stack([b[:nf] for b in bands], axis=1).astype("float64")
    ok = bands[0][:nf] != bv.MISSING

    pos, kind, share = s["c_pos"], s["c_kind"], s["c_share"]
    Rv = R[s["fuel_ci"][pos], kind]
    known = ~np.isnan(Rv[:, 0])
    w = share[known].astype("float64")
    ksum = np.bincount(pos[known], weights=w, minlength=nf)
    x = np.empty_like(obs)
    for b in range(3):
        csum = np.bincount(pos[known], weights=w * Rv[known, b],
                           minlength=nf)
        x[:, b] = (obs[:, b] - csum) / (1 - ksum)
    plausible = ((x >= 0) & (x <= 10000)).all(axis=1) & (
        x[:, 1] + x[:, 0] > 0) & (x[:, 1] + x[:, 2] > 0)
    x = np.where((ok & plausible)[:, None], x, obs)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (x[:, 1] - x[:, 0]) / (x[:, 1] + x[:, 0])
        nbr = (x[:, 1] - x[:, 2]) / (x[:, 1] + x[:, 2])

    def i16(a):
        return np.where(ok, np.round(np.clip(a, -1, 1) * 1e4),
                        bv.MISSING).astype("int16")

    np.savez(out, ndvi=i16(ndvi), nbr=i16(nbr), refs=R.astype("float32"),
             ref_src=src.astype("int8"))
    # Where the references used come from, weighted by contaminant share.
    used = ok[pos]
    by_src = np.bincount(src[s["fuel_ci"][pos[used]], kind[used]],
                         weights=share[used], minlength=4)
    return date, int(ok.sum()), int((ok & (ksum > 0)).sum()), int(
        (ok & ~plausible).sum()), by_src


def correct(dates, grid_cells):
    CORRECTED.mkdir(parents=True, exist_ok=True)
    cells_ixy = (grid_cells["ix"].to_numpy(), grid_cells["iy"].to_numpy())
    rows = []
    with ProcessPoolExecutor(bv.WORKERS, initializer=_init,
                             initargs=(cells_ixy,)) as pool:
        for date, n, n_corr, n_bad, src in pool.map(correct_one, dates):
            rows.append((date, n, n_corr, n_bad, *src))
    st = pd.DataFrame(rows, columns=["date", "usable", "corrected",
                                     "implausible", "ref_cell",
                                     "ref_neighbours", "ref_national",
                                     "ref_none"])
    st.to_parquet(bv.WORK / "correction_stats.parquet", index=False)
    ref = st[["ref_cell", "ref_neighbours", "ref_national",
              "ref_none"]].sum()
    print(f"correction: {st['corrected'].sum() / st['usable'].sum():.1%} of "
          f"the usable fuel pixel values corrected, "
          f"{st['implausible'].sum() / st['usable'].sum():.3%} implausible "
          "(kept observed); references from the cell / neighbours / "
          f"national / none: {(ref / ref.sum()).round(3).to_dict()}",
          flush=True)


def run(dates, pixels, grid):
    """Step 2b: setup, reflectances, correction."""
    setup(pixels, grid)
    s = static()
    extract(dates, pixels["pix"].to_numpy(), s["ref_pix"], grid[2])
    correct(list(dates), pd.read_parquet(INTERIM / "grid.parquet"))
