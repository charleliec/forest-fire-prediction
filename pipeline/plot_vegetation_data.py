"""Figures of the MODIS data behind the vegetation features.

- ``vegetation_usable.png``: why a 250 m pixel is used or not. Status of
  every MODIS pixel around the cell with most fires of an example day (the
  summer day of plot_vegetation.py), the split of France's pixels between
  fuel pixels, mixed pixels and no fuel, the reasons fuel pixels are
  rejected per month, and how many rejected pixels the previous one or two
  composites (16 / 32 days before) would recover;
- ``vegetation_raw.png``: raw NDVI and NBR of that composite, France at
  1 km and the window at 250 m, every pixel (fuel or not), quality flags
  ignored;
- ``vegetation_raw_series.png``: raw NDVI and NBR of the usable fuel pixels
  of six classes, median and interquartile range per composite;
- ``vegetation_correction.png``: the mixture correction (mixture.py): the
  bias of the less pure fuel pixels per month, observed and corrected, the
  per-cell shift it causes in July-August, the size of the corrections, and
  one pixel's series beside the local references.

    python pipeline/plot_vegetation_data.py [--seed 0]

Writes to ``data/processed/``. Needs build_vegetation.py and assemble.py.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from rasterio.warp import Resampling

import build_vegetation as bv
from common import CELL_SIZE, INTERIM, PROCESSED
from plot_map import INK, ramp, style_colorbar
from plot_vegetation import (CATEGORICAL, FIRE_C, GRID, LABELS, MUTED,
                             NAN_C, NBR_C, NDVI_C, example_days, style_axes,
                             to_l93)

RES = 250
# Pixel status, in drawing order. Greys: never used; hues: fuel pixels.
STATUS = [
    ("No BD Forêt fuel (crops, urban...)", "#eceae4"),
    ("Partly fuel (< 75 % of the pixel)", "#d0cec6"),
    ("Mostly fuel (≥ 75 %), but no class ≥ 75 %", "#a9a7a0"),
    ("Fuel pixel, usable", CATEGORICAL[0]),
    ("Fuel pixel, rejected by the quality flags", CATEGORICAL[1]),
    ("Fuel pixel, recently burnt", CATEGORICAL[2]),
]
# Reasons a fuel pixel is not usable, for the monthly breakdown.
REASONS = [("Cloud or cloud shadow", CATEGORICAL[1]),
           ("Snow or ice", CATEGORICAL[6]),
           ("Other quality (aerosol, low usefulness, missing band)",
            CATEGORICAL[3]),
           ("Recently burnt", CATEGORICAL[2])]
SERIES_CLASSES = ["broadleaf_decid", "fir_spruce_douglas", "pine_maritime",
                  "pine_aleppo", "heath_med", "grass"]
L93_FRANCE = (90_000, 6_030_000, 1_250_000, 7_120_000)  # xmin, ymin, ...


def composite_of(day, dates):
    """Last composite whose 16 days all precede ``day``."""
    return max(d for d in dates
               if d + timedelta(days=bv.COMPOSITE_DAYS) <= day)


def window(grid, day, half=3):
    fires = pd.read_parquet(INTERIM / "fires.parquet")
    fires = fires[fires["date"] == day]
    top = fires["cell_id"].value_counts().index[0]
    g = grid.set_index("cell_id")
    n = 2 * half + 1
    x0 = (g.loc[top, "ix"] - half) * CELL_SIZE
    y0 = (g.loc[top, "iy"] - half) * CELL_SIZE
    return fires, (x0, y0, x0 + n * CELL_SIZE, y0 + n * CELL_SIZE)


def read(path):
    with rasterio.open(path) as src:
        a = src.read(1)
        return a, src.nodata


def quality_reason(qa):
    """0 usable by the flags, 1 cloud / shadow, 2 snow, 3 other quality,
    as build_vegetation.quality_ok decides (same bits)."""
    qa = qa.astype("uint32")
    r = np.zeros(qa.shape, dtype="int8")
    r[(((qa >> 2) & 15) > 11) | (((qa >> 6) & 3) == 3)] = 3
    r[((qa & 3) >= 2) | ((qa >> 10) & 1 == 1) | ((qa >> 15) & 1 == 1)] = 1
    r[(qa >> 14) & 1 == 1] = 2
    r[qa == 65535] = 3
    return r


def recently_burnt(events, date, n):
    end = date + timedelta(days=bv.COMPOSITE_DAYS - 1)
    e = events[(events["known_start"] <= date) & (
        events["burn_start"] > end - timedelta(days=1826))]
    out = np.zeros(n, dtype=bool)
    out[e["pos"].to_numpy()] = True
    return out


# --- usable pixels -----------------------------------------------------------

def monthly_reasons(dates, pix, events):
    """Share of the fuel pixels per reason, and the share still unusable
    when falling back to the previous one / two composites, per
    composite."""
    n = len(pix)

    def one(item):
        t, date = item
        qa, _ = read(dates[date]["qa"])
        return t, quality_reason(qa.ravel()[pix])

    rows, usable = [], []
    days = list(dates)
    with ThreadPoolExecutor(6) as pool:
        for t, r in pool.map(one, enumerate(days)):
            date = days[t]
            ndvi, _, _ = bv.load(date)
            r[(r == 0) & (ndvi == bv.MISSING)] = 3  # missing band value
            r[(r == 0) & recently_burnt(events, date, n)] = 4
            ok = r == 0
            usable.append(ok)
            prev1 = usable[t - 1] if t >= 1 else np.zeros(n, bool)
            prev2 = usable[t - 2] if t >= 2 else np.zeros(n, bool)
            if t >= 3:
                usable[t - 3] = None
            row = {"date": date, "unusable": 1 - ok.mean(),
                   "unusable_16d": (~ok & ~prev1).mean(),
                   "unusable_32d": (~ok & ~prev1 & ~prev2).mean()}
            for k, (name, _) in enumerate(REASONS, 1):
                row[name] = (r == k).mean()
            rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def usable_figure(grid, day, dates, events):
    composite = composite_of(day, dates)
    src = bv.modis_grid()
    _, transform, shape = src
    fires, bounds = window(grid, day)

    pixels = pd.read_parquet(bv.WORK / "fuel_pixels.parquet")
    pix = pixels["pix"].to_numpy()
    counts = pd.concat([pd.read_parquet(p, columns=["pix", "n"])
                        for p in bv.FUEL_CACHE.glob("*.parquet")])
    fuel_share = counts.groupby("pix")["n"].sum() / (
        abs(transform.a * transform.e) / bv.RES ** 2)
    any_fuel = fuel_share.index.to_numpy()
    mostly = fuel_share.index[fuel_share >= bv.PURITY].to_numpy()

    # Status of every MODIS pixel in that composite.
    status = np.zeros(shape[0] * shape[1], dtype="float32")
    status[any_fuel] = 1
    status[mostly] = 2
    qa, _ = read(dates[composite]["qa"])
    reason = quality_reason(qa.ravel()[pix])
    ndvi, _, _ = bv.load(composite)
    reason[(reason == 0) & (ndvi == bv.MISSING)] = 3
    burnt = recently_burnt(events, composite, len(pix)) & (reason == 0)
    status[pix] = np.where(reason == 0, 3, 4)
    status[pix[burnt]] = 5
    # Sea, lakes and no data: left out (white), unless a fuel pixel.
    raw, nodata = read(dates[composite]["ndvi"])
    water = ((raw == nodata) | (raw < 0)).ravel() & (status < 3)
    status[water] = np.nan
    img = to_l93(status.reshape(shape), src, bounds, RES)

    print("monthly reasons over all composites...", flush=True)
    m = monthly_reasons(dates, pix, events)

    fig = plt.figure(figsize=(19, 13), dpi=130)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=.32,
                          wspace=.28)
    ax = fig.add_subplot(gs[0, :2])
    cmap = ListedColormap([c for _, c in STATUS])
    cmap.set_bad("white")
    ax.imshow(img, extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
              cmap=cmap,
              norm=BoundaryNorm(np.arange(-.5, len(STATUS)), len(STATUS)),
              interpolation="nearest")
    for k in range(int((bounds[2] - bounds[0]) / CELL_SIZE) + 1):
        ax.axvline(bounds[0] + k * CELL_SIZE, color="white", lw=.6)
        ax.axhline(bounds[1] + k * CELL_SIZE, color="white", lw=.6)
    ax.scatter(fires["x"], fires["y"], s=22, facecolor="none",
               edgecolor=FIRE_C, linewidths=1.2, zorder=3)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Every 250 m pixel around the cell with most fires on "
                 f"{day:%d %B %Y}\n(composite of {composite:%d %b %Y}; "
                 "circles: fires of the day; white: water)", color=INK,
                 fontsize=11, loc="left")

    # Window shares, ranked bars.
    ax = fig.add_subplot(gs[0, 2])
    valid = img[~np.isnan(img)].astype(int)
    share = np.bincount(valid, minlength=len(STATUS)) / len(valid) * 100
    y = np.arange(len(STATUS))[::-1]
    ax.barh(y, share, color=[c for _, c in STATUS], height=.7)
    for yy, s in zip(y, share):
        ax.text(s + 1, yy, f"{s:.1f} %", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [name for name, _ in STATUS], fontsize=9)
    ax.set_xlim(0, max(share) * 1.25)
    ax.set_title("Pixels of this window", color=INK, fontsize=11, loc="left")
    style_axes(ax)
    ax.grid(False)

    # France: what the 250 m pixels are.
    ax = fig.add_subplot(gs[1, 0])
    land = pd.read_parquet(INTERIM / "landcover_cell.parquet")
    land_px = land["land_share"].sum() * CELL_SIZE ** 2 / abs(
        transform.a * transform.e)
    parts = [("Fuel pixels", len(pix), CATEGORICAL[0]),
             ("Mostly fuel, no class ≥ 75 %", len(mostly) - len(pix),
              STATUS[2][1]),
             ("Partly fuel", len(any_fuel) - len(mostly), STATUS[1][1]),
             ("No BD Forêt fuel", land_px - len(any_fuel), STATUS[0][1])]
    y = np.arange(len(parts))[::-1]
    shares = [v / land_px * 100 for _, v, _ in parts]
    ax.barh(y, shares, color=[c for _, _, c in parts], height=.7)
    for yy, (_, v, _), sh in zip(y, parts, shares):
        ax.text(sh + 1, yy, f"{sh:.0f} %  ({v / 1e6:.1f} M)", va="center",
                color=INK, fontsize=9)
    ax.set_yticks(y, [name for name, _, _ in parts], fontsize=9)
    ax.set_xlim(0, max(shares) * 1.45)
    ax.set_title(f"France's {land_px / 1e6:.1f} M land pixels (250 m)",
                 color=INK, fontsize=11, loc="left")
    style_axes(ax)
    ax.grid(False)

    # Reasons per month.
    ax = fig.add_subplot(gs[1, 1])
    month = m.groupby(m.index.month).mean() * 100
    bottom = np.zeros(12)
    for name, c in REASONS:
        ax.bar(month.index, month[name], bottom=bottom, color=c, width=.75,
               label=name, edgecolor="white", lw=1)
        bottom += month[name].to_numpy()
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_ylabel("% of the fuel pixels", color=INK, fontsize=9)
    ax.set_title("Why fuel pixels are not usable (mean per month, "
                 "2014-2025)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center")
    style_axes(ax)

    # Fallback on the previous composites.
    ax = fig.add_subplot(gs[1, 2])
    for col, name, c in (
            ("unusable", "this composite only", "#0d366b"),
            ("unusable_16d", "or the previous one (−16 days)", "#2a78d6"),
            ("unusable_32d", "or the two previous (−32 days)", "#86b6ef")):
        v = month[col]
        ax.plot(v.index, v.values, color=c, lw=2, marker="o", markersize=5,
                label=name)
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_ylim(0, None)
    ax.set_ylabel("% of the fuel pixels still unusable", color=INK,
                  fontsize=9)
    ax.set_title("Falling back on older composites", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style_axes(ax)

    fig.suptitle("Which 250 m pixels the vegetation features use: fuel "
                 "pixels (one BD Forêt class on ≥ 75 % of the pixel) whose "
                 "value passes the quality flags", color=INK, fontsize=13,
                 x=.02, ha="left")
    out = PROCESSED / "vegetation_usable.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    summary = month[["unusable", "unusable_16d", "unusable_32d"]].round(1)
    print(f"usable pixels -> {out}\nwindow shares (%): "
          f"{dict(zip([s for s, _ in STATUS], share.round(1)))}\n"
          f"unusable per month (%):\n{summary.T}")


# --- raw values --------------------------------------------------------------

def raw_layers(date, dates):
    """Raw NDVI and NBR of a composite on the MODIS grid (NaN: no data)."""
    a, nodata = read(dates[date]["ndvi"])
    ndvi = np.where(a == nodata, np.nan, a * 1e-4)
    nir, nn = read(dates[date]["nir"])
    mir, mn = read(dates[date]["mir"])
    nir, mir = nir.astype("float64"), mir.astype("float64")
    ok = (nir != nn) & (mir != mn) & (nir + mir > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        nbr = np.where(ok, (nir - mir) / (nir + mir), np.nan)
    return ndvi, nbr


def raw_figure(grid, day, dates):
    composite = composite_of(day, dates)
    src = bv.modis_grid()
    fires, bounds = window(grid, day)
    ndvi, nbr = raw_layers(composite, dates)
    maps = [(ndvi, "NDVI", ramp("Greens"), Normalize(0, 1)),
            (nbr, "NBR", ramp("Blues"), Normalize(-.2, .8))]
    fig, axes = plt.subplots(2, 2, figsize=(15, 15), dpi=130,
                             gridspec_kw={"hspace": .12, "wspace": .05})
    for col, (a, name, cmap, norm) in enumerate(maps):
        cmap = cmap.copy()
        cmap.set_bad(NAN_C)
        for row, (b, res, label) in enumerate(
                ((L93_FRANCE, 1000, "France, 1 km (mean of the pixels)"),
                 (bounds, RES, "window of vegetation_pixels.png, 250 m"))):
            img = to_l93(a, src, b, res, Resampling.average
                         if res > RES else Resampling.nearest)
            ax = axes[row, col]
            im = ax.imshow(img, extent=(b[0], b[2], b[1], b[3]), cmap=cmap,
                           norm=norm, interpolation="nearest")
            if row == 0:
                x0, y0, x1, y1 = bounds
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                        color=FIRE_C, lw=1)
            else:
                ax.scatter(fires["x"], fires["y"], s=22, facecolor="none",
                           edgecolor=FIRE_C, linewidths=1.2, zorder=3)
            ax.set_xlim(b[0], b[2])
            ax.set_ylim(b[1], b[3])
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(f"Raw {name}, {label}", color=INK, fontsize=11,
                         loc="left")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                              shrink=.7, pad=.02, aspect=35, extend="both")
            style_colorbar(cb, name)
    fig.suptitle(f"Raw MODIS values of the composite of "
                 f"{composite:%d %B %Y} (16 days), every pixel, quality "
                 "flags ignored (grey: no data; black square: the window "
                 f"below; circles: fires of {day:%d %B %Y})", color=INK,
                 fontsize=12, x=.02, ha="left")
    out = PROCESSED / "vegetation_raw.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"raw maps -> {out}")


def raw_series(dates, years=(2019, 2023)):
    pixels = pd.read_parquet(bv.WORK / "fuel_pixels.parquet")
    cls = pixels["cls"].to_numpy()
    codes = [bv.CODE[c] for c in SERIES_CLASSES]
    rows = []
    for date in dates:
        if not years[0] <= date.year <= years[1]:
            continue
        ndvi, nbr, _ = bv.load(date)
        ok = ndvi != bv.MISSING
        for code in codes:
            m = ok & (cls == code)
            if m.sum() < 100:
                continue
            for name, v in (("ndvi", ndvi), ("nbr", nbr)):
                q = np.percentile(v[m] * 1e-4, [25, 50, 75])
                rows.append((date, code, name, *q))
    s = pd.DataFrame(rows, columns=["date", "cls", "idx", "q25", "q50",
                                    "q75"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), dpi=130, sharey=True,
                             gridspec_kw={"hspace": .35, "wspace": .06})
    for ax, name in zip(axes.flat, SERIES_CLASSES):
        for year in range(years[0], years[1] + 1):
            ax.axvspan(pd.Timestamp(f"{year}-07-01"),
                       pd.Timestamp(f"{year}-08-31"), color=GRID, lw=0,
                       zorder=0)
        for idx, color in (("ndvi", NDVI_C), ("nbr", NBR_C)):
            d = s[(s["cls"] == bv.CODE[name]) & (s["idx"] == idx)]
            ax.fill_between(d["date"], d["q25"], d["q75"], color=color,
                            alpha=.18, lw=0)
            ax.plot(d["date"], d["q50"], color=color, lw=1.6,
                    label=idx.upper())
        ax.set_title(LABELS[name], color=INK, fontsize=11, loc="left")
        ax.set_ylim(-.2, 1)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK,
                      loc="lower left", ncol=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("raw index", color=INK, fontsize=9)
    fig.suptitle(f"Raw NDVI and NBR of the usable fuel pixels per class, "
                 f"{years[0]}-{years[1]}: median and interquartile range "
                 "per composite (grey bands: July-August)", color=INK,
                 fontsize=13, x=.02, ha="left")
    out = PROCESSED / "vegetation_raw_series.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"raw series -> {out}")


# --- mixture correction ---------------------------------------------------

def correction_figure(dates, years=(2019, 2023)):
    """Does the mixture correction remove the bias of the less pure fuel
    pixels, and are the corrected values plausible?"""
    import mixture as mx

    pixels = pd.read_parquet(bv.WORK / "fuel_pixels.parquet")
    s = mx.static()
    main_share = s["main"]
    cls, cell = pixels["cls"].to_numpy(), pixels["cell_id"].to_numpy()
    group = np.where(main_share >= .95, 1, np.where(main_share < .90, 0, -1))
    key = cell.astype("int64") * 100 + cls
    ref = np.load(bv.WORK / "reference.npz")
    h = ref["ndvi"].astype("float64")
    with np.errstate(invalid="ignore"):
        table = (np.cumsum(h, 1) - h / 2) / h.sum(1, keepdims=True)

    gaps, shifts, corr = [], [], []
    for date in dates:
        if not years[0] <= date.year <= years[1]:
            continue
        for name, corrected in (("observed", False), ("corrected", True)):
            ndvi, nbr, _ = bv.load(date, corrected=corrected)
            ok = (ndvi != bv.MISSING) & (group >= 0)
            d = pd.DataFrame({"key": key[ok], "g": group[ok],
                              "ndvi": ndvi[ok] * 1e-4, "nbr": nbr[ok] * 1e-4})
            m = d.groupby(["key", "g"])[["ndvi", "nbr"]].agg(["mean", "size"])
            m = m.unstack("g").dropna()
            m = m[(m[("ndvi", "size", 0)] >= 10)
                  & (m[("ndvi", "size", 1)] >= 10)]
            gaps.append({"date": date, "version": name, **{
                idx: (m[(idx, "mean", 0)] - m[(idx, "mean", 1)]).mean()
                for idx in ("ndvi", "nbr")}})
            if date.month in (7, 8):
                ok = ndvi != bv.MISSING
                p = np.full(len(ndvi), np.nan)
                p[ok] = table[cls[ok], ndvi[ok].astype(int) + bv.OFFSET]
                d = pd.DataFrame({"cell": cell, "p": p,
                                  "pure": main_share >= .95}).dropna()
                a = d.groupby("cell")["p"].agg(["mean", "size"])
                b = d[d["pure"]].groupby("cell")["p"].agg(["mean", "size"])
                ab = a.merge(b, left_index=True, right_index=True,
                             suffixes=("", "_pure"))
                ab = ab[(ab["size"] >= 10) & (ab["size_pure"] >= 10)]
                shifts.append(pd.DataFrame({
                    "version": name, "cell": ab.index.to_numpy(),
                    "shift": (ab["mean"] - ab["mean_pure"]).to_numpy()}))
            if corrected:
                raw, _, _ = bv.load(date)
                ok = (raw != bv.MISSING) & (main_share < .90)
                corr.append(pd.DataFrame({
                    "month": date.month,
                    "delta": (ndvi[ok].astype(int) - raw[ok]) * 1e-4}))
    gaps = pd.DataFrame(gaps)
    shifts = pd.concat(shifts)
    shifts = shifts.groupby(["version", "cell"])["shift"].mean().unstack(0)
    corr = pd.concat(corr)

    fig = plt.figure(figsize=(18, 11.5), dpi=130)
    gs = fig.add_gridspec(2, 3, hspace=.38, wspace=.22)
    before_c, after_c = "#898781", CATEGORICAL[0]
    for k, idx in enumerate(("ndvi", "nbr")):
        ax = fig.add_subplot(gs[0, k])
        for version, color in (("observed", before_c),
                               ("corrected", after_c)):
            g = gaps[gaps["version"] == version]
            month = g.groupby(g["date"].dt.month)[idx].mean()
            ax.plot(month.index, month.values, color=color, lw=2,
                    marker="o", markersize=5, label=version)
        ax.axhline(0, color=MUTED, lw=.8)
        ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
        ax.set_title(f"{idx.upper()}: pixels 75-90 % pure minus pixels "
                     "≥ 95 % pure\n(same cell and class, mean per month)",
                     color=INK, fontsize=11, loc="left")
        ax.set_ylabel(f"{idx.upper()} difference", color=INK, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        style_axes(ax)

    ax = fig.add_subplot(gs[0, 2])
    land = pd.read_parquet(INTERIM / "landcover_cell.parquet").set_index(
        "cell_id")["lc_crops"]
    bins = np.linspace(-.12, .06, 37)
    for version, color in (("observed", before_c), ("corrected", after_c)):
        v = shifts[version].dropna()
        r = np.corrcoef(v, land.reindex(v.index))[0, 1]
        ax.hist(v, bins=bins, histtype="step", lw=2, color=color,
                label=f"{version}: median {v.median():+.3f}, "
                f"corr. with lc_crops {r:+.2f}")
    ax.axvline(0, color=MUTED, lw=.8)
    ax.set_title("Per cell, July-August: mean NDVI percentile of all fuel\n"
                 "pixels minus that of the ≥ 95 % pure ones", color=INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("shift of the cell's ndvi_pct", color=INK, fontsize=9)
    ax.set_ylabel("cells", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    style_axes(ax)

    ax = fig.add_subplot(gs[1, 0])
    q = corr.groupby("month")["delta"].quantile([.1, .25, .5, .75, .9])
    q = q.unstack()
    ax.fill_between(q.index, q[.1], q[.9], color=after_c, alpha=.12, lw=0,
                    label="10-90 %")
    ax.fill_between(q.index, q[.25], q[.75], color=after_c, alpha=.25,
                    lw=0, label="25-75 %")
    ax.plot(q.index, q[.5], color=after_c, lw=2, marker="o", markersize=5,
            label="median")
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_title("Correction of the NDVI (corrected − observed),\npixels "
                 "under 90 % pure", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style_axes(ax)

    # Example: a deciduous pixel with the largest cropland share among the
    # cells with references for both.
    crops_kind = len(bv.CLASSES) + mx.TYPES.index("crops")
    m = s["c_kind"] == crops_kind
    crop_share = np.zeros(len(main_share))
    crop_share[s["c_pos"][m]] = s["c_share"][m]
    cand = np.flatnonzero((cls == bv.CODE["broadleaf_decid"])
                          & (crop_share >= .15))
    p = cand[np.argmax(crop_share[cand])] if len(cand) else 0
    ci = s["fuel_ci"][p]
    series = []
    for date in dates:
        if not 2021 <= date.year <= 2022:
            continue
        raw, _, _ = bv.load(date)
        cor, _, _ = bv.load(date, corrected=True)
        with np.load(mx.CORRECTED / f"{date:%Y-%m-%d}.npz") as z:
            R = z["refs"][ci]
        nd = (R[:, 1] - R[:, 0]) / (R[:, 1] + R[:, 0])
        series.append((date, raw[p] * 1e-4 if raw[p] != bv.MISSING
                       else np.nan, cor[p] * 1e-4 if cor[p] != bv.MISSING
                       else np.nan, nd[cls[p] - 1], nd[crops_kind]))
    series = pd.DataFrame(series, columns=["date", "observed", "corrected",
                                           "class", "crops"])
    ax = fig.add_subplot(gs[1, 1:])
    for col, color, label in (
            ("crops", CATEGORICAL[1], "cropland reference of the cell"),
            ("class", CATEGORICAL[2], "deciduous reference of the cell "
             "(≥ 95 % pure pixels)"),
            ("observed", before_c, "pixel, observed"),
            ("corrected", after_c, "pixel, corrected")):
        ax.plot(series["date"], series[col], color=color, lw=2, marker="o",
                markersize=4, label=label)
    ax.set_ylabel("NDVI", color=INK, fontsize=9)
    ax.set_title(f"Example: deciduous fuel pixel with {crop_share[p]:.0%} "
                 f"cropland ({main_share[p]:.0%} deciduous), cell "
                 f"{pixels['cell_id'].iloc[p]}, 2021-2022", color=INK,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2,
              loc="lower left")
    style_axes(ax)

    st = pd.read_parquet(bv.WORK / "correction_stats.parquet")
    fig.suptitle(f"Mixture correction of the fuel pixels, {years[0]}-"
                 f"{years[1]}: "
                 f"{st['corrected'].sum() / st['usable'].sum():.0%}"
                 " of the values corrected, "
                 f"{st['implausible'].sum() / st['usable'].sum():.2%} "
                 "implausible (kept as observed)", color=INK, fontsize=13,
                 x=.02, ha="left")
    out = PROCESSED / "vegetation_correction.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    month_gap = gaps.groupby([gaps["version"], gaps["date"].dt.month])[
        ["ndvi", "nbr"]].mean().unstack(0).round(3)
    print(f"correction -> {out}\ngap 75-90 % minus >= 95 % per month:\n"
          f"{month_gap}\nper-cell July-August shift:\n"
          f"{shifts.describe().round(3)}\ncorrelation with lc_crops: "
          f"{shifts.corrwith(land.reindex(shifts.index)).round(2).to_dict()}"
          f"\nexample pixel {p}: crop share {crop_share[p]:.2f}")


def main(seed):
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    df = pd.read_parquet(PROCESSED / "firegrid.parquet", columns=["date"])
    day = example_days(df, seed)[0]
    dates = bv.composites()
    events = pd.read_parquet(bv.WORK / "burns.parquet")
    raw_figure(grid, day, dates)
    raw_series(dates)
    usable_figure(grid, day, dates, events)
    correction_figure(dates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args().seed)
