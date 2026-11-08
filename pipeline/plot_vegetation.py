"""Figures of the vegetation features (build_vegetation.py).

- ``fuel_maps.png``: share of each cell in the 14 fuel classes, number of
  fuel pixels and the share of the fuel area they hold;
- ``vegetation_dates.png``: the vegetation features of every cell on two
  example days drawn at random (one in summer, one in late winter), with
  the fires of that day;
- ``vegetation_pixels.png``, ``vegetation_pixels_winter.png``: every
  250 m pixel around the cell with most fires on the summer / winter
  example day: fuel class and NDVI / NBR class percentiles of the fuel
  pixels (lighter when the value comes from the previous composite),
  pixels that are not fuel (black), mixed-class pixels left out (blue)
  and fuel pixels without a usable value (grey);
- ``vegetation_series.png``: national mean of the features per composite,
  2015-2025, and their mean seasonal cycle;
- ``vegetation_fire_rate.png``: share of (cell, day) rows with a fire per
  decile of each feature, June-September of the training years;
- ``burns.png``: every burn detected, and the burnt area per year.

    python pipeline/plot_vegetation.py [--seed 0]

Writes to ``data/processed/``. Needs build_vegetation.py and assemble.py.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

import build_vegetation as bv
from common import CELL_SIZE, CRS_L93, INTERIM, PROCESSED, TEST_YEARS
from plot_map import (CELL_EDGE, CELL_FILL, INK, cell_squares, frame, ramp,
                      style_colorbar)

MUTED, GRID = "#6b6a64", "#e6e4dd"
NAN_C = "#a9a7a0"  # NaN cells: neutral grey, apart from every scale
# Reference categorical palette, fixed order; past 7 classes: "Other".
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#6250d6"]
OTHER_C = "#898781"
# Categorical slots 1-2 of the reference palette (validated pair).
NDVI_C, NBR_C = "#2a78d6", "#eb6834"
FIRE_C = "#0b0b0b"
TRAIN = ~pd.Index(range(2015, 2026)).isin(TEST_YEARS)
DRY = ramp("YlOrBr_r")  # class percentile: dark = low

LABELS = {
    "broadleaf_decid": "Deciduous broadleaf", "broadleaf_med":
    "Mediterranean broadleaf", "oak_evergreen": "Holm and cork oak",
    "pine_maritime": "Maritime pine", "pine_aleppo": "Aleppo pine",
    "pine_mountain": "Other pines", "fir_spruce_douglas":
    "Fir, spruce, Douglas", "larch": "Larch", "conifer_mixed":
    "Conifers, none dominant", "mixed": "Mixed forest", "open_forest":
    "Open forest", "heath_med": "Mediterranean heath",
    "heath_other": "Other heath", "grass": "Herbaceous",
}


def style_axes(ax):
    ax.grid(True, axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CELL_EDGE)
    ax.tick_params(labelsize=8, colors=INK, length=0)


def cell_map(ax, cells, values, cmap, norm, title, note=None):
    poly = PolyCollection(cells, array=np.ma.masked_invalid(values),
                          cmap=cmap, norm=norm, edgecolor="white",
                          linewidth=.3)
    poly.cmap.set_bad(NAN_C)
    ax.add_collection(poly)
    frame(ax, cells)
    ax.set_title(title, color=INK, fontsize=10, loc="left",
                 pad=16 if note else 6)
    if note:
        ax.text(0, 1.01, note, transform=ax.transAxes, color=MUTED,
                fontsize=8, va="bottom")
    return poly


# --- static fuel classes ---------------------------------------------------

def fuel_maps(grid):
    fuel = grid[["cell_id"]].merge(
        pd.read_parquet(INTERIM / "fuel_cell.parquet"), on="cell_id")
    cells = cell_squares(grid)
    fig, axes = plt.subplots(2, 8, figsize=(30, 9.5), dpi=130,
                             gridspec_kw={"wspace": .03, "hspace": .12})
    specs = [(f"fuel_{c}", LABELS[c], "Greens", 100., "% of the cell")
             for c in bv.CLASSES]
    specs += [("fuel_px", "Fuel pixels (250 m)", "Greys", 1., "pixels"),
              ("fuel_pure_share", "Fuel area in fuel pixels", "Greys",
               100., "% of the cell's fuel area")]
    for ax, (col, name, cmap, scale, unit) in zip(axes.flat, specs):
        v = fuel[col].to_numpy(dtype=float) * scale
        hi = np.nanpercentile(v, 99) or np.nanmax(v) or 1.
        note = (f"France {np.nansum(fuel[col]) * CELL_SIZE ** 2 / 1e6:,.0f}"
                " km²" if col.startswith("fuel_") and scale == 100.
                and col != "fuel_pure_share" else
                f"median {np.nanmedian(v):,.0f}")
        poly = cell_map(ax, cells, v, ramp(cmap), Normalize(0, hi), name,
                        note)
        cb = fig.colorbar(poly, ax=ax, orientation="horizontal",
                          extend="max", shrink=.75, pad=.02, aspect=30)
        style_colorbar(cb, unit)
    fig.suptitle(f"Fuel classes per {CELL_SIZE // 1000} km cell (BD Forêt "
                 "v2 regrouped by phenology; `fuel_*` columns), and the "
                 "fuel pixels the vegetation features use (grey)",
                 color=INK, fontsize=13, x=.02, ha="left")
    out = PROCESSED / "fuel_maps.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"fuel maps -> {out}")


# --- example days ----------------------------------------------------------

def example_days(df, seed):
    """One random day of July-August and one of February-March, training
    years."""
    rng = np.random.default_rng(seed)
    days = pd.Series(df["date"].unique())
    days = days[days.dt.year.isin(pd.Index(range(2015, 2026))[TRAIN])]
    summer = days[days.dt.month.isin([7, 8])].to_numpy()
    winter = days[days.dt.month.isin([2, 3])].to_numpy()
    return [pd.Timestamp(rng.choice(summer)), pd.Timestamp(rng.choice(winter))]


VEG_MAPS = [
    ("ndvi_pct", "NDVI class percentile (mean)", DRY, Normalize(0, 1),
     "0 = lowest value of the class"),
    ("nbr_pct", "NBR class percentile (mean)", DRY, Normalize(0, 1),
     "0 = lowest value of the class"),
    ("nbr_pct_p10", "NBR class percentile (lowest 10 %)", DRY,
     Normalize(0, 1), "10th percentile of the pixels"),
    ("nbr_extreme", "NBR extreme share", ramp("Reds"), Normalize(0, .5),
     "share of pixels < 0.1"),
    ("dry_gap", "dry_gap = nbr_pct − ndvi_pct", "RdBu",
     TwoSlopeNorm(0, -.3, .3), "red: drying faster than browning"),
    ("vi_valid_share", "Usable fuel pixels", ramp("Greys"), Normalize(0, 1),
     "share of the fuel pixels"),
]


def vegetation_dates(grid, df, days):
    cells = cell_squares(grid)
    centre = grid.set_index("cell_id")
    fig, axes = plt.subplots(len(days), len(VEG_MAPS),
                             figsize=(4 * len(VEG_MAPS), 4.6 * len(days)),
                             dpi=130, gridspec_kw={"wspace": .03,
                                                   "hspace": .18})
    for row, day in zip(axes, days):
        d = grid[["cell_id"]].merge(df[df["date"] == day], on="cell_id",
                                    how="left")
        fire = d.loc[d["fire"] == 1, "cell_id"]
        composite = day - pd.Timedelta(days=int(d["vi_age"].iloc[0]) + 15)
        for ax, (col, name, cmap, norm, note) in zip(row, VEG_MAPS):
            poly = cell_map(ax, cells, d[col].to_numpy(dtype=float), cmap,
                            norm, name, note)
            # Cell centres in km -> m.
            ax.scatter(centre.loc[fire, "x"], centre.loc[fire, "y"], s=9,
                       c=FIRE_C, marker="o", linewidths=0, zorder=3)
            cb = fig.colorbar(poly, ax=ax, orientation="horizontal",
                              shrink=.75, pad=.02, aspect=30)
            style_colorbar(cb, "")
        row[0].text(-.06, .5, f"{day:%d %B %Y}\ncomposite of "
                    f"{composite:%d %b}\n{len(fire)} cells with a fire",
                    transform=row[0].transAxes, ha="right", va="center",
                    color=INK, fontsize=11)
    fig.suptitle("Vegetation features on two random days (dots: cells "
                 "with a fire that day; grey cells: NaN, fewer than 10 "
                 "usable fuel pixels). In late winter dormant vegetation "
                 "gives most classes their lowest values: low percentiles "
                 "everywhere", color=INK, fontsize=13, x=.02, ha="left")
    out = PROCESSED / "vegetation_dates.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"example days {[f'{d:%Y-%m-%d}' for d in days]} -> {out}")


def to_l93(a, src, bounds, res, resampling=Resampling.nearest,
           nodata=np.nan):
    """Raster ``a`` on the MODIS grid -> Lambert-93 grid over ``bounds``."""
    crs, transform, _ = src
    xmin, ymin, xmax, ymax = bounds
    out = np.full((int((ymax - ymin) / res), int((xmax - xmin) / res)),
                  nodata, dtype="float32")
    reproject(a.astype("float32"), out, src_transform=transform,
              src_crs=crs, dst_transform=from_origin(xmin, ymax, res, res),
              dst_crs=CRS_L93, resampling=resampling, src_nodata=np.nan,
              dst_nodata=nodata)
    return out


# Status of a MODIS pixel in the pixel view.
NOT_FUEL, MIXED, NO_VALUE, OBSERVED, PREVIOUS = range(5)
NOT_FUEL_C = "#000000"
# Blue marks the mixed pixels (far from the yellow-brown percentile scale);
# the class panel never uses it.
MIXED_C = CATEGORICAL[0]
CLASS_C = CATEGORICAL[1:6]


def pixel_status(composite, src):
    """Status (NOT_FUEL...PREVIOUS) of every MODIS pixel on the MODIS grid,
    the corrected NDVI / NBR the features use for the fuel pixels (observed,
    or carried over from the previous composite), and the fuel pixels."""
    _, transform, shape = src
    pixels = pd.read_parquet(bv.WORK / "fuel_pixels.parquet")
    pix = pixels["pix"].to_numpy()
    counts = pd.concat([pd.read_parquet(f) for f in
                        bv.FUEL_CACHE.glob("*.parquet")])
    counts = counts.groupby(["pix", "cls"])["n"].sum()
    full = abs(transform.a * transform.e) / bv.RES ** 2
    fuel = counts.groupby("pix").sum() / full
    top = counts.groupby("pix").max() / full
    status = np.full(shape[0] * shape[1], NOT_FUEL, dtype="float32")
    status[fuel.index[(fuel >= bv.PURITY) & (top < bv.PURITY)]] = MIXED

    dates = list(bv.composites())
    prev = dates[dates.index(composite) - 1]
    events = pd.read_parquet(bv.WORK / "burns.parquet")

    def burnt(date):
        end = date + pd.Timedelta(days=bv.COMPOSITE_DAYS - 1)
        e = events[(events["known_start"] <= date) & (
            events["burn_start"] > end - pd.Timedelta(days=1826))]
        out = np.zeros(len(pix), dtype=bool)
        out[e["pos"].to_numpy()] = True
        return out

    ndvi, nbr, _ = bv.load(composite, corrected=True)
    p_ndvi, p_nbr, _ = bv.load(prev, corrected=True)
    excl = burnt(composite)
    observed = (ndvi != bv.MISSING) & ~excl
    previous = ((ndvi == bv.MISSING) & ~excl & (p_ndvi != bv.MISSING)
                & ~burnt(prev))
    status[pix] = NO_VALUE
    status[pix[observed]] = OBSERVED
    status[pix[previous]] = PREVIOUS
    values = {name: np.where(observed, v, np.where(previous, pv, bv.MISSING))
              for name, v, pv in (("ndvi", ndvi, p_ndvi),
                                  ("nbr", nbr, p_nbr))}
    return status.reshape(shape), values, pixels


def compose(status, value, cmap, norm):
    """RGBA image: black not fuel, violet mixed, grey no value, percentile
    colours, lighter when carried over from the previous composite."""
    rgba = np.zeros(status.shape + (4,))
    rgba[:] = to_rgba(NOT_FUEL_C)
    rgba[status == MIXED] = to_rgba(MIXED_C)
    rgba[status == NO_VALUE] = to_rgba(NAN_C)
    used = (status == OBSERVED) | (status == PREVIOUS)
    col = cmap(norm(np.nan_to_num(value)))
    rgba[used] = col[used]
    light = status == PREVIOUS
    rgba[light, :3] = .45 * col[light, :3] + .55
    return rgba


def vegetation_pixels(grid, df, day, out_name="vegetation_pixels.png",
                      half=3):
    """Every MODIS pixel in the (2 half + 1)^2 cells around the cell with
    most fires of ``day``: fuel class and NDVI / NBR class percentiles of
    the fuel pixels (as the features use them), the other pixels by
    status."""
    d = df[df["date"] == day]
    composite = (day - pd.Timedelta(days=int(d["vi_age"].iloc[0]) + 15)
                 ).to_pydatetime()
    fires = pd.read_parquet(INTERIM / "fires.parquet")
    fires = fires[fires["date"] == day]
    top = fires["cell_id"].value_counts().index[0]
    g = grid.set_index("cell_id")
    n = 2 * half + 1
    x0 = (g.loc[top, "ix"] - half) * CELL_SIZE
    y0 = (g.loc[top, "iy"] - half) * CELL_SIZE
    bounds = (x0, y0, x0 + n * CELL_SIZE, y0 + n * CELL_SIZE)
    src = bv.modis_grid()
    shape = src[2]

    status, values, pixels = pixel_status(composite, src)
    pix, cls = pixels["pix"].to_numpy(), pixels["cls"].to_numpy()
    st = to_l93(status, src, bounds, 250)
    ref = np.load(bv.WORK / "reference.npz")
    pct = {}
    for name in ("ndvi", "nbr"):
        h = ref[name].astype("float64")
        with np.errstate(invalid="ignore"):  # class 0 (not fuel) is empty
            table = (np.cumsum(h, axis=1) - h / 2) / h.sum(axis=1,
                                                           keepdims=True)
        v = values[name]
        a = np.full(shape[0] * shape[1], np.nan, dtype="float32")
        ok = v != bv.MISSING
        a[pix[ok]] = table[cls[ok], v[ok].astype(int) + bv.OFFSET]
        pct[name] = to_l93(a.reshape(shape), src, bounds, 250)

    # Fuel classes of the window by pixel count: 5 colours, then Other.
    a = np.full(shape[0] * shape[1], np.nan, dtype="float32")
    a[pix] = cls
    cls_img = to_l93(a.reshape(shape), src, bounds, 250)
    fuel = ~np.isnan(cls_img)
    counts = pd.Series(cls_img[fuel].astype(int)).value_counts()
    shown = list(counts.index[:len(CLASS_C)])
    palette = CLASS_C[:len(shown)] + [OTHER_C]
    slot = np.full(len(bv.CLASSES) + 1, len(shown))
    slot[shown] = range(len(shown))
    cls_rgba = np.zeros(st.shape + (4,))
    cls_rgba[:] = to_rgba(NOT_FUEL_C)
    cls_rgba[st == MIXED] = to_rgba(MIXED_C)
    lut = np.array([to_rgba(c) for c in palette])
    cls_rgba[fuel] = lut[slot[cls_img[fuel].astype(int)]]

    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.4), dpi=130,
                             gridspec_kw={"wspace": .05})
    panels = [(cls_rgba, "Fuel class of each fuel pixel", None),
              (compose(st, pct["ndvi"], DRY, Normalize(0, 1)),
               "NDVI class percentile", DRY),
              (compose(st, pct["nbr"], DRY, Normalize(0, 1)),
               "NBR class percentile", DRY)]
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, extent=extent, interpolation="nearest")
        for k in range(n + 1):
            ax.axvline(x0 + k * CELL_SIZE, color=CELL_EDGE, lw=.6)
            ax.axhline(y0 + k * CELL_SIZE, color=CELL_EDGE, lw=.6)
        # White rings outlined in black: visible on any background.
        ax.scatter(fires["x"], fires["y"], s=30, facecolor="none",
                   edgecolor="black", linewidths=2.6, zorder=3)
        ax.scatter(fires["x"], fires["y"], s=30, facecolor="none",
                   edgecolor="white", linewidths=1.2, zorder=4)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        if cmap is None:
            handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k],
                                     label=LABELS[bv.CLASSES[c - 1]])
                       for k, c in enumerate(shown)]
            if len(counts) > len(shown):
                handles.append(plt.Rectangle((0, 0), 1, 1, color=OTHER_C,
                                             label="Other classes"))
            ax.legend(handles=handles, loc="upper left",
                      bbox_to_anchor=(0, -.01), ncol=2, frameon=False,
                      fontsize=8, labelcolor=INK)
        else:
            cb = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap),
                              cax=ax.inset_axes([.1, -.07, .8, .035]),
                              orientation="horizontal")
            style_colorbar(cb, "0 = lowest value of the class (lighter: "
                           "value of the previous composite)")
    keys = [(NOT_FUEL_C, "Not fuel (under 75 % fuel, no fuel, water)"),
            (MIXED_C, "At least 75 % fuel but no class at 75 %: excluded"),
            (NAN_C, "Fuel pixel without a usable value (cloudy in both "
             "composites, or recently burnt)")]
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=t)
                        for c, t in keys],
               loc="upper right", bbox_to_anchor=(.9, .04), ncol=1,
               frameon=False, fontsize=9, labelcolor=INK)
    win = st[~np.isnan(st)].astype(int)
    used = np.isin(win, [OBSERVED, PREVIOUS]).sum()
    fig.suptitle(f"250 m pixels around the cell with most fires on "
                 f"{day:%d %B %Y} (composite of {composite:%d %b}; "
                 "circles: fires of the day at their commune centroid; grid: "
                 f"25 km cells)\nFuel pixels used: {used:,}, of which "
                 f"{(win == PREVIOUS).sum():,} from the previous composite;"
                 f" mixed-class pixels excluded: {(win == MIXED).sum():,}",
                 color=INK, fontsize=12, x=.02, ha="left")
    out = PROCESSED / out_name
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    names = ["not fuel", "mixed", "no value", "observed", "previous"]
    print(f"pixels around cell {top} on {day:%Y-%m-%d} -> {out}; window: "
          f"{dict(zip(names, np.bincount(win, minlength=5).tolist()))}")


# --- national series ---------------------------------------------------------

def vegetation_series():
    v = pd.read_parquet(INTERIM / "vegetation_cell.parquet")
    w = v["vi_px"].astype(float)

    def national(cols):
        out = {c: (v[c] * w).groupby(v["composite_start"]).sum()
               / w.where(v[c].notna()).groupby(v["composite_start"]).sum()
               for c in cols}
        return pd.DataFrame(out)

    nat = national(["ndvi_pct", "nbr_pct", "ndvi_extreme", "nbr_extreme"])
    fig = plt.figure(figsize=(17, 12.5), dpi=130)
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=.42, wspace=.12)
    axes = [(fig.add_subplot(gs[r, 0]), fig.add_subplot(gs[r, 1]))
            for r in range(2)]
    specs = [("pct", "Mean class percentile of the usable fuel pixels "
              "(France)", (0, 1)),
             ("extreme", "Share of usable fuel pixels below the class's "
              "10th percentile (France)", (0, None))]
    for (kind, title, ylim), (ax, ax2) in zip(specs, axes):
        for idx, color in (("ndvi", NDVI_C), ("nbr", NBR_C)):
            s = nat[f"{idx}_{kind}"]
            ax.plot(s.index, s.values, color=color, lw=1.6)
            ax.text(s.index[-1] + pd.Timedelta(days=25), s.values[-1],
                    idx.upper(), color=INK, fontsize=9, va="center")
            clim = s.groupby(s.index.month).mean()
            ax2.plot(clim.index, clim.values, color=color, lw=2,
                     marker="o", markersize=4, label=idx.upper())
        for year in TEST_YEARS:
            ax.axvspan(pd.Timestamp(f"{year}-01-01"),
                       pd.Timestamp(f"{year}-12-31"), color=GRID, lw=0,
                       zorder=0)
            ax.text(pd.Timestamp(f"{year}-07-01"), 1.0, "test year",
                    transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", color=MUTED, fontsize=8)
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=16)
        ax.set_ylim(*ylim)
        style_axes(ax)
        ax2.set_title("Mean per month", color=INK, fontsize=11, loc="left",
                      pad=16)
        ax2.set_xticks(range(1, 13), list("JFMAMJJASOND"))
        ax2.set_ylim(*ylim)
        ax2.legend(frameon=False, fontsize=8, labelcolor=INK)
        style_axes(ax2)
    # Fire season, one bar pair per year.
    ax = fig.add_subplot(gs[2, :])
    summer = nat[nat.index.month.isin([7, 8])].groupby(
        lambda d: d.year)[["ndvi_pct", "nbr_pct"]].mean()
    for col, color in (("ndvi_pct", NDVI_C), ("nbr_pct", NBR_C)):
        ax.plot(summer.index, summer[col], color=color, lw=2, marker="o",
                markersize=7, label=col.split("_")[0].upper())
    low = summer.mean(axis=1).idxmin()
    ax.annotate(f"{low}: driest summer", (low, summer.loc[low].min()),
                xytext=(0, -22), textcoords="offset points", ha="center",
                color=INK, fontsize=9)
    for year in TEST_YEARS:
        ax.axvspan(year - .5, year + .5, color=GRID, lw=0, zorder=0)
        ax.text(year, 1.0, "test year", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=MUTED, fontsize=8)
    ax.set_xticks(summer.index)
    ax.set_ylim(.5, .75)
    ax.set_title("Fire season: mean class percentile of the July-August "
                 "composites (NDVI / NBR), lower = drier", color=INK,
                 fontsize=11, loc="left", pad=16)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2,
              loc="upper left")
    style_axes(ax)
    fig.suptitle("NDVI and NBR class percentiles over time: weighted by usable fuel "
                 "pixels, one point per 16-day composite", color=INK,
                 fontsize=13, x=.02, ha="left")
    out = PROCESSED / "vegetation_series.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    july = nat[nat.index.month.isin([7, 8])].groupby(
        lambda d: d.year).mean().round(3)
    print(f"series -> {out}; July-August means:\n{july}")


# --- fire rate per decile ------------------------------------------------------

def fire_rate(df):
    """Fire rate per decile, deciles over all rows (rows 1-2: dry places
    and dry periods mixed) and within each cell (row 3: dry periods of a
    cell only)."""
    d = df[df["date"].dt.month.isin([6, 7, 8, 9])
           & ~df["date"].dt.year.isin(TEST_YEARS)]
    base = d["fire"].mean() * 100
    feats = [("ndvi_pct", "NDVI class percentile (mean)"),
             ("ndvi_pct_p10", "NDVI class percentile (lowest 10 %)"),
             ("nbr_pct", "NBR class percentile (mean)"),
             ("nbr_pct_p10", "NBR class percentile (lowest 10 %)"),
             ("nbr_extreme", "NBR extreme share"),
             ("dry_gap", "dry_gap")]
    within = [("ndvi_pct_p10", "NDVI pct (lowest 10 %), within cell"),
              ("nbr_pct_p10", "NBR pct (lowest 10 %), within cell"),
              ("dry_gap", "dry_gap, within cell")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5), dpi=130,
                             gridspec_kw={"hspace": .55, "wspace": .12},
                             sharey=True)

    def bars(ax, dec, title, ticks, xlabel):
        rate = d.groupby(dec)["fire"].mean() * 100
        ax.bar(rate.index, rate.values, width=.8, color=NDVI_C)
        ax.axhline(base, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_xticks(rate.index, ticks or rate.index, fontsize=7)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
        style_axes(ax)

    for ax, (col, name) in zip(axes[:2].flat, feats):
        dec = pd.qcut(d[col].rank(method="first"), 10, labels=False) + 1
        low = d.groupby(dec)[col].min()
        bars(ax, dec, name, [f"{v:.2f}" for v in low],
             "decile over all rows, labelled by its lowest value")
    for ax, (col, name) in zip(axes[2], within):
        r = d.groupby("cell_id")[col].rank(method="first", pct=True)
        dec = np.ceil(r * 10).clip(1, 10).astype("Int64")
        bars(ax, dec, name, None, "decile within each cell (1 = lowest)")
    for ax in axes[:, 0]:
        ax.set_ylabel("% of (cell, day) rows with a fire", color=INK,
                      fontsize=9)
    fig.text(.02, .335, "Within each cell: the cell's own deciles, so "
             "only the dry periods of a place count, not dry places",
             color=INK, fontsize=11)
    fig.suptitle("Fire rate per decile of each feature: June-September, "
                 "training years, all cells, NaN rows left out (dashed: "
                 f"all rows, {base:.2f} %)",
                 color=INK, fontsize=13, x=.02, ha="left")
    out = PROCESSED / "vegetation_fire_rate.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"fire rate per decile -> {out}")


# --- burns -------------------------------------------------------------------

def burns_map(grid):
    b = pd.read_parquet(bv.WORK / "burns.parquet")
    crs, transform, shape = bv.modis_grid()
    rows, cols = np.divmod(b["pix"].to_numpy(), shape[1])
    xs, ys = transform * (cols + .5, rows + .5)
    x, y = Transformer.from_crs(crs, CRS_L93, always_xy=True).transform(
        xs, ys)
    px_km2 = abs(transform.a * transform.e) / 1e6
    cells = cell_squares(grid)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 9), dpi=130,
                                  gridspec_kw={"width_ratios": [2, 1],
                                               "wspace": .08})
    ax.add_collection(PolyCollection(cells, facecolor=CELL_FILL,
                                     edgecolor=CELL_EDGE, linewidth=.3))
    ax.scatter(x, y, s=1.5, c=NBR_C, linewidths=0, zorder=3)
    frame(ax, cells)
    ax.set_title(f"{len(b):,} burnt fuel pixels (250 m) detected, "
                 f"{len(b) * px_km2:,.0f} km²", color=INK, fontsize=11,
                 loc="left")
    year = b.groupby(b["burn_start"].dt.year).size() * px_km2
    year = year.reindex(range(2016, 2026), fill_value=0)
    ax2.barh(year.index, year.values, height=.7, color=NBR_C)
    for yr, val in year.items():
        ax2.text(val + year.max() * .01, yr, f"{val:.0f}", va="center",
                 color=INK, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_yticks(year.index)
    ax2.set_title("Burnt fuel area per year of the burn (km²)", color=INK,
                  fontsize=11, loc="left")
    style_axes(ax2)
    ax2.grid(True, axis="x", color=GRID, lw=.6)
    ax2.grid(False, axis="y")
    fig.suptitle("Burns detected on the NBR series (burnt_1y, burnt_5y): "
                 "searched from February 2016, composites starting "
                 "18 Feb-30 Sep; clear-cuts count too", color=INK,
                 fontsize=13, x=.02, ha="left")
    out = PROCESSED / "burns.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"burns -> {out}")


def main(seed):
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    grid["x"], grid["y"] = ((grid["ix"] + .5) * CELL_SIZE,
                            (grid["iy"] + .5) * CELL_SIZE)
    cols = ["cell_id", "date", "fire", "vi_age", "vi_px", "vi_valid_share",
            "ndvi_pct", "ndvi_pct_p10", "ndvi_extreme", "nbr_pct",
            "nbr_pct_p10", "nbr_extreme", "dry_gap"]
    df = pd.read_parquet(PROCESSED / "firegrid.parquet", columns=cols)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    days = example_days(df, seed)
    fuel_maps(grid)
    vegetation_dates(grid, df, days)
    vegetation_pixels(grid, df, days[0])
    vegetation_pixels(grid, df, days[1], "vegetation_pixels_winter.png")
    vegetation_series()
    fire_rate(df)
    burns_map(grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args().seed)
