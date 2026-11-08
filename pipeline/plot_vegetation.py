"""Figures of the vegetation features (build_vegetation.py).

- ``fuel_maps.png``: share of each cell in the 14 fuel classes, number of
  fuel pixels and the share of the fuel area they hold;
- ``vegetation_dates.png``: the vegetation features of every cell on two
  example days drawn at random (one in summer, one in late winter), with
  the fires of that day;
- ``vegetation_pixels.png``: the 250 m fuel pixels behind those features,
  NDVI and NBR class percentiles, around the cell with most fires on the
  summer example day;
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
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from pyproj import Transformer

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


def vegetation_pixels(grid, df, day, half=3):
    """Pixel percentiles in the (2 half + 1)^2 cells around the cell with
    most fires of ``day``."""
    d = df[df["date"] == day]
    composite = day - pd.Timedelta(days=int(d["vi_age"].iloc[0]) + 15)
    fires = pd.read_parquet(INTERIM / "fires.parquet")
    fires = fires[fires["date"] == day]
    top = fires["cell_id"].value_counts().index[0]
    g = grid.set_index("cell_id")
    ix0, iy0 = g.loc[top, "ix"] - half, g.loc[top, "iy"] - half
    n = 2 * half + 1
    x0, y0 = ix0 * CELL_SIZE, iy0 * CELL_SIZE

    pixels = pd.read_parquet(bv.WORK / "fuel_pixels.parquet")
    crs, transform, shape = bv.modis_grid()
    rows, cols = np.divmod(pixels["pix"].to_numpy(), shape[1])
    xs, ys = transform * (cols + .5, rows + .5)
    x, y = Transformer.from_crs(crs, CRS_L93, always_xy=True).transform(
        xs, ys)
    res = 250
    size = n * CELL_SIZE // res
    i = ((y0 + n * CELL_SIZE - y) // res).astype(int)
    j = ((x - x0) // res).astype(int)
    inside = (i >= 0) & (i < size) & (j >= 0) & (j < size)

    ref = np.load(bv.WORK / "reference.npz")
    ndvi, nbr, _ = bv.load(composite.to_pydatetime(), corrected=True)
    cls = pixels["cls"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6), dpi=130,
                             gridspec_kw={"wspace": .05})
    images = {}
    for name, v in (("ndvi", ndvi), ("nbr", nbr)):
        h = ref[name].astype("float64")
        with np.errstate(invalid="ignore"):  # class 0 (not fuel) is empty
            table = (np.cumsum(h, axis=1) - h / 2) / h.sum(axis=1,
                                                           keepdims=True)
        ok = inside & (v != bv.MISSING)
        img = np.full((size, size), np.nan)
        img[i[ok], j[ok]] = table[cls[ok], v[ok].astype(int) + bv.OFFSET]
        images[name] = img
    # Classes of the window by pixel count; the 8th and next fold in Other.
    counts = pd.Series(cls[inside]).value_counts()
    shown = list(counts.index[:len(CATEGORICAL)])
    slot = {c: k for k, c in enumerate(shown)}
    cls_img = np.full((size, size), np.nan)
    cls_img[i[inside], j[inside]] = [slot.get(c, len(shown))
                                     for c in cls[inside]]

    extent = (x0, x0 + n * CELL_SIZE, y0, y0 + n * CELL_SIZE)
    palette = CATEGORICAL[:len(shown)] + [OTHER_C]
    panels = [
        (cls_img, ListedColormap(palette), Normalize(-.5, len(shown) + .5),
         "Fuel class of each fuel pixel"),
        (images["ndvi"], DRY, Normalize(0, 1), "NDVI class percentile"),
        (images["nbr"], DRY, Normalize(0, 1), "NBR class percentile"),
    ]
    for ax, (img, cmap, norm, title) in zip(axes, panels):
        cmap = cmap.copy()
        cmap.set_bad(NAN_C)
        im = ax.imshow(img, extent=extent, cmap=cmap, norm=norm,
                       interpolation="nearest")
        for k in range(n + 1):
            ax.axvline(x0 + k * CELL_SIZE, color=CELL_EDGE, lw=.6)
            ax.axhline(y0 + k * CELL_SIZE, color=CELL_EDGE, lw=.6)
        ax.scatter(fires["x"], fires["y"], s=22, facecolor="none",
                   edgecolor=FIRE_C, linewidths=1.2, zorder=3)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        if title.startswith("Fuel"):
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
            cb = fig.colorbar(im, cax=ax.inset_axes([.1, -.07, .8, .035]),
                              orientation="horizontal")
            style_colorbar(cb, "0 = lowest value of the class")
    fig.suptitle(f"250 m fuel pixels around the cell with most fires on "
                 f"{day:%d %B %Y} (composite of {composite:%d %b}; circles: "
                 f"fires of the day at their commune centroid; grid: 25 km "
                 "cells; grey: not fuel, or not usable)", color=INK,
                 fontsize=12, x=.02, ha="left")
    out = PROCESSED / "vegetation_pixels.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"pixels around cell {top} on {day:%Y-%m-%d} -> {out}")


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
    vegetation_series()
    fire_rate(df)
    burns_map(grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args().seed)
