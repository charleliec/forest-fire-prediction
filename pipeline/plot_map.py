"""Maps of the data, train and test years merged (FIRST_YEAR..LAST_YEAR).

- ``map.png``: grid cells, national roads, and every BDIFF fire that falls in
  a grid cell, drawn at its commune centroid, sized by burnt area;
- ``meteo_maps.png``: one map per key weather variable, each cell coloured
  by its average over all days;
- ``population_maps.png``: population density and scattered housing per
  cell (INSEE 1 km grid), and the yearly fire days of a cell against its
  population density;
- ``landcover_maps.png``: share of each cell in the main Corine Land Cover
  groups (build_landcover.py);
- ``forest_maps.png``: share of each cell in each BD Forêt forest type and
  dominant species (build_forest.py).

    python pipeline/plot_map.py

Writes to ``data/processed/``. Needs the whole pipeline to have run
(``build_roads.py`` for the shapefile, ``build_population.py``,
``assemble.py`` for the weather).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D

from build_roads import SHP
from common import CELL_SIZE, FIRST_YEAR, INTERIM, LAST_YEAR, PROCESSED

CELL_FILL, CELL_EDGE = "#f1efe8", "#c8c5bb"
ROAD, FIRE, INK = "#3d3d3a", "#d9480f", "#2b2b29"


# (column, title, unit, colormap, aggregation): single-hue sequential scales.
METEO = [
    ("t_max", "Daily max temperature", "°C, mean", "Reds", "mean"),
    ("rain", "Rainfall", "mm per year", "Blues", "year"),
    ("days_since_rain", "Days since last rain", "days, mean", "Oranges",
     "mean"),
    ("rh", "Relative humidity", "%, mean", "Greens", "mean"),
    ("wind", "Wind speed at 10 m", "m/s, mean", "Purples", "mean"),
    ("swi", "Soil wetness index", "mean", "Greys", "mean"),
]


def cell_squares(grid):
    """(n, 4, 2) corners of the cells, Lambert-93 (m)."""
    x0 = grid["ix"].to_numpy() * CELL_SIZE
    y0 = grid["iy"].to_numpy() * CELL_SIZE
    c = CELL_SIZE
    return np.stack([np.c_[x0, y0], np.c_[x0 + c, y0],
                     np.c_[x0 + c, y0 + c], np.c_[x0, y0 + c]], axis=1)


def frame(ax, cells):
    """Fit the axes on the cells, equal aspect, no axes."""
    pad = CELL_SIZE
    ax.set_xlim(cells[..., 0].min() - pad, cells[..., 0].max() + pad)
    ax.set_ylim(cells[..., 1].min() - pad, cells[..., 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()


def size(area_m2):
    """Marker area (pt^2), log of the burnt area: 100 m2 -> 1, 1 km2 -> 13."""
    return 1. + 3. * np.log10(np.clip(area_m2, 100., None) / 100.)


def fire_map(grid):
    fires = pd.read_parquet(INTERIM / "fires.parquet")
    fires = fires[fires["cell_id"].isin(grid["cell_id"])
                  & fires["year"].between(FIRST_YEAR, LAST_YEAR)]
    fires = fires.sort_values("area_m2")  # big fires on top

    cells = cell_squares(grid)

    reader = shapefile.Reader(str(SHP), encoding="utf-8")
    lines = {True: [], False: []}
    for rec, shp in zip(reader.iterRecords(), reader.iterShapes()):
        pts = np.asarray(shp.points)
        bounds = list(shp.parts) + [len(pts)]
        lines[rec["route"].startswith("A")] += [
            pts[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 1]

    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    ax.add_collection(PolyCollection(cells, facecolor=CELL_FILL,
                                     edgecolor=CELL_EDGE, linewidth=.4))
    ax.add_collection(LineCollection(lines[False], colors=ROAD,
                                     linewidths=.5, alpha=.7))
    ax.add_collection(LineCollection(lines[True], colors=ROAD,
                                     linewidths=1.1))
    ax.scatter(fires["x"], fires["y"], s=size(fires["area_m2"]), c=FIRE,
               alpha=.45, linewidths=0, zorder=3)

    frame(ax, cells)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CELL_FILL, edgecolor=CELL_EDGE,
                      label=f"{len(grid)} cells of {CELL_SIZE // 1000} km"),
        Line2D([], [], color=ROAD, lw=1.1, label="Motorway"),
        Line2D([], [], color=ROAD, lw=.5, alpha=.7, label="National road"),
        *[Line2D([], [], ls="", marker="o", color=FIRE, alpha=.6,
                 markeredgewidth=0, markersize=np.sqrt(size(a)), label=lab)
          for a, lab in [(1e2, "Fire, 100 m²"), (1e4, "Fire, 1 ha"),
                         (1e6, "Fire, 1 km²"), (1e8, "Fire, 100 km²")]],
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              labelcolor=INK, fontsize=9)
    ax.set_title(f"{len(fires):,} fires (BDIFF, {FIRST_YEAR}-{LAST_YEAR}, "
                 "train and test), national road network (TMJA 2019)",
                 color=INK, fontsize=11, loc="left")

    out = PROCESSED / "map.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"{len(fires)} fires, {len(lines[True])} motorway and "
          f"{len(lines[False])} other road parts -> {out}")


def meteo_maps(grid):
    cols = [m[0] for m in METEO]
    df = pd.read_parquet(PROCESSED / "firegrid.parquet",
                         columns=["cell_id", *cols])
    per_cell = df.groupby("cell_id")[cols].mean()
    per_cell = grid[["cell_id"]].join(per_cell, on="cell_id")
    cells = cell_squares(grid)

    fig, axes = plt.subplots(2, 3, figsize=(15, 11), dpi=150,
                             gridspec_kw={"wspace": .02, "hspace": .08})
    for ax, (col, title, unit, cmap, agg) in zip(axes.flat, METEO):
        v = per_cell[col].to_numpy() * (365.25 if agg == "year" else 1.)
        # Clip the scale so a few outliers (Alps) don't flatten the rest.
        lo, hi = np.nanpercentile(v, [2, 98])
        # Start the ramp above white so low cells stay visible.
        ramp = ListedColormap(plt.get_cmap(cmap)(np.linspace(.12, 1., 256)))
        poly = PolyCollection(cells, array=v, cmap=ramp, clim=(lo, hi),
                              edgecolor="white", linewidth=.3)
        ax.add_collection(poly)
        frame(ax, cells)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        cb = fig.colorbar(poly, ax=ax, orientation="horizontal", extend="both",
                          shrink=.7, pad=.02, aspect=30)
        cb.set_label(unit, color=INK, fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=INK, length=0)
    fig.suptitle(f"Weather per {CELL_SIZE // 1000} km cell, average over "
                 f"{FIRST_YEAR}-{LAST_YEAR} (SIM2, train and test)",
                 color=INK, fontsize=13, x=.02, ha="left")
    out = PROCESSED / "meteo_maps.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"{len(cols)} weather maps -> {out}")


def ramp(cmap):
    """Single-hue colormap starting above white, so low cells stay visible."""
    return ListedColormap(plt.get_cmap(cmap)(np.linspace(.12, 1., 256)))


def style_colorbar(cb, label):
    cb.set_label(label, color=INK, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK, length=0)


def population_maps(grid):
    pop = grid[["cell_id"]].merge(
        pd.read_parquet(INTERIM / "population_cell.parquet"), on="cell_id")
    fires = pd.read_parquet(INTERIM / "fires_cell_day.parquet")
    fires = fires[fires["date"].dt.year.between(FIRST_YEAR, LAST_YEAR)]
    n_years = LAST_YEAR - FIRST_YEAR + 1
    pop["fire_days"] = pop["cell_id"].map(
        fires.groupby("cell_id").size() / n_years).fillna(0.)
    cells = cell_squares(grid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), dpi=150,
                             gridspec_kw={"wspace": .08,
                                          "width_ratios": [1, 1, 1.15]})
    maps = [
        ("pop_density", "Population density",
         "inhabitants per km² (log scale)", "Purples",
         LogNorm(1, np.nanpercentile(pop["pop_density"], 99.5))),
        ("sparse_share", "Scattered housing",
         "share of the cell's 1 km squares with 1-50 inhabitants", "Greens",
         None),
    ]
    for ax, (col, title, unit, cmap, norm) in zip(axes, maps):
        v = pop[col].to_numpy()
        poly = PolyCollection(cells, array=np.clip(v, 1, None) if norm else v,
                              cmap=ramp(cmap), norm=norm,
                              edgecolor="white", linewidth=.3)
        ax.add_collection(poly)
        frame(ax, cells)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        cb = fig.colorbar(poly, ax=ax, orientation="horizontal",
                          extend="both" if norm else "neither",
                          shrink=.7, pad=.02, aspect=30)
        style_colorbar(cb, unit)

    # Fire days against density: each cell, and the mean per density decile.
    ax = axes[2]
    x = pop["pop_density"].clip(lower=1.)
    ax.scatter(x, pop["fire_days"], s=9, c=CELL_EDGE, linewidths=0,
               zorder=2)
    dec = pd.qcut(x, 10)
    binned = pop.assign(x=x).groupby(dec, observed=True).agg(
        x=("x", "median"), y=("fire_days", "mean"))
    ax.plot(binned["x"], binned["y"], color=FIRE, lw=2, marker="o", ms=8,
            markeredgecolor="white", markeredgewidth=2, zorder=3)
    peak = binned["y"].idxmax()
    ax.annotate(f"{binned.loc[peak, 'y']:.1f} fire days / year",
                (binned.loc[peak, "x"], binned.loc[peak, "y"]),
                xytext=(0, 12), textcoords="offset points", ha="center",
                color=INK, fontsize=9)
    ax.set_xscale("log")
    # Linear below 1, log above: the many quiet cells and the busy ones.
    ax.set_yscale("symlog", linthresh=1., linscale=.6)
    ax.set_ylim(0, pop["fire_days"].max() * 1.3)
    ax.yaxis.set_major_formatter("{x:g}")
    ax.set_xlabel("inhabitants per km² (log scale)", color=INK, fontsize=9)
    ax.set_ylabel("fire days per year", color=INK, fontsize=9)
    ax.set_title("Fire days per year against density", color=INK,
                 fontsize=11, loc="left", pad=20)
    ax.text(0, 1.015, "each cell (grey), mean per density decile (line)",
            transform=ax.transAxes, color="#6b6a64", fontsize=9,
            va="bottom")
    ax.grid(True, color="#e6e4dd", lw=.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CELL_EDGE)
    ax.tick_params(labelsize=8, colors=INK, length=0)
    ax.set_box_aspect(1)

    fig.suptitle(f"Population per {CELL_SIZE // 1000} km cell (INSEE "
                 f"Filosofi 2019, 1 km grid) and fires {FIRST_YEAR}-"
                 f"{LAST_YEAR} (BDIFF, train and test)",
                 color=INK, fontsize=13, x=.02, ha="left")
    out = PROCESSED / "population_maps.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"population maps -> {out}; fire days per year by density decile:"
          f"\n{binned.round(2)}")


# (column, title, colormap): the hue says the family of the land cover.
LANDCOVER = [
    ("lc_urban", "Artificial surfaces", "Greys"),
    ("lc_crops", "Arable land", "Oranges"),
    ("lc_vine_orchard", "Vineyards, orchards, olive groves", "Oranges"),
    ("lc_pasture", "Pastures", "Oranges"),
    ("lc_agri_natural", "Farmland with natural vegetation", "Oranges"),
    ("lc_forest", "Forest", "Greens"),
    ("lc_grassland", "Natural grassland", "Greens"),
    ("lc_shrub", "Moors, heath, woodland-shrub", "Greens"),
    ("lc_maquis", "Maquis and garrigue", "Greens"),
    ("lc_sparse", "Rocks, sand, sparse vegetation", "Greys"),
]
FOREST = [
    ("forest_broadleaf", "Broadleaf forest", "Greens"),
    ("forest_conifer", "Conifer forest", "Greens"),
    ("forest_mixed", "Mixed forest", "Greens"),
    ("forest_open", "Open forest (10-40 % canopy)", "Greens"),
    ("heath", "Heathland", "Greens"),
    ("pine_maritime", "Maritime pine", "Purples"),
    ("pine_aleppo", "Aleppo pine", "Purples"),
    ("pine_other", "Other pines", "Purples"),
    ("fir_spruce", "Fir, spruce, Douglas, larch", "Purples"),
    ("oak_evergreen", "Holm and cork oak", "Purples"),
]


def share_maps(grid, table, specs, title, out):
    """2 x 5 maps of static shares of the cell, each on its own scale."""
    df = grid[["cell_id"]].merge(table, on="cell_id", how="left")
    cells = cell_squares(grid)
    fig, axes = plt.subplots(2, 5, figsize=(22, 10.5), dpi=150,
                             gridspec_kw={"wspace": .03, "hspace": .1})
    for ax, (col, name, cmap) in zip(axes.flat, specs):
        v = df[col].to_numpy() * 100.
        # Clip the top so a few cells don't flatten the rest.
        hi = np.nanpercentile(v, 99) or np.nanmax(v) or 1.
        poly = PolyCollection(cells, array=v, cmap=ramp(cmap), clim=(0, hi),
                              edgecolor="white", linewidth=.3)
        ax.add_collection(poly)
        frame(ax, cells)
        ax.set_title(name, color=INK, fontsize=11, loc="left")
        ax.text(0, .985, f"France mean {np.nanmean(v):.1f} %",
                transform=ax.transAxes, color="#6b6a64", fontsize=9,
                va="top")
        cb = fig.colorbar(poly, ax=ax, orientation="horizontal",
                          extend="max", shrink=.75, pad=.02, aspect=30)
        style_colorbar(cb, "% of the cell")
    fig.suptitle(title, color=INK, fontsize=13, x=.02, ha="left")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"{len(specs)} maps -> {out}")


def landcover_maps(grid):
    share_maps(grid, pd.read_parquet(INTERIM / "landcover_cell.parquet"),
               LANDCOVER, f"Land cover per {CELL_SIZE // 1000} km cell "
               "(Corine Land Cover 2018): grey artificial or bare, orange "
               "farmland, green natural vegetation",
               PROCESSED / "landcover_maps.png")


def forest_maps(grid):
    share_maps(grid, pd.read_parquet(INTERIM / "forest_cell.parquet"),
               FOREST, f"Forest type per {CELL_SIZE // 1000} km cell (IGN "
               "BD Forêt v2): green composition and canopy, purple dominant "
               "species", PROCESSED / "forest_maps.png")


def main():
    grid = pd.read_parquet(INTERIM / "grid.parquet")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    fire_map(grid)
    meteo_maps(grid)
    population_maps(grid)
    landcover_maps(grid)
    forest_maps(grid)


if __name__ == "__main__":
    main()
