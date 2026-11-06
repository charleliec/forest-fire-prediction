"""Shared paths and constants of the (private) data-building pipeline.

The pipeline turns the raw sources into one table, ``data/processed/
firegrid.parquet``: one row per (25 km cell, day) over metropolitan France,
2015-2025, with the weather of the day, antecedent weather, the last MODIS
NDVI composite available before the day, and the fire label.

    python pipeline/build_grid.py      # SIM2 points -> 25 km cells
    python pipeline/build_fires.py     # BDIFF -> fires per (cell, day)
    python pipeline/build_weather.py   # SIM2 -> weather per (cell, day)
    python pipeline/build_ndvi.py      # AppEEARS MOD13Q1 -> NDVI per cell
    python pipeline/build_roads.py     # national road traffic -> per cell
    python pipeline/build_population.py  # INSEE 1 km grid -> per cell
    python pipeline/build_landcover.py   # Corine Land Cover -> per cell
    python pipeline/build_forest.py      # IGN BD Forêt v2 -> per cell
    python pipeline/assemble.py        # -> data/processed/firegrid.parquet

``benchmark/datasets/real.py`` then cuts the competition splits from it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

BDIFF_CSV = RAW / "Incendies.csv"
SIM2_DIR = RAW / "sim2"
GEO_DIR = RAW / "geo"
NDVI_DIR = RAW / "ndvi"
ROADS_DIR = RAW / "roads"
POP_DIR = RAW / "population"
LANDCOVER_DIR = RAW / "landcover"
BDFORET_DIR = RAW / "bdforet"

FIRST_YEAR, LAST_YEAR = 2015, 2025
# Antecedent features (rolling rain, NDVI) need data before FIRST_YEAR.
WARMUP_YEAR = FIRST_YEAR - 1
TEST_YEARS = (2018, 2025)

# Grid: square cells in Lambert-93 (EPSG:2154), metres.
CELL_SIZE = 25_000
CRS_L93 = "EPSG:2154"
# SIM2 LAMBX/LAMBY are Lambert II etendu coordinates in hectometres.
CRS_SIM2 = "EPSG:27572"


def sim2_file(year):
    return SIM2_DIR / f"QUOT_SIM2_{year}.csv.gz"


def cell_index(x, y):
    """Integer (ix, iy) of the 25 km cell containing Lambert-93 (x, y)."""
    import numpy as np
    return (np.floor_divide(x, CELL_SIZE).astype(int),
            np.floor_divide(y, CELL_SIZE).astype(int))
