# Data dictionary

**One row = one (25 km cell, day)** over metropolitan France (977 cells,
Corsica included), 2015-2025. Built by `pipeline/` into
`data/processed/firegrid.parquet`, then split by `benchmark/datasets/real.py`.

| Split | Years | Label | File |
|---|---|---|---|
| train | 2015-2017, 2019-2024 | yes | `train.parquet` |
| public test | 2018 | yes | `public.parquet` |
| private test | 2025 | hidden | `private.parquet` (features only) |

Files live in `$BENCHOPT_DATA_HOME/real/` (default `benchmark/data/real/`).
About 0.6% of rows have a fire.

## Target

| Column | Description |
|---|---|
| `fire` | 1 if at least one wildfire alert (BDIFF, ≥ 10 m²) started in the cell that day, else 0 |

`n_fires` and `area_m2` (fire count, burned area) exist in the source table
but are **dropped**. They are outcomes of the fire, never features.

## Features

### Cell (static)

| Column | Description |
|---|---|
| `cell_id` | Cell identifier (integer, with gaps) |
| `x`, `y` | Cell centre, Lambert-93 (km) |
| `lon`, `lat` | Cell centre, WGS84 (degrees) |

### Roads (static, DGITM 2019 national network; NaN in Corsica)

| Column | Description |
|---|---|
| `road_km` | Length of national roads + motorways in the cell (km) |
| `motorway_km` | Of which motorways (km) |
| `road_traffic` | Traffic on them, thousands of vehicle-km per day |
| `road_dist` | Distance from cell centre to nearest national road (km) |

### Population (static, INSEE 1 km grid, 2019)

| Column | Description |
|---|---|
| `pop_density` | Inhabitants per km² of the cell (sea included, so coastal cells read low) |
| `inhabited_share` | Share of the cell's 1 km squares with ≥ 1 inhabitant |
| `sparse_share` | Share of 1 km squares with 1-50 inhabitants (scattered housing, wildland-urban interface proxy) |

### Day

| Column | Description |
|---|---|
| `date` | Day of the row (datetime). Also the grouping key of the metric |

### Weather of the day (Météo-France SIM2, averaged over the cell)

| Column | Description |
|---|---|
| `t_mean`, `t_min`, `t_max` | Mean / min / max 2 m temperature (°C) |
| `rain` | Liquid precipitation (mm) |
| `snow` | Solid precipitation (mm) |
| `wind` | 10 m wind speed (m/s) |
| `rh` | Relative humidity (%) |
| `q` | Specific humidity (g/kg) |
| `sw_rad` | Shortwave (solar) radiation (J/cm²) |
| `lw_rad` | Longwave radiation (J/cm²) |
| `etp` | Potential evapotranspiration, Penman (mm) |
| `evap` | Actual evapotranspiration (mm) |
| `swi` | Soil wetness index |
| `wg_root` | Root-zone soil water content |
| `snow_depth` | Snow depth (m) |

### Antecedent weather (days strictly **before** the row's day)

| Column | Description |
|---|---|
| `rain_3d`, `rain_7d`, `rain_30d` | Rain sum over the previous 3 / 7 / 30 days (mm) |
| `days_since_rain` | Days since the last day with rain ≥ 1 mm |
| `t_max_7d` | Mean `t_max` over the previous 7 days (°C) |
| `rh_7d` | Mean `rh` over the previous 7 days (%) |
| `swi_prev` | `swi` of the previous day |

## Being added (not in the current files yet)

| Column(s) | Description |
|---|---|
| `ndvi` | Last MODIS NDVI composite (250 m, 16 days) completed before the day, cell mean |
| `ndvi_anom` | `ndvi` minus the cell's usual NDVI for that period (training years) |
| `ndvi_age` | Age of that composite (days) |
| `land_share` | Share of the cell that is land (< 1 on coasts / borders) |
| `lc_*` | Corine Land Cover 2018 shares of the cell: `urban`, `crops`, `vine_orchard`, `pasture`, `agri_natural`, `forest`, `grassland`, `shrub`, `maquis`, `sparse`, `water` |
| `forest_broadleaf`, `forest_conifer`, `forest_mixed`, `forest_open`, `heath` | BD Forêt v2 shares of the cell by forest type |
| `pine_maritime`, `pine_aleppo`, `pine_other`, `fir_spruce`, `oak_evergreen` | BD Forêt v2 shares of the cell by dominant species |

Sources, licenses and build details: the docstrings of `pipeline/build_*.py`.
