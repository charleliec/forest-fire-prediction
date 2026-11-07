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

### Fire outcomes (every split)

| Column | Description |
|---|---|
| `n_fires` | Number of wildfire alerts that started in the cell that day (`fire` = `n_fires` > 0) |
| `area_m2` | Total burned area of those fires (m²), 0 when no fire |

Kept with the label so the objective can move to them (fire count, area
weighting...). They are outcomes of the fire, not known beforehand: **never
features**.

- `train.parquet`, `public.parquet`: columns next to `fire`.
- private test: hidden like `fire`, in `reference/private.csv` (`target`,
  `n_fires`, `area_m2`, `date`), not in `private.parquet`.
- `get_data` (Real and Simulated) returns them apart from `X`, as
  `outcomes` / `outcomes_public`; the objective splits them per fold with `y`.

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
| `lc_*` | Corine Land Cover 2018 shares of the cell: `urban`, `crops`, `vine_orchard`, `pasture`, `agri_natural`, `forest`, `grassland`, `shrub`, `maquis`, `sparse` (beaches, rocks, sparse vegetation, burnt areas, glaciers), `water` |
| `forest_broadleaf`, `forest_conifer`, `forest_mixed`, `forest_open`, `heath` | BD Forêt v2 shares of the cell by forest type |
| `sp_*` | BD Forêt v2 shares of the cell by dominant species, one column per species value of the source: pines `sp_pine_maritime`, `sp_pine_aleppo`, `sp_pine_scots`, `sp_pine_black` (laricio, black), `sp_pine_mountain` (mountain, stone), `sp_pine_other`, `sp_pine_mixed`; other conifers `sp_fir_spruce`, `sp_douglas`, `sp_larch`, `sp_conifer_mixed`; broadleaves `sp_oak_evergreen` (holm, cork), `sp_oak_deciduous`, `sp_beech`, `sp_chestnut`, `sp_robinia`, `sp_poplar`, `sp_broadleaf_mixed`; `sp_mixed` (broadleaves and conifers). Forest with no species recorded counts in none. The most fire-prone are the Mediterranean and Atlantic pines (`sp_pine_maritime`, `sp_pine_aleppo`, `sp_pine_black`) and the evergreen oaks (`sp_oak_evergreen`) |

Sources, licenses and build details: the docstrings of `pipeline/build_*.py`.
