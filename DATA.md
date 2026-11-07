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

`train.parquet` and `public.parquet` hold every column below, in this
order. `private.parquet` holds the same columns minus `fire`, `n_fires` and
`area_m2`, which are hidden in `reference/private.csv` (`target`,
`n_fires`, `area_m2`, `date`) for the scoring program. `get_data` (Real and
Simulated) returns `fire` as `y` and the outcomes `n_fires`, `area_m2`
apart from `X`, as `outcomes` / `outcomes_public` (split per fold with `y`
under local CV).

**Static** columns are constant over the days of a cell. Words in *italics*
(fuel pixel, usable pixel, class percentile...) are defined in
[Details](#details).

## Columns

### Cell (static)

| Column | Description |
|---|---|
| `cell_id` | Cell identifier (integer, with gaps) |
| `date` | Day of the row (datetime). Also the grouping key of the metric |
| `x`, `y` | Cell centre, Lambert-93 (km) |
| `lon`, `lat` | Cell centre, WGS84 (degrees) |

### Roads (static, DGITM 2019 national network; NaN in Corsica)

| Column | Description |
|---|---|
| `road_km` | Length of national roads and motorways in the cell (km) |
| `motorway_km` | Of which motorways (km) |
| `road_traffic` | Traffic on them, thousands of vehicle-km per day |
| `road_dist` | Distance from the cell centre to the nearest national road or motorway (km) |

### Population (static, INSEE 1 km grid, 2019)

| Column | Description |
|---|---|
| `pop_density` | Inhabitants per km² of the cell (sea included, so coastal cells read low) |
| `inhabited_share` | Share of the cell's 1 km squares with ≥ 1 inhabitant |
| `sparse_share` | Share of the cell's 1 km squares with 1-50 inhabitants (scattered housing, a proxy of the wildland-urban interface) |

### Land cover (static, Corine Land Cover 2018)

Shares of the whole cell (625 km²). The Corine classes of each group:
[Land cover groups](#land-cover-groups).

| Column | Description |
|---|---|
| `land_share` | Share of the cell inland, in France: everything but the sea and foreign countries, lakes and rivers included (< 1 on coasts and borders) |
| `lc_urban` | Artificial surfaces: urban, industrial, roads, mines, green urban areas |
| `lc_crops` | Arable land and annual crops |
| `lc_vine_orchard` | Vineyards, orchards, olive groves |
| `lc_pasture` | Pastures |
| `lc_agri_natural` | Agriculture mixed with natural vegetation, agro-forestry |
| `lc_forest` | Broadleaf, coniferous and mixed forest |
| `lc_grassland` | Natural grassland |
| `lc_shrub` | Moors, heathland, transitional woodland-shrub |
| `lc_maquis` | Sclerophyllous vegetation (maquis, garrigue) |
| `lc_sparse` | Beaches, rocks, sparse vegetation, burnt areas, glaciers |
| `lc_water` | Wetlands, inland and coastal water |

### Forest types (static, BD Forêt v2)

Shares of the whole cell (625 km²) covered by BD Forêt polygons of that
type: [BD Forêt](#bd-forêt).

| Column | Description |
|---|---|
| `forest_broadleaf` | Broadleaf forest, closed or open, and poplar plantations |
| `forest_conifer` | Conifer forest, closed or open |
| `forest_mixed` | Mixed broadleaf-conifer forest, closed or open |
| `forest_open` | Open forest (tree cover 10-40 %), any composition (also counted in the three above) |
| `heath` | Heath (lande) |

### Dominant species (static, BD Forêt v2)

Shares of the whole cell covered by BD Forêt polygons of that dominant
species (a polygon with no species recorded counts in none). The most
fire-prone are the Mediterranean and Atlantic pines and the evergreen oaks.

| Column | Description |
|---|---|
| `sp_pine_maritime` | Maritime pine |
| `sp_pine_aleppo` | Aleppo pine |
| `sp_pine_scots` | Scots pine |
| `sp_pine_black` | Laricio and black pine |
| `sp_pine_mountain` | Mountain pine and Swiss stone pine |
| `sp_pine_other` | Other pines |
| `sp_pine_mixed` | Mixed pines |
| `sp_fir_spruce` | Fir, spruce |
| `sp_douglas` | Douglas fir |
| `sp_larch` | Larch |
| `sp_conifer_mixed` | Conifers with no dominant species |
| `sp_oak_evergreen` | Holm and cork oak |
| `sp_oak_deciduous` | Deciduous oaks |
| `sp_beech` | Beech |
| `sp_chestnut` | Chestnut |
| `sp_robinia` | Robinia (black locust) |
| `sp_poplar` | Poplar |
| `sp_broadleaf_mixed` | Broadleaves with no dominant species |
| `sp_mixed` | Broadleaves and conifers mixed |

### Fuel classes (static, BD Forêt v2)

The 14 vegetation classes on which the vegetation features are computed
(same phenology within a class): an alternative to the forest types and
species above. Shares of the whole cell (625 km²) covered by BD Forêt
polygons of that class: [Fuel classes](#fuel-classes).

| Column | Description |
|---|---|
| `fuel_broadleaf_decid` | Closed deciduous broadleaf forest (oaks, beech, chestnut...) |
| `fuel_broadleaf_med` | Closed broadleaf forest with no dominant species, Mediterranean zone |
| `fuel_oak_evergreen` | Closed holm and cork oak forest |
| `fuel_pine_maritime` | Closed maritime pine forest |
| `fuel_pine_aleppo` | Closed Aleppo pine forest |
| `fuel_pine_mountain` | Closed forest of other pines (Scots, black, mountain...) |
| `fuel_fir_spruce_douglas` | Closed fir, spruce and Douglas fir forest |
| `fuel_larch` | Closed larch forest |
| `fuel_conifer_mixed` | Closed conifer forest with no dominant species |
| `fuel_mixed` | Closed mixed broadleaf-conifer forest |
| `fuel_open_forest` | Open forest (tree cover 10-40 %) |
| `fuel_heath_med` | Heath in the Mediterranean zone (garrigue, maquis) |
| `fuel_heath_other` | Heath elsewhere (gorse, heather, broom, fern) |
| `fuel_grass` | Herbaceous formations |
| `fuel_px` | Number of *fuel pixels* of the cell, the 250 m pixels the vegetation features use (a cell holds about 11 600 pixels) |
| `fuel_pure_share` | Share of the cell's fuel area that lies in fuel pixels: low in fragmented landscapes, where the vegetation features rest on few pixels |

### Weather of the day (Météo-France SIM2, averaged over the cell)

| Column | Description |
|---|---|
| `t_mean` | Mean air temperature 2 m above the ground (°C) |
| `t_min` | Minimum air temperature 2 m above the ground (°C) |
| `t_max` | Maximum air temperature 2 m above the ground (°C) |
| `rain` | Liquid precipitation (mm) |
| `snow` | Solid precipitation (mm) |
| `wind` | Wind speed 10 m above the ground (m/s) |
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
| `rain_3d` | Rain sum over the previous 3 days (mm) |
| `rain_7d` | Rain sum over the previous 7 days (mm) |
| `rain_30d` | Rain sum over the previous 30 days (mm) |
| `days_since_rain` | Days since the last day with rain ≥ 1 mm |
| `t_max_7d` | Mean `t_max` over the previous 7 days (°C) |
| `rh_7d` | Mean `rh` over the previous 7 days (%) |
| `swi_prev` | `swi` of the previous day |

### Vegetation (MODIS, last 16-day composite completed before the day)

Computed on the cell's *usable* *fuel pixels* only. A *class percentile*
compares a pixel with every state its fuel class takes over the training
years: 0 = the driest, 1 = the greenest. Formulas:
[Vegetation columns](#vegetation-columns).

| Column | Description |
|---|---|
| `vi_px` | Number of usable fuel pixels behind the vegetation features of the row (they are NaN below 10) |
| `vi_valid_share` | `vi_px` / `fuel_px`: share of the fuel pixels usable in that composite (not cloudy, snowy, of poor quality or recently burnt) |
| `ndvi_pct` | Mean NDVI (greenness) class percentile of the usable fuel pixels |
| `ndvi_pct_p10` | 10th percentile of those pixel percentiles: the driest tenth of the cell's fuel |
| `ndvi_extreme` | Share of the usable fuel pixels with an NDVI percentile < 0.1 |
| `nbr_pct` | Mean NBR (water content) class percentile of the usable fuel pixels |
| `nbr_pct_p10` | 10th percentile of those pixel percentiles |
| `nbr_extreme` | Share of the usable fuel pixels with an NBR percentile < 0.1 |
| `dry_gap` | `nbr_pct` − `ndvi_pct`: negative when the vegetation dries faster than it browns |
| `burnt_1y` | Share of the fuel pixels detected as burnt (or cleared) in the year before the end of the composite |
| `burnt_5y` | Same over 5 years |
| `vi_age` | Days from the last day of the composite to the row's day (1 to 16) |

`vi_px`, `vi_valid_share`, `vi_age`, `fuel_px` and `fuel_pure_share` are
not meant as predictors (nothing forbids it): they tell how much data the
vegetation features rest on.

### Target and fire outcomes

| Column | Description |
|---|---|
| `fire` | Target: 1 if at least one wildfire alert started in the cell that day, else 0 |
| `n_fires` | Number of wildfire alerts that started in the cell that day (`fire` = `n_fires` > 0) |
| `area_m2` | Total burned area of those fires (m²), 0 when no fire |

`n_fires` and `area_m2` are kept with the label so that the objective can
move to them (fire count, area weighting...). They are outcomes of the
fire, not known beforehand: **never features**.

## Details

### Fire label

From the BDIFF, the French forest-fire database: every wildfire alert of
10 m² or more. A fire
is located at the centroid of its commune (the BDIFF gives no coordinates)
and goes to the cell holding that centroid, so a cell without any commune
centroid can never have a fire; such cells are not in the grid.

### Land cover groups

Corine Land Cover 2018 (100 m), level-3 codes of each `lc_*` group:

| Column | Corine codes |
|---|---|
| `lc_urban` | 111-142 |
| `lc_crops` | 211, 212, 213, 241, 242 |
| `lc_vine_orchard` | 221, 222, 223 |
| `lc_pasture` | 231 |
| `lc_agri_natural` | 243, 244 |
| `lc_forest` | 311, 312, 313 |
| `lc_grassland` | 321 |
| `lc_shrub` | 322, 324 |
| `lc_maquis` | 323 |
| `lc_sparse` | 331-335 |
| `lc_water` | 411-423, 511, 512, 521, 522 |
| `land_share` | Every code but 523 (sea and ocean), so lakes and rivers included |

The groups cover every code but 523, so `land_share` is the sum of the
`lc_*` columns.

### BD Forêt

The BD Forêt v2 (IGN) maps the forests and semi-natural vegetation of
France as polygons of 0.5 ha or more, photo-interpreted between 2007 and
2018 depending on the département. Each polygon has a vegetation type
(closed forest, tree cover ≥ 40 %, or open forest, 10-40 %, of broadleaves,
conifers or both; poplar plantation; heath; herbaceous formation; forest
without tree cover, i.e. clear-cut, storm-felled or burnt when mapped) and,
for forests, a dominant species when one covers at least 75 % of the
canopy. It is rasterised at 50 m (forest types and species) or 25 m (fuel
classes) to compute the shares of the cells.

### Fuel classes

The BD Forêt polygons grouped into 14 classes of similar phenology (how
the greenness and water content of the vegetation vary over the year):

| Class | BD Forêt polygons |
|---|---|
| `broadleaf_decid` | Closed broadleaf forest of deciduous oaks, beech, chestnut or robinia, or with no dominant species outside the Mediterranean zone |
| `broadleaf_med` | Closed broadleaf forest with no dominant species in the Mediterranean zone (mixed holm and downy oak...) |
| `oak_evergreen` | Closed forest of holm or cork oak |
| `pine_maritime` | Closed forest of maritime pine |
| `pine_aleppo` | Closed forest of Aleppo pine |
| `pine_mountain` | Closed forest of Scots, laricio / black, mountain or Swiss stone pine, other or mixed pines |
| `fir_spruce_douglas` | Closed forest of fir, spruce or Douglas fir |
| `larch` | Closed forest of larch (a conifer that sheds its needles) |
| `conifer_mixed` | Closed conifer forest with no dominant species |
| `mixed` | Closed forest mixing broadleaves and conifers |
| `open_forest` | Open forest, any composition |
| `heath_med` | Heath in the Mediterranean zone: garrigue, maquis |
| `heath_other` | Heath outside the Mediterranean zone: gorse, heather, broom, fern |
| `grass` | Herbaceous formation |

The **Mediterranean zone** is the union of the IGN sylvoecoregions of the
grandes régions écologiques J (Méditerranée) and K (Corse).

**Not fuel**: poplar plantations, forest without tree cover, and
everything the BD Forêt does not map (crops, pastures, vineyards, urban
areas, water, rocks...).

`fuel_<class>` is the share of the whole cell covered by the class: the BD
Forêt is rasterised at 25 m, and each 25 m square of the class counts in
the cell that holds the centre of its MODIS pixel (see below), i.e. in its
own cell up to ~120 m at the cell borders. The 14 shares sum to at most 1.

### NDVI and NBR

Both are normalised differences of two reflectances measured by the MODIS
sensor (satellite Terra), between -1 and 1:

- **NDVI** = (NIR − Red) / (NIR + Red), near infrared (0.86 µm) against
  red (0.65 µm). Chlorophyll absorbs red light and leaf tissue reflects
  near infrared, so the NDVI measures the amount of *green* vegetation:
  about 0.8-0.9 for a dense green canopy, 0.2-0.4 for dry grass or sparse
  shrubs, about 0 for bare soil, below 0 for water and snow. It falls when
  vegetation browns (senescence, leaf fall, curing of grass).
- **NBR** (normalised burn ratio) = (NIR − SWIR) / (NIR + SWIR), with the
  short-wave infrared at 2.1 µm (MODIS band 7), which leaf water absorbs.
  The NBR falls when leaves lose water and collapses on burnt ground (char
  and ash reflect SWIR, not NIR).

Lower means drier for both. They complement each other: evergreen shrubs
and conifers (maquis, holm oak, pines) lose water while staying green, so
their NBR falls before or without their NDVI.

The values come from the MODIS product MOD13Q1 version 6.1: one image
every 16 days (a **composite**: for each pixel, the best-quality
observation of the 16 days), on a 231.7 m sinusoidal grid (5.4 ha per
pixel), with a quality flag per pixel (VI Quality). NDVI, NIR and SWIR
(MIR) are distributed in units of 0.0001.

### Composite of a row

A composite covers 16 days, from its start date to start + 15 days. A row
uses the last composite whose 16 days all fall *before* the row's day, so
that a fire of the day (or of the following days) never shows in it;
`vi_age` = row's day − last day of that composite.

### Fuel pixels

A MODIS pixel is a **fuel pixel** of class *c* when *c* covers at least
75 % of its area (estimated from the 25 m squares whose centre falls in the
pixel). Pixels mixing classes, or fuel and non-fuel (forest edges, small
woods among fields), are left out. A fuel pixel belongs to the cell that
holds its centre. France has about 1.73 million fuel pixels.

### Usable pixels

A fuel pixel is **usable** in a composite when:

1. its VI Quality flag says the value was produced (MODLAND QA 0 or 1),
   its usefulness index is not one of the four lowest (≤ 11), the aerosol
   load is not high, and there are no mixed clouds, no snow or ice and no
   shadow;
2. its NDVI, NIR and SWIR values are present;
3. it is not a *burn candidate* in that composite, and no burn of that
   pixel known by that composite started in the 5 years (1826 days) before
   the composite's last day (see [Burn detection](#burn-detection)): burn
   scars would otherwise read as extreme drought.

The NDVI and NBR features of a composite use the same usable pixels.

### Class percentiles

For each fuel class *c* and each index (NDVI or NBR), the **reference
distribution** is the set of all the values of the usable pixels of class
*c* in all the **reference composites**: those whose 16 days all fall in
the training years 2015-2017 and 2019-2024 (the test years 2018 and 2025
are never used). It pools the pixels of the class all over France and all
the composites of the year, so it spans every state the class goes
through, from its driest summer to its greenest spring.

The **class percentile** of a usable pixel *p* of class *c* in a
composite, of value *v* (NDVI or NBR, in units of 0.0001), is its mid-rank
in that distribution:

```
pct(p) = ( #{reference values of c < v} + ½ · #{reference values of c = v} )
         / #{reference values of c}
```

pct = 0.1 means that 10 % of all the observations of that vegetation class
were drier (lower). The percentile is relative to the class, not to the
pixel's own history: a pixel that is drier than the rest of its class every
summer (south-facing slope, shallow soil, hotter region) reads low every
summer, and a class whose index barely moves (pines) gets percentiles as
spread as one whose index swings (deciduous oaks). Dormant vegetation
(leafless broadleaves, cured grass) is the driest state of most classes:
percentiles are low in winter.

### Burn detection

A burn makes the NBR collapse (and the NDVI fall) from one composite to
the next, and the scar lasts for months. Leaf fall, the curing of grass and
image artefacts can also lower the NBR quickly, so a drop only counts as a
burn when it is unusual for that pixel, for that time of year and for its
surroundings. Composite after composite, on the series of each fuel pixel:

- **recent baselines**: the median of the pixel's last 4 usable NBR values
  that were not burn candidates (at least 2 needed), and the same for the
  NDVI;
- **seasonal baseline**: the lowest usable NBR value of the pixel at the
  same time of year in the two previous years (same 16-day period, or the
  one before or after);
- **cell context**: for each NBR baseline, the median NBR drop of all the
  usable fuel pixels of the cell. When it is positive (and rests on at
  least 10 pixels), it is subtracted from the pixel's drop: a fire is
  local, while haze or an early leaf fall lowers a whole region;
- the pixel is a **burn candidate** in a composite where its NBR is at
  least 0.27 below both its recent and its seasonal baseline (after the
  cell context), and its NDVI at least 0.1 below its recent NDVI baseline;
- the burn is **confirmed** by the pixel's next usable composite if it
  passes the same tests (against the candidate's recent baselines and its
  own seasonal baseline); otherwise it was a false alarm (e.g. an
  undetected cloud). A composite whose NDVI and NBR equal the candidate's
  is skipped: the last composite of a year and the first of the next share
  3 days and can hold the same observation. The burn *starts* in the
  candidate composite and is *known* from the confirming one. After a
  confirmed burn the recent baselines restart from post-fire values.

Burns are only searched in composites starting between 18 February and 30
September (day of year 49 to 273), which also feed the baselines: with a
low autumn and winter sun, the NBR of mountain forests falls far below 0
every year while their NDVI stays high. A burn of late autumn or winter
(pastoral burning in the Pyrenees...) is detected, and dated, at the first
composite from 18 February where its scar still shows.

The seasonal baseline needs a year of data, and the rasters start in
November 2014: burns are detected from February 2016 on. 0.27 is the usual
lower bound of moderate burn severity for the NBR drop (Key & Benson,
2006). Fires much smaller than a pixel (5.4 ha) go unseen, and a clear-cut
or a storm can be detected as a burn: all remove fuel.

### Vegetation columns

For a cell *i* and the composite *t* of the row: *F* = the fuel pixels of
*i*; *U* = the pixels of *F* usable in *t*; pct = class percentiles. The
columns marked (NaN < 10) are NaN when |*U*| < 10.

| Column | Definition |
|---|---|
| `fuel_px` | \|*F*\| |
| `fuel_pure_share` | Area of the 25 m fuel squares (any class) lying in the fuel pixels of *i*, divided by the area of all the 25 m fuel squares lying in MODIS pixels whose centre is in *i* (NaN if that area is 0) |
| `vi_px` | \|*U*\| |
| `vi_valid_share` | \|*U*\| / \|*F*\|; 0 if \|*F*\| = 0 |
| `ndvi_pct` | Mean of the NDVI pct of the pixels of *U* (NaN < 10) |
| `ndvi_pct_p10` | 10th percentile (linear interpolation) of the NDVI pct of the pixels of *U* (NaN < 10) |
| `ndvi_extreme` | Share of the pixels of *U* with NDVI pct < 0.1 (NaN < 10) |
| `nbr_pct`, `nbr_pct_p10`, `nbr_extreme` | Same with the NBR pct (NaN < 10) |
| `dry_gap` | `nbr_pct` − `ndvi_pct` (NaN < 10) |
| `burnt_1y` | Number of pixels of *F* with a burn known by *t* that started within the 365 days before the last day of *t*, divided by \|*F*\| (NaN if \|*F*\| = 0) |
| `burnt_5y` | Same over 1826 days |
| `vi_age` | Row's day − (start of *t* + 15 days), in days |

Burns are detected from February 2016 only: `burnt_1y` and `burnt_5y` are
0 before, and `burnt_5y` covers less than 5 years until early 2021.

Sources, licenses and build details: the docstrings of `pipeline/build_*.py`.
