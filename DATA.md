# Data dictionary

**One row = one (25 km cell, day)** over metropolitan France (977 cells,
Corsica included), 2015-2025. Built by `pipeline/` into
`data/processed/firegrid.parquet`, then split by `benchmark/datasets/real.py`.
Download: [GitHub release data-v1](https://github.com/charleliec/forest-fire-prediction/releases/tag/data-v1)
(setup steps in [CONTRIBUTING.md](CONTRIBUTING.md#2-data)).

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

### Calendar (school and public holidays of the day)

People start most fires, and they are out more on days off. School
holidays come from the Ministry of Education calendar per académie
(data.education.gouv.fr) from September 2017, and from the zone A / B / C
compilation of A. Augusti before (zones were redrawn in September 2015;
Corsica has its own calendar, in no zone). Holidays run from the first
day off to the Sunday before school resumes. These columns are the same
for every cell on a given day.

| Column | Description |
|---|---|
| `holiday_pop_share` | Share of the inhabitants of metropolitan France whose académie is on school holiday (INSEE 1 km grid, 2019) |
| `holiday_zone_a`, `holiday_zone_b`, `holiday_zone_c` | 1 if zone A / B / C is on school holiday |
| `summer_holiday` | 1 during the summer school holidays (zones A / B / C) |
| `short_holiday` | 1 during any other school holiday (Toussaint, Christmas, winter, spring, Ascension bridge) of at least one zone |
| `public_holiday` | 1 on a national public holiday (the two extra days of Alsace-Moselle are not counted) |
| `bridge_day` | 1 on a working day between two days off ("pont", e.g. the Friday after Ascension) |
| `long_weekend` | 1 on a day of a run of 3 or more days off (weekends, public holidays, bridge days) |

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
years, all seasons together: 0 = the lowest value seen for the class, 1 =
the highest. For most classes the lowest values come in late winter
(leafless or dormant vegetation), not in summer: read them with the date.
Formulas: [Vegetation columns](#vegetation-columns).

| Column | Description |
|---|---|
| `vi_px` | Number of usable fuel pixels behind the vegetation features of the row (they are NaN below 10) |
| `vi_valid_share` | `vi_px` / `fuel_px`: share of the fuel pixels with a usable value, in that composite or the previous one (not cloudy, snowy, of poor quality or recently burnt) |
| `vi_prev_share` | Share of the `vi_px` pixels whose value comes from the previous composite (16 days earlier), because it had no usable value in this one (cloud, snow...) |
| `ndvi_pct` | Mean NDVI (greenness) class percentile of the usable fuel pixels |
| `ndvi_pct_p10` | 10th percentile of those pixel percentiles: the lowest tenth of the cell's fuel |
| `ndvi_extreme` | Share of the usable fuel pixels with an NDVI percentile < 0.1 |
| `nbr_pct` | Mean NBR (water content) class percentile of the usable fuel pixels |
| `nbr_pct_p10` | 10th percentile of those pixel percentiles |
| `nbr_extreme` | Share of the usable fuel pixels with an NBR percentile < 0.1 |
| `dry_gap` | `nbr_pct` − `ndvi_pct`: negative when the vegetation dries faster than it browns |
| `burnt_1y` | Share of the fuel pixels detected as burnt (or cleared) in the year before the end of the composite |
| `burnt_5y` | Same over 5 years |
| `vi_age` | Days from the last day of the composite to the row's day (1 to 16) |

`vi_px`, `vi_valid_share`, `vi_prev_share`, `vi_age`, `fuel_px` and
`fuel_pure_share` are
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
woods among fields), are left out; the rest of a fuel pixel (up to 25 %)
is corrected for ([Mixture correction](#mixture-correction)). A fuel
pixel belongs to the cell that holds its centre. France has about 1.73
million fuel pixels.

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

A fuel pixel that fails 1 or 2 in a composite (cloud, snow, poor quality,
missing value), but not 3, takes the value it had in the **previous
composite** (16 days earlier), if it was usable there: it then counts as
usable, and `vi_prev_share` counts it. Only values observed in the previous
composite carry over, so a value is never used more than 16 days after its
own composite. The reference distributions and the burn detection use
the values of each composite itself only, never carried-over ones.

The NDVI and NBR features of a composite use the same usable pixels.

### Mixture correction

A fuel pixel is at least 75 % its main class; the rest of it (8.7 % on
average: 5 % non-fuel land, 3.7 % other fuel classes) adds its own signal.
A field harvested in July inside a forest pixel, for instance, lowers the
pixel's NDVI just when fires happen. Before the percentiles, the values of
every usable fuel pixel are corrected to those its main class would have
alone:

1. **What else is in the pixel.** The 25 m BD Forêt squares give the
   share of each other fuel class. The rest of the pixel is non-fuel,
   split between the Corine Land Cover groups found in the pixel:
   cropland, vineyards and orchards, pastures, agriculture mixed with
   natural vegetation, urban, wetlands, open water, sparse vegetation and
   rock. The part that Corine maps as forest or natural vegetation there
   (Corine ignores patches under 25 ha) is **other non-fuel**.
2. **Local references**, per cell and composite: the median reflectance
   (red, NIR, SWIR) of
   - another fuel class: its usable fuel pixels at least 95 % pure in the
     cell;
   - a Corine group: up to 200 pixels of the cell (drawn once) with less
     than 10 % BD Forêt fuel and at least 75 % of that group;
   - other non-fuel: up to 200 pixels of the cell with less than 10 % BD
     Forêt fuel, any group.

   A reference needs 10 usable pixels; otherwise it is the median of the
   references of the 3 × 3 cells around, otherwise the national median of
   that composite.
3. **Correction**, band by band, on the reflectances (light mixes
   linearly within a pixel, NDVI and NBR do not; the red is recovered as
   NIR × (1 − NDVI) / (1 + NDVI)):

   ```
   main = (observed − Σ_k share_k × reference_k) / (1 − Σ_k share_k)
   ```

   over the other contents *k* of the pixel, then NDVI and NBR are
   recomputed from the corrected reflectances. A content with no reference
   at all is assumed to look like the main class (it stays in the
   denominator). If a corrected reflectance falls outside 0-1, the pixel
   keeps its observed values.

The reference distributions and the features use the corrected values; the
burn detection uses the observed ones.

77 % of the usable values are corrected (the others belong to pixels made
of their main class only), 0.5 % give an implausible reflectance and are
kept as observed. 90 % of the contaminant share has a reference from its
own cell, 7 % from the cells around, 3 % national.

**Effect** (2019-2023). Before the correction, the fuel pixels 75-90 %
pure read lower than the ≥ 95 % pure ones of the same cell and class,
most in July-August (NDVI −0.019, NBR −0.027): this lowered the
July-August `ndvi_pct` of a typical cell by 0.035, most in cropland cells
(correlation −0.53 with `lc_crops`). After it the gap is 0.000 for the
NDVI and −0.007 for the NBR, the median cell shift +0.002 (correlation
−0.20). The price is noise at pixel level: a pixel's own neighbours (a
wheat or a maize field) differ from the cell's median, which widens the
spread of the corrected pixels around their class reference by about 10 %
in summer and 25 % in winter; averages over the cell's pixels smooth it
out.

### Class percentiles

For each fuel class *c* and each index (NDVI or NBR), the **reference
distribution** is the set of all the (corrected) values of the usable
pixels of class *c* in all the **reference composites**: those whose 16 days all fall in
the training years 2015-2017 and 2019-2024 (the test years 2018 and 2025
are never used). It pools the pixels of the class all over France and all
the composites of the year, so it spans every state the class goes
through, from its leafless or dormant winter and its driest summer to its
greenest spring.

The **class percentile** of a usable pixel *p* of class *c* in a
composite, of value *v* (NDVI or NBR, in units of 0.0001), is its mid-rank
in that distribution:

```
pct(p) = ( #{reference values of c < v} + ½ · #{reference values of c = v} )
         / #{reference values of c}
```

pct = 0.1 means that 10 % of all the observations of that vegetation class
were lower (less green for the NDVI, less leaf water for the NBR). The
percentile is relative to the class, not to the pixel's own history: a
pixel that is drier than the rest of its class every summer (south-facing
slope, shallow soil, hotter region) reads low every summer, and a class
whose index barely moves (pines) gets percentiles as spread as one whose
index swings (deciduous oaks).

The reference mixes all seasons, so the percentile follows the yearly
cycle of each class: leaf fall, cured grass and the low winter sun give
most classes their lowest values in February-March, and their highest in
May-June (deciduous broadleaves: about 0.15 in February-March, 0.70 in
July-August). Only the Mediterranean classes (Aleppo pine, Mediterranean
heath) reach their lowest values in summer. A low percentile thus means
"dry" in summer but "leafless" in winter: the date tells which.

### Burn detection

A burn makes the NBR collapse (and the NDVI fall) from one composite to
the next, and the scar lasts for months. Leaf fall, the curing of grass and
image artefacts can also lower the NBR quickly, so a drop only counts as a
burn when it is unusual for that pixel, for that time of year and for its
surroundings. Composite after composite, on the observed (uncorrected)
series of each fuel pixel:

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

**After snow.** When the snow melts, the flattened, still-brown vegetation
underneath has a low NBR and NDVI for a few weeks, as if burnt, and the
seasonal baseline does not catch it because the melt comes weeks earlier
or later from one year to the next. A pixel is *snowy* in a composite when
its VI Quality has the snow / ice bit or says cloudy (MODIS often takes
snow for cloud), while SIM2 has at least 1 cm of snow on average over its
cell during the composite (the snow / ice bit alone also flags summer
clouds, on the Channel coast for instance). In the 4 composites (64 days) after a pixel's
last snowy one, a burn candidate of that pixel waits: it can be neither
confirmed nor dismissed, and the values the pixel shows meanwhile stay out
of its recent baselines. The first usable composite after that decides: a
real scar is still there, while the after-melt browning is gone. Such a
burn keeps its start; it is only known later. This removes about 50 km²
of false burns (2016-2025), mostly in the Alps and Pyrenees from February
to May. Its cost: of the BDIFF fires of 5 ha or more with a burn detected
within 6 km and 32 days without this rule (248), 14 lose it, all
late-winter pastoral burns in snowy mountain cells (Lozère, Corsica,
Pyrenees, Alpes-Maritimes) whose grass or heath scar fades within the 64
days.

The seasonal baseline needs a year of data, and the rasters start in
November 2014: burns are detected from February 2016 on. 0.27 is the usual
lower bound of moderate burn severity for the NBR drop (Key & Benson,
2006). Fires much smaller than a pixel (5.4 ha) go unseen, and a clear-cut
or a storm can be detected as a burn: all remove fuel.

### Vegetation columns

For a cell *i* and the composite *t* of the row: *F* = the fuel pixels of
*i*; *U* = the pixels of *F* usable in *t* (with their value of *t*, or of
the previous composite, see [Usable pixels](#usable-pixels)); *P* = the
pixels of *U* whose value comes from the previous composite; pct = class
percentiles. The columns marked (NaN < 10) are NaN when |*U*| < 10.

| Column | Definition |
|---|---|
| `fuel_px` | \|*F*\| |
| `fuel_pure_share` | Area of the 25 m fuel squares (any class) lying in the fuel pixels of *i*, divided by the area of all the 25 m fuel squares lying in MODIS pixels whose centre is in *i* (NaN if that area is 0) |
| `vi_px` | \|*U*\| |
| `vi_valid_share` | \|*U*\| / \|*F*\|; 0 if \|*F*\| = 0 |
| `vi_prev_share` | \|*P*\| / \|*U*\|; 0 if \|*U*\| = 0 |
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


## Sample rows

Twelve rows of `train.parquet`, one column per row: 1-10 drawn at random
(`DataFrame.sample(10, random_state=0)`), all with `fire` = 0 as 99.4% of
the rows; 11-12 drawn at random among the rows with `fire` = 1
(`sample(2, random_state=0)`). Each group is sorted by date.

| Column | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cell_id` | 454 | 271 | 1002 | 1021 | 1085 | 897 | 1068 | 113 | 426 | 192 | 1007 | 298 |
| `date` | 2015-05-29 | 2016-08-18 | 2017-01-04 | 2019-01-19 | 2019-01-26 | 2020-08-17 | 2021-01-02 | 2021-03-03 | 2024-03-28 | 2024-10-16 | 2019-02-16 | 2020-06-26 |
| `x` | 387.5 | 912.5 | 962.5 | 712.5 | 612.5 | 612.5 | 1162.5 | 787.5 | 637.5 | 962.5 | 362.5 | 562.5 |
| `y` | 6687.5 | 6812.5 | 6262.5 | 6237.5 | 6137.5 | 6337.5 | 6187.5 | 6912.5 | 6712.5 | 6862.5 | 6237.5 | 6787.5 |
| `lon` | -1.131 | 5.87 | 6.24 | 3.154 | 1.94 | 1.907 | 8.637 | 4.203 | 2.17 | 6.576 | -1.148 | 1.15 |
| `lat` | 47.21 | 48.38 | 43.41 | 43.24 | 42.33 | 44.13 | 42.64 | 49.31 | 47.51 | 48.81 | 43.16 | 48.17 |
| `road_km` | 21.22 | 27.18 | 41.05 | 35.22 | 7.995 | 0 | NaN | 40.44 | 19.54 | 0 | 0 | 22.18 |
| `motorway_km` | 0 | 27.18 | 41.05 | 35.22 | 0 | 0 | NaN | 23.76 | 19.54 | 0 | 0 | 22.18 |
| `road_traffic` | 636 | 518 | 1167.6 | 2277.5 | 30.72 | 0 | NaN | 844.8 | 472.3 | 0 | 0 | 686.6 |
| `road_dist` | 8.589 | 0.7491 | 1.088 | 0.7945 | 11.25 | 25.31 | NaN | 3.575 | 12.09 | 24.98 | 35.09 | 8.484 |
| `pop_density` | 110.1 | 26.73 | 97.87 | 234 | 5.303 | 18.73 | 7.767 | 191.6 | 24.63 | 25.84 | 17.88 | 33.34 |
| `inhabited_share` | 0.9824 | 0.3664 | 0.8016 | 0.576 | 0.0448 | 0.8864 | 0.0208 | 0.3152 | 0.4656 | 0.3568 | 0.5344 | 0.888 |
| `sparse_share` | 0.6576 | 0.2144 | 0.5168 | 0.3424 | 0.0256 | 0.8256 | 0.008 | 0.1424 | 0.3984 | 0.2096 | 0.4464 | 0.7808 |
| `land_share` | 1 | 1 | 1 | 0.8198 | 0.1647 | 1 | 0.042 | 1 | 1 | 1 | 0.9806 | 1 |
| `lc_urban` | 0.05498 | 0.0287 | 0.0763 | 0.09696 | 0.004464 | 0.00872 | 0.0064 | 0.08552 | 0.0293 | 0.02509 | 0.01051 | 0.01958 |
| `lc_crops` | 0.5877 | 0.2531 | 0.05597 | 0.1565 | 0.01816 | 0.2904 | 0.001696 | 0.7976 | 0.141 | 0.5483 | 0.1559 | 0.8405 |
| `lc_vine_orchard` | 0.08696 | 0.003744 | 0.2194 | 0.3459 | 0 | 0.01664 | 0.001104 | 0.01077 | 0 | 0.001616 | 0.001472 | 0 |
| `lc_pasture` | 0.2023 | 0.3502 | 0.005056 | 0.01826 | 0.01368 | 0.2296 | 0 | 0.000912 | 0.0399 | 0.2141 | 0.1311 | 0.05714 |
| `lc_agri_natural` | 0.02304 | 0.01322 | 0.0269 | 0.01594 | 0.001728 | 0.06442 | 0.000304 | 0.00416 | 0.07227 | 0.008544 | 0.02522 | 0.01016 |
| `lc_forest` | 0.04437 | 0.3253 | 0.5189 | 0.02653 | 0.06845 | 0.3523 | 0 | 0.06944 | 0.6983 | 0.1932 | 0.2749 | 0.07149 |
| `lc_grassland` | 0 | 0 | 0 | 0.004512 | 0.02574 | 0.007328 | 0.01458 | 0 | 0 | 0 | 0.3315 | 0 |
| `lc_shrub` | 0 | 0.02574 | 0.0155 | 0.00104 | 0.01414 | 0.0305 | 0 | 0.03005 | 0.01008 | 0.00544 | 0.03778 | 0.000544 |
| `lc_maquis` | 0 | 0 | 0.07238 | 0.08899 | 0 | 0 | 0.01435 | 0 | 0 | 0 | 0 | 0 |
| `lc_sparse` | 0 | 0 | 0.00792 | 0.005296 | 0.01829 | 0 | 0.003568 | 0 | 0 | 0 | 0.01211 | 0 |
| `lc_water` | 0.000672 | 0 | 0.001664 | 0.05987 | 0 | 0 | 0 | 0.001568 | 0.009136 | 0.00376 | 0 | 0.000624 |
| `forest_broadleaf` | 0.07539 | 0.3394 | 0.2058 | 0.01921 | 0.01117 | 0.3988 | 0.01978 | 0.05081 | 0.4304 | 0.2002 | 0.3452 | 0.09294 |
| `forest_conifer` | 0.004628 | 0.007048 | 0.129 | 0.07339 | 0.07283 | 0.009168 | 0.001416 | 0.02389 | 0.1941 | 0.003376 | 0.001268 | 0.00184 |
| `forest_mixed` | 0.003084 | 0.0274 | 0.3335 | 0.008092 | 0.00326 | 0.008276 | 0.0019 | 0.02235 | 0.09572 | 0.007228 | 0.003288 | 0.002348 |
| `forest_open` | 0.001132 | 0.001764 | 0.06285 | 0.0357 | 0.00474 | 0.02783 | 0.01062 | 0.004944 | 0.0109 | 0.001868 | 0.04879 | 0.000252 |
| `heath` | 0.001396 | 0.000268 | 0.0263 | 0.1055 | 0.02965 | 0.01139 | 0.01136 | 0.008612 | 0.01505 | 0.002 | 0.279 | 0.002852 |
| `sp_pine_maritime` | 0.00212 | 0 | 0.006988 | 0 | 0 | 0 | 0.000116 | 0 | 0.00482 | 0 | 0 | 0.000172 |
| `sp_pine_aleppo` | 0 | 0 | 0.1043 | 0.03996 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `sp_pine_scots` | 0.0004 | 0.00116 | 0 | 0 | 0.001728 | 1.2e-05 | 0 | 0.01198 | 0.1081 | 0.001484 | 0 | 0 |
| `sp_pine_black` | 0.000996 | 0.000676 | 9.6e-05 | 0 | 0.000268 | 0.004504 | 0 | 0.00868 | 0.03854 | 0.00028 | 0.000184 | 0.001176 |
| `sp_pine_mountain` | 0 | 0 | 0 | 0 | 0.06692 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `sp_pine_other` | 0 | 0 | 0.006156 | 0.00634 | 0 | 1.6e-05 | 0.000788 | 0 | 3.6e-05 | 0 | 0 | 0 |
| `sp_pine_mixed` | 0 | 0 | 0.007612 | 0.003332 | 0 | 3.6e-05 | 0.000284 | 0.002832 | 0.0344 | 0 | 0 | 0 |
| `sp_fir_spruce` | 0 | 0.003892 | 0 | 0 | 9.6e-05 | 0.0001 | 0 | 9.6e-05 | 0.000384 | 0.000764 | 0 | 0 |
| `sp_douglas` | 0.001008 | 0.000452 | 0 | 0 | 0 | 0.002536 | 0 | 0 | 0.002632 | 0.000344 | 0.000284 | 0.000444 |
| `sp_larch` | 0 | 0.000312 | 0 | 0 | 0.000104 | 0 | 0 | 0 | 0 | 0.000324 | 0.000272 | 0 |
| `sp_conifer_mixed` | 0.000104 | 0.000556 | 0.003844 | 0.02376 | 0.003712 | 0.001964 | 0.000228 | 0.0003 | 0.005256 | 0.00018 | 0.000528 | 4.8e-05 |
| `sp_oak_evergreen` | 0 | 0 | 0.05046 | 0.000492 | 0 | 8e-06 | 3.6e-05 | 0 | 0 | 0.000484 | 0 | 0 |
| `sp_oak_deciduous` | 0.009032 | 0.05864 | 0.03774 | 0 | 0.000204 | 0.2639 | 0 | 0.000116 | 0.02232 | 0.05995 | 0.06496 | 0.03725 |
| `sp_beech` | 0 | 0.05988 | 0 | 0 | 0 | 0 | 0 | 0.000204 | 0 | 0.008464 | 0.1199 | 0 |
| `sp_chestnut` | 0.002932 | 0 | 0.00158 | 0 | 0 | 0.004912 | 0 | 0 | 0.00134 | 0 | 0.001792 | 0 |
| `sp_robinia` | 6.8e-05 | 0 | 0 | 4e-05 | 0 | 0 | 0 | 4.8e-05 | 0.000156 | 0 | 9.2e-05 | 0 |
| `sp_poplar` | 0.00288 | 0.000348 | 0 | 0 | 0 | 0.001736 | 0 | 0.003056 | 0.00232 | 0.00032 | 4.8e-05 | 0.000988 |
| `sp_broadleaf_mixed` | 0.06048 | 0.2206 | 0.116 | 0.01868 | 0.01096 | 0.1283 | 0.01974 | 0.04739 | 0.4043 | 0.131 | 0.1584 | 0.0547 |
| `sp_mixed` | 0.003084 | 0.0274 | 0.3335 | 0.008092 | 0.00326 | 0.008276 | 0.0019 | 0.02235 | 0.09572 | 0.007228 | 0.003288 | 0.002348 |
| `fuel_broadleaf_decid` | 0.07168 | 0.3376 | 0.03931 | 4.2e-05 | 0.01006 | 0.3703 | 2e-06 | 0.04484 | 0.4203 | 0.1975 | 0.2983 | 0.09132 |
| `fuel_broadleaf_med` | 0 | 0 | 0.08947 | 0.009229 | 0 | 0 | 0.0102 | 0 | 0 | 0 | 0 | 0 |
| `fuel_oak_evergreen` | 0 | 0 | 0.05026 | 0.000481 | 0 | 3e-06 | 3.1e-05 | 0 | 0 | 0.000488 | 0 | 0 |
| `fuel_pine_maritime` | 0.002121 | 0 | 0.006985 | 0 | 0 | 0 | 0.00011 | 0 | 0.004779 | 0 | 0 | 0.000174 |
| `fuel_pine_aleppo` | 0 | 0 | 0.1044 | 0.03974 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `fuel_pine_mountain` | 0.001396 | 0.001785 | 0.01355 | 0.009715 | 0.06909 | 0.004731 | 0.001131 | 0.02348 | 0.1809 | 0.001768 | 0.000194 | 0.001141 |
| `fuel_fir_spruce_douglas` | 0.000971 | 0.004343 | 0 | 0 | 9e-05 | 0.00257 | 0 | 9.5e-05 | 0.003036 | 0.001113 | 0.000279 | 0.00043 |
| `fuel_larch` | 0 | 0.000318 | 0 | 0 | 0.000116 | 0 | 0 | 0 | 0 | 0.000327 | 0.000258 | 0 |
| `fuel_conifer_mixed` | 0.000107 | 0.000454 | 0.000588 | 0.001854 | 0.000221 | 0.001806 | 0.000108 | 0.000184 | 0.003759 | 0.000156 | 3.5e-05 | 4.5e-05 |
| `fuel_mixed` | 0.003101 | 0.02683 | 0.3011 | 0.003723 | 0.003064 | 0.007362 | 0.000793 | 0.0207 | 0.09414 | 0.006983 | 0.00202 | 0.002362 |
| `fuel_open_forest` | 0.001156 | 0.00181 | 0.06198 | 0.03573 | 0.004637 | 0.02766 | 0.01077 | 0.004874 | 0.01093 | 0.00177 | 0.04872 | 0.000219 |
| `fuel_heath_med` | 0 | 0 | 0.0267 | 0.1048 | 0 | 0 | 0.01135 | 0 | 0 | 0 | 0 | 0 |
| `fuel_heath_other` | 0.001348 | 0.000276 | 0 | 0 | 0.02955 | 0.01128 | 9e-06 | 0.008618 | 0.0151 | 0.002011 | 0.279 | 0.002897 |
| `fuel_grass` | 0.000106 | 0.000205 | 0.00043 | 0.002477 | 0.017 | 0.002471 | 0 | 0.000275 | 0.001646 | 0.000463 | 0.03876 | 0 |
| `fuel_px` | 248 | 3030 | 2971 | 825 | 1087 | 2401 | 134 | 479 | 4024 | 1765 | 4109 | 463 |
| `fuel_pure_share` | 0.2487 | 0.6978 | 0.356 | 0.3306 | 0.6932 | 0.4656 | 0.3247 | 0.3809 | 0.4559 | 0.7067 | 0.5188 | 0.3788 |
| `holiday_pop_share` | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0.4571 | 0 | 0 | 0.7151 | 0 |
| `holiday_zone_a` | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| `holiday_zone_b` | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| `holiday_zone_c` | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `summer_holiday` | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `short_holiday` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| `public_holiday` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `bridge_day` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `long_weekend` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| `t_mean` | 14.84 | 17.38 | 6.611 | 5 | 2 | 19.26 | 9.5 | 8.85 | 8.867 | 13.97 | 10.49 | 22.72 |
| `t_min` | 11.96 | 13.86 | 0.2778 | 0.8875 | -0.375 | 13.53 | 6.4 | 2.917 | 5.467 | 11.78 | 0.3889 | 19.48 |
| `t_max` | 18.49 | 20.63 | 11.09 | 9.262 | 6.45 | 23.82 | 12.6 | 14.68 | 11.17 | 16.44 | 16.57 | 28.83 |
| `rain` | 0.4444 | 2.117 | 0 | 0.45 | 0.05 | 0.03333 | 4.6 | 0.375 | 3.733 | 0.7556 | 0 | 0 |
| `snow` | 0 | 0 | 0 | 0 | 0.025 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `wind` | 3.133 | 2.158 | 4.456 | 1.438 | 1.5 | 0.8 | 3.7 | 1.092 | 7.3 | 3.267 | 3.178 | 3.608 |
| `rh` | 71.83 | 87.21 | 51.62 | 78.45 | 54.5 | 72.23 | 79.2 | 75.7 | 77.17 | 93.82 | 48.47 | 72.79 |
| `q` | 7.432 | 11.12 | 3.018 | 4.2 | 2.84 | 10.09 | 5.803 | 5.337 | 5.481 | 9.512 | 3.738 | 12.47 |
| `sw_rad` | 2489.9 | 1073.5 | 854.1 | 714.2 | 1016.2 | 1493.5 | 305.6 | 807.6 | 1357.6 | 313.4 | 1394.6 | 1904.6 |
| `lw_rad` | 2748.0 | 3216.4 | 2243.5 | 2509.9 | 2140.4 | 3037.3 | 2655.7 | 2685.8 | 2517.4 | 3129.2 | 2402.5 | 3241.2 |
| `etp` | 4.144 | 2.108 | 1.733 | 0.65 | 0.925 | 2.811 | 1.1 | 1.108 | 2.033 | 0.7222 | 2.522 | 4.908 |
| `evap` | 2.756 | 1.725 | 1.322 | 0.5 | 0.125 | 1.389 | 0.6 | 0.8 | 2.2 | 0.2778 | 2.233 | 1.842 |
| `swi` | 0.3837 | 0.4281 | 0.7857 | 0.6553 | 0.6037 | 0.217 | 1.047 | 0.7517 | 0.9261 | 0.772 | 0.9796 | 0.3078 |
| `wg_root` | 0.2089 | 0.286 | 0.2943 | 0.2819 | 0.2442 | 0.2222 | 0.32 | 0.2962 | 0.2288 | 0.3401 | 0.3164 | 0.2323 |
| `snow_depth` | 0 | 0 | 0 | 0 | 0.07925 | 0 | 0 | 0 | 0 | 0 | 0.002556 | 0 |
| `rain_3d` | 0.01111 | 4.9 | 0.1222 | 0 | 0.95 | 5.056 | 15.6 | 0 | 18.01 | 3.756 | 0.1111 | 0.9333 |
| `rain_7d` | 0.01111 | 4.9 | 0.3222 | 0.1 | 1 | 16.4 | 49.5 | 1.675 | 21.83 | 23.39 | 13.77 | 3.05 |
| `rain_30d` | 96.93 | 44.72 | 27.43 | 0.3625 | 1.25 | 18.22 | 140.2 | 35.22 | 69.44 | 103 | 315.2 | 43.13 |
| `days_since_rain` | 10 | 1 | 15 | 34 | 41 | 2 | 1 | 13 | 1 | 2 | 5 | 5 |
| `t_max_7d` | 19.27 | 25.4 | 11.25 | 11.25 | -0.05 | 29.08 | 11.56 | 12.92 | 14.42 | 14.85 | 12.84 | 24.89 |
| `rh_7d` | 78.4 | 68.61 | 76.15 | 71.32 | 70.6 | 65.47 | 74.23 | 75.2 | 79.67 | 87.99 | 68.27 | 69.33 |
| `swi_prev` | 0.3997 | 0.4178 | 0.7922 | 0.658 | 0.592 | 0.223 | 1.035 | 0.759 | 0.9267 | 0.7747 | 0.9996 | 0.3128 |
| `vi_px` | 248 | 3030 | 2953 | 812 | 283 | 2400 | 115 | 445 | 4024 | 1763 | 4086 | 462 |
| `vi_valid_share` | 1 | 1 | 0.9939 | 0.9842 | 0.2603 | 0.9996 | 0.8582 | 0.929 | 1 | 0.9989 | 0.9944 | 0.9978 |
| `vi_prev_share` | 0 | 0 | 0 | 0.003695 | 0.3958 | 0 | 0.113 | 0 | 0 | 0.688 | 0.9298 | 0 |
| `ndvi_pct` | 0.7964 | 0.7804 | 0.4793 | 0.6655 | 0.5474 | 0.6358 | 0.5203 | 0.1527 | 0.2886 | 0.6239 | 0.3202 | 0.7332 |
| `ndvi_pct_p10` | 0.6298 | 0.5199 | 0.1538 | 0.07821 | 0.03281 | 0.4368 | 0.1131 | 0.01479 | 0.07191 | 0.4773 | 0.09199 | 0.4281 |
| `ndvi_extreme` | 0 | 0.00363 | 0.0535 | 0.1084 | 0.1696 | 0.001667 | 0.09565 | 0.5191 | 0.1692 | 0.003403 | 0.1072 | 0.006494 |
| `nbr_pct` | 0.7366 | 0.7399 | 0.4355 | 0.6527 | 0.6155 | 0.5785 | 0.5237 | 0.1646 | 0.3398 | 0.6851 | 0.3532 | 0.7063 |
| `nbr_pct_p10` | 0.5206 | 0.4701 | 0.09482 | 0.2572 | 0.02166 | 0.3479 | 0.0212 | 0.01424 | 0.128 | 0.467 | 0.006027 | 0.4136 |
| `nbr_extreme` | 0 | 0.0006601 | 0.106 | 0.04803 | 0.1696 | 0.002917 | 0.1739 | 0.4562 | 0.05641 | 0.001134 | 0.2543 | 0.002165 |
| `dry_gap` | -0.05973 | -0.04046 | -0.04381 | -0.01289 | 0.06815 | -0.05736 | 0.003408 | 0.01187 | 0.05118 | 0.06123 | 0.03299 | -0.02697 |
| `burnt_1y` | 0 | 0 | 0.006059 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `burnt_5y` | 0 | 0 | 0.006059 | 0.001212 | 0 | 0.0004165 | 0 | 0 | 0 | 0 | 0.004867 | 0 |
| `vi_age` | 5 | 7 | 2 | 3 | 10 | 6 | 16 | 14 | 8 | 2 | 15 | 2 |
| `fire` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `n_fires` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `area_m2` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 50005 | 190000 |
