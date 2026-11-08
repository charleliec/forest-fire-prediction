"""Météo-France SIM2 (SAFRAN-ISBA, daily, 8 km) -> weather per (cell, day).

Each 25 km cell gets the mean of the SIM2 points it contains. Source:
"Données changement climatique - SIM quotidienne", Météo-France, Licence
Ouverte (https://www.data.gouv.fr/fr/datasets/
donnees-changement-climatique-sim-quotidienne/).

Writes ``data/interim/weather_cell_day.parquet``.
"""

import pandas as pd

from common import INTERIM, LAST_YEAR, WARMUP_YEAR, sim2_file

# SIM2 column -> feature name (units from the SIM2 documentation).
VARIABLES = {
    "T": "t_mean",            # mean 2 m temperature (°C)
    "TINF_H": "t_min",        # daily minimum temperature (°C)
    "TSUP_H": "t_max",        # daily maximum temperature (°C)
    "PRELIQ": "rain",         # liquid precipitation (mm)
    "PRENEI": "snow",         # solid precipitation (mm)
    "FF": "wind",             # 10 m wind speed (m/s)
    "HU": "rh",               # relative humidity (%)
    "Q": "q",                 # specific humidity (g/kg)
    "SSI": "sw_rad",          # shortwave radiation (J/cm2)
    "DLI": "lw_rad",          # longwave radiation (J/cm2)
    "ETP": "etp",             # potential evapotranspiration, Penman (mm)
    "EVAP": "evap",           # actual evapotranspiration (mm)
    "SWI": "swi",             # soil wetness index
    "WG_RACINE": "wg_root",   # root-zone soil water content
    "HTEURNEIGE": "snow_depth",  # snow depth (m)
}


def main():
    pts = pd.read_parquet(INTERIM / "sim2_points.parquet")
    parts = []
    for year in range(WARMUP_YEAR, LAST_YEAR + 1):
        df = pd.read_csv(sim2_file(year), sep=";",
                         usecols=["LAMBX", "LAMBY", "DATE", *VARIABLES])
        df = df.merge(pts, on=["LAMBX", "LAMBY"], how="inner")
        day = (df.drop(columns=["LAMBX", "LAMBY"])
               .groupby(["cell_id", "DATE"]).mean().reset_index())
        day["date"] = pd.to_datetime(day.pop("DATE").astype(str),
                                     format="%Y%m%d")
        parts.append(day.rename(columns=VARIABLES).astype(
            {v: "float32" for v in VARIABLES.values()}))
        print(f"{year}: {len(df)} SIM2 rows -> {len(day)} (cell, day)")
    weather = pd.concat(parts, ignore_index=True)
    weather = weather[["cell_id", "date", *VARIABLES.values()]]
    weather.to_parquet(INTERIM / "weather_cell_day.parquet", index=False)
    n_days = weather["date"].nunique()
    print(f"weather: {len(weather)} rows, {weather['cell_id'].nunique()} "
          f"cells x {n_days} days")


if __name__ == "__main__":
    main()
