"""School holidays and public holidays -> calendar features per day.

Sources (both under the Licence Ouverte / Etalab):
- Ministère de l'Éducation nationale, "Calendrier scolaire"
  (https://data.education.gouv.fr/explore/dataset/fr-en-calendrier-scolaire):
  start and end of every holiday of every académie. It covers zones A / B / C
  only from the 2017-2018 school year (Corsica since 2009), and records the
  holidays as they happened (e.g. the single national spring holiday of
  April 2021, moved for COVID-19) and the school Ascension bridges.
- A. Augusti, "vacances-scolaires"
  (https://github.com/AntoineAugusti/vacances-scolaires): one row per day,
  whether zone A / B / C is on holiday, compiled from the ministry's
  calendars. Used before September 2017 only.

Both give the days off: from the first Saturday (or the day after the last
class) to the Sunday before school starts again.

A zone is a set of académies, and the zones were redrawn in September 2015
(the 2014-2015 map is ``OLD_ZONES``). Each académie is a set of
départements (``ACADEMIES``). The population of an académie sums the
inhabited 1 km squares of the INSEE grid (see build_population.py) of its
départements, read from the communes each square covers.

Per day, the same for every cell:
- ``holiday_pop_share``: share of the inhabitants of metropolitan France
  whose académie is on school holiday (how many holiday-makers there are,
  wherever they go);
- ``holiday_zone_a`` / ``_b`` / ``_c``: zone A / B / C on school holiday;
- ``summer_holiday``: summer school holidays (any of zones A / B / C);
- ``short_holiday``: any other school holiday (Toussaint, Christmas,
  winter, spring, Ascension bridge) in at least one of zones A / B / C;
- ``public_holiday``: national public holiday (jour férié; the two extra
  days of Alsace-Moselle are ignored);
- ``bridge_day``: a working day between two days off (e.g. the Friday after
  Ascension Thursday), which many take off ("faire le pont");
- ``long_weekend``: the day is in a run of 3 days off or more (weekends,
  public holidays and bridge days).

All are known in advance: no leak.

    python pipeline/build_calendar.py

Writes ``data/interim/calendar_day.parquet``.
"""

import numpy as np
import pandas as pd
import requests
from dateutil.easter import easter

from common import CALENDAR_DIR, FIRST_YEAR, INTERIM, LAST_YEAR
from build_population import CSV as POP_CSV
from build_population import download as download_population

OFFICIAL_URL = ("https://data.education.gouv.fr/api/explore/v2.1/catalog/"
                "datasets/fr-en-calendrier-scolaire/exports/csv")
COMMUNITY_URL = ("https://raw.githubusercontent.com/AntoineAugusti/"
                 "vacances-scolaires/master/data.csv")
OFFICIAL_CSV = CALENDAR_DIR / "fr-en-calendrier-scolaire.csv"
COMMUNITY_CSV = CALENDAR_DIR / "vacances-scolaires.csv"
# First day taken from the official calendar (school year 2017-2018).
OFFICIAL_FROM = pd.Timestamp("2017-09-01")
# Zones were redrawn for the school year 2015-2016.
NEW_ZONES_FROM = pd.Timestamp("2015-09-01")

ACADEMIES = {
    "Aix-Marseille": ["04", "05", "13", "84"],
    "Amiens": ["02", "60", "80"],
    "Besançon": ["25", "39", "70", "90"],
    "Bordeaux": ["24", "33", "40", "47", "64"],
    "Caen": ["14", "50", "61"],
    "Clermont-Ferrand": ["03", "15", "43", "63"],
    "Corse": ["2A", "2B"],
    "Créteil": ["77", "93", "94"],
    "Dijon": ["21", "58", "71", "89"],
    "Grenoble": ["07", "26", "38", "73", "74"],
    "Lille": ["59", "62"],
    "Limoges": ["19", "23", "87"],
    "Lyon": ["01", "42", "69"],
    "Montpellier": ["11", "30", "34", "48", "66"],
    "Nancy-Metz": ["54", "55", "57", "88"],
    "Nantes": ["44", "49", "53", "72", "85"],
    "Nice": ["06", "83"],
    "Orléans-Tours": ["18", "28", "36", "37", "41", "45"],
    "Paris": ["75"],
    "Poitiers": ["16", "17", "79", "86"],
    "Reims": ["08", "10", "51", "52"],
    "Rennes": ["22", "29", "35", "56"],
    "Rouen": ["27", "76"],
    "Strasbourg": ["67", "68"],
    "Toulouse": ["09", "12", "31", "32", "46", "65", "81", "82"],
    "Versailles": ["78", "91", "92", "95"],
}
# Since September 2015. Corsica has its own calendar, in no zone.
NEW_ZONES = {
    "A": ["Besançon", "Bordeaux", "Clermont-Ferrand", "Dijon", "Grenoble",
          "Limoges", "Lyon", "Poitiers"],
    "B": ["Aix-Marseille", "Amiens", "Caen", "Lille", "Nancy-Metz", "Nantes",
          "Nice", "Orléans-Tours", "Reims", "Rennes", "Rouen", "Strasbourg"],
    "C": ["Créteil", "Montpellier", "Paris", "Toulouse", "Versailles"],
}
OLD_ZONES = {
    "A": ["Caen", "Clermont-Ferrand", "Grenoble", "Lyon", "Montpellier",
          "Nancy-Metz", "Nantes", "Rennes", "Toulouse"],
    "B": ["Aix-Marseille", "Amiens", "Besançon", "Dijon", "Lille",
          "Limoges", "Nice", "Orléans-Tours", "Poitiers", "Reims", "Rouen",
          "Strasbourg"],
    "C": ["Bordeaux", "Créteil", "Paris", "Versailles"],
}
# The académies of Caen and Rouen merged in 2020.
MERGED = {"Normandie": ["Caen", "Rouen"]}


def download():
    CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
    for url, path in [(OFFICIAL_URL, OFFICIAL_CSV),
                      (COMMUNITY_URL, COMMUNITY_CSV)]:
        if path.exists():
            continue
        r = requests.get(url, timeout=300, params={"delimiter": ";"}
                         if url == OFFICIAL_URL else None,
                         headers={"User-Agent": "forest-fire-pipeline"})
        r.raise_for_status()
        path.write_bytes(r.content)


def official(days):
    """(day, académie) days off and summer flags from the ministry."""
    cal = pd.read_csv(OFFICIAL_CSV, sep=";", encoding="utf-8-sig")
    # Pupils' holidays ("Élèves", or "Élèves du premier / second degré"
    # when they differ by a day), not teachers'.
    pupils = cal["population"].eq("-") | cal["population"].str.startswith(
        "Élèves")
    known = cal["location"].isin(ACADEMIES) | cal["location"].isin(MERGED)
    cal = cal[pupils & known]
    off = pd.DataFrame(False, index=days, columns=list(ACADEMIES))
    summer = off.copy()
    for row in cal.itertuples():
        start, end = (pd.to_datetime(d, utc=True).tz_convert("Europe/Paris")
                      .tz_localize(None).normalize()
                      for d in (row.start_date, row.end_date))
        # End is the day school starts again; a one-day break has end ==
        # start.
        span = (days >= start) & (days < max(end, start + pd.Timedelta(
            days=1)))
        for academie in MERGED.get(row.location, [row.location]):
            off.loc[span, academie] = True
            if "été" in row.description.lower():
                summer.loc[span, academie] = True
    return off, summer


def community(days):
    """(day, académie) days off and summer flags from the zone A/B/C file."""
    cal = pd.read_csv(COMMUNITY_CSV, parse_dates=["date"]).set_index("date")
    cal = cal.reindex(days)
    if cal["vacances_zone_a"].isna().any():
        raise SystemExit("the zone calendar misses some days")
    is_summer = cal["nom_vacances"].str.lower().str.contains("été",
                                                             na=False)
    off = pd.DataFrame(False, index=days, columns=list(ACADEMIES))
    for zones, period in [(OLD_ZONES, days < NEW_ZONES_FROM),
                          (NEW_ZONES, days >= NEW_ZONES_FROM)]:
        for zone, academies in zones.items():
            flag = cal.loc[period, f"vacances_zone_{zone.lower()}"]
            for academie in academies:
                off.loc[period, academie] = flag.astype(bool).to_numpy()
    summer = off.apply(lambda col: col & is_summer.to_numpy())
    return off, summer


def school_days_off(days):
    old = days < OFFICIAL_FROM
    off_o, summer_o = official(days)
    off_c, summer_c = community(days)
    # Corsica (no zone) comes from the official calendar on every day.
    off_c["Corse"], summer_c["Corse"] = off_o["Corse"], summer_o["Corse"]
    off, summer = off_o.copy(), summer_o.copy()
    off.loc[old], summer.loc[old] = off_c.loc[old], summer_c.loc[old]
    per_year = off.groupby(off.index.year).sum()
    if (per_year < 100).any().any():
        raise SystemExit(f"académies with few holidays:\n{per_year}")
    return off, summer


def zone_of(days):
    """(day, académie) zone letter, '' for Corsica."""
    out = pd.DataFrame("", index=days, columns=list(ACADEMIES))
    for zones, period in [(OLD_ZONES, days < NEW_ZONES_FROM),
                          (NEW_ZONES, days >= NEW_ZONES_FROM)]:
        for zone, academies in zones.items():
            out.loc[period, academies] = zone
    return out


def academy_population():
    """Inhabitants of each académie (INSEE 1 km grid, 2019)."""
    download_population()
    sq = pd.read_csv(POP_CSV, usecols=["lcog_geo", "ind"],
                     dtype={"lcog_geo": str})
    # A square on a border lists several 5-character commune codes: split
    # its inhabitants evenly between them.
    sq["commune"] = sq["lcog_geo"].str.findall(r".{5}")
    sq["ind"] /= sq["commune"].str.len()
    sq = sq.explode("commune")
    dep_to_acad = {d: a for a, deps in ACADEMIES.items() for d in deps}
    sq["academie"] = sq["commune"].str[:2].map(dep_to_acad)
    unknown = sq.loc[sq["academie"].isna(), "commune"].unique()
    if len(unknown):
        raise SystemExit(f"unknown départements: {unknown}")
    return sq.groupby("academie")["ind"].sum().reindex(list(ACADEMIES))


def public_holidays(years):
    out = []
    for year in years:
        e = pd.Timestamp(easter(year))
        out += [pd.Timestamp(f"{year}-{md}") for md in
                ("01-01", "05-01", "05-08", "07-14", "08-15", "11-01",
                 "11-11", "12-25")]
        # Easter Monday, Ascension Thursday, Whit Monday.
        out += [e + pd.Timedelta(days=d) for d in (1, 39, 50)]
    return pd.DatetimeIndex(out)


def day_off_features(days):
    # A week of margin each side for the bridges and runs at the edges.
    ext = pd.date_range(days[0] - pd.Timedelta(days=7),
                        days[-1] + pd.Timedelta(days=7))
    public = ext.isin(public_holidays(range(ext[0].year, ext[-1].year + 1)))
    weekend = ext.day_of_week >= 5
    off = public | weekend
    bridge = ~off & np.r_[False, off[:-1]] & np.r_[off[1:], False]
    off |= bridge
    run_id = np.cumsum(np.r_[True, off[1:] != off[:-1]])
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts())
    long_weekend = off & (run_len.to_numpy() >= 3)
    keep = ext.isin(days)
    return pd.DataFrame({"public_holiday": public[keep],
                         "bridge_day": bridge[keep],
                         "long_weekend": long_weekend[keep]},
                        index=days).astype("int8")


def main():
    download()
    days = pd.date_range(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR}-12-31")
    off, summer = school_days_off(days)
    zone = zone_of(days)
    pop = academy_population()

    daily = day_off_features(days)
    for z in "ABC":
        in_zone = zone == z
        # A zone is on holiday when most of its académies are (they differ
        # by a day at most, around the Ascension bridge).
        daily[f"holiday_zone_{z.lower()}"] = (
            (off & in_zone).sum(axis=1) * 2 > in_zone.sum(axis=1)).astype(
                "int8")
    abc = zone != ""
    daily["summer_holiday"] = (summer & abc).any(axis=1).astype("int8")
    daily["short_holiday"] = (off & ~summer & abc).any(axis=1).astype("int8")
    daily["holiday_pop_share"] = (off.astype(float) @ (pop / pop.sum())
                                  ).astype("float32")
    cols = ["holiday_pop_share", "holiday_zone_a", "holiday_zone_b",
            "holiday_zone_c", "summer_holiday", "short_holiday",
            "public_holiday", "bridge_day", "long_weekend"]
    cal = daily[cols].rename_axis("date").reset_index()

    INTERIM.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(INTERIM / "calendar_day.parquet", index=False)
    print(f"{len(cal)} days; days per year with each flag:")
    print(cal.groupby(cal["date"].dt.year)[cols].sum().round(1).to_string())


if __name__ == "__main__":
    main()
