"""INSEE 1 km population grid -> static population features per cell.

Source: INSEE, "Revenus, pauvreté et niveau de vie en 2019 - Données
carroyées, carreaux de 1 km (y compris données imputées)" (Filosofi 2019,
https://www.insee.fr/fr/statistiques/7655464): the number of inhabitants
(``ind``, fiscal population) of every inhabited 1 km square of the European
INSPIRE grid (ETRS89-LAEA, EPSG:3035), metropolitan France with Corsica.
Uninhabited squares are absent.

Each 1 km square goes to the 25 km cell containing its centre; the squares
of the few cells dropped from the grid (coastline, border) are dropped too.

Per cell:
- ``pop_density``: inhabitants per km² of the cell (625 km², sea included,
  so coastal cells read low);
- ``inhabited_share``: share of the cell's 1 km squares with at least one
  inhabitant;
- ``sparse_share``: share of the cell's 1 km squares with 1 to 50
  inhabitants, i.e. scattered housing, a proxy of the wildland-urban
  interface where most fires start.

    python pipeline/build_population.py

Writes ``data/interim/population_cell.parquet``.
"""

import io
import subprocess
import zipfile

import pandas as pd
import requests
from pyproj import Transformer

from common import CELL_SIZE, CRS_L93, INTERIM, POP_DIR, cell_index

URL = ("https://www.insee.fr/fr/statistiques/fichier/7655464/"
       "Filosofi2019_carreaux_1km_csv.zip")
CSV = POP_DIR / "carreaux_1km_met.csv"
SPARSE_MAX = 50  # inhabitants in a 1 km square
SQUARES_PER_CELL = (CELL_SIZE // 1000) ** 2


def download():
    """The zip holds a .7z; Windows' tar (bsdtar) and libarchive read it."""
    if CSV.exists():
        return
    POP_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL, timeout=300,
                     headers={"User-Agent": "forest-fire-pipeline"})
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(POP_DIR)
    archive = next(POP_DIR.glob("*.7z"))
    subprocess.run(["tar", "-xf", archive.name], cwd=POP_DIR, check=True)


def main():
    download()
    sq = pd.read_csv(CSV, usecols=["idcar_1km", "ind"])
    # "CRS3035RES1000mN2029000E4252000": lower-left corner, metres.
    ne = sq["idcar_1km"].str.extract(r"N(\d+)E(\d+)").astype("float64")
    to_l93 = Transformer.from_crs("EPSG:3035", CRS_L93, always_xy=True)
    x, y = to_l93.transform(ne[1] + 500., ne[0] + 500.)
    sq["ix"], sq["iy"] = cell_index(x, y)

    grid = pd.read_parquet(INTERIM / "grid.parquet")
    sq = sq.merge(grid[["ix", "iy", "cell_id"]], on=["ix", "iy"], how="left")
    out = sq["cell_id"].isna()
    print(f"{len(sq)} inhabited 1 km squares, {sq['ind'].sum() / 1e6:.2f} M "
          f"inhabitants; outside the grid: {out.sum()} squares, "
          f"{sq.loc[out, 'ind'].sum():.0f} inhabitants (dropped)")
    sq = sq[~out].copy()
    sq["sparse"] = sq["ind"] <= SPARSE_MAX

    per_cell = sq.groupby("cell_id").agg(
        pop=("ind", "sum"), n=("ind", "size"), n_sparse=("sparse", "sum"))
    pop = grid[["cell_id"]].join(per_cell, on="cell_id").fillna(0.)
    pop["pop_density"] = pop["pop"] / SQUARES_PER_CELL
    pop["inhabited_share"] = pop["n"] / SQUARES_PER_CELL
    pop["sparse_share"] = pop["n_sparse"] / SQUARES_PER_CELL
    features = ["pop_density", "inhabited_share", "sparse_share"]
    pop = pop[["cell_id", *features]].astype({f: "float32" for f in features})

    INTERIM.mkdir(parents=True, exist_ok=True)
    pop.to_parquet(INTERIM / "population_cell.parquet", index=False)
    print(f"{len(pop)} cells, {(pop['pop_density'] == 0).sum()} uninhabited")
    print(pop[features].describe().round(3))


if __name__ == "__main__":
    main()
