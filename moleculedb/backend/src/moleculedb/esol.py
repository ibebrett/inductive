"""ESOL (Delaney) solubility dataset: raw CSV + its mmpdb database.

Compound IDs (esol_0001, ...) come from CSV row order. Property: measured logS (mol/L).
Build the database with `python -m moleculedb.esol` (or `just esol`).
"""

import csv
import os
import urllib.request
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

from moleculedb.mmp import MMPDatabase, build_database

# Data root: $MOLECULEDB_DATA if set (e.g. in Modal containers), else moleculedb/data in the repo.
DATA_DIR = Path(
    os.environ.get("MOLECULEDB_DATA") or Path(__file__).resolve().parents[3] / "data"
)
ESOL_DIR = DATA_DIR / "esol"
CSV_PATH = ESOL_DIR / "delaney-processed.csv"
MMPDB_PATH = ESOL_DIR / "esol.mmpdb"
CSV_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
)


def download() -> None:
    """Fetch the ESOL CSV (MoleculeNet / DeepChem mirror) if not already present."""
    if not CSV_PATH.exists():
        ESOL_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)


@cache
def _rows() -> dict[str, dict[str, str]]:
    with open(CSV_PATH, newline="") as f:
        return {f"esol_{i:04d}": row for i, row in enumerate(csv.DictReader(f), 1)}


@dataclass(frozen=True)
class Record:
    id: str
    name: str
    smiles: str
    logS: float  # measured log10 solubility, mol/L


def records() -> list[Record]:
    """All 1,128 ESOL compounds (downloads the CSV if needed)."""
    download()
    return [
        Record(
            cid,
            row["Compound ID"],
            row["smiles"].strip(),
            float(row["measured log solubility in mols per litre"]),
        )
        for cid, row in _rows().items()
    ]


def names() -> dict[str, str]:
    """Compound ID -> common name from the CSV (mmpdb only stores IDs)."""
    return {r.id: r.name for r in records()}


def build() -> MMPDatabase:
    recs = records()
    return build_database(
        smiles={r.id: r.smiles for r in recs},
        properties={r.id: {"logS": r.logS} for r in recs},
        output=MMPDB_PATH,
    )


@cache
def database() -> MMPDatabase:
    return MMPDatabase(MMPDB_PATH)


def graph_json(max_size: int) -> dict:
    """Paired compounds + pairs, shaped for a force-graph frontend."""
    db = database()
    pairs = db.pairs(max_size)
    compounds = db.compounds()
    ids = {p.source for p in pairs} | {p.target for p in pairs}
    nodes = [
        {
            "id": c.id,
            "name": names().get(c.id, c.id),
            "smiles": c.smiles,
            "logS": c.properties["logS"],
        }
        for c in compounds.values()
        if c.id in ids
    ]
    links = [
        {
            **asdict(p),
            "delta_logS": round(
                compounds[p.target].properties["logS"]
                - compounds[p.source].properties["logS"],
                3,
            ),
        }
        for p in pairs
    ]
    return {"nodes": nodes, "links": links}


if __name__ == "__main__":
    db = build()
    print(f"Built {db.path} ({len(db.compounds())} paired compounds)")
