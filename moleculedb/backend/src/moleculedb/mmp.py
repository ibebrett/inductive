"""The only interface to mmpdb (matched molecular pairs): building and querying databases.

Nothing else in the codebase should call the mmpdb CLI, import mmpdblib, or read the
SQLite file directly; add what you need here instead.

    db = build_database({"c1": "c1ccccc1O", ...}, {"c1": {"logS": -0.1}}, "out.mmpdb")
    db = MMPDatabase("data/esol/esol.mmpdb")
    db.property_names()            # ["logS"]
    db.compounds()                 # {public_id: Compound}
    db.pairs(max_size=2)           # [Pair], one per compound pair (smallest transform)
    db.neighbors("esol_0042")      # [Pair] touching one compound, oriented away from it
    db.graph(max_size=2)           # networkx.Graph of compounds joined by pairs

"size" is the number of heavy atoms in the larger side of a transform
(e.g. [*:1][H] >> [*:1]Cl has size 1), a good proxy for how small a change is.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import networkx as nx


def build_database(
    smiles: dict[str, str],
    properties: dict[str, dict[str, float]],
    output: str | Path,
) -> MMPDatabase:
    """Fragment + index compounds into an mmpdb database at `output` (overwritten).

    smiles:     compound id -> SMILES (ids must not contain whitespace)
    properties: compound id -> {property name: value}; every compound needs the same names
    """
    names = sorted({name for props in properties.values() for name in props})
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        smi, props, fragdb = (
            tmpdir / "in.smi",
            tmpdir / "props.tsv",
            tmpdir / "in.fragdb",
        )
        smi.write_text("".join(f"{s} {cid}\n" for cid, s in smiles.items()))
        props.write_text(
            "\t".join(["ID", *names])
            + "\n"
            + "".join(
                "\t".join([cid, *(str(properties[cid].get(n, "*")) for n in names)])
                + "\n"
                for cid in smiles
                if cid in properties
            )
        )

        def mmpdb(*args: str | Path) -> None:
            subprocess.run(
                [sys.executable, "-m", "mmpdblib", *map(str, args)], check=True
            )

        mmpdb("fragment", smi, "-o", fragdb)
        mmpdb("index", fragdb, "--properties", props, "-o", output)
    return MMPDatabase(output)


@dataclass(frozen=True)
class Compound:
    id: str  # mmpdb public_id
    smiles: str
    properties: dict[str, float]


@dataclass(frozen=True)
class Pair:
    """Two compounds that differ by a single fragment swap: source + from >> to = target."""

    source: str
    target: str
    from_smiles: str  # variable fragment in source, e.g. "[*:1]Cl"
    to_smiles: str  # variable fragment in target
    size: int
    rule_count: int  # pairs in the dataset sharing this transform (radius 0)

    def reversed(self) -> Pair:
        return replace(
            self,
            source=self.target,
            target=self.source,
            from_smiles=self.to_smiles,
            to_smiles=self.from_smiles,
        )


# Every pair has a radius-0 rule environment, so this yields each stored pair exactly once.
_PAIRS_SQL = """
SELECT c1.public_id, c2.public_id, fs.smiles, ts.smiles,
       max(fs.num_heavies, ts.num_heavies) AS size, re.num_pairs
FROM pair p
JOIN rule_environment re ON re.id = p.rule_environment_id AND re.radius = 0
JOIN rule r ON r.id = re.rule_id
JOIN rule_smiles fs ON fs.id = r.from_smiles_id
JOIN rule_smiles ts ON ts.id = r.to_smiles_id
JOIN compound c1 ON c1.id = p.compound1_id
JOIN compound c2 ON c2.id = p.compound2_id
"""


class MMPDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)

    def property_names(self) -> list[str]:
        with self._connect() as conn:
            return [name for (name,) in conn.execute("SELECT name FROM property_name")]

    @cached_property
    def _compounds(self) -> dict[str, Compound]:
        with self._connect() as conn:
            props: dict[int, dict[str, float]] = {}
            for cid, name, value in conn.execute(
                "SELECT cp.compound_id, pn.name, cp.value FROM compound_property cp "
                "JOIN property_name pn ON pn.id = cp.property_name_id"
            ):
                props.setdefault(cid, {})[name] = value
            return {
                public_id: Compound(public_id, smiles, props.get(cid, {}))
                for cid, public_id, smiles in conn.execute(
                    "SELECT id, public_id, clean_smiles FROM compound"
                )
            }

    def compounds(self) -> dict[str, Compound]:
        return self._compounds

    @cached_property
    def _pairs(self) -> list[Pair]:
        """All compound pairs, canonical direction (source < target), smallest transform each."""
        best: dict[tuple[str, str], Pair] = {}
        with self._connect() as conn:
            for row in conn.execute(_PAIRS_SQL):
                pair = Pair(*row)
                if pair.source > pair.target:
                    pair = pair.reversed()
                key = (pair.source, pair.target)
                if key not in best or pair.size < best[key].size:
                    best[key] = pair
        return sorted(best.values(), key=lambda p: (p.size, p.source, p.target))

    def pairs(self, max_size: int | None = None) -> list[Pair]:
        if max_size is None:
            return list(self._pairs)
        return [p for p in self._pairs if p.size <= max_size]

    def neighbors(self, compound_id: str, max_size: int | None = None) -> list[Pair]:
        """Pairs touching compound_id, oriented so compound_id is the source."""
        return [
            p if p.source == compound_id else p.reversed()
            for p in self.pairs(max_size)
            if compound_id in (p.source, p.target)
        ]

    def graph(self, max_size: int | None = None) -> nx.Graph:
        """Compounds as nodes (attrs: smiles + properties), pairs as edges (attrs: Pair fields).

        Edge attrs are stored in the canonical source < target direction.
        """
        g = nx.Graph()
        for c in self.compounds().values():
            g.add_node(c.id, smiles=c.smiles, **c.properties)
        for p in self.pairs(max_size):
            g.add_edge(
                p.source,
                p.target,
                from_smiles=p.from_smiles,
                to_smiles=p.to_smiles,
                size=p.size,
                rule_count=p.rule_count,
            )
        return g
