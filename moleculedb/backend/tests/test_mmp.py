import pytest

from moleculedb import esol

pytestmark = pytest.mark.skipif(
    not esol.MMPDB_PATH.exists(), reason="run `just esol` first"
)


def test_pairs_are_canonical_and_unique() -> None:
    pairs = esol.database().pairs()
    keys = [(p.source, p.target) for p in pairs]
    assert len(keys) == len(set(keys))
    assert all(s < t for s, t in keys)


def test_neighbors_oriented_from_compound() -> None:
    db = esol.database()
    cid = db.pairs(max_size=1)[0].source
    neighbors = db.neighbors(cid, max_size=1)
    assert neighbors and all(p.source == cid for p in neighbors)


def test_graph() -> None:
    g = esol.database().graph(max_size=2)
    assert g.number_of_edges() == len(esol.database().pairs(max_size=2))
    assert "logS" in next(iter(g.nodes.values()))
