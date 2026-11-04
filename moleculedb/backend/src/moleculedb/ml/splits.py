"""Train/val/test splits as index arrays.

Random splits put near-identical molecules on both sides, so they flatter models.
Scaffold splits keep each Bemis-Murcko scaffold (ring system + linkers) entirely in one
partition, so the test set contains chemical series the model never saw: a harder,
more realistic estimate.
"""

from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def random_split(n: int, seed: int, frac: tuple[float, float] = (0.8, 0.1)) -> Split:
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def scaffold(smiles: str) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles))


def scaffold_split(
    smiles: list[str], seed: int, frac: tuple[float, float] = (0.8, 0.1)
) -> Split:
    """Balanced scaffold split (as in Chemprop): big scaffold groups always go to train,
    the remaining groups are shuffled by seed and dealt to train/val/test."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[scaffold(s)].append(i)

    n = len(smiles)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    n_test = n - n_train - n_val
    big = [g for g in groups.values() if len(g) > n_val / 2 or len(g) > n_test / 2]
    small = [
        g for g in groups.values() if not (len(g) > n_val / 2 or len(g) > n_test / 2)
    ]
    rng = np.random.default_rng(seed)
    rng.shuffle(small)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for g in sorted(big, key=len, reverse=True) + small:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(val) + len(g) <= n_val:
            val += g
        else:
            test += g
    return np.array(train), np.array(val), np.array(test)
