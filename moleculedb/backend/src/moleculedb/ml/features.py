"""Hand-crafted molecular features (fixed, not learned)."""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def morgan_counts(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Morgan/ECFP count fingerprints: hashed counts of circular substructures up to `radius`."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.stack(
        [gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles]
    ).astype(np.float32)


def rdkit_descriptors(smiles: list[str]) -> np.ndarray:
    """~200 RDKit physicochemical descriptors (MolWt, MolLogP, TPSA, ...)."""
    rows = [Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)) for s in smiles]
    x = np.array([list(r.values()) for r in rows], dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -1e6, 1e6).astype(np.float32)
