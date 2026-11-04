"""ML experiments: hand-crafted molecular features vs learned graph representations.

splits    random and scaffold (chemical-series-aware) train/val/test splits
features  Morgan fingerprints and RDKit descriptors (hand-crafted)
gnn       message-passing neural network on the molecular graph (learned)
benchmark `python -m moleculedb.ml.benchmark`: compare them on ESOL
"""
