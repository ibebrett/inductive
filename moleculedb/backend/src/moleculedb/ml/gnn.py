"""A small message-passing neural network (GIN-E) on the molecular graph.

    atoms/bonds --embed--> h⁰ --K x message passing--> hᴷ --sum pool--> h_G --MLP--> logS

Atom and bond features are categorical (element, degree, charge, Hs, hybridisation,
aromatic, ring, chirality / bond type, stereo, conjugation), from PyG's `from_smiles`.
Each feature gets a learned embedding; an atom's initial vector is their sum.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_add_pool
    from torch_geometric.utils.smiles import e_map, from_smiles, x_map


@dataclass
class GNNConfig:
    hidden: int = 128
    layers: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 150
    batch_size: int = 32
    seed: int = 0


class MPNN(nn.Module):
    def __init__(self, cfg: GNNConfig):
        super().__init__()
        h = cfg.hidden
        self.atom_emb = nn.ModuleList(nn.Embedding(len(v), h) for v in x_map.values())
        self.bond_emb = nn.ModuleList(nn.Embedding(len(v), h) for v in e_map.values())
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h)), edge_dim=h
            )
            for _ in range(cfg.layers)
        )
        self.norms = nn.ModuleList(nn.BatchNorm1d(h) for _ in range(cfg.layers))
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1)
        )

    def embed(self, data: Data) -> torch.Tensor:
        """Molecule-level representation h_G (one vector per molecule in the batch)."""
        x = sum(emb(data.x[:, i]) for i, emb in enumerate(self.atom_emb))
        e = sum(emb(data.edge_attr[:, i]) for i, emb in enumerate(self.bond_emb))
        for conv, norm in zip(self.convs, self.norms):
            x = x + self.dropout(
                torch.relu(norm(conv(x, data.edge_index, e)))
            )  # residual
        return global_add_pool(x, data.batch)

    def forward(self, data: Data) -> torch.Tensor:
        return self.head(self.embed(data)).squeeze(-1)


def featurize(smiles: list[str], y: np.ndarray) -> list[Data]:
    graphs = []
    for s, target in zip(smiles, y):
        d = from_smiles(s)
        d.y = torch.tensor([target], dtype=torch.float32)
        graphs.append(d)
    return graphs


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict(model: MPNN, loader: DataLoader) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        return torch.cat([model(b.to(dev)) for b in loader]).cpu().numpy()


def train_gnn(
    train: list[Data],
    val: list[Data],
    test: list[Data],
    cfg: GNNConfig,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[MPNN, np.ndarray]:
    """Train on `train`, keep the epoch with best val RMSE, return (model, test predictions).

    Uses CUDA if available. `on_epoch(epoch, {"train_loss", "val_rmse", "lr"})` is called
    after each epoch (e.g. to log to W&B).
    """
    torch.manual_seed(cfg.seed)
    dev = device()
    y_train = torch.cat([d.y for d in train])
    mean, std = y_train.mean().item(), y_train.std().item()

    model = MPNN(cfg).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val, batch_size=256)
    y_val = torch.cat([d.y for d in val]).numpy()

    best_rmse, best_state = float("inf"), None
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(dev)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(batch), (batch.y - mean) / std)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        lr = sched.get_last_lr()[0]
        sched.step()
        val_pred = _predict(model, val_loader) * std + mean
        rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if on_epoch:
            on_epoch(
                epoch,
                {"train_loss": float(np.mean(losses)), "val_rmse": rmse, "lr": lr},
            )

    model.load_state_dict(best_state)
    test_pred = _predict(model, DataLoader(test, batch_size=256)) * std + mean
    return model, test_pred
