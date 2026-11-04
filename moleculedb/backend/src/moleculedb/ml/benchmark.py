"""Compare hand-crafted features vs a learned graph model on ESOL logS.

    uv run python -m moleculedb.ml.benchmark [--seeds 3] [--epochs 150] [--wandb PROJECT]
    uv run modal run -m moleculedb.ml.modal_app [--seeds 3] [--epochs 150] [--gpu L4]

Each run (split type x seed) trains:
  mean        predict the training-set mean (sanity floor)
  rf-morgan   random forest on Morgan count fingerprints
  rf-desc     random forest on RDKit descriptors
  gnn         message-passing neural network (moleculedb.ml.gnn)
and the summary reports test RMSE / R² (mean ± std over seeds).
"""

import argparse
import time
from dataclasses import asdict
from functools import cache

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from moleculedb import esol
from moleculedb.ml import features, gnn, splits

SPLITS = ("random", "scaffold")
Metrics = dict[str, tuple[float, float]]  # model -> (RMSE, R²)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


@cache
def _dataset():
    recs = esol.records()
    smiles = [r.smiles for r in recs]
    y = np.array([r.logS for r in recs])
    X = {
        "rf-morgan": features.morgan_counts(smiles),
        "rf-desc": features.rdkit_descriptors(smiles),
    }
    return smiles, y, X, gnn.featurize(smiles, y)


def run_one(
    split: str, seed: int, epochs: int = 150, wandb_project: str | None = None
) -> Metrics:
    """Train every model on one split; return test metrics.

    With `wandb_project`, logs GNN training curves and all test metrics to one W&B run.
    """
    smiles, y, X, graphs = _dataset()
    cfg = gnn.GNNConfig(epochs=epochs, seed=seed)
    run = None
    if wandb_project:
        import wandb

        run = wandb.init(
            project=wandb_project,
            group=f"esol-{split}",
            name=f"{split}-seed{seed}",
            config={"split": split, "device": str(gnn.device()), **asdict(cfg)},
            reinit="create_new",
        )
    if split == "random":
        tr, va, te = splits.random_split(len(y), seed)
    else:
        tr, va, te = splits.scaffold_split(smiles, seed)

    preds = {"mean": np.full(len(te), y[tr].mean())}
    for name, x in X.items():
        rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=seed)
        preds[name] = rf.fit(x[np.r_[tr, va]], y[np.r_[tr, va]]).predict(x[te])
    t0 = time.time()
    _, preds["gnn"] = gnn.train_gnn(
        [graphs[i] for i in tr],
        [graphs[i] for i in va],
        [graphs[i] for i in te],
        cfg,
        on_epoch=(lambda e, m: run.log({f"gnn/{k}": v for k, v in m.items()}, step=e))
        if run
        else None,
    )
    metrics = {k: (rmse(y[te], p), r2(y[te], p)) for k, p in preds.items()}
    if run:
        for k, (r, q) in metrics.items():
            run.summary[f"{k}/test_rmse"] = r
            run.summary[f"{k}/test_r2"] = q
        run.summary["gnn/seconds"] = time.time() - t0
        run.finish()
    print(
        f"{split:8s} seed {seed}: "
        + "  ".join(f"{k} {m[0]:.3f}" for k, m in metrics.items())
        + f"   (gnn {time.time() - t0:.0f}s)",
        flush=True,
    )
    return metrics


def summarize(runs: list[tuple[str, int, Metrics]]) -> str:
    """Table of mean ± std test RMSE / R² per (split, model)."""
    by_key: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for split, _, metrics in runs:
        for model, m in metrics.items():
            by_key.setdefault((split, model), []).append(m)
    n_seeds = len({seed for _, seed, _ in runs})
    lines = [
        f"Test RMSE / R² (logS units), mean ± std over {n_seeds} seeds",
        "",
        f"{'split':10s}{'model':12s}{'RMSE':>16s}{'R²':>16s}",
    ]
    for (split, model), vals in sorted(
        by_key.items(), key=lambda kv: SPLITS.index(kv[0][0])
    ):
        r, q = np.array(vals).T
        lines.append(
            f"{split:10s}{model:12s}{r.mean():>10.3f} ± {r.std():.3f}{q.mean():>10.3f} ± {q.std():.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()
    runs = [
        (s, seed, run_one(s, seed, args.epochs))
        for s in SPLITS
        for seed in range(args.seeds)
    ]
    print("\n" + summarize(runs))


if __name__ == "__main__":
    main()
