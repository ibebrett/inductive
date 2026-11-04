# moleculedb

Exploration project: a Python backend (data science + FastAPI) and a TypeScript/React frontend.

```
moleculedb/
├── backend/    Python, managed by uv (FastAPI, RDKit, mmpdb)
├── frontend/   TypeScript + React, built with Vite (RDKit.js)
├── data/       Datasets: downloaded/generated locally, not committed
└── notebooks/  Jupyter notebooks for exploration
```

## Backend layout

- `moleculedb.mmp`: **the only interface to mmpdb.** Builds databases (`build_database`) and queries them (`MMPDatabase`: compounds, pairs, neighbors, networkx graph). Nothing else calls the mmpdb CLI, imports `mmpdblib`, or reads the SQLite file; add what you need here.
- `moleculedb.esol`: ESOL dataset (CSV paths, compound names, `build()`), on top of `mmp`.
- `moleculedb.ml`: ML experiments: splits, hand-crafted features (Morgan, descriptors), a message-passing GNN, and `python -m moleculedb.ml.benchmark` (`just bench`).
- `moleculedb.api`: FastAPI routes, e.g. `GET /api/esol/graph?max_size=N`.

## Dependencies

### Toolchain

| Tool | Purpose | Install |
| --- | --- | --- |
| [uv](https://docs.astral.sh/uv/) | Python version + package management for `backend/` | `dnf install uv` or see uv docs |
| [fnm](https://github.com/Schniz/fnm) | Node version manager; reads `frontend/.node-version` | release binary into `~/.local/bin`, then add `eval "$(fnm env --use-on-cd --shell bash)"` to `~/.bashrc` |
| Node 26 (LTS) | JavaScript runtime | `fnm install --lts` |
| [pnpm](https://pnpm.io/) | Frontend package manager | `npm install -g pnpm` (Node 26 no longer ships corepack) |
| [just](https://github.com/casey/just) | Task runner (`justfile`) | release binary into `~/.local/bin` |
| [process-compose](https://github.com/F1bonacc1/process-compose) | Runs backend + frontend dev servers together | release binary into `~/.local/bin` |

### Backend (`backend/pyproject.toml`)

- Python 3.13 (pinned in the repo-root `.python-version`; 3.15 broke PyTorch Geometric). Install with `uv python install 3.13` if missing.
- [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/): API server
- [RDKit](https://www.rdkit.org/): cheminformatics
- [mmpdb](https://github.com/rdkit/mmpdb): matched molecular pair analysis (CLI: `uv run mmpdb ...`)
- [networkx](https://networkx.org/): matched-pair graph
- ML group (`ml`): CPU-only [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pyg.org/), scikit-learn
- Dev: pytest, httpx, ruff, jupyterlab, pandas

### Frontend (`frontend/package.json`)

- [React](https://react.dev/) 19
- [Vite](https://vite.dev/) 8: dev server + bundler
- [RDKit.js](https://rdkitjs.com/) (`@rdkit/rdkit`): RDKit compiled to WebAssembly
- [react-force-graph-2d](https://github.com/vasturiano/react-force-graph): interactive graph view
- [d3-scale-chromatic](https://github.com/d3/d3-scale-chromatic): colour scales
- Dev: TypeScript, oxlint

## Datasets

Small public datasets with measured properties, useful for matched molecular pair (MMP) analysis:

| Dataset | Size | Property | Source | Status |
| --- | --- | --- | --- | --- |
| ESOL (Delaney) | 1,128 | aqueous solubility, logS (mol/L) | [MoleculeNet / DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv) | `just esol` |
| FreeSolv | ~640 | hydration free energy | MoleculeNet | not yet |
| Lipophilicity | ~4,200 | logD (pH 7.4), from ChEMBL | MoleculeNet | not yet |
| ChEMBL, single target | 100s to 1000s | potency (IC50/Ki) per target | [ChEMBL](https://www.ebi.ac.uk/chembl/) / `chembl_webresource_client` | not yet |

ESOL: `just esol` (`python -m moleculedb.esol`) downloads the CSV into `data/esol/` and builds `data/esol/esol.mmpdb` (~25 s, ~220 MB). Explore it in `notebooks/esol.ipynb` (`just notebook`).

## Usage

```sh
just install   # uv sync + pnpm install
just dev       # backend on :8000, frontend on http://localhost:5173 (proxies /api -> backend)
just test
just lint
just build
just data       # download datasets into data/
just esol       # build the ESOL mmpdb database (downloads first)
just notebook   # JupyterLab on notebooks/
just bench      # ESOL: Morgan/descriptor random forests vs GNN
```
