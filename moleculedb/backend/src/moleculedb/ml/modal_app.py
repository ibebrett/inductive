"""Run the ESOL benchmark on Modal, one container per (split, seed), in parallel.

    uv run modal run --detach -m moleculedb.ml.modal_app                       # CPU containers
    uv run modal run --detach -m moleculedb.ml.modal_app --gpu L4 --epochs 300  # GPU for the GNN

--detach keeps the app running on Modal if your laptop disconnects; results are in the
Modal app logs (orchestrate prints the summary) and in W&B.

Images are built from this project's uv.lock: the CPU image uses the `cpu` torch group,
the GPU image the `gpu` (CUDA) group. Runs log to W&B via the Modal secret `wandb`
(WANDB_API_KEY + WANDB_BASE_URL); pass --wandb-project "" to disable.
"""

from pathlib import Path

import modal

BACKEND = Path(__file__).resolve().parents[3]


def _image(torch_group: str) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.13")
        .uv_sync(
            str(BACKEND),
            groups=["ml", torch_group],
            extra_options="--no-default-groups",
        )
        .env({"MOLECULEDB_DATA": "/data"})
        .add_local_python_source("moleculedb")
    )


app = modal.App("moleculedb-esol-benchmark")
secrets = [modal.Secret.from_name("wandb")]


def _run(split: str, seed: int, epochs: int, wandb_project: str):
    from moleculedb.ml import benchmark

    return split, seed, benchmark.run_one(split, seed, epochs, wandb_project or None)


@app.function(image=_image("cpu"), cpu=8, memory=8192, timeout=3600, secrets=secrets)
def run_cpu(split: str, seed: int, epochs: int, wandb_project: str):
    return _run(split, seed, epochs, wandb_project)


@app.function(
    image=_image("gpu"), gpu="L4", cpu=4, memory=16384, timeout=3600, secrets=secrets
)
def run_gpu(split: str, seed: int, epochs: int, wandb_project: str):
    return _run(split, seed, epochs, wandb_project)


@app.function(image=_image("cpu"), timeout=4 * 3600, secrets=secrets)
def orchestrate(seeds: int, epochs: int, gpu: str, wandb_project: str) -> str:
    """Fan out all (split, seed) runs and return the summary table. Runs on Modal, so with
    `modal run --detach` it survives the local machine disconnecting or sleeping."""
    from moleculedb.ml import benchmark

    fn = run_gpu.with_options(gpu=gpu) if gpu else run_cpu
    jobs = [
        (s, seed, epochs, wandb_project)
        for s in benchmark.SPLITS
        for seed in range(seeds)
    ]
    table = benchmark.summarize(list(fn.starmap(jobs)))
    print(table)
    return table


@app.local_entrypoint()
def main(
    seeds: int = 3, epochs: int = 150, gpu: str = "", wandb_project: str = "moleculedb"
):
    print("\n" + orchestrate.remote(seeds, epochs, gpu, wandb_project))
