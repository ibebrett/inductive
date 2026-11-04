from fastapi import FastAPI, HTTPException, Query

from moleculedb import esol

app = FastAPI(title="moleculedb")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/esol/graph")
def esol_graph(max_size: int = Query(2, ge=1, le=10)) -> dict:
    """Compounds plus matched-pair edges whose transform has <= max_size heavy atoms."""
    if not esol.MMPDB_PATH.exists():
        raise HTTPException(503, "ESOL mmpdb database not built; run `just esol`")
    return esol.graph_json(max_size)
