from functools import lru_cache
from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "producer_clusters.pkl"


@lru_cache(maxsize=1)
def _load_cluster_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo de clusters nao encontrado em {MODEL_PATH}. Rode training/train_clusters.py."
        )
    return joblib.load(MODEL_PATH)


def get_client_cluster(cliente_id: int) -> dict:
    artifact = _load_cluster_model()
    cid = int(cliente_id)
    cluster = artifact["cliente_clusters"].get(cid)
    if cluster is None:
        return {}

    cluster_stats = next(
        (item for item in artifact["cluster_stats"] if int(item["cluster"]) == int(cluster)),
        None,
    )
    return {
        "cliente_id": cid,
        "cluster": int(cluster),
        "cluster_stats": cluster_stats,
        "top_products": artifact.get("cluster_top_products", {}).get(int(cluster), []),
    }


def get_cluster_top_products_for_client(cliente_id: int) -> list[str]:
    artifact = _load_cluster_model()
    cid = int(cliente_id)
    cluster = artifact["cliente_clusters"].get(cid)
    if cluster is None:
        return []
    return artifact.get("cluster_top_products", {}).get(int(cluster), [])


def get_all_client_clusters() -> dict[int, int]:
    artifact = _load_cluster_model()
    return {int(k): int(v) for k, v in artifact.get("cliente_clusters", {}).items()}
