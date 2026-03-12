from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "client_similarity.pkl"


@lru_cache(maxsize=1)
def _load_similarity():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo de similaridade nao encontrado em {MODEL_PATH}. Rode training/train_content_based.py."
        )
    payload = joblib.load(MODEL_PATH)
    cliente_ids = [int(cid) for cid in payload["cliente_ids"]]
    similarity = payload["similarity"]
    return cliente_ids, similarity


def find_similar(cliente_id: int, top_n: int = 3) -> list[int]:
    scored = find_similar_with_scores(cliente_id, top_n=top_n)
    return [item["cliente_id"] for item in scored]


def find_similar_with_scores(cliente_id: int, top_n: int = 3) -> list[dict]:
    cliente_ids, similarity = _load_similarity()
    cliente_id = int(cliente_id)

    if cliente_id not in cliente_ids:
        return []

    idx = cliente_ids.index(cliente_id)
    sim_scores = similarity[idx]
    similar_indexes = np.argsort(sim_scores)[::-1]
    similar_indexes = [int(i) for i in similar_indexes if i != idx][:top_n]

    return [
        {"cliente_id": cliente_ids[i], "similarity_score": float(sim_scores[i])}
        for i in similar_indexes
    ]
