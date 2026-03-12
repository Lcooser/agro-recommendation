from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "collaborative_matrix.pkl"


@lru_cache(maxsize=1)
def _load_collaborative():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo colaborativo nao encontrado em {MODEL_PATH}. Rode training/train_collaborative.py."
        )

    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, tuple):
        matrix, similarity = payload
    else:
        matrix = payload["matrix"]
        similarity = payload["similarity"]

    matrix.index = matrix.index.astype(int)
    return matrix, similarity


def get_all_products() -> list[str]:
    matrix, _ = _load_collaborative()
    return matrix.columns.tolist()


def get_products_for_clients(cliente_ids: list[int], exclude_cliente_id: int | None = None) -> list[str]:
    matrix, _ = _load_collaborative()

    products: list[str] = []
    seen: set[str] = set()

    owned: set[str] = set()
    if exclude_cliente_id is not None and exclude_cliente_id in matrix.index:
        owned = set(matrix.columns[matrix.loc[exclude_cliente_id] == 1])

    for cid in cliente_ids:
        if cid not in matrix.index:
            continue
        row = matrix.loc[cid]
        for product, value in row.items():
            if value != 1:
                continue
            if product in owned or product in seen:
                continue
            seen.add(product)
            products.append(product)

    return products


def recommend(cliente_id: int, top_n_clients: int = 3) -> list[str]:
    scores = recommend_with_scores(cliente_id, top_n_clients=top_n_clients)
    return [item["produto"] for item in scores]


def recommend_with_scores(cliente_id: int, top_n_clients: int = 3) -> list[dict]:
    matrix, similarity = _load_collaborative()
    cliente_id = int(cliente_id)

    if cliente_id not in matrix.index:
        return []

    idx = matrix.index.get_loc(cliente_id)
    sim_scores = similarity[idx]

    similar_clients = np.argsort(sim_scores)[::-1]
    similar_clients = [int(i) for i in similar_clients if i != idx][:top_n_clients]

    own_products = set(matrix.columns[matrix.loc[cliente_id] == 1])
    weighted_products: dict[str, float] = {}

    for client_idx in similar_clients:
        row = matrix.iloc[client_idx]
        weight = float(sim_scores[client_idx])
        for product, value in row.items():
            if value != 1 or product in own_products:
                continue
            weighted_products[product] = weighted_products.get(product, 0.0) + weight

    if not weighted_products:
        return []

    max_score = max(weighted_products.values())
    ranked = sorted(weighted_products.items(), key=lambda item: item[1], reverse=True)
    return [
        {"produto": product, "collaborative_score": float(score / max_score)}
        for product, score in ranked
    ]


def score_products_from_client_weights(
    client_weights: dict[int, float],
    exclude_cliente_id: int | None = None,
) -> dict[str, float]:
    matrix, _ = _load_collaborative()
    weighted_products: dict[str, float] = {}

    owned: set[str] = set()
    if exclude_cliente_id is not None and exclude_cliente_id in matrix.index:
        owned = set(matrix.columns[matrix.loc[exclude_cliente_id] == 1])

    for cid, weight in client_weights.items():
        if cid not in matrix.index:
            continue
        row = matrix.loc[cid]
        for product, value in row.items():
            if value != 1 or product in owned:
                continue
            weighted_products[product] = weighted_products.get(product, 0.0) + float(weight)

    if not weighted_products:
        return {}

    max_score = max(weighted_products.values())
    return {product: float(score / max_score) for product, score in weighted_products.items()}


def score_products_by_frequency(
    similar_cliente_ids: list[int],
    exclude_cliente_id: int | None = None,
) -> dict[str, float]:
    matrix, _ = _load_collaborative()
    if not similar_cliente_ids:
        return {}

    owned: set[str] = set()
    if exclude_cliente_id is not None and exclude_cliente_id in matrix.index:
        owned = set(matrix.columns[matrix.loc[exclude_cliente_id] == 1])

    counts: dict[str, int] = {}
    valid_clients = [cid for cid in similar_cliente_ids if cid in matrix.index]
    total = len(valid_clients)
    if total == 0:
        return {}

    for cid in valid_clients:
        row = matrix.loc[cid]
        for product, value in row.items():
            if value != 1 or product in owned:
                continue
            counts[product] = counts.get(product, 0) + 1

    return {product: float(count / total) for product, count in counts.items()}


def products_bought_by_clients(
    similar_cliente_ids: list[int],
    exclude_cliente_id: int | None = None,
) -> dict[str, set[int]]:
    matrix, _ = _load_collaborative()
    buyers: dict[str, set[int]] = {}

    owned: set[str] = set()
    if exclude_cliente_id is not None and exclude_cliente_id in matrix.index:
        owned = set(matrix.columns[matrix.loc[exclude_cliente_id] == 1])

    for cid in similar_cliente_ids:
        if cid not in matrix.index:
            continue
        row = matrix.loc[cid]
        for product, value in row.items():
            if value != 1 or product in owned:
                continue
            buyers.setdefault(product, set()).add(int(cid))

    return buyers
