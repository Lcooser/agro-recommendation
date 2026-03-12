from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import psycopg2
from psycopg2 import Error as PsycopgError

from recommender.config import get_config

EMBEDDINGS_PATH = Path(__file__).resolve().parents[1] / "models" / "vector_embeddings.pkl"


def _db_params() -> dict:
    cfg = get_config().get("postgres", {})
    return {
        "host": cfg.get("host", "localhost"),
        "port": int(cfg.get("port", 5432)),
        "dbname": cfg.get("database", "agro"),
        "user": cfg.get("user", "postgres"),
        "password": cfg.get("password", "postgres"),
    }


@contextmanager
def _conn():
    conn = psycopg2.connect(**_db_params())
    try:
        yield conn
    finally:
        conn.close()


@lru_cache(maxsize=1)
def _load_local_embeddings() -> dict:
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Arquivo de embeddings nao encontrado: {EMBEDDINGS_PATH}")
    return joblib.load(EMBEDDINGS_PATH)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _similar_clients_vector_local(cliente_id: int, top_n: int) -> list[dict]:
    data = _load_local_embeddings()
    client_ids = data["cliente_ids"]
    client_embeddings = _normalize_rows(np.asarray(data["client_embeddings"], dtype=float))

    if cliente_id not in client_ids:
        return []

    idx = client_ids.index(cliente_id)
    query = client_embeddings[idx]
    scores = client_embeddings @ query

    order = np.argsort(scores)[::-1]
    out: list[dict] = []
    for i in order:
        cid = int(client_ids[i])
        if cid == int(cliente_id):
            continue
        out.append({"cliente_id": cid, "similarity_score": float(scores[i])})
        if len(out) >= top_n:
            break
    return out


def _top_products_for_client_vector_local(cliente_id: int, top_n: int) -> list[dict]:
    data = _load_local_embeddings()
    client_ids = data["cliente_ids"]
    products = data["produtos"]
    client_embeddings = _normalize_rows(np.asarray(data["client_embeddings"], dtype=float))
    product_embeddings = _normalize_rows(np.asarray(data["product_embeddings"], dtype=float))

    if cliente_id not in client_ids:
        return []

    idx = client_ids.index(cliente_id)
    query = client_embeddings[idx]
    scores = product_embeddings @ query
    order = np.argsort(scores)[::-1][:top_n]

    return [{"produto": str(products[i]), "vector_score": float(scores[i])} for i in order]


def similar_clients_vector(cliente_id: int, top_n: int = 5) -> list[dict]:
    query = """
    SELECT
        ce2.cliente_id,
        (1 - (ce1.embedding <=> ce2.embedding)) AS similarity_score
    FROM client_embeddings ce1
    JOIN client_embeddings ce2 ON ce1.cliente_id <> ce2.cliente_id
    WHERE ce1.cliente_id = %s
    ORDER BY ce2.embedding <-> ce1.embedding
    LIMIT %s
    """

    try:
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(query, (int(cliente_id), int(top_n)))
            rows = cur.fetchall()
        return [{"cliente_id": int(r[0]), "similarity_score": float(r[1])} for r in rows]
    except (UnicodeDecodeError, PsycopgError):
        return _similar_clients_vector_local(cliente_id=cliente_id, top_n=top_n)


def top_products_for_client_vector(cliente_id: int, top_n: int = 5) -> list[dict]:
    query = """
    SELECT
        pe.produto,
        (1 - (pe.embedding <=> ce.embedding)) AS vector_score
    FROM client_embeddings ce
    CROSS JOIN product_embeddings pe
    WHERE ce.cliente_id = %s
    ORDER BY pe.embedding <-> ce.embedding
    LIMIT %s
    """

    try:
        with _conn() as conn, conn.cursor() as cur:
            cur.execute(query, (int(cliente_id), int(top_n)))
            rows = cur.fetchall()
        return [{"produto": str(r[0]), "vector_score": float(r[1])} for r in rows]
    except (UnicodeDecodeError, PsycopgError):
        return _top_products_for_client_vector_local(cliente_id=cliente_id, top_n=top_n)
