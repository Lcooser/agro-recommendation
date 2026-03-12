from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender.config import get_config


def train_embeddings() -> Path:
    cfg = get_config()
    emb_cfg = cfg.get("embeddings", {})
    dims = int(emb_cfg.get("dimensions", 16))
    neighbors = int(emb_cfg.get("nearest_neighbors", 20))

    purchases_path = PROJECT_ROOT / "data" / "purchases.csv"
    output_path = PROJECT_ROOT / "models" / "vector_embeddings.pkl"

    df = pd.read_csv(purchases_path)
    matrix = pd.crosstab(df["cliente_id"], df["produto"])
    matrix = matrix.sort_index()

    svd_dims = max(2, min(dims, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=svd_dims, random_state=42)
    client_embeddings = svd.fit_transform(matrix.values)

    product_embeddings = svd.components_.T

    # Pad to configured dimensions to keep a fixed vector contract (e.g., vector(64)).
    if svd_dims < dims:
        client_pad = np.zeros((client_embeddings.shape[0], dims - svd_dims))
        product_pad = np.zeros((product_embeddings.shape[0], dims - svd_dims))
        client_embeddings = np.hstack([client_embeddings, client_pad])
        product_embeddings = np.hstack([product_embeddings, product_pad])

    client_embeddings = normalize(client_embeddings)
    product_embeddings = normalize(product_embeddings)

    nn = NearestNeighbors(
        n_neighbors=min(neighbors, len(matrix.index)),
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(client_embeddings)

    artifact = {
        "cliente_ids": matrix.index.astype(int).tolist(),
        "produtos": matrix.columns.tolist(),
        "client_embeddings": client_embeddings,
        "product_embeddings": product_embeddings,
        "nn": nn,
        "svd_explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        "dimensions": int(dims),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return output_path


if __name__ == "__main__":
    out = train_embeddings()
    print(f"Embeddings treinados: {out}")
