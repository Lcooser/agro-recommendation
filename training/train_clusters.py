from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_client_vectors import load_clients
from recommender.config import get_config


def train_clusters(n_clusters: int | None = None) -> Path:
    if n_clusters is None:
        cfg = get_config()
        n_clusters = int(cfg.get("clusters", {}).get("cluster_size", 4))

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "models" / "producer_clusters.pkl"
    dataset_path = project_root / "data" / "dataset.csv"

    profiles = load_clients()
    feature_cols = [col for col in profiles.columns if col != "cliente_id"]
    X = profiles[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_ids = kmeans.fit_predict(X_scaled)
    profiles["cluster"] = cluster_ids

    cluster_stats = (
        profiles.groupby("cluster", as_index=False)
        .agg(
            total_clientes=("cliente_id", "count"),
            media_hectares=("hectares", "mean"),
            pct_tem_pulverizador=("possui_pulverizador", "mean"),
            pct_tem_plantadeira=("possui_plantadeira", "mean"),
            pct_tem_colheitadeira=("possui_colheitadeira", "mean"),
        )
        .to_dict(orient="records")
    )

    df = pd.read_csv(dataset_path)
    purchases = df[df["comprou"] == 1][["cliente_id", "produto"]].copy()
    profile_clusters = profiles[["cliente_id", "cluster"]]
    merged = purchases.merge(profile_clusters, on="cliente_id", how="inner")

    cluster_top_products: dict[int, list[str]] = {}
    if not merged.empty:
        counts = (
            merged.groupby(["cluster", "produto"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        for cluster_id, grp in counts.groupby("cluster"):
            top = grp.sort_values("count", ascending=False).head(8)["produto"].tolist()
            cluster_top_products[int(cluster_id)] = top

    artifact = {
        "cliente_clusters": {
            int(row["cliente_id"]): int(row["cluster"]) for _, row in profiles.iterrows()
        },
        "cluster_stats": cluster_stats,
        "cluster_top_products": cluster_top_products,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "kmeans": kmeans,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return output_path


if __name__ == "__main__":
    out = train_clusters()
    print(f"Clusters treinados: {out}")
