from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recommender.config import get_config

DATASET_PATH = ROOT / "data" / "dataset.csv"
PURCHASES_PATH = ROOT / "data" / "purchases.csv"
EMBEDDINGS_PATH = ROOT / "models" / "vector_embeddings.pkl"
SEED_DIR = ROOT / "data" / "seed"


def _to_pgvector(values, target_dim: int) -> str:
    vals = [float(v) for v in values]
    if len(vals) < target_dim:
        vals = vals + [0.0] * (target_dim - len(vals))
    elif len(vals) > target_dim:
        vals = vals[:target_dim]
    return "[" + ",".join(f"{v:.8f}" for v in vals) + "]"


def export_clients(df: pd.DataFrame) -> None:
    clients = (
        df.groupby("cliente_id", as_index=False)
        .agg(
            hectares=("hectares", "first"),
            cultura=("cultura", "first"),
            regiao=("regiao", "first"),
        )
        .sort_values("cliente_id")
    )
    clients.to_csv(SEED_DIR / "clients.csv", index=False)


def export_client_features(df: pd.DataFrame) -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    features = (
        df.groupby("cliente_id", as_index=False)
        .agg(
            visitas_ultimos_6_meses=("visitas_ultimos_6_meses", "max"),
            num_oportunidades=("num_oportunidades", "max"),
            valor_oportunidades=("valor_oportunidades", "max"),
            maquinas_por_hectare=("maquinas_por_hectare", "max"),
            crescimento_area=("crescimento_area", "max"),
            possui_pulverizador=("possui_pulverizador", "max"),
            possui_plantadeira=("possui_plantadeira", "max"),
            possui_colheitadeira=("possui_colheitadeira", "max"),
            possui_trator=("possui_trator", "max"),
            idade_pulverizador=("idade_pulverizador", "max"),
            idade_plantadeira=("idade_plantadeira", "max"),
            idade_colheitadeira=("idade_colheitadeira", "max"),
        )
        .sort_values("cliente_id")
    )
    features = features[
        [
            "cliente_id",
            "visitas_ultimos_6_meses",
            "num_oportunidades",
            "valor_oportunidades",
            "maquinas_por_hectare",
            "crescimento_area",
            "possui_pulverizador",
            "possui_plantadeira",
            "possui_colheitadeira",
            "possui_trator",
            "idade_pulverizador",
            "idade_plantadeira",
            "idade_colheitadeira",
        ]
    ]
    features.to_csv(SEED_DIR / "client_features.csv", index=False)


def export_embeddings() -> None:
    cfg = get_config()
    target_dim = int(cfg.get("embeddings", {}).get("dimensions", 64))
    artifact = joblib.load(EMBEDDINGS_PATH)

    client_rows = []
    for i, cid in enumerate(artifact["cliente_ids"]):
        emb = artifact["client_embeddings"][i]
        client_rows.append(
            {
                "cliente_id": int(cid),
                "embedding": _to_pgvector(emb, target_dim),
            }
        )
    pd.DataFrame(client_rows).to_csv(SEED_DIR / "client_embeddings.csv", index=False)

    product_rows = []
    for i, produto in enumerate(artifact["produtos"]):
        emb = artifact["product_embeddings"][i]
        product_rows.append(
            {
                "produto": str(produto),
                "embedding": _to_pgvector(emb, target_dim),
            }
        )
    pd.DataFrame(product_rows).to_csv(SEED_DIR / "product_embeddings.csv", index=False)


def main() -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    purchases = pd.read_csv(PURCHASES_PATH)
    purchases.to_csv(SEED_DIR / "purchases.csv", index=False)
    export_clients(df)
    export_client_features(df)
    export_embeddings()
    print(f"Seed exports generated in: {SEED_DIR}")


if __name__ == "__main__":
    main()
