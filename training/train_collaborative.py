from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def train_collaborative() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    purchases_path = project_root / "data" / "purchases.csv"
    output_path = project_root / "models" / "collaborative_matrix.pkl"

    df = pd.read_csv(purchases_path)
    matrix = pd.crosstab(df["cliente_id"], df["produto"])
    similarity = cosine_similarity(matrix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"matrix": matrix, "similarity": similarity}, output_path)

    return output_path


if __name__ == "__main__":
    output = train_collaborative()
    print(f"Collaborative model treinado: {output}")
