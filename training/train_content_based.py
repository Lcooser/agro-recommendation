from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from build_client_vectors import load_clients


def train_content_based() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "models" / "client_similarity.pkl"
    scaler_path = project_root / "models" / "scaler.pkl"

    profiles = load_clients()

    feature_cols = [col for col in profiles.columns if col != "cliente_id"]
    features = profiles[feature_cols].copy()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    similarity_matrix = cosine_similarity(features_scaled)

    payload = {
        "cliente_ids": profiles["cliente_id"].tolist(),
        "similarity": similarity_matrix,
        "feature_columns": feature_cols,
        "profiles": profiles,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    joblib.dump(scaler, scaler_path)

    return output_path


if __name__ == "__main__":
    output = train_content_based()
    print(f"Content-based model treinado: {output}")
