from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recommender.config import get_config


PRODUCT_AGE_MAP = {
    "pulverizador": "idade_pulverizador",
    "plantadeira": "idade_plantadeira",
    "colheitadeira": "idade_colheitadeira",
}


def _phase_from_month(month: int) -> str:
    if month in (9, 10, 11, 12):
        return "plantio"
    if month in (1, 2, 3, 4):
        return "colheita"
    return "entressafra"


def _build_temporal_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cfg = get_config()
    default_month = int(cfg.get("temporal", {}).get("default_month", 3))

    temporal = df.copy()
    temporal["idade_maquina"] = temporal.apply(
        lambda r: float(r.get(PRODUCT_AGE_MAP.get(str(r["produto"]).lower(), ""), 0)),
        axis=1,
    )
    temporal["anos_desde_compra"] = temporal["idade_maquina"]
    temporal["mes_atual"] = default_month
    temporal["fase_safra"] = temporal["mes_atual"].apply(_phase_from_month)
    temporal["comprou_12m"] = temporal["comprou"].astype(int)
    return temporal


def train_temporal() -> Path:
    dataset_path = PROJECT_ROOT / "data" / "dataset.csv"
    df = pd.read_csv(dataset_path)
    temporal = _build_temporal_dataset(df)

    feature_cols = [
        "cliente_id",
        "produto",
        "hectares",
        "cultura",
        "regiao",
        "idade_maquina",
        "anos_desde_compra",
        "visitas_ultimos_6_meses",
        "num_oportunidades",
        "valor_oportunidades",
        "maquinas_por_hectare",
        "crescimento_area",
        "mes_atual",
        "fase_safra",
        "possui_pulverizador",
        "possui_plantadeira",
        "possui_colheitadeira",
        "possui_trator",
        "idade_pulverizador",
        "idade_plantadeira",
        "idade_colheitadeira",
    ]
    feature_cols = [c for c in feature_cols if c in temporal.columns]

    X = temporal[feature_cols].copy()
    y = temporal["comprou_12m"].astype(int)
    X = pd.get_dummies(X, columns=[c for c in ["produto", "cultura", "regiao", "fase_safra"] if c in X.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(max_depth=6, n_estimators=300, learning_rate=0.05)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    model_path = PROJECT_ROOT / "models" / "temporal_model.pkl"
    metrics_path = PROJECT_ROOT / "models" / "temporal_metrics.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "expected_features": X_train.columns.tolist(),
    }
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Temporal metrics:", metrics)
    return model_path


if __name__ == "__main__":
    out = train_temporal()
    print(f"Temporal model treinado: {out}")
