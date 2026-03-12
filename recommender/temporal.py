from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from recommender.config import get_config


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "temporal_model.pkl"
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


@lru_cache(maxsize=1)
def _load_temporal():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo temporal nao encontrado em {MODEL_PATH}. Rode training/train_temporal.py."
        )
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact["expected_features"]
    model = artifact
    return model, model.get_booster().feature_names


def _build_temporal_row(payload: dict) -> pd.DataFrame:
    cfg = get_config()
    temporal_cfg = cfg.get("temporal", {})
    month = int(payload.get("mes_atual", temporal_cfg.get("default_month", datetime.now().month)))
    produto = str(payload.get("produto", "")).lower()
    age_col = PRODUCT_AGE_MAP.get(produto)
    idade_maquina = float(payload.get(age_col, 0)) if age_col else 0.0

    row = payload.copy()
    row["idade_maquina"] = idade_maquina
    row["anos_desde_compra"] = idade_maquina
    row["mes_atual"] = month
    row["fase_safra"] = str(payload.get("fase_safra", _phase_from_month(month)))
    return pd.DataFrame([row])


def _window_distribution(prob_12m: float, idade_maquina: float) -> dict:
    age_factor = min(max(idade_maquina / 10.0, 0.0), 1.0)
    w0 = prob_12m * (0.7 + age_factor)
    w1 = prob_12m * (1.1 + age_factor)
    w2 = (1 - prob_12m) * (0.8 - 0.2 * age_factor) + 0.2
    total = max(w0 + w1 + w2, 1e-9)
    return {
        "0-6 meses": float(w0 / total),
        "6-12 meses": float(w1 / total),
        "12-24 meses": float(w2 / total),
    }


def predict_temporal(cliente_features: dict) -> dict:
    model, expected_features = _load_temporal()
    row = _build_temporal_row(cliente_features)
    row = pd.get_dummies(row, columns=[c for c in ["produto", "cultura", "regiao", "fase_safra"] if c in row.columns])
    row = row.reindex(columns=expected_features, fill_value=0)
    prob_12m = float(model.predict_proba(row)[0][1])

    idade_maquina = float(row.get("idade_maquina", pd.Series([0.0])).iloc[0])
    windows = _window_distribution(prob_12m, idade_maquina)
    best_window = max(windows, key=windows.get)

    reasons = []
    hectares = float(cliente_features.get("hectares", 0))
    if idade_maquina >= 8:
        reasons.append("idade_maquina_alta")
    if hectares >= 800:
        reasons.append("area_grande")
    if float(cliente_features.get("visitas_ultimos_6_meses", 0)) >= 5:
        reasons.append("alto_engajamento_comercial")
    if int(cliente_features.get("num_oportunidades", 0)) >= 2:
        reasons.append("pipeline_comercial_aquecido")

    return {
        "probabilidade_12_meses": prob_12m,
        "janela_esperada_compra": best_window,
        "janela_probabilidades": windows,
        "reasons": reasons,
    }


def predict_temporal_12m(cliente_features: dict) -> float:
    return float(predict_temporal(cliente_features)["probabilidade_12_meses"])
