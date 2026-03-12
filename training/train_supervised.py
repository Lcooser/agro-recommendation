from pathlib import Path
import json
import sys

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering import load_dataset
from recommender.config import get_config


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    rel = y_true[order]
    total_relevant = rel.sum()
    if total_relevant == 0:
        return 0.0

    precisions = []
    hits = 0
    for i, r in enumerate(rel, start=1):
        if r == 1:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0


def _dcg_at_k(rel: np.ndarray, k: int) -> float:
    rel_k = rel[:k]
    if len(rel_k) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rel_k) + 2))
    return float(np.sum(rel_k / discounts))


def _ranking_metrics_by_client(
    X_test,
    y_test,
    y_prob: np.ndarray,
    k: int,
) -> dict:
    prob_series = y_test.copy().astype(float)
    prob_series.loc[:] = y_prob

    clients = X_test["cliente_id"].astype(int)
    precision_vals = []
    recall_vals = []
    ndcg_vals = []
    ap_vals = []

    for cid in clients.unique():
        idx = clients[clients == cid].index
        y_true = y_test.loc[idx].to_numpy(dtype=int)
        y_score = prob_series.loc[idx].to_numpy(dtype=float)

        if y_true.sum() == 0:
            continue

        order = np.argsort(-y_score)
        rel_sorted = y_true[order]
        topk = rel_sorted[:k]

        precision_vals.append(float(topk.sum() / k))
        recall_vals.append(float(topk.sum() / y_true.sum()))

        dcg = _dcg_at_k(rel_sorted, k)
        ideal = _dcg_at_k(np.sort(y_true)[::-1], k)
        ndcg_vals.append(float(dcg / ideal) if ideal > 0 else 0.0)

        ap_vals.append(_average_precision(y_true, y_score))

    def _safe_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        f"precision@{k}": _safe_mean(precision_vals),
        f"recall@{k}": _safe_mean(recall_vals),
        f"ndcg@{k}": _safe_mean(ndcg_vals),
        "map": _safe_mean(ap_vals),
    }


def train_supervised() -> Path:
    cfg = get_config()
    ranking_k = int(cfg.get("ranking_metrics", {}).get("k", 3))

    df = load_dataset()

    X = df.drop("comprou", axis=1)
    y = df["comprou"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = [
        col
        for col in [
            "hectares",
            "idade_pulverizador",
            "idade_plantadeira",
            "idade_colheitadeira",
            "visitas_ultimos_6_meses",
            "num_oportunidades",
            "valor_oportunidades",
            "maquinas_por_hectare",
            "crescimento_area",
        ]
        if col in X_train.columns
    ]
    scaler = StandardScaler()
    if numeric_cols:
        train_scaled = scaler.fit_transform(X_train[numeric_cols].astype("float64"))
        test_scaled = scaler.transform(X_test[numeric_cols].astype("float64"))
        for idx, col in enumerate(numeric_cols):
            X_train[col] = train_scaled[:, idx]
            X_test[col] = test_scaled[:, idx]

    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    ranking_metrics = _ranking_metrics_by_client(X_test, y_test, y_prob, ranking_k)
    metrics.update(ranking_metrics)

    gain_importance = model.get_booster().get_score(importance_type="gain")
    gain_importance = dict(sorted(gain_importance.items(), key=lambda x: x[1], reverse=True))

    model_path = Path(__file__).resolve().parents[1] / "models" / "supervised_model.pkl"
    metrics_path = Path(__file__).resolve().parents[1] / "models" / "supervised_metrics.json"
    ranking_metrics_path = Path(__file__).resolve().parents[1] / "models" / "ranking_metrics.json"
    importance_path = Path(__file__).resolve().parents[1] / "models" / "feature_importance_gain.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "expected_features": X_train.columns.tolist(),
        "scaler": scaler if numeric_cols else None,
        "numeric_cols": numeric_cols,
    }
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ranking_metrics_path.write_text(json.dumps(ranking_metrics, indent=2), encoding="utf-8")
    importance_path.write_text(json.dumps(gain_importance, indent=2), encoding="utf-8")
    print("Metricas:", metrics)
    print("Top feature importance (gain):", list(gain_importance.items())[:10])

    return model_path


if __name__ == "__main__":
    output = train_supervised()
    print(f"Supervised model treinado: {output}")
