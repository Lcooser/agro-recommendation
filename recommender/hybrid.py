from functools import lru_cache
import math
from pathlib import Path

import joblib
import pandas as pd

from recommender.collaborative import (
    products_bought_by_clients,
    score_products_by_frequency,
)
from recommender.candidates import generate_candidates
from recommender.config import get_config


SUPERVISED_PATH = Path(__file__).resolve().parents[1] / "models" / "supervised_model.pkl"
_cfg = get_config()
UPGRADE_MIN_AGE_YEARS = int(_cfg.get("rules", {}).get("upgrade_min_age_years", 6))
PRODUCT_OWNERSHIP_MAP = {
    "pulverizador": ("possui_pulverizador", "idade_pulverizador"),
    "plantadeira": ("possui_plantadeira", "idade_plantadeira"),
    "colheitadeira": ("possui_colheitadeira", "idade_colheitadeira"),
}
weights_cfg = _cfg.get("weights", {})
WEIGHTS = {
    "ml": float(weights_cfg.get("ml", 0.65)),
    "similarity": float(weights_cfg.get("similarity", 0.20)),
    "collaborative": float(weights_cfg.get("collaborative", 0.15)),
}
COLLAB_SCORE_CAP = float(_cfg.get("rules", {}).get("collab_score_cap", 0.8))


@lru_cache(maxsize=1)
def _load_supervised():
    if not SUPERVISED_PATH.exists():
        raise FileNotFoundError(
            f"Modelo supervisionado nao encontrado em {SUPERVISED_PATH}. Rode training/train_supervised.py."
        )
    artifact = joblib.load(SUPERVISED_PATH)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        expected_features = artifact.get("expected_features", model.get_booster().feature_names)
        scaler = artifact.get("scaler")
        numeric_cols = artifact.get("numeric_cols", [])
        return model, expected_features, scaler, numeric_cols

    # Backward compatibility with older single-object model file.
    model = artifact
    expected_features = model.get_booster().feature_names
    return model, expected_features, None, []


def _score_product(cliente_features: dict, produto: str) -> float:
    model, expected_features, scaler, numeric_cols = _load_supervised()

    features = cliente_features.copy()
    features["produto"] = produto

    row = pd.DataFrame([features])
    row = pd.get_dummies(row, columns=["cultura", "regiao", "produto"])
    row = row.reindex(columns=expected_features, fill_value=0)
    if scaler is not None and numeric_cols:
        cols = [col for col in numeric_cols if col in row.columns]
        if cols:
            scaled = scaler.transform(row[cols].astype("float64"))
            for idx, col in enumerate(cols):
                row[col] = scaled[:, idx]

    prob = model.predict_proba(row)[0][1]
    return float(prob)


def _owned_product_names(cliente_features: dict) -> set[str]:
    owned = set()
    for product, (ownership_field, age_field) in PRODUCT_OWNERSHIP_MAP.items():
        has_product = int(cliente_features.get(ownership_field, 0)) == 1
        age = float(cliente_features.get(age_field, 0))
        if has_product and age <= UPGRADE_MIN_AGE_YEARS:
            owned.add(product)
    return owned


def _should_block_probability(cliente_features: dict, produto: str) -> bool:
    fields = PRODUCT_OWNERSHIP_MAP.get(str(produto).lower())
    if not fields:
        return False
    ownership_field, age_field = fields
    has_product = int(cliente_features.get(ownership_field, 0)) == 1
    age = float(cliente_features.get(age_field, 0))
    return has_product and age <= UPGRADE_MIN_AGE_YEARS


def _business_constraint_bonus(cliente_features: dict, produto: str) -> tuple[float, str | None]:
    cultura = str(cliente_features.get("cultura", "")).lower()
    if cultura == "soja" and produto == "plantadeira":
        return 0.08, "cultura soja favorece plantadeira"
    if cultura == "milho" and produto == "colheitadeira":
        return 0.08, "cultura milho favorece colheitadeira"
    return 0.0, None


def _penalize_similarity(
    raw_similarity: float,
    collaborative_score: float,
    n_buyers: int,
    n_similar_clients: int,
) -> float:
    if n_buyers <= 0 or n_similar_clients <= 0:
        return 0.0
    support = math.log1p(n_buyers) / math.log1p(n_similar_clients)
    return float(raw_similarity * collaborative_score * support)


def hybrid_recommend(cliente_features: dict, cliente_id: int, top_n: int = 3) -> list[dict]:
    candidate_payload = generate_candidates(
        cliente_features=cliente_features,
        cliente_id=cliente_id,
    )
    candidates = candidate_payload["candidates"]
    similar_rank = candidate_payload["similar_rank"]
    similar_ids = [item["cliente_id"] for item in similar_rank]
    collab_scores = score_products_by_frequency(
        similar_ids,
        exclude_cliente_id=cliente_id,
    )
    buyers_by_product = products_bought_by_clients(
        similar_ids,
        exclude_cliente_id=cliente_id,
    )
    similarity_by_client = {
        item["cliente_id"]: float(item["similarity_score"]) for item in similar_rank
    }
    similarity_scores = {}
    for product, buyers in buyers_by_product.items():
        vals = [similarity_by_client[cid] for cid in buyers if cid in similarity_by_client]
        if vals:
            similarity_scores[product] = float(sum(vals) / len(vals))

    owned_products = _owned_product_names(cliente_features)

    scores: list[dict] = []
    for product in candidates:
        if product in owned_products:
            continue

        ml_score = _score_product(cliente_features, product)
        collaborative_score = min(float(collab_scores.get(product, 0.0)), COLLAB_SCORE_CAP)
        raw_similarity = float(similarity_scores.get(product, 0.0))
        n_buyers = len(buyers_by_product.get(product, set()))
        similarity_score = _penalize_similarity(
            raw_similarity=raw_similarity,
            collaborative_score=collaborative_score,
            n_buyers=n_buyers,
            n_similar_clients=max(1, len(similar_ids)),
        )
        bonus, bonus_reason = _business_constraint_bonus(cliente_features, product)
        final_score = min(
            1.0,
            WEIGHTS["ml"] * ml_score
            + WEIGHTS["similarity"] * similarity_score
            + WEIGHTS["collaborative"] * collaborative_score
            + bonus,
        )
        reasons = []
        if similarity_score > 0:
            reasons.append("clientes semelhantes compraram")
        if collaborative_score > 0:
            reasons.append("historico colaborativo favoravel")
        if ml_score >= 0.5:
            reasons.append("alta probabilidade no modelo supervisionado")
        if cliente_features.get("possui_plantadeira", 0) == 0 and product == "plantadeira":
            reasons.append("cliente nao possui plantadeira")
        if bonus_reason:
            reasons.append(bonus_reason)

        scores.append(
            {
                "produto": product,
                "score_final": float(final_score),
                "ml_score": float(ml_score),
                "similarity_score": similarity_score,
                "collaborative_score": collaborative_score,
                "reasons": reasons,
                "candidate_sources": [
                    key
                    for key, products in candidate_payload["sources"].items()
                    if product in products
                ],
            }
        )

    scores.sort(key=lambda item: item["score_final"], reverse=True)
    return scores[:top_n]


def predict_probability(cliente_features: dict) -> float:
    produto = cliente_features.get("produto")
    if not produto:
        raise ValueError("Campo 'produto' e obrigatorio para previsao supervisionada.")
    if _should_block_probability(cliente_features, produto):
        return 0.0
    return _score_product(cliente_features, produto)
