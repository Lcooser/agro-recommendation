from pathlib import Path
import math

import pandas as pd

from recommender.clusters import get_all_client_clusters
from recommender.config import get_config
from recommender.hybrid import predict_probability
from recommender.temporal import predict_temporal_12m


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"


def _load_client_profiles() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    profiles = (
        df.groupby("cliente_id", as_index=False)
        .agg(
            hectares=("hectares", "first"),
            cultura=("cultura", "first"),
            possui_pulverizador=("possui_pulverizador", "max"),
            possui_plantadeira=("possui_plantadeira", "max"),
            possui_colheitadeira=("possui_colheitadeira", "max"),
            possui_trator=("possui_trator", "max"),
            regiao=("regiao", "first"),
            idade_pulverizador=("idade_pulverizador", "max"),
            idade_plantadeira=("idade_plantadeira", "max"),
            idade_colheitadeira=("idade_colheitadeira", "max"),
            visitas_ultimos_6_meses=("visitas_ultimos_6_meses", "max"),
            num_oportunidades=("num_oportunidades", "max"),
            valor_oportunidades=("valor_oportunidades", "max"),
            maquinas_por_hectare=("maquinas_por_hectare", "max"),
            crescimento_area=("crescimento_area", "max"),
        )
    )
    return profiles


def sales_opportunity(produto: str, top_n: int = 20) -> list[dict]:
    cfg = get_config()
    opp_cfg = cfg.get("opportunity", {})
    default_cluster_weight = float(opp_cfg.get("default_cluster_weight", 1.0))
    temporal_cfg = cfg.get("temporal", {})
    use_temporal = bool(temporal_cfg.get("use_in_opportunity", True))
    temporal_weight = float(temporal_cfg.get("opportunity_weight", 1.0))
    urgency_cfg = opp_cfg.get("urgency", {})
    urgency_base = float(urgency_cfg.get("base", 1.0))
    urgency_visit_weight = float(urgency_cfg.get("visit_weight", 0.03))
    urgency_growth_weight = float(urgency_cfg.get("growth_weight", 0.6))
    urgency_min = float(urgency_cfg.get("min", 0.8))
    urgency_max = float(urgency_cfg.get("max", 1.8))
    cluster_weights = {
        int(k): float(v) for k, v in opp_cfg.get("cluster_weights", {}).items()
    }

    profiles = _load_client_profiles()
    cluster_map = get_all_client_clusters()

    ranking = []
    for _, row in profiles.iterrows():
        payload = row.to_dict()
        cid = int(payload["cliente_id"])
        payload["produto"] = produto

        prob = predict_probability(payload)
        temporal_prob = predict_temporal_12m(payload) if use_temporal else 1.0
        cluster_id = cluster_map.get(cid)
        cluster_weight = cluster_weights.get(
            cluster_id,
            default_cluster_weight,
        )
        valor = float(payload.get("valor_oportunidades", 0.0))
        visitas = float(payload.get("visitas_ultimos_6_meses", 0.0))
        crescimento = max(0.0, float(payload.get("crescimento_area", 0.0)))
        urgency = urgency_base + urgency_visit_weight * visitas + urgency_growth_weight * crescimento
        urgency = max(urgency_min, min(urgency_max, urgency))

        score = float(
            prob
            * (temporal_prob ** temporal_weight)
            * float(payload["hectares"])
            * math.log1p(max(0.0, valor))
            * cluster_weight
            * urgency
        )
        reasons = []
        if prob >= 0.7:
            reasons.append("alta probabilidade")
        if temporal_prob >= 0.65:
            reasons.append("janela_temporal_favoravel")
        if float(payload["hectares"]) >= 1000:
            reasons.append("grande area")
        if cluster_weight > 1.0:
            reasons.append("cluster com alta conversao")
        if urgency > 1.1:
            reasons.append("urgencia comercial elevada")

        ranking.append(
            {
                "cliente_id": cid,
                "produto": produto,
                "probabilidade_compra": float(prob),
                "probabilidade_temporal_12m": float(temporal_prob),
                "hectares": float(payload["hectares"]),
                "cluster": cluster_id,
                "cluster_weight": float(cluster_weight),
                "urgencia": float(urgency),
                "opportunity_score": score,
                "reasons": reasons,
            }
        )

    ranking.sort(key=lambda item: item["opportunity_score"], reverse=True)
    return ranking[:top_n]
