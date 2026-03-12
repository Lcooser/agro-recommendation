from recommender.client_similarity import find_similar_with_scores
from recommender.collaborative import get_all_products, get_products_for_clients
from recommender.clusters import get_cluster_top_products_for_client
from recommender.config import get_config


def _rule_candidates(cliente_features: dict) -> set[str]:
    cultura = str(cliente_features.get("cultura", "")).lower()
    hectares = float(cliente_features.get("hectares", 0))

    candidates: set[str] = set()
    if cultura == "soja":
        candidates.update({"plantadeira", "pulverizador", "distribuidor_adubo"})
    if cultura == "milho":
        candidates.update({"colheitadeira", "plantadeira", "pulverizador"})
    if cultura == "trigo":
        candidates.update({"plantadeira", "colheitadeira"})
    if hectares >= 800:
        candidates.update({"colheitadeira", "piloto_automatico", "drone"})
    elif hectares <= 120:
        candidates.update({"pulverizador", "monitoramento"})

    return candidates


def generate_candidates(
    cliente_features: dict,
    cliente_id: int,
    top_n_similar_clients: int | None = None,
    fallback_all_products_limit: int | None = None,
) -> dict:
    config = get_config()
    candidate_cfg = config.get("candidates", {})
    if top_n_similar_clients is None:
        top_n_similar_clients = int(candidate_cfg.get("top_n_similar_clients", 8))
    if fallback_all_products_limit is None:
        fallback_all_products_limit = int(candidate_cfg.get("fallback_all_products_limit", 100))

    similar_rank = find_similar_with_scores(cliente_id, top_n=top_n_similar_clients)
    similar_ids = [item["cliente_id"] for item in similar_rank]

    collaborative = set(get_products_for_clients(similar_ids, exclude_cliente_id=cliente_id))
    from_similarity = set(collaborative)
    from_rules = _rule_candidates(cliente_features)
    try:
        from_cluster = set(get_cluster_top_products_for_client(cliente_id))
    except FileNotFoundError:
        from_cluster = set()

    all_candidates = collaborative | from_similarity | from_rules | from_cluster
    if not all_candidates:
        all_products = get_all_products()
        all_candidates = set(all_products[:fallback_all_products_limit])

    return {
        "candidates": all_candidates,
        "similar_rank": similar_rank,
        "sources": {
            "collaborative": collaborative,
            "similarity": from_similarity,
            "rules": from_rules,
            "cluster": from_cluster,
        },
    }
