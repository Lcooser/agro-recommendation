from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from recommender.client_similarity import find_similar_with_scores
from recommender.clusters import get_client_cluster
from recommender.candidates import generate_candidates
from recommender.hybrid import hybrid_recommend, predict_probability
from recommender.opportunity import sales_opportunity
from recommender.temporal import predict_temporal
from recommender.vector_search import similar_clients_vector, top_products_for_client_vector

app = FastAPI()

class ClienteBaseInput(BaseModel):
    cliente_id: int
    hectares: float
    cultura: str
    possui_pulverizador: int
    possui_plantadeira: int
    possui_colheitadeira: int
    possui_trator: int
    regiao: str
    idade_pulverizador: int = 0
    idade_plantadeira: int = 0
    idade_colheitadeira: int = 0
    visitas_ultimos_6_meses: int = 0
    num_oportunidades: int = 0
    valor_oportunidades: float = 0.0
    maquinas_por_hectare: float = 0.0
    crescimento_area: float = 0.0


class ClienteProdutoInput(ClienteBaseInput):
    produto: str


class ClienteProdutoTemporalInput(ClienteProdutoInput):
    mes_atual: int = 3
    fase_safra: str = ""


@app.post("/recommend")
def recommend(cliente: ClienteBaseInput, debug: bool = Query(default=False)):
    payload = cliente.model_dump()
    try:
        ranking = hybrid_recommend(payload, cliente_id=cliente.cliente_id, top_n=3)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = {"cliente_id": cliente.cliente_id, "recomendacoes": ranking}
    if debug:
        cands = generate_candidates(payload, cliente.cliente_id)
        response["candidate_debug"] = {
            "total_candidates": len(cands["candidates"]),
            "by_source": {
                "collaborative": len(cands["sources"]["collaborative"]),
                "similarity": len(cands["sources"]["similarity"]),
                "rules": len(cands["sources"]["rules"]),
                "cluster": len(cands["sources"]["cluster"]),
            },
        }
    return response


@app.post("/recommend/probability")
def recommend_probability(cliente: ClienteProdutoInput):
    try:
        prob = predict_probability(cliente.model_dump())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"probabilidade_compra": prob}


@app.post("/temporal/purchase")
def temporal_purchase(cliente: ClienteProdutoTemporalInput):
    try:
        out = predict_temporal(cliente.model_dump())
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return out


@app.get("/similar/{cliente_id}")
def similar_clients(cliente_id: int, top_n: int = Query(default=3, ge=1, le=20)):
    try:
        similar = find_similar_with_scores(cliente_id, top_n=top_n)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not similar:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {cliente_id} nao encontrado na base de similaridade.",
        )

    return {"cliente_id": cliente_id, "similar_clientes": similar}


@app.get("/cluster/{cliente_id}")
def producer_cluster(cliente_id: int):
    try:
        cluster_info = get_client_cluster(cliente_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not cluster_info:
        raise HTTPException(status_code=404, detail=f"Cliente {cliente_id} nao encontrado.")

    return cluster_info


@app.get("/sales/opportunity")
def sales_opportunity_endpoint(
    produto: str = Query(..., description="Produto alvo para priorizacao comercial"),
    top_n: int = Query(default=20, ge=1, le=500),
):
    try:
        ranking = sales_opportunity(produto=produto, top_n=top_n)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"produto": produto, "top_n": top_n, "ranking": ranking}


@app.get("/sales/opportunities")
def sales_opportunities_endpoint(
    produto: str = Query(..., description="Produto alvo para priorizacao comercial"),
    top_n: int = Query(default=20, ge=1, le=500),
):
    return sales_opportunity_endpoint(produto=produto, top_n=top_n)


@app.get("/similar_vector/{cliente_id}")
def similar_vector(cliente_id: int, top_n: int = Query(default=5, ge=1, le=100)):
    try:
        similar = similar_clients_vector(cliente_id=cliente_id, top_n=top_n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not similar:
        raise HTTPException(status_code=404, detail=f"Cliente {cliente_id} nao encontrado.")
    return {"cliente_id": cliente_id, "similar_clientes_vector": similar}


@app.get("/similar_vector/{cliente_id}/products")
def similar_vector_products(cliente_id: int, top_n: int = Query(default=5, ge=1, le=100)):
    try:
        products = top_products_for_client_vector(cliente_id=cliente_id, top_n=top_n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not products:
        raise HTTPException(status_code=404, detail=f"Cliente {cliente_id} nao encontrado.")
    return {"cliente_id": cliente_id, "produtos_similares_vector": products}
