
# Agro Recommendation Engine

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Sistema de recomendacao hibrido para agronegocio, com foco em inteligencia comercial para vendas tecnicas.

## 1. Visao Geral do Projeto

O **Agro Recommendation Engine** resolve um problema comum em operacoes comerciais no agro:

- qual produto recomendar para cada produtor
- qual cliente priorizar na agenda do vendedor
- quando a chance de compra e maior

Em vez de depender apenas de "feeling comercial", o sistema combina sinais de dados para gerar recomendacoes com prioridade objetiva.

### Por que recomendacao importa no agro

No agronegocio, a decisao de compra de maquina e implemento envolve:

- capacidade produtiva (hectares, cultura)
- maturidade tecnologica do cliente
- parque de maquinas ja instalado
- ciclo de renovacao (idade de equipamento)
- contexto comercial (visitas e pipeline)

Com recomendacao, a equipe comercial passa a atuar de forma mais eficiente:

- mais foco nos clientes com melhor potencial
- menos tempo em oportunidades frias
- melhor previsibilidade de funil e receita

### Como o sistema prioriza clientes e produtos

O motor combina 3 perguntas:

1. O que recomendar? (ranking por produto)
2. Para quem vender primeiro? (opportunity score)
3. Quando vender? (temporal prediction)

Isso caracteriza um sistema de **Sales Intelligence**: decisao comercial orientada por modelos de ML + regras de negocio.

### Conceitos-chave (explicacao simples)

- **Sistema de recomendacao**: algoritmo que sugere produtos com base em perfil e historico.
- **Ranking**: ordenacao dos produtos do mais promissor para o menos promissor.
- **Vector search**: busca por proximidade entre vetores (embeddings) para achar clientes/produtos semanticamente parecidos.
- **Temporal prediction**: previsao da chance de compra em uma janela de tempo (ex.: 12 meses).

---

## 2. Quick Start

Para rodar ponta a ponta rapidamente:

```bash
git clone https://github.com/Lcooser/agro-recommendation.git
cd agro-recommendation
python -m pip install -r requirements.txt

python training/train_supervised.py
python training/train_collaborative.py
python training/train_content_based.py
python training/train_clusters.py
python training/train_temporal.py
python embeddings/train_embeddings.py

python scripts/export_postgres_seed.py
docker compose up -d
python -m uvicorn api.main:app --reload
```

Swagger:

- `http://localhost:8000/docs`

---
flowchart LR

DATA[(Dataset + Purchases)]
TRAIN[Offline Training]
MODELS[(Models)]
API[FastAPI]
DB[(PostgreSQL + pgvector)]

DATA --> TRAIN
TRAIN --> MODELS
MODELS --> API
DB --> API


## 3. Arquitetura do Sistema

Pipeline principal:

1. Candidate Generation
2. Ranking ML
3. Regras de negocio
4. Opportunity Engine
5. Temporal Prediction
6. Vector Search

### Diagrama textual

```text
Dados (dataset.csv, purchases.csv)
    |
    +--> Treino Offline (supervisionado, collaborative, content, cluster, temporal, embeddings)
    |         |
    |         +--> Artefatos em models/
    |
FastAPI (api/main.py)
    |
    +--> /recommend ------------------> Candidate Generation + Ranking + Regras
    +--> /recommend/probability ------> Score supervisionado por cliente/produto
    +--> /temporal/purchase ----------> Probabilidade de compra em 12 meses
    +--> /sales/opportunity ----------> Priorizacao comercial de clientes
    +--> /similar --------------------> Similaridade content-based
    +--> /cluster --------------------> Segmento KMeans do cliente
    +--> /similar_vector -------------> Busca vetorial (pgvector principal + fallback local)
```

### Diagrama Mermaid

```mermaid
flowchart LR
    A[data/dataset.csv + data/purchases.csv] --> B[Offline Training Pipeline]
    B --> B1[train_supervised.py]
    B --> B2[train_collaborative.py]
    B --> B3[train_content_based.py]
    B --> B4[train_clusters.py]
    B --> B5[train_temporal.py]
    B --> B6[embeddings/train_embeddings.py]
    B --> C[models/*.pkl + *.json]
    C --> D[FastAPI api/main.py]

    D --> E1[/recommend]
    E1 --> CG[Candidate Generation]
    CG --> R1[Collaborative]
    CG --> R2[Client Similarity]
    CG --> R3[Business Rules]
    CG --> R4[Cluster Candidates]
    R1 --> RK[Ranking XGBoost + weighted scoring]
    R2 --> RK
    R3 --> RK
    R4 --> RK
    RK --> BR[Ownership + upgrade rules]
    BR --> OUT1[Top recomendacoes]

    D --> E2[/recommend/probability]
    E2 --> OUT2[Probabilidade supervisionada]

    D --> E3[/temporal/purchase]
    E3 --> OUT3[Probabilidade 12m + janela esperada]

    D --> E4[/sales/opportunity]
    E4 --> OUT4[Ranking de clientes para visita]

    D --> E5[/similar_vector]
    E5 --> PG[(PostgreSQL + pgvector)]
    PG --> OUT5[Clientes/produtos similares]
```

### Papel de cada modulo

- `recommender/candidates.py`: gera candidatos para ranking (collaborative, similarity, rules, cluster).
- `recommender/hybrid.py`: calcula score final com pesos e regras.
- `recommender/collaborative.py`: sinais por frequencia e compras de clientes similares.
- `recommender/client_similarity.py`: similaridade de clientes por content-based.
- `recommender/clusters.py`: consulta cluster e produtos dominantes por cluster.
- `recommender/temporal.py`: previsao temporal por janela.
- `recommender/opportunity.py`: priorizacao comercial por score.
- `recommender/vector_search.py`: tenta pgvector primeiro; se houver erro de conexao/encoding do driver, usa fallback local em `models/vector_embeddings.pkl`.

---

## 4. Fluxo dos dados no sistema

```text
data/dataset.csv
  -> treino supervisionado
  -> treino temporal
  -> treino de clusters
  -> similaridade content-based

data/purchases.csv
  -> collaborative filtering
  -> embeddings cliente-produto

models/*.pkl e models/*.json
  -> consumidos pela API FastAPI na inferencia online

data/seed/*.csv
  -> consumidos pelo PostgreSQL via COPY (db/init/02_seed.sql)

PostgreSQL + pgvector
  -> consumido pelos endpoints vetoriais
```

---

## 5. Decisoes de arquitetura e por que

- **XGBoost no ranking supervisionado**
  - bom desempenho em dados tabulares heterogeneos
  - pre-processamento simples
  - boa relacao entre qualidade, velocidade e interpretabilidade

- **Collaborative Filtering**
  - captura padroes de co-compra entre clientes parecidos
  - complementa regras fixas de negocio

- **Content-Based Similarity**
  - ajuda no cold start de clientes
  - usa perfil agricola e parque de maquinas

- **KMeans para clusters**
  - segmentacao comercial rapida e interpretavel
  - reforca geracao de candidatos e priorizacao

- **Modelo temporal separado**
  - separa "vai comprar?" de "quando vai comprar?"
  - melhora aplicacao comercial em planejamento

- **PostgreSQL + pgvector**
  - persistencia e consulta vetorial via SQL
  - simplifica operacao local e evolucao para cloud

- **Docker no banco**
  - ambiente reprodutivel
  - schema e seed automatizados

---

## 6. Tecnologias Utilizadas

| Tecnologia | Papel no projeto |
|---|---|
| Python | Linguagem principal de treino, inferencia e API |
| FastAPI | Exposicao dos endpoints HTTP |
| XGBoost | Modelos supervisionado e temporal |
| Scikit-learn | Similaridade, clustering, escalonamento, SVD |
| PostgreSQL | Persistencia relacional e consultas SQL |
| pgvector | Vetor nativo no Postgres (`vector(64)`) e busca por distancia |
| Docker / Docker Compose | Subida padronizada do banco em ambiente local |
| psycopg2 | Driver Python para acesso ao PostgreSQL |
| pandas / joblib | Transformacoes de dados e persistencia de artefatos |

---

## 7. Estrutura de Pastas do Projeto

```text
agro-recommendation/
  api/
    main.py
  config/
    config.json
  data/
    dataset.csv
    purchases.csv
    seed/
      clients.csv
      client_features.csv
      purchases.csv
      client_embeddings.csv
      product_embeddings.csv
  db/
    init/
      01_schema.sql
      02_seed.sql
      03_vector_functions.sql
  embeddings/
    train_embeddings.py
  models/
    supervised_model.pkl
    supervised_metrics.json
    ranking_metrics.json
    feature_importance_gain.json
    collaborative_matrix.pkl
    client_similarity.pkl
    scaler.pkl
    producer_clusters.pkl
    temporal_model.pkl
    temporal_metrics.json
    vector_embeddings.pkl
  postman/
    agro-recommendation-engine.postman_collection.json
  recommender/
    candidates.py
    hybrid.py
    collaborative.py
    client_similarity.py
    clusters.py
    temporal.py
    opportunity.py
    vector_search.py
    config.py
  scripts/
    export_postgres_seed.py
  training/
    feature_engineering.py
    build_client_vectors.py
    train_supervised.py
    train_collaborative.py
    train_content_based.py
    train_clusters.py
    train_temporal.py
    train_model.py
    sanity_check.py
  docker-compose.yml
  requirements.txt
  README.md
```

### Resumo por pasta

- `config/`: pesos, parametros de candidatos, cluster, temporal, embeddings e conexao Postgres.
- `api/`: camada HTTP.
- `recommender/`: logica de recomendacao e servicos de inferencia.
- `training/`: pipeline offline para treinamento e metricas.
- `embeddings/`: treino dos embeddings cliente/produto.
- `models/`: artefatos gerados no treino.
- `data/`: datasets de entrada e seeds para banco.
- `db/`: SQL de schema, carga e indices vetoriais.
- `postman/`: collection pronta para testes de API.
- `scripts/`: utilitarios operacionais (export de seed).

---

## 8. Pre-requisitos

Instale:

- Python 3.11+ (recomendado 3.12/3.13)
- pip
- Docker
- Docker Compose (plugin `docker compose`)
- Git

Verifique:

```bash
python --version
pip --version
docker --version
docker compose version
git --version
```

---

## 9. Como clonar o projeto

```bash
git clone https://github.com/Lcooser/agro-recommendation.git
cd agro-recommendation
```

---

## 10. Instalacao das dependencias

```bash
python -m pip install -r requirements.txt
```

Dependencias principais:

- `fastapi`, `uvicorn`: API e servidor ASGI
- `pandas`: ETL/tabular
- `joblib`: salvar/carregar modelos
- `xgboost`: classificadores supervisionados
- `scikit-learn`: clustering, similaridade, preprocessing
- `psycopg2-binary`: conexao com PostgreSQL

---

## 11. Dataset

### 11.1 `data/dataset.csv`

Cada linha representa um par `cliente-produto` com features e target (`comprou`).

| Coluna | Tipo | Descricao |
|---|---|---|
| `cliente_id` | int | Identificador unico do produtor |
| `produto` | str | Produto avaliado para recomendacao/compra |
| `hectares` | float | Tamanho da area produtiva do cliente |
| `cultura` | str | Cultura principal (ex.: soja, milho, trigo) |
| `regiao` | str | Regiao geografica do cliente |
| `possui_pulverizador` | int (0/1) | Se ja possui pulverizador |
| `idade_pulverizador` | int | Idade do pulverizador em anos |
| `possui_plantadeira` | int (0/1) | Se ja possui plantadeira |
| `idade_plantadeira` | int | Idade da plantadeira em anos |
| `possui_colheitadeira` | int (0/1) | Se ja possui colheitadeira |
| `idade_colheitadeira` | int | Idade da colheitadeira em anos |
| `possui_trator` | int (0/1) | Se possui trator |
| `visitas_ultimos_6_meses` | int | Quantidade de interacoes comerciais recentes |
| `num_oportunidades` | int | Oportunidades abertas no CRM |
| `valor_oportunidades` | float | Valor monetario potencial das oportunidades |
| `maquinas_por_hectare` | float | Intensidade tecnologica relativa da operacao |
| `crescimento_area` | float | Variacao da area em relacao ao periodo anterior |
| `comprou` | int (0/1) | Target supervisionado (se comprou o produto) |

### 11.2 `data/purchases.csv`

Historico de compras para collaborative filtering.

| Coluna | Tipo | Descricao |
|---|---|---|
| `cliente_id` | int | Cliente comprador |
| `produto` | str | Produto comprado |

---

## 12. Pipeline de Treino

### 12.1 `training/train_supervised.py`

Treina o modelo supervisionado principal (XGBoost) para estimar `probabilidade_compra`.

Saidas:

- `models/supervised_model.pkl`
- `models/supervised_metrics.json`
- `models/ranking_metrics.json`
- `models/feature_importance_gain.json`

Metricas de classificacao:

- ROC-AUC
- Precision
- Recall
- F1

Metricas de ranking:

- Precision@K
- Recall@K
- NDCG@K
- MAP

### 12.2 `training/train_collaborative.py`

Gera matriz cliente-produto (`crosstab`) e similaridade cosseno para sinais colaborativos.

Saida:

- `models/collaborative_matrix.pkl`

### 12.3 `training/train_content_based.py`

Cria vetor de perfil por cliente (one-hot de cultura/regiao), aplica `StandardScaler` e calcula similaridade cliente-cliente.

Saidas:

- `models/client_similarity.pkl`
- `models/scaler.pkl`

### 12.4 `training/train_clusters.py`

Treina KMeans para segmentar produtores e gerar estatisticas por cluster + produtos dominantes.

Saida:

- `models/producer_clusters.pkl`

### 12.5 `training/train_temporal.py`

Treina modelo temporal para prever compra em 12 meses.

Saidas:

- `models/temporal_model.pkl`
- `models/temporal_metrics.json`

### 12.6 `embeddings/train_embeddings.py`

Treina embeddings de clientes e produtos para busca vetorial.

Saida:

- `models/vector_embeddings.pkl`

---

## 13. Embeddings e Busca Vetorial

### O que sao embeddings

Embeddings sao vetores numericos densos que representam entidades (cliente/produto) em um espaco semantico.

### Como sao gerados

Arquivo: `embeddings/train_embeddings.py`

- entrada: matriz cliente x produto de `purchases.csv`
- reducao com `TruncatedSVD`
- normalizacao L2
- pad para dimensao fixa configurada (`embeddings.dimensions`, ex.: 64)

### Como o vector search funciona neste projeto

O sistema usa **pgvector como principal** e **fallback local em memoria** apenas em falhas de conexao/driver:

- modo principal: SQL no Postgres (`ORDER BY embedding <-> ...`)
- fallback: busca no `models/vector_embeddings.pkl`

Isso evita indisponibilidade do endpoint vetorial em ambiente local Windows com problemas de encoding do driver.

Consulta SQL tipica:

```sql
SELECT cliente_id
FROM client_embeddings
ORDER BY embedding <-> '[0.1,0.2,...]'::vector
LIMIT 10;
```

`<->` representa distancia vetorial (quanto menor, mais proximo).

---

## 14. Banco de Dados

Schema principal (`db/init/01_schema.sql`):

- `clients`
- `client_features`
- `purchases`
- `client_embeddings` (`vector(64)`)
- `product_embeddings` (`vector(64)`)

### Tabelas

| Tabela | Objetivo |
|---|---|
| `clients` | Cadastro basico do produtor |
| `client_features` | Features agregadas para inteligencia comercial |
| `purchases` | Historico de compras |
| `client_embeddings` | Vetor do cliente para similaridade |
| `product_embeddings` | Vetor do produto para busca vetorial |

### Indices vetoriais

Arquivo: `db/init/03_vector_functions.sql`

- `idx_client_embedding` com `ivfflat` + `vector_cosine_ops`
- `idx_product_embedding` com `ivfflat` + `vector_cosine_ops`

---

## 15. Como subir o banco com Docker

Subir:

```bash
docker compose up -d
```

Parar e remover volume (reset completo):

```bash
docker compose down -v
```

Scripts em `db/init/`:

- `01_schema.sql`: extension `vector` e tabelas
- `02_seed.sql`: `COPY` dos CSVs em `/seed`
- `03_vector_functions.sql`: indices vetoriais e `ANALYZE`

---

## 16. Como gerar os seeds

```bash
python scripts/export_postgres_seed.py
```

Arquivos gerados em `data/seed/`:

- `clients.csv`
- `client_features.csv`
- `purchases.csv`
- `client_embeddings.csv`
- `product_embeddings.csv`

---

## 17. Como treinar os modelos

Ordem recomendada:

```bash
python training/train_supervised.py
python training/train_collaborative.py
python training/train_content_based.py
python training/train_clusters.py
python training/train_temporal.py
python embeddings/train_embeddings.py
```

Compatibilidade legado:

```bash
python training/train_model.py
```

`train_model.py` hoje roda apenas o treino supervisionado.

---

## 18. Como rodar a API

Recomendado:

```bash
python -m uvicorn api.main:app --reload
```

Swagger:

- `http://localhost:8000/docs`

---

## 19. Endpoints disponiveis

### 19.1 `POST /recommend`

Retorna top recomendacoes hibridas.

### 19.2 `POST /recommend/probability`

Retorna probabilidade supervisionada para um cliente-produto.

### 19.3 `POST /temporal/purchase`

Retorna probabilidade de compra em 12 meses, janela esperada e motivos.

### 19.4 `GET /similar/{cliente_id}`

Retorna clientes similares (content-based).

### 19.5 `GET /cluster/{cliente_id}`

Retorna cluster do cliente e estatisticas.

### 19.6 `GET /sales/opportunity`

Prioriza clientes para venda de um produto.

### 19.7 `GET /similar_vector/{cliente_id}`

Retorna clientes semanticamente similares via embeddings.

Observacao:

> Em datasets pequenos ou embeddings pouco dispersos, e possivel ver `similarity_score` muito alto (inclusive proximo de `1.0`). Em producao, isso tende a ficar mais distribuido conforme aumentam volume e diversidade de dados.

### 19.8 `GET /similar_vector/{cliente_id}/products`

Retorna produtos proximos ao vetor do cliente.

### 19.9 Glossario das metricas de resposta

| Metrica | Endpoint(s) | O que significa | Faixa tipica |
|---|---|---|---|
| `ml_score` | `/recommend` | Probabilidade do modelo supervisionado para um produto candidato. | 0 a 1 |
| `collaborative_score` | `/recommend` | Forca colaborativa baseada em frequencia entre clientes similares. | 0 a 1 (capado) |
| `similarity_score` (ranking) | `/recommend` | Similaridade content-based penalizada por suporte colaborativo. | 0 a 1 |
| `score_final` | `/recommend` | Score hibrido final para ordenacao. | 0 a 1 |
| `probabilidade_compra` | `/recommend/probability`, `/sales/opportunity` | Chance prevista de compra (classificador supervisionado). | 0 a 1 |
| `probabilidade_12_meses` | `/temporal/purchase` | Chance de compra em 12 meses (modelo temporal). | 0 a 1 |
| `janela_probabilidades` | `/temporal/purchase` | Distribuicao de probabilidade por janela temporal. | soma ~ 1 |
| `opportunity_score` | `/sales/opportunity` | Prioridade comercial final do cliente para um produto. | sem limite fixo |
| `similarity_score` (content) | `/similar/{cliente_id}` | Similaridade entre clientes no modelo content-based. | geralmente 0 a 1 |
| `similarity_score` (vector) | `/similar_vector/{cliente_id}` | Similaridade vetorial (`1 - distancia cosseno`). | geralmente 0 a 1 |
| `vector_score` | `/similar_vector/{cliente_id}/products` | Proximidade vetorial cliente-produto. | geralmente 0 a 1 |
| `urgencia` | `/sales/opportunity` | Fator de urgencia comercial por visitas e crescimento. | ~0.8 a 1.8 |
| `cluster_weight` | `/sales/opportunity` | Peso multiplicativo por cluster de conversao. | configuravel |

---

## 20. Parametros importantes (`config/config.json`)

| Chave | Significado |
|---|---|
| `weights.ml` | peso do score supervisionado no ranking hibrido |
| `weights.similarity` | peso do score de similaridade no ranking hibrido |
| `weights.collaborative` | peso do score colaborativo no ranking hibrido |
| `rules.collab_score_cap` | limite superior do score colaborativo |
| `rules.upgrade_min_age_years` | idade minima para permitir recomendacao de upgrade |
| `clusters.cluster_size` | quantidade de clusters no KMeans |
| `embeddings.dimensions` | dimensao dos vetores cliente/produto |
| `embeddings.nearest_neighbors` | numero maximo de vizinhos em busca local |
| `ranking_metrics.k` | K usado em Precision@K, Recall@K e NDCG@K |
| `temporal.default_month` | mes default da inferencia temporal |
| `temporal.use_in_opportunity` | habilita uso do temporal no opportunity score |
| `temporal.opportunity_weight` | expoente/peso do componente temporal no score |
| `postgres.host` | host do PostgreSQL |
| `postgres.port` | porta do PostgreSQL |
| `postgres.database` | nome do banco |
| `postgres.user` | usuario do banco |
| `postgres.password` | senha do banco |

---

## 21. Como interpretar metricas no contexto comercial

- **Precision@K**: entre os top K produtos recomendados, quantos eram de fato relevantes.
- **Recall@K**: quanto do universo relevante foi recuperado nos top K.
- **MAP**: qualidade media da ordenacao considerando a posicao dos acertos.
- **NDCG**: mede se os itens mais relevantes aparecem no topo.
- **ROC-AUC**: capacidade geral de separar compradores vs nao compradores.
- **F1**: equilibrio entre precision e recall no classificador.

Leitura pratica:

- modelo com bom `Precision@K` tende a gerar listas mais "vendaveis" no topo.
- modelo com bom `Recall@K` tende a perder menos oportunidades.
- modelo com bom `ROC-AUC` tende a separar melhor carteira quente e fria.

---

## 22. Troubleshooting

### Container `agro-db` ja existe

```bash
docker rm -f agro-db
docker compose up -d
```

### Seed nao recarrega

Os scripts de init do Postgres so rodam no primeiro boot do volume:

```bash
docker compose down -v
docker compose up -d
```

### Erro de modelo ausente

Rode novamente os treinos:

```bash
python training/train_supervised.py
python training/train_collaborative.py
python training/train_content_based.py
python training/train_clusters.py
python training/train_temporal.py
python embeddings/train_embeddings.py
```

### Endpoints vetoriais falhando

Verifique:

- banco no ar
- tabelas `client_embeddings` e `product_embeddings` populadas
- `config/config.json` apontando para banco correto
- logs da API para fallback local

---

## 23. Limitacoes conhecidas

- embeddings atuais usam abordagem baseada em SVD (nao two-tower).
- modelo temporal preve compra em 12 meses (nao tempo continuo ate evento).
- dataset de exemplo pode ser pequeno/sintetico para casos reais.
- `opportunity_score` depende da qualidade de `valor_oportunidades`.
- busca vetorial pode gerar scores muito altos em base pequena.
- regra de upgrade hoje cobre principalmente plantadeira, pulverizador e colheitadeira.

---

## 24. Postman Collection

Collection pronta no repositorio:

- `postman/agro-recommendation-engine.postman_collection.json`

Como usar:

1. abrir Postman
2. `Import` do arquivo
3. ajustar `base_url` (default `http://127.0.0.1:8000`)
4. executar requests

---

## 25. Casos de uso

- recomendar produtos para um cliente especifico
- priorizar carteira de visitas por produto
- descobrir clientes semelhantes
- identificar clusters de produtores
- prever janela de compra
- explorar proximidade vetorial entre clientes e produtos

---

## 26. Como contribuir

Fluxo sugerido:

1. fork do repositorio
2. branch de feature (`git checkout -b feat/minha-feature`)
3. commits pequenos e claros
4. rodar sanity checks locais
5. abrir pull request com objetivo, impacto e validacao

Comando util:

```bash
python training/sanity_check.py
python -m uvicorn api.main:app --reload
```

---

## 27. Roadmap futuro

- dashboard web para time comercial
- integracao com CRM
- ingestao de dados climaticos e sazonais
- integracao com dados de satelite
- mapa geoespacial de produtores
- feature store e retrain automatizado
- observabilidade de recomendacao

---

## 28. Glossario de IA

- **ML**: Machine Learning.
- **XGBoost**: algoritmo de gradient boosting para dados tabulares.
- **Embedding**: vetor numerico que representa uma entidade.
- **Vector Search**: busca por proximidade entre vetores.
- **Collaborative Filtering**: recomendacao baseada em comportamento de clientes parecidos.
- **Content-Based**: recomendacao baseada em caracteristicas do cliente/produto.
- **KMeans**: algoritmo de agrupamento nao supervisionado.
- **pgvector**: extensao do PostgreSQL para vetores.
- **NDCG**: metrica de qualidade de ranking.
- **MAP**: Mean Average Precision.
- **ROC-AUC**: capacidade de separacao entre classes.
- **Feature Engineering**: transformacao/criacao de variaveis para o modelo.



---

## Comandos rapidos (resumo operacional)

```bash
# 1) instalar dependencias
python -m pip install -r requirements.txt

# 2) treinar modelos
python training/train_supervised.py
python training/train_collaborative.py
python training/train_content_based.py
python training/train_clusters.py
python training/train_temporal.py
python embeddings/train_embeddings.py

# 3) gerar seeds para postgres
python scripts/export_postgres_seed.py

# 4) subir banco
docker compose up -d

# 5) subir API
python -m uvicorn api.main:app --reload
```

