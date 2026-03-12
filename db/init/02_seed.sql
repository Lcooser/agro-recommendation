TRUNCATE TABLE
    purchases,
    client_features,
    client_embeddings,
    product_embeddings,
    clients
RESTART IDENTITY;

COPY clients (
    cliente_id,
    hectares,
    cultura,
    regiao
)
FROM '/seed/clients.csv'
WITH (FORMAT csv, HEADER true);

COPY purchases (
    cliente_id,
    produto
)
FROM '/seed/purchases.csv'
WITH (FORMAT csv, HEADER true);

COPY client_features (
    cliente_id,
    visitas_ultimos_6_meses,
    num_oportunidades,
    valor_oportunidades,
    maquinas_por_hectare,
    crescimento_area,
    possui_pulverizador,
    possui_plantadeira,
    possui_colheitadeira,
    possui_trator,
    idade_pulverizador,
    idade_plantadeira,
    idade_colheitadeira
)
FROM '/seed/client_features.csv'
WITH (FORMAT csv, HEADER true);

COPY client_embeddings (
    cliente_id,
    embedding
)
FROM '/seed/client_embeddings.csv'
WITH (FORMAT csv, HEADER true);

COPY product_embeddings (
    produto,
    embedding
)
FROM '/seed/product_embeddings.csv'
WITH (FORMAT csv, HEADER true);
