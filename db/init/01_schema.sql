CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS clients (
    cliente_id INTEGER PRIMARY KEY,
    hectares INTEGER NOT NULL,
    cultura TEXT NOT NULL,
    regiao TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS client_features (
    cliente_id INTEGER PRIMARY KEY REFERENCES clients(cliente_id) ON DELETE CASCADE,
    visitas_ultimos_6_meses INTEGER NOT NULL,
    num_oportunidades INTEGER NOT NULL,
    valor_oportunidades NUMERIC NOT NULL,
    maquinas_por_hectare DOUBLE PRECISION NOT NULL,
    crescimento_area DOUBLE PRECISION NOT NULL,
    possui_pulverizador SMALLINT NOT NULL,
    possui_plantadeira SMALLINT NOT NULL,
    possui_colheitadeira SMALLINT NOT NULL,
    possui_trator SMALLINT NOT NULL,
    idade_pulverizador INTEGER NOT NULL,
    idade_plantadeira INTEGER NOT NULL,
    idade_colheitadeira INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS purchases (
    cliente_id INTEGER NOT NULL REFERENCES clients(cliente_id) ON DELETE CASCADE,
    produto TEXT NOT NULL,
    PRIMARY KEY (cliente_id, produto)
);

CREATE TABLE IF NOT EXISTS client_embeddings (
    cliente_id INTEGER PRIMARY KEY REFERENCES clients(cliente_id) ON DELETE CASCADE,
    embedding vector(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS product_embeddings (
    produto TEXT PRIMARY KEY,
    embedding vector(64) NOT NULL
);
