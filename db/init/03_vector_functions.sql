CREATE INDEX IF NOT EXISTS idx_client_embedding
ON client_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_product_embedding
ON product_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

ANALYZE client_embeddings;
ANALYZE product_embeddings;
