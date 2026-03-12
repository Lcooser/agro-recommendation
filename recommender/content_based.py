from sklearn.metrics.pairwise import cosine_similarity


def recommend(cliente_vector, produtos_matrix):
    return cosine_similarity([cliente_vector], produtos_matrix)
