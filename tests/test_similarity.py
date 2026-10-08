from app.rag.similarity import cosine_similarity


vector_a = [1, 0, 0]
vector_b = [1, 0, 0]

similarity = cosine_similarity(
    vector_a,
    vector_b,
)

print(f"Similarity: {similarity}")