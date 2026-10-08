from app.rag.vector_store import VectorStore


store = VectorStore(dimension=3)


documents = [
    {
        "text": "You have 14 days to cancel your order.",
        "source": "return_policy.pdf",
        "page": 2,
    },
    {
        "text": "Refunds are processed after the returned product is received.",
        "source": "return_policy.pdf",
        "page": 2,
    },
    {
        "text": "Gift refunds are sent to the original purchaser.",
        "source": "return_policy.pdf",
        "page": 3,
    },
]


embeddings = [
    [1.0, 0.0, 0.0],
    [0.7, 0.7, 0.0],
    [0.0, 1.0, 0.0],
]


store.add(
    embeddings=embeddings,
    documents=documents,
)


query_embedding = [0.95, 0.05, 0.0]


results = store.search(
    query_embedding=query_embedding,
    top_k=2,
)


print("\nSearch results:\n")

for result in results:

    print("=" * 60)

    print(f"Score: {result['score']:.4f}")
    print(f"Source: {result['source']}")
    print(f"Page: {result['page']}")
    print(f"Text: {result['text']}")