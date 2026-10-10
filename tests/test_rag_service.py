
from app.rag.rag_service import RAGService


def main():
    rag = RAGService()

    question = input("Ask a question about the document: ").strip()

    result = rag.ask(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved sources:")
    for source in result["sources"]:
        print(
            f"- {source['source']}, "
            f"page {source['page']}, "
            f"chunk {source['chunk']}, "
            f"score {source['score']:.4f}"
        )


if __name__ == "__main__":
    main()
