
from app.rag.generation import ClaudeGenerationService


def main():
    generator = ClaudeGenerationService()

    context_chunks = [
        {
            "text": (
                "Los clientes pueden devolver productos "
                "dentro de los 30 días posteriores a la compra. "
                "El producto debe estar sin usar y conservar "
                "su embalaje original."
            ),
            "source": "return_policy.pdf",
            "page": 1,
        }
    ]

    question = "¿Cuántos días tengo para devolver un producto?"

    answer = generator.generate_answer(
        question=question,
        context_chunks=context_chunks,
    )

    print("\nPregunta:")
    print(question)

    print("\nRespuesta de Claude:")
    print(answer)


if __name__ == "__main__":
    main()
