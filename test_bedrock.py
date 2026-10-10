
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise RuntimeError(
        "No se encontró ANTHROPIC_API_KEY en el archivo .env"
    )

client = anthropic.Anthropic(api_key=api_key)

try:
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    "Explica en español qué es RAG en inteligencia "
                    "artificial en un máximo de tres frases."
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "text":
            print("\nRespuesta de Claude:\n")
            print(block.text)

    print(f"\nTokens de entrada: {response.usage.input_tokens}")
    print(f"Tokens de salida: {response.usage.output_tokens}")

except anthropic.AuthenticationError:
    print("Error: la API key no es válida o no está autorizada.")

except anthropic.RateLimitError:
    print("Error: se alcanzó un límite de uso de la API.")

except anthropic.APIStatusError as error:
    print(f"Error de Anthropic: HTTP {error.status_code}")
    print(error.message)

except anthropic.APIConnectionError:
    print("Error de conexión con la API de Anthropic.")
