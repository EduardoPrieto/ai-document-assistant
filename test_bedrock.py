import boto3
from botocore.exceptions import ClientError


REGION = "us-east-1"

MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


def ask_claude(prompt: str) -> str:
    """Send a prompt to Claude through Amazon Bedrock."""

    client = boto3.client(
        "bedrock-runtime",
        region_name=REGION,
    )

    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt,
                    }
                ],
            }
        ],
        inferenceConfig={
            "maxTokens": 300,
            "temperature": 0.2,
        },
    )

    return response["output"]["message"]["content"][0]["text"]


if __name__ == "__main__":
    try:
        response = ask_claude(
            "Explica en español qué es RAG en inteligencia artificial "
            "en máximo 3 frases."
        )

        print("\nRespuesta de Claude:\n")
        print(response)

    except ClientError as error:
        print("\nError de AWS/Bedrock:")
        print(error)

    except Exception as error:
        print("\nError inesperado:")
        print(error)