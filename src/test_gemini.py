import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set in .env"
    )

client = genai.Client(
    api_key=api_key
)

print("Gemini connection successful.")
print()
print("Available generation models:")

count = 0

for model in client.models.list():

    actions = getattr(
        model,
        "supported_actions",
        []
    )

    if "generateContent" in actions:

        print(model.name)
        count += 1

        if count >= 10:
            break

print()
print("Models shown:", count)