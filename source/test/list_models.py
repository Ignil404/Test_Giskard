import os
from dotenv import load_dotenv
from google import genai
from groq import Groq


load_dotenv()

print("GEMINI MODELS")
client1 = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
models1 = client1.models.list()
for model in models1:
    print(f"- {model.name}")

print("GROQ MODELS")
client2 = Groq(api_key=os.getenv('GROQ_API_KEY'))
models2 = client2.models.list()
for model in models2.data:
    print(f"- {model.id}")