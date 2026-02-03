import os
from dotenv import load_dotenv
from google import genai
from groq import Groq
from giskard.llm.client import LLMClient
from giskard.llm.client.base import ChatMessage
from logger import get_logger

load_dotenv()
logger = get_logger(__name__)

def _format_messages(messages: list[ChatMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages if m.content]

def _chat_completion(client, model: str, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None):
    params = {"model": model, "messages": messages}
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    return client.chat.completions.create(**params)

class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = model

    def complete(self, messages: list[ChatMessage], temperature: float = 0.5, max_tokens: int = 1000, **kwargs) -> ChatMessage:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in _format_messages(messages))
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        )
        return ChatMessage(role="assistant", content=response.text or "")

    def get_config(self) -> dict:
        return {"model": self.model, "api": "google-genai"}

    def chat(self, messages: list[ChatMessage]) -> str:
        return self.complete(messages).content


class GroqClient(LLMClient):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = model

    def complete(self, messages: list[ChatMessage], temperature: float = 0.5, max_tokens: int = 150, **kwargs) -> ChatMessage:
        formatted_messages = _format_messages(messages)
        response = _chat_completion(
            self.client,
            self.model,
            formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return ChatMessage(role="assistant", content=response.choices[0].message.content)

    def get_config(self) -> dict:
        return {"model": self.model, "api": "groq"}

    def chat(self, messages: list[ChatMessage]) -> str:
        formatted_messages = _format_messages(messages)
        response = _chat_completion(self.client, self.model, formatted_messages)
        return response.choices[0].message.content


def get_llm_client(provider: str | None = None) -> LLMClient:
    provider = (provider or os.getenv("LLM_CLIENT") or "gemini").lower()
    if provider == "groq":
        return GroqClient()
    return GeminiClient()
