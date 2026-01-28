import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from giskard.llm.client import LLMClient
from giskard.llm.client.base import ChatMessage

load_dotenv()

class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.5-flash-lite", base_url: str = "https://api.aitunnel.ru/v1/"):
        self.client = OpenAI(
            api_key=os.getenv('GEMINI_API_KEY'),
            base_url=base_url
        )
        self.model = model
        self.base_url = base_url

    def complete(self, messages: list[ChatMessage], temperature: float = 0.5, max_tokens: int = 1000, **kwargs) -> ChatMessage:
        formatted_messages = [
            {"role": m.role, "content": m.content}
            for m in messages if m.content
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return ChatMessage(role="assistant", content=response.choices[0].message.content)

    def get_config(self) -> dict:
        return {"model": self.model, "api": "google-genai", "base_url": self.base_url}

    def chat(self, messages: list[ChatMessage]) -> str:
        formatted_messages = [
            {"role": message.role, "content": message.content}
            for message in messages if message.content
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
        )
        return response.choices[0].message.content


class GroqClient(LLMClient):
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = model

    def complete(self, messages: list[ChatMessage], temperature: float = 0.5, max_tokens: int = 150, **kwargs) -> ChatMessage:
        formatted_messages = [
            {"role": m.role, "content": m.content}
            for m in messages if m.content
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return ChatMessage(role="assistant", content=response.choices[0].message.content)

    def get_config(self) -> dict:
        return {"model": self.model, "api": "groq"}

    def chat(self, messages: list[ChatMessage]) -> str:
        formatted_messages = [
            {"role": message.role, "content": message.content}
            for message in messages if message.content
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
        )
        return response.choices[0].message.content
