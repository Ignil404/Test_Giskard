import os
from dotenv import load_dotenv
from knowledge_base import get_mm
from llm_client import GeminiClient, GroqClient
from giskard.llm.client.base import ChatMessage

load_dotenv()


class RAGSystem:
    def __init__(self):
        #self.client = GeminiClient()
        self.client = GroqClient()
        self.knowledge_base = get_mm()

    def answer(self, question: str) -> str:
        prompt = f"""На основе следующего текста ответь на вопрос.
        Текст:
        {self.knowledge_base}
        Вопрос: {question}
        Ответ должен быть основан только на информации из текста. Если информации нет в тексте, скажи "Информации нет в тексте"."""
        messages = [ChatMessage(role="user", content=prompt)]
        response_msg = self.client.complete(messages)
        return response_msg.content