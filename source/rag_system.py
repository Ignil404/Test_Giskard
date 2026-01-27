import warnings
from dotenv import load_dotenv
from knowledge_base import prepare_vector_store
from llm_client import GeminiClient, GroqClient
from giskard.llm.client.base import ChatMessage
from logger import get_logger, configure_logging

warnings.filterwarnings("ignore")
load_dotenv()

configure_logging()
logger = get_logger(__name__)

class RAGSystem:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.client = GeminiClient()
        self.vector_store = prepare_vector_store(persist_directory)
        #self.client = GroqClient()
        logger.info("RAGSystem initialized")

    def get_retrieved_documents(self, query: str, top_k: int = 5) -> list[str]:
        logger.info("Retrieving documents", query=query, top_k=top_k)
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
        retrieved_docs = [doc.page_content for doc, score in docs_with_scores]
        logger.info("Documents retrieved", num_docs=len(retrieved_docs))
        return retrieved_docs

    def answer(self, question: str, top_k: int = 5) -> str:
        logger.info("Answering question", question=question)
        retrieved_docs = self.get_retrieved_documents(question, top_k)
        context = "\n".join(retrieved_docs)
        prompt = f"""На основе следующего текста ответь на вопрос.
        Текст:
        {context}
        Вопрос: {question}
        Ответ должен быть основан только на информации из текста. Если информации нет в тексте, скажи "Информации нет в тексте"."""
        messages = [ChatMessage(role="user", content=prompt)]
        response_msg = self.client.complete(messages)
        return response_msg.content

if __name__ == "__main__":
    rag = RAGSystem()
    logger.info("RAGSystem initialized")