import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from logger import get_logger
from dotenv import load_dotenv

logger = get_logger(__name__)

def load_knowledge_base(file_path: str) -> str:
    logger.info("Loading knowledge base", file_path=file_path)
    with open(file_path, 'r', encoding="utf-8") as file:
        logger.info("Knowledge base loaded", file_path=file_path)
        return file.read()

def get_mm() -> str:
    return load_knowledge_base("data/MM.txt")

def get_mm_paragraphs() -> list[str]:
    text = get_mm()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    logger.info("Knowledge base split into chunks", num_chunks=len(chunks))
    return chunks

def prepare_vector_store(persist_directory: str = "data/chroma_db") -> Chroma:
    load_dotenv()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=os.getenv("GEMINI_API_KEY"))

    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    #     model_kwargs={'device': 'cpu'},
    #     encode_kwargs={'normalize_embeddings': True}
    # )

    if os.path.exists(persist_directory):
        logger.info("Loading existing vector store", path=persist_directory)
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    logger.info("Creating new vector store")
    chunks = get_mm_paragraphs()
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    logger.info("Vector store created and persisted", path=persist_directory)
    return vector_store

if __name__ == "__main__":
    prepare_vector_store()
