import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from logger import get_logger

logger = get_logger(__name__)

def load_knowledge_base(file_path: str) -> str:
    logger.info("Loading knowledge base", file_path=file_path)
    with open(file_path, 'r', encoding="utf-8") as file:
        logger.info("Knowledge base loaded", file_path=file_path)
        return file.read()

def get_mm() -> str:
    return load_knowledge_base("data/MM.txt")

def get_mm_paragraphs(text: str) -> list[str]:
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
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if os.path.exists(persist_directory):
        logger.info("Loading existing vector store", path=persist_directory)
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    
    logger.info("Creating new vector store")
    chunks = get_mm_paragraphs(get_mm())
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    logger.info("Vector store created and persisted", path=persist_directory)
    return vector_store