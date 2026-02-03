import os
import warnings
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from giskard.rag import QATestset, evaluate, AgentAnswer, KnowledgeBase
from giskard.rag.metrics.ragas_metrics import ragas_context_precision, ragas_context_recall
from giskard.llm.client import set_default_client
from llm_client import get_llm_client
from rag_system import RAGSystem
from knowledge_base import get_mm_paragraphs
from logger import configure_logging, get_logger

warnings.filterwarnings('ignore')

configure_logging()
logger = get_logger(__name__)

load_dotenv()
llm_client = get_llm_client()
set_default_client(llm_client)

try:
    import litellm
    litellm.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    litellm._set_verbosity("ERROR")
    logger.info("Configured litellm with provided key")
except Exception as e:
    logger.debug("litellm not configured (not installed or failed)", error=str(e))


def load_testset():
    candidates = [
        "data/testset.json",
        "data/testset.jsonl",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                logger.info("Loading testset", path=p)
                return QATestset.load(p)
            except Exception as e:
                logger.error("Failed to load testset", path=p, error=str(e))
    raise FileNotFoundError("No testset found in data;")


testset = load_testset()
logger.info("Initializing RAGSystem")
rag_system = RAGSystem()
knowledge_base = KnowledgeBase.from_pandas(
    pd.DataFrame({"text": get_mm_paragraphs()}),
    columns=["text"],
    llm_client=llm_client,
    chunk_size=5000,
)

def answer_fn(question: str, history: list[dict] = None) -> AgentAnswer:
    logger.info("Answering question", question=question)
    answer_text = rag_system.answer(question)
    retrieved_docs_with_scores = rag_system.vector_store.similarity_search_with_score(question, k=2)
    retrieved_docs = [str(doc.page_content) for doc, score in retrieved_docs_with_scores]
    
    logger.info(f"Retrieved {len(retrieved_docs)} documents")
    return AgentAnswer(message=answer_text, documents=retrieved_docs)

metrics_list = [ragas_context_precision, ragas_context_recall]
logger.info("Using metrics", metrics=[m.name for m in metrics_list])

try:
    rag_report = evaluate(
        answer_fn,
        testset=testset,
        knowledge_base=knowledge_base,
        metrics=metrics_list,
        llm_client=llm_client
    )
    logger.info("Evaluation complete")
    
    kb = rag_report._knowledge_base
    if kb:
        kb._documents_index = {
            key: doc
            for idx, doc in enumerate(kb._documents)
            for key in (doc.id, idx)
        }
        if "metadata" in rag_report._dataframe.columns and kb._documents:
            rag_report._dataframe["metadata"] = rag_report._dataframe["metadata"].apply(
                lambda md: {**md, "seed_document_id": 0} if isinstance(md, dict) and "seed_document_id" in md else md
            )
    report_dir = Path("data/rag_evaluation_report")
    
    if report_dir.exists():
        import shutil
        shutil.rmtree(report_dir)
    
    report_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving report to: {report_dir.absolute()}")
    rag_report.save(str(report_dir))
    
    saved_files = list(report_dir.iterdir())
    logger.info(f"Successfully saved {len(saved_files)} files: {[f.name for f in saved_files]}")
    logger.info(f"Overall correctness: {rag_report.correctness:.2%}")
    logger.info(f"Report available at: {report_dir / 'report.html'}")
    
except Exception as e:
    logger.error("Evaluation or save failed", error=str(e), exc_info=True)
