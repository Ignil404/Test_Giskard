import os
import pandas as pd
from dotenv import load_dotenv
from giskard.rag import QATestset, evaluate, AgentAnswer, KnowledgeBase, RAGReport, generate_testset
from giskard.rag.metrics import CorrectnessMetric
from giskard.llm.client import set_default_client
from llm_client import GroqClient
from rag_system import RAGSystem
from knowledge_base import get_mm, get_mm_paragraphs
from giskard.rag.metrics.ragas_metrics import ragas_context_precision, ragas_context_recall
from logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
load_dotenv()
groq_client = GroqClient()
set_default_client(groq_client)

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

rag_system = RAGSystem()

paragraphs = get_mm_paragraphs()
knowledge_base_df = pd.DataFrame({"text": paragraphs})
knowledge_base = KnowledgeBase.from_pandas(knowledge_base_df, columns=["text"])


def answer_fn(question: str, history: list[dict] = None) -> AgentAnswer:
    answer_text = rag_system.answer(question)
    return AgentAnswer(message=answer_text, documents=paragraphs)


metrics_list = [CorrectnessMetric(name="correctness")]
logger.info("Using metrics", metrics=[m.name for m in metrics_list])

try:
    rag_report = evaluate(
        answer_fn,
        testset=testset,
        knowledge_base=knowledge_base,
        metrics=metrics_list,
        llm_client=groq_client)
    logger.info("Evaluation complete")
    try:
        rag_report.save("data/rag_evaluation_report_groq")
        logger.info("Saved RAGReport to data/rag_evaluation_report_groq/", path="data/rag_evaluation_report_groq/")
    except Exception as e:
        logger.warning("Failed to save RAGReport", path="data/rag_evaluation_report_groq/", error=str(e))

except Exception as e:
    logger.error("Evaluation failed", error=str(e))