import os
import json
import pandas as pd
from dotenv import load_dotenv
from giskard import scan
from giskard.rag import QATestset, evaluate, AgentAnswer, KnowledgeBase, RAGReport, generate_testset
from giskard.rag.metrics import CorrectnessMetric
from giskard.llm.client import set_default_client
from llm_client import GeminiClient, GroqClient
from rag_system import RAGSystem
from knowledge_base import get_mm
from giskard.rag.metrics.ragas_metrics import ragas_context_precision, ragas_context_recall

load_dotenv()
gemini_client = GeminiClient()
#groq_client = GroqClient()
set_default_client(gemini_client)
#set_default_client(groq_client)

def load_testset():
    candidates = [
        "data/testset.json",
        "data/testset.jsonl",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                print(f"Loading testset from {p}")
                return QATestset.load(p)
            except Exception as e:
                print(f"Failed to load {p}: {e}")
    raise FileNotFoundError("No testset found in data;")


testset = load_testset()

rag_system = RAGSystem()

text = get_mm()
paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
knowledge_base_df = pd.DataFrame({"text": paragraphs})
knowledge_base = KnowledgeBase.from_pandas(knowledge_base_df, columns=["text"])


def answer_fn(question: str, history: list[dict] = None) -> AgentAnswer:
    answer_text = rag_system.answer(question)
    return AgentAnswer(message=answer_text, documents=paragraphs)


metrics_list = [ragas_context_precision, ragas_context_recall]
print("Using metrics:", [m.name for m in metrics_list])


try:
    rag_report = evaluate(
        answer_fn,
        testset=testset,
        knowledge_base=knowledge_base,
        metrics=metrics_list,
        llm_client=gemini_client)
    print("Evaluation complete")
    try:
        rag_report.save("data/rag_evaluation_report")
        print("Saved RAGReport to data/rag_evaluation_report/")
    except Exception as e:
        print(f"Warning: failed to save RAGReport: {e}")

except Exception as e:
    print(f"Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
