import os
import warnings
import pandas as pd
from dotenv import load_dotenv
from giskard.rag import generate_testset, KnowledgeBase
from giskard.rag.question_generators import (
    simple_questions,
    complex_questions,
    distracting_questions,
)
from giskard.llm.client import set_default_client
from llm_client import GeminiClient, GroqClient
from knowledge_base import get_mm

warnings.filterwarnings("ignore")

load_dotenv()

llm_client1 = GeminiClient()
# set_default_client(llm_client1)
llm_client2 = GroqClient()
set_default_client(llm_client1)
text = get_mm()

paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
df = pd.DataFrame({"text": paragraphs})

knowledge_base = KnowledgeBase.from_pandas(df, columns=["text"])
testset = generate_testset(
    knowledge_base,
    num_questions=5,
    question_generators=[
        simple_questions,
        complex_questions,
        distracting_questions,
    ],
    language='ru',
    agent_description="Система для ответов на вопросы по роману 'Мастер и Маргарита'.",
)
testset.save("data/testset.json")