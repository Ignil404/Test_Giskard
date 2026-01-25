from giskard.rag import QATestset
from logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
testset = QATestset.load('data/testset.json')

logger.info(f"Всего вопросов: {len(testset)}")

for i, item in enumerate(testset.samples, 1):
    logger.info(f"Вопрос {i}: {item.question}")
    if hasattr(item, 'reference_answer') and item.reference_answer:
        logger.info(f"Референсный ответ: {item.reference_answer}")
    print("-" * 80)