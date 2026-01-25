from giskard.rag import QATestset
from logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
testset = QATestset.load('data/testset.json')

print(f"Всего вопросов: {len(testset)}")
print("=" * 80)

for i, item in enumerate(testset.samples, 1):
    print(f"\nВопрос {i}: {item.question}")
    if hasattr(item, 'reference_answer') and item.reference_answer:
        print(f"Референсный ответ: {item.reference_answer}")
    if hasattr(item, 'reference_context') and item.reference_context:
        print(f"Референсный контекст: {item.reference_context[:200]}...")
    print("-" * 80)