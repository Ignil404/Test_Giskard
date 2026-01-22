from giskard.rag import QATestset

testset = QATestset.load('data/testset.json')

print(f"Всего вопросов: {len(testset)}")

for i, item in enumerate(testset.samples, 1):
    print(f"Вопрос {i}: {item.question}")
    if hasattr(item, 'reference_answer') and item.reference_answer:
        print(f"Референсный ответ: {item.reference_answer}")
    print("-" * 80)