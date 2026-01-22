from rag import RAGSystem

rag = RAGSystem()

questions = [
    "Где происходит действие в начале романа?",
    "Как звали редактора журнала?",
    "Кто такой Бездомный?"
]
for question in questions: 
    print (f"Вопрос: {question}")
    print("Ответ:", rag.answer_question(question))
