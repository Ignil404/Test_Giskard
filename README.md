# Test_Giskard

RAG-система для романа «Мастер и Маргарита» и набор утилит для генерации вопросов и оценки качества ответов через Giskard.

## Что внутри

Проект делает три вещи:
- готовит базу знаний и векторное хранилище
- отвечает на вопросы по контексту
- оценивает ответы метриками Giskard и строит HTML-отчёт

## Быстрый старт

1. Установите зависимости:
```bash
uv sync
```

2. Создайте `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=gemini
```

3. Запустите основной сценарий:
```bash
uv run python main.py
```

Отчёт будет в `data/rag_evaluation_report/report.html`.

## Команды

Запуск оценки:
```bash
uv run python source/evaluate_rag.py
```

Генерация тестсета:
```bash
uv run python source/generate_questions.py
```

Просмотр тестсета:
```bash
uv run python source/view_questions.py
```

## Как это работает

1. `knowledge_base.py` читает `data/MM.txt`, режет текст на чанки и строит `Chroma`-векторное хранилище.
2. `rag_system.py` достаёт релевантные чанки, формирует промпт и вызывает LLM.
3. `evaluate_rag.py` прогоняет тестсет и считает метрики (Context Precision/Recall), затем формирует HTML-отчёт.

## Структура проекта

```
Test_Giskard/
├── source/
│   ├── llm_client.py            # LLM клиенты
│   ├── rag_system.py            # RAG пайплайн и ответы
│   ├── knowledge_base.py        # База знаний и векторное хранилище
│   ├── evaluate_rag.py          # Оценка и отчёт
│   ├── generate_questions.py    # Генерация тестсета
│   ├── view_questions.py        # Просмотр тестсета
│   └── test/                    # Черновые тесты
├── data/
│   ├── MM.txt                   # База знаний
│   ├── testset.json             # Тестсет (JSONL)
│   ├── testset.jsonl            # Альтернативный формат
│   └── rag_evaluation_report/   # Отчёт
├── pyproject.toml
└── .env
```

## Конфигурация

Обязательные переменные окружения:
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
 - `LLM_CLIENT` (`gemini` или `groq`)

Дополнительно:
- можно сменить модель через `GeminiClient(model="...")` в `source/llm_client.py`
- можно переключиться на Groq через `LLM_CLIENT=groq`

## Метрики

- **CorrectnessMetric**: Оценка корректности ответов на основе базы данных
- **Context Precision**: Точность извлеченного контекста
- **Context Recall**: Полнота извлеченного контекста
