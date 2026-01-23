# Test_Giskard - RAG System Evaluation

Система для тестирования и оценки RAG (Retrieval Augmented Generation) с использованием Giskard.

## Описание

Этот проект реализует RAG систему на базе знаний (knowledge base) с поддержкой:
- **Gemini** и **Groq** LLM клиентов
- Оценки качества ответов с помощью Giskard
- Извлечения релевантной информации из базы знаний
- Генерации и обработки набора тестовых вопросов

## Структура проекта

```
Test_Giskard/
├── source/
│   ├── llm_client.py            # LLM клиенты (Gemini, Groq)
│   ├── rag_system.py            # Система RAG с поиском по знаниям
│   ├── knowledge_base.py        # Загрузка базы знаний
│   ├── evaluate_rag.py          # Скрипт оценки с Context Precision и Context Recall
│   ├── evaluate_rag2_gemini.py  # Скрипт оценки с CorrectnessMetric(меньше нагрузка на api)
│   ├── evaluate_rag2_groq.py    # Скрипт оценки с CorrectnessMetric
│   ├── generate_questions.py    # Генерация тестовых вопросов
│   ├── view_questions.py        # Просмотр вопросов
│   └── test/                    # Тесты(Использовались на начальном этапе, сейчас бесполезны)
├── data/
│   ├── MM.txt                   # База знаний
│   ├── testset.json             # Набор тестовых вопросов
│   ├── testset.jsonl            # Вопросы, которые не помещаются в запрос
│   └── evaluation_report/       # Результаты оценки
├── pyproject.toml               # Конфигурация проекта
└── .env                         # API ключи
```

## Требования

- Python 3.11
- UV package manager

## Установка

1. Клонируйте репозиторий или перейдите в папку проекта
2. Создайте файл `.env` с необходимыми ключами API:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

3. Установите зависимости:

```bash
uv sync
```

## Использование

### Запуск оценки RAG системы

```bash
cd Test_Giskard
uv run python source/evaluate_rag.py

или ( с более щадящей нагрузкой на лимиты)
uv run python source/evaluate_rag2_gemini.py
uv run python source/evaluate_rag2_groq.py
```

### Генерация тестовых вопросов

```bash
uv run python source/generate_questions.py
```

### Просмотр вопросов из набора тестов

```bash
uv run python source/view_questions.py
```

## Компоненты

### LLM Client (`llm_client.py`)
Поддерживает два LLM провайдера:
- **GeminiClient**: использует Google Gemini API
- **GroqClient**: использует Groq API

### RAG System (`rag_system.py`)
Система ответов на основе знаний:
- Загружает базу знаний из файла
- Ищет релевантную информацию по ключевым словам вопроса
- Формирует промпт с контекстом для LLM
- Возвращает ответ на основе предоставленного контекста

### Knowledge Base (`knowledge_base.py`)
Загрузка и управление базой знаний:
- Читает информацию из файла `data/MM.txt`
- Предоставляет текстовый контент для RAG системы

## Ограничения и советы

### Rate Limit при использовании
- Groq free имеет лимит 6000/12000 токенов в минуту
- Gemini free имеет 5 запросов в минуту(но больше контекста)


## Результаты

Результаты оценки сохраняются в:
- `data/evaluation_report/` - полный отчет Giskard
- `data/rag_evaluation_report(gemini/groq)/report.html` - HTML визуализация

## Поддерживаемые метрики

- **CorrectnessMetric**: Оценка корректности ответов на основе базы данных
- **Context Precision**: Точность извлеченного контекста
- **Context Recall**: Полнота извлеченного контекста
