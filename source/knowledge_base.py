import os

def load_knowledge_base(file_path: str) -> str:
    with open(file_path, 'r', encoding="utf-8") as file:
        return file.read()

def get_mm() -> str:
    return load_knowledge_base("data/MM.txt")

def get_mm_paragraphs() -> list[str]:
    mm_content = get_mm()
    return [para.strip() for para in mm_content.split('\n') if para.strip()]