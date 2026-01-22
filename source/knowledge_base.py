import os

def load_knowledge_base(file_path: str) -> str:
    with open(file_path, 'r', encoding="utf-8") as file:
        return file.read()

def get_mm() -> str:
    return load_knowledge_base("data/MM.txt")