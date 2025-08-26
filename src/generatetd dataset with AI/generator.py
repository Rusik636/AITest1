import requests
import json
from pathlib import Path
from tqdm import tqdm  # библиотека для прогресс-бара

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "openhermes-2.5-mistral-7b.Q5_K_S:latest"
DATASET_PATH = Path("dataset.jsonl")

PROMPT_TEMPLATE = """
Сгенерируй {n} примеров в формате JSON:
{{"text": "<реплика на русском>", "intent": "<одна метка из [buy, sell, cancel, support]>"}}
Требования:
- реплики должны быть на русском языке
- стиль — повседневная речь, как в чате
- разнообразные формулировки и ситуации
- не используй англоязычные слова
"""

def generate_batch(n=20):
    """Отправка промпта в Ollama и получение батча синтетики из streaming JSON"""
    prompt = PROMPT_TEMPLATE.format(n=n)
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt},
        stream=True  # включаем стриминг
    )
    response.raise_for_status()

    full_text = ""
    for line in response.iter_lines(decode_unicode=True):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
            full_text += chunk.get("response", "")
            if chunk.get("done", False):
                break
        except json.JSONDecodeError:
            continue
    return full_text

def parse_json_lines(text: str):
    """Парсит строку с JSON-объектами, возвращает список словарей"""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            lines.append(obj)
        except json.JSONDecodeError:
            continue
    return lines

def main(total=200, batch_size=20):
    """Основная функция для генерации датасета"""
    DATASET_PATH.unlink(missing_ok=True)
    collected = 0
    # Создаём tqdm прогресс-бар
    with tqdm(total=total, desc="Генерация датасета", ncols=100) as pbar:
        while collected < total:
            raw = generate_batch(batch_size)
            examples = parse_json_lines(raw)
            with open(DATASET_PATH, "a", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            collected += len(examples)
            pbar.update(len(examples))  # обновляем прогресс-бар

if __name__ == "__main__":
    main(total=200, batch_size=20)
