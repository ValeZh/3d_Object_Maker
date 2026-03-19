import requests
import json
import re

OLLAMA_URL = "http://127.0.0.1:11434/v1/completions"

def send_text_to_ollama(text: str):
    payload = {
        "model": "llama2",
        "prompt": f"""
You are a text normalizer and structured data extractor. Your tasks:

1. Extract all relevant information from the input text about objects, including:
   - Main color
   - Shape
   - Texture
   - Additional features (like secondary colors, patterns, or parts)

2. Normalize all words:
   - Correct spelling mistakes.
   - Convert words written in mixed case, caps, or informal forms to standard lowercase English.
   - Map all shape words to standard forms (e.g., 'CUBE', 'cubick' -> 'cube'; 'circle' -> 'sphere').
   - Convert diminutives or casual color forms to standard color names (e.g., 'reddish', 'rosy' -> 'red').

3. Determine the **main color** as the primary color of the object and treat other colors or parts as **additional features** (e.g., 'reddish cubick with a blue top' -> color: red, additional_features: "blue top").

4. Return only JSON with these fields: 
   {{
     "shape": "...",
     "color": "...",
     "additional_features": "..."
   }}

Input text: '{text}'
""",
        "max_tokens": 200
    }

    response = requests.post(OLLAMA_URL, json=payload)
    data = response.json()

    # Получаем текст из выбора
    text_completion = data['choices'][0]['text']

    # Ищем JSON внутри текста (между фигурными скобками)
    match = re.search(r"\{.*\}", text_completion, re.DOTALL)
    if match:
        json_text = match.group(0)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print("Не удалось распарсить JSON из найденного текста:")
            print(json_text)
            return None
    else:
        print("JSON не найден в ответе Ollama:")
        print(text_completion)
        return None


if __name__ == "__main__":
    text = "Tell me about reddish CuBe with black color on top"
    result = send_text_to_ollama(text)
    print("Результат:", result)