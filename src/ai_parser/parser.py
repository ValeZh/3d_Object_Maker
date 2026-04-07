import requests
import json
import re

DEEPSEEK_API_KEY = "sk-7f5f5e5858b64a8d8d6b62bad95938e9"

def send_text_to_deepseek(text: str):
    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Extract shape, color, texture and additional features from text.
Normalize words (cubick→cube, reddish→red).
Return ONLY JSON:
{{
  "shape": "...",
  "color": "...",
  "texture": "...",
  "additional_features": "..."
}}

Text: "{text}"
"""

    payload = {
        "model": "deepseek-chat",  # основная модель
        "messages": [
            {"role": "system", "content": "You are a strict JSON generator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        content = data["choices"][0]["message"]["content"]

        print("Ответ DeepSeek:", content)

        # Ищем JSON
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        return None

    except Exception as e:
        print("Ошибка DeepSeek:", e)
        return None