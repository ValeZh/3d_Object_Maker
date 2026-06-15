"""
parser_modules.py — DeepSeek интеграция для парсинга ОТДЕЛЬНЫХ МОДУЛЕЙ
Переписано для извлечения параметров модулей (стена, окно, дверь, балкон, вход)
"""

import requests
import json
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ⚠️ ВАЖНО: Замени на твой реальный API ключ
DEEPSEEK_API_KEY = "sk-7f5f5e5858b64a8d8d6b62bad95938e9"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


def send_module_text_to_deepseek(text: str, module_type: Optional[str] = None) -> dict:
    """
    Отправляет текст описания МОДУЛЯ в DeepSeek и извлекает параметры.

    Args:
        text: Описание модуля (стена, окно, дверь, балкон, вход)
        module_type: Подсказка о типе модуля (опционально)

    Returns:
        dict: Извлеченные параметры модуля

    Примеры:
        - "стена 3м высота, 2м ширина, бетон" →
          {"module_type": "wall", "height": 3, "width": 2, "material": "concrete"}

        - "окно 1.2м ширина, 1.5м высота, двойное" →
          {"module_type": "window", "width": 1.2, "height": 1.5, "style": "double"}

        - "дверь входная 2.1м, 0.9м деревянная" →
          {"module_type": "door", "height": 2.1, "width": 0.9, "material": "wood"}
    """

    # Промпт для парсинга отдельного модуля
    type_hint = f"Известно, что это модуль типа: {module_type}. " if module_type else ""

    prompt = f"""
Ты архитектор. Извлеки параметры ОТДЕЛЬНОГО МОДУЛЯ из описания на русском языке.

{type_hint}

Определи тип модуля (wall, window, door, balcony, entrance) и верни ТОЛЬКО JSON (без пояснений):

Для СТЕНЫ (wall):
{{
  "module_type": "wall",
  "height": <высота в метрах, 1.5-5.0>,
  "width": <ширина в метрах, 0.5-5.0>,
  "color": "<цвет: hex или название>",
  "material": "<бетон, кирпич, камень и т.д.>",
  "thickness": <толщина в метрах, по умолчанию 0.3>
}}

Для ОКНА (window):
{{
  "module_type": "window",
  "width": <ширина в метрах, 0.8-2.5>,
  "height": <высота в метрах, 0.8-2.0>,
  "style": "<single или double>",
  "frame_color": "<цвет рамы>",
  "glass_color": "<цвет стекла>"
}}

Для ДВЕРИ (door):
{{
  "module_type": "door",
  "height": <высота в метрах, 2.0-2.5>,
  "width": <ширина в метрах, 0.7-1.2>,
  "style": "<standard, modern или glass>",
  "material": "<дерево, металл, стекло>",
  "frame_color": "<цвет рамы>"
}}

Для БАЛКОНА (balcony):
{{
  "module_type": "balcony",
  "depth": <глубина в метрах, 0.8-2.0>,
  "width": <ширина в метрах, 1.0-4.0>,
  "style": "<open или enclosed>",
  "has_roof": <true/false — крыша сверху, для лоджии/enclosed обычно true>,
  "roof_thickness": <толщина крыши в метрах, 0.1-0.25>,
  "roof_overhang": <свес крыши в метрах, 0-0.15>,
  "parapet_height": <высота перил в метрах, по умолчанию 1.1>,
  "color": "<цвет стен>",
  "roof_color": "<цвет крыши, hex #RRGGBB>"
}}

Для ВХОДА/ПОДЪЕЗДА (entrance):
{{
  "module_type": "entrance",
  "width": <ширина в метрах, 1.5-3.0>,
  "height": <высота в метрах, 2.2-3.5>,
  "depth": <глубина входа в метрах, 0.8-1.5>,
  "color": "<цвет>"
}}

ПРАВИЛА:
1. Определи тип модуля из текста (стена, окно, дверь, балкон, вход)
2. Извлеки все числовые параметры (метры!)
3. Извлеки строковые параметры (цвет, стиль, материал)
4. Если что-то не указано — используй разумные defaults
5. Проверь диапазоны и ограничь значения
6. Верни ТОЛЬКО JSON, ничего больше

Описание модуля: "{text}"
"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "Ты архитектор. Извлекай параметры модулей (стена, окно, дверь, балкон, вход) из текста и возвращай ТОЛЬКО валидный JSON без пояснений."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }

    try:
        logger.info(f"DeepSeek: парсим модуль '{text[:50]}...'")

        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 402:
            logger.warning("DeepSeek: недостаточно средств на балансе (402) — module parse")
            return {}
        if response.status_code != 200:
            logger.error(f"DeepSeek ошибка {response.status_code}: {response.text}")
            return {}

        data = response.json()

        if "choices" not in data or len(data["choices"]) == 0:
            logger.error(f"DeepSeek неожиданный ответ: {data}")
            return {}

        content = data["choices"][0]["message"]["content"]
        logger.info(f"DeepSeek ответ: {content[:100]}...")

        # Ищем JSON в ответе
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                logger.info(f"✓ Параметры извлечены: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка JSON парсинга: {e}")
                return {}

        logger.warning(f"JSON не найден в ответе: {content}")
        return {}

    except requests.exceptions.Timeout:
        logger.error("DeepSeek timeout (30 сек)")
        return {}
    except requests.exceptions.ConnectionError:
        logger.error("DeepSeek ConnectionError")
        return {}
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        return {}


def extract_module_parameters(text: str, module_type: Optional[str] = None) -> dict:
    """
    Обертка над send_module_text_to_deepseek с дополнительной обработкой

    Args:
        text: Описание модуля
        module_type: Подсказка о типе (опционально)

    Returns:
        dict: Параметры модуля с defaults для пропущенных значений
    """

    # Defaults для каждого типа
    defaults_by_type = {
        "wall": {
            "module_type": "wall",
            "height": 3.0,
            "width": 2.0,
            "color": "#888888",
            "thickness": 0.3,
        },
        "window": {
            "module_type": "window",
            "width": 1.5,
            "height": 1.2,
            "frame_color": "#444444",
            "glass_color": "#87CEEB",
        },
        "door": {
            "module_type": "door",
            "height": 2.1,
            "width": 0.9,
            "frame_color": "#8B4513",
        },
        "balcony": {
            "module_type": "balcony",
            "depth": 1.15,
            "width": 2.0,
            "parapet_height": 1.1,
            "color": "#AAAAAA",
        },
        "entrance": {
            "module_type": "entrance",
            "width": 2.0,
            "height": 2.5,
            "depth": 1.0,
            "color": "#CCCCCC",
        },
    }

    # Получаем результаты от DeepSeek
    result = send_module_text_to_deepseek(text, module_type)

    # Определяем тип из результата или используем подсказку
    detected_type = result.get("module_type", module_type or "wall")

    # Получаем defaults для этого типа
    defaults = defaults_by_type.get(detected_type, defaults_by_type["wall"])

    # Объединяем с defaults
    return {**defaults, **result}


def parse_building_text(text: str) -> Optional[dict]:
    """
    Отправляет текст описания ЗДАНИЯ в DeepSeek и возвращает validated dict
    или None при ошибке.
    """
    prompt = f"""You are an architect assistant. A user described a building in natural language (possibly with grammar mistakes or typos).
Correct any errors mentally, then extract building parameters and return ONLY a valid JSON object.

The JSON must have exactly these fields:
{{
  "floors": <integer 1-25>,
  "sections": <integer 1-10, number of entrances/sections>,
  "width": <integer 6-30, number of facade columns>,
  "depth": <integer 1-6>,
  "has_balconies": <boolean>,
  "window_cols": <integer 2-width>,
  "texture_scale": <integer 1-8>
}}

Defaults if not mentioned: floors=9, sections=3, width=18, depth=2, has_balconies=true, window_cols=8, texture_scale=3.
Clamp all values. window_cols <= width. Return ONLY JSON.

User description: "{text}"
"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an architect. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }

    try:
        logger.info(f"DeepSeek: парсим здание '{text[:60]}'")
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)

        if response.status_code == 402:
            logger.warning("DeepSeek: недостаточно средств на балансе (402) — используется regex fallback")
            return None
        if response.status_code != 200:
            logger.error(f"DeepSeek ошибка {response.status_code}: {response.text[:200]}")
            return None

        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info(f"DeepSeek building response: {content[:200]}")

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            logger.warning("JSON не найден в ответе DeepSeek (building)")
            return None

        parsed = json.loads(match.group(0))

        floors = max(1, min(25, int(parsed.get("floors", 9))))
        sections = max(1, min(10, int(parsed.get("sections", 3))))
        width = max(6, min(30, int(parsed.get("width", 18))))
        depth = max(1, min(6, int(parsed.get("depth", 2))))
        has_balconies = bool(parsed.get("has_balconies", True))
        window_cols = max(2, min(width, int(parsed.get("window_cols", 8))))
        texture_scale = max(1, min(8, int(parsed.get("texture_scale", 3))))

        result = {
            "house": {
                "floors": floors,
                "sections": sections,
                "width": width,
                "depth": depth,
                "has_balconies": has_balconies,
                "window_cols": window_cols,
                "facade": {"texture_url": "", "texture_scale": texture_scale},
            }
        }
        logger.info(f"✓ Building params: {result}")
        return result

    except requests.exceptions.Timeout:
        logger.error("DeepSeek timeout (20s) — building parse")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("DeepSeek ConnectionError — building parse")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.error(f"Building parse error: {exc}")
        return None
    except Exception as exc:
        logger.error(f"DeepSeek error: {exc}")
        return None


# ============== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ==============

if __name__ == "__main__":
    print("=" * 70)
    print("ПРИМЕРЫ ПАРСИНГА МОДУЛЕЙ")
    print("=" * 70)

    examples = [
        ("стена высота 3 метра, ширина 2 метра, бетон серый", "wall"),
        ("окно 1.2 метра ширина, 1.5 метра высота, двойное", "window"),
        ("дверь входная 2.1м высота, 0.9м ширина, деревянная", "door"),
        ("балкон глубина 1.5м, ширина 2м, открытый", "balcony"),
        ("подъезд вход 2м ширина, 2.5м высота, стандартный", "entrance"),
    ]

    for text, expected_type in examples:
        print(f"\n{'='*70}")
        print(f"Входной текст: '{text}'")
        print(f"Ожидаемый тип: {expected_type}")
        print(f"{'='*70}")

        result = extract_module_parameters(text, expected_type)
        print(f"Результат: {json.dumps(result, ensure_ascii=False, indent=2)}\n")