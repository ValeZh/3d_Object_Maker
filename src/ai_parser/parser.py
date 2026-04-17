"""
src/ai_parser/parser.py — DeepSeek интеграция для домов
Переписано для извлечения параметров ДОМОВ (этажи, окна, балконы и т.д.)
вместо объектов (кубики, цвет, текстура)
"""

import requests
import json
import re
import logging

logger = logging.getLogger(__name__)

# ⚠️ ВАЖНО: Замени на твой реальный API ключ
DEEPSEEK_API_KEY = "sk-7f5f5e5858b64a8d8d6b62bad95938e9"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


def send_text_to_deepseek(text: str) -> dict:
    """
    Отправляет текст описания ДОМА в DeepSeek и извлекает параметры здания.

    Параметры извлекаются:
    - floors: количество этажей
    - wall_height: высота одного этажа (м)
    - building_length: длина здания (м)
    - building_width: ширина здания (м)
    - windows_per_floor: количество окон на этаж
    - balconies_per_floor: количество балконов на этаж
    - entrance_count: количество входов/подъездов
    - balcony_depth: глубина балконов (м)
    - window_width: ширина окна (м)
    - window_height: высота окна (м)
    - door_height: высота двери (м)
    - door_width: ширина двери (м)

    Args:
        text: Описание дома на русском языке

    Returns:
        dict: Извлеченные параметры дома или пустой dict при ошибке
    """

    # Новый промпт для домов (вместо старого для объектов)
    prompt = f"""
Ты архитектор. Извлеки параметры ЖИЛОГО ДОМА из описания на русском языке.

Ищи эти параметры и верни ТОЛЬКО JSON (без пояснений):
{{
  "floors": <число этажей, 1-25>,
  "wall_height": <высота этажа в метрах, 2.5-5.0>,
  "building_length": <длина дома в метрах, 10-100>,
  "building_width": <ширина дома в метрах, 8-40>,
  "windows_per_floor": <окон на этаж, 1-12>,
  "balconies_per_floor": <балконов на этаж, 0-6>,
  "entrance_count": <входов/подъездов, 1-4>,
  "balcony_depth": <глубина балконов в метрах, 0.8-2.0>,
  "window_width": <ширина окна в метрах, 1.0-2.0>,
  "window_height": <высота окна в метрах, 0.8-1.5>,
  "door_height": <высота двери в метрах, 2.0-2.3>,
  "door_width": <ширина двери в метрах, 0.8-1.0>,
  "description": "<краткое описание что получилось>"
}}

ПРАВИЛА:
1. Если что-то не указано — используй разумные defaults (5 этажей, 3м высота, 4 окна и т.д.)
2. Все числа — метры
3. Проверь диапазоны и ограничь значения
4. Исправь ошибки в русском языке если есть (дома, а не домов)
5. Верни ТОЛЬКО JSON, ничего больше

Описание дома: "{text}"
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
                "content": "Ты архитектор. Извлекай параметры домов из текста и возвращай ТОЛЬКО валидный JSON без пояснений."
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
        logger.info(f"Отправляем запрос к DeepSeek: '{text[:50]}...'")

        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"DeepSeek вернул статус {response.status_code}: {response.text}")
            return {}

        data = response.json()

        if "choices" not in data or len(data["choices"]) == 0:
            logger.error(f"Неожиданный ответ DeepSeek: {data}")
            return {}

        content = data["choices"][0]["message"]["content"]
        logger.info(f"Ответ DeepSeek: {content[:100]}...")

        # Ищем JSON в ответе
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                logger.info(f"Успешно извлечены параметры: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга JSON: {e}")
                return {}

        logger.warning(f"JSON не найден в ответе: {content}")
        return {}

    except requests.exceptions.Timeout:
        logger.error("Timeout: DeepSeek не ответил в течение 30 секунд")
        return {}
    except requests.exceptions.ConnectionError:
        logger.error("ConnectionError: Не удалось подключиться к DeepSeek API")
        return {}
    except Exception as e:
        logger.error(f"Ошибка при обращении к DeepSeek: {e}")
        return {}


def extract_building_parameters(text: str) -> dict:
    """
    Обертка над send_text_to_deepseek с дополнительной обработкой

    Args:
        text: Описание дома

    Returns:
        dict: Параметры дома с defaults для пропущенных значений
    """

    # Defaults для всех параметров
    defaults = {
        "floors": 5,
        "wall_height": 3.0,
        "building_length": 25.0,
        "building_width": 12.0,
        "windows_per_floor": 4,
        "balconies_per_floor": 2,
        "entrance_count": 1,
        "balcony_depth": 1.15,
        "window_width": 1.5,
        "window_height": 1.2,
        "door_height": 2.1,
        "door_width": 0.9,
        "description": "Жилой многоэтажный дом"
    }

    # Получаем результаты от DeepSeek
    result = send_text_to_deepseek(text)

    # Объединяем с defaults
    return {**defaults, **result}


if __name__ == "__main__":
    # Примеры для тестирования

    print("=" * 60)
    print("ТЕСТ 1: Простое описание")
    print("=" * 60)

    text1 = "3 этажа, 4 окна на каждый этаж, 2 балкона, стены 3 метра высотой"
    result1 = extract_building_parameters(text1)
    print(f"Входной текст: '{text1}'")
    print(f"Результат: {json.dumps(result1, ensure_ascii=False, indent=2)}\n")

    print("=" * 60)
    print("ТЕСТ 2: Детальное описание")
    print("=" * 60)

    text2 = """
    5-этажное жилое здание на углу с протяженностью 30 метров.
    По 6 окон на фасаде каждого этажа.
    3 балкона с каждой стороны на всех этажах.
    Высота одного этажа 3.2 метра.
    2 входа в подъезды.
    Балконы глубиной 1.3 метра.
    """
    result2 = extract_building_parameters(text2)
    print(f"Входной текст: '{text2}'")
    print(f"Результат: {json.dumps(result2, ensure_ascii=False, indent=2)}\n")

    print("=" * 60)
    print("ТЕСТ 3: Минимальное описание")
    print("=" * 60)

    text3 = "4 этажа"
    result3 = extract_building_parameters(text3)
    print(f"Входной текст: '{text3}'")
    print(f"Результат: {json.dumps(result3, ensure_ascii=False, indent=2)}\n")