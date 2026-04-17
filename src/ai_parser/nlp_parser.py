"""
nlp_parser.py — Парсер текста в численные параметры
Извлекает из русского текста параметры дома (этажи, размеры, окна, балконы и т.д.)
"""

import re
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class BuildingParams:
    """Параметры здания, извлеченные из текста"""
    floors: int
    wall_height: float
    building_length: float
    building_width: float
    windows_per_floor: int
    balconies_per_floor: int
    entrance_count: int
    balcony_depth: float
    balcony_width: float = 2.0
    parapet_height: float = 1.1
    door_height: float = 2.1
    door_width: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BuildingTextParser:
    """Парсер текста описания дома"""

    # Значения по умолчанию
    DEFAULTS = {
        "floors": 5,
        "wall_height": 3.0,
        "building_length": 25.0,
        "building_width": 12.0,
        "windows_per_floor": 4,
        "balconies_per_floor": 2,
        "entrance_count": 1,
        "balcony_depth": 1.15,
        "balcony_width": 2.0,
        "parapet_height": 1.1,
        "door_height": 2.1,
        "door_width": 0.9,
    }

    # Диапазоны допустимых значений
    RANGES = {
        "floors": (1, 25),
        "wall_height": (2.5, 5.0),
        "building_length": (10.0, 100.0),
        "building_width": (8.0, 40.0),
        "windows_per_floor": (1, 12),
        "balconies_per_floor": (0, 6),
        "entrance_count": (1, 4),
        "balcony_depth": (0.8, 2.0),
        "balcony_width": (1.5, 4.0),
        "parapet_height": (0.9, 1.5),
        "door_height": (2.0, 2.3),
        "door_width": (0.8, 1.0),
    }

    def __init__(self):
        """Инициализация парсера с регулярными выражениями"""
        self.patterns = {
            "floors": [
                r"(\d+)\s*этажа?(?:х)?",  # 3 этажа, 5 этажей
                r"(\d+)\s*-?\s*эт(?:аж)?",  # 3-этажный
                r"(\d+)\s*уровн[ей]?",  # 3 уровня
            ],
            "wall_height": [
                r"стены?\s+(\d+(?:[,.]\d+)?)\s*м(?:етр)?",  # стены 3м, стены 3.5м
                r"высот[аы]\s+(?:стен)?\s*(\d+(?:[,.]\d+)?)\s*м",  # высота стен 3м
                r"(?:высот[аы])?\s+(\d+(?:[,.]\d+)?)\s*м\s+(?:стен)?",  # 3м стен
            ],
            "building_length": [
                r"длин[аы]\s+(\d+(?:[,.]\d+)?)\s*м(?:етр)?",  # длина 25м
                r"протяжен[ность]?\s+(\d+(?:[,.]\d+)?)\s*м",  # протяженность 25м
                r"(?:здани)?[яе]\s+(\d+(?:[,.]\d+)?)\s*м\s+в\s+длин(?:у|ы)?",  # здания 25м в длину
            ],
            "building_width": [
                r"ширин[аы]\s+(\d+(?:[,.]\d+)?)\s*м(?:етр)?",  # ширина 12м
                r"(?:здани)?[яе]\s+(\d+(?:[,.]\d+)?)\s*м\s+в\s+ширин(?:у|ы)?",  # здания 12м в ширину
            ],
            "windows_per_floor": [
                r"(\d+)\s*окн[аами]?(?:\s+на\s+этаж)?",  # 4 окна, 4 окна на этаж
                r"по\s+(\d+)\s*окн[аами]?",  # по 4 окна
            ],
            "balconies_per_floor": [
                r"(\d+)\s*балкон[ов]?(?:\s+на\s+этаж)?",  # 2 балкона, 2 балкона на этаж
                r"по\s+(\d+)\s*балкон[ов]?",  # по 2 балкона
            ],
            "entrance_count": [
                r"(\d+)\s*подъезд[ов]?",  # 2 подъезда
                r"(\d+)\s*входо?в?",  # 2 входа
                r"(\d+)\s*раздел[а]?",  # 2 раздела
            ],
            "balcony_depth": [
                r"балкон[ы]?\s+(\d+(?:[,.]\d+)?)\s*м(?:етр)?(?:\s+(?:в|глубин))?",  # балконы 1.2м
                r"глубин[аы]\s+балкон[а]?\s+(\d+(?:[,.]\d+)?)\s*м",  # глубина балкона 1.2м
            ],
        }

    def _normalize_number(self, text: str) -> float:
        """Преобразует строку с числом в float (заменяет запятую на точку)"""
        return float(text.replace(",", "."))

    def _extract_value(self, text: str, param_name: str) -> Optional[float]:
        """Извлекает значение параметра из текста по регулярным выражениям"""
        if param_name not in self.patterns:
            return None

        for pattern in self.patterns[param_name]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = self._normalize_number(match.group(1))
                    return value
                except (ValueError, IndexError):
                    continue

        return None

    def _clamp_value(self, param_name: str, value: float) -> float:
        """Ограничивает значение диапазоном допустимых значений"""
        if param_name not in self.RANGES:
            return value

        min_val, max_val = self.RANGES[param_name]
        return max(min_val, min(value, max_val))

    def parse(self, text: str) -> BuildingParams:
        """
        Парсит текст и возвращает параметры здания

        Args:
            text: Описание дома на русском языке

        Returns:
            BuildingParams: Объект с извлеченными параметрами
        """

        params = {}

        # Извлекаем каждый параметр
        for param_name in self.DEFAULTS.keys():
            value = self._extract_value(text, param_name)

            if value is not None:
                # Ограничиваем диапазоном и применяем целочисленность если нужно
                value = self._clamp_value(param_name, value)
                if param_name in ["floors", "windows_per_floor", "balconies_per_floor", "entrance_count"]:
                    value = int(value)
                params[param_name] = value
            else:
                params[param_name] = self.DEFAULTS[param_name]

        return BuildingParams(**params)

    def debug_parse(self, text: str) -> Dict[str, Any]:
        """
        Парсит текст с подробной информацией о том, что было найдено
        Полезно для отладки
        """
        print(f"\n📝 Парсинг текста: '{text}'\n")

        result = {}
        found_params = []

        for param_name in self.DEFAULTS.keys():
            value = self._extract_value(text, param_name)

            if value is not None:
                original_value = value
                value = self._clamp_value(param_name, value)

                if param_name in ["floors", "windows_per_floor", "balconies_per_floor", "entrance_count"]:
                    value = int(value)

                found_params.append(f"✓ {param_name}: {original_value} → {value}")
                result[param_name] = value
            else:
                default = self.DEFAULTS[param_name]
                found_params.append(f"○ {param_name}: используется дефолт {default}")
                result[param_name] = default

        print("Найденные параметры:")
        for msg in found_params:
            print(f"  {msg}")
        print()

        return result


# ============== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ==============

if __name__ == "__main__":
    parser = BuildingTextParser()

    # Пример 1: Простой текст
    text1 = "3 этажа, 4 окна, 2 балкона по 1.2м, стены 3м"
    params1 = parser.parse(text1)
    print("Пример 1:")
    print(f"  Входной текст: '{text1}'")
    print(f"  Результат: {params1.to_dict()}\n")

    # Пример 2: С ошибками (заведомо неправильный текст)
    text2 = "дом с 5 окнов и 2 балконов на 4 этажа высотой 3.5 метров"
    params2 = parser.parse(text2)
    print("Пример 2:")
    print(f"  Входной текст: '{text2}'")
    print(f"  Результат: {params2.to_dict()}\n")

    # Пример 3: С деталями
    text3 = "5 этажей, ширина 15м, длина 30м, по 6 окон на этаж, 3 балкона по 1.5м"
    params3 = parser.parse(text3)
    print("Пример 3:")
    print(f"  Входной текст: '{text3}'")
    print(f"  Результат: {params3.to_dict()}\n")

    # Пример с отладкой
    print("=" * 60)
    print("ОТЛАДКА ПАРСЕРА:")
    print("=" * 60)
    parser.debug_parse(text1)
    parser.debug_parse(text2)