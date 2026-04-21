"""
nlp_parser.py — Парсер текста в параметры ОТДЕЛЬНЫХ МОДУЛЕЙ
Извлекает из русского текста параметры стен, окон, дверей, балконов, входов
"""

import re
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class ModuleType(str, Enum):
    """Типы модулей"""
    WALL = "wall"
    WINDOW = "window"
    DOOR = "door"
    BALCONY = "balcony"
    ENTRANCE = "entrance"


@dataclass
class ModuleParams:
    """Параметры модуля"""
    module_type: ModuleType
    module_name: str
    params: Dict[str, Any]
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_type": self.module_type.value,
            "module_name": self.module_name,
            "params": self.params,
            "confidence": self.confidence
        }


class ModuleTextParser:
    """Парсер текста для модулей"""

    # Определение типов по ключевым словам
    MODULE_TYPE_PATTERNS = {
        ModuleType.WALL: [
            r"стен[аы]",
            r"панел[ь]?",
            r"фасад",
            r"кирпич",
            r"бетон",
        ],
        ModuleType.WINDOW: [
            r"окн[оа]",
            r"стекл[оа]",
            r"окошк[оа]",
        ],
        ModuleType.DOOR: [
            r"дверь",
            r"входн[ая]?",
            r"дверц[аы]",
        ],
        ModuleType.BALCONY: [
            r"балкон",
            r"лоджи[я]",
        ],
        ModuleType.ENTRANCE: [
            r"подъезд",
            r"входн[ая]?",
        ],
    }

    # Маппинг цветов на HEX коды
    COLOR_MAP = {
        "красный": "#c74a4a",
        "red": "#c74a4a",
        "синий": "#4a6fc7",
        "blue": "#4a6fc7",
        "зелёный": "#4aa36c",
        "зеленый": "#4aa36c",
        "green": "#4aa36c",
        "серый": "#888888",
        "серая": "#888888",
        "grey": "#888888",
        "gray": "#888888",
        "белый": "#d9d9d9",
        "белая": "#d9d9d9",
        "white": "#d9d9d9",
        "чёрный": "#2a2a2a",
        "черный": "#2a2a2a",
        "чёрная": "#2a2a2a",
        "черная": "#2a2a2a",
        "black": "#2a2a2a",
        "коричневый": "#8b6a4e",
        "коричневая": "#8b6a4e",
        "brown": "#8b6a4e",
        "бежевый": "#c9b28f",
        "бежевая": "#c9b28f",
        "beige": "#c9b28f",
        "жёлтый": "#f0ad4e",
        "желтый": "#f0ad4e",
        "жёлтая": "#f0ad4e",
        "желтая": "#f0ad4e",
        "yellow": "#f0ad4e",
        "оранжевый": "#ff9800",
        "оранжевая": "#ff9800",
        "orange": "#ff9800",
        "фиолетовый": "#9c27b0",
        "фиолетовая": "#9c27b0",
        "purple": "#9c27b0",
    }

    # Дефолты для каждого типа
    DEFAULTS = {
        ModuleType.WALL: {
            "height": 3.0,
            "width": 2.0,
            "color": "#888888",
            "material": "concrete",
            "thickness": 0.3,
        },
        ModuleType.WINDOW: {
            "width": 1.5,
            "height": 1.2,
            "style": "double",
            "frame_color": "#444444",
            "glass_color": "#87CEEB",
        },
        ModuleType.DOOR: {
            "height": 2.1,
            "width": 0.9,
            "style": "standard",
            "material": "wood",
            "frame_color": "#8B4513",
        },
        ModuleType.BALCONY: {
            "depth": 1.15,
            "width": 2.0,
            "style": "open",
            "parapat_height": 1.1,
            "color": "#AAAAAA",
        },
        ModuleType.ENTRANCE: {
            "width": 2.0,
            "height": 2.5,
            "depth": 1.0,
            "style": "standard",
            "color": "#CCCCCC",
        },
    }

    def __init__(self):
        """Инициализация парсера с регулярными выражениями"""

        # Регулярные выражения для парсинга параметров
        self.PARAM_PATTERNS = {
            "height": [
                r"(\d+(?:[.,]\d+)?)\s*(?:м(?:етр)?(?:ов?)?)?\s*высот[аы]",  # 4 высоты или 4м высоты
                r"высот[аы]\s+(\d+(?:[.,]\d+)?)\s*м?(?:етр)?",  # высота 4 или высота 4м
            ],
            "width": [
                r"(\d+(?:[.,]\d+)?)\s*(?:м(?:етр)?(?:ов?)?)?\s*ширин[аы]",  # 1 ширины или 1м ширины
                r"ширин[аы]\s+(\d+(?:[.,]\d+)?)\s*м?(?:етр)?",  # ширина 1 или ширина 1м
            ],
            "depth": [
                r"(\d+(?:[.,]\d+)?)\s*(?:м(?:етр)?(?:ов?)?)?\s*глубин[аы]",
                r"глубин[аы]\s+(\d+(?:[.,]\d+)?)\s*м?(?:етр)?",
            ],
            "color": [
                r"цвет[:]?\s+(#?[a-fA-F0-9]{6}|[а-яА-Я]+)",  # hex или название
                r"(белый|черный|серый|красный|синий|зеленый|желтый|коричневый|оранжевый|фиолетовый|белая|красная|черная|синяя|зеленая|зелёная|желтая)",
            ],
        }

    def _normalize_number(self, text: str) -> float:
        """Преобразует строку с числом в float (заменяет запятую на точку)"""
        return float(text.replace(",", "."))

    def _detect_module_type(self, text: str) -> ModuleType:
        """Определяет тип модуля по тексту"""
        text_lower = text.lower()

        # Подсчитываем совпадения для каждого типа
        scores = {}
        for module_type, patterns in self.MODULE_TYPE_PATTERNS.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            scores[module_type] = score

        # Возвращаем тип с наибольшим количеством совпадений
        if max(scores.values()) == 0:
            # Если ничего не найдено, по умолчанию стена
            return ModuleType.WALL

        return max(scores, key=scores.get)

    def _extract_value(self, text: str, param_name: str) -> Optional[float]:
        """Извлекает численное значение параметра"""
        if param_name not in self.PARAM_PATTERNS:
            return None

        for pattern in self.PARAM_PATTERNS[param_name]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = self._normalize_number(match.group(1))
                    return value
                except (ValueError, IndexError):
                    continue

        return None

    def _extract_string(self, text: str, param_name: str) -> Optional[str]:
        """Извлекает строковое значение параметра"""
        if param_name not in self.PARAM_PATTERNS:
            return None

        for pattern in self.PARAM_PATTERNS[param_name]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1).lower()
                    return value
                except (ValueError, IndexError):
                    continue

        return None

    def _get_color_hex(self, color_name: str) -> str:
        """Преобразует название цвета в HEX код"""
        if not color_name:
            return None

        color_lower = color_name.lower().strip()

        # Если уже HEX код
        if color_lower.startswith("#"):
            return color_lower

        # Ищем в маппинге
        if color_lower in self.COLOR_MAP:
            return self.COLOR_MAP[color_lower]

        # Ищем частичное совпадение
        for key, hex_code in self.COLOR_MAP.items():
            if key in color_lower or color_lower in key:
                return hex_code

        return None

    def _normalize_style(self, style: str) -> str:
        """Нормализует значение style"""
        if not style:
            return None

        style = style.lower().strip()

        # Маппинг синонимов
        style_map = {
            "одинарн": "single",
            "одно": "single",
            "двойн": "double",
            "двухи": "double",
            "стандарт": "standard",
            "обычн": "standard",
            "модерн": "modern",
            "открыт": "open",
            "закрыт": "enclosed",
            "остекл": "enclosed",
            "стекл": "glass",
        }

        for key, val in style_map.items():
            if key in style:
                return val

        return style

    def parse(self, text: str) -> ModuleParams:
        """
        Парсит текст описания модуля и возвращает параметры

        Args:
            text: Описание модуля на русском языке

        Returns:
            ModuleParams: Объект с извлеченными параметрами
        """

        # Определяем тип модуля
        module_type = self._detect_module_type(text)

        # Получаем defaults для этого типа
        defaults = self.DEFAULTS[module_type].copy()

        # Парсим параметры в зависимости от типа
        params = {}

        if module_type == ModuleType.WALL:
            # Для стены: height, width, color, material
            height = self._extract_value(text, "height")
            width = self._extract_value(text, "width")

            if height is not None:
                params["height"] = height
            if width is not None:
                params["width"] = width

            color = self._extract_string(text, "color")
            if color:
                hex_color = self._get_color_hex(color)
                if hex_color:
                    params["color"] = hex_color

            material = self._extract_string(text, "material")
            if material:
                params["material"] = material

        elif module_type == ModuleType.WINDOW:
            # Для окна: width, height, style
            width = self._extract_value(text, "width")
            height = self._extract_value(text, "height")

            if width is not None:
                params["width"] = width
            if height is not None:
                params["height"] = height

            style = self._extract_string(text, "style")
            if style:
                params["style"] = self._normalize_style(style)

        elif module_type == ModuleType.DOOR:
            # Для двери: height, width, style, material
            height = self._extract_value(text, "height")
            width = self._extract_value(text, "width")

            if height is not None:
                params["height"] = height
            if width is not None:
                params["width"] = width

            style = self._extract_string(text, "style")
            if style:
                params["style"] = self._normalize_style(style)

            material = self._extract_string(text, "material")
            if material:
                params["material"] = material

        elif module_type == ModuleType.BALCONY:
            # Для балкона: depth, width, style
            depth = self._extract_value(text, "depth")
            width = self._extract_value(text, "width")

            if depth is not None:
                params["depth"] = depth
            if width is not None:
                params["width"] = width

            style = self._extract_string(text, "style")
            if style:
                params["style"] = self._normalize_style(style)

        elif module_type == ModuleType.ENTRANCE:
            # Для входа: width, height, depth, style
            width = self._extract_value(text, "width")
            height = self._extract_value(text, "height")
            depth = self._extract_value(text, "depth")

            if width is not None:
                params["width"] = width
            if height is not None:
                params["height"] = height
            if depth is not None:
                params["depth"] = depth

            style = self._extract_string(text, "style")
            if style:
                params["style"] = self._normalize_style(style)

        # Объединяем с defaults
        final_params = {**defaults, **params}

        # Создаем имя модуля
        module_name = f"{module_type.value}_{len(str(final_params).encode()) % 10000}"

        # Вычисляем confidence (найдено ли хоть что-то)
        found_count = sum(1 for k, v in params.items() if v is not None)
        confidence = min(1.0, 0.5 + (found_count * 0.15))

        return ModuleParams(
            module_type=module_type,
            module_name=module_name,
            params=final_params,
            confidence=confidence
        )

    def _get_color_hex(self, color_name: str) -> str:
        """Преобразует название цвета в HEX код"""
        if not color_name:
            return None

        color_lower = color_name.lower().strip()

        if color_lower.startswith("#"):
            return color_lower

        COLOR_MAP = {
            "красный": "#c74a4a",
            "синий": "#4a6fc7",
            "зелёный": "#4aa36c",
            "зеленый": "#4aa36c",
            "серый": "#888888",
            "белый": "#d9d9d9",
            "чёрный": "#2a2a2a",
            "черный": "#2a2a2a",
            "коричневый": "#8b6a4e",
            "оранжевый": "#ff9800",
            "фиолетовый": "#9c27b0",
        }

        return COLOR_MAP.get(color_lower, None)

    def debug_parse(self, text: str) -> Dict[str, Any]:
        """Парсит с подробной информацией для отладки"""
        print(f"\n📝 Парсинг модуля: '{text}'\n")

        result = self.parse(text)

        print(f"✓ Тип модуля: {result.module_type.value}")
        print(f"✓ Уверенность: {result.confidence:.0%}")
        print(f"✓ Параметры:")
        for key, val in result.params.items():
            print(f"    {key}: {val}")
        print()

        return result.to_dict()


# ============== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ==============

if __name__ == "__main__":
    parser = ModuleTextParser()

    examples = [
        "стена 3 метра высоты и 2 метра ширины, красная",
        "окно 1.2м ширина и 1.5м высота, двойное",
        "дверь входная 2.1м высота, 0.9м ширина, деревянная",
        "балкон 1.5м глубина, 2м ширина, открытый",
        "подъезд 2м ширина, 2.5м высота",
    ]

    print("=" * 70)
    print("ПРИМЕРЫ ПАРСИНГА МОДУЛЕЙ")
    print("=" * 70)

    for text in examples:
        parser.debug_parse(text)