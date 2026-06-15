"""
nlp_parser.py — Парсер текста в параметры ОТДЕЛЬНЫХ МОДУЛЕЙ
Извлекает из русского текста параметры стен, окон, дверей, балконов, входов
"""

import re
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
logger = logging.getLogger(__name__)

class ModuleType(str, Enum):
    """Типы модулей"""
    WALL = "wall"
    WINDOW = "window"
    DOOR = "door"
    BALCONY = "balcony"
    ENTRANCE = "entrance"
    ROOF = "roof"


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
            r"стен[аы]|wall",
            r"панел[ь]?|panel",
            r"фасад|facade",
            r"кирпич|brick",
            r"бетон|concrete",
        ],
        ModuleType.WINDOW: [
            r"окн[оа]|window",
            r"стекл[оа]|glass",
            r"окошк[оа]",
        ],
        ModuleType.DOOR: [
            r"дверь|door",
            r"входн[ая]?|entrance",
            r"дверц[аы]",
        ],
        ModuleType.BALCONY: [
            r"балкон|balcony",
            r"лоджи[я]|loggia",
        ],
        ModuleType.ENTRANCE: [
            r"подъезд|entry",
            r"вход\s*в\s*здание",
        ],
        ModuleType.ROOF: [
            r"крыш[аи]|roof",
            r"кровл[яи]|кровель",
            r"двускатн",
            r"пирамид",
        ],
        ModuleType.ROOF: [
            r"крыш[аи]|roof",
            r"плоская\s+крыша|flat\s+roof",
            r"двускатн|gable",
            r"пирамид|pyramid",
            r"кровл[яи]|кровля",
        ],
    }

    # Маппинг цветов на HEX коды
    COLOR_MAP = {
        "красный": "#c74a4a",
        "red": "#c74a4a",
        "синий": "#4a6fc7",
        "синяя": "#4a6fc7",
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
        "розовый": "#ff88ff",
        "розовая": "#ff88ff",
        "pink": "#ff88ff",
    }

    # Дефолты для каждого типа
    DEFAULTS = {
        ModuleType.WALL: {
            "height": 3.0,
            "width": 2.0,
            "color": "#C9B28F",
            "material": "plaster",
            "thickness": 0.3,
        },
        ModuleType.WINDOW: {
            "width": 1.5,
            "height": 1.2,
            "depth": 0.12,
            "style": "double",
            "frame_color": "#5C4A3A",
            "glass_color": "#87CEEB",
        },
        ModuleType.DOOR: {
            "height": 2.1,
            "width": 0.9,
            "depth": 0.08,
            "style": "standard",
            "material": "wood",
            "color": "#6B4A33",
            "frame_color": "#6B4A33",
        },
        ModuleType.BALCONY: {
            "height": 2.15,
            "depth": 1.15,
            "width": 2.0,
            "style": "open",
            "parapat_height": 1.1,
            "color": "#B8B0A8",
            "has_roof": True,
        },
        ModuleType.ENTRANCE: {
            "width": 2.0,
            "height": 2.5,
            "depth": 1.0,
            "style": "standard",
            "color": "#CCCCCC",
        },
        ModuleType.ROOF: {
            "length": 3.0,
            "width": 3.0,
            "height": 0.28,
            "roof_type": "flat",
            "color": "#7A523E",
        },
    }

    def __init__(self):
        """Инициализация парсера с регулярными выражениями"""

        # Регулярные выражения для парсинга параметров
        self.PARAM_PATTERNS = {
            "height": [
                r"(?:height|высот[аы])\s+(\d+(?:[.,]\d+)?)\s*м?(?:етр)?",  # height 1.75, высота 1.75м
                r"(\d+(?:[.,]\d+)?)\s*м?(?:etres?|метр(?:а|ов)?)?\s*(?:high|height|высот[аы])",
                # 1.75м height, 1.75 метра высоты
            ],
            "width": [
                r"(?:width|ширин[аы])\s+(\d+(?:[.,]\d+)?)\s*м?(?:етр)?",  # width 0.3, ширина 0.3м
                r"(\d+(?:[.,]\d+)?)\s*м?(?:etres?|метр(?:а|ов)?)?\s*(?:wide|ширин[аы])",  # 0.3м width, 0.3 метра ширины
            ],
            "depth": [
                r"(?:depth|глубин[аы])\s+(\d+(?:[.,]\d+)?)\s*м?(?:етр)?",  # depth 0.2, глубина 0.2м
                r"(\d+(?:[.,]\d+)?)\s*м?(?:etres?|метр(?:а|ов)?)?\s*(?:deep|глубин[аы])",  # 0.2м depth
            ],
            "color": [
                r"(?:color|цвет)[:]?\s+(#?[a-fA-F0-9]{6})",
                # Не использовать «[ый|ая]» внутри [] — это один символ из класса, а не «ый/ая».
                r"\b((?:красн(?:ый|ая)|син(?:ий|яя)|зелён(?:ый|ая)|зелен(?:ый|ая)|"
                r"жёлт(?:ый|ая)|желт(?:ый|ая)|бел(?:ый|ая)|чёрн(?:ый|ая)|черн(?:ый|ая)|"
                r"сер(?:ый|ая)|оранжев(?:ый|ая)|фиолетов(?:ый|ая)|розов(?:ый|ая)|"
                r"коричнев(?:ый|ая)|red|blue|green|white|black|gray|grey))\b",
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

    def _extract_value(self, text: str, param_name: str) -> float | None:
        """Извлекает число, которое стоит ПРЯМО ПЕРЕД названием параметра"""

        # Регулярные выражения - ищут число ПЕРЕД словом параметра
        patterns = {
            # Ищет: [число] [опционально метры] [height/высота]
            # Но НЕ перед словом width
            "height": r"(\d+(?:\.\d+)?)\s*(?:(?:м|m|meters?)\s+)?(?:high|height|высота)",

            # Ищет: [число] [опционально метры] [width/ширина]
            # Но НЕ перед словом height
            "width": r"(\d+(?:\.\d+)?)\s*(?:(?:м|m|meters?)\s+)?(?:width|ширина)",

            "depth": r"(\d+(?:\.\d+)?)\s*(?:(?:м|m|meters?)\s+)?(?:deep|depth|глубина)",

            "length": r"(\d+(?:\.\d+)?)\s*(?:(?:м|m|meters?)\s+)?(?:length|длина|длин[аы])",
        }

        pattern = patterns.get(param_name)
        if not pattern:
            return None

        # Ищем ВСЕ совпадения и берем ПОСЛЕДНЕЕ
        # (потому что пользователь может написать "height 4, width 2")
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        if matches:
            # Берем последнее совпадение (оно точнее)
            match = matches[-1]
            value = float(match.group(1))
            logger.info(f"🔍 Найдено {param_name}: {value}")
            return value

        logger.info(f"🔍 {param_name} не найден в тексте: '{text}'")
        return None

    def _extract_bool(self, text: str, positive_pattern: str, negative_pattern: str) -> Optional[bool]:
        """
        Extract boolean. Negative pattern takes precedence over positive.
        Returns True, False, or None if neither matches.
        """
        if re.search(negative_pattern, text, re.IGNORECASE):
            return False
        if re.search(positive_pattern, text, re.IGNORECASE):
            return True
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
            # Для окна: width, height, depth, style, color (рама)
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

            color_raw = self._extract_string(text, "color")
            if color_raw:
                hx = self._get_color_hex(color_raw)
                if hx:
                    params["frame_color"] = hx
                    params["color"] = hx

            if re.search(r"blue\s+glass|glass\s+blue|голуб", text, re.IGNORECASE):
                params["glass_color"] = "#87CEEB"

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
            # Для балкона: height, depth, width, style, color, has_roof
            depth = self._extract_value(text, "depth")
            width = self._extract_value(text, "width")
            height = self._extract_value(text, "height")

            logger.info(f"[nlp_parser] balcony extract: height={height} depth={depth} width={width} (text='{text}')")

            if height is not None:
                params["height"] = height
            if depth is not None:
                params["depth"] = depth
            if width is not None:
                params["width"] = width

            style = self._extract_string(text, "style")
            if style:
                params["style"] = self._normalize_style(style)

            color_raw = self._extract_string(text, "color")
            if color_raw:
                hx = self._get_color_hex(color_raw)
                if hx:
                    params["color"] = hx

            has_roof = self._extract_bool(
                text,
                positive_pattern=r"(?:with\s+)?(?:крыш[аи]|потолок|roof|ceiling)\b",
                negative_pattern=r"(?:without|no|без)\s+(?:крыш[иеа]|потолк[аеу]|roof|ceiling)",
            )
            if has_roof is not None:
                params["has_roof"] = has_roof

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

        elif module_type == ModuleType.ROOF:
            length = self._extract_value(text, "length") or self._extract_value(text, "width")
            width = self._extract_value(text, "width")
            height = self._extract_value(text, "height")
            depth = self._extract_value(text, "depth")

            if length is not None:
                params["length"] = length
            if width is not None:
                params["width"] = width
            if depth is not None and width is None:
                params["width"] = depth
            if height is not None:
                params["height"] = height

            t = text.lower()
            if re.search(r"плоска|flat|горизонтал|slab|плит", t):
                params["roof_type"] = "flat"
            elif re.search(r"пирамид|pyramid|четырёхскат|четырехскат|hip", t):
                params["roof_type"] = "pyramid"
            elif re.search(r"двускатн|gable|конёк|конек|triangle|треугольн|shed", t):
                params["roof_type"] = "gable"

            color_raw = self._extract_string(text, "color")
            if color_raw:
                hx = self._get_color_hex(color_raw)
                if hx:
                    params["color"] = hx

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


class BuildingTextParser:
    """Regex-парсер описания ЗДАНИЯ — используется как fallback если AI недоступен."""

    def parse(self, text: str) -> dict:
        t = text.lower()

        def _int(pattern, default):
            m = re.search(pattern, t)
            return int(m.group(1)) if m else default

        def _float(pattern, default):
            m = re.search(pattern, t)
            return float(m.group(1).replace(",", ".")) if m else default

        floors = _int(r"(\d+)\s*(?:этаж|floor|storey|story)", 9)
        sections = _int(r"(\d+)\s*(?:секци|подъезд|entranc|section)", 3)
        width = _int(r"(?:ширин[аы]|width)\s*(\d+)", 18)
        depth = _int(r"(?:глубин[аы]|depth)\s*(\d+)", 2)
        window_cols = _int(r"(\d+)\s*(?:окон|window)", 8)
        balcony_rate = _float(r"балкон[ыа]?\s*(\d+(?:[.,]\d+)?)", 0.3)
        if balcony_rate > 1.0:
            balcony_rate /= 100.0
        has_balconies = bool(
            re.search(r"балкон|balcon", t) and
            not re.search(r"без\s*балкон|no\s*balcon", t)
        )

        return {
            "house": {
                "floors": max(1, min(floors, 25)),
                "sections": max(1, min(sections, 10)),
                "width": max(6, min(width, 30)),
                "depth": max(1, min(depth, 6)),
                "has_balconies": has_balconies,
                "balcony_rate": round(max(0.0, min(balcony_rate, 1.0)), 2),
                "window_cols": max(2, min(window_cols, width)),
                "facade": {"texture_url": "", "texture_scale": 3},
            }
        }


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