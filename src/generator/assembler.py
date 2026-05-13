"""
assembler.py — Сборка панельного дома из модульной сетки фасадов
ИСПРАВЛЕНИЕ: Правильная ориентация модулей и координатная система
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from enum import Enum
import trimesh
import numpy as np

logger = logging.getLogger(__name__)


class FacadeType(Enum):
    """Типы фасадных модулей"""
    EMPTY_WALL = "wall"
    WINDOW = "window"
    BALCONY = "balcony"
    ENTRANCE = "door"


class FacadeGrid:
    """2D сетка фасада (columns × floors)"""

    def __init__(self, columns: int, floors: int):
        self.columns = columns
        self.floors = floors
        # grid[floor][column] = FacadeType
        self.grid = [[FacadeType.EMPTY_WALL for _ in range(columns)] for _ in range(floors)]

    def set_cell(self, column: int, floor: int, facade_type: FacadeType):
        """Установить тип модуля в ячейку сетки"""
        if 0 <= column < self.columns and 0 <= floor < self.floors:
            self.grid[floor][column] = facade_type


class PanelBuildingAssembler:
    """Собирает панельный дом из модульной сетки"""

    def __init__(self, params: Dict[str, Any], models_dir: Path):
        """
        Args:
            params: {
                floors: int,
                columns: int,
                sections: int,
                module_width: float (default 4.0m),
                module_height: float (default 3.0m),
                depth: float (modules in depth, default 2),
                ...
            }
            models_dir: Папка с модулями (modules/)
        """
        self.params = params
        self.models_dir = Path(models_dir)

        # === ПАРАМЕТРЫ ЗДАНИЯ ===
        self.floors = params.get("floors", 5)
        self.columns = params.get("columns", 18)
        self.sections = params.get("sections", 3)
        self.module_width = params.get("module_width", 4.0)
        self.module_height = params.get("module_height", 3.0)
        self.depth = params.get("depth", 2)

        # === РАЗМЕРЫ ЗДАНИЯ ===
        self.building_width = self.columns * self.module_width
        self.building_height = self.floors * self.module_height
        self.building_depth = self.depth * self.module_width

        # === СЕТКИ ФАСАДОВ ===
        self.front_grid = FacadeGrid(self.columns, self.floors)
        self.back_grid = FacadeGrid(self.columns, self.floors)
        self.left_grid = FacadeGrid(self.depth, self.floors)
        self.right_grid = FacadeGrid(self.depth, self.floors)

        # === ЗАГРУЖЕННЫЕ МОДУЛИ ===
        self.modules = {}

        logger.info(f"🏗️ Инициализация здания:")
        logger.info(f"   Размер: {self.columns}×{self.floors} (модули)")
        logger.info(f"   Размеры: {self.building_width:.1f}m × {self.building_height:.1f}m × {self.building_depth:.1f}m")
        logger.info(f"   Секций: {self.sections}")

    def load_modules(self) -> bool:
        """Загружает все модули из директории"""
        logger.info("📥 Загрузка модулей...")

        for facade_type in FacadeType:
            mesh = self._load_module(facade_type.value)
            if mesh is not None:
                self.modules[facade_type] = mesh
                logger.info(f"   ✓ {facade_type.name}: загружен")
            else:
                logger.warning(f"   ⚠️ {facade_type.name}: не найден")

        if not self.modules:
            logger.error("❌ Не загружено ни одного модуля!")
            return False

        return True

    def _load_module(self, module_type: str) -> trimesh.Trimesh:
        """Загружает модуль и исправляет ориентацию"""
        module_dir = self.models_dir / module_type

        if not module_dir.exists():
            logger.warning(f"⚠️ Папка не найдена: {module_dir}")
            return None

        obj_files = list(module_dir.glob(f"*/{module_type}.obj"))

        if not obj_files:
            logger.warning(f"⚠️ OBJ файлы не найдены в: {module_dir}")
            return None

        try:
            mesh = trimesh.load(str(obj_files[0]), process=False)
            logger.info(f"✓ Загружен {module_type}: {obj_files[0]}")

            try:
                bounds = mesh.bounds
                size_x = bounds[1][0] - bounds[0][0]
                size_y = bounds[1][1] - bounds[0][1]
                size_z = bounds[1][2] - bounds[0][2]

                logger.info(f"   Размеры OBJ: X={size_x:.2f}m, Y={size_y:.2f}m, Z={size_z:.2f}m")

                # ИСПРАВЛЯЕМ ОРИЕНТАЦИЮ если модуль на боку
                if size_z < 0.5 and size_y > 0.5:
                    logger.info(f"   ⚠️ Модуль {module_type} перевернут, исправляем...")
                    rotate_matrix = trimesh.transformations.rotation_matrix(
                        np.pi / 2, [1, 0, 0]  # +90° вокруг X
                    )
                    mesh.apply_transform(rotate_matrix)
                    bounds = mesh.bounds
                    size_x = bounds[1][0] - bounds[0][0]
                    size_y = bounds[1][1] - bounds[0][1]
                    size_z = bounds[1][2] - bounds[0][2]
                    logger.info(f"   ✓ После исправления: X={size_x:.2f}m, Y={size_y:.2f}m, Z={size_z:.2f}m")

                # Масштабируем
                if size_x < 0.01 or size_z < 0.01:
                    logger.warning(f"⚠️ {module_type} имеет размер близкий к нулю")
                    return mesh

                scale_x = self.module_width / size_x
                scale_z = self.module_height / size_z

                logger.info(f"   Масштабирование: X={scale_x:.2f}, Z={scale_z:.2f}")

                # Масштабируем правильно (вокруг центра)
                center = mesh.centroid
                mesh.apply_translation(-center)

                scale_matrix = np.diag([scale_x, 1.0, scale_z, 1.0])
                mesh.apply_transform(scale_matrix)

                mesh.apply_translation(center)

                logger.info(f"   ✓ {module_type.upper()} масштабирован успешно")

            except Exception as scale_error:
                logger.error(f"❌ Ошибка масштабирования {module_type}: {scale_error}")

            return mesh

        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {module_type}: {e}", exc_info=True)
            return None

    def generate_facade_rules(self):
        """Применяет правила размещения модулей в сетке"""
        logger.info("📐 Применение правил фасада...")

        scale = self.params.get("texture_scale", 1)
        scale = max(1, min(scale, 8))

        wall_prob = max(0, 0.7 - (scale - 1) * 0.087)
        balcony_prob = 0.1 + (scale - 1) * 0.037

        logger.info(f"   Texture Scale: {scale}")
        logger.info(
            f"   Вероятности - Стены: {wall_prob:.1%}, Балконы: {balcony_prob:.1%}, Окна: {1 - wall_prob - balcony_prob:.1%}")

        # === РАСЧЕТ ВХОДОВ ===
        entrance_columns = []
        if self.sections > 0:
            section_width = self.columns / (self.sections + 1)
            for i in range(self.sections):
                col = int((i + 1) * section_width)
                col = max(0, min(col, self.columns - 1))
                entrance_columns.append(col)

        logger.info(f"   Входы в колонках: {entrance_columns}")

        # === ЗАПОЛНЯЕМ ФАСАДЫ ===
        for floor in range(self.floors):
            for col in range(self.columns):
                if floor == 0:
                    # ТОЛЬКО ПЕРВЫЙ ЭТАЖ - входы или окна
                    if col in entrance_columns:
                        self.front_grid.set_cell(col, floor, FacadeType.ENTRANCE)
                        self.back_grid.set_cell(col, floor, FacadeType.ENTRANCE)  # Двери есть и сзади
                    else:
                        self.front_grid.set_cell(col, floor, FacadeType.WINDOW)
                        self.back_grid.set_cell(col, floor, FacadeType.WINDOW)
                else:
                    # ВЕРХНИЕ ЭТАЖИ
                    if col in entrance_columns:
                        # НАД ВХОДОМ ВСЕГДА ОКНА
                        self.front_grid.set_cell(col, floor, FacadeType.WINDOW)
                        self.back_grid.set_cell(col, floor, FacadeType.WINDOW)
                    else:
                        # Рандомные по вероятности
                        import random
                        r = random.random()
                        if r < wall_prob:
                            element = FacadeType.EMPTY_WALL
                        elif r < wall_prob + balcony_prob:
                            element = FacadeType.BALCONY
                        else:
                            element = FacadeType.WINDOW

                        self.front_grid.set_cell(col, floor, element)
                        self.back_grid.set_cell(col, floor, element)

        # === БОКОВЫЕ ФАСАДЫ (только стены) ===
        for floor in range(self.floors):
            for d in range(self.depth):
                self.left_grid.set_cell(d, floor, FacadeType.EMPTY_WALL)
                self.right_grid.set_cell(d, floor, FacadeType.EMPTY_WALL)

    def assemble_building(self) -> trimesh.Trimesh:
        """Собирает панельный дом из всех компонентов"""

        # Загружаем модули
        if not self.load_modules():
            return None

        # Генерируем правила
        self.generate_facade_rules()

        building_meshes = []

        # === ПЕРЕДНИЙ ФАСАД (y=0, смотрит в направлении +Y) ===
        logger.info("🔲 Размещение переднего фасада...")
        front_meshes = self._place_facade_grid(
            self.front_grid,
            y_offset=0,
            rotate_z=0  # ← НЕ rotate, а rotate_z!
        )
        building_meshes.extend(front_meshes)

        # === ЗАДНИЙ ФАСАД (y=depth, смотрит в направлении -Y) ===
        logger.info("🔲 Размещение заднего фасада...")
        back_meshes = self._place_facade_grid(
            self.back_grid,
            y_offset=self.building_depth,
            rotate_z=np.pi  # ← 180°
        )
        building_meshes.extend(back_meshes)

        # === ЛЕВЫЙ ФАСАД (x=0, смотрит в направлении +X) ===
        logger.info("🔲 Размещение левого фасада...")
        left_meshes = self._place_side_facade(
            self.left_grid,
            x_offset=0,
            is_left=True
        )
        building_meshes.extend(left_meshes)

        # === ПРАВЫЙ ФАСАД (x=width, смотрит в направлении -X) ===
        logger.info("🔲 Размещение правого фасада...")
        right_meshes = self._place_side_facade(
            self.right_grid,
            x_offset=self.building_width,
            is_left=False
        )
        building_meshes.extend(right_meshes)

        # === ОБЪЕДИНЕНИЕ ===
        logger.info(f"🔗 Объединение {len(building_meshes)} компонентов...")

        try:
            combined = trimesh.util.concatenate(building_meshes)
            combined.merge_vertices()

            logger.info(f"✅ Здание собрано!")
            logger.info(f"   Вершин: {len(combined.vertices)}")
            logger.info(f"   Граней: {len(combined.faces)}")

            return combined
        except Exception as e:
            logger.error(f"❌ Ошибка объединения: {e}")
            return None

    def _place_facade_grid(self, grid: FacadeGrid, y_offset: float, rotate_z: float) -> List[trimesh.Trimesh]:
        """Размещает модули в фасадной сетке (front/back)"""
        meshes = []

        for floor in range(grid.floors):
            for col in range(grid.columns):
                facade_type = grid.grid[floor][col]

                if facade_type not in self.modules:
                    continue

                module = self.modules[facade_type].copy()

                # === ПОВОРОТ ДЛЯ БАЛКОНОВ И ВХОДОВ ===
                # Они ВСЕГДА ротируются на 180° чтобы смотреть наружу
                if facade_type in [FacadeType.BALCONY, FacadeType.ENTRANCE]:
                    module.apply_transform(
                        trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
                    )

                # === ВЫРАВНИВАЕМ ПО НИЖНЕЙ ГРАНИ ===
                bounds = module.bounds
                z_min = bounds[0][2]
                module.apply_translation([0, 0, -z_min])

                # === ЦЕНТРИРУЕМ ПО X И Y (но НЕ по Z!) ===
                center = module.centroid
                module.apply_translation([-center[0], -center[1], 0])

                # === ПОЗИЦИОНИРУЕМ ===
                x = col * self.module_width + self.module_width / 2
                y = y_offset
                z = floor * self.module_height
                module.apply_translation([x, y, z])

                # === ПОВОРОТ ДЛЯ BACK ФАСАДА ===
                if rotate_z != 0:
                    center_point = module.centroid
                    # Вращаем вокруг центра
                    module.apply_translation([-center_point[0], -center_point[1], 0])
                    module.apply_transform(
                        trimesh.transformations.rotation_matrix(rotate_z, [0, 0, 1])
                    )
                    module.apply_translation([center_point[0], center_point[1], 0])

                meshes.append(module)

        logger.info(f"   ✓ Размещено {len(meshes)} модулей")
        return meshes

    def _place_side_facade(
        self,
        grid: FacadeGrid,
        x_offset: float,
        is_left: bool
    ) -> List[trimesh.Trimesh]:
        """
        Размещает ТОЛЬКО СТЕНЫ на боковых фасадах (left/right)

        ИСПРАВЛЕНИЕ:
        - Left: x=0, смотрит в +X
        - Right: x=width, смотрит в -X
        """
        meshes = []

        if FacadeType.EMPTY_WALL not in self.modules:
            logger.warning("⚠️ EMPTY_WALL модуль не найден для боковых фасадов")
            return meshes

        wall_module = self.modules[FacadeType.EMPTY_WALL]

        for floor in range(self.floors):
            for d in range(self.depth):
                module = wall_module.copy()

                # === ВЫРАВНИВАЕМ ПО НИЖНЕЙ ГРАНИ (z=0) ===
                bounds = module.bounds
                z_min = bounds[0][2]
                module.apply_translation([0, 0, -z_min])

                # === ЦЕНТРИРУЕМ ПО X И Y (но НЕ по Z!) ===
                center = module.centroid
                module.apply_translation([-center[0], -center[1], 0])

                # === ПОЗИЦИОНИРУЕМ В СЕТКЕ ===
                # Для боковых фасадов Y идет вдоль глубины дома
                x = x_offset
                y = d * self.module_width + self.module_width / 2
                z = floor * self.module_height
                module.apply_translation([x, y, z])

                # === ПОВОРОТ БОКОВЫХ ФАСАДОВ ===
                center_point = module.centroid

                if is_left:
                    # Left (x=0): поворот -90° вокруг Z (смотрит в +X)
                    module.apply_translation([-center_point[0], -center_point[1], 0])
                    module.apply_transform(
                        trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
                    )
                    module.apply_translation([center_point[0], center_point[1], 0])
                else:
                    # Right (x=width): поворот +90° вокруг Z (смотрит в -X)
                    module.apply_translation([-center_point[0], -center_point[1], 0])
                    module.apply_transform(
                        trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
                    )
                    module.apply_translation([center_point[0], center_point[1], 0])

                meshes.append(module)

        logger.info(f"   ✓ Боковой фасад: {len(meshes)} стен")
        return meshes

    def export_to_obj(self, output_path: Path) -> bool:
        """Экспортирует здание в OBJ"""
        try:
            building = self.assemble_building()

            if building is None:
                logger.error("❌ Не удалось собрать здание")
                return False

            building.export(str(output_path))
            logger.info(f"✅ Здание экспортировано в {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка экспорта: {e}")
            return False


# === ФУНКЦИЯ ДЛЯ ВЫЗОВА ИЗ SERVER.PY ===

def assemble_building(params: Dict[str, Any], models_dir: Path, output_path: Path) -> bool:
    """Собирает панельный дом"""

    assembler = PanelBuildingAssembler(params, models_dir)
    return assembler.export_to_obj(output_path)