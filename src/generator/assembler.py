"""
assembler.py — Сборка дома из отдельных компонентов
Объединяет window.obj, balcony.obj, door.obj, entrance.obj в один building.obj
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import trimesh
import numpy as np

logger = logging.getLogger(__name__)


class BuildingAssembler:
    """Собирает дом из процедурно сгенерированных компонентов"""

    def __init__(self, models_dir: Path, building_params: Dict[str, Any]):
        """
        Args:
            models_dir: Папка с .obj файлами (models/)
            building_params: Параметры дома {floors, windows_per_floor, ...}
        """
        self.models_dir = Path(models_dir)
        self.params = building_params

        # Параметры здания
        self.floors = building_params.get("floors", 5)
        self.wall_height = building_params.get("wall_height", 3.0)
        self.building_length = building_params.get("building_length", 25.0)
        self.building_width = building_params.get("building_width", 12.0)
        self.windows_per_floor = building_params.get("windows_per_floor", 4)
        self.balconies_per_floor = building_params.get("balconies_per_floor", 2)
        self.entrance_count = building_params.get("entrance_count", 1)
        self.balcony_depth = building_params.get("balcony_depth", 1.15)

        # Загруженные mesh'и
        self.meshes = {}

    def load_component(self, component_name: str) -> trimesh.Trimesh:
        """
        Загружает компонент из .obj файла

        Args:
            component_name: "window", "balcony", "door", "entrance"

        Returns:
            Загруженный mesh или None если файла нет
        """
        obj_path = self.models_dir / f"{component_name}.obj"

        if not obj_path.exists():
            logger.warning(f"⚠️ Файл не найден: {obj_path}")
            return None

        try:
            mesh = trimesh.load(str(obj_path), process=False)
            logger.info(f"✓ Загружен {component_name}: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
            return mesh
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки {component_name}: {e}")
            return None

    def assemble_building(self) -> trimesh.Trimesh:
        """
        Собирает полный дом из компонентов

        Алгоритм:
        1. Загружаем компоненты
        2. Размножаем окна по этажам и фасадам
        3. Добавляем балконы
        4. Добавляем двери
        5. Добавляем входы
        6. Объединяем всё в один mesh
        """

        logger.info("=" * 60)
        logger.info("🏗️ СБОРКА ДОМА")
        logger.info("=" * 60)

        # Список mesh'ей для объединения
        building_meshes = []

        # === ЗАГРУЖАЕМ КОМПОНЕНТЫ ===
        logger.info("📥 Загрузка компонентов...")

        window_mesh = self.load_component("window")
        balcony_mesh = self.load_component("balcony")
        door_mesh = self.load_component("door")
        entrance_mesh = self.load_component("entrance")

        # === ОКНА ===
        if window_mesh is not None:
            logger.info(
                f"🪟 Размножение окон ({self.windows_per_floor} × {self.floors} = {self.windows_per_floor * self.floors})...")
            window_meshes = self._arrange_windows(window_mesh)
            building_meshes.extend(window_meshes)

        # === БАЛКОНЫ ===
        if balcony_mesh is not None:
            logger.info(
                f"🏠 Размножение балконов ({self.balconies_per_floor} × {self.floors} = {self.balconies_per_floor * self.floors})...")
            balcony_meshes = self._arrange_balconies(balcony_mesh)
            building_meshes.extend(balcony_meshes)

        # === ДВЕРИ ===
        if door_mesh is not None:
            logger.info(f"🚪 Добавление дверей ({self.entrance_count})...")
            door_meshes = self._arrange_doors(door_mesh)
            building_meshes.extend(door_meshes)

        # === ВХОДЫ/ПОДЪЕЗДЫ ===
        if entrance_mesh is not None:
            logger.info(f"🚶 Добавление подъездов ({self.entrance_count})...")
            entrance_meshes = self._arrange_entrances(entrance_mesh)
            building_meshes.extend(entrance_meshes)

        # === ОБЪЕДИНЕНИЕ ===
        if not building_meshes:
            logger.error("❌ Нет компонентов для сборки!")
            return None

        logger.info(f"🔗 Объединение {len(building_meshes)} компонентов...")

        try:
            combined = trimesh.util.concatenate(building_meshes)
            combined.merge_vertices()

            logger.info(f"✅ Дом собран!")
            logger.info(f"   - Вершин: {len(combined.vertices)}")
            logger.info(f"   - Граней: {len(combined.faces)}")
            logger.info(f"   - Объем: {combined.volume:.2f}")

            return combined

        except Exception as e:
            logger.error(f"❌ Ошибка объединения: {e}")
            return None

    def _arrange_windows(self, window_mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        """
        Размножает окна по фасадам и этажам

        Расстояние между окнами рассчитывается из длины здания
        """
        windows = []

        # Расстояние между окнами вдоль фасада
        window_spacing = self.building_length / (self.windows_per_floor + 1)

        # Для каждого этажа
        for floor in range(self.floors):
            floor_z = floor * self.wall_height

            # Для каждого окна на этаже
            for window_idx in range(self.windows_per_floor):
                # Позиция окна вдоль фасада
                window_x = window_spacing * (window_idx + 1)

                # Копируем mesh окна
                w = window_mesh.copy()

                # Позиционируем
                # Сдвиг вверх: подоконник обычно 0.9м
                window_z = floor_z + 0.9

                # Переводим в центр
                center = w.centroid
                w.apply_translation([-center[0], -center[1], -center[2]])

                # Позиционируем в правильное место
                w.apply_translation([window_x, 0, window_z])

                windows.append(w)

        logger.info(f"   ✓ Окна расставлены: {len(windows)} шт")
        return windows

    def _arrange_balconies(self, balcony_mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        """
        Размножает балконы по фасадам и этажам

        Балконы выступают из фасада на глубину balcony_depth
        """
        balconies = []

        # Расстояние между балконами вдоль фасада
        balcony_spacing = self.building_length / (self.balconies_per_floor + 1)

        # Для каждого этажа
        for floor in range(self.floors):
            floor_z = floor * self.wall_height

            # Для каждого балкона на этаже
            for balcony_idx in range(self.balconies_per_floor):
                # Позиция балкона вдоль фасада
                balcony_x = balcony_spacing * (balcony_idx + 1)

                # Копируем mesh балкона
                b = balcony_mesh.copy()

                # Позиционируем
                center = b.centroid
                b.apply_translation([-center[0], -center[1], -center[2]])

                # Балкон выступает из фасада
                balcony_y = self.balcony_depth

                b.apply_translation([balcony_x, balcony_y, floor_z])

                balconies.append(b)

        logger.info(f"   ✓ Балконы расставлены: {len(balconies)} шт")
        return balconies

    def _arrange_doors(self, door_mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        """
        Добавляет двери на фасад
        """
        doors = []

        # Двери ставим в центре фасада на первом этаже
        center_x = self.building_length / 2

        for entrance_idx in range(self.entrance_count):
            # Если несколько входов, расставляем их по фасаду
            if self.entrance_count > 1:
                spacing = self.building_length / (self.entrance_count + 1)
                center_x = spacing * (entrance_idx + 1)

            d = door_mesh.copy()

            center = d.centroid
            d.apply_translation([-center[0], -center[1], -center[2]])

            # Дверь находится на фасаде (y=0) на земле (z=0)
            d.apply_translation([center_x, 0, 0])

            doors.append(d)

        logger.info(f"   ✓ Двери установлены: {len(doors)} шт")
        return doors

    def _arrange_entrances(self, entrance_mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        """
        Добавляет подъезды/входы перед зданием
        """
        entrances = []

        # Подъезды перед входами
        for entrance_idx in range(self.entrance_count):
            e = entrance_mesh.copy()

            center = e.centroid
            e.apply_translation([-center[0], -center[1], -center[2]])

            # Позиция входа (перед дверью)
            if self.entrance_count > 1:
                spacing = self.building_length / (self.entrance_count + 1)
                entrance_x = spacing * (entrance_idx + 1)
            else:
                entrance_x = self.building_length / 2

            # Подъезд находится перед зданием (y отрицательный)
            entrance_y = -2.0

            e.apply_translation([entrance_x, entrance_y, 0])

            entrances.append(e)

        logger.info(f"   ✓ Подъезды установлены: {len(entrances)} шт")
        return entrances

    def export_to_obj(self, output_path: Path) -> bool:
        """
        Экспортирует собранный дом в OBJ файл
        """
        try:
            building = self.assemble_building()

            if building is None:
                logger.error("❌ Не удалось собрать дом")
                return False

            # Экспортируем
            building.export(str(output_path))

            logger.info(f"✅ Дом экспортирован в {output_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка экспорта: {e}")
            return False


# ==================== ИНТЕГРАЦИЯ С SERVER.PY ====================

def assemble_building(params: Dict[str, Any], models_dir: Path, output_path: Path) -> bool:
    """
    Функция для вызова из server.py

    Args:
        params: Параметры здания
        models_dir: Папка с компонентами
        output_path: Путь для сохранения building.obj
    """

    assembler = BuildingAssembler(models_dir, params)
    return assembler.export_to_obj(output_path)


if __name__ == "__main__":
    # Пример использования
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Использование: python assembler.py <models_dir> [output_path]")
        sys.exit(1)

    models_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else models_dir / "building.obj"

    # Пример параметров
    params = {
        "floors": 5,
        "wall_height": 3.0,
        "building_length": 25.0,
        "building_width": 12.0,
        "windows_per_floor": 4,
        "balconies_per_floor": 2,
        "entrance_count": 1,
        "balcony_depth": 1.15
    }

    success = assemble_building(params, models_dir, output_path)
    sys.exit(0 if success else 1)