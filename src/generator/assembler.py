"""
assembler.py — Simple building assembler.

Receives:
  floors, sections (entrance count), columns (width in cells), depth (in cells),
  has_balcony, texture_scale, and module IDs for wall / wall_window / balcony / entrance.

The wall_window module is pre-built by the server (wall params + window params merged
via the batch runner) before this assembler is called.

Grid: columns × floors cells per front/back facade. Cell size = wall module native size.

Facade rules:
  front  — entrances (floor 0) + balcony / wall_window (texture_scale density)
  back   — balcony / wall_window (no entrances)
  left   — wall only  (depth × floors)
  right  — wall only  (depth × floors)

Entrance column positions  (0-indexed):
  sections=3, columns=18  →  columns [3, 9, 15]
  formula: col[i] = int((i + 0.5) × columns / sections)

Module density per cell = 0.1 + texture_scale × 0.1
  texture_scale 1 → 20 %,  texture_scale 8 → 90 %

No scaling. No per-cell conditional rotation. No reservation system.
Each module is loaded at its native size, centered, given a fixed face rotation,
then translated to its grid position.

Assembly space: Z-up.  Final global -90°X rotation → Y-up for Three.js.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


# ── Module loader ──────────────────────────────────────────────────────────────

class ModuleLoader:
    """Load and cache OBJ/MTL modules from  modules_dir/{type}/{uuid}/{type}.obj."""

    def __init__(self, modules_dir: Path, preferred_ids: Optional[Dict[str, str]] = None):
        self.modules_dir = Path(modules_dir)
        self._preferred  = preferred_ids or {}
        self._cache: Dict[str, List[trimesh.Trimesh]] = {}

    def parts(self, module_type: str) -> List[trimesh.Trimesh]:
        if module_type not in self._cache:
            self._cache[module_type] = self._load(module_type)
        return self._cache[module_type]

    def _locate(self, module_type: str) -> Optional[Path]:
        d = self.modules_dir / module_type
        if not d.exists():
            logger.warning(f"Module directory missing: {d}")
            return None
        preferred = self._preferred.get(module_type)
        if preferred:
            p = d / preferred / f"{module_type}.obj"
            if p.exists():
                return p
            logger.warning(f"Preferred {module_type}/{preferred} not found, using fallback.")
        files = sorted(d.glob(f"*/{module_type}.obj"))
        if not files:
            logger.warning(f"No OBJ file found for module type '{module_type}'")
            return None
        return files[0]

    def _load(self, module_type: str) -> List[trimesh.Trimesh]:
        # Всегда загружаем свежим - не кешируем visual
        path = self._locate(module_type)
        if path is None:
            return []

        try:
            import os
            old_cwd = os.getcwd()
            os.chdir(path.parent)
            try:
                loaded = trimesh.load(str(path), process=False, skip_materials=False)
            finally:
                os.chdir(old_cwd)
        except Exception as e:
            logger.error(f"Failed to load '{module_type}': {e}", exc_info=True)
            return []

        if isinstance(loaded, trimesh.Scene):
            parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        elif isinstance(loaded, trimesh.Trimesh):
            parts = [loaded]
        else:
            logger.warning(f"Unexpected type for '{module_type}': {type(loaded)}")
            return []

        if parts:
            mn = np.min([p.bounds[0] for p in parts], axis=0)
            mx = np.max([p.bounds[1] for p in parts], axis=0)
            d = mx - mn
            logger.info(
                f"Loaded '{module_type}' from {path.parent.name}/"
                f"{path.name}  [{d[0]:.2f} × {d[1]:.2f} × {d[2]:.2f} m,  {len(parts)} part(s)]"
            )

        return parts


# ── Geometry utilities ─────────────────────────────────────────────────────────

def _combined_bounds(parts: List[trimesh.Trimesh]) -> Tuple[np.ndarray, np.ndarray]:
    mn = np.min([p.bounds[0] for p in parts], axis=0)
    mx = np.max([p.bounds[1] for p in parts], axis=0)
    return mn, mx


def _transform_bounds(mn, mx, T):
    corners = np.array([
        [mn[0], mn[1], mn[2], 1], [mx[0], mn[1], mn[2], 1],
        [mn[0], mx[1], mn[2], 1], [mx[0], mx[1], mn[2], 1],
        [mn[0], mn[1], mx[2], 1], [mx[0], mn[1], mx[2], 1],
        [mn[0], mx[1], mx[2], 1], [mx[0], mx[1], mx[2], 1],
    ], dtype=float)
    t = (T @ corners.T).T[:, :3]
    return t.min(axis=0), t.max(axis=0)


def _orient_matrix(parts: List[trimesh.Trimesh]) -> np.ndarray:
    """
    If the module is Y-tall (height along Y, typical for wall/balcony/entrance OBJs),
    apply +90°X to bring it into Z-up space.  Wall_window is already Z-up → identity.
    """
    if not parts:
        return np.eye(4)
    mn, mx = _combined_bounds(parts)
    d = mx - mn
    if d[2] < 0.5 and d[1] > 0.5:
        return trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    return np.eye(4)


# ── Assembler ──────────────────────────────────────────────────────────────────

class SimpleAssembler:
    """
    Builds a box building by placing identical modules on a grid.

    Per-face orientations (Z-up space, module face direction is -Y by default):
      front  Y = 0      →  identity        (faces -Y outward toward street)
      back   Y = bld_d  →  180 ° Z         (faces +Y outward)
      left   X = 0      →  -90 ° Z         (faces -X outward)
      right  X = bld_w  →  +90 ° Z         (faces +X outward)
    These are fixed constants, not per-cell logic.
    """

    _FACE_ROT: Dict[str, np.ndarray] = {
        "front": np.eye(4),
        "back":  trimesh.transformations.rotation_matrix( np.pi,        [0, 0, 1]),
        "left":  trimesh.transformations.rotation_matrix(-np.pi / 2,    [0, 0, 1]),
        "right": trimesh.transformations.rotation_matrix( np.pi / 2,    [0, 0, 1]),
    }

    def __init__(self, params: Dict[str, Any], modules_dir: Path):
        preferred_ids: Dict[str, str] = {}
        if params.get("wall_module_id"):         preferred_ids["wall"]        = params["wall_module_id"]
        if params.get("wall_window_module_id"):  preferred_ids["wall_window"] = params["wall_window_module_id"]
        if params.get("balcony_module_id"):      preferred_ids["balcony"]     = params["balcony_module_id"]
        if params.get("entrance_module_id"):     preferred_ids["door"]        = params["entrance_module_id"]

        self.loader = ModuleLoader(Path(modules_dir), preferred_ids)

        self.floors        = max(1, int(params.get("floors",        5)))
        self.cols          = max(1, int(params.get("columns",       10)))
        self.sections      = max(0, int(params.get("sections",       3)))
        self.depth         = max(1, int(params.get("depth",          2)))
        self.texture_scale = max(1, min(8, int(params.get("texture_scale", 1))))
        self.has_balcony   = bool(params.get("has_balcony", False))

        wall_parts = self.loader.parts("wall")
        if not wall_parts:
            raise RuntimeError("Wall module not found — cannot determine cell size.")

        T_or = _orient_matrix(wall_parts)
        mn, mx = _combined_bounds(wall_parts)
        mn_o, mx_o = _transform_bounds(mn, mx, T_or)
        d = mx_o - mn_o

        self.cell_w = float(d[0]) if d[0] > 0.01 else 4.0
        self.cell_h = float(d[2]) if d[2] > 0.01 else 3.0

        self.bld_w = self.cols   * self.cell_w
        self.bld_h = self.floors * self.cell_h
        self.bld_d = self.depth  * self.cell_w

        logger.info(
            f"SimpleAssembler  grid={self.cols}×{self.floors}  "
            f"cell={self.cell_w:.2f}×{self.cell_h:.2f} m  "
            f"building={self.bld_w:.1f}×{self.bld_h:.1f}×{self.bld_d:.1f} m  "
            f"scale={self.texture_scale}  balcony={self.has_balcony}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _entrance_cols(self) -> Set[int]:
        """
        Evenly space entrance columns across the facade width.
        sections=3, cols=18  →  {3, 9, 15}
        """
        if self.sections <= 0:
            return set()
        spacing = self.cols / self.sections
        return {int((i + 0.5) * spacing) for i in range(self.sections)}

    def _density(self) -> float:
        """Module density: texture_scale 1 → 20 %, texture_scale 8 → 90 %."""
        return 0.1 + self.texture_scale * 0.1

    def _pick(self, col: int, floor: int, entrance_cols: Set[int], has_entrance: bool) -> str:
        """Return the module type for one grid cell."""
        if has_entrance and floor == 0 and col in entrance_cols:
            return "door"
        if random.random() < self._density():
            if self.has_balcony and floor >= 1 and random.random() < 0.5:
                return "balcony"
            return "wall_window"
        return "wall"

    def _place(
            self,
            meshes: List[trimesh.Trimesh],
            module_type: str,
            x: float,
            y: float,
            z: float,
            face: str,
            debug: bool = False,
    ) -> None:
        parts = self.loader._load(module_type)
        if not parts and module_type != "wall":
            parts = self.loader._load("wall")
        if not parts:
            if debug:
                logger.warning(f"❌ [{face}] {module_type}: NO PARTS")
            return

        if debug:
            logger.info(f"✓ [{face}] {module_type}: loaded {len(parts)} part(s)")

        T_or = _orient_matrix(parts)
        mn, mx = _combined_bounds(parts)
        mn_o, mx_o = _transform_bounds(mn, mx, T_or)
        centroid = (mn_o + mx_o) / 2

        T_co = trimesh.transformations.translation_matrix(-centroid)
        T_pos = trimesh.transformations.translation_matrix([x, y, z])
        R = self._FACE_ROT[face]

        if module_type == "balcony":
            R_balcony = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            T = T_pos @ R @ R_balcony @ T_co @ T_or
        else:
            T = T_pos @ R @ T_co @ T_or

        for part in parts:
            m = part.copy()

            has_visual = hasattr(part, 'visual') and part.visual is not None
            has_image = has_visual and hasattr(part.visual, 'image') and part.visual.image is not None

            if debug:
                logger.info(
                    f"    Part: visual={type(part.visual).__name__ if has_visual else 'None'}, image={has_image}")

            if has_visual:
                m.visual = part.visual.copy()

            m.apply_transform(T)
            meshes.append(m)

    # ── Facade builders ────────────────────────────────────────────────────────

    def _build_front_back(
            self,
            face: str,
            y_pos: float,
            has_entrance: bool,
    ) -> List[trimesh.Trimesh]:
        """Build front or back facade."""
        entrance_cols = self._entrance_cols() if has_entrance else set()
        meshes: List[trimesh.Trimesh] = []

        back_count = 0  # Счетчик для логирования back

        for floor in range(self.floors):
            z = floor * self.cell_h + self.cell_h / 2
            for col in range(self.cols):
                x = (col + 0.5) * self.cell_w
                mod = self._pick(col, floor, entrance_cols, has_entrance)

                if mod in ("balcony", "door"):
                    self._place(meshes, "wall", x, y_pos, z, face)
                    self._place(meshes, mod, x, y_pos, z, face, debug=(face == "back" and back_count < 3))
                    back_count += 1
                else:
                    self._place(meshes, mod, x, y_pos, z, face, debug=(face == "back" and back_count < 3))
                    back_count += 1

        logger.info(f"{face.capitalize()} facade: {len(meshes)} mesh parts added\n")
        return meshes

    def _build_side(self, face: str, x_pos: float) -> List[trimesh.Trimesh]:
        """Build left or right facade — wall panels only."""
        meshes: List[trimesh.Trimesh] = []

        for floor in range(self.floors):
            z = floor * self.cell_h + self.cell_h / 2
            for row in range(self.depth):
                y = (row + 0.5) * self.cell_w
                self._place(meshes, "wall", x_pos, y, z, face)

        logger.info(f"{face.capitalize()} facade: {len(meshes)} mesh parts")
        return meshes

    # ── Main assembly ──────────────────────────────────────────────────────────

    def assemble(self) -> trimesh.Scene:
        """
        Assemble all four facades + structural base into a trimesh.Scene.
        Final -90°X converts from Z-up assembly space to Three.js Y-up.
        """
        meshes: List[trimesh.Trimesh] = []

        # Solid interior volume so the building does not look hollow
        base = trimesh.creation.box(extents=[self.bld_w, self.bld_d, self.bld_h])
        base.apply_translation([self.bld_w / 2, self.bld_d / 2, self.bld_h / 2])
        meshes.append(base)

        meshes.extend(self._build_front_back("front", y_pos=0.0,        has_entrance=True))
        meshes.extend(self._build_front_back("back",  y_pos=self.bld_d, has_entrance=False))
        meshes.extend(self._build_side("left",  x_pos=0.0))
        meshes.extend(self._build_side("right", x_pos=self.bld_w))

        rot_yup = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        scene   = trimesh.Scene()
        for i, m in enumerate(meshes):
            m.apply_transform(rot_yup)
            scene.add_geometry(m, node_name=f"mesh_{i:04d}")

        logger.info(f"Assembly complete: {len(meshes)} mesh parts")
        return scene

    def export(self, output_path: Path) -> bool:
        try:
            scene = self.assemble()

            # Скопировать текстуры и переписать MTL пути
            self._prepare_for_export(output_path)

            scene.export(str(output_path))
            logger.info(f"Exported: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return False

    def _prepare_for_export(self, output_path: Path) -> None:
        """Скопировать текстуры и переписать MTL чтобы указывал на них."""
        import shutil

        output_dir = output_path.parent
        texture_dir = output_dir / "textures"
        texture_dir.mkdir(exist_ok=True)

        # Копируем ВСЕ PNG текстуры из всех модулей в одну папку
        copied_textures = set()
        for module_type in ["wall", "window", "wall_window", "balcony", "door"]:
            type_dir = self.loader.modules_dir / module_type
            if type_dir.exists():
                for png_file in type_dir.rglob("*.png"):
                    if png_file.name not in copied_textures:
                        dest = texture_dir / png_file.name
                        shutil.copy2(png_file, dest)
                        copied_textures.add(png_file.name)

        logger.info(f"Copied {len(copied_textures)} textures to {texture_dir}")

        # Переписать MTL чтобы указывал на textures/
        mtl_path = output_dir / "house.mtl"
        mtl_content = "# Combined materials\n\n"

        for module_type in ["wall", "window", "wall_window", "balcony", "door"]:
            type_dir = self.loader.modules_dir / module_type
            if type_dir.exists():
                for mtl_file in type_dir.rglob("*.mtl"):
                    content = mtl_file.read_text()
                    # Заменить все пути на textures/
                    content = content.replace("map_Kd ", "map_Kd textures/")
                    content = content.replace("map_Bump ", "map_Bump textures/")
                    content = content.replace("map_Pr ", "map_Pr textures/")
                    mtl_content += content + "\n"

        mtl_path.write_text(mtl_content)
        logger.info(f"Created MTL: {mtl_path}")


# ── Public API ─────────────────────────────────────────────────────────────────

def assemble_building(params: Dict[str, Any], models_dir: Path, output_path: Path) -> bool:
    """Entry point called by server.py."""
    return SimpleAssembler(params, models_dir).export(output_path)
