"""
assembler.py — Single-phase panel building assembler using Trimesh.

Pipeline per facade:
  1. Build a 2D FacadeType grid (WALL / WINDOW / BALCONY / ENTRANCE)
  2. Place module meshes in one pass; wide/tall overlays reserve adjacent cells
  3. Collect all parts into a trimesh.Scene (per-mesh materials preserved)

wall/window → scaled to exact cell; balcony/door → natural size with wall behind.
Side facades: wall panels rotated ±90°Z. Assembly is Z-up; -90°X at end → Y-up.
"""

import logging
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


class FacadeType(Enum):
    WALL     = "wall"
    WINDOW   = "window"    # served by wall_window composite
    BALCONY  = "balcony"
    ENTRANCE = "entrance"


# ── ModuleLoader ──────────────────────────────────────────────────────────────

class ModuleLoader:
    """
    Loads OBJ modules and caches geometry parts without vertex modification.

    preferred_ids maps module_type → UUID subdirectory so the assembler loads
    the freshly generated module from server.py, not an alphabetical fallback.
    """

    def __init__(
        self,
        modules_dir: Path,
        preferred_ids: Optional[Dict[str, str]] = None,
    ):
        self.modules_dir = Path(modules_dir)
        self._preferred  = preferred_ids or {}
        self._cache: Dict[str, List[trimesh.Trimesh]] = {}

    def parts(self, module_type: str) -> List[trimesh.Trimesh]:
        """Return cached OBJ parts. Raw parts are never modified after caching."""
        if module_type not in self._cache:
            self._cache[module_type] = self._load(module_type)
        return self._cache[module_type]

    def _locate_obj(self, module_type: str) -> Optional[Path]:
        module_dir = self.modules_dir / module_type
        if not module_dir.exists():
            logger.warning(f"Module dir missing: {module_dir}")
            return None
        preferred = self._preferred.get(module_type)
        if preferred:
            p = module_dir / preferred / f"{module_type}.obj"
            if p.exists():
                return p
            logger.warning(
                f"Preferred module {module_type}/{preferred} not found; "
                "falling back to first alphabetical match."
            )
        files = sorted(module_dir.glob(f"*/{module_type}.obj"))
        if not files:
            logger.warning(f"No OBJ found for '{module_type}'")
            return None
        return files[0]

    def _load(self, module_type: str) -> List[trimesh.Trimesh]:
        path = self._locate_obj(module_type)
        if path is None:
            return []
        try:
            loaded = trimesh.load(str(path), process=False)
        except Exception as e:
            logger.error(f"Failed to load '{module_type}': {e}", exc_info=True)
            return []
        if isinstance(loaded, trimesh.Scene):
            parts = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        elif isinstance(loaded, trimesh.Trimesh):
            parts = [loaded]
        else:
            logger.warning(f"Unexpected geometry type for '{module_type}': {type(loaded)}")
            return []
        if parts:
            mn, mx = _combined_bounds(parts)
            d = mx - mn
            logger.info(
                f"Loaded '{module_type}': {path.parent.name}/{path.name} "
                f"[{d[0]:.2f}×{d[1]:.2f}×{d[2]:.2f}m, {len(parts)} part(s)]"
            )
        return parts


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _combined_bounds(parts: List[trimesh.Trimesh]) -> Tuple[np.ndarray, np.ndarray]:
    mn = np.min([p.bounds[0] for p in parts], axis=0)
    mx = np.max([p.bounds[1] for p in parts], axis=0)
    return mn, mx


def _transform_bounds(
    mn: np.ndarray,
    mx: np.ndarray,
    T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a 4×4 transform to all 8 AABB corners and return the new AABB."""
    corners = np.array([
        [mn[0], mn[1], mn[2], 1], [mx[0], mn[1], mn[2], 1],
        [mn[0], mx[1], mn[2], 1], [mx[0], mx[1], mn[2], 1],
        [mn[0], mn[1], mx[2], 1], [mx[0], mn[1], mx[2], 1],
        [mn[0], mx[1], mx[2], 1], [mx[0], mx[1], mx[2], 1],
    ], dtype=float)
    transformed = (T @ corners.T).T[:, :3]
    return transformed.min(axis=0), transformed.max(axis=0)


def _orient_matrix(parts: List[trimesh.Trimesh]) -> np.ndarray:
    """
    Return +90°X if the module is Y-tall (height along Y → corrects to Z-up).
    Wall, balcony, door OBJs are typically Y-tall. Wall_window is already Z-up.
    """
    if not parts:
        return np.eye(4)
    mn, mx = _combined_bounds(parts)
    sz, sy = float((mx - mn)[2]), float((mx - mn)[1])
    if sz < 0.5 and sy > 0.5:
        return trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    return np.eye(4)


def _oriented_extents(
    parts: List[trimesh.Trimesh],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (T_orient, mn_oriented, mx_oriented, centroid_oriented)."""
    T_or = _orient_matrix(parts)
    mn, mx = _combined_bounds(parts)
    mn_o, mx_o = _transform_bounds(mn, mx, T_or)
    centroid = (mn_o + mx_o) * 0.5
    return T_or, mn_o, mx_o, centroid


def _clone_transformed(
    parts: List[trimesh.Trimesh],
    T: np.ndarray,
) -> List[trimesh.Trimesh]:
    """Clone all parts and apply transform T. Raw cached parts are never modified."""
    result = []
    for p in parts:
        m = p.copy()
        m.apply_transform(T)
        result.append(m)
    return result


# ── Assembler ─────────────────────────────────────────────────────────────────

class BuildingAssembler:
    """
    Single-phase grid facade assembler for panel buildings.

    Assigns FacadeType to each cell, places modules in one pass per facade.
    Wide/tall overlays reserve adjacent cells during placement — no pre-planning.
    Front+back: all four types. Left+right sides: wall only, rotated ±90°Z.
    """

    def __init__(self, params: Dict[str, Any], modules_dir: Path):
        preferred_ids: Dict[str, str] = {}
        if params.get("wall_module_id"):
            preferred_ids["wall"] = params["wall_module_id"]
        if params.get("window_module_id"):
            preferred_ids["window"] = params["window_module_id"]
        if params.get("balcony_module_id"):
            preferred_ids["balcony"] = params["balcony_module_id"]

        self.loader = ModuleLoader(Path(modules_dir), preferred_ids)

        self.floors        = max(1, int(params.get("floors",        5)))
        self.cols          = max(1, int(params.get("columns",       10)))
        self.sections      = max(0, int(params.get("sections",       3)))
        self.depth         = max(1, int(params.get("depth",          2)))
        self.texture_scale = max(1, min(8, int(params.get("texture_scale", 3))))
        self.has_balcony   = bool(params.get("has_balcony", False))

        wall_parts = self.loader.parts("wall")
        if not wall_parts:
            raise RuntimeError("Wall module required but not found.")

        _, mn_o, mx_o, _ = _oriented_extents(wall_parts)
        d = mx_o - mn_o
        self.cell_w = float(d[0]) if d[0] > 0.01 else 4.0
        self.cell_h = float(d[2]) if d[2] > 0.01 else 3.0
        self.wall_d = float(d[1]) if d[1] > 0.01 else 0.3

        self.bld_w = self.cols   * self.cell_w
        self.bld_h = self.floors * self.cell_h
        self.bld_d = self.depth  * self.cell_w

        logger.info(
            f"BuildingAssembler: grid {self.cols}×{self.floors} | "
            f"cell {self.cell_w:.2f}×{self.cell_h:.2f}m | "
            f"building {self.bld_w:.1f}×{self.bld_h:.1f}×{self.bld_d:.1f}m | "
            f"texture_scale={self.texture_scale} has_balcony={self.has_balcony}"
        )

    # ── Grid building ─────────────────────────────────────────────────────────

    def _entrance_cols(self) -> Set[int]:
        """Column indices that contain entrance doors (evenly spaced per section)."""
        if self.sections <= 0:
            return set()
        spacing = self.cols / self.sections
        return {
            max(0, min(self.cols - 1, int(round((i + 0.5) * spacing - 0.5))))
            for i in range(self.sections)
        }

    def _random_cell_type(self) -> FacadeType:
        """Pick WALL / WINDOW / BALCONY using texture_scale-based probabilities."""
        s      = self.texture_scale
        wall_p = max(0.0, 0.7 - (s - 1) * 0.1)
        win_p  = (0.2 + (s - 1) * 0.057) if self.has_balcony else (0.3 + (s - 1) * 0.1)
        r      = random.random()
        if r < wall_p:
            return FacadeType.WALL
        if r < wall_p + win_p:
            return FacadeType.WINDOW
        return FacadeType.BALCONY if self.has_balcony else FacadeType.WINDOW

    def _build_front_grid(self, entrance_cols: Set[int]) -> List[List[FacadeType]]:
        """
        Front facade grid:
          floor 0  — entrance at entrance columns, window everywhere else
          floor ≥1 — window above entrance columns, probabilistic otherwise
        """
        grid = [[FacadeType.WALL] * self.cols for _ in range(self.floors)]
        for col in range(self.cols):
            grid[0][col] = (
                FacadeType.ENTRANCE if col in entrance_cols else FacadeType.WINDOW
            )
        for floor in range(1, self.floors):
            for col in range(self.cols):
                grid[floor][col] = (
                    FacadeType.WINDOW if col in entrance_cols
                    else self._random_cell_type()
                )
        return grid

    def _build_back_grid(self) -> List[List[FacadeType]]:
        """
        Back facade grid:
          floor 0  — all windows (no doors on back)
          floor ≥1 — probabilistic WALL / WINDOW / BALCONY
        """
        grid = [[FacadeType.WALL] * self.cols for _ in range(self.floors)]
        for col in range(self.cols):
            grid[0][col] = FacadeType.WINDOW
        for floor in range(1, self.floors):
            for col in range(self.cols):
                grid[floor][col] = self._random_cell_type()
        return grid

    # ── Module placement ──────────────────────────────────────────────────────

    def _place_scaled(
        self,
        meshes: List[trimesh.Trimesh],
        module_type: str,
        cx: float,
        z_bottom: float,
        y_center: float,
        is_front: bool,
    ) -> None:
        """Scale wall or window to exact cell dimensions. Front modules flipped 180°Z."""
        parts = self.loader.parts(module_type)
        if not parts:
            return

        T_or, mn_o, mx_o, oc = _oriented_extents(parts)
        sx = mx_o[0] - mn_o[0]
        sz = mx_o[2] - mn_o[2]
        if sx < 1e-6 or sz < 1e-6:
            return

        T_co = trimesh.transformations.translation_matrix(-oc)
        T_sc = np.diag([self.cell_w / sx, 1.0, self.cell_h / sz, 1.0])
        T_fl = (trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
                if is_front else np.eye(4))
        T_ps = trimesh.transformations.translation_matrix(
            [cx, y_center, z_bottom + self.cell_h / 2]
        )
        T = T_ps @ T_fl @ T_sc @ T_co @ T_or
        meshes.extend(_clone_transformed(parts, T))

    def _place_overlay(
        self,
        meshes: List[trimesh.Trimesh],
        module_type: str,
        cx: float,
        z_bottom: float,
        y_overlay: float,
        is_front: bool,
        reserved: Set[Tuple[int, int]],
        floor: int,
        col: int,
    ) -> None:
        """
        Place balcony or door at natural size (no scaling).
        Wide modules reserve the right neighbour and center across two cells.
        Tall modules reserve the cell above (becomes plain wall).
        """
        parts = self.loader.parts(module_type)
        if not parts:
            return

        T_or, mn_o, mx_o, oc = _oriented_extents(parts)
        mod_w = mx_o[0] - mn_o[0]
        mod_h = mx_o[2] - mn_o[2]

        # Wide module: span two cells
        if mod_w > self.cell_w and col + 1 < self.cols:
            reserved.add((floor, col + 1))
            cx_place = (col + 1) * self.cell_w   # midpoint between cell col and col+1
        else:
            cx_place = cx

        # Tall module: block the cell above
        if mod_h > self.cell_h and floor + 1 < self.floors:
            reserved.add((floor + 1, col))

        T_co = trimesh.transformations.translation_matrix(-oc)
        T_fl = (trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
                if is_front else np.eye(4))
        T_ps = trimesh.transformations.translation_matrix(
            [cx_place, y_overlay, z_bottom + mod_h / 2]
        )
        T = T_ps @ T_fl @ T_co @ T_or
        meshes.extend(_clone_transformed(parts, T))

    # ── Facade builders ───────────────────────────────────────────────────────

    def _build_fb_facade(
        self,
        grid: List[List[FacadeType]],
        y_center: float,
        is_front: bool,
        has_doors: bool,
    ) -> List[trimesh.Trimesh]:
        """
        Build a front or back facade.
        WALL → wall mesh; WINDOW → wall_window (falls back to wall if missing);
        BALCONY/ENTRANCE → wall underneath + overlay in front.
        Reserved cells (overflow from wide/tall overlays) become plain wall.
        """
        meshes: List[trimesh.Trimesh] = []
        win_parts  = self.loader.parts("window")
        bal_parts  = self.loader.parts("balcony") if self.has_balcony else []
        door_parts = self.loader.parts("door")   if has_doors         else []

        # Overlays sit just outside the wall surface
        outward   = -1.0 if is_front else 1.0
        overlay_y = y_center + outward * (self.wall_d / 2 + 0.01)
        reserved: Set[Tuple[int, int]] = set()

        for floor in range(self.floors):
            z_bottom = floor * self.cell_h
            for col in range(self.cols):
                cx        = (col + 0.5) * self.cell_w
                cell_type = FacadeType.WALL if (floor, col) in reserved else grid[floor][col]

                if cell_type == FacadeType.WALL:
                    self._place_scaled(meshes, "wall", cx, z_bottom, y_center, is_front)

                elif cell_type == FacadeType.WINDOW:
                    mod = "window" if win_parts else "wall"
                    self._place_scaled(meshes, mod, cx, z_bottom, y_center, is_front)

                elif cell_type == FacadeType.BALCONY:
                    self._place_scaled(meshes, "wall", cx, z_bottom, y_center, is_front)
                    if bal_parts:
                        self._place_overlay(
                            meshes, "balcony", cx, z_bottom, overlay_y,
                            is_front, reserved, floor, col,
                        )

                elif cell_type == FacadeType.ENTRANCE:
                    self._place_scaled(meshes, "wall", cx, z_bottom, y_center, is_front)
                    if door_parts:
                        self._place_overlay(
                            meshes, "door", cx, z_bottom, overlay_y,
                            is_front, reserved, floor, col,
                        )

        label = "front" if is_front else "back"
        logger.info(f"{label.capitalize()} facade: {len(meshes)} mesh parts")
        return meshes

    def _build_side_facade(self, x_pos: float, is_left: bool) -> List[trimesh.Trimesh]:
        """Wall panels only, scaled to cell, rotated ±90°Z to align with building depth."""
        meshes: List[trimesh.Trimesh] = []
        wall_parts = self.loader.parts("wall")
        if not wall_parts:
            logger.error("Wall module missing — side facade skipped.")
            return meshes

        T_or, mn_o, mx_o, oc = _oriented_extents(wall_parts)
        sx = mx_o[0] - mn_o[0]
        sz = mx_o[2] - mn_o[2]
        if sx < 1e-6 or sz < 1e-6:
            return meshes

        angle  = -np.pi / 2 if is_left else np.pi / 2
        T_co   = trimesh.transformations.translation_matrix(-oc)
        T_sc   = np.diag([self.cell_w / sx, 1.0, self.cell_h / sz, 1.0])
        T_side = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])

        for floor in range(self.floors):
            z_bottom = floor * self.cell_h
            for d in range(self.depth):
                depth_center = (d + 0.5) * self.cell_w
                T_ps = trimesh.transformations.translation_matrix(
                    [x_pos, depth_center, z_bottom + self.cell_h / 2]
                )
                T = T_ps @ T_side @ T_sc @ T_co @ T_or
                meshes.extend(_clone_transformed(wall_parts, T))

        label = "left" if is_left else "right"
        logger.info(f"{label.capitalize()} side facade: {len(meshes)} mesh parts")
        return meshes

    # ── Main assembly ─────────────────────────────────────────────────────────

    def assemble(self) -> trimesh.Scene:
        """
        Assemble all facades into a trimesh.Scene in Y-up orientation.
        Unique node names preserve per-mesh materials on export.
        Final -90°X converts from Z-up assembly space to Three.js Y-up.
        """
        entrance_cols = self._entrance_cols()
        logger.info(f"Assembly start — entrance cols: {sorted(entrance_cols)}")

        front_grid = self._build_front_grid(entrance_cols)
        back_grid  = self._build_back_grid()

        all_meshes: List[trimesh.Trimesh] = []

        # Solid structural base volume
        base = trimesh.creation.box(extents=[self.bld_w, self.bld_d, self.bld_h])
        base.apply_translation([self.bld_w / 2, self.bld_d / 2, self.bld_h / 2])
        all_meshes.append(base)

        # Front facade — doors present, modules flipped 180°Z to face outward (-Y)
        all_meshes.extend(
            self._build_fb_facade(front_grid, y_center=0.0, is_front=True, has_doors=True)
        )
        # Back facade — no doors, modules face +Y outward by default
        all_meshes.extend(
            self._build_fb_facade(back_grid, y_center=self.bld_d, is_front=False, has_doors=False)
        )
        # Side facades — plain wall panels only
        all_meshes.extend(self._build_side_facade(x_pos=0.0,       is_left=True))
        all_meshes.extend(self._build_side_facade(x_pos=self.bld_w, is_left=False))

        # Z-up → Y-up conversion for Three.js compatibility
        rot_yup = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        scene   = trimesh.Scene()
        for i, mesh in enumerate(all_meshes):
            mesh.apply_transform(rot_yup)
            scene.add_geometry(mesh, node_name=f"mesh_{i:04d}")

        logger.info(f"Assembly complete — {len(all_meshes)} mesh parts in scene")
        return scene

    def export(self, output_path: Path) -> bool:
        """Assemble and export to OBJ + MTL at output_path."""
        try:
            scene = self.assemble()
            scene.export(str(output_path))
            logger.info(f"Exported: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return False


# ── Public API ────────────────────────────────────────────────────────────────

def assemble_building(params: Dict[str, Any], models_dir: Path, output_path: Path) -> bool:
    """Entry point called by server.py."""
    assembler = BuildingAssembler(params, models_dir)
    return assembler.export(output_path)
