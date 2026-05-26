"""
assembler.py — Strict grid-based panel building assembler

Cell size:
  cellWidth  = wall module bounding-box X
  cellHeight = wall module bounding-box Z  (internal Z-up space)

Per-facade pipeline (TWO PHASES):

  PHASE 1 — Plan (no meshes, pure state assignment)
    Step 1  All cells initialised as WALL
    Step 2  Door cells reserved  (floor 0, front only, highest priority)
    Step 3  Balcony cells reserved  (floors ≥ 1, deterministic stagger)
            — cell state set to BALCONY/BALCONY_UPPER to block window promotion
            — a plain WALL mesh is still emitted for every balcony cell
    Step 4  Remaining WALL cells promoted to WALL_WINDOW
            using a deterministic interval derived from texture_scale

  PHASE 2 — Build (one mesh per cell, plus overlays)
    DOOR         → plain wall mesh behind door (same as BALCONY) PLUS door
                   module overlay offset outward; 180° Z-flip on front facade
    BALCONY      → plain wall mesh (underlying) PLUS balcony overlay in front;
                   180° Z-flip on front facade overlay only
    BALCONY_UPPER→ plain wall mesh only (overlay handled by BALCONY row)
    WALL_WINDOW  → wall_window composite, scaled to cell; 180° Z-flip on front
                   facade only — module convention: window face at +Y, which is
                   already the back facade outward direction (no back flip needed)
    WALL         → plain wall, scaled to cell; 180° Z-flip on front facade only

    Door and front-facade balcony overlays are offset outward so their back
    face sits flush with the wall outer surface (no wall clipping).

    After all facades are assembled: a -90° X rotation is applied to the
    entire building scene, converting from internal Z-up to Three.js Y-up.

ASSET IMMUTABILITY CONTRACT
  ModuleLoader caches raw OBJ parts without ANY vertex modifications.
  The orientation-fix matrix is computed analytically from the raw bounding
  box and stored separately.  Every mesh placement:
    1. Copies the raw part(s)          — src.copy()
    2. Applies a composed 4×4 matrix  — m.apply_transform(T)
     where T = T_pos @ T_flip @ T_scale @ T_center @ T_orient
  The raw templates in cache are NEVER modified after loading.

Rules:
  • No standalone window module — wall_window is the only window representation
  • Door and balcony reservations happen BEFORE any window logic
  • Reserved cells cannot be overridden by window replacement
  • texture_scale drives window density deterministically (no randomness)
  • Balcony is an overlay — it never removes the wall behind it
  • Assembler is placement-only — it does not generate geometry
"""

import logging
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

import trimesh
import numpy as np

logger = logging.getLogger(__name__)


# ========================= CELL STATE =========================

class CellState(Enum):
    WALL         = "wall"           # plain wall (default)
    WALL_WINDOW  = "wall_window"    # wall replaced by wall_window composite
    DOOR         = "door"           # structural — entrance door span
    BALCONY      = "balcony"        # overlay balcony; underlying wall still emitted
    BALCONY_UPPER = "balcony_upper" # upper cell of 2-floor balcony; wall still emitted

    def is_structural(self) -> bool:
        """Structural cells cannot be overridden by window replacement."""
        return self in (CellState.DOOR, CellState.BALCONY, CellState.BALCONY_UPPER)

    def needs_wall_mesh(self) -> bool:
        """Every cell produces a wall (or wall_window) mesh in phase 2.
        DOOR cells emit a plain wall behind the door overlay."""
        return True


class FacadeGrid:
    """
    2D planning grid (cols × floors).
    Starts fully initialised as WALL.
    Structural reservations block window promotion but do NOT suppress wall meshes
    for BALCONY/BALCONY_UPPER cells.
    """

    def __init__(self, cols: int, floors: int):
        self.cols   = cols
        self.floors = floors
        self._state: List[List[CellState]] = [
            [CellState.WALL] * cols for _ in range(floors)
        ]

    def state(self, col: int, floor: int) -> CellState:
        if 0 <= col < self.cols and 0 <= floor < self.floors:
            return self._state[floor][col]
        return CellState.DOOR  # out-of-bounds treated as reserved

    def is_wall(self, col: int, floor: int) -> bool:
        """True only for plain WALL — eligible for wall_window promotion."""
        return self.state(col, floor) == CellState.WALL

    def can_reserve(self, cols_r: List[int], floors_r: List[int]) -> bool:
        return all(self.state(c, f) == CellState.WALL
                   for c in cols_r for f in floors_r)

    def reserve(self, cols_r: List[int], floors_r: List[int], state: CellState) -> None:
        for c in cols_r:
            for f in floors_r:
                if 0 <= c < self.cols and 0 <= f < self.floors:
                    self._state[f][c] = state

    def promote_to_wall_window(self, col: int, floor: int) -> None:
        if self.state(col, floor) == CellState.WALL:
            self._state[floor][col] = CellState.WALL_WINDOW


# ========================= MODULE LOADER =========================

class ModuleLoader:
    """
    Loads OBJ assets and caches them as immutable raw parts.

    CONTRACT: raw OBJ vertices are NEVER modified after loading.

    The orientation-fix transform (needed because procedural exporters write
    OBJs with -90°X applied) is stored as a 4×4 matrix and composed into the
    per-placement transform at assembly time, not applied to the template.

    Wall-window OBJs contain two `usemtl` groups; keeping them as separate
    Trimesh parts (never concatenating) preserves both materials.
    """

    def __init__(self, modules_dir: Path, preferred_ids: Optional[Dict[str, str]] = None):
        self.modules_dir = Path(modules_dir)
        # Maps module_type → specific UUID subdirectory to load preferentially.
        self._preferred: Dict[str, str] = preferred_ids or {}
        # Raw OBJ parts per module — vertices NEVER modified after this cache is populated.
        self._parts_cache: Dict[str, List[trimesh.Trimesh]] = {}
        # Orientation-fix matrix per module — computed analytically, no vertex mutation.
        self._orient_cache: Dict[str, np.ndarray] = {}

    # ── Public API ────────────────────────────────────────────────

    def raw_parts(self, module_type: str) -> List[trimesh.Trimesh]:
        """Return immutable raw OBJ parts. Never apply transforms directly to these."""
        if module_type not in self._parts_cache:
            self._parts_cache[module_type] = self._load_raw(module_type)
        return self._parts_cache[module_type]

    def orient_matrix(self, module_type: str) -> np.ndarray:
        """4×4 matrix that corrects the module's export-time coordinate rotation.

        Procedural exporters apply -90°X before writing OBJ so the file looks
        upright in standard viewers.  The correction is +90°X.  Wall_window is
        already Z-up; its correction matrix is identity.

          wall:      raw sy(3m, height) > raw sz(0.25m, thickness) → +90°X ✓
          balcony:   raw sy > raw sz                                → +90°X ✓
          door:      raw sy > raw sz                                → +90°X ✓
          wall_win:  raw sy(0.25m) < raw sz(3m)                   → identity ✓
        """
        if module_type not in self._orient_cache:
            parts = self.raw_parts(module_type)
            self._orient_cache[module_type] = _detect_orient_matrix(parts)
        return self._orient_cache[module_type]

    def bbox(self, module_type: str) -> Tuple[float, float, float]:
        """(width_x, depth_y, height_z) in Z-up oriented space.

        Computed analytically from raw bounds + orient matrix — no vertex mutation.
        """
        parts = self.raw_parts(module_type)
        if not parts:
            return (1.0, 0.3, 1.0)
        mn, mx = _oriented_bounds(parts, self.orient_matrix(module_type))
        d = mx - mn
        return (float(d[0]), float(d[1]), float(d[2]))

    # ── Internal ──────────────────────────────────────────────────

    def _locate_obj(self, module_type: str) -> Optional[Path]:
        module_dir = self.modules_dir / module_type
        if not module_dir.exists():
            logger.warning(f"Module dir not found: {module_dir}")
            return None
        preferred_id = self._preferred.get(module_type)
        if preferred_id:
            preferred_path = module_dir / preferred_id / f"{module_type}.obj"
            if preferred_path.exists():
                logger.debug(f"Using preferred module {module_type}/{preferred_id}")
                return preferred_path
            logger.warning(
                f"Preferred module {module_type}/{preferred_id} not found at "
                f"{preferred_path}; falling back to alphabetical search."
            )
        obj_files = sorted(module_dir.glob(f"*/{module_type}.obj"))
        if not obj_files:
            logger.warning(f"No OBJ for module '{module_type}'")
            return None
        return obj_files[0]

    def _load_raw(self, module_type: str) -> List[trimesh.Trimesh]:
        """Load OBJ and return all sub-geometry parts.  NO vertex modifications."""
        obj_path = self._locate_obj(module_type)
        if obj_path is None:
            return []
        try:
            loaded = trimesh.load(str(obj_path), process=False)
        except Exception as exc:
            logger.error(f"Failed to load '{module_type}': {exc}", exc_info=True)
            return []

        if isinstance(loaded, trimesh.Scene):
            parts = [g for g in loaded.geometry.values()
                     if isinstance(g, trimesh.Trimesh)]
        elif isinstance(loaded, trimesh.Trimesh):
            parts = [loaded]
        else:
            logger.warning(f"Unexpected type for '{module_type}': {type(loaded)}")
            return []

        if parts:
            raw_mins = np.min([p.bounds[0] for p in parts], axis=0)
            raw_maxs = np.max([p.bounds[1] for p in parts], axis=0)
            d = raw_maxs - raw_mins
            logger.info(
                f"Loaded '{module_type}': {obj_path.parent.name}/{obj_path.name} "
                f"[raw {d[0]:.2f}×{d[1]:.2f}×{d[2]:.2f}m, {len(parts)} part(s)]"
            )
        return parts


# ========================= PLACEMENT MATH =========================
# All functions are pure — they take raw parts + parameters and return
# a 4×4 matrix or (min, max) bounds.  No meshes are mutated.


def _detect_orient_matrix(parts: List[trimesh.Trimesh]) -> np.ndarray:
    """Return +90°X rotation if the OBJ was exported with -90°X, else identity."""
    if not parts:
        return np.eye(4)
    raw_mins = np.min([p.bounds[0] for p in parts], axis=0)
    raw_maxs = np.max([p.bounds[1] for p in parts], axis=0)
    sx, sy, sz = raw_maxs - raw_mins
    if sy > sz:
        return trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    return np.eye(4)


def _oriented_bounds(
    parts: List[trimesh.Trimesh],
    T_orient: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bounding box after T_orient, computed analytically — no vertex mutation."""
    raw_mins = np.min([p.bounds[0] for p in parts], axis=0)
    raw_maxs = np.max([p.bounds[1] for p in parts], axis=0)
    # Transform all 8 corners of the raw AABB and compute the resulting AABB.
    mn, mx = raw_mins, raw_maxs
    corners = np.array([
        [mn[0], mn[1], mn[2], 1.0],
        [mx[0], mn[1], mn[2], 1.0],
        [mn[0], mx[1], mn[2], 1.0],
        [mx[0], mx[1], mn[2], 1.0],
        [mn[0], mn[1], mx[2], 1.0],
        [mx[0], mn[1], mx[2], 1.0],
        [mn[0], mx[1], mx[2], 1.0],
        [mx[0], mx[1], mx[2], 1.0],
    ], dtype=float)
    transformed = (T_orient @ corners.T).T[:, :3]
    return transformed.min(axis=0), transformed.max(axis=0)


def _build_cell_transform(
    T_orient: np.ndarray,
    parts: List[trimesh.Trimesh],
    target_x: float,
    target_z: float,
    center_x: float,
    z_bottom: float,
    y_center: float,
    is_front: bool,
) -> np.ndarray:
    """Compose a 4×4 placement transform for a grid cell.

    Pipeline (right-to-left application order):
      T_orient  — correct export-time axis rotation
      T_center  — translate oriented bbox centre to origin (scale/flip pivot)
      T_scale   — stretch X to target_x, Z to target_z (Y depth unchanged)
      T_flip    — 180° Z rotation for front-facade outward facing (omitted for rear)
      T_pos     — translate to (center_x, y_center, z_bottom + target_z/2)

    Returns identity if parts are degenerate.
    """
    mn, mx = _oriented_bounds(parts, T_orient)
    sx = mx[0] - mn[0]
    sz = mx[2] - mn[2]
    if sx < 1e-6 or sz < 1e-6:
        return np.eye(4)

    oc = (mn + mx) * 0.5          # oriented bbox centre — scale/flip pivot

    T_co = trimesh.transformations.translation_matrix(-oc)
    T_sc = np.diag([target_x / sx, 1.0, target_z / sz, 1.0])
    T_fl = (trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
            if is_front else np.eye(4))
    T_ps = trimesh.transformations.translation_matrix(
        [center_x, y_center, z_bottom + target_z / 2]
    )
    return T_ps @ T_fl @ T_sc @ T_co @ T_orient


def _build_fit_transform(
    T_orient: np.ndarray,
    parts: List[trimesh.Trimesh],
    max_x: float,
    max_z: float,
    center_x: float,
    z_bottom: float,
    y_center: float,
    is_front: bool,
    center_vertically: bool,
) -> np.ndarray:
    """Compose a 4×4 placement transform that fits parts within max_x × max_z.

    Uses uniform scale (preserves aspect ratio, never enlarges).
    If center_vertically, the placed mesh is centred within max_z.
    """
    mn, mx = _oriented_bounds(parts, T_orient)
    sx = mx[0] - mn[0]
    sz = mx[2] - mn[2]
    if sx < 1e-6 or sz < 1e-6:
        return np.eye(4)

    scale    = min(max_x / sx, max_z / sz, 1.0)   # never enlarge
    placed_z = sz * scale
    z_offset = (max_z - placed_z) / 2 if center_vertically else 0.0

    oc = (mn + mx) * 0.5

    T_co = trimesh.transformations.translation_matrix(-oc)
    T_sc = np.diag([scale, scale, scale, 1.0])
    T_fl = (trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
            if is_front else np.eye(4))
    T_ps = trimesh.transformations.translation_matrix(
        [center_x, y_center, z_bottom + z_offset + placed_z / 2]
    )
    return T_ps @ T_fl @ T_sc @ T_co @ T_orient


def _emit(
    meshes: List[trimesh.Trimesh],
    parts: List[trimesh.Trimesh],
    T: np.ndarray,
) -> None:
    """Clone each part, apply composed transform T, append to meshes list."""
    for part in parts:
        m = part.copy()
        m.apply_transform(T)
        meshes.append(m)


# ========================= ASSEMBLER =========================

class GridFacadeAssembler:
    """
    Strict two-phase grid assembler.

    Phase 1 — Planning: build a 2D CellState map with no meshes.
    Phase 2 — Building: emit exactly one wall mesh per non-DOOR cell plus
               balcony overlays on top, and door span meshes for DOOR cells.

    The "window" module slot holds the wall_window composite OBJ produced
    by server.py's create_wall_window_module().  It is used as a full-cell
    replacement for plain wall panels, never as an overlay.

    Final output: a trimesh.Scene in Y-up orientation (Three.js compatible).
    """

    def __init__(self, params: Dict[str, Any], modules_dir: Path):
        self.params = params

        preferred_ids: Dict[str, str] = {}
        if params.get("wall_module_id"):
            preferred_ids["wall"] = params["wall_module_id"]
        if params.get("window_module_id"):
            preferred_ids["window"] = params["window_module_id"]
        if params.get("balcony_module_id"):
            preferred_ids["balcony"] = params["balcony_module_id"]

        self.loader = ModuleLoader(Path(modules_dir), preferred_ids=preferred_ids)

        self.floors:      int = max(1, int(params.get("floors",        5)))
        self.cols:        int = max(1, int(params.get("columns",       10)))
        self.sections:    int = max(0, int(params.get("sections",       3)))
        self.depth_cells: int = max(1, int(params.get("depth",          2)))
        self.texture_scale: int = max(1, min(8, int(params.get("texture_scale", 3))))
        self.balcony_rate: float = max(0.0, min(1.0,
                                       float(params.get("balcony_rate", 0.25))))

        wall_parts = self.loader.raw_parts("wall")
        if not wall_parts:
            raise RuntimeError("Wall module is required but could not be loaded.")

        ww, wd, wh     = self.loader.bbox("wall")
        self.cell_width:  float = ww if ww > 0.01 else 4.0
        self.cell_height: float = wh if wh > 0.01 else 3.0
        self.wall_depth:  float = wd if wd > 0.01 else 0.3

        self.building_width:  float = self.cols        * self.cell_width
        self.building_height: float = self.floors      * self.cell_height
        self.building_depth:  float = self.depth_cells * self.cell_width

        logger.info(
            f"GridFacadeAssembler | grid {self.cols}×{self.floors} | "
            f"cell {self.cell_width:.2f}m×{self.cell_height:.2f}m | "
            f"building {self.building_width:.1f}×"
            f"{self.building_height:.1f}×{self.building_depth:.1f}m | "
            f"texture_scale={self.texture_scale}"
        )

    # ── Helpers ───────────────────────────────────────────────────

    def _entrance_cols(self) -> List[int]:
        if self.sections <= 0:
            return []
        spacing = self.cols / self.sections
        result = []
        for i in range(self.sections):
            col = int(round((i + 0.5) * spacing - 0.5))
            result.append(max(0, min(self.cols - 1, col)))
        return result

    @staticmethod
    def _window_density(texture_scale: int) -> float:
        return round(min(0.90, max(0.10, 0.10 + (texture_scale - 1) * 0.114)), 3)

    @staticmethod
    def _is_window_col(col: int, floor: int, step: int) -> bool:
        if step <= 1:
            return True
        offset = (step // 2) * (floor % 2)
        return ((col + offset) % step) == 0

    @staticmethod
    def _is_balcony_position(candidate_idx: int, floor: int, step: int) -> bool:
        if step <= 1:
            return True
        offset = (step // 2) * (floor % 2)
        return ((candidate_idx + offset) % step) == 0

    # ========================
    # PHASE 1 — PLANNING
    # ========================

    def _plan_step2_doors(self,
                          grid: FacadeGrid,
                          entrance_cols: List[int]) -> List[Tuple[int, int]]:
        placements: List[Tuple[int, int]] = []

        door_parts = self.loader.raw_parts("door")
        if not door_parts:
            logger.warning("Door module missing — entrance columns will be plain wall.")
            return placements

        dw, _, _ = self.loader.bbox("door")
        h_span   = max(1, math.ceil(dw / self.cell_width))

        for center_col in entrance_cols:
            start  = center_col - h_span // 2
            cols_r = list(range(start, start + h_span))

            if any(c < 0 or c >= self.cols for c in cols_r):
                continue
            if not grid.can_reserve(cols_r, [0]):
                continue

            grid.reserve(cols_r, [0], CellState.DOOR)
            placements.append((start, h_span))
            logger.debug(f"Door reserved: cols={cols_r}, floor=0")

        return placements

    def _plan_step3_balconies(self,
                              grid: FacadeGrid,
                              entrance_cols: List[int]) -> List[Tuple[int, int, int, int]]:
        """
        Reserve balcony cells (blocks window promotion) and return placement list.
        BALCONY/BALCONY_UPPER states do NOT suppress wall meshes — only window
        promotion is blocked.
        """
        placements: List[Tuple[int, int, int, int]] = []

        if self.balcony_rate <= 0:
            return placements

        bal_parts = self.loader.raw_parts("balcony")
        if not bal_parts:
            return placements

        bw, _, bh = self.loader.bbox("balcony")
        h_span    = max(1, math.ceil(bw / self.cell_width))
        v_span    = 2 if bh > self.cell_height else 1

        blocked   = set(entrance_cols)
        bal_step  = max(1, round(1.0 / self.balcony_rate))

        for floor in range(1, self.floors):
            if v_span == 2 and floor + 1 >= self.floors:
                continue

            candidates = [
                col for col in range(0, self.cols - h_span + 1, max(1, h_span))
                if not any(c in blocked for c in range(col, col + h_span))
            ]

            for idx, col in enumerate(candidates):
                if not self._is_balcony_position(idx, floor, bal_step):
                    continue

                cols_r  = list(range(col, col + h_span))
                floor_r = [floor, floor + 1] if v_span == 2 else [floor]

                if not grid.can_reserve(cols_r, floor_r):
                    continue

                grid.reserve(cols_r, [floor], CellState.BALCONY)
                if v_span == 2:
                    grid.reserve(cols_r, [floor + 1], CellState.BALCONY_UPPER)

                placements.append((col, floor, h_span, v_span))
                logger.debug(f"Balcony reserved: cols={cols_r}, floor={floor}, v_span={v_span}")

        return placements

    def _plan_step4_windows(self, grid: FacadeGrid, entrance_cols: List[int]) -> None:
        density = self._window_density(self.texture_scale)
        step    = max(1, round(1.0 / density))
        blocked = set(entrance_cols)

        promoted = 0
        for floor in range(self.floors):
            for col in range(self.cols):
                if floor == 0 and col in blocked:
                    continue
                if not grid.is_wall(col, floor):
                    continue
                if self._is_window_col(col, floor, step):
                    grid.promote_to_wall_window(col, floor)
                    promoted += 1

        logger.info(
            f"Window step4: density={density:.2f}, interval={step} cols, "
            f"promoted {promoted} cells to WALL_WINDOW"
        )

    def _plan_facade(self,
                     is_front: bool,
                     entrance_cols: List[int]
                     ) -> Tuple[FacadeGrid,
                                List[Tuple[int, int]],
                                List[Tuple[int, int, int, int]]]:
        grid = FacadeGrid(self.cols, self.floors)

        door_placements: List[Tuple[int, int]] = []
        if is_front:
            door_placements = self._plan_step2_doors(grid, entrance_cols)

        bal_placements = self._plan_step3_balconies(grid, entrance_cols)
        self._plan_step4_windows(grid, entrance_cols)

        return grid, door_placements, bal_placements

    # ========================
    # PHASE 2 — BUILD
    # ========================

    def _build_from_plan(self,
                         grid: FacadeGrid,
                         door_placements: List[Tuple[int, int]],
                         bal_placements: List[Tuple[int, int, int, int]],
                         y_center: float,
                         is_front: bool) -> List[trimesh.Trimesh]:
        """
        Build meshes from the completed plan.

        For every cell: clone raw OBJ part(s), apply a single composed 4×4
        transform (orient → center → scale → flip → position), add to list.
        Raw templates in ModuleLoader cache are NEVER modified.

        Wall-window parts are kept separate (not concatenated) so each retains
        its own TextureVisuals (wall albedo + window atlas).
        """
        meshes: List[trimesh.Trimesh] = []

        wall_parts = self.loader.raw_parts("wall")
        ww_parts   = self.loader.raw_parts("window")    # wall_window composite parts
        door_parts = self.loader.raw_parts("door")
        bal_parts  = self.loader.raw_parts("balcony")

        wall_T = self.loader.orient_matrix("wall")
        ww_T   = self.loader.orient_matrix("window")

        if not wall_parts:
            return meshes

        # ── Per-cell wall / wall_window meshes ────────────────────
        for floor in range(self.floors):
            z_bottom = floor * self.cell_height
            for col in range(self.cols):
                s  = grid.state(col, floor)
                cx = (col + 0.5) * self.cell_width

                if s == CellState.WALL_WINDOW and ww_parts:
                    src_parts = ww_parts
                    T_orient  = ww_T
                else:
                    # WALL, DOOR, BALCONY, BALCONY_UPPER — all emit plain wall
                    src_parts = wall_parts
                    T_orient  = wall_T

                T = _build_cell_transform(
                    T_orient, src_parts,
                    self.cell_width, self.cell_height,
                    cx, z_bottom, y_center, is_front,
                )
                _emit(meshes, src_parts, T)

        # ── Door span overlays ────────────────────────────────────
        if door_parts:
            door_T            = self.loader.orient_matrix("door")
            door_mn, door_mx  = _oriented_bounds(door_parts, door_T)
            dd  = door_mx[1] - door_mn[1]   # oriented door depth (Y)
            dh  = min(door_mx[2] - door_mn[2], self.cell_height)  # oriented height, capped
            # Offset door outward so its back face aligns with wall outer surface.
            outward = -1.0 if is_front else 1.0
            door_y  = y_center + outward * (self.wall_depth / 2.0 + max(dd, 0.01) / 2.0)

            for start_col, h_span in door_placements:
                cx = (start_col + h_span / 2) * self.cell_width
                T  = _build_cell_transform(
                    door_T, door_parts,
                    h_span * self.cell_width, dh,
                    cx, 0.0, door_y, is_front,
                )
                _emit(meshes, door_parts, T)

        # ── Balcony overlays (placed IN FRONT of wall) ───────────
        if bal_parts and bal_placements:
            bal_T           = self.loader.orient_matrix("balcony")
            bal_mn, bal_mx  = _oriented_bounds(bal_parts, bal_T)
            bd  = bal_mx[1] - bal_mn[1]   # oriented balcony depth (Y)
            bh  = bal_mx[2] - bal_mn[2]   # oriented balcony height (Z)
            outward = -1.0 if is_front else 1.0
            bal_y   = y_center + outward * (self.wall_depth / 2.0 + max(bd, 0.01) / 2.0)

            for start_col, floor, h_span, v_span in bal_placements:
                span_w    = h_span * self.cell_width
                z_bottom  = floor  * self.cell_height
                cx        = (start_col + h_span / 2) * self.cell_width
                # v_span==1: fit to one cell height, centre vertically.
                # v_span==2: fit to the balcony's natural height (bh).
                max_z      = self.cell_height if v_span == 1 else bh
                center_v   = v_span == 1
                T = _build_fit_transform(
                    bal_T, bal_parts,
                    span_w, max_z,
                    cx, z_bottom, bal_y, is_front, center_v,
                )
                _emit(meshes, bal_parts, T)

        return meshes

    # ── Front / back facade (full pipeline) ──────────────────────

    def _build_facade(self,
                      y_center: float,
                      is_front: bool,
                      entrance_cols: List[int]) -> List[trimesh.Trimesh]:
        grid, door_pl, bal_pl = self._plan_facade(is_front, entrance_cols)

        wall_count = sum(
            1 for f in range(self.floors) for c in range(self.cols)
            if grid.state(c, f) == CellState.WALL
        )
        ww_count = sum(
            1 for f in range(self.floors) for c in range(self.cols)
            if grid.state(c, f) == CellState.WALL_WINDOW
        )
        door_cells = sum(
            1 for f in range(self.floors) for c in range(self.cols)
            if grid.state(c, f) == CellState.DOOR
        )
        bal_cells = sum(
            1 for f in range(self.floors) for c in range(self.cols)
            if grid.state(c, f) in (CellState.BALCONY, CellState.BALCONY_UPPER)
        )
        label = "front" if is_front else "back"
        logger.info(
            f"{label.capitalize()} facade plan: "
            f"WALL={wall_count} WALL_WINDOW={ww_count} "
            f"DOOR_cells={door_cells} BALCONY_cells={bal_cells}"
        )

        meshes = self._build_from_plan(grid, door_pl, bal_pl, y_center, is_front)
        logger.info(f"{label.capitalize()} facade built: {len(meshes)} meshes")
        return meshes

    # ── Side facades (walls only) ─────────────────────────────────

    def _build_side_facade(self, x_pos: float, is_left: bool) -> List[trimesh.Trimesh]:
        """
        Side walls: scale to cell dimensions, then rotate ±90° around Z so the
        panel width axis (X) becomes the building depth axis (Y).
        Composed as a single matrix applied to copies of the raw template.
        """
        meshes: List[trimesh.Trimesh] = []
        wall_parts = self.loader.raw_parts("wall")
        if not wall_parts:
            logger.error("Wall module missing — side facade empty.")
            return meshes

        wall_T         = self.loader.orient_matrix("wall")
        wall_mn, wall_mx = _oriented_bounds(wall_parts, wall_T)
        sx = wall_mx[0] - wall_mn[0]
        sz = wall_mx[2] - wall_mn[2]
        if sx < 1e-6 or sz < 1e-6:
            return meshes

        oc      = (wall_mn + wall_mx) * 0.5
        scale_x = self.cell_width  / sx
        scale_z = self.cell_height / sz
        angle   = -np.pi / 2 if is_left else np.pi / 2

        T_co   = trimesh.transformations.translation_matrix(-oc)
        T_sc   = np.diag([scale_x, 1.0, scale_z, 1.0])
        T_side = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])

        for floor in range(self.floors):
            z_bottom     = floor * self.cell_height
            for d in range(self.depth_cells):
                depth_center = (d + 0.5) * self.cell_width
                T_ps = trimesh.transformations.translation_matrix(
                    [x_pos, depth_center, z_bottom + self.cell_height / 2]
                )
                T = T_ps @ T_side @ T_sc @ T_co @ wall_T
                _emit(meshes, wall_parts, T)

        label = "left" if is_left else "right"
        logger.info(f"{label.capitalize()} side facade: {len(meshes)} walls")
        return meshes

    # ── Main assembly ─────────────────────────────────────────────

    def assemble_building(self) -> Optional[trimesh.Scene]:
        """
        Assemble all facade meshes into a trimesh.Scene in Y-up orientation.

        Internal assembly uses Z-up (height = Z).  A -90° rotation around X
        is applied to every mesh at the end to convert to Three.js Y-up so the
        exported OBJ renders upright without further client-side transforms.
        """
        entrance_cols = self._entrance_cols()
        logger.info(
            f"Assembly start — entrance cols: {entrance_cols}, "
            f"balcony_rate: {self.balcony_rate:.2f}, "
            f"texture_scale: {self.texture_scale} "
            f"(window density: {self._window_density(self.texture_scale):.2f})"
        )

        all_meshes: List[trimesh.Trimesh] = []

        # Solid structural base volume
        base = trimesh.creation.box(
            extents=[self.building_width, self.building_depth, self.building_height]
        )
        base.apply_translation([
            self.building_width  / 2,
            self.building_depth  / 2,
            self.building_height / 2,
        ])
        all_meshes.append(base)

        # Front facade (y=0, has doors, modules flipped to face outward)
        all_meshes.extend(self._build_facade(0.0,                 True,  entrance_cols))
        # Back facade (y=depth, no doors, default outward orientation is correct)
        all_meshes.extend(self._build_facade(self.building_depth, False, entrance_cols))
        # Side facades (walls only, rotated 90° around Z)
        all_meshes.extend(self._build_side_facade(0.0,                 is_left=True))
        all_meshes.extend(self._build_side_facade(self.building_width, is_left=False))

        # Convert from internal Z-up to Three.js Y-up by rotating -90° around X.
        rot_yup = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])

        scene = trimesh.Scene()
        for i, mesh in enumerate(all_meshes):
            mesh.apply_transform(rot_yup)
            scene.add_geometry(mesh, node_name=f'mesh_{i:04d}')

        total = len(all_meshes)
        logger.info(f"Assembly complete — {total} mesh components in scene")
        return scene

    def export_to_obj(self, output_path: Path) -> bool:
        scene = self.assemble_building()
        if scene is None:
            logger.error("Assembly failed — nothing to export.")
            return False
        try:
            scene.export(str(output_path))
            logger.info(f"Exported: {output_path}")
            return True
        except Exception as exc:
            logger.error(f"Export failed: {exc}", exc_info=True)
            return False


# ========================= PUBLIC API =========================

def assemble_building(params: Dict[str, Any], models_dir: Path, output_path: Path) -> bool:
    """Entry point called by server.py — signature unchanged."""
    assembler = GridFacadeAssembler(params, models_dir)
    return assembler.export_to_obj(output_path)
