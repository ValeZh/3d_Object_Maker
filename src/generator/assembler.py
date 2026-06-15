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

  PHASE 2 — Build (one mesh per cell, no overlaps)
    DOOR         → door module, scaled to span; 180° Z-flip on front facade
    BALCONY      → wall mesh (underlying) + balcony overlay placed in front
    BALCONY_UPPER→ wall mesh (underlying wall continues behind upper span)
    WALL_WINDOW  → wall_window composite, scaled to cell
    WALL         → plain wall, scaled to cell

    After all cell meshes: each facade-mesh is 180° Z-flipped on the FRONT
    facade so modules face outward.

    After all facades are assembled: a -90° X rotation is applied to the
    entire building scene, converting from internal Z-up to Three.js Y-up.

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
        """All non-DOOR cells produce a wall or wall_window mesh in phase 2."""
        return self != CellState.DOOR


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
    """Loads, orients, and caches trimesh objects from OBJ files."""

    def __init__(self, modules_dir: Path, preferred_ids: Optional[Dict[str, str]] = None):
        self.modules_dir = Path(modules_dir)
        # Maps module_type → specific UUID subdirectory to load.
        # When set, the loader targets that UUID directory directly instead of
        # picking the first alphabetical match — critical for freshly generated
        # composites that must not be confused with older modules of the same type.
        self._preferred: Dict[str, str] = preferred_ids or {}
        self._cache: Dict[str, Optional[trimesh.Trimesh]] = {}
        self._parts_cache: Dict[str, List[trimesh.Trimesh]] = {}

    def load(self, module_type: str) -> Optional[trimesh.Trimesh]:
        if module_type not in self._cache:
            self._cache[module_type] = self._from_disk(module_type)
        return self._cache[module_type]

    def load_parts(self, module_type: str) -> List[trimesh.Trimesh]:
        """Return all geometry parts for multi-material modules (e.g. wall_window)."""
        self.load(module_type)  # populates _parts_cache for multi-part modules
        cached = self._parts_cache.get(module_type)
        if cached:
            return cached
        single = self._cache.get(module_type)
        return [single] if single is not None else []

    def bbox(self, module_type: str) -> Tuple[float, float, float]:
        """(width_x, depth_y, height_z) from bounding box."""
        mesh = self.load(module_type)
        if mesh is None:
            return (1.0, 0.3, 1.0)
        b = mesh.bounds
        return (b[1][0] - b[0][0], b[1][1] - b[0][1], b[1][2] - b[0][2])

    def _from_disk(self, module_type: str) -> Optional[trimesh.Trimesh]:
        module_dir = self.modules_dir / module_type
        if not module_dir.exists():
            logger.warning(f"Module dir not found: {module_dir}")
            return None

        # Preferred UUID takes priority; fall back to alphabetical first match.
        preferred_id = self._preferred.get(module_type)
        if preferred_id:
            preferred_path = module_dir / preferred_id / f"{module_type}.obj"
            if preferred_path.exists():
                obj_files = [preferred_path]
                logger.debug(f"Using preferred module {module_type}/{preferred_id}")
            else:
                logger.warning(
                    f"Preferred module {module_type}/{preferred_id} not found at "
                    f"{preferred_path}; falling back to alphabetical search."
                )
                obj_files = sorted(module_dir.glob(f"*/{module_type}.obj"))
        else:
            obj_files = sorted(module_dir.glob(f"*/{module_type}.obj"))

        if not obj_files:
            logger.warning(f"No OBJ for module '{module_type}'")
            return None

        try:
            loaded = trimesh.load(str(obj_files[0]), process=False)
        except Exception as exc:
            logger.error(f"Failed to load {module_type}: {exc}", exc_info=True)
            return None

        # For multi-geometry Scenes, keep parts separate to preserve per-part materials.
        if isinstance(loaded, trimesh.Scene):
            parts = [g for g in loaded.geometry.values()
                     if isinstance(g, trimesh.Trimesh)]
            if not parts:
                return None
            if len(parts) > 1:
                fixed = [self._orient_loaded_mesh(p, module_type) for p in parts]
                self._parts_cache[module_type] = fixed
                ww, wd, wh = fixed[0].bounds[1] - fixed[0].bounds[0]
                logger.info(
                    f"Loaded '{module_type}' ({len(fixed)} parts): "
                    f"{obj_files[0].parent.name}/{obj_files[0].name} "
                    f"[{ww:.2f}×{wd:.2f}×{wh:.2f}m]"
                )
                return fixed[0]
            loaded = parts[0]

        if not isinstance(loaded, trimesh.Trimesh):
            logger.warning(f"Unexpected type for {module_type}: {type(loaded)}")
            return None

        mesh = self._orient_loaded_mesh(loaded, module_type)
        ww, wd, wh = mesh.bounds[1] - mesh.bounds[0]
        logger.info(
            f"Loaded '{module_type}': {obj_files[0].parent.name}/{obj_files[0].name} "
            f"[{ww:.2f}×{wd:.2f}×{wh:.2f}m]"
        )
        return mesh

    def _orient_loaded_mesh(self, mesh: trimesh.Trimesh, module_type: str) -> trimesh.Trimesh:
        """Apply auto Z-up fix unless the assembler handles orientation separately."""
        # Door OBJ is Y-up (height=Y, depth=Z). _build_from_plan applies 90°X (+ Z-flip).
        # Running _fix_orientation first on thin slabs (depth << height) would Z-up them
        # here, then the assembler rotates again — leaving the door flat on the ground.
        if module_type == "door":
            return mesh
        return self._fix_orientation(mesh)

    @staticmethod
    def _fix_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Rotate so Z is the vertical (height) axis — internal Z-up convention."""
        b = mesh.bounds
        sx, sy, sz = b[1] - b[0]
        if sz < 0.3 * max(sx, sy, 1e-6) and sy > sz:
            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            mesh.apply_transform(rot)
        return mesh


# ========================= MESH UTILITIES =========================

def _bbox_center(mesh: trimesh.Trimesh) -> np.ndarray:
    return (mesh.bounds[0] + mesh.bounds[1]) * 0.5


def _scale_exact(mesh: trimesh.Trimesh, target_x: float, target_z: float) -> None:
    """Scale mesh so X span = target_x, Z span = target_z. In-place."""
    b   = mesh.bounds
    sx  = b[1][0] - b[0][0]
    sz  = b[1][2] - b[0][2]
    if sx < 1e-6 or sz < 1e-6:
        return
    c = _bbox_center(mesh)
    mesh.apply_translation(-c)
    mesh.apply_transform(np.diag([target_x / sx, 1.0, target_z / sz, 1.0]))
    mesh.apply_translation(c)


def _scale_fit(mesh: trimesh.Trimesh, max_x: float, max_z: float) -> None:
    """Scale mesh DOWN to fit within max_x × max_z, aspect ratio preserved. In-place."""
    b  = mesh.bounds
    sx = b[1][0] - b[0][0]
    sz = b[1][2] - b[0][2]
    if sx < 1e-6 or sz < 1e-6:
        return
    scale = min(max_x / sx, max_z / sz)
    if scale >= 1.0:
        return
    c = _bbox_center(mesh)
    mesh.apply_translation(-c)
    mesh.apply_transform(np.diag([scale, scale, scale, 1.0]))
    mesh.apply_translation(c)


def _position(mesh: trimesh.Trimesh,
              center_x: float,
              z_bottom: float,
              y_center: float) -> None:
    """Move mesh: X center = center_x, Z bottom = z_bottom, Y center = y_center."""
    b = mesh.bounds
    mesh.apply_translation([
        center_x - (b[0][0] + b[1][0]) * 0.5,
        y_center  - (b[0][1] + b[1][1]) * 0.5,
        z_bottom  - b[0][2],
    ])


def _flip_facing(mesh: trimesh.Trimesh) -> None:
    """
    Rotate 180° around Z axis so the module faces the other Y direction.
    Applied to front-facade modules so they face outward instead of inward.
    In-place, pivoted around the mesh's own bounding-box centre.
    """
    c = _bbox_center(mesh)
    mesh.apply_translation(-c)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
    mesh.apply_translation(c)


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
        if params.get("wall_window_module_id"):
            preferred_ids["wall_window"] = params["wall_window_module_id"]
        if params.get("balcony_module_id"):
            preferred_ids["balcony"] = params["balcony_module_id"]
        if params.get("door_module_id"):
            preferred_ids["door"] = params["door_module_id"]
        if params.get("roof_module_id"):
            preferred_ids["roof"] = params["roof_module_id"]

        self.loader = ModuleLoader(Path(modules_dir), preferred_ids=preferred_ids)

        self.floors:      int = max(1, int(params.get("floors",        5)))
        self.cols:        int = max(1, int(params.get("columns",       10)))
        self.sections:    int = max(0, int(params.get("sections",       3)))
        self.depth_cells: int = max(1, int(params.get("depth",          2)))
        self.texture_scale: int = max(1, min(8, int(params.get("texture_scale", 3))))
        self.balcony_rate: float = max(0.0, min(1.0,
                                       float(params.get("balcony_rate", 0.25))))

        self._export_dir: Optional[Path] = None

        wall = self.loader.load("wall")
        if wall is None:
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

        door_orig = self.loader.load("door")
        if door_orig is None:
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

    def _build_house_flat_roof(
        self,
        roof_w: float,
        roof_d: float,
        roof_thickness: float,
        roof_params: Dict[str, Any],
    ) -> trimesh.Trimesh:
        """Flat roof slab with UV + visible material for house export."""
        from PIL import Image
        from trimesh.visual.material import SimpleMaterial
        from trimesh.visual.texture import TextureVisuals

        from src.generator.procedural.procedural_roof import (
            _resolve_roof_textures,
            build_flat_roof_mesh,
        )
        from src.generator.procedural.unfolding import faceted_triplanar_uv
        from src.generator.procedural.texturing.color_tint import parse_texture_color_tint

        mesh = build_flat_roof_mesh(roof_w, roof_d, roof_thickness)
        mesh_uv, uv = faceted_triplanar_uv(mesh)

        tex_img: Optional[Image.Image] = None
        if self._export_dir is not None:
            tex_dir = self._export_dir / "textures"
            tex_dir.mkdir(parents=True, exist_ok=True)

            roof_id = self.params.get("roof_module_id")
            if roof_id:
                mod_dir = self.loader.modules_dir / "roof" / str(roof_id)
                for candidate in (
                    mod_dir / "roof_diffuse.png",
                    mod_dir / "_proc_roof_diffuse.png",
                ):
                    if candidate.exists():
                        tex_img = Image.open(candidate).convert("RGB")
                        break

            if tex_img is None:
                tint = None
                raw_color = roof_params.get("color") or roof_params.get("roof_color")
                if isinstance(raw_color, str) and raw_color.strip().startswith("#"):
                    tint = parse_texture_color_tint(raw_color.strip())
                _resolve_roof_textures(
                    tex_dir,
                    roof_texture=None,
                    roof_normal_texture=None,
                    roof_roughness_texture=None,
                    roof_texture_color=tint,
                    use_procedural_maps=True,
                    roof_color_preset=str(
                        roof_params.get("roof_color_preset", "roof_shingles")
                    ),
                    bump_strength=0.7,
                )
                for candidate in (
                    tex_dir / "roof_diffuse.png",
                    tex_dir / "_proc_roof_diffuse.png",
                ):
                    if candidate.exists():
                        tex_img = Image.open(candidate).convert("RGB")
                        break

        if tex_img is not None:
            mesh_uv.visual = TextureVisuals(
                uv=uv,
                material=SimpleMaterial(name="roof", image=tex_img),
            )
        else:
            mesh_uv.visual = trimesh.visual.ColorVisuals(
                mesh=mesh_uv,
                face_colors=np.tile(
                    np.array([122, 82, 61, 255], dtype=np.uint8),
                    (len(mesh_uv.faces), 1),
                ),
            )

        return mesh_uv

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

        bal_orig = self.loader.load("balcony")
        if bal_orig is None:
            return placements

        bw, _, bh = self.loader.bbox("balcony")
        h_span    = 1  # always one cell; mesh is scaled to fit by _scale_fit
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

        Every non-DOOR cell gets a wall or wall_window mesh.
        BALCONY/BALCONY_UPPER cells get a plain wall mesh (the balcony overlaid
        on top of it is placed as a separate overlay mesh with a Y offset so it
        sits in front of the wall surface — not replacing it).

        Front-facade modules are flipped 180° around Z so they face outward.
        """
        meshes: List[trimesh.Trimesh] = []

        wall_orig = self.loader.load("wall")
        ww_orig   = self.loader.load("wall_window")   # wall_window composite
        door_orig = self.loader.load("door")
        bal_orig  = self.loader.load("balcony")

        # ── Per-cell wall / wall_window meshes ────────────────────
        for floor in range(self.floors):
            z_bottom = floor * self.cell_height
            for col in range(self.cols):
                s = grid.state(col, floor)

                if s == CellState.DOOR:
                    continue  # handled as span mesh below

                if s == CellState.WALL_WINDOW:
                    ww_parts = self.loader.load_parts("wall_window")
                    if len(ww_parts) > 1:
                        copies = [p.copy() for p in ww_parts]
                        cb_min = np.min([m.bounds[0] for m in copies], axis=0)
                        cb_max = np.max([m.bounds[1] for m in copies], axis=0)
                        if not is_front:
                            cc = (cb_min + cb_max) * 0.5
                            rot = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
                            for m in copies:
                                m.apply_translation(-cc)
                                m.apply_transform(rot)
                                m.apply_translation(cc)
                            cb_min = np.min([m.bounds[0] for m in copies], axis=0)
                            cb_max = np.max([m.bounds[1] for m in copies], axis=0)
                        sx = cb_max[0] - cb_min[0]
                        sz = cb_max[2] - cb_min[2]
                        if sx > 1e-6 and sz > 1e-6:
                            scale = min(self.cell_width / sx, self.cell_height / sz)
                            if scale < 1.0:
                                cc = (cb_min + cb_max) * 0.5
                                s_mat = np.diag([scale, scale, scale, 1.0])
                                for m in copies:
                                    m.apply_translation(-cc)
                                    m.apply_transform(s_mat)
                                    m.apply_translation(cc)
                                cb_min = np.min([m.bounds[0] for m in copies], axis=0)
                                cb_max = np.max([m.bounds[1] for m in copies], axis=0)
                        t = np.array([
                            (col + 0.5) * self.cell_width - (cb_min[0] + cb_max[0]) * 0.5,
                            y_center - (cb_min[1] + cb_max[1]) * 0.5,
                            z_bottom - cb_min[2],
                        ])
                        for m in copies:
                            m.apply_translation(t)
                        meshes.extend(copies)
                        continue
                    src = ww_parts[0] if ww_parts else (ww_orig if ww_orig is not None else wall_orig)
                else:
                    # WALL, BALCONY, BALCONY_UPPER all get a plain wall
                    src = wall_orig

                if src is None:
                    continue

                m = src.copy()
                c = _bbox_center(m)
                m.apply_translation(-c)
                m.apply_translation(c)
                if not is_front:
                    _flip_facing(m)
                _scale_fit(m, self.cell_width, self.cell_height)
                _position(m, (col + 0.5) * self.cell_width, z_bottom, y_center)
                meshes.append(m)

        # ── Door span meshes ──────────────────────────────────────
        # Wall behind each door cell so the wall surface is visible through/around doors.
        if wall_orig is not None:
            for start_col, h_span in door_placements:
                for dc in range(h_span):
                    wall_behind = wall_orig.copy()
                    if not is_front:
                        _flip_facing(wall_behind)
                    _scale_fit(wall_behind, self.cell_width, self.cell_height)
                    _position(wall_behind, (start_col + dc + 0.5) * self.cell_width, 0.0, y_center)
                    meshes.append(wall_behind)

        # Door mesh offset slightly forward so it protrudes from the wall face.
        # OBJ is Y-up: index 1 = height, index 2 = depth.
        # After 90°X + 180°Z rotations: OBJ-Y→Z, OBJ-Z→Y, so we read depth from index 2.
        _, _, door_depth = self.loader.bbox("door")  # OBJ Z span = depth (lands in Y after rotations)
        _door_y_offset = door_depth / 2 + 0.18
        door_y = y_center - _door_y_offset if is_front else y_center + _door_y_offset
        if door_orig is not None:
            _, dh, _ = self.loader.bbox("door")  # OBJ Y span = height (lands in Z after 90°X rotation)
            for start_col, h_span in door_placements:
                door = door_orig.copy()

                c = _bbox_center(door)
                door.apply_translation(-c)
                door.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
                door.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
                door.apply_translation(c)

                _scale_exact(door, h_span * self.cell_width, min(dh, self.cell_height))
                if is_front:
                    _flip_facing(door)
                cx = (start_col + h_span / 2) * self.cell_width
                _position(door, cx, 0.0, door_y)
                meshes.append(door)

        # ── Balcony overlay meshes (placed IN FRONT of wall) ─────
        if bal_orig is not None and bal_placements:
            _, bd, bh = self.loader.bbox("balcony")
            # Outward direction: front facade → -Y; back facade → +Y
            outward = -1.0 if is_front else 1.0
            # Place balcony so its back face meets the wall's outer face
            bal_y = y_center + outward * (self.wall_depth / 2.0 + max(bd, 0.01) / 2.0 - 1.45)

            for start_col, floor, h_span, v_span in bal_placements:
                bal      = bal_orig.copy()
                span_w   = h_span * self.cell_width
                z_bottom = floor  * self.cell_height

                c = _bbox_center(bal)
                bal.apply_translation(-c)
                bal.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
                bal.apply_translation(c)

                if not is_front:
                    _flip_facing(bal)

                if v_span == 1:
                    _scale_fit(bal, span_w, self.cell_height)
                    placed_h = bal.bounds[1][2] - bal.bounds[0][2]
                    z_bottom += (self.cell_height - placed_h) / 2
                else:
                    _scale_fit(bal, span_w, bh)

                cx = (start_col + h_span / 2) * self.cell_width
                _position(bal, cx, z_bottom, bal_y)
                meshes.append(bal)

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
        meshes: List[trimesh.Trimesh] = []
        wall_orig = self.loader.load("wall")
        if wall_orig is None:
            logger.error("Wall module missing — side facade empty.")
            return meshes

        angle = -np.pi / 2 if is_left else np.pi / 2


        for floor in range(self.floors):
            z_bottom = floor * self.cell_height
            for d in range(self.depth_cells):
                w = wall_orig.copy()
                _scale_exact(w, self.cell_width, self.cell_height)

                c = _bbox_center(w)
                w.apply_translation(-c)
                w.apply_transform(
                    trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
                )
                w.apply_translation(c)

                b = w.bounds
                w.apply_translation([
                    x_pos                             - (b[0][0] + b[1][0]) * 0.5,
                    (d + 0.5) * self.cell_width       - (b[0][1] + b[1][1]) * 0.5,
                    z_bottom                          - b[0][2],
                ])
                meshes.append(w)

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

        # ── Roof ──────────────────────────────────────────────────────
        roof_params = self.params.get("roof_params") or {}
        roof_type = str(roof_params.get("roof_type", "flat")).strip().lower()
        flat_kinds = ("flat", "plate", "slab", "плоская", "плоская крыша")

        if roof_type in flat_kinds:
            roof_thickness = max(0.35, float(roof_params.get("height", 0.45)))
            overhang = float(roof_params.get("overhang", 0.4))
            roof_w = self.building_width + 2 * overhang
            roof_d = self.building_depth + 2 * overhang
            roof = self._build_house_flat_roof(
                roof_w, roof_d, roof_thickness, roof_params
            )
            # Lift slightly above the structural base top to avoid z-fighting.
            _position(
                roof,
                self.building_width / 2,
                self.building_height + 0.02,
                self.building_depth / 2,
            )
            all_meshes.append(roof)
            logger.info(
                f"Flat roof added — {roof_w:.1f}×{roof_d:.1f}×{roof_thickness:.2f}m"
            )
        else:
            # Gable / pyramid — load roof module OBJ and scale to footprint.
            roof_orig = self.loader.load("roof")
            if roof_orig is not None:
                roof = roof_orig.copy()

                roof_x = self.building_width
                roof_y = self.building_depth
                roof_z = float(roof_params.get("height", 3.0))

                bw, bd, bh = self.loader.bbox("roof")
                if bw > 1e-6 and bd > 1e-6 and bh > 1e-6:
                    c = _bbox_center(roof)
                    roof.apply_translation(-c)
                    roof.apply_transform(np.diag([
                        roof_x / bw,
                        roof_y / bd,
                        roof_z / bh,
                        1.0,
                    ]))
                    roof.apply_translation(c)

                _position(
                    roof,
                    self.building_width / 2,
                    self.building_height,
                    self.building_depth / 2,
                )

                all_meshes.append(roof)
                logger.info(
                    f"Roof added — type={roof_type!r}, "
                    f"target {roof_x:.1f}×{roof_y:.1f}×{roof_z:.1f}m"
                )
            else:
                logger.warning("Roof module not found — building exported without roof")

        # Convert from internal Z-up to Three.js Y-up by rotating -90° around X.
        rot_yup = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])

        scene = trimesh.Scene()
        for i, mesh in enumerate(all_meshes):
            mesh.apply_transform(rot_yup)
            scene.add_geometry(mesh, node_name=f'mesh_{i:04d}')

        total = len(all_meshes)
        logger.info(f"Assembly complete — {total} mesh components in scene")
        return scene

    def _prepare_for_export(self, output_path: Path) -> None:
        """Copy all textures and create combined MTL with correct paths."""
        import shutil

        output_dir = output_path.parent
        texture_dir = output_dir / "textures"
        texture_dir.mkdir(exist_ok=True)

        # Copy ALL PNG textures from all module types
        copied = set()
        for module_type in ["wall", "window", "door", "balcony", "wall_window", "roof"]:
            type_dir = self.loader.modules_dir / module_type
            if type_dir.exists():
                for png in type_dir.rglob("*.png"):
                    if png.name not in copied:
                        shutil.copy2(png, texture_dir / png.name)
                        copied.add(png.name)

        # Flat-roof procedural maps generated during assembly
        for name in ("roof_diffuse.png", "roof_normal.png", "roof_roughness.png",
                     "_proc_roof_diffuse.png", "_proc_roof_normal.png", "_proc_roof_roughness.png"):
            src = texture_dir / name
            if src.exists():
                copied.add(name)

        logger.info(f"Copied {len(copied)} textures to {texture_dir}")

        # Create combined MTL with correct paths
        mtl_path = output_dir / "house.mtl"
        mtl_content = ""

        for module_type in ["wall", "window", "door", "balcony", "wall_window", "roof"]:
            type_dir = self.loader.modules_dir / module_type
            if type_dir.exists():
                for mtl_file in type_dir.rglob("*.mtl"):
                    content = mtl_file.read_text()
                    content = content.replace("map_Kd ", "map_Kd textures/")
                    content = content.replace("map_Bump ", "map_Bump textures/")
                    content = content.replace("map_Pr ", "map_Pr textures/")
                    mtl_content += content + "\n"

        roof_diffuse = texture_dir / "roof_diffuse.png"
        if not roof_diffuse.exists():
            roof_diffuse = texture_dir / "_proc_roof_diffuse.png"
        if roof_diffuse.exists():
            mtl_content += (
                "newmtl roof\n"
                "Ka 1 1 1\n"
                "Kd 0.48 0.32 0.24\n"
                "Ks 0 0 0\n"
                f"map_Kd textures/{roof_diffuse.name}\n\n"
            )

        if mtl_content:
            mtl_path.write_text(mtl_content)
            logger.info(f"Created MTL: {mtl_path}")

    def export_to_obj(self, output_path: Path) -> bool:
        self._export_dir = output_path.parent
        scene = self.assemble_building()
        if scene is None:
            logger.error("Assembly failed — nothing to export.")
            return False
        try:
            self._prepare_for_export(output_path)
            scene.export(str(output_path))
            self._ensure_roof_obj_material(output_path)
            import shutil
            roof_png = output_path.parent / "roof.png"
            texture_dir = output_path.parent / "textures"
            if roof_png.exists() and texture_dir.exists():
                shutil.copy2(roof_png, texture_dir / "roof_diffuse.png")
            logger.info(f"Exported: {output_path}")
            return True
        except Exception as exc:
            logger.error(f"Export failed: {exc}", exc_info=True)
            return False

    def _ensure_roof_obj_material(self, output_path: Path) -> None:
        """Fallback: tag the last geometry block with roof material if export omitted it."""
        obj_path = Path(output_path)
        mtl_path = obj_path.parent / "material.mtl"
        if not obj_path.exists():
            return

        text = obj_path.read_text(encoding="utf-8")
        if "usemtl roof" in text:
            return

        lines = text.splitlines()
        last_geo = max((i for i, line in enumerate(lines) if line.startswith("o ")), default=-1)
        if last_geo < 0:
            return

        patched: List[str] = []
        for idx, line in enumerate(lines):
            patched.append(line)
            if idx == last_geo:
                patched.append("usemtl roof")
        obj_path.write_text("\n".join(patched) + "\n", encoding="utf-8")

        if mtl_path.exists() and "newmtl roof" not in mtl_path.read_text(encoding="utf-8"):
            roof_png = obj_path.parent / "roof.png"
            map_line = f"map_Kd {roof_png.name}\n" if roof_png.exists() else ""
            mtl_path.write_text(
                mtl_path.read_text(encoding="utf-8")
                + "\nnewmtl roof\nKa 1 1 1\nKd 0.48 0.32 0.24\nKs 0 0 0\n"
                + map_line,
                encoding="utf-8",
            )


# ========================= PUBLIC API =========================

def assemble_building(params: Dict[str, Any], models_dir: Path, output_path: Path) -> bool:
    """Entry point called by server.py — signature unchanged."""
    assembler = GridFacadeAssembler(params, models_dir)
    return assembler.export_to_obj(output_path)
