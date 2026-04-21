"""Текстуры и экспорт материалов/OBJ для процедурных объектов."""

from src.generator.procedural.texturing.color_tint import (
    apply_texture_color_tint,
    parse_texture_color_tint,
)
from src.generator.procedural.texturing.height_map_paths import (
    data_height_maps_dir,
    default_height_maps_dir,
    resolve_height_map_in_defaults,
    resolve_height_map_path,
)
from src.generator.procedural.texturing.entrance_atlas import (
    ENTRANCE_ATLAS_NUM_TILES,
    TILE_DOOR,
    TILE_ROOF,
    TILE_WALL,
    concatenate_entrance_uv_meshes,
    entrance_part_tile_index,
    make_entrance_atlas,
    scale_uv_to_atlas_tile,
)
from src.generator.procedural.texturing.wall_window_obj_export import write_wall_window_mtl, write_wall_window_obj
from src.generator.procedural.texturing.window_texture_assets import (
    default_textures_dir,
    ensure_window_textures,
    make_atlas_from_sources,
    make_normal_atlas_from_sources,
    make_window_atlas,
    make_window_frame_texture,
    make_window_glass_texture,
    resolve_texture_path,
)

__all__ = [
    "apply_texture_color_tint",
    "parse_texture_color_tint",
    "data_height_maps_dir",
    "default_height_maps_dir",
    "resolve_height_map_in_defaults",
    "resolve_height_map_path",
    "ENTRANCE_ATLAS_NUM_TILES",
    "TILE_DOOR",
    "TILE_ROOF",
    "TILE_WALL",
    "concatenate_entrance_uv_meshes",
    "entrance_part_tile_index",
    "make_entrance_atlas",
    "scale_uv_to_atlas_tile",
    "default_textures_dir",
    "ensure_window_textures",
    "make_atlas_from_sources",
    "make_normal_atlas_from_sources",
    "make_window_atlas",
    "make_window_frame_texture",
    "make_window_glass_texture",
    "resolve_texture_path",
    "write_wall_window_mtl",
    "write_wall_window_obj",
]
