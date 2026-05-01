"""Процедурные цветовые текстуры и карты нормалей (без отдельной карты высот как продукта пайплайна)."""

from src.generator.procedural.procedural_texture_maps.normal_map import (
    make_ceramic_tile_normal_map,
    make_fine_noise_normal_map,
    make_neutral_flat_normal_map,
    make_soft_frosted_glass_normal_map,
    make_stucco_like_normal_map,
    make_wood_grain_normal_map,
    normals_from_scalar_slopes,
)
from src.generator.procedural.procedural_texture_maps.procedural_color_texture import (
    make_ceramic_tile_color_texture,
    make_plaster_facade_texture,
    make_uniform_noise_texture,
    make_vertical_stripes_texture,
    make_wood_plank_color_texture,
)

__all__ = [
    "make_fine_noise_normal_map",
    "make_ceramic_tile_color_texture",
    "make_ceramic_tile_normal_map",
    "make_neutral_flat_normal_map",
    "make_plaster_facade_texture",
    "make_soft_frosted_glass_normal_map",
    "make_stucco_like_normal_map",
    "make_uniform_noise_texture",
    "make_vertical_stripes_texture",
    "make_wood_grain_normal_map",
    "make_wood_plank_color_texture",
    "normals_from_scalar_slopes",
]
