"""Развёртки (UV) для процедурных мешей: faceted triplanar, triplanar стена."""

from src.generator.procedural.unfolding.faceted_uv import faceted_triplanar_uv, frame_glass_atlas_uv_mesh
from src.generator.procedural.unfolding.wall_triplanar import wall_mesh_expanded_uv

__all__ = ["faceted_triplanar_uv", "frame_glass_atlas_uv_mesh", "wall_mesh_expanded_uv"]
