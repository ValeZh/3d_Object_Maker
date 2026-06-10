"""
Сборка секции конфига процедурного оркестратора из params API/UI.
Полная схема: docs/procedural_orchestrator_params.txt
Поддерживаемые ключи экспортёров передаются плоским dict и/или через вложенный ``texture``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

ALL_BATCH_SECTIONS = ("balcony", "entrance", "entrance_textured", "window", "wall", "wall_window")

__all__ = [
    "API_TO_BATCH_SECTION",
    "ALL_BATCH_SECTIONS",
    "assemble_full_disabled_config",
    "build_section_for_api_module",
    "deep_merge",
]


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v) if isinstance(v, (dict, list)) else v
    return out


API_TO_BATCH_SECTION: Dict[str, str] = {
    "wall": "wall",
    "window": "window",
    "door": "entrance",
    "balcony": "balcony",
    "entrance": "entrance_textured",
    "wall_window": "wall_window",
}


def _split_overlay(params: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    flat = {k: copy.deepcopy(v) for k, v in params.items() if k != "texture"}
    tex = params.get("texture")
    if isinstance(tex, dict):
        return flat, copy.deepcopy(tex)
    return flat, None


def _maybe_hex_rgb(hex_to_rgb: Callable[[str], list], hex_str: Optional[str]) -> Optional[list]:
    if not isinstance(hex_str, str) or not hex_str.strip().startswith("#"):
        return None
    return hex_to_rgb(hex_str.strip())


def _load_json(p: Path) -> Dict[str, Any]:
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def build_section_for_api_module(
    module_type_api: str,
    *,
    params: Mapping[str, Any],
    output_dir: Path,
    project_root: Path,
    hex_to_rgb: Callable[[str], list],
    batch_defaults_path: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any]]:
    section_key = API_TO_BATCH_SECTION.get(module_type_api)
    if not section_key:
        raise ValueError(f"Unknown API module_type: {module_type_api}")

    cfg_path = batch_defaults_path or (
        project_root / "scripts" / "balcony_examples" / "batch_generators_config.json"
    )
    tpl_root = _load_json(cfg_path)
    sub = copy.deepcopy(tpl_root[section_key]) if isinstance(tpl_root.get(section_key), dict) else {}
    overlay_flat, overlay_tex = _split_overlay(params)
    sub_wo_ent = {
        k: v
        for k, v in sub.items()
        if k != "texture" and (module_type_api != "door" or k != "enabled")
    }

    section: Dict[str, Any]

    if module_type_api == "wall":
        defaults = {
            "enabled": True,
            "out_dir": str(output_dir),
            "wall_length": float(overlay_flat.get("wall_length") or overlay_flat.get("width") or 2.0),
            "wall_thickness": float(overlay_flat.get("wall_thickness") or overlay_flat.get("thickness") or 0.3),
            "wall_height": float(overlay_flat.get("wall_height") or overlay_flat.get("height") or 3.0),
            "no_view": True,
            "texture": {
                "use_procedural_maps": True,
                "wall_color_preset": "plaster",
                "generate_normal": True,
                "generate_roughness": True,
            },
        }
        section = deep_merge(defaults, sub_wo_ent)
        section = deep_merge(section, overlay_flat)
        if overlay_tex:
            section["texture"] = deep_merge(section.get("texture") or {}, overlay_tex)
        rgb = _maybe_hex_rgb(hex_to_rgb, overlay_flat.get("color")) or _maybe_hex_rgb(
            hex_to_rgb, overlay_flat.get("wall_color")
        )
        if rgb:
            section.setdefault("texture", {}).setdefault("wall_tex_color", rgb)

    elif module_type_api == "window":
        defaults = {
            "enabled": True,
            "out_dir": str(output_dir),
            "width": float(overlay_flat.get("width") or 1.5),
            "height": float(overlay_flat.get("height") or 1.2),
            "depth": float(overlay_flat.get("depth") or 0.12),
            "profile": str(overlay_flat.get("profile") or "rect"),
            "kind": str(overlay_flat.get("kind") or "fixed"),
            "mullions_vertical": int(overlay_flat.get("mullions_vertical", 1)),
            "mullions_horizontal": int(overlay_flat.get("mullions_horizontal", 0)),
            "atlas_half_size": int(overlay_flat.get("atlas_half_size", 256)),
            "no_view": True,
            "texture": {
                "use_procedural_maps": True,
                "frame_color_preset": "plaster",
                "glass_color_preset": "uniform_noise",
                "frame_normal_preset": "fine_noise",
                "generate_normal": True,
                "generate_roughness": True,
            },
        }
        section = deep_merge(defaults, sub_wo_ent)
        section = deep_merge(section, overlay_flat)
        if overlay_tex:
            section["texture"] = deep_merge(section.get("texture") or {}, overlay_tex)
        hf = overlay_flat.get("color") or overlay_flat.get("frame_color")
        hg = overlay_flat.get("glass_color")
        r1 = _maybe_hex_rgb(hex_to_rgb, hf) if isinstance(hf, str) else None
        r2 = _maybe_hex_rgb(hex_to_rgb, hg) if isinstance(hg, str) else None
        tex = section.setdefault("texture", {})
        if r1 is not None:
            tex.setdefault("frame_tex_color", r1)
        if r2 is not None:
            tex.setdefault("glass_tex_color", r2)

    elif module_type_api == "door":
        defaults = {
            "enabled": True,
            "out_dir": str(output_dir),
            "entrance_style": str(overlay_flat.get("entrance_style") or "canopy"),
            "width": float(overlay_flat.get("width") or 2.0),
            "depth": float(overlay_flat.get("depth") or 1.75),
            "has_left_wall": bool(overlay_flat.get("has_left_wall", True)),
            "has_right_wall": bool(overlay_flat.get("has_right_wall", True)),
            "doors": overlay_flat.get("doors")
            or [{"u0": 0.1, "u1": 0.9, "z_bottom": 0.12, "z_top": 2.05}],
            "no_view": True,
        }
        section = deep_merge(defaults, sub_wo_ent)
        section = deep_merge(section, overlay_flat)
        if overlay_tex:
            section["texture"] = deep_merge(section.get("texture") or {}, overlay_tex)

    elif module_type_api == "balcony":
        ceramic = project_root / "scripts" / "balcony_examples" / "batch_balcony_ceramic_tile.json"
        tpl_b = copy.deepcopy(_load_json(ceramic).get("balcony") or {})
        defaults = {"enabled": True, "out_dir": str(output_dir), "no_view": True}
        section = deep_merge(tpl_b, {})
        section = deep_merge(section, defaults)
        section = deep_merge(section, overlay_flat)
        if overlay_tex:
            section["texture"] = deep_merge(section.get("texture") or {}, overlay_tex)
        wf = float(section.get("width_front") or overlay_flat.get("width") or 2.0)
        wb = float(section.get("width_back", wf))
        if overlay_flat.get("width") is not None:
            wf = wb = float(overlay_flat["width"])
        section["width_front"], section["width_back"] = wf, wb
        if overlay_flat.get("depth") is not None:
            section["depth"] = float(overlay_flat["depth"])
        if overlay_flat.get("height") is not None:
            section["height"] = float(overlay_flat["height"])
        if overlay_flat.get("has_roof") is not None:
            section["has_roof"] = bool(overlay_flat["has_roof"])
        elif str(overlay_flat.get("style", "")).lower() in ("enclosed", "closed", "лоджия", "закрытый"):
            section["has_roof"] = True
        if overlay_flat.get("roof_thickness") is not None:
            section["roof_thickness"] = float(overlay_flat["roof_thickness"])
        if overlay_flat.get("roof_overhang") is not None:
            section["roof_overhang"] = float(overlay_flat["roof_overhang"])
        roof_hex = overlay_flat.get("roof_color")
        roof_rgb = _maybe_hex_rgb(hex_to_rgb, roof_hex) if isinstance(roof_hex, str) else None
        if roof_rgb:
            section.setdefault("roof_tex_color", roof_rgb)
        hx = overlay_flat.get("color") or overlay_flat.get("frame_color")
        rgb = _maybe_hex_rgb(hex_to_rgb, hx) if isinstance(hx, str) else None
        if rgb:
            for k in (
                "wall_lower_tex_color",
                "wall_upper_tex_color",
                "side_jamb_tex_color",
                "side_separator_tex_color",
                "side_basket_tex_color",
            ):
                section.setdefault(k, rgb)
        fg = overlay_flat.get("glass_color")
        grgb = _maybe_hex_rgb(hex_to_rgb, fg) if isinstance(fg, str) else None
        if grgb:
            section.setdefault("glass_tex_color", grgb)

        iw = overlay_flat.get("inner_wall_windows")
        if (
            isinstance(iw, list)
            and len(iw) > 0
            and "inner_wall_doors" not in overlay_flat
        ):
            section["inner_wall_doors"] = []

    elif module_type_api == "entrance":
        defaults = {
            "enabled": True,
            "out_dir": str(output_dir),
            "atlas_tile": int(overlay_flat.get("atlas_tile", 256)),
            "width": float(overlay_flat.get("width", 3.6)),
            "depth": float(overlay_flat.get("depth", 1.75)),
            "entrance_style": str(overlay_flat.get("entrance_style") or "canopy"),
            "no_view": True,
            "texture": {
                "use_procedural_maps": True,
                "wall_color_preset": "plaster",
                "roof_color_preset": "plaster",
                "door_color_preset": "wood",
                "generate_normal": True,
                "generate_roughness": True,
            },
        }
        section = deep_merge(defaults, sub_wo_ent)
        section = deep_merge(section, overlay_flat)
        if overlay_tex:
            section["texture"] = deep_merge(section.get("texture") or {}, overlay_tex)
        c = overlay_flat.get("color") or overlay_flat.get("wall_color")
        rgb = _maybe_hex_rgb(hex_to_rgb, c) if isinstance(c, str) else None
        if rgb:
            section.setdefault("texture", {}).setdefault("wall_tex_color", rgb)
        dd = overlay_flat.get("door_color")
        drgb = _maybe_hex_rgb(hex_to_rgb, dd) if isinstance(dd, str) else None
        if drgb:
            section.setdefault("texture", {}).setdefault("door_tex_color", drgb)

    elif module_type_api == "wall_window":
        wlen = float(
            overlay_flat.get("wall_length")
            or overlay_flat.get("panel_length")
            or overlay_flat.get("width")
            or 4.0
        )
        wt = float(overlay_flat.get("wall_thickness") or overlay_flat.get("thickness") or 0.25)
        wh = float(overlay_flat.get("wall_height") or overlay_flat.get("panel_height") or 3.0)
        ww = float(overlay_flat.get("window_width") or 1.2)
        whh = float(overlay_flat.get("window_height") or 1.5)
        sill = float(overlay_flat.get("window_sill_z") or overlay_flat.get("sill_z") or 1.0)
        cx = float(overlay_flat.get("window_center_x") or 0.0)
        defaults = {
            "enabled": True,
            "out_dir": str(output_dir),
            "wall_length": wlen,
            "wall_thickness": wt,
            "wall_height": wh,
            "window_center_x": cx,
            "window_sill_z": sill,
            "width": ww,
            "height": whh,
            "depth": float(overlay_flat.get("window_depth") or 0.12),
            "mullions_vertical": int(overlay_flat.get("mullions_vertical", 1)),
            "mullions_horizontal": int(overlay_flat.get("mullions_horizontal", 0)),
            "atlas_half_size": int(overlay_flat.get("atlas_half_size", 256)),
            "no_view": True,
            "texture": {"generate_normal": True, "generate_roughness": True},
        }
        section = deep_merge(defaults, sub_wo_ent)
        section = deep_merge(section, overlay_flat)
        if overlay_tex:
            section["texture"] = deep_merge(section.get("texture") or {}, overlay_tex)
        hf = overlay_flat.get("color") or overlay_flat.get("frame_color")
        r1 = _maybe_hex_rgb(hex_to_rgb, hf) if isinstance(hf, str) else None
        if r1 is not None:
            section.setdefault("texture", {}).setdefault("frame_tex_color", r1)
        hg = overlay_flat.get("glass_color")
        r2 = _maybe_hex_rgb(hex_to_rgb, hg) if isinstance(hg, str) else None
        if r2 is not None:
            section.setdefault("texture", {}).setdefault("glass_tex_color", r2)

    else:
        section = deep_merge(dict(sub_wo_ent), overlay_flat) if overlay_flat else dict(sub_wo_ent)

    section["enabled"] = True
    section["out_dir"] = str(output_dir)
    section.setdefault("no_view", True)
    return section_key, section


def assemble_full_disabled_config(section_key: str, section_payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {k: {"enabled": False} for k in ALL_BATCH_SECTIONS}
    cfg[section_key] = section_payload
    return cfg
