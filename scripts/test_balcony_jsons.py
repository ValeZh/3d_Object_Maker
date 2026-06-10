"""Smoke-test all balcony JSON examples."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import trimesh

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.generator.procedural.procedural_batch_json_parser import load_batch_config
from src.generator.procedural.procedural_batch_runner import run_all_generators
from src.generator.procedural.procedural_balcony import USER_BALCONY, build_balcony_meshes

EXAMPLES = _REPO / "scripts" / "balcony_examples"
OUT = _REPO / "data" / "out_batch_test" / "json_smoke"


def _balcony_kwargs(bal: dict) -> dict:
    p = {**USER_BALCONY, **bal}
    ph = p.get("parapet_height")
    return dict(
        width_back=float(p["width_back"]),
        width_front=float(p["width_front"]),
        depth=float(p["depth"]),
        height=float(p["height"]),
        floor_thickness=float(p.get("floor_thickness", 0.14)),
        parapet_z_frac=float(p.get("parapet_z_frac", 0.42)),
        parapet_height=float(ph) if ph is not None else None,
        window_mode=str(p.get("window_mode", "none")),
        front_window_mode=p.get("front_window_mode"),
        window_depth=float(p.get("window_depth", 0.14)),
        tilt_left_deg=float(p.get("tilt_left_deg", 0.0)),
        tilt_right_deg=float(p.get("tilt_right_deg", 0.0)),
        wall_upper_z_frac=float(p.get("wall_upper_z_frac", 0.35)),
        has_roof=bool(p.get("has_roof", False)),
        roof_thickness=float(p.get("roof_thickness", 0.14)),
        roof_overhang=float(p.get("roof_overhang", 0.06)),
        inner_wall_windows=p.get("inner_wall_windows"),
        inner_wall_doors=p.get("inner_wall_doors"),
        simple_box=bool(p.get("simple_box", False)),
        window_left_wall=bool(p.get("window_left_wall", False)),
        window_right_wall=bool(p.get("window_right_wall", False)),
        side_parapet_split_frac=float(p.get("side_parapet_split_frac", 0.0)),
        open_left_above_parapet=bool(p.get("open_left_above_parapet", False)),
        open_right_above_parapet=bool(p.get("open_right_above_parapet", False)),
        wall_thickness=float(p.get("wall_thickness", 0.0)),
    )


def _check_roof(bal: dict) -> str:
    p = {**USER_BALCONY, **bal}
    wp, win = build_balcony_meshes(**_balcony_kwargs(bal))
    roof = [m for n, m in wp if n == "roof"]
    H = float(p["height"])
    rt = max(float(p.get("roof_thickness", 0.14)), 0.04)
    H_body = H - rt
    zmax = max(m.vertices[:, 2].max() for m in roof)
    zmin_side = min(m.vertices[:, 2].min() for m in roof[1:]) if len(roof) > 1 else zmax
    wall_max = max(
        (m.vertices[:, 2].max() for n, m in wp + win if n != "roof" and len(m.vertices)),
        default=0.0,
    )
    issues: list[str] = []
    if len(roof) != 5:
        issues.append(f"roof_parts={len(roof)}")
    if abs(zmax - H) > 0.02:
        issues.append(f"roof_top={zmax:.3f}!=H={H}")
    if abs(wall_max - H_body) > 0.03:
        issues.append(f"wall_top={wall_max:.3f}!=H_body={H_body:.3f}")
    if zmin_side > H_body + 0.03:
        issues.append(f"roof_base={zmin_side:.3f}>H_body")
    return "roof OK" if not issues else "; ".join(issues)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, str]] = []

    for cfg_path in sorted(EXAMPLES.glob("*.json")):
        try:
            cfg = load_batch_config(cfg_path)
        except Exception as exc:
            results.append((cfg_path.name, "SKIP", f"bad json: {exc}"))
            continue

        bal = cfg.get("balcony")
        if not isinstance(bal, dict) or not bal.get("enabled", True):
            results.append((cfg_path.name, "SKIP", "balcony disabled"))
            continue

        name = cfg_path.stem
        out_root = OUT / name
        out_root.mkdir(parents=True, exist_ok=True)
        test_cfg = json.loads(json.dumps(cfg))
        test_cfg["balcony"]["no_view"] = True
        test_cfg["balcony"]["out_dir"] = str(out_root / "balcony")
        for sec in ("entrance", "entrance_textured", "window", "wall", "wall_window"):
            if sec in test_cfg and isinstance(test_cfg[sec], dict):
                test_cfg[sec]["enabled"] = False

        try:
            run_all_generators(test_cfg, default_out_root=out_root)
            obj = out_root / "balcony" / "balcony.obj"
            if not obj.is_file():
                results.append((cfg_path.name, "FAIL", "no balcony.obj"))
                continue
            mesh = trimesh.load(str(obj), process=False)
            notes = [f"faces={len(mesh.faces)}"]
            if bal.get("has_roof"):
                notes.append(_check_roof(bal))
            else:
                wp, _ = build_balcony_meshes(**_balcony_kwargs(bal))
                if any(n == "roof" for n, _ in wp):
                    notes.append("unexpected roof without has_roof")
            results.append((cfg_path.name, "OK", ", ".join(notes)))
        except Exception as exc:
            results.append((cfg_path.name, "FAIL", str(exc)))
            traceback.print_exc()

    print("BALCONY JSON SMOKE TEST")
    print("-" * 72)
    for name, status, detail in results:
        print(f"{status:4}  {name:40}  {detail}")
    print("-" * 72)
    ok = sum(1 for r in results if r[1] == "OK")
    fail = sum(1 for r in results if r[1] == "FAIL")
    skip = sum(1 for r in results if r[1] == "SKIP")
    print(f"total={len(results)} ok={ok} fail={fail} skip={skip}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
