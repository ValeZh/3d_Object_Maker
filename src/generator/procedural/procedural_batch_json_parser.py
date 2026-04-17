from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.generator.procedural.procedural_batch_runner import run_all_generators

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "scripts" / "balcony_examples" / "batch_generators_config.json"
_DEFAULT_OUT_ROOT = _REPO_ROOT / "data" / "out_batch"


def load_batch_config(config_path: Path) -> Dict[str, Any]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON config must be an object with generator sections.")
    return data


def parse_and_run(config_path: Path, out_root: Path) -> dict[str, Path]:
    config = load_batch_config(config_path)
    return run_all_generators(config, default_out_root=out_root)


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="JSON parser for batch procedural generator calls.")
    ap.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to JSON config with sections for procedural generators.",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default=str(_DEFAULT_OUT_ROOT),
        help="Default output root for generators without explicit out_dir.",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = _build_cli().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    result = parse_and_run(config_path, out_root)
    if not result:
        print("[INFO] No enabled generator sections found in config.")
        return

    print("[OK] Batch export completed:")
    for name, path in result.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
