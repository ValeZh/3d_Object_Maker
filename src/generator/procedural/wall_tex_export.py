"""
Экспорт стены с окном с обязательной текстурой стены.

Те же аргументы, что у `procedural_wall_window export`, но флаг `--wall-tex` обязателен
(иначе нет смысла в этом входе — используйте `procedural_wall_window` без текстуры стены).

Пример (одна строка):
  python -m src.generator.procedural.wall_tex_export export -o data/wall_win --wall-length 5 --wall-thickness 0.35 --wall-height 3.2 --window-center-x 0 --window-sill-z 0.95 --wall-tex data/textures/brick.jpg --width 1.2 --height 1.5 --depth 0.12
"""
from __future__ import annotations

import sys


def _has_wall_tex_flag(argv: list[str]) -> bool:
    for a in argv:
        if a == "--wall-tex":
            return True
        if a.startswith("--wall-tex="):
            return True
    return False


def _is_help(argv: list[str]) -> bool:
    return "-h" in argv or "--help" in argv


def main() -> None:
    argv = sys.argv[1:]
    if not _is_help(argv) and not _has_wall_tex_flag(argv):
        print(
            "wall_tex_export: нужен флаг --wall-tex ПУТЬ (текстура фасада).\n"
            "Справка: python -m src.generator.procedural.wall_tex_export export --help",
            file=sys.stderr,
        )
        sys.exit(2)
    from src.generator.procedural.procedural_wall_window import main as wall_window_main

    wall_window_main(argv)


if __name__ == "__main__":
    main()
