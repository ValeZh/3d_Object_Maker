"""
Процедурные текстуры окна: рама (ПВХ/дерево) и стекло — отдельные файлы + атлас для UV.

Файлы в data/textures/:
  window_frame.png   — рама и импосты
  window_glass.png   — стекло (мороз/оттенок)
  window_atlas.png   — слева рама, справа стекло (для одного map_Kd в OBJ)

Свои картинки: make_atlas_from_sources(frame_path=..., glass_path=...) или CLI ниже.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.generator.procedural.texturing.color_tint import apply_texture_color_tint, parse_texture_color_tint


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_textures_dir() -> Path:
    return _repo_root() / "data" / "textures"


def resolve_texture_path(path: str | Path | None) -> Path | None:
    """Абсолютный путь к существующему файлу или None."""
    if path is None:
        return None
    r = Path(path).expanduser().resolve()
    return r if r.is_file() else None


def make_window_frame_texture(size: int = 512) -> Image.Image:
    """Белая рама: почти чистый белый, лёгкий шум и едва заметные вертикальные швы."""
    rng = np.random.default_rng(42)
    s = max(size, 64)
    base = np.ones((s, s, 3), dtype=np.float32) * 255.0
    noise = rng.normal(0, 1.8, (s, s, 3)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    for x in range(0, s, max(s // 24, 8)):
        img[:, x : x + 1, :] = np.clip(img[:, x : x + 1, :].astype(np.int16) - 5, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def make_window_glass_texture(size: int = 512) -> Image.Image:
    """Глянцево-серое стекло: нейтральный серый, плавный блик, мало шума."""
    rng = np.random.default_rng(17)
    s = max(size, 64)
    yy = np.linspace(0.0, 1.0, s, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, s, dtype=np.float32)[None, :]
    base = 142.0 + 28.0 * yy + 20.0 * (1.0 - xx)
    rgb = np.stack([base, base, base], axis=-1)
    yg, xg = np.ogrid[:s, :s]
    cx, cy = s * 0.58, s * 0.36
    d2 = (xg - cx) ** 2 + (yg - cy) ** 2
    spec = np.exp(-d2 / (2 * (s * 0.2) ** 2)) * 62.0
    rgb[..., 0] += spec
    rgb[..., 1] += spec
    rgb[..., 2] += spec
    rgb += rng.normal(0, 2.2, (s, s, 3)).astype(np.float32)
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def make_window_atlas(half: int = 512) -> Image.Image:
    """Ширина 2*half: левая половина — рама, правая — стекло."""
    fr = make_window_frame_texture(half)
    gl = make_window_glass_texture(half)
    atlas = Image.new("RGB", (half * 2, half))
    atlas.paste(fr, (0, 0))
    atlas.paste(gl, (half, 0))
    return atlas


def _open_image_rgb(path: Path) -> Image.Image:
    """Загрузка PNG/JPG и т.п. в RGB; RGBA — на белый фон."""
    im = Image.open(path)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA")
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im


def _boost_dark_glass_visible(img: Image.Image, luminance_threshold: float = 72.0) -> Image.Image:
    """
    Тёмные карты вроде glass_black.png почти не видны в превью и на светлом фоне сцены.
    Слегка поднимаем яркость и контраст только если средняя яркость ниже порога.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    lum = float(
        np.dot(a.reshape(-1, 3).mean(axis=0), np.array([0.299, 0.587, 0.114], dtype=np.float64))
    )
    if lum >= luminance_threshold:
        return img
    gain = 1.0 + (luminance_threshold - lum) / 100.0
    bias = (luminance_threshold - lum) * 0.45
    b = np.clip(a * gain + bias, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(b, mode="RGB")


def make_atlas_from_sources(
    *,
    frame_path: Path | str | None = None,
    glass_path: Path | str | None = None,
    half_size: int = 512,
    frame_color: Any = None,
    glass_color: Any = None,
) -> Image.Image:
    """
    Атлас для UV: слева [0,0.5] — рама, справа [0.5,1] — стекло.
    Для каждой стороны: если путь задан и файл есть — картинка (масштаб под квадрат half×half),
    иначе процедурная текстура.

    ``frame_color`` / ``glass_color`` — опциональный тинт RGB: ``[r,g,b]`` (0–255) или 0–1,
    см. ``parse_texture_color_tint``.
    """
    half = max(int(half_size), 64)
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    fp = Path(frame_path).expanduser().resolve() if frame_path else None
    gp = Path(glass_path).expanduser().resolve() if glass_path else None

    fc = parse_texture_color_tint(frame_color)
    gc = parse_texture_color_tint(glass_color)

    if fp is not None and fp.is_file():
        fr = apply_texture_color_tint(_open_image_rgb(fp).resize((half, half), resample), fc)
    else:
        fr = apply_texture_color_tint(make_window_frame_texture(half), fc)

    if gp is not None and gp.is_file():
        gl = _boost_dark_glass_visible(_open_image_rgb(gp).resize((half, half), resample))
        gl = apply_texture_color_tint(gl, gc)
    else:
        gl = apply_texture_color_tint(make_window_glass_texture(half), gc)

    atlas = Image.new("RGB", (half * 2, half))
    atlas.paste(fr, (0, 0))
    atlas.paste(gl, (half, 0))
    return atlas


def make_normal_atlas_from_sources(
    *,
    frame_path: Path | str | None = None,
    glass_path: Path | str | None = None,
    half_size: int = 512,
) -> Image.Image:
    """
    Атлас карт нормалей под те же UV, что у ``make_atlas_from_sources`` (слева рама, справа стекло).
    Если файл не найден: рама — процедурная нормалка; стекло — плоская однотонная (без рельефа).
    """
    from src.generator.procedural.procedural_texture_maps.normal_map import (
        make_fine_noise_normal_map,
        make_neutral_flat_normal_map,
    )

    half = max(int(half_size), 64)
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    fp = Path(frame_path).expanduser().resolve() if frame_path else None
    gp = Path(glass_path).expanduser().resolve() if glass_path else None

    if fp is not None and fp.is_file():
        fr = _open_image_rgb(fp).resize((half, half), resample)
    else:
        fr = make_fine_noise_normal_map(half, strength=10.0, seed=41)

    if gp is not None and gp.is_file():
        gl = _open_image_rgb(gp).resize((half, half), resample)
    else:
        gl = make_neutral_flat_normal_map(half)

    atlas = Image.new("RGB", (half * 2, half))
    atlas.paste(fr, (0, 0))
    atlas.paste(gl, (half, 0))
    return atlas


def make_window_roughness_atlas(
    half_size: int,
    *,
    frame_roughness: float = 0.86,
    glass_roughness: float = 0.08,
    frame_spatial_variation: float = 0.05,
    seed: int = 11,
) -> Image.Image:
    """
    Атлас шероховатости под UV окна (слева рама | справа стекло), для PBR (Open3D ``roughness_img``).

    Светлее — шершавее, темнее — глаже (высокий / низкий roughness в [0, 1]).
    """
    half = max(int(half_size), 64)
    w = half * 2
    rng = np.random.default_rng(seed)
    fr = np.clip(
        float(frame_roughness)
        + rng.normal(0.0, float(frame_spatial_variation), (half, half)).astype(np.float64),
        0.0,
        1.0,
    )
    gl = np.full((half, half), float(glass_roughness), dtype=np.float64)
    rr = np.zeros((half, w), dtype=np.float64)
    rr[:, 0:half] = fr
    rr[:, half:w] = gl
    g8 = (np.clip(rr, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = np.stack([g8, g8, g8], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def ensure_window_textures(
    out_dir: Path | None = None, *, half_size: int = 512, force: bool = False
) -> dict[str, Path]:
    """Создаёт PNG в out_dir, если их ещё нет (или force=True — перезаписать). Возвращает пути."""
    out_dir = out_dir or default_textures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    p_frame = out_dir / "window_frame.png"
    p_glass = out_dir / "window_glass.png"
    p_atlas = out_dir / "window_atlas.png"
    if force or not p_frame.is_file():
        make_window_frame_texture(half_size).save(p_frame)
    if force or not p_glass.is_file():
        make_window_glass_texture(half_size).save(p_glass)
    if force or not p_atlas.is_file():
        make_window_atlas(half_size).save(p_atlas)
    paths["frame"] = p_frame
    paths["glass"] = p_glass
    paths["atlas"] = p_atlas
    return paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="PNG текстуры окна: процедурные или свои изображения → атлас для UV."
    )
    ap.add_argument("--force", action="store_true", help="Перезаписать процедурные файлы в data/textures/")
    ap.add_argument("--size", type=int, default=512, metavar="N", help="Сторона квадрата половины атласа")
    ap.add_argument("--frame", type=str, default=None, metavar="PATH", help="Изображение для рамы (jpg/png)")
    ap.add_argument("--glass", type=str, default=None, metavar="PATH", help="Изображение для стекла")
    ap.add_argument(
        "-o",
        "--atlas-out",
        type=str,
        default=None,
        metavar="PATH",
        help="Сохранить только атлас в этот файл (иначе — стандартный ensure в data/textures)",
    )
    args = ap.parse_args()
    half = max(args.size, 64)
    if args.atlas_out:
        out = Path(args.atlas_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        atlas = make_atlas_from_sources(frame_path=args.frame, glass_path=args.glass, half_size=half)
        atlas.save(out)
        print("Atlas written:", out)
    else:
        p = ensure_window_textures(half_size=half, force=args.force)
        print("Written:", p)
