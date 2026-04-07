import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

# --------- НАСТРОЙКИ ПОД ТВОЙ ПРОЕКТ ---------

# Папка с исходными картинками домов
SOURCE_DIR = Path(r"D:\4course_1sem\semestr_project\3d_Object_Maker\data\zip2_h")

# Папка, куда сохранить выровненные версии
OUT_DIR = Path(r"D:\4course_1sem\semestr_project\3d_Object_Maker\data\zip2_h_rectified")

# Рекурсивно обходить подпапки?
RECURSIVE = True

# Минимальный |угол|, при котором есть смысл что‑то крутить (в градусах)
MIN_DEG = 0.5

# Максимальный |угол|, чтобы не ломать перспективу (в градусах)
MAX_DEG = 20.0

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(src: Path, recursive: bool) -> Iterable[Path]:
    if src.is_file():
        if src.suffix.lower() in IMG_EXTS:
            yield src
        return

    if recursive:
        for p in src.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        for p in src.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def estimate_skew_deg(img_rgb: np.ndarray) -> float:
    """
    Оценивает наклон фасада по линиям (Canny + HoughLinesP).
    Возвращает угол (в градусах), который нужно КОМПЕНСИРОВАТЬ
    поворотом на -угол.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0.0

    angles = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue

        angle = np.degrees(np.arctan2(dy, dx))  # 0 = горизонталь

        # приводим к [-90, 90]
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180

        # нас интересуют линии, почти горизонтальные или вертикальные
        if abs(angle) <= 30 or abs(angle - 90) <= 30:
            angles.append(angle)

    if len(angles) == 0:
        return 0.0

    med = float(np.median(np.array(angles, dtype=np.float32)))
    return med


def rotate_keep_size(img_rgb: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-6:
        return img_rgb
    h, w = img_rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    rotated = cv2.warpAffine(
        img_rgb,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return rotated


def main():
    assert SOURCE_DIR.exists(), f"SOURCE_DIR not found: {SOURCE_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(SOURCE_DIR, RECURSIVE))
    print(f"Найдено картинок: {len(images)}")

    fixed = 0
    copied = 0
    failed = 0

    for i, p in enumerate(images, start=1):
        try:
            img = Image.open(p).convert("RGB")
            arr = np.array(img)
        except Exception as e:
            print(f"[FAIL] {p.name}: {e}")
            failed += 1
            continue

        skew = estimate_skew_deg(arr)

        if abs(skew) < MIN_DEG or abs(skew) > MAX_DEG:
            # сохраняем как есть
            out_path = OUT_DIR / p.name
            img.save(out_path, quality=95)
            copied += 1
        else:
            # компенсируем наклон
            rotated = rotate_keep_size(arr, -skew)
            out_path = OUT_DIR / p.name
            Image.fromarray(rotated).save(out_path, quality=95)
            fixed += 1

        if i % 100 == 0:
            print(f"Progress {i}/{len(images)} | fixed={fixed} copied={copied} failed={failed}")

    print("ГОТОВО.")
    print(f"Выровняно: {fixed}, без изменений: {copied}, ошибок: {failed}")
    print("Результат в:", OUT_DIR)


if __name__ == "__main__":
    main()

