import argparse
import csv
import logging
import math
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("make_panel_dataset_yolo")


PROJECT_ROOT = Path(r"D:\4course_1sem\semestr_project\3d_Object_Maker")
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "door_window" / "yolov8n_panel" / "weights" / "best.pt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ===================== IO =====================
def iter_images(source: Path, recursive: bool) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMG_EXTS:
            yield source
        return

    it = source.rglob("*") if recursive else source.iterdir()
    for p in it:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


# ===================== GEOMETRY =====================
def make_square_bbox(x1, y1, x2, y2, img_w, img_h):
    w, h = x2 - x1, y2 - y1
    side = max(w, h)

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    nx1 = int(cx - side / 2)
    ny1 = int(cy - side / 2)
    nx2 = nx1 + side
    ny2 = ny1 + side

    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(img_w, nx2), min(img_h, ny2)

    return nx1, ny1, nx2, ny2


def make_square_bbox_with_margin(x1, y1, x2, y2, img_w, img_h, margin):
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img_w, x2 + margin)
    y2 = min(img_h, y2 + margin)
    return make_square_bbox(x1, y1, x2, y2, img_w, img_h)


# ===================== DESKEW (УЛУЧШЕННЫЙ) =====================
def estimate_skew_angle_from_image(img: Image.Image) -> float:
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        angle = math.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if len(angles) < 10:
        return 0.0

    return float(np.median(angles))


def rotate_image_keep_size(img: Image.Image, angle_deg: float) -> Image.Image:
    if abs(angle_deg) < 0.1:
        return img

    w, h = img.size
    rotated = img.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
    rw, rh = rotated.size

    left = (rw - w) // 2
    top = (rh - h) // 2
    return rotated.crop((left, top, left + w, top + h))


# ===================== RECTIFY WINDOW =====================
def rectify_crop_by_window_lines(crop: Image.Image, max_abs_deg=15.0):
    img_np = np.array(crop)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 70, 180)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60,
                            minLineLength=40, maxLineGap=5)

    if lines is None:
        return crop, 0.0

    angles = []

    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx, dy = x2 - x1, y2 - y1

        if dx == 0 and dy == 0:
            continue

        angle = math.degrees(math.atan2(dy, dx))

        # нормализация
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180

        # строгий фильтр
        if abs(angle) <= 10:
            angles.append(angle)
        elif abs(abs(angle) - 90) <= 10:
            angles.append(angle - 90)

    if len(angles) < 5:
        return crop, 0.0

    skew = float(np.median(angles))

    if abs(skew) > max_abs_deg:
        return crop, 0.0

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -skew, 1.0)

    rotated = cv2.warpAffine(
        img_np, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101
    )

    return Image.fromarray(rotated), skew


# ===================== MAIN =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="panels")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--margin", type=int, default=5)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--per-box", action="store_true")
    # 0 = window (CMP), 1 = door, -1 = все классы
    parser.add_argument("--only-class", type=int, default=0)
    parser.add_argument("--square", action="store_true")
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--rectify-window", action="store_true")

    args = parser.parse_args()

    model = YOLO(args.weights)
    images = list(iter_images(Path(args.source), True))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for img_path in images:
        img = Image.open(img_path).convert("RGB")

        # ===== DESKEW (НОВЫЙ) =====
        if args.deskew:
            angle = estimate_skew_angle_from_image(img)
            if abs(angle) > 0.2:
                img = rotate_image_keep_size(img, -angle)

        results = model(np.array(img), conf=args.conf, verbose=False)
        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        for i, (bb, c) in enumerate(zip(xyxy, cls)):
            if args.per_box and args.only_class != -1 and c != args.only_class:
                continue

            x1, y1, x2, y2 = map(int, bb)

            x1, y1, x2, y2 = make_square_bbox_with_margin(
                x1, y1, x2, y2,
                img.size[0], img.size[1],
                args.margin
            )

            crop = img.crop((x1, y1, x2, y2))

            # ===== RECTIFY КАЖДОГО ОБЪЕКТА =====
            # выравниваем кроп по линиям именно этого окна/объекта
            if args.rectify_window:
                # если пользователь указал конкретный класс --only-class,
                # выравниваем только его; при -1 — все объекты
                if args.only_class == -1 or c == args.only_class:
                    crop, _ = rectify_crop_by_window_lines(crop)

            crop = crop.resize((args.size, args.size), Image.BICUBIC)

            out_path = out_dir / f"panel_{saved:06d}.jpg"
            crop.save(out_path, quality=95)
            saved += 1

    print(f"Saved: {saved}")


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()