from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

# -----------------------------------------
# НАСТРОЙКИ ПОД ТВОЙ ПРОЕКТ
# -----------------------------------------

# Путь к папке с CMP масками/картинками
BASE_DIR = Path(r"D:\4course_1sem\semestr_project\3d_Object_Maker\data\CMP_facade_DB_base\base")

# Куда сохранить YOLO-датасет
OUT_ROOT = Path(r"D:\4course_1sem\semestr_project\3d_Object_Maker\data\CMP_facade_DB_base\yolo")

# Доля train (остальное -> val)
TRAIN_SPLIT = 0.8

# Маппинг: значение пикселя в маске CMP (1..12) -> id класса YOLO (0..)
# По label_names.txt:
# 1 background, 2 facade, 3 window, 4 door, 5 cornice,
# 6 sill, 7 balcony, 8 blind, 9 deco, 10 molding, 11 pillar, 12 shop
VALUE_TO_CLASS: Dict[int, int] = {
    3: 0,   # window
    4: 1,   # door
    7: 2,   # balcony
    6: 3,   # sill
    5: 4,   # cornice
    8: 5,   # blind
    9: 6,   # deco
    10: 7,  # molding
    11: 8,  # pillar
    12: 9,  # shop
    # 1,2 игнорируем
}

MIN_COMPONENT_AREA = 20  # минимум пикселей в компоненте, чтобы не считать шум


def find_components(mask_bin: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Связные компоненты в бинарной маске -> список bbox (x1, y1, x2, y2)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin.astype(np.uint8), connectivity=8
    )
    bboxes: List[Tuple[int, int, int, int]] = []
    for label in range(1, num_labels):  # 0 = фон
        x, y, w, h, area = stats[label]
        if area < MIN_COMPONENT_AREA:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        bboxes.append((x1, y1, x2, y2))
    return bboxes


def convert_single(img_path: Path, mask_path: Path, out_img_path: Path, out_lbl_path: Path) -> None:
    """Конвертирует одну пару (jpg + png маска) в YOLO-лейбл."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    mask = Image.open(mask_path)  # палитровый PNG, значения 0..12
    mask_idx = np.array(mask)

    lines: List[str] = []

    for val, cls_id in VALUE_TO_CLASS.items():
        cls_mask = (mask_idx == val)
        if not cls_mask.any():
            continue

        bboxes = find_components(cls_mask)
        for x1, y1, x2, y2 in bboxes:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            cx_n = cx / w
            cy_n = cy / h
            bw_n = bw / w
            bh_n = bh / h

            lines.append(f"{cls_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_lbl_path.parent.mkdir(parents=True, exist_ok=True)

    img.save(out_img_path)
    with open(out_lbl_path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")


def main():
    assert BASE_DIR.exists(), f"base dir not found: {BASE_DIR}"

    img_paths = sorted(
        [p for p in BASE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}]
    )

    print(f"Всего JPEG фасадов: {len(img_paths)}")

    n_train = int(len(img_paths) * TRAIN_SPLIT)
    train_imgs = img_paths[:n_train]
    val_imgs = img_paths[n_train:]

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    def process(subset_imgs: List[Path], subset: str):
        for img_path in subset_imgs:
            mask_path = img_path.with_suffix(".png")
            if not mask_path.exists():
                print(f"[WARN] нет маски для {img_path.name}, пропускаю")
                continue

            out_img = OUT_ROOT / "images" / subset / img_path.name
            out_lbl = OUT_ROOT / "labels" / subset / (img_path.stem + ".txt")

            convert_single(img_path, mask_path, out_img, out_lbl)

    process(train_imgs, "train")
    process(val_imgs, "val")
    print("Готово. YOLO-датасет в:", OUT_ROOT)


if __name__ == "__main__":
    main()