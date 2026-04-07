import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from ultralytics import YOLO


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("draw_boxes_folder")


PROJECT_ROOT = Path(r"D:\4course_1sem\semestr_project\3d_Object_Maker")
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "door_window" / "yolov8n_panel" / "weights" / "best.pt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(source: Path, recursive: bool) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMG_EXTS:
            yield source
        return

    if recursive:
        for p in source.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        for p in source.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def parse_classes(s: str) -> Optional[List[int]]:
    """
    '--classes 0,2' -> [0,2]
    пусто/None -> None (все классы)
    """
    s = (s or "").strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser(description="Draw YOLO boxes on all images in a folder and save results.")
    parser.add_argument("--source", type=str, required=True, help="Папка или файл с изображениями.")
    parser.add_argument("--out", type=str, required=True, help="Папка, куда сохранять изображения с bbox.")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Путь к best.pt")
    parser.add_argument("--conf", type=float, default=0.4, help="Порог confidence.")
    parser.add_argument("--recursive", action="store_true", help="Искать изображения в подпапках.")
    parser.add_argument("--classes", type=str, default="", help="Какие классы рисовать, например '0,1'. Пусто = все.")
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out)
    weights = Path(args.weights)
    cls_filter = parse_classes(args.classes)

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading YOLO weights: %s", weights)
    model = YOLO(str(weights))

    images = list(iter_images(source, recursive=bool(args.recursive)))
    logger.info("Found %d images in %s", len(images), source)

    saved = 0
    skipped = 0

    for i, img_path in enumerate(images, start=1):
        results = model(str(img_path), conf=float(args.conf), verbose=False)
        if not results:
            skipped += 1
            continue

        r = results[0]

        if cls_filter is not None and r.boxes is not None and r.boxes.cls is not None and len(r.boxes.cls) > 0:
            # фильтруем боксы по классам
            keep = []
            for idx, c in enumerate(r.boxes.cls.cpu().numpy().astype(int).tolist()):
                if c in cls_filter:
                    keep.append(idx)
            if keep:
                r.boxes = r.boxes[keep]
            else:
                skipped += 1
                continue

        # рисуем и сохраняем
        try:
            # Ultralytics сам сохранит картинку, если передать filename
            out_path = out_dir / img_path.name
            r.plot(save=True, filename=str(out_path))
            saved += 1
        except Exception:
            skipped += 1

        if i % 100 == 0:
            logger.info("Progress %d/%d | saved=%d | skipped=%d", i, len(images), saved, skipped)

    logger.info("DONE | saved=%d | skipped=%d", saved, skipped)
    logger.info("Output folder: %s", out_dir)


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()

