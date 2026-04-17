"""
Отбор лучших сгенерированных «окон» с помощью YOLO (метрика — насколько детектор
узнаёт в картинке окно, как на реальных кропах).

Дополнительно (опционально):
  - целевое соотношение сторон бокса окна (w/h) и доля площади кадра — мягкий гауссовский скор;
  - схожесть с эталонным изображением (--reference): MSE в RGB → [0,1] после нормализации.

Итог: score_combined = w_y*yolo + w_s*shape + w_m*sim (веса по умолчанию сбалансированы;
если эталона нет, w_m распределяется на yolo и shape).

1) Сгенерировать кандидатов из DCGAN и выбрать лучшие:
   python src/generator/texture/rank_gan_windows_yolo.py ^
     --gan-weights runs/window_gan_v2/window_generator.pt ^
     --yolo-weights runs/door_window/yolov8n_panel/weights/best.pt ^
     --candidates 64 --top-k 8 --out runs/window_gan_v2/yolo_picked

2) С эталоном и целевой формой (пример):
   ... --reference runs/ref_window.png --target-aspect 0.85 --target-area-frac 0.45

Эвристика: YOLO учился на реальных кропах; размытый GAN может получить низкий скор —
это ожидаемо, а не баг скрипта.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, utils
from ultralytics import YOLO

from train_window_gan import load_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _resolve_model_file(arg: str, root: Path, what: str) -> Path:
    """Путь от корня проекта или абсолютный; срезает случайный ` с конца строки из PowerShell."""
    s = arg.strip().strip("`").strip('"').strip("'")
    p = Path(s)
    if p.is_file():
        return p.resolve()
    cand = (root / s).resolve()
    if cand.is_file():
        return cand
    norm = root / Path(s.replace("\\", "/"))
    if norm.is_file():
        return norm.resolve()
    if "yolo" in what.lower() and cand.suffix.lower() == ".pt" and cand.name.lower() == "last.pt":
        alt = cand.parent / "best.pt"
        if alt.is_file():
            logger.warning("last.pt не найден, берём %s", alt)
            return alt.resolve()
    raise FileNotFoundError(
        f"Не найден {what}: {arg!r}\n"
        f"  как есть: {p}\n"
        f"  от корня проекта: {cand}\n"
        f"Корень проекта: {root}"
    )


def _list_images(folder: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _largest_window_box(result, window_cls: int) -> Optional[Tuple[float, float, float, float, float]]:
    """
    Самый крупный по площади бокс класса window: (w, h, aspect_wh, area_frac, conf).
    """
    boxes = result.boxes
    im_h, im_w = result.orig_shape
    area_img = float(im_h * im_w)
    if boxes is None or len(boxes) == 0:
        return None
    best_area = -1.0
    out: Optional[Tuple[float, float, float, float, float]] = None
    for i in range(len(boxes)):
        if int(boxes.cls[i].item()) != window_cls:
            continue
        xyxy = boxes.xyxy[i].cpu().numpy().ravel()
        w = float(xyxy[2] - xyxy[0])
        h = float(xyxy[3] - xyxy[1])
        area = w * h
        if area > best_area:
            best_area = area
            conf = float(boxes.conf[i].item())
            aspect = w / max(h, 1e-6)
            frac = area / max(area_img, 1.0)
            out = (w, h, aspect, frac, conf)
    return out


def _yolo_metric_max(result, window_cls: int, metric: str) -> float:
    """Как раньше: лучший скор среди боксов window по выбранной метрике."""
    boxes = result.boxes
    im_h, im_w = result.orig_shape
    area_img = float(im_h * im_w)
    if boxes is None or len(boxes) == 0:
        return 0.0
    best = 0.0
    for i in range(len(boxes)):
        if int(boxes.cls[i].item()) != window_cls:
            continue
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy().ravel()
        w = float(xyxy[2] - xyxy[0])
        h = float(xyxy[3] - xyxy[1])
        frac = (w * h) / max(area_img, 1.0)
        if metric == "max_conf":
            s = conf
        elif metric == "conf_sqrt_area":
            s = conf * math.sqrt(max(frac, 1e-8))
        else:
            raise ValueError(metric)
        best = max(best, s)
    return best


def _shape_score(
    aspect: float,
    area_frac: float,
    target_aspect: Optional[float],
    target_area_frac: Optional[float],
    aspect_sigma: float,
    area_sigma: float,
) -> float:
    """1.0 если цели не заданы; иначе произведение гауссов отклонений."""
    if target_aspect is None and target_area_frac is None:
        return 1.0
    sa = 1.0
    sf = 1.0
    if target_aspect is not None:
        sa = math.exp(-((aspect - target_aspect) / max(aspect_sigma, 1e-6)) ** 2)
    if target_area_frac is not None:
        sf = math.exp(-((area_frac - target_area_frac) / max(area_sigma, 1e-6)) ** 2)
    return sa * sf


def _load_reference_tensor(ref_path: Path, size: int, device: str) -> torch.Tensor:
    """[3,H,W] float 0..1, квадрат size."""
    img = Image.open(ref_path).convert("RGB")
    t = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )(img)
    return t.to(device)


def _similarity_mse(ref: torch.Tensor, path: Path, size: int, device: str) -> float:
    """1 / (1 + k*mse), выше = ближе к эталону."""
    img = Image.open(path).convert("RGB")
    t = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )(img).to(device)
    mse = F.mse_loss(t, ref).item()
    return 1.0 / (1.0 + 25.0 * mse)


def _score_one_image(
    result,
    window_cls: int,
    metric: str,
    target_aspect: Optional[float],
    target_area_frac: Optional[float],
    aspect_sigma: float,
    area_sigma: float,
    ref_tensor: Optional[torch.Tensor],
    sim_size: int,
    device: str,
    path: Path,
    weight_yolo: float,
    weight_shape: float,
    weight_sim: float,
) -> Tuple[float, Dict[str, Any]]:
    boxes = result.boxes
    n_all = 0 if boxes is None else len(boxes)
    lb = _largest_window_box(result, window_cls)
    n_win = 0
    if boxes is not None:
        n_win = sum(1 for i in range(len(boxes)) if int(boxes.cls[i].item()) == window_cls)

    yolo_s = _yolo_metric_max(result, window_cls, metric)
    yolo_n = min(1.0, yolo_s) if metric == "max_conf" else min(1.0, yolo_s / 2.0)

    sim_s = _similarity_mse(ref_tensor, path, sim_size, device) if ref_tensor is not None else 0.0

    if lb is None:
        combined = weight_yolo * yolo_n + weight_sim * sim_s
        meta = {
            "n_det_window": n_win,
            "n_det_all": n_all,
            "aspect_wh": None,
            "area_frac": None,
            "score_yolo": round(yolo_n, 6),
            "score_shape": 0.0,
            "score_sim": round(sim_s, 6) if ref_tensor is not None else None,
            "score_combined": round(combined, 6),
        }
        return combined, meta

    _w, _h, aspect, area_frac, _cf = lb
    shape_s = _shape_score(aspect, area_frac, target_aspect, target_area_frac, aspect_sigma, area_sigma)

    combined = (
        weight_yolo * yolo_n + weight_shape * shape_s + weight_sim * sim_s
    )

    meta = {
        "n_det_window": n_win,
        "n_det_all": n_all,
        "aspect_wh": round(aspect, 4),
        "area_frac": round(area_frac, 4),
        "score_yolo": round(yolo_n, 6),
        "score_shape": round(shape_s, 6),
        "score_sim": round(sim_s, 6) if ref_tensor is not None else None,
        "score_combined": round(combined, 6),
    }
    return combined, meta


def generate_from_gan(
    gan_weights: Path,
    out_dir: Path,
    n: int,
    device: str,
    seed: int,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    g, z_dim, _ = load_generator(gan_weights, device)
    paths: List[Path] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    with torch.no_grad():
        for i in range(n):
            z = torch.randn(1, z_dim, 1, 1, device=device, generator=gen)
            img = (g(z).cpu()[0] * 0.5 + 0.5).clamp(0, 1)
            p = out_dir / f"candidate_{i:04d}.png"
            utils.save_image(img, p)
            paths.append(p)
    logger.info("Сгенерировано %d кандидатов в %s", n, out_dir)
    return paths


def main():
    p = argparse.ArgumentParser(description="Ранг сгенерированных окон по YOLO-детекции.")
    p.add_argument("--images", type=str, default="", help="Папка с PNG/JPG (если нет --gan-weights).")
    p.add_argument(
        "--gan-weights",
        type=str,
        default="",
        help="window_generator.pt — сгенерировать кандидатов в --work-dir/candidates.",
    )
    p.add_argument("--candidates", type=int, default=32, help="Сколько сгенерировать (--gan-weights).")
    p.add_argument("--work-dir", type=str, default="", help="Рабочая папка; по умолчанию = --out.")
    p.add_argument("--out", type=str, required=True, help="Отчёт + best_k копии.")
    p.add_argument("--yolo-weights", type=str, required=True, help="best.pt детектора.")
    p.add_argument("--window-class", type=int, default=0, help="Индекс класса window в модели.")
    p.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Мин. confidence при инференсе (ниже — больше слабых срабатываний).",
    )
    p.add_argument(
        "--metric",
        choices=("max_conf", "conf_sqrt_area"),
        default="max_conf",
        help="max_conf или conf*sqrt(доля площади бокса).",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--seed", type=int, default=46)
    p.add_argument(
        "--reference",
        type=str,
        default="",
        help="Эталонное фото окна (PNG/JPG) для метрики схожести по MSE после ресайза.",
    )
    p.add_argument(
        "--sim-size",
        type=int,
        default=128,
        help="Размер стороны для сравнения с эталоном (квадрат).",
    )
    p.add_argument(
        "--target-aspect",
        type=float,
        default=None,
        help="Целевое w/h самого крупного бокса окна (например 0.7 для «выше, чем шире»).",
    )
    p.add_argument(
        "--target-area-frac",
        type=float,
        default=None,
        help="Целевая доля площади кадра, занятая боксом (0..1).",
    )
    p.add_argument("--aspect-sigma", type=float, default=0.28, help="Допуск по aspect (гаусс).")
    p.add_argument("--area-sigma", type=float, default=0.14, help="Допуск по доле площади (гаусс).")
    p.add_argument("--weight-yolo", type=float, default=0.4, help="Вес нормализованного YOLO-скора.")
    p.add_argument("--weight-shape", type=float, default=0.35, help="Вес совпадения формы/размера бокса.")
    p.add_argument("--weight-sim", type=float, default=0.25, help="Вес схожести с эталоном.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    work = Path(args.work_dir) if args.work_dir else out_root
    if not work.is_absolute():
        work = PROJECT_ROOT / work
    work.mkdir(parents=True, exist_ok=True)

    yolo_w = _resolve_model_file(args.yolo_weights, PROJECT_ROOT, "yolo-weights")

    if args.gan_weights:
        gpath = _resolve_model_file(args.gan_weights, PROJECT_ROOT, "gan-weights")
        cand_dir = work / "candidates"
        image_paths = generate_from_gan(gpath, cand_dir, args.candidates, device, args.seed)
    else:
        if not args.images:
            raise ValueError("Нужно --gan-weights или --images")
        img_root = Path(args.images)
        if not img_root.is_absolute():
            img_root = PROJECT_ROOT / img_root
        if not img_root.is_dir():
            raise FileNotFoundError(img_root)
        image_paths = _list_images(img_root, args.recursive)
        if not image_paths:
            raise RuntimeError(f"Нет изображений в {img_root}")

    logger.info("Загрузка YOLO: %s", yolo_w)
    model = YOLO(str(yolo_w))

    ref_t: Optional[torch.Tensor] = None
    ref_path_resolved: Optional[str] = None
    if args.reference.strip():
        ref_path = Path(args.reference.strip().strip("`").strip('"'))
        if not ref_path.is_file():
            ref_path = PROJECT_ROOT / ref_path
        if not ref_path.is_file():
            raise FileNotFoundError(f"--reference не найден: {args.reference}")
        ref_path_resolved = str(ref_path.resolve())
        ref_t = _load_reference_tensor(ref_path, int(args.sim_size), device)
        logger.info("Эталон для схожести: %s (%d×%d)", ref_path, args.sim_size, args.sim_size)

    ta = args.target_aspect
    tf = args.target_area_frac
    if ta is not None or tf is not None:
        logger.info("Цели по боксу: aspect=%s area_frac=%s", ta, tf)

    wy, ws, wm = float(args.weight_yolo), float(args.weight_shape), float(args.weight_sim)
    if ref_t is None:
        wm = 0.0
    s = wy + ws + wm
    if s < 1e-9:
        wy, ws, wm = (0.5, 0.5, 0.0) if ref_t is None else (0.4, 0.3, 0.3)
        s = wy + ws + wm
    wy, ws, wm = wy / s, ws / s, wm / s
    logger.info("Веса (норм.): yolo=%.3f shape=%.3f sim=%.3f", wy, ws, wm)

    rows: List[Dict[str, Any]] = []
    for path in image_paths:
        results = model(str(path), conf=float(args.conf), verbose=False)
        if not results:
            score, meta = 0.0, {
                "n_det_window": 0,
                "n_det_all": 0,
                "aspect_wh": None,
                "area_frac": None,
                "score_yolo": 0.0,
                "score_shape": 0.0,
                "score_sim": None,
                "score_combined": 0.0,
            }
        else:
            score, meta = _score_one_image(
                results[0],
                args.window_class,
                args.metric,
                ta,
                tf,
                float(args.aspect_sigma),
                float(args.area_sigma),
                ref_t,
                int(args.sim_size),
                device,
                path,
                wy,
                ws,
                wm,
            )
        rows.append({"path": str(path.resolve()), "score": score, **meta})

    rows.sort(key=lambda r: r["score_combined"], reverse=True)

    best_dir = out_root / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "yolo_weights": str(yolo_w),
        "window_class": args.window_class,
        "metric": args.metric,
        "conf_threshold": args.conf,
        "reference": ref_path_resolved,
        "target_aspect": ta,
        "target_area_frac": tf,
        "weights": {"yolo": wy, "shape": ws, "sim": wm},
        "rankings": rows,
    }
    (out_root / "yolo_rankings.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out_root / "yolo_rankings.csv"
    csv_fields = [
        "rank",
        "score_combined",
        "score_yolo",
        "score_shape",
        "score_sim",
        "aspect_wh",
        "area_frac",
        "n_det_window",
        "n_det_all",
        "path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        wr.writeheader()
        for i, r in enumerate(rows, start=1):
            row: Dict[str, Any] = {"rank": i}
            row["score_combined"] = f"{r.get('score_combined', r['score']):.6f}"
            row["score_yolo"] = f"{r.get('score_yolo', 0):.6f}"
            row["score_shape"] = f"{r.get('score_shape', 0):.6f}"
            ss = r.get("score_sim")
            row["score_sim"] = "" if ss is None else f"{ss:.6f}"
            row["aspect_wh"] = "" if r.get("aspect_wh") is None else r["aspect_wh"]
            row["area_frac"] = "" if r.get("area_frac") is None else r["area_frac"]
            row["n_det_window"] = r["n_det_window"]
            row["n_det_all"] = r["n_det_all"]
            row["path"] = r["path"]
            wr.writerow(row)

    k = max(1, min(args.top_k, len(rows)))
    for i in range(k):
        src = Path(rows[i]["path"])
        sc = rows[i].get("score_combined", rows[i]["score"])
        dst = best_dir / f"best_{i+1:02d}_combined_{sc:.4f}{src.suffix}"
        shutil.copy2(src, dst)

    logger.info("Топ-%d сохранены в %s", k, best_dir)
    logger.info("Отчёты: %s, %s", out_root / "yolo_rankings.json", csv_path)
    if rows:
        top = rows[0]
        logger.info(
            "Лучший combined=%.4f (yolo=%s shape=%s sim=%s) | %s",
            top.get("score_combined", top["score"]),
            top.get("score_yolo"),
            top.get("score_shape"),
            top.get("score_sim"),
            top["path"],
        )


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
