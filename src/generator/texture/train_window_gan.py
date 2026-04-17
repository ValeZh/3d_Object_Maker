"""
DCGAN по кропам окон (папка JPG/PNG).

По умолчанию конфиг заточен под небольшой датасет кропов из panels/: hinge, batch 16, 128×128,
300 эпох, рекурсивный обход папок, вывод в runs/window_gan_panels, drop_last=False.

Обучение (достаточно без аргументов, если данные в папке panels/):
  python src/generator/texture/train_window_gan.py

Генерация:
  python src/generator/texture/train_window_gan.py --generate --weights runs/window_gan_panels/window_generator.pt --out runs/window_gan_panels --n 20

Одна строка лога = среднее за эпоху по батчам; при малом числе батчей смотрите samples_epoch_*.png.

Чекпоинты: out/checkpoints/best.pt, out/checkpoints/epoch_XXXXX.pt каждые --checkpoint-every эпох.
При необходимости отбрасывать неполный последний батч: --drop-last.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def weights_init_dcgan(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class WindowsDataset(Dataset):
    def __init__(self, root: Path, image_size: int, recursive: bool, augment: bool):
        self.root = root
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        if recursive:
            self.paths: List[Path] = [
                p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
            ]
        else:
            self.paths = [
                p for p in root.iterdir()
                if p.is_file() and p.suffix.lower() in exts
            ]
        if not self.paths:
            raise RuntimeError(f"Нет изображений в {root} (recursive={recursive})")

        tfms = [transforms.Resize((image_size, image_size))]
        if augment:
            tfms.append(transforms.RandomHorizontalFlip(p=0.5))
        tfms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.transform = transforms.Compose(tfms)
        logger.info("Датасет: %d изображений из %s", len(self.paths), root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        image_size: int,
        img_channels: int = 3,
        ngf: int = 64,
    ):
        super().__init__()
        if not _is_power_of_two(image_size) or image_size < 64:
            raise ValueError(f"image_size: степень двойки и >= 64, получено {image_size}")
        self.z_dim = z_dim
        n_up = int(round(math.log2(image_size))) - 2
        layers: List[nn.Module] = [
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
        ]
        c = ngf * 8
        for _ in range(n_up - 1):
            c_next = max(ngf, c // 2)
            layers += [
                nn.ConvTranspose2d(c, c_next, 4, 2, 1, bias=False),
                nn.BatchNorm2d(c_next),
                nn.ReLU(True),
            ]
            c = c_next
        layers += [
            nn.ConvTranspose2d(c, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, image_size: int, img_channels: int = 3, ndf: int = 64):
        super().__init__()
        if not _is_power_of_two(image_size) or image_size < 64:
            raise ValueError(f"image_size: степень двойки и >= 64, получено {image_size}")
        layers: List[nn.Module] = [
            nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        c = ndf
        h = image_size // 2
        while h > 4:
            c_next = min(c * 2, ndf * 8)
            layers += [
                nn.Conv2d(c, c_next, 4, 2, 1, bias=False),
                nn.BatchNorm2d(c_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            c = c_next
            h //= 2
        layers += [nn.Conv2d(c, 1, 4, 1, 0, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(-1)


def train_loop(
    data_dir: Path,
    out_dir: Path,
    image_size: int,
    batch_size: int,
    z_dim: int,
    epochs: int,
    device: str,
    recursive: bool,
    augment: bool,
    num_workers: int,
    label_smooth: float,
    lr_g: float,
    lr_d: float,
    d_steps: int,
    grad_clip: float,
    gan_loss: str,
    checkpoint_every: int,
    drop_last: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ds = WindowsDataset(data_dir, image_size, recursive, augment)
    if len(ds) < 256:
        logger.warning(
            "Мало примеров (%d): метрики шумные; добавьте кропы или попробуйте --image-size 64, --batch-size 16.",
            len(ds),
        )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=drop_last,
    )

    G = Generator(z_dim, image_size).to(device)
    D = Discriminator(image_size).to(device)
    G.apply(weights_init_dcgan)
    D.apply(weights_init_dcgan)

    gan_loss = gan_loss.lower().strip()
    if gan_loss not in ("bce", "hinge"):
        raise ValueError("gan_loss: bce или hinge")
    bce = nn.BCEWithLogitsLoss()
    opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    if d_steps < 1:
        raise ValueError("d_steps >= 1")

    real_hi = 1.0 - label_smooth
    fake_hi = label_smooth
    fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)

    n_batches_loader = len(loader)
    r = len(ds) % batch_size
    if not drop_last and r != 0:
        logger.info(
            "Батчей за эпоху: %d (последний батч = %d прим.; drop_last=False — все примеры за эпоху).",
            n_batches_loader,
            r,
        )
    else:
        logger.info(
            "Батчей за эпоху: %d (мало батчей → сильный разброс D_loss/G_loss между эпохами; это не баг).",
            n_batches_loader,
        )

    cfg = {
        "z_dim": z_dim,
        "image_size": image_size,
        "n_samples": len(ds),
        "data_dir": str(data_dir.resolve()),
        "batch_size": batch_size,
        "drop_last": drop_last,
        "lr_g": lr_g,
        "lr_d": lr_d,
        "d_steps": d_steps,
        "grad_clip": grad_clip,
        "gan_loss": gan_loss,
    }
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    ema_d: float | None = None
    ema_g: float | None = None
    ema_alpha = 0.15
    best_ema_g = float("inf")
    best_epoch = 0

    def _ckpt_payload(ep: int, d_ep: float, g_ep: float) -> dict:
        return {
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "z_dim": z_dim,
            "image_size": image_size,
            "epoch": ep,
            "d_loss": d_ep,
            "g_loss": g_ep,
            "ema_d": ema_d,
            "ema_g": ema_g,
        }

    for epoch in range(1, epochs + 1):
        g_acc = 0.0
        d_acc = 0.0
        n_batches = 0

        for real in loader:
            real = real.to(device)
            bsz = real.size(0)
            n_batches += 1

            loss_d_mean = 0.0
            for _ in range(d_steps):
                opt_d.zero_grad(set_to_none=True)
                noise = torch.randn(bsz, z_dim, 1, 1, device=device)
                fake_det = G(noise).detach()
                dr = D(real)
                df = D(fake_det)
                if gan_loss == "bce":
                    loss_d = bce(dr, torch.full_like(dr, real_hi)) + bce(
                        df, torch.full_like(df, fake_hi)
                    )
                else:
                    loss_d = F.relu(1.0 - dr).mean() + F.relu(1.0 + df).mean()
                loss_d.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(D.parameters(), grad_clip)
                opt_d.step()
                loss_d_mean += loss_d.item()
            loss_d_mean /= d_steps

            opt_g.zero_grad(set_to_none=True)
            noise = torch.randn(bsz, z_dim, 1, 1, device=device)
            fake = G(noise)
            dg = D(fake)
            if gan_loss == "bce":
                loss_g = bce(dg, torch.ones_like(dg))
            else:
                loss_g = -dg.mean()
            loss_g.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(G.parameters(), grad_clip)
            opt_g.step()

            g_acc += loss_g.item()
            d_acc += loss_d_mean

        d_epoch = d_acc / max(n_batches, 1)
        g_epoch = g_acc / max(n_batches, 1)
        if ema_d is None:
            ema_d, ema_g = d_epoch, g_epoch
        else:
            ema_d = (1 - ema_alpha) * ema_d + ema_alpha * d_epoch
            ema_g = (1 - ema_alpha) * ema_g + ema_alpha * g_epoch

        logger.info(
            "Эпоха %d/%d | D=%.4f G=%.4f | сглаж. D~%.4f G~%.4f (%s)",
            epoch,
            epochs,
            d_epoch,
            g_epoch,
            ema_d,
            ema_g,
            gan_loss,
        )

        if ema_g < best_ema_g:
            best_ema_g = ema_g
            best_epoch = epoch
            best_path = ckpt_dir / "best.pt"
            torch.save(_ckpt_payload(epoch, d_epoch, g_epoch), best_path)
            (ckpt_dir / "best_meta.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "ema_g": ema_g,
                        "ema_d": ema_d,
                        "g_loss_epoch": g_epoch,
                        "d_loss_epoch": d_epoch,
                        "metric": "min_ema_g",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("Лучший чекпоинт (min G~) → %s (эпоха %d, G~=%.4f)", best_path, epoch, ema_g)

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ep_path = ckpt_dir / f"epoch_{epoch:05d}.pt"
            torch.save(_ckpt_payload(epoch, d_epoch, g_epoch), ep_path)
            logger.info("Чекпоинт каждые %d эпох: %s", checkpoint_every, ep_path)

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            with torch.no_grad():
                s = G(fixed_noise).cpu() * 0.5 + 0.5
                utils.save_image(s, out_dir / f"samples_epoch_{epoch:03d}.png", nrow=4)

    wpath = out_dir / "window_generator.pt"
    torch.save(
        {"G_state": G.state_dict(), "D_state": D.state_dict(), "z_dim": z_dim, "image_size": image_size},
        wpath,
    )
    logger.info("Сохранено: %s", wpath)


def load_generator(weights_path: Path, device: str) -> tuple[Generator, int, int]:
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "G_state" in ckpt:
        z_dim = int(ckpt.get("z_dim", 128))
        image_size = int(ckpt.get("image_size", 128))
        state = ckpt["G_state"]
    else:
        z_dim, image_size, state = 128, 128, ckpt

    g = Generator(z_dim, image_size).to(device)
    g.load_state_dict(state, strict=True)
    g.eval()
    return g, z_dim, image_size


def generate_images(weights_path: Path, out_dir: Path, n_images: int, device: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    g, z_dim, _ = load_generator(weights_path, device)
    with torch.no_grad():
        for i in range(n_images):
            z = torch.randn(1, z_dim, 1, 1, device=device)
            img = (g(z).cpu()[0] * 0.5 + 0.5).clamp(0, 1)
            utils.save_image(img, out_dir / f"gen_window_{i:04d}.png")
    logger.info("Сгенерировано %d в %s", n_images, out_dir)


def main():
    p = argparse.ArgumentParser(description="DCGAN: окна по кропам (дефолты — малый датасет panels/).")
    p.add_argument("--data", type=str, default=str(PROJECT_ROOT / "panels"))
    p.add_argument("--out", type=str, default=str(PROJECT_ROOT / "runs" / "window_gan_panels"))
    p.add_argument("--size", "--image-size", dest="size", type=int, default=128)
    p.add_argument(
        "--batch",
        "--batch-size",
        dest="batch",
        type=int,
        default=16,
        help="По умолчанию 16: больше шагов за эпоху при малом числе кропов.",
    )
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="По умолчанию 300 для сходимости на малом датасите окон.",
    )
    _rec = p.add_mutually_exclusive_group()
    _rec.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Искать изображения в подпапках (по умолчанию уже включено).",
    )
    _rec.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Только файлы прямо в корне --data.",
    )
    p.set_defaults(recursive=True)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Потоки DataLoader; на Windows при сбоях оставьте 0.",
    )
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument(
        "--loss",
        choices=("bce", "hinge"),
        default="hinge",
        help="По умолчанию hinge для малого датасита; bce — классический вариант.",
    )
    p.add_argument("--lr", type=float, default=None, help="Один lr для G и D.")
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--d-steps", type=int, default=1, help="Шагов D на один шаг G.")
    p.add_argument("--grad-clip", type=float, default=0.0, help="0 = выкл.")
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Полный чекпоинт каждые N эпох (по умолчанию 50 при длинном обучении; 0 = выкл.).",
    )
    p.add_argument(
        "--drop-last",
        action="store_true",
        help="Отбрасывать неполный последний батч (по умолчанию он учитывается — лучше при малом датасите).",
    )
    p.add_argument("--generate", action="store_true")
    p.add_argument("--weights", type=str, default="")
    p.add_argument("--n", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Устройство: %s", device)
    if device == "cpu" and not args.generate:
        logger.warning(
            "CUDA нет — обучение GAN на CPU очень медленное. "
            "GPU + torch с CUDA, или меньше --image-size / --epochs для экспериментов."
        )

    lr_g = args.lr if args.lr is not None else args.lr_g
    lr_d = args.lr if args.lr is not None else args.lr_d

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    if args.generate:
        wp = Path(args.weights) if args.weights else out_dir / "window_generator.pt"
        if not wp.is_file():
            raise FileNotFoundError(f"Нет весов: {wp}")
        generate_images(wp, out_dir / "generated", args.n, device)
        return

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Нет папки: {data_dir}\nУкажите путь от {PROJECT_ROOT} или абсолютный."
        )

    train_loop(
        data_dir=data_dir,
        out_dir=out_dir,
        image_size=args.size,
        batch_size=args.batch,
        z_dim=args.z_dim,
        epochs=args.epochs,
        device=device,
        recursive=args.recursive,
        augment=not args.no_augment,
        num_workers=args.workers,
        label_smooth=args.label_smooth,
        lr_g=lr_g,
        lr_d=lr_d,
        d_steps=args.d_steps,
        grad_clip=args.grad_clip,
        gan_loss=args.loss,
        checkpoint_every=args.checkpoint_every,
        drop_last=args.drop_last,
    )


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()
