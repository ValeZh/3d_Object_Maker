import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# ================== НАСТРОЙКИ ==================

PROJECT_ROOT = Path(r"D:\4course_1sem\semestr_project\3d_Object_Maker")
DATA_DIR = PROJECT_ROOT / "panels"          # папка с кропами окон
OUT_DIR = PROJECT_ROOT / "window_gan"       # куда сохранять модели и примеры

IMAGE_SIZE = 128        # размер, на котором учится GAN
BATCH_SIZE = 32
Z_DIM = 128
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================== DATASET ==================

class PanelsDataset(Dataset):
    def __init__(self, root: Path, image_size: int):
        self.root = root
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.paths: List[Path] = [
            p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts
        ]
        if not self.paths:
            raise RuntimeError(f"В папке {root} нет картинок для обучения GAN")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # в [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def get_dataloader():
    ds = PanelsDataset(DATA_DIR, IMAGE_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    return loader


# ================== МОДЕЛИ ==================

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, img_channels=3, feature_g=64):
        super().__init__()
        self.net = nn.Sequential(
            # input: Z_DIM x 1 x 1
            nn.ConvTranspose2d(z_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),  # [-1,1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, feature_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


# ================== ОБУЧЕНИЕ ==================

def train():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    loader = get_dataloader()

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    criterion = nn.BCELoss()
    optG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, Z_DIM, 1, 1, device=DEVICE)

    for epoch in range(1, EPOCHS + 1):
        for real in loader:
            real = real.to(DEVICE)
            bsz = real.size(0)

            # ---- train D ----
            noise = torch.randn(bsz, Z_DIM, 1, 1, device=DEVICE)
            fake = G(noise).detach()

            D_real = D(real)
            D_fake = D(fake)

            lossD_real = criterion(D_real, torch.ones_like(D_real))
            lossD_fake = criterion(D_fake, torch.zeros_like(D_fake))
            lossD = lossD_real + lossD_fake

            optD.zero_grad()
            lossD.backward()
            optD.step()

            # ---- train G ----
            noise = torch.randn(bsz, Z_DIM, 1, 1, device=DEVICE)
            fake = G(noise)
            D_fake_for_G = D(fake)
            lossG = criterion(D_fake_for_G, torch.ones_like(D_fake_for_G))

            optG.zero_grad()
            lossG.backward()
            optG.step()

        print(f"Epoch {epoch}/{EPOCHS} | D: {lossD.item():.4f} | G: {lossG.item():.4f}")

        if epoch % 10 == 0 or epoch == EPOCHS:
            with torch.no_grad():
                samples = G(fixed_noise).cpu()
                samples = (samples * 0.5 + 0.5)  # обратно в [0,1]
                utils.save_image(
                    samples,
                    OUT_DIR / f"samples_epoch_{epoch:03d}.png",
                    nrow=4,
                )

    torch.save(G.state_dict(), OUT_DIR / "window_generator.pt")
    print("Обучение завершено, веса сохранены в", OUT_DIR)


# ================== ГЕНЕРАЦИЯ ==================

def generate(n_images: int = 10):
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(OUT_DIR / "window_generator.pt", map_location=DEVICE))
    G.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i in range(n_images):
            z = torch.randn(1, Z_DIM, 1, 1, device=DEVICE)
            img = G(z).cpu()[0]
            img = (img * 0.5 + 0.5).clamp(0, 1)
            utils.save_image(img, OUT_DIR / f"gen_window_{i:03d}.png")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    train()
    # после обучения можно вызвать generate() отдельно, если нужно
    # generate(20)