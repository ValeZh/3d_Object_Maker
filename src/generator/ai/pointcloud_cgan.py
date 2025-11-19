#!/usr/bin/env python3
# pointcloud_cgan.py
# Conditional GAN (WGAN-GP + Chamfer) for 3D point clouds with PointNet discriminator.

import os
import sqlite3
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Chamfer Distance (замена EMD, без pytorch3d) ---
def chamfer_distance(A, B):
    """Chamfer Distance между двумя облаками точек (A, B: (B, N, 3))."""
    # Расстояния от A к B
    dist1 = torch.cdist(A, B).min(dim=2)[0].mean(dim=1)  # (B,)
    # Расстояния от B к A
    dist2 = torch.cdist(B, A).min(dim=2)[0].mean(dim=1)  # (B,)
    return dist1 + dist2

try:
    from plyfile import PlyData, PlyElement
    PLY_AVAILABLE = True
except Exception:
    PLY_AVAILABLE = False

# -------------------------------
# CONFIG
# -------------------------------
CFG = {
    "PROJECT_ROOT": os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
    "DATA_DIR": None,
    "DB_NAME": "objects_data.db",
    "OUTPUT_SUBDIR": "point_cgan_pointnet_output",

    "NUM_POINTS": 2048,           # Увеличено для детализации
    "LATENT_DIM": 128,
    "COND_DIM": 64,
    "BATCH_SIZE": 8,
    "NUM_EPOCHS": 100,
    "LR_G": 1e-4,
    "LR_D": 1e-4,
    "BETA1": 0.0,
    "BETA2": 0.9,
    "SAVE_EVERY": 10,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # WGAN-GP
    "CRITIC_ITERS": 5,
    "LAMBDA_GP": 10.0,

    # Loss weights
    "GEO_LOSS_WEIGHT": 1.0,
    "CHAMFER_WEIGHT": 0.1,        # Замена EMD
    "HEIGHT_LOSS_WEIGHT": 0.5,

    "MAX_ITEMS_FOR_RADII": 200,
    "CLIP_GRAD": 1.0
}

# --- EXPORT FOR VISUALIZER ---
NUM_POINTS = CFG["NUM_POINTS"]
LATENT_DIM = CFG["LATENT_DIM"]
COND_DIM   = CFG["COND_DIM"]
DEVICE     = torch.device(CFG["DEVICE"])

# Paths
if CFG["DATA_DIR"] is None:
    CFG["DATA_DIR"] = os.path.join(CFG["PROJECT_ROOT"], "data")
DB_PATH = os.path.join(CFG["DATA_DIR"], CFG["DB_NAME"])
OUTPUT_DIR = os.path.join(CFG["DATA_DIR"], CFG["OUTPUT_SUBDIR"])
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------
# Utils
# -------------------------------
def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def obj_to_pointcloud(obj_blob, num_points=CFG["NUM_POINTS"]):
    tmp = os.path.join(OUTPUT_DIR, "_tmp_obj.obj")
    with open(tmp, "wb") as f:
        f.write(obj_blob)
    try:
        mesh = trimesh.load(tmp, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            geoms = [g for g in mesh.geometry.values()]
            mesh = trimesh.util.concatenate(geoms)
        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
    except Exception:
        pts = np.random.normal(scale=0.05, size=(num_points, 3)).astype(np.float32)
    finally:
        try: os.remove(tmp)
        except Exception: pass

    pts = pts - pts.mean(axis=0)
    norm = np.max(np.linalg.norm(pts, axis=1))
    if norm > 0:
        pts /= norm
    return pts.astype(np.float32)

def save_pc_image(points, path, title=""):
    ensure_dir(os.path.dirname(path))
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=2, c=points[:,2], cmap='viridis')
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

def save_pointcloud_ply(points, path):
    ensure_dir(os.path.dirname(path))
    if PLY_AVAILABLE:
        verts = np.array([tuple(p) for p in points], dtype=[('x','f4'),('y','f4'),('z','f4')])
        PlyData([PlyElement.describe(verts, 'vertex')], text=True).write(path)
    else:
        np.save(path + ".npy", points)

def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# -------------------------------
# Dataset
# -------------------------------
class ObjectsSQLiteDataset(Dataset):
    def __init__(self, db_path=DB_PATH, num_points=CFG["NUM_POINTS"], max_items=None):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(
            "SELECT o.obj_data, o.shape_id, s.name "
            "FROM objects o JOIN shapes s ON o.shape_id = s.id "
            "WHERE o.obj_data IS NOT NULL"
        )
        rows = c.fetchall()
        conn.close()
        if max_items:
            rows = rows[:max_items]
        if not rows:
            raise RuntimeError("No objects found in DB.")
        self.rows = rows
        self.num_points = num_points

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        obj_blob, shape_id, shape_name = self.rows[idx]
        pts = obj_to_pointcloud(obj_blob, self.num_points)
        return torch.tensor(pts), torch.tensor(shape_id - 1), shape_name


# -------------------------------
# Models
# -------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim, cond_dim, num_points, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, cond_dim)
        h = 512
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, h),
            nn.ReLU(True),
            nn.LayerNorm(h),
            nn.Linear(h, h * 2),
            nn.ReLU(True),
            nn.LayerNorm(h * 2),
            nn.Linear(h * 2, h),
            nn.ReLU(True),
            nn.LayerNorm(h),
            nn.Linear(h, num_points * 3),
            nn.Tanh()
        )
        self.num_points = num_points

    def forward(self, z, labels):
        cond = self.label_emb(labels)
        x = torch.cat([z, cond], dim=1)
        out = self.net(x)
        return out.view(-1, self.num_points, 3)


class PointNetDiscriminator(nn.Module):
    def __init__(self, num_points, num_classes, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.label_emb = nn.Embedding(num_classes, cond_dim)
        self.fc = nn.Sequential(
            nn.Linear(512 + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, pts, labels):
        x = pts.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, dim=2)[0]
        l = self.label_emb(labels)
        return self.fc(torch.cat([x, l], dim=1))


# -------------------------------
# WGAN-GP Gradient Penalty
# -------------------------------
def gradient_penalty(D, real, fake, labels):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=DEVICE).expand_as(real)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = D(interpolates, labels)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.reshape(B, -1)  # ИСПРАВЛЕНО: reshape вместо view
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# -------------------------------
# Compute class radii
# -------------------------------
def compute_class_radii(db_path, num_points=512, max_items=None):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, name FROM shapes")
    shape_rows = c.fetchall()
    shape_map = {r[0]-1: r[1] for r in shape_rows}
    norms_per_class = {cid: [] for cid in shape_map}
    c.execute("SELECT o.obj_data, o.shape_id FROM objects o WHERE o.obj_data IS NOT NULL")
    objs = c.fetchall()
    conn.close()
    if max_items:
        objs = objs[:max_items]
    for blob, sid in objs:
        try:
            pts = obj_to_pointcloud(blob, num_points)
            norms_per_class[sid-1].append(np.mean(np.linalg.norm(pts, axis=1)))
        except Exception:
            pass
    class_radius = {cid: float(np.mean(v)) if v else 0.5 for cid, v in norms_per_class.items()}
    return class_radius, shape_map


# -------------------------------
# Training
# -------------------------------
def train():
    class_radius, shape_map = compute_class_radii(DB_PATH, CFG["NUM_POINTS"], CFG["MAX_ITEMS_FOR_RADII"])
    num_classes = len(shape_map)
    ds = ObjectsSQLiteDataset(DB_PATH, CFG["NUM_POINTS"])
    loader = DataLoader(ds, batch_size=CFG["BATCH_SIZE"], shuffle=True, drop_last=True)

    G = Generator(CFG["LATENT_DIM"], CFG["COND_DIM"], CFG["NUM_POINTS"], num_classes).to(DEVICE)
    D = PointNetDiscriminator(CFG["NUM_POINTS"], num_classes, CFG["COND_DIM"]).to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = optim.Adam(G.parameters(), lr=CFG["LR_G"], betas=(CFG["BETA1"], CFG["BETA2"]))
    opt_D = optim.Adam(D.parameters(), lr=CFG["LR_D"], betas=(CFG["BETA1"], CFG["BETA2"]))

    iteration = 0
    for epoch in range(1, CFG["NUM_EPOCHS"] + 1):
        for real_pts, shape_ids, _ in tqdm(loader, desc=f"Epoch {epoch}/{CFG['NUM_EPOCHS']}"):
            real_pts, shape_ids = real_pts.to(DEVICE), shape_ids.to(DEVICE)
            B = real_pts.size(0)
            iteration += 1

            # --- Train Critic (D) ---
            for _ in range(CFG["CRITIC_ITERS"]):
                opt_D.zero_grad()
                real_logits = D(real_pts, shape_ids)
                z = torch.randn(B, CFG["LATENT_DIM"], device=DEVICE)
                fake_pts = G(z, shape_ids).detach()
                fake_logits = D(fake_pts, shape_ids)
                gp = gradient_penalty(D, real_pts, fake_pts, shape_ids)
                loss_D = -real_logits.mean() + fake_logits.mean() + CFG["LAMBDA_GP"] * gp
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), CFG["CLIP_GRAD"])
                opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            z2 = torch.randn(B, CFG["LATENT_DIM"], device=DEVICE)
            gen_pts = G(z2, shape_ids)
            gen_logits = D(gen_pts, shape_ids)

            # WGAN loss
            loss_adv = -gen_logits.mean()

            # Geometric radius
            target_radii = torch.tensor([class_radius[int(cid.item())] for cid in shape_ids],
                                      dtype=torch.float32, device=DEVICE).view(B, 1)
            gen_radii = torch.norm(gen_pts, dim=2).mean(dim=1, keepdim=True)
            loss_geo = nn.functional.mse_loss(gen_radii, target_radii)

            # Height loss
            height_gen = gen_pts[:,:,2].max(dim=1)[0] - gen_pts[:,:,2].min(dim=1)[0]
            height_real = real_pts[:,:,2].max(dim=1)[0] - real_pts[:,:,2].min(dim=1)[0]
            loss_height = nn.functional.mse_loss(height_gen, height_real)

            # Chamfer Distance (замена EMD)
            loss_chamfer = chamfer_distance(gen_pts, real_pts).mean()

            loss_G = (loss_adv +
                      CFG["GEO_LOSS_WEIGHT"] * loss_geo +
                      CFG["HEIGHT_LOSS_WEIGHT"] * loss_height +
                      CFG["CHAMFER_WEIGHT"] * loss_chamfer)

            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), CFG["CLIP_GRAD"])
            opt_G.step()

            # --- Save samples ---
            if iteration % CFG["SAVE_EVERY"] == 0:
                with torch.no_grad():
                    G.eval()
                    for cid, name in shape_map.items():
                        z_eval = torch.randn(1, CFG["LATENT_DIM"], device=DEVICE)
                        label_eval = torch.tensor([cid], dtype=torch.long, device=DEVICE)
                        gen = G(z_eval, label_eval)[0].cpu().numpy()
                        save_pointcloud_ply(gen, os.path.join(OUTPUT_DIR, f"iter{iteration}_{name}.ply"))
                        save_pc_image(gen, os.path.join(OUTPUT_DIR, f"iter{iteration}_{name}.png"), name)
                    G.train()
                torch.save(
                    {"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch, "iter": iteration},
                    os.path.join(OUTPUT_DIR, f"ckpt_iter{iteration}.pt")
                )
                print(f"[Iter {iteration}] Saved samples & checkpoint")

        print(f"Epoch {epoch} completed.")

    final_path = os.path.join(OUTPUT_DIR, "model_final.pt")
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, final_path)
    print(f"Training finished. Final model: {final_path}")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    train()