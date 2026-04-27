from __future__ import annotations

import numpy as np
from PIL import Image


def _to_gray01(image: Image.Image) -> np.ndarray:
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    g = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    g -= float(g.min())
    g /= max(float(g.max()), 1e-6)
    return g


def make_normal_map_from_albedo(image: Image.Image, *, strength: float = 3.5) -> Image.Image:
    """Approximate tangent-space normal map from albedo luminance gradients."""
    h = _to_gray01(image)
    dx = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
    dy = np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)
    nx = -dx * float(strength)
    ny = -dy * float(strength)
    nz = np.ones_like(h, dtype=np.float32)
    n = np.stack([nx, ny, nz], axis=-1)
    ln = np.sqrt((n * n).sum(axis=-1, keepdims=True))
    n /= np.clip(ln, 1e-6, None)
    rgb = ((n + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def make_roughness_map_from_albedo(
    image: Image.Image,
    *,
    min_roughness: float = 0.35,
    max_roughness: float = 0.9,
    invert: bool = False,
) -> Image.Image:
    """Generate roughness map from albedo luminance (RGB grayscale output)."""
    g = _to_gray01(image)
    if invert:
        g = 1.0 - g
    lo = float(min(min_roughness, max_roughness))
    hi = float(max(min_roughness, max_roughness))
    r = lo + (hi - lo) * g
    g8 = np.clip(r * 255.0, 0, 255).astype(np.uint8)
    rgb = np.repeat(g8[:, :, None], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")

