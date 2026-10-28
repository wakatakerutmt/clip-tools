"""Pure numeric helpers for the composite engine. All float32 in [0, 1]."""

from __future__ import annotations

import numpy as np


def union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Porter-Duff ``src OVER dst`` for premultiplied alpha: ``a + b - a*b``."""
    return a + b - a * b


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe elementwise ``a / b`` returning 1.0 wherever the result is non-finite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        c = a / b
    c[~np.isfinite(c)] = 1.0
    return c


def clip(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def paste(
    dst: np.ndarray, src: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    """Paste ``src`` into ``dst`` at ``box = (left, top, right, bottom)``.

    Both arrays share the same channel count. Returns ``dst`` (modified in place).
    """
    left, top, right, bottom = box
    dst_h, dst_w = dst.shape[:2]
    src_h, src_w = src.shape[:2]

    sx0 = max(0, -left)
    sy0 = max(0, -top)
    sx1 = min(src_w, dst_w - left)
    sy1 = min(src_h, dst_h - top)
    if sx1 <= sx0 or sy1 <= sy0:
        return dst

    dx0 = max(0, left)
    dy0 = max(0, top)
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)

    dst[dy0:dy1, dx0:dx1] = src[sy0:sy1, sx0:sx1]
    return dst
