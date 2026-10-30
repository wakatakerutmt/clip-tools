"""Blend function table. Port of ``psd_tools.composite.blend``.

All functions take ``(Cb, Cs)`` as float32 ndarrays in ``[0, 1]`` and return a
float32 ndarray in ``[0, 1]``. ``BLEND_FUNC`` maps clip-tools'
:py:class:`~clip_tools.constants.LayerComposite` enum to the corresponding
function. ``PASS_THROUGH`` is handled at the group level in the Compositor,
not here. ``GLOW_DODGE`` and ``ADD_GLOW`` are CSP-specific; they fall back
to ``color_dodge`` and ``linear_dodge`` respectively — a close approximation
to the standard Photoshop semantics that the existing engine already uses
elsewhere.
"""

from __future__ import annotations

import functools
import logging

import numpy as np

from clip_tools.constants import LayerComposite

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Separable blend functions
# ---------------------------------------------------------------------------


def normal(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return Cs


def multiply(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return Cb * Cs


def screen(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return Cb + Cs - (Cb * Cs)


def overlay(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return hard_light(Cs, Cb)


def darken(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return np.minimum(Cb, Cs)


def lighten(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return np.maximum(Cb, Cs)


def color_dodge(Cb: np.ndarray, Cs: np.ndarray, s: float = 1.0) -> np.ndarray:
    B = np.zeros_like(Cb, dtype=np.float32)
    B[Cs == 1] = 1
    B[Cb == 0] = 0
    index = (Cs != 1) & (Cb != 0)
    B[index] = np.minimum(1, Cb[index] / (s * (1 - Cs[index] + 1e-9)))
    return B


def color_burn(Cb: np.ndarray, Cs: np.ndarray, s: float = 1.0) -> np.ndarray:
    B = np.zeros_like(Cb, dtype=np.float32)
    B[Cb == 1] = 1
    index = (Cb != 1) & (Cs != 0)
    B[index] = 1 - np.minimum(1, (1 - Cb[index]) / (s * Cs[index] + 1e-9))
    return B


def linear_dodge(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return np.minimum(1, Cb + Cs)


def linear_burn(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return np.maximum(0, Cb + Cs - 1)


def hard_light(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    index = Cs > 0.5
    B = multiply(Cb, 2 * Cs)
    B[index] = screen(Cb, 2 * Cs - 1)[index]
    return B


def soft_light(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    index = Cs <= 0.25
    index_not = ~index
    D = np.zeros_like(Cb, dtype=np.float32)
    D[index] = ((16 * Cb[index] - 12) * Cb[index] + 4) * Cb[index]
    D[index_not] = np.sqrt(Cb[index_not])

    index = Cs <= 0.5
    index_not = ~index
    B = np.zeros_like(Cb, dtype=np.float32)
    B[index] = Cb[index] - (1 - 2 * Cs[index]) * Cb[index] * (1 - Cb[index])
    B[index_not] = Cb[index_not] + (2 * Cs[index_not] - 1) * (
        D[index_not] - Cb[index_not]
    )
    return B


def vivid_light(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    Cs2 = Cs * 2
    index = Cs > 0.5
    B = color_burn(Cb, Cs2)
    D = color_dodge(Cb, Cs2 - 1)
    B[index] = D[index]
    return B


def linear_light(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    index = Cs > 0.5
    B = linear_burn(Cb, 2 * Cs)
    B[index] = linear_dodge(Cb, 2 * Cs - 1)[index]
    return B


def pin_light(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    index = Cs > 0.5
    B = darken(Cb, 2 * Cs)
    B[index] = lighten(Cb, 2 * Cs - 1)[index]
    return B


def difference(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return np.abs(Cb - Cs)


def exclusion(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return Cb + Cs - 2 * Cb * Cs


def subtract(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return np.maximum(0, Cb - Cs)


def hard_mix(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    B = np.zeros_like(Cb, dtype=np.float32)
    B[(Cb + 0.999999 * Cs) >= 1] = 1
    return B


def divide_blend(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    B = Cb / (Cs + 1e-9)
    B[B > 1] = 1
    return B


# ---------------------------------------------------------------------------
# Non-separable (HSL) blend functions
# ---------------------------------------------------------------------------


def non_separable(k: str = "s"):
    """Wrap a non-separable blend for CMYK handling (RGB passes through)."""

    def decorator(func):
        @functools.wraps(func)
        def _blend_fn(Cb, Cs):
            if Cs.shape[2] == 4:
                K = Cs[:, :, 3:4] if k == "s" else Cb[:, :, 3:4]
                Cb_, Cs_ = _cmyk2rgb(Cb), _cmyk2rgb(Cs)
                return np.concatenate((_rgb2cmy(func(Cb_, Cs_), K), K), axis=2)
            return func(Cb, Cs)

        return _blend_fn

    return decorator


def _cmyk2rgb(C: np.ndarray) -> np.ndarray:
    return np.stack([(1.0 - C[:, :, i]) * (1.0 - C[:, :, 3]) for i in range(3)], axis=2)


def _rgb2cmy(C: np.ndarray, K: np.ndarray) -> np.ndarray:
    K = np.repeat(K, 3, axis=2)
    color = np.zeros((C.shape[0], C.shape[1], 3))
    index = K < 1.0
    color[index] = (1.0 - C[index] - K[index]) / (1.0 - K[index] + 1e-9)
    return color


@non_separable()
def hue(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return _set_lum(_set_sat(Cs, _sat(Cb)), _lum(Cb))


@non_separable()
def saturation(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return _set_lum(_set_sat(Cb, _sat(Cs)), _lum(Cb))


@non_separable()
def color_blend(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return _set_lum(Cs, _lum(Cb))


@non_separable("s")
def luminosity(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    return _set_lum(Cb, _lum(Cs))


@non_separable()
def darker_color(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    index = np.repeat(_lum(Cs) < _lum(Cb), 3, axis=2)
    B = Cb.copy()
    B[index] = Cs[index]
    return B


@non_separable()
def lighter_color(Cb: np.ndarray, Cs: np.ndarray) -> np.ndarray:
    index = np.repeat(_lum(Cs) > _lum(Cb), 3, axis=2)
    B = Cb.copy()
    B[index] = Cs[index]
    return B


# Helper functions from PDF reference.
def _lum(C: np.ndarray) -> np.ndarray:
    return 0.3 * C[:, :, 0:1] + 0.59 * C[:, :, 1:2] + 0.11 * C[:, :, 2:3]


def _set_lum(C: np.ndarray, L: np.ndarray) -> np.ndarray:
    d = L - _lum(C)
    return _clip_color(C + d)


def _clip_color(C: np.ndarray) -> np.ndarray:
    L = np.repeat(_lum(C), 3, axis=2)
    C_min = np.repeat(np.min(C, axis=2, keepdims=True), 3, axis=2)
    C_max = np.repeat(np.max(C, axis=2, keepdims=True), 3, axis=2)

    index = C_min < 0.0
    L_i = L[index]
    C[index] = L_i + (C[index] - L_i) * L_i / (L_i - C_min[index] + 1e-9)

    index = C_max > 1.0
    L_i = L[index]
    C[index] = L_i + (C[index] - L_i) * (1 - L_i) / (C_max[index] - L_i + 1e-9)

    C[C < 0.0] = 0
    C[C > 1] = 1
    return C


def _sat(C):
    return np.max(C, axis=2, keepdims=True) - np.min(C, axis=2, keepdims=True)


def _set_sat(C: np.ndarray, s: np.ndarray) -> np.ndarray:
    s = np.repeat(s, 3, axis=2)

    C_max = np.repeat(np.max(C, axis=2, keepdims=True), 3, axis=2)
    C_mid = np.repeat(np.median(C, axis=2, keepdims=True), 3, axis=2)
    C_min = np.repeat(np.min(C, axis=2, keepdims=True), 3, axis=2)

    B = np.zeros_like(C, dtype=np.float32)

    index_diff = C_max > C_min
    index_mid = C == C_mid
    index_max = (C == C_max) & ~index_mid
    index_min = C == C_min

    index = index_mid & index_diff
    B[index] = (
        (C_mid[index] - C_min[index]) * s[index] / (C_max[index] - C_min[index] + 1e-9)
    )
    index = index_max & index_diff
    B[index] = s[index]

    B[~index_diff & index_mid] = 0
    B[~index_diff & index_max] = 0

    B[index_min] = 0

    return B


# ---------------------------------------------------------------------------
# BLEND_FUNC table keyed by clip-tools' LayerComposite enum.
# ---------------------------------------------------------------------------

BLEND_FUNC = {
    LayerComposite.NORMAL: normal,
    LayerComposite.DARKEN: darken,
    LayerComposite.MULTIPLY: multiply,
    LayerComposite.COLOR_BURN: color_burn,
    LayerComposite.LINEAR_BURN: linear_burn,
    LayerComposite.SUBTRACT: subtract,
    LayerComposite.DARKER_COLOR: darker_color,
    LayerComposite.LIGHTEN: lighten,
    LayerComposite.SCREEN: screen,
    LayerComposite.COLOR_DODGE: color_dodge,
    # CSP-specific GLOW_DODGE / ADD_GLOW: closest standard Photoshop match.
    # Bit-exact parity for these is a separate follow-up (see clip-tools#17).
    LayerComposite.GLOW_DODGE: color_dodge,
    LayerComposite.ADD: linear_dodge,
    LayerComposite.ADD_GLOW: linear_dodge,
    LayerComposite.LIGHTER_COLOR: lighter_color,
    LayerComposite.OVERLAY: overlay,
    LayerComposite.SOFT_LIGHT: soft_light,
    LayerComposite.HARD_LIGHT: hard_light,
    LayerComposite.VIVID_LIGHT: vivid_light,
    LayerComposite.LINEAR_LIGHT: linear_light,
    LayerComposite.PIN_LIGHT: pin_light,
    LayerComposite.HARD_MIX: hard_mix,
    LayerComposite.DIFFERENCE: difference,
    LayerComposite.EXCLUSION: exclusion,
    LayerComposite.HUE: hue,
    LayerComposite.SATURATION: saturation,
    LayerComposite.COLOR: color_blend,
    LayerComposite.LUMINOSITY: luminosity,
    LayerComposite.DIVIDE: divide_blend,
    # PASS_THROUGH is NOT in BLEND_FUNC — handled at the group level.
}
