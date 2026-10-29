# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython batched inner kernels for the vector rasterizer.

Replaces the per-stamp ``disc_alpha_into`` call in
``clip_tools.rendering.render_vector_line_stamp``'s innermost loop
with a single native call that walks the entire sample list of a
stroke. Keeps per-stamp Python / numpy overhead out of the hot path.

Scope: circular stamps only (``rx == ry``), which covers the common
hard- and soft-pen paths and is what the Y4 profile shows as the hot
path. Elliptical and rotated stamps still go through the per-stamp
numpy path in ``rendering.py`` (``disc_alpha_into``). Both branches
remain reachable from Python, so stroke-by-stroke dispatch picks the
fast path when it applies and falls back otherwise.

Semantics match ``disc_alpha_into`` for the circular case bit-for-bit
within float32 (max abs diff 5.96e-08 against the numpy reference —
below the uint8 quantization threshold).
"""

from libc.math cimport sqrt, floor, ceil
import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t f32_t


def stamp_discs_circular(
    cnp.ndarray[f32_t, ndim=2] stroke_alpha,
    cnp.ndarray[cnp.float64_t, ndim=1] cxs,
    cnp.ndarray[cnp.float64_t, ndim=1] cys,
    cnp.ndarray[cnp.float64_t, ndim=1] radii,
    cnp.ndarray[cnp.float64_t, ndim=1] opacities,
    double hardness,
    bint accumulate,
):
    """Apply a batch of circular stamps.

    - ``hardness >= 1.0``: inside the circle, src_a = min(r + 0.5 - dist, 1).
      Matches ``disc_alpha_into``'s hard path.
    - ``hardness < 1.0``: inner ``hardness * r`` is full opacity, falls off
      linearly to 0 at ``r``. Matches the soft path.
    - ``accumulate=False``: MAX accumulation (hard pens).
    - ``accumulate=True``: Porter-Duff within-stroke accumulation (soft brushes).
    """
    cdef Py_ssize_t n = cxs.shape[0]
    cdef Py_ssize_t buf_h = stroke_alpha.shape[0]
    cdef Py_ssize_t buf_w = stroke_alpha.shape[1]
    cdef Py_ssize_t i, py, px, x0, y0, x1, y1
    cdef double cx, cy, r, alpha, outer, dx, dy, dist, src_a, inner_r, falloff_span
    cdef f32_t src_f, old
    cdef bint hard = hardness >= 1.0

    for i in range(n):
        cx = cxs[i]
        cy = cys[i]
        r = radii[i]
        if r < 0.5:
            r = 0.5
        alpha = opacities[i]
        outer = r + 2.0
        x0 = <Py_ssize_t>floor(cx - outer)
        y0 = <Py_ssize_t>floor(cy - outer)
        x1 = <Py_ssize_t>ceil(cx + outer)
        y1 = <Py_ssize_t>ceil(cy + outer)
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > buf_w:
            x1 = buf_w
        if y1 > buf_h:
            y1 = buf_h
        if x1 <= x0 or y1 <= y0:
            continue

        if hard:
            for py in range(y0, y1):
                dy = <double>py + 0.5 - cy
                for px in range(x0, x1):
                    dx = <double>px + 0.5 - cx
                    dist = sqrt(dx * dx + dy * dy)
                    src_a = r + 0.5 - dist
                    if src_a <= 0.0:
                        continue
                    if src_a > 1.0:
                        src_a = 1.0
                    src_f = <f32_t>(src_a * alpha)
                    if accumulate:
                        old = stroke_alpha[py, px]
                        stroke_alpha[py, px] = old + src_f * (<f32_t>1.0 - old)
                    else:
                        old = stroke_alpha[py, px]
                        if src_f > old:
                            stroke_alpha[py, px] = src_f
        else:
            # Soft: inner_r full-opacity, linear falloff to 0 at r.
            inner_r = (hardness if hardness > 0.0 else 0.0) * r
            falloff_span = r - inner_r
            if falloff_span < 1e-3:
                falloff_span = 1e-3
            for py in range(y0, y1):
                dy = <double>py + 0.5 - cy
                for px in range(x0, x1):
                    dx = <double>px + 0.5 - cx
                    dist = sqrt(dx * dx + dy * dy)
                    if dist > r:
                        continue
                    if dist <= inner_r:
                        src_a = 1.0
                    else:
                        src_a = (r - dist) / falloff_span
                        if src_a < 0.0:
                            src_a = 0.0
                        elif src_a > 1.0:
                            src_a = 1.0
                    src_f = <f32_t>(src_a * alpha)
                    if accumulate:
                        old = stroke_alpha[py, px]
                        stroke_alpha[py, px] = old + src_f * (<f32_t>1.0 - old)
                    else:
                        old = stroke_alpha[py, px]
                        if src_f > old:
                            stroke_alpha[py, px] = src_f
