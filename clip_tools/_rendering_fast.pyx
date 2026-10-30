# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython batched inner kernels for the vector rasterizer.

Three kernels, each replacing a hot numpy loop in
``clip_tools.rendering.render_vector_line_stamp`` with a single
native pass over its data:

* ``stamp_discs_circular`` — circular disc stamps (round 4).
* ``composite_alpha_onto`` — per-stroke Porter-Duff alpha-over of a
  stroke's accumulated alpha plane onto the supersampled float32
  buffer (round 5). Called once per stroke on the stroke's bbox slice.
* ``reduce_and_cast`` — final ss-reduce + uint8 quantization pass on
  the union-bbox region of the supersampled buffer (round 5).

Elliptical and rotated stamps still go through the per-stamp numpy
path in ``rendering.py``; both branches remain reachable from Python,
so stroke-by-stroke dispatch picks the fast path when it applies and
falls back otherwise.

Semantics match the numpy references within float32 LSB (< 1 unit at
uint8 output on the files the project tracks).
"""

from libc.math cimport sqrt, floor, ceil, fabs
import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t f32_t
ctypedef cnp.uint8_t u8_t
ctypedef cnp.float64_t f64_t


def eval_curve_graph_sorted(
    cnp.ndarray[f64_t, ndim=2] pts,
    double x,
):
    """Evaluate a brush-dynamics curve at ``x``.

    ``pts`` is an ``(n, 2)`` float64 array of control points **already
    sorted by x** (caller invariant). Transcribes the pure-Python
    reference in ``clip_tools.brush_dynamics.eval_curve_graph``
    one-for-one, including the clamping, linear-only endpoint segments,
    and 8-iteration Newton solve inside the cubic Bezier for interior
    segments. Semantics are bit-exact against the reference when the
    caller feeds pre-sorted input (the reference's own ``sorted(...)``
    call is a stable sort on x, so ``build_effector_curves`` can hoist
    the sort to parse time without changing values).

    Returns 0.0 when ``n < 2``, mirroring the reference's early return.
    """
    cdef Py_ssize_t n = pts.shape[0]
    cdef Py_ssize_t i
    cdef double x0, y0, x1, y1
    cdef double p0x, p0y, p1x, p1y, h1x, h1y, h2x, h2y
    cdef double t, u, bx, dbx
    cdef double px_prev_x, px_prev_y, px_next_x, px_next_y
    cdef double seg_start_x, seg_start_y, seg_end_x, seg_end_y
    cdef int it

    if n < 2:
        return 0.0

    # Clamp to endpoints (matches reference's early returns).
    if x <= pts[0, 0]:
        return pts[0, 1]
    if x >= pts[n - 1, 0]:
        return pts[n - 1, 1]

    # n == 2: linear.
    if n == 2:
        x0 = pts[0, 0]
        y0 = pts[0, 1]
        x1 = pts[1, 0]
        y1 = pts[1, 1]
        if x1 == x0:
            return y0
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    # Find segment i such that pts[i].x <= x < pts[i+1].x.
    i = 0
    while i + 1 < n and x >= pts[i + 1, 0]:
        i += 1
    seg_start_x = pts[i, 0]
    seg_start_y = pts[i, 1]
    seg_end_x = pts[i + 1, 0]
    seg_end_y = pts[i + 1, 1]

    # Endpoint segments are always linear.
    if i == 0 or i == n - 2:
        if seg_end_x == seg_start_x:
            return seg_start_y
        return (
            seg_start_y
            + (x - seg_start_x) * (seg_end_y - seg_start_y)
            / (seg_end_x - seg_start_x)
        )

    # Interior segment: cubic Bezier with neighbor-midpoint handles.
    px_prev_x = pts[i - 1, 0]
    px_prev_y = pts[i - 1, 1]
    px_next_x = pts[i + 2, 0]
    px_next_y = pts[i + 2, 1]

    p0x = seg_start_x
    p0y = seg_start_y
    p1x = seg_end_x
    p1y = seg_end_y
    h1x = (seg_start_x + seg_end_x) * 0.5
    h1y = (seg_start_y + seg_end_y) * 0.5
    h2x = (seg_end_x + px_next_x) * 0.5
    h2y = (seg_end_y + px_next_y) * 0.5

    if p1x == p0x:
        return p0y
    # Newton init: linear interp t.
    t = (x - p0x) / (p1x - p0x)
    for it in range(8):
        u = 1.0 - t
        bx = (
            u * u * u * p0x
            + 3.0 * u * u * t * h1x
            + 3.0 * u * t * t * h2x
            + t * t * t * p1x
        )
        dbx = (
            3.0 * u * u * (h1x - p0x)
            + 6.0 * u * t * (h2x - h1x)
            + 3.0 * t * t * (p1x - h2x)
        )
        if fabs(dbx) < 1e-9:
            break
        t -= (bx - x) / dbx
        if t < 0.0:
            t = 0.0
            break
        if t > 1.0:
            t = 1.0
            break
    u = 1.0 - t
    return (
        u * u * u * p0y
        + 3.0 * u * u * t * h1y
        + 3.0 * u * t * t * h2y
        + t * t * t * p1y
    )


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


def composite_alpha_onto(
    cnp.ndarray[f32_t, ndim=3] buffer,
    cnp.ndarray[f32_t, ndim=2] stroke_alpha,
    double color_r,
    double color_g,
    double color_b,
):
    """Porter-Duff alpha-over a stroke's accumulated alpha plane onto the
    supersampled RGBA buffer.

    Equivalent to the numpy version in ``rendering.py``:

        dst_a = buffer[..., 3]
        out_a = stroke_alpha + dst_a * (1.0 - stroke_alpha)
        safe_a = where(out_a > 1e-6, out_a, 1.0)
        for ch in 0..2:
            buffer[..., ch] = (color[ch]/255 * stroke_alpha
                               + buffer[..., ch] * dst_a * (1 - stroke_alpha)) / safe_a
        buffer[..., 3] = out_a

    The three channels share ``dst_a`` and ``stroke_alpha``, read once
    per pixel. Writes land in place; a scratch allocation the size of
    the stroke bbox is avoided (numpy allocates ``out_a``, ``safe_a``
    and three temporaries per channel). Caller passes
    ``color[ch] / 255.0`` so no divide happens inside the kernel.
    """
    cdef Py_ssize_t h = buffer.shape[0]
    cdef Py_ssize_t w = buffer.shape[1]
    cdef Py_ssize_t py, px
    cdef f32_t sa, da, inv_sa, out_a, safe_a
    cdef f32_t r = <f32_t>color_r
    cdef f32_t g = <f32_t>color_g
    cdef f32_t b = <f32_t>color_b
    cdef f32_t one = <f32_t>1.0
    cdef f32_t eps = <f32_t>1e-6

    for py in range(h):
        for px in range(w):
            sa = stroke_alpha[py, px]
            da = buffer[py, px, 3]
            inv_sa = one - sa
            out_a = sa + da * inv_sa
            safe_a = out_a if out_a > eps else one
            buffer[py, px, 0] = (r * sa + buffer[py, px, 0] * da * inv_sa) / safe_a
            buffer[py, px, 1] = (g * sa + buffer[py, px, 1] * da * inv_sa) / safe_a
            buffer[py, px, 2] = (b * sa + buffer[py, px, 2] * da * inv_sa) / safe_a
            buffer[py, px, 3] = out_a


def reduce_and_cast(
    cnp.ndarray[f32_t, ndim=3] buf,
    cnp.ndarray[u8_t, ndim=3] rendered,
    Py_ssize_t ub_y0,
    Py_ssize_t ub_x0,
    Py_ssize_t region_h,
    Py_ssize_t region_w,
    Py_ssize_t ss,
    double inv_ss2,
):
    """Supersample reduce + uint8 quantize in one pass over the painted region.

    Equivalent to the numpy version in ``render_vector_line_stamp``:

        tiled = buf[ub_y0:ub_y1, ub_x0:ub_x1].reshape(region_h, ss, region_w, ss, 4)
        reduced = einsum("hswSc->hwc", tiled) * (1 / ss²)
        rendered[dst] = minimum(255, (reduced * 256).astype(int32)).astype(uint8)

    Rather than materializing the ``reduced`` float32 buffer (~¼ of
    the supersampled region's bytes but still large for Y4) and then
    reading it back for the uint8 cast, write each output pixel from
    the ``ss²`` subpixel accumulator in a single tight loop. ``inv_ss2``
    is passed in to avoid a per-pixel divide; passing it from Python
    also matches numpy's float32 scalar casting behaviour.

    Output rows cover ``[ub_y0//ss, ub_y0//ss + region_h)`` and columns
    ``[ub_x0//ss, ub_x0//ss + region_w)``. ``ub_y0`` and ``ub_x0`` are
    pre-aligned to ``ss`` tile boundaries by the caller.
    """
    cdef Py_ssize_t out_y0 = ub_y0 // ss
    cdef Py_ssize_t out_x0 = ub_x0 // ss
    cdef Py_ssize_t oy, ox, c, sy, sx
    cdef Py_ssize_t base_y, base_x
    cdef f32_t acc0, acc1, acc2, acc3
    cdef f32_t inv_f = <f32_t>inv_ss2
    cdef int v0, v1, v2, v3

    if ss == 1:
        # No reduction — straight cast.
        for oy in range(region_h):
            base_y = ub_y0 + oy
            for ox in range(region_w):
                base_x = ub_x0 + ox
                v0 = <int>(buf[base_y, base_x, 0] * <f32_t>256.0)
                v1 = <int>(buf[base_y, base_x, 1] * <f32_t>256.0)
                v2 = <int>(buf[base_y, base_x, 2] * <f32_t>256.0)
                v3 = <int>(buf[base_y, base_x, 3] * <f32_t>256.0)
                rendered[out_y0 + oy, out_x0 + ox, 0] = <u8_t>(255 if v0 > 255 else v0)
                rendered[out_y0 + oy, out_x0 + ox, 1] = <u8_t>(255 if v1 > 255 else v1)
                rendered[out_y0 + oy, out_x0 + ox, 2] = <u8_t>(255 if v2 > 255 else v2)
                rendered[out_y0 + oy, out_x0 + ox, 3] = <u8_t>(255 if v3 > 255 else v3)
        return

    for oy in range(region_h):
        base_y = ub_y0 + oy * ss
        for ox in range(region_w):
            base_x = ub_x0 + ox * ss
            acc0 = <f32_t>0.0
            acc1 = <f32_t>0.0
            acc2 = <f32_t>0.0
            acc3 = <f32_t>0.0
            for sy in range(ss):
                for sx in range(ss):
                    acc0 = acc0 + buf[base_y + sy, base_x + sx, 0]
                    acc1 = acc1 + buf[base_y + sy, base_x + sx, 1]
                    acc2 = acc2 + buf[base_y + sy, base_x + sx, 2]
                    acc3 = acc3 + buf[base_y + sy, base_x + sx, 3]
            v0 = <int>(acc0 * inv_f * <f32_t>256.0)
            v1 = <int>(acc1 * inv_f * <f32_t>256.0)
            v2 = <int>(acc2 * inv_f * <f32_t>256.0)
            v3 = <int>(acc3 * inv_f * <f32_t>256.0)
            rendered[out_y0 + oy, out_x0 + ox, 0] = <u8_t>(255 if v0 > 255 else v0)
            rendered[out_y0 + oy, out_x0 + ox, 1] = <u8_t>(255 if v1 > 255 else v1)
            rendered[out_y0 + oy, out_x0 + ox, 2] = <u8_t>(255 if v2 > 255 else v2)
            rendered[out_y0 + oy, out_x0 + ox, 3] = <u8_t>(255 if v3 > 255 else v3)
