# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""Cython inner kernels for the composite engine.

Replaces the hottest numpy expression in
``clip_tools.composite.compositor.Compositor._apply_source`` with a
single native pass. Scope is deliberately narrow:

* NORMAL blend mode (``LayerComposite.NORMAL``)
* non-knockout (``knockout=False``) — all CSP leaf layers currently

Everything else (MULTIPLY/SCREEN/… blends, PASS_THROUGH knockout group
nesting) stays on the numpy fallback by letting ``_apply_source``
dispatch at the top of the function. The two paths remain
byte-equivalent because the kernel transcribes numpy's per-element
expression order one-for-one (same ``*`` associativity, same float32
intermediate rounding).

The numpy expression (specialized to NORMAL non-knockout) is:

    shape_g'  = shape_g + shape - shape_g * shape
    alpha_g'  = alpha_g + alpha - alpha_g * alpha
    alpha'    = alpha_0 + alpha_g' - alpha_0 * alpha_g'
    color_t   = (shape - alpha) * alpha_prev * color_state
              + alpha * ((1 - alpha_prev) * color + alpha_prev * color)
    numerator = (1 - shape) * alpha_prev * color_state + color_t
    color'    = clip(numerator / alpha')   # divide fallback: 1.0 where alpha' == 0

Shapes:
* ``color_state`` (self._color):  (H, W, 3) float32  — mutated in place
* ``shape_g``, ``alpha_g``, ``alpha_state``, ``alpha_0``, ``src_shape``,
  ``src_alpha``:  (H, W, 1) float32
* ``src_color``:  (H, W, 3) float32

``alpha_state`` is mutated so the caller does not need to assign the
new value; ``shape_g`` and ``alpha_g`` are mutated too.
"""

import numpy as np
cimport numpy as cnp

ctypedef cnp.float32_t f32_t


def apply_source_normal(
    cnp.ndarray[f32_t, ndim=3] color_state,
    cnp.ndarray[f32_t, ndim=3] shape_g,
    cnp.ndarray[f32_t, ndim=3] alpha_g,
    cnp.ndarray[f32_t, ndim=3] alpha_state,
    cnp.ndarray[f32_t, ndim=3] alpha_0,
    cnp.ndarray[f32_t, ndim=3] src_color,
    cnp.ndarray[f32_t, ndim=3] src_shape,
    cnp.ndarray[f32_t, ndim=3] src_alpha,
):
    """Inner kernel for NORMAL + non-knockout path of ``_apply_source``.

    Caller guarantees shapes match and ``blend_mode == NORMAL``,
    ``knockout == False``. On return, ``color_state``, ``shape_g``,
    ``alpha_g`` and ``alpha_state`` reflect the post-apply state.
    """
    cdef Py_ssize_t h = color_state.shape[0]
    cdef Py_ssize_t w = color_state.shape[1]
    cdef Py_ssize_t py, px

    cdef f32_t s, a_src, a_prev, a0, sg, ag
    cdef f32_t cr, cg, cb
    cdef f32_t cr_state, cg_state, cb_state
    cdef f32_t sg_new, ag_new, a_new
    cdef f32_t one_minus_s, s_minus_a, one_minus_a_prev
    cdef f32_t inner_r, inner_g, inner_b
    cdef f32_t color_t_r, color_t_g, color_t_b
    cdef f32_t num_r, num_g, num_b
    cdef f32_t out_r, out_g, out_b
    # Explicit per-step temporaries. Each is a distinct f32 round-trip
    # that mirrors numpy's per-array materialization order; writing the
    # intermediates to typed locals prevents the C compiler from fusing
    # the multiply-add into a single FMA (which has infinite internal
    # precision and would silently diverge from numpy by 1 ULP every op).
    cdef f32_t t_sg_s, t_ag_a, t_a0_ag_new
    cdef f32_t t_sma_ap, t_oms_ap
    cdef f32_t t_inner_a_r, t_inner_a_g, t_inner_a_b
    cdef f32_t t_inner_b_r, t_inner_b_g, t_inner_b_b
    cdef f32_t t_ct_a_r, t_ct_a_g, t_ct_a_b
    cdef f32_t t_num_a_r, t_num_a_g, t_num_a_b
    cdef f32_t one = <f32_t>1.0
    cdef f32_t zero = <f32_t>0.0

    for py in range(h):
        for px in range(w):
            # Read inputs for this pixel.
            s = src_shape[py, px, 0]
            a_src = src_alpha[py, px, 0]
            cr = src_color[py, px, 0]
            cg = src_color[py, px, 1]
            cb = src_color[py, px, 2]

            sg = shape_g[py, px, 0]
            ag = alpha_g[py, px, 0]
            a_prev = alpha_state[py, px, 0]
            a0 = alpha_0[py, px, 0]
            cr_state = color_state[py, px, 0]
            cg_state = color_state[py, px, 1]
            cb_state = color_state[py, px, 2]

            # Group accumulators via Porter-Duff union: a + b - a*b.
            # Numpy evaluates each binary op as its own array temp, so we
            # mirror that: materialize `a*b` into an f32 before subtract.
            t_sg_s = sg * s
            sg_new = sg + s
            sg_new = sg_new - t_sg_s

            t_ag_a = ag * a_src
            ag_new = ag + a_src
            ag_new = ag_new - t_ag_a

            # a_new = union(a0, ag_new) = a0 + ag_new - a0 * ag_new.
            t_a0_ag_new = a0 * ag_new
            a_new = a0 + ag_new
            a_new = a_new - t_a0_ag_new

            # Derived scalars (shared across 3 color channels).
            s_minus_a = s - a_src
            one_minus_a_prev = one - a_prev
            one_minus_s = one - s
            t_sma_ap = s_minus_a * a_prev   # f32 ("shape - alpha" * alpha_prev)
            t_oms_ap = one_minus_s * a_prev   # f32 ("(1 - shape)" * alpha_prev)

            # For NORMAL + non-knockout:
            #   inner  = (1 - alpha_prev) * color + alpha_prev * color
            #   color_t = (shape - alpha) * alpha_prev * color_state
            #           + alpha * inner
            #   num = (1 - shape) * alpha_prev * color_state + color_t
            # Each * and + is an independent f32 statement, stored into a
            # typed local, matching numpy's per-op precision exactly.
            t_inner_a_r = one_minus_a_prev * cr
            t_inner_b_r = a_prev * cr
            inner_r = t_inner_a_r + t_inner_b_r

            t_inner_a_g = one_minus_a_prev * cg
            t_inner_b_g = a_prev * cg
            inner_g = t_inner_a_g + t_inner_b_g

            t_inner_a_b = one_minus_a_prev * cb
            t_inner_b_b = a_prev * cb
            inner_b = t_inner_a_b + t_inner_b_b

            t_ct_a_r = t_sma_ap * cr_state
            color_t_r = a_src * inner_r
            color_t_r = t_ct_a_r + color_t_r

            t_ct_a_g = t_sma_ap * cg_state
            color_t_g = a_src * inner_g
            color_t_g = t_ct_a_g + color_t_g

            t_ct_a_b = t_sma_ap * cb_state
            color_t_b = a_src * inner_b
            color_t_b = t_ct_a_b + color_t_b

            t_num_a_r = t_oms_ap * cr_state
            num_r = t_num_a_r + color_t_r

            t_num_a_g = t_oms_ap * cg_state
            num_g = t_num_a_g + color_t_g

            t_num_a_b = t_oms_ap * cb_state
            num_b = t_num_a_b + color_t_b

            # divide + clip. cu.divide replaces non-finite (0/0, x/0) with 1.0,
            # then cu.clip clamps to [0, 1]; 1.0 is in-range so the fallback
            # path short-circuits to 1.0 directly.
            if a_new == zero:
                out_r = one
                out_g = one
                out_b = one
            else:
                out_r = num_r / a_new
                out_g = num_g / a_new
                out_b = num_b / a_new
                # Clip.
                if out_r < zero:
                    out_r = zero
                elif out_r > one:
                    out_r = one
                if out_g < zero:
                    out_g = zero
                elif out_g > one:
                    out_g = one
                if out_b < zero:
                    out_b = zero
                elif out_b > one:
                    out_b = one

            # Write back.
            color_state[py, px, 0] = out_r
            color_state[py, px, 1] = out_g
            color_state[py, px, 2] = out_b
            shape_g[py, px, 0] = sg_new
            alpha_g[py, px, 0] = ag_new
            alpha_state[py, px, 0] = a_new
