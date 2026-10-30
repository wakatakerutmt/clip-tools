"""Composite engine ported from psd-tools.

Produces (color, shape, alpha) float32 triples with Porter-Duff over blending,
pass-through / isolated / knockout group semantics, and backdrop-aware color
output. Replaces the earlier uint8 single-RGBA path.
"""

from clip_tools.composite.compositor import composite, composite_pil

__all__ = ["composite", "composite_pil"]
