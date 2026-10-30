"""Build hook for the Cython extension in ``clip_tools``.

Project metadata lives in ``pyproject.toml``; this file exists only
because setuptools needs an explicit ``ext_modules`` list to invoke
``cythonize``. The extension is optional at import time — ``rendering.py``
falls back to the pure-numpy path when ``_rendering_fast`` isn't present
— but ``uv sync`` / ``pip install`` builds it by default.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Composite fast kernel is bit-exact against the numpy reference, which
# depends on each float32 op being its own round-trip. Disable the
# compiler's floating-point contraction (which would silently fuse
# `a*b + c` into a single FMA with infinite internal precision) so the
# per-element arithmetic matches numpy's byte-for-byte.
_composite_ext = Extension(
    name="clip_tools.composite._composite_fast",
    sources=["clip_tools/composite/_composite_fast.pyx"],
    extra_compile_args=["-ffp-contract=off"],
)

setup(
    ext_modules=cythonize(
        [
            "clip_tools/_rendering_fast.pyx",
            _composite_ext,
        ],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
)
