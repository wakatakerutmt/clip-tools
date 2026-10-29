"""Build hook for the Cython extension in ``clip_tools``.

Project metadata lives in ``pyproject.toml``; this file exists only
because setuptools needs an explicit ``ext_modules`` list to invoke
``cythonize``. The extension is optional at import time — ``rendering.py``
falls back to the pure-numpy path when ``_rendering_fast`` isn't present
— but ``uv sync`` / ``pip install`` builds it by default.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import setup

setup(
    ext_modules=cythonize(
        ["clip_tools/_rendering_fast.pyx"],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[np.get_include()],
)
