from setuptools import setup, Extension
import numpy as np

# Configuración de la extensión Cython
extensions = [
    Extension(
        name="cython_assembler",
        sources=["src/fem_shell/core/pyx/cython_assembler.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION")],
        extra_compile_args=["-O3", "-Wno-cpp"], 
    )
]

setup(ext_modules=extensions)