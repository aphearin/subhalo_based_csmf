from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(ext_modules=cythonize("subhalo_selection_kernel.pyx"),
    include_dirs=[np.get_include()])

# compile instructions:
# python setup.py build_ext --inplace
