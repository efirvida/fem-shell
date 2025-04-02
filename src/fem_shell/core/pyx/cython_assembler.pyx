# cython_assembler.pyx
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION
# cython: language_level=3

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def assemble_global_matrix(double[:, ::1] K, long[:, ::1] dofs_array, double[:, :, ::1] ke_array):
    cdef int num_elements = dofs_array.shape[0]
    cdef int dofs_per_element = dofs_array.shape[1]
    cdef int e, i, j
    cdef long gi, gj
    cdef double ke_val

    for e in range(num_elements):
        for i in range(dofs_per_element):
            gi = dofs_array[e, i]
            for j in range(dofs_per_element):
                gj = dofs_array[e, j]
                ke_val = ke_array[e, i, j]
                K[gi, gj] += ke_val

@cython.boundscheck(False)
@cython.wraparound(False)
def assemble_global_vector(double[::1] f, long[:, ::1] dofs_array, double[:, ::1] fe_array):
    cdef int num_elements = dofs_array.shape[0]
    cdef int dofs_per_element = dofs_array.shape[1]
    cdef int e, i
    cdef long gi
    cdef double fe_val

    for e in range(num_elements):
        for i in range(dofs_per_element):
            gi = dofs_array[e, i]
            fe_val = fe_array[e, i]
            f[gi] += fe_val