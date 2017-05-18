"""
"""
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport floor
import numpy as np


cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def subhalo_index_selection(long[:] lower_indices, long[:] upper_indices):
    cdef double u
    cdef int i
    cdef long low, high
    cdef int num_draws = len(upper_indices)
    cdef long[:] result = np.zeros(num_draws).astype('i8')
    for i in range(num_draws):
        u = random_uniform()
        low = lower_indices[i]
        high = upper_indices[i]
        result[i] = <long>floor(u*(high-low) + low)
    return result

