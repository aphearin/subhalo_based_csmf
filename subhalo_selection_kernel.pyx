"""
"""
cimport cython
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport floor
import numpy as np

__all__ = ('subhalo_index_kernel', 'subhalo_index_selection')


cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def subhalo_index_selection(long[:] lower_indices, long[:] upper_indices):
    """
    """
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



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def subhalo_index_kernel(long[:] satellite_bin_indices, long[:] subhalo_bin_indices):
    """
    Parameters
    ----------
    satellite_bin_indices : ndarray
        Array of long integers of shape (num_sats, ) storing the host halo mass bin of
        every satellite galaxy in the mock.
        All values must be between 0 and num_bins - 2, inclusive

    subhalo_bin_indices : ndarray
        Array of long integers of shape (num_bins, ) storing the first index
        where a subhalo belongs in the next host mass bin

    Returns
    -------
    subhalo_indices : ndarray
        Array of long integers of shape (num_sats, ) that can be used
        as an indexing array into subhalos
    """
    cdef int num_draws = len(satellite_bin_indices)
    cdef int i, bin_index
    cdef double u
    cdef long low, high
    cdef long[:] result = np.zeros(num_draws).astype('i8')

    for i in range(num_draws):
        bin_index = satellite_bin_indices[i]
        low = subhalo_bin_indices[bin_index]
        high = subhalo_bin_indices[bin_index+1]
        u = random_uniform()
        result[i] = <long>floor(u*(high-low) + low)
    return result



