""" Moduling storing the Numpy-based kernel function ``select_subhalo_indices``
that can be used to provide a mapping between satellites onto subhalos
"""
import numpy as np
from astropy.utils.misc import NumpyRNGContext


def select_subhalo_indices(satellite_host_property,
        bin_edges, sorted_subhalo_host_property, seed=None):
    """ Calculate an array of integers that can be used as an indexing array
    into the input subhalos.

    Parameters
    ----------
    satellite_host_property : ndarray
        Numpy array of shape (num_satellites, ) storing, for example,
        the host halo mass of each model satellite

    bin_edges : ndarray
        Numpy array of shape (num_bins, ) defining the binning of satellites and subhalos

    sorted_subhalo_host_property : ndarray
        Monotonically increasing array of shape (num_subhalos, ) storing, for example,
        the host halo mass of each subhalo.
        Each subhalo in this array will be treated as a candidate satellite.

    seed : int, optional
        Random number seed used in the Monte Carlo selection of subhalos.
        Default is None for stochastic results

    Returns
    -------
    subhalo_indices : ndarray
        Integer array of shape (num_satellites, ) storing the index providing the
        correspondence between satellites and subhalos
    """
    __ = _check_bins_span_satellite_range(satellite_host_property, bin_edges)
    satellite_bin_indices = np.digitize(satellite_host_property, bin_edges) - 1

    assert np.all(np.diff(sorted_subhalo_host_property) >= 0), "subhalo property must be sorted"
    subhalo_bin_indices = np.searchsorted(sorted_subhalo_host_property, bin_edges)
    __ = _check_at_least_one_sub_per_bin(subhalo_bin_indices)

    high = subhalo_bin_indices[satellite_bin_indices+1]
    low = subhalo_bin_indices[satellite_bin_indices]

    with NumpyRNGContext(seed):
        uran = np.random.rand(len(low))

    return np.array(np.floor(uran*(high - low) + low)).astype(int)


def _check_bins_span_satellite_range(x, bin_edges):
    """
    """
    msg = ("Smallest bin edge = {0} must be strictly less than\n"
        "min(host_halo_binning_property) = {1}".format(
            bin_edges[0], np.min(x)))
    assert bin_edges[0] < np.min(x), msg

    msg = ("Largest bin edge = {0} must be strictly larger than\n"
        "max(host_halo_binning_property) = {1}".format(
            bin_edges[-1], np.max(x)))
    assert bin_edges[-1] > np.max(x), msg


def _check_at_least_one_sub_per_bin(subhalo_bin_indices):
    assert np.all(np.diff(subhalo_bin_indices) > 0), "Must have at least one subhalo per bin"



