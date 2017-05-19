"""
"""
import numpy as np


def select_subhalo_indices(satellite_host_property,
        bin_edges, sorted_subhalo_host_property):
    """
    """
    __ = _check_bins_span_range(satellite_host_property, bin_edges)

    satellite_bin_indices = np.digitize(satellite_host_property, bin_edges) - 1
    subhalo_bin_indices = np.searchsorted(sorted_subhalo_host_property, bin_edges)
    high = subhalo_bin_indices[satellite_bin_indices+1]
    low = subhalo_bin_indices[satellite_bin_indices]
    result = np.floor(np.random.rand(len(low))*(high - low) + low)
    return result.astype(int)


def _check_bins_span_range(x, bin_edges):
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




