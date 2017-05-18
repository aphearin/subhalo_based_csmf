"""
"""
import numpy as np
from subhalo_selection_kernel import subhalo_index_selection
from halotools.utils import unsorting_indices


def select_subhalo_indices(host_halo_occupations, host_halo_binning_property,
        bin_edges, subhalo_binning_property):
    """
    """
    __ = _check_bins_span_range(host_halo_binning_property, bin_edges)

    satellite_host_property = np.repeat(host_halo_binning_property, host_halo_occupations)

    idx_satellite_host_property = np.argsort(satellite_host_property)
    sorted_satellite_host_property = satellite_host_property[idx_satellite_host_property]

    upper_indices = np.searchsorted(bin_edges, sorted_satellite_host_property)
    lower_indices = upper_indices - 1

    subhalo_indices = subhalo_index_selection(lower_indices, upper_indices)

    idx_unsorted = unsorting_indices(idx_satellite_host_property)
    return subhalo_indices[idx_unsorted]


def _check_bins_span_range(host_halo_binning_property, bin_edges):
    """
    """
    assert np.all(np.diff(bin_edges) > 0), "``bin_edges`` must be strictly monotonically increasing"

    msg = ("Smallest bin edge = {0} must be strictly less than\n"
        "min(host_halo_binning_property) = {1}".format(
            bin_edges[0], np.min(host_halo_binning_property)))
    assert bin_edges[0] < np.min(host_halo_binning_property), msg

    msg = ("Largest bin edge = {0} must be strictly larger than\n"
        "max(host_halo_binning_property) = {1}".format(
            bin_edges[-1], np.min(host_halo_binning_property)))
    assert bin_edges[-1] > np.max(host_halo_binning_property), msg




