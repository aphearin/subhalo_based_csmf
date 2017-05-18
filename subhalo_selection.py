"""
"""
import numpy as np
from subhalo_selection_kernel import subhalo_index_selection
from halotools.utils import unsorting_indices


def f(host_halo_occupations, host_halo_binning_property, bin_edges, subhalo_binning_property):
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

    subhalo_host_property = np.repeat(host_halo_binning_property, host_halo_occupations)

    idx_sorted = np.argsort(subhalo_binning_property)
    sorted_subhalos = subhalo_binning_property[idx_sorted]

    high_indices = np.searchsorted(bin_edges, host_halo_binning_property)
    low_indices = high_indices - 1

    subhalo_bin_indices = np.digitize(subhalo_host_property, bin_edges)
    subhalo_bin_indices = np.where(subhalo_bin_indices < 0, 0, subhalo_bin_indices)
    subhalo_bin_indices = np.where(subhalo_bin_indices >= len(bin_edges)-1,
        len(bin_edges)-1, subhalo_bin_indices)

    raise NotImplementedError("Left off here")
    #  Not sure about the following lines
    subhalo_indices = select_subhalo_indices(subhalo_bin_indices)
    return tuple((subhalo_property[subhalo_indices] for subhalo_property in subhalo_properties))


def select_subhalo_indices(subhalo_bin_indices):
    """
    """
    idx_sorted = np.argsort(subhalo_bin_indices)

    sorted_subhalo_bin_indices = subhalo_bin_indices[idx_sorted]

    idx_unsorted = unsorting_indices(idx_sorted)
    raise NotImplementedError("Unfinished function - first needs a clearer API for the specific use-case")
