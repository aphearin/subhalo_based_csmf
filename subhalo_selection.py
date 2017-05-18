"""
"""
import numpy as np
from subhalo_selection_kernel import subhalo_index_selection
from halotools.utils import unsorting_indices


def select_subhalo_indices(subhalo_property, bins):
    """
    """
    idx_sorted = np.argsort(subhalo_property)
    sorted_subhalo_property = subhalo_property[idx_sorted]
    raise NotImplementedError("Unfinished function - first needs a clearer API for the specific use-case")
