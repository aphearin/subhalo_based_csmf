"""
"""
import numpy as np
import pytest

from subhalo_selection import select_subhalo_indices


def test_select_subhalo_indices1():

    satellite_host_property = [0.5, 1, 2.5, 4.5]
    num_sats = int(1e5)
    satellite_host_property = np.random.uniform(0.5, 4.5, num_sats)
    bin_edges = [0, 2, 4, 6]

    sorted_subhalo_host_property = [0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 5.5, 5.5, 5.5]

    result = select_subhalo_indices(satellite_host_property,
        bin_edges, sorted_subhalo_host_property)

    assert len(result) == num_sats

    ibin = 0
    bin_mask = satellite_host_property > bin_edges[ibin]
    bin_mask *= satellite_host_property < bin_edges[ibin]
    assert np.all(result[bin_mask] >= 0)
    assert np.all(result[bin_mask] <= 2)

    ibin = 1
    bin_mask = satellite_host_property > bin_edges[ibin]
    bin_mask *= satellite_host_property < bin_edges[ibin]
    assert np.all(result[bin_mask] >= 3)
    assert np.all(result[bin_mask] <= 5)

    ibin = 2
    bin_mask = satellite_host_property > bin_edges[ibin]
    bin_mask *= satellite_host_property < bin_edges[ibin]
    assert np.all(result[bin_mask] >= 6)
    assert np.all(result[bin_mask] <= 8)


def test_select_subhalo_indices2():

    satellite_host_property = [0.5, 1, 2.5, 4.5]
    num_sats = int(1e5)
    satellite_host_property = np.linspace(0.5, 4.5, num_sats)
    bin_edges = [0, 2, 4, 6]

    sorted_subhalo_host_property = [0.5, 0.5, 0.5, 5.5, 5.5, 5.5]

    with pytest.raises(AssertionError) as err:
        result = select_subhalo_indices(satellite_host_property,
            bin_edges, sorted_subhalo_host_property)
    substr = "Must have at least one subhalo per bin"
    assert substr in err.value.args[0]
