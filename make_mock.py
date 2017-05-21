"""
"""
from halotools.empirical_models import Behroozi10SmHm
from halotools.sim_manager import CachedHaloCatalog
import numpy as np
from jiang_usmf import monte_carlo_subhalo_population, best_fit_param_dict, _check_bins
from copy import deepcopy
best_fit_param_dict = deepcopy(best_fit_param_dict)
del best_fit_param_dict['chi2']


def retrieve_halo_catalog(log10_msub_min, log10_mhost_min, **sim_manager_keys):
    msg = "log10_msub_min = {0} must be greater than log10_mhost_min = {1}"
    assert log10_msub_min >= log10_mhost_min, msg.format(log10_msub_min, log10_mhost_min)
    halocat = CachedHaloCatalog(simname='bolplanck')
    host_mask = halocat.halo_table['halo_upid'] == -1
    hosts = halocat.halo_table[host_mask]
    subs = halocat.halo_table[~host_mask]

    subhalo_mpeak_cut = 10**log10_msub_min
    subhalo_sample_mask = subs['halo_mpeak'] > subhalo_mpeak_cut
    subhalo_sample_mask *= subs['halo_mvir_host_halo'] >= 10**log10_mhost_min
    subs = subs[subhalo_sample_mask]

    host_halo_sample_mask = hosts['halo_mvir'] > 10**log10_mhost_min
    hosts = hosts[host_halo_sample_mask]

    hosts.sort(['halo_mvir', 'halo_id'])
    subs.sort(['halo_mvir_host_halo', 'halo_upid'])

    return hosts, subs


def make_mock(hosts, subs, **kwargs):
    """
    """
    try:
        log10_mhost_bin_edges = kwargs['log10_mhost_bin_edges']
    except:
        num_mhost_bins = kwargs.get('num_mhost_bins', 20)
        log10_mhost_max = kwargs.get('log10_mhost_max', np.log10(hosts['halo_mvir'].max()) + 0.01)
        log10_mhost_min = kwargs.get('log10_mhost_min', np.log10(hosts['halo_mvir'].min()))
        log10_mhost_bin_edges = np.linspace(log10_mhost_min, log10_mhost_max, num_mhost_bins)
    mhost_bin_edges = 10**log10_mhost_bin_edges

    _check_bins(hosts['halo_mvir'], mhost_bin_edges)

    log10_msub_min = np.log10(subs['halo_mpeak'].min())
    mc_nsub, mc_subhalo_mpeak = monte_carlo_subhalo_population(
        hosts['halo_mvir'], log10_msub_min, np.log10(mhost_bin_edges), **best_fit_param_dict)

    mpeak_mock = np.append(hosts['halo_mpeak'], mc_subhalo_mpeak)

    model = Behroozi10SmHm(redshift=0)
    sm_satellites = model.mc_stellar_mass(prim_haloprop=mc_subhalo_mpeak)
    sm_centrals = model.mc_stellar_mass(prim_haloprop=hosts['halo_mpeak'])
    sm_mock = np.append(sm_centrals, sm_satellites)

    mhost_mock = np.append(hosts['halo_mvir'], np.repeat(hosts['halo_mvir'], mc_nsub))

    return mpeak_mock, sm_mock, mhost_mock
