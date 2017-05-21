"""
"""
from halotools.sim_manager import CachedHaloCatalog
from halotools.utils import SampleSelector
import numpy as np
from jiang_usmf import monte_carlo_subhalo_population, best_fit_param_dict
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

