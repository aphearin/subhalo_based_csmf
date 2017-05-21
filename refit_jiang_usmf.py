"""
"""
import numpy as np
from halotools.sim_manager import CachedHaloCatalog
from halotools.utils import SampleSelector
from halotools.mock_observables import hod_from_mock
from jiang_usmf import monte_carlo_subhalo_population


halocat = CachedHaloCatalog(simname='bolplanck')
hosts, subs = SampleSelector.host_halo_selection(table=halocat.halo_table, return_subhalos=True)


log10_msub_min = 11
subhalo_mpeak_cut = 10**log10_msub_min

log10_mhost_min = max(11.75, log10_msub_min)
log10_mhost_max = 14.75

subhalo_sample_mask = subs['halo_mpeak'] > subhalo_mpeak_cut
subhalo_sample_mask *= subs['halo_mvir_host_halo'] > 10**log10_mhost_min
host_halo_sample_mask = hosts['halo_mvir'] > 10**log10_mhost_min

num_mhost_bins = 25
haloprop_bins = np.logspace(log10_mhost_min, log10_mhost_max, num_mhost_bins)
log10_bin_mids = 0.5*(np.log10(haloprop_bins[:-1]) + np.log10(haloprop_bins[1:]))
bin_mids = 10**log10_bin_mids


mean_occupation_bolshoi, bin_edges = hod_from_mock(
    subs['halo_mvir_host_halo'][subhalo_sample_mask],
    hosts['halo_mvir'][host_halo_sample_mask], haloprop_bins)

host_halo_counts_bolshoi = np.histogram(
    hosts['halo_mvir'][host_halo_sample_mask], bins=haloprop_bins)[0].astype('f4')
subhalo_counts_bolshoi = np.histogram(
    subs['halo_mvir_host_halo'][subhalo_sample_mask], bins=haloprop_bins)[0].astype('f4')


def subhalo_counts_prediction(beta, zeta, gamma1, alpha1, gamma2, alpha2):
    params = dict(beta=beta, zeta=zeta, gamma1=gamma1, alpha1=alpha1,
                 gamma2=gamma2, alpha2=alpha2)
    mc_nsub, mc_subhalo_mpeak = monte_carlo_subhalo_population(
        hosts['halo_mvir'][host_halo_sample_mask], log10_msub_min, np.log10(haloprop_bins), **params)
    model_subs_mhost = np.repeat(hosts['halo_mvir'][host_halo_sample_mask], mc_nsub)
    return np.histogram(model_subs_mhost, bins=haloprop_bins)[0].astype('f4')


def chi2_subhalo_counts(params):
    beta, zeta, gamma1, alpha1, gamma2, alpha2 = params

    num_mocks = 1
    chi2_arr = np.zeros(num_mocks)
    for i in range(num_mocks):
        subhalo_counts_model = subhalo_counts_prediction(beta, zeta, gamma1, alpha1, gamma2, alpha2)
        chi2_arr[i] = np.sum((subhalo_counts_model - subhalo_counts_bolshoi)**2/host_halo_counts_bolshoi)
    return np.mean(chi2_arr)

