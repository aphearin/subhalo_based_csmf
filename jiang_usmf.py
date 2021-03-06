""" Module storing kernel functions for generating Monte Carlo realizations of
the Jiang & van den Bosch 2014 fitting function for the unevolved subhalo mass function
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import poisson
from collections import OrderedDict


__all__ = ('unevolved_subhalo_mass_function', )


beta_jiang14 = 5.67
zeta_jiang14 = 1.19
gamma1_jiang14 = 0.13
alpha1_jiang14 = -0.83
gamma2_jiang14 = 1.33
alpha2_jiang14 = -0.02
jiang14_param_dict = OrderedDict()
jiang14_param_dict['beta'] = beta_jiang14
jiang14_param_dict['zeta'] = zeta_jiang14
jiang14_param_dict['gamma1'] = gamma1_jiang14
jiang14_param_dict['alpha1'] = alpha1_jiang14
jiang14_param_dict['gamma2'] = gamma2_jiang14
jiang14_param_dict['alpha2'] = alpha2_jiang14


def unevolved_subhalo_mass_function(log10_mu,
        beta=beta_jiang14, zeta=zeta_jiang14, gamma1=gamma1_jiang14,
        alpha1=alpha1_jiang14, gamma2=gamma2_jiang14, alpha2=alpha2_jiang14):
    """ Average number of subhalos of mass mu = Msub/Mhost.

    ((gamma1 * mu**alpha1) + (gamma2 * mu**alpha2)) * exp(-beta * mu**zeta)
    """
    mu = 10.**log10_mu
    prefactor_term1 = gamma1*mu**alpha1
    prefactor_term2 = gamma2*mu**alpha2
    prefactor = prefactor_term1 + prefactor_term2

    exparg = beta*mu**zeta

    return prefactor*np.exp(-exparg)


def mean_nsub_mu_range(log10_mu_min, log10_mu_max,
        beta=beta_jiang14, zeta=zeta_jiang14, gamma1=gamma1_jiang14,
        alpha1=alpha1_jiang14, gamma2=gamma2_jiang14, alpha2=alpha2_jiang14):
    """
    """
    params = (beta, zeta, gamma1, alpha1, gamma2, alpha2)
    return quad(unevolved_subhalo_mass_function, log10_mu_min, log10_mu_max, params)[0]


def mean_nsub_vs_mhost(mhost_array, log10_msub_min,
        log10_mu_max=0., log10_mhost_max=15.5, **kwargs):
    """
    """
    npts_mhost_grid = 100
    log10_mhost_min = log10_msub_min
    log10_mhost_grid = np.linspace(log10_mhost_min, log10_mhost_max, npts_mhost_grid)
    log10_mu_min_grid = log10_msub_min - log10_mhost_grid
    mean_nsub_grid = list(mean_nsub_mu_range(log10_mu_min, log10_mu_max, **kwargs)
        for log10_mu_min in log10_mu_min_grid)
    return np.interp(np.log10(mhost_array), log10_mhost_grid, mean_nsub_grid)


def monte_carlo_subhalo_mass(mhost, log10_msub_min, counts, randoms=None, **kwargs):
    """ For an input host halo mass, generate a Monte Carlo realization
    of the Mpeak values of a subhalo population according to the unevolved mass function
    given by Eq. (21) of Jiang & van den Bosch (2014), arXiv:1311.5225.

    Parameters
    ----------
    mhost : float
        Mass of the host halo in Msun/h

    log10_msub_min : float
        Lower limit on the returned subhalo masses, typically determined by
        the resolution of the simulation

    counts : int
        Number of satellites in the halo

    randoms : ndarray, optional
        Array to use as uniform randoms in the Monte Carlo realization.
        Smaller values of randoms will correlate with larger values of mu.
        Default is None for stochastic Mpeak values.

    Returns
    --------
    msub_array : ndarray
        Numpy array of shape (counts, ) storing a Monte Carlo realization
        of the mpeak value of every subhalo.

    """
    npts_mu_grid = 100
    log10_mu_min = log10_msub_min - np.log10(mhost)
    log10_mu_max = 0
    log10_mu_grid = np.linspace(log10_mu_min, log10_mu_max, npts_mu_grid)[::-1]
    num_subs_above_mu_min = mean_nsub_mu_range(log10_mu_min, log10_mu_max, **kwargs)
    num_subs_grid = np.array(list(mean_nsub_mu_range(log10_mu, log10_mu_max, **kwargs)
        for log10_mu in log10_mu_grid))
    mu_cdf_grid = num_subs_grid/num_subs_above_mu_min

    if randoms is None:
        randoms = np.random.rand(counts)
    else:
        assert np.shape(randoms) == (counts, )

    mc_log10_mu = np.interp(randoms, mu_cdf_grid, log10_mu_grid)
    return 10**mc_log10_mu


def monte_carlo_subhalo_population(mhost_array, log10_msub_min, log10_mhost_bin_edges,
        **kwargs):
    """
    """
    __ = _check_bins(mhost_array, 10**log10_mhost_bin_edges)

    log10_mhost_bin_edges[-1] = log10_mhost_bin_edges[-1] + 0.001
    mean_nsub = mean_nsub_vs_mhost(mhost_array, log10_msub_min, **kwargs)

    try:
        uniform_variate = kwargs['percentile']
    except:
        uniform_variate = np.random.rand(len(mean_nsub))
    mc_nsub = poisson.isf(1 - uniform_variate, np.maximum(mean_nsub, 0)).astype('i4')

    log10_mhost_array = np.log10(mhost_array)

    num_subs = int(np.sum(mc_nsub))
    mc_subhalo_mpeak = np.zeros(num_subs, dtype='f8')

    for i in range(len(log10_mhost_bin_edges)-1):
        log10_mhost_low, log10_mhost_high = log10_mhost_bin_edges[i], log10_mhost_bin_edges[i+1]
        mhost_mid = 10**(0.5*(log10_mhost_low + log10_mhost_high))
        host_halo_mask = (log10_mhost_array >= log10_mhost_low) & (log10_mhost_array < log10_mhost_high)
        num_hosts_ibin = np.count_nonzero(host_halo_mask)
        num_subs_ibin = int(np.sum(mc_nsub[host_halo_mask]))
        msg = ("\nMust have at least one subhalo per bin:\n"
            "ibin = {0}, log10_mhost_low = {1:.2f}, log10_mhost_high = {2:.2f}\n"
            "num_hosts_ibin = {3}, num_subs_ibin = {4}\n"
            "Modify your mass bins or use a different host/subhalo catalog")
        assert num_subs_ibin > 0, msg.format(i, log10_mhost_low, log10_mhost_high,
            num_hosts_ibin, num_subs_ibin)
        subhalo_masses_ibin = monte_carlo_subhalo_mass(mhost_mid, log10_msub_min,
            num_subs_ibin, **kwargs)
        subhalo_mask = np.repeat(host_halo_mask, mc_nsub)
        mc_subhalo_mpeak[subhalo_mask] = subhalo_masses_ibin*np.repeat(
            mhost_array[host_halo_mask], mc_nsub[host_halo_mask])

    return mc_nsub, mc_subhalo_mpeak


def read_best_fit_params(fname):
    d = OrderedDict()
    with open(fname, 'r') as f:
        for raw_line in f:
            line = raw_line.strip().split(',')
            d[line[0]] = float(line[1])
    return d


best_fit_param_dict = read_best_fit_params('best_fit_param_dict.txt')


def _check_bins(x, bins):
    assert np.all(np.diff(bins) > 0), "bins must be monotonically increasing"
    assert np.all(np.diff(x) >= 0), "subhalos must be sorted"
    assert np.min(x) > bins[0], "All mhost values must be greater than the smallest bin edge"
    assert np.max(x) < bins[-1], "All mhost values must be less than the largest bin edge"
