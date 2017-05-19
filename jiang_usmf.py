""" Module storing kernel functions for generating Monte Carlo realizations of
the Jiang & van den Bosch 2014 fitting function for the unevolved subhalo mass function
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import poisson


__all__ = ('unevolved_subhalo_mass_function', )


beta = 5.67
zeta = 1.19
gamma1 = 0.13
alpha1 = -0.83
gamma2 = 1.33
alpha2 = -0.02


def unevolved_subhalo_mass_function(log10_mu, beta=beta, zeta=zeta,
        gamma1=gamma1, alpha1=alpha1, gamma2=gamma2, alpha2=alpha2):
    """ Average number of subhalos of mass mu = Msub/Mhost
    """
    mu = 10.**log10_mu
    prefactor_term1 = gamma1*mu**alpha1
    prefactor_term2 = gamma2*mu**alpha2
    prefactor = prefactor_term1 + prefactor_term2

    exparg = beta*mu**zeta

    return prefactor*np.exp(-exparg)


def mean_nsub_mu_range(log10_mu_min, log10_mu_max, beta=beta, zeta=zeta,
        gamma1=gamma1, alpha1=alpha1, gamma2=gamma2, alpha2=alpha2):
    """
    """
    csmf_mu = quad(unevolved_subhalo_mass_function, log10_mu_min, log10_mu_max)[0]
    return csmf_mu


def mean_nsub_vs_mhost(mhost_array, log10_msub_min, log10_mu_max=0., log10_mhost_max=15.5,
        beta=beta, zeta=zeta, gamma1=gamma1, alpha1=alpha1, gamma2=gamma2, alpha2=alpha2):
    """
    """
    npts_mhost_grid = 100
    log10_mhost_min = log10_msub_min
    log10_mhost_grid = np.linspace(log10_mhost_min, log10_mhost_max, npts_mhost_grid)
    log10_mu_min_grid = log10_msub_min - log10_mhost_grid
    mean_nsub_grid = list(mean_nsub_mu_range(log10_mu_min, log10_mu_max)
        for log10_mu_min in log10_mu_min_grid)
    return np.interp(np.log10(mhost_array), log10_mhost_grid, mean_nsub_grid)


def monte_carlo_subhalo_mass(mhost, log10_msub_min, counts):
    """ For an input host halo mass, generate a Monte Carlo realization
    of the Mpeak values of a subhalo population according to the unevolved mass function
    given by Eq. (21) of Jiang & van den Bosch (2014), arXiv:1311.5225.

    Parameters
    ----------
    mhost : float
        Mass of the host halo in Msun/h

    counts : int
        Number of satellites in the halo

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
    num_subs_above_mu_min = mean_nsub_mu_range(log10_mu_min, log10_mu_max)
    num_subs_grid = np.array(list(mean_nsub_mu_range(log10_mu, log10_mu_max)
        for log10_mu in log10_mu_grid))
    mu_cdf_grid = num_subs_grid/num_subs_above_mu_min

    uran = np.random.rand(counts)
    mc_log10_mu = np.interp(uran, mu_cdf_grid, log10_mu_grid)
    return 10**mc_log10_mu


def monte_carlo_subhalo_population(mhost_array, log10_msub_min, log10_mhost_bins):
    """
    """
    mean_nsub = mean_nsub_vs_mhost(mhost_array, log10_msub_min)
    mc_nsub = poisson.rvs(mean_nsub)

    log10_mhost_array = np.log10(mhost_array)
    result = np.zeros(np.sum(mc_nsub))

    for i in range(len(log10_mhost_bins)-1):
        log10_mhost_low, log10_mhost_high = log10_mhost_bins[i], log10_mhost_bins[i+1]
        mhost_mid = 10**(0.5*(log10_mhost_low + log10_mhost_high))
        host_halo_mask = (log10_mhost_array >= log10_mhost_low) & (log10_mhost_array < log10_mhost_high)
        subhalo_masses_ibin = monte_carlo_subhalo_mass(mhost_mid, log10_msub_min,
            np.sum(mc_nsub[host_halo_mask]))
        subhalo_mask = np.repeat(host_halo_mask, mc_nsub)
        result[subhalo_mask] = subhalo_masses_ibin*np.repeat(mhost_array[host_halo_mask],
            mc_nsub[host_halo_mask])

    return result
