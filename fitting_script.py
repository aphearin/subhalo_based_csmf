"""
"""
import numpy as np
from itertools import product
from jiang_usmf import jiang14_param_dict
from copy import deepcopy
from time import time
from refit_jiang_usmf import chi2_subhalo_counts
import csv


beta_min, beta_max, num_beta = 4.5, 7.5, 5
beta_grid_default = np.linspace(beta_min, beta_max, num_beta)

alpha1_min, alpha1_max, num_alpha1 = -0.8, -1, 5
alpha1_grid_default = np.linspace(alpha1_min, alpha1_max, num_alpha1)

gamma2_min, gamma2_max, num_gamma2 = 0.1, 1.5, 5
gamma2_grid_default = np.linspace(gamma2_min, gamma2_max, num_gamma2)


def param_grid_search_generator(**param_iterators):
    param_names = list(param_iterators.keys())
    param_combination_generator = product(*list(param_iterators.values()))
    for param_combination in param_combination_generator:
        yield {param_names[i]: param_combination[i] for i in range(len(param_names))}


def refit_params(**param_iterators):

    total_grid_size = len(list(param_grid_search_generator(
        alpha1=alpha1_grid_default, beta=beta_grid_default, gamma2=gamma2_grid_default)))
    print("Total grid size = {0}".format(total_grid_size))

    gen = param_grid_search_generator(
        alpha1=alpha1_grid_default, beta=beta_grid_default, gamma2=gamma2_grid_default)

    best_fit_param_dict = deepcopy(jiang14_param_dict)

    start = time()
    best_fit_chi2 = chi2_subhalo_counts(best_fit_param_dict.values())
    end = time()
    print("Projected runtime = {0} minutes".format((end-start)*total_grid_size/60.))

    best_fit_param_dict['chi2'] = best_fit_chi2
    with open('best_fit_param_dict.txt', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in best_fit_param_dict.items():
           writer.writerow([key, value])

    start = time()
    for param_dict in gen:
        beta = param_dict.get('beta', jiang14_param_dict['beta'])
        zeta = param_dict.get('zeta', jiang14_param_dict['zeta'])
        gamma1 = param_dict.get('gamma1', jiang14_param_dict['gamma1'])
        alpha1 = param_dict.get('alpha1', jiang14_param_dict['alpha1'])
        gamma2 = param_dict.get('gamma2', jiang14_param_dict['gamma2'])
        alpha2 = param_dict.get('alpha2', jiang14_param_dict['alpha2'])
        params = (beta, zeta, gamma1, alpha1, gamma2, alpha2)
        chi2 = chi2_subhalo_counts(params)
        if chi2 < best_fit_chi2:
            best_fit_chi2 = chi2
            best_fit_param_dict.update(param_dict)
            best_fit_param_dict['chi2'] = best_fit_chi2
            with open('best_fit_param_dict.txt', 'wb') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in best_fit_param_dict.items():
                   writer.writerow([key, value])




