#!/usr/bin/env python
"""
Compute the DSigma profiles for different lenses
"""

import os
import pickle

from astropy.table import Table

from jianbing import scatter
from jianbing import wlensing

TOPN_DIR = '/tigress/sh19/work/topn/'

# Lensing data using medium photo-z quality cut
s16a_lensing = os.path.join(TOPN_DIR, 'prepare', 's16a_weak_lensing_medium.hdf5')

# Random
s16a_rand = Table.read(s16a_lensing, path='random')

# Pre-compute results using medium photo-z quality cut
s16a_precompute_med = os.path.join(
    TOPN_DIR, 'precompute', 'topn_public_s16a_medium_precompute.hdf5')

# Pre-compute results for each individual samples
# HSC massive galaxies
hsc = Table.read(s16a_precompute_med, path='hsc_extra')

# TopN bins
topn_bins = Table.read(
    os.path.join(TOPN_DIR, 'precompute', 'topn_bins.fits'))

# Tablulated simulation results
sim_cat = Table.read(
    os.path.join(TOPN_DIR, 'precompute', 'sim_merge_all_dsig.fits'))

# HSC properties to use
# Stellar or halo mass measurements for HSC galaxies
hsc_mass = [
    'logm_cmod', 'logm_5', 'logm_10', 'logm_15', 'logm_25',
    'logm_30', 'logm_40', 'logm_50', 'logm_60',
    'logm_75', 'logm_100', 'logm_120', 'logm_150', 'logm_max',
    'logmh_vir_forest', 'logmh_vir_plane', 'logmh_vir_symbol',
    'logm_extra_120', 'logm_extra_150', 'logm_extra_200', 'logm_extra_300',
    'logm_r50', 'logm_r50_half', 'logm_2_r50', 'logm_3_r50',
    'logm_4_r50', 'logm_5_r50', 'logm_6_r50',
    'logm_10_100', 'logm_30_100', 'logm_40_100', 'logm_50_100', 'logm_60_100',
    'logm_50_150', 'logm_60_150', 'logm_75_150', 'logm_40_120', 'logm_50_120',
    'logm_60_120', 'logm_75_120',
    'logm_50_120_extra', 'logm_50_150_extra', 'logm_50_200_extra', 'logm_50_300_extra',
    'logm_2_4_r50', 'logm_2_6_r50', 'logm_3_4_r50', 'logm_3_5_r50',
    'logm_3_6_r50', 'logm_4_6_r50'
]

# Size measurements for HSC galaxies
hsc_size = ['r50_100', 'r80_100', 'r90_100', 'r50_120', 'r80_120', 'r90_120',
            'r50_max', 'r80_max', 'r90_max', 'logr_vir_forest']

# S18A bright star mask
bsm_s18a = hsc['flag'] > 0

# General mask for HSC galaxies
mask = (
    (hsc['c82_100'] <= 18.) & (hsc['logm_100'] - hsc['logm_50'] <= 0.2) &
    bsm_s18a
)

# General mask for HSC size measurements
size_mask = (
    mask & (hsc['logm_max'] >= 11.3) & (hsc['r80_100'] <= 60.0) & (hsc['r90_100'] <= 60.0)
)

# Mask to select "central" galaxies
cen_mask_1 = hsc['cen_mask_1'] > 0
cen_mask_2 = hsc['cen_mask_2'] > 0
cen_mask_3 = hsc['cen_mask_3'] > 0

n_rand = 200000
n_boot = 1000
n_jobs = 8

topn_galaxies = {}
topn_galaxies_sum = {}

# Stellar mass related
for col in hsc_mass:

    # Default test with both jackknife and bootstrap error
    topn_galaxies[col] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=mask, n_rand=n_rand,
        n_boot=n_boot, verbose=True, n_jobs=n_jobs)

    topn_galaxies_sum[col] = scatter.compare_model_dsigma(
        topn_galaxies[col], sim_cat, model_err=False, poly=True, verbose=True)

    # The whole sample, without applying any mask; no bootstrap error
    topn_galaxies[col + '_all'] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=None, n_rand=n_rand,
        verbose=False, n_jobs=n_jobs, n_boot=200)

    topn_galaxies_sum[col + '_all'] = scatter.compare_model_dsigma(
        topn_galaxies[col + '_all'], sim_cat, model_err=False, poly=True, verbose=False)

    # Applying central mask 1; no bootstrap error
    topn_galaxies[col + '_cen1'] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=(mask & cen_mask_1), n_rand=n_rand,
        verbose=False, n_jobs=n_jobs, n_boot=200)

    topn_galaxies_sum[col + '_cen1'] = scatter.compare_model_dsigma(
        topn_galaxies[col + '_cen1'], sim_cat, model_err=False, poly=True, verbose=False)

    # Applying central mask 2; no bootstrap error
    topn_galaxies[col + '_cen2'] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=(mask & cen_mask_2), n_rand=n_rand,
        verbose=False, n_jobs=n_jobs, n_boot=200)

    topn_galaxies_sum[col + '_cen2'] = scatter.compare_model_dsigma(
        topn_galaxies[col + '_cen2'], sim_cat, model_err=False, poly=True, verbose=False)

    # Applying central mask 3; no bootstrap error
    topn_galaxies[col + '_cen3'] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=(mask & cen_mask_3), n_rand=n_rand,
        verbose=False, n_jobs=n_jobs, n_boot=200)

    topn_galaxies_sum[col + '_cen3'] = scatter.compare_model_dsigma(
        topn_galaxies[col + '_cen3'], sim_cat, model_err=False, poly=True, verbose=False)

# Galaxy size related
for col in hsc_size:

    # Default test with both jackknife and bootstrap error
    topn_galaxies[col] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=(mask & size_mask), n_rand=n_rand,
        n_boot=n_boot, verbose=True, n_jobs=n_jobs)

    topn_galaxies_sum[col] = scatter.compare_model_dsigma(
        topn_galaxies[col], sim_cat, model_err=False, poly=True, verbose=False)

    # The whole sample, without applying any mask; no bootstrap error
    topn_galaxies[col + '_all'] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=None, n_rand=n_rand,
        verbose=False, n_jobs=n_jobs, n_boot=200)

    topn_galaxies_sum[col + '_all'] = scatter.compare_model_dsigma(
        topn_galaxies[col + '_all'], sim_cat, model_err=False, poly=True, verbose=False)

    # Applying central mask 1; no bootstrap error
    topn_galaxies[col + '_cen1'] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=(mask & size_mask & cen_mask_1), n_rand=n_rand,
        n_boot=n_boot, verbose=False, n_jobs=n_jobs)

    topn_galaxies_sum[col + '_cen1'] = scatter.compare_model_dsigma(
        topn_galaxies[col + '_cen1'], sim_cat, model_err=False, poly=True, verbose=False)

    # Applying central mask 2; no bootstrap error
    topn_galaxies[col + '_cen2'] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=(mask & size_mask & cen_mask_2), n_rand=n_rand,
        n_boot=n_boot, verbose=False)

    topn_galaxies_sum[col + '_cen2'] = scatter.compare_model_dsigma(
        topn_galaxies[col + '_cen2'], sim_cat, model_err=False, poly=True, verbose=False)

    # Applying central mask 3; no bootstrap error
    topn_galaxies[col + '_cen3'] = wlensing.gather_topn_dsigma_profiles(
        hsc, s16a_rand, topn_bins, col, mask=(mask & size_mask & cen_mask_3), n_rand=n_rand,
        n_boot=n_boot, verbose=False)

    topn_galaxies_sum[col + '_cen3'] = scatter.compare_model_dsigma(
        topn_galaxies[col + '_cen3'], sim_cat, model_err=False, poly=True, verbose=False)


pickle.dump(
    topn_galaxies, open(os.path.join(TOPN_DIR, 'topn_galaxies.pkl'), "wb"))

pickle.dump(
    topn_galaxies_sum, open(os.path.join(TOPN_DIR, 'topn_galaxies_sum.pkl'), "wb"))
