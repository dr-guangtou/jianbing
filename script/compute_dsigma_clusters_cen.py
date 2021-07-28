#!/usr/bin/env python
"""
Compute the DSigma profiles for different clusters using just the central galaxies.
"""

import os
import pickle

from astropy.table import Table

from jianbing import scatter
from jianbing import wlensing

TOPN_DIR = '/tigress/sh19/work/topn'

# Lensing data using medium photo-z quality cut
s16a_lensing = os.path.join(TOPN_DIR, 'prepare', 's16a_weak_lensing_medium.hdf5')

# Random
s16a_rand = Table.read(s16a_lensing, path='random')

# Pre-compute results using medium photo-z quality cut
s16a_precompute_med = os.path.join(
    TOPN_DIR, 'precompute', 'topn_public_s16a_medium_precompute.hdf5')

# S16A HSC redMaPPer catalog
redm_hsc = Table.read(s16a_precompute_med, path='redm_hsc_specz')
redm_hsc_photoz = Table.read(s16a_precompute_med, path='redm_hsc')

# SDSS DR8 redMaPPer catalog
redm_sdss = Table.read(s16a_precompute_med, path='redm_sdss_specz')

# S16A HSC CAMIRA catalog
cam_s16a = Table.read(s16a_precompute_med, path='cam_s16a_specz')
cam_s16a_photoz = Table.read(s16a_precompute_med, path='cam_s16a')

# TopN bins
topn_bins = Table.read(
    os.path.join(TOPN_DIR, 'precompute', 'topn_bins.fits'))

# Tablulated simulation results
sim_cat = Table.read(
    os.path.join(TOPN_DIR, 'precompute', 'sim_mdpl2_cen_dsig.fits'))

n_rand = 200000
n_boot = 1000
n_jobs = 8

topn_clusters = {}
topn_clusters_sum = {}

# CAMIRA clusters; n_mem; use spec-z
mask = cam_s16a['flag'] > 0

topn_clusters['cam_s16a_n_mem'] = wlensing.gather_topn_dsigma_profiles(
    cam_s16a, s16a_rand, topn_bins, 'n_mem', mask=mask, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['cam_s16a_n_mem'] = scatter.compare_model_dsigma(
    topn_clusters['cam_s16a_n_mem'], sim_cat, model_err=False, poly=True, verbose=True)

topn_clusters['cam_s16a_n_mem_all'] = wlensing.gather_topn_dsigma_profiles(
    cam_s16a, s16a_rand, topn_bins, 'n_mem', mask=None, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['cam_s16a_n_mem_all'] = scatter.compare_model_dsigma(
    topn_clusters['cam_s16a_n_mem_all'], sim_cat, model_err=False, poly=True, verbose=True)

# CAMIRA clusters; logms; use spec-z
topn_clusters['cam_s16a_logms'] = wlensing.gather_topn_dsigma_profiles(
    cam_s16a, s16a_rand, topn_bins, 'logms', mask=mask, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['cam_s16a_logms'] = scatter.compare_model_dsigma(
    topn_clusters['cam_s16a_logms'], sim_cat, model_err=False, poly=True, verbose=True)

# CAMIRA clusters; n_mem; use spec-z
mask = cam_s16a_photoz['flag'] > 0

topn_clusters['cam_s16a_photoz_n_mem'] = wlensing.gather_topn_dsigma_profiles(
    cam_s16a_photoz, s16a_rand, topn_bins, 'n_mem', mask=mask, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['cam_s16a_photoz_n_mem'] = scatter.compare_model_dsigma(
    topn_clusters['cam_s16a_photoz_n_mem'], sim_cat, model_err=False, poly=True, verbose=True)

topn_clusters['cam_s16a_photoz_n_mem_all'] = wlensing.gather_topn_dsigma_profiles(
    cam_s16a_photoz, s16a_rand, topn_bins, 'n_mem', mask=None, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['cam_s16a_photoz_n_mem_all'] = scatter.compare_model_dsigma(
    topn_clusters['cam_s16a_photoz_n_mem_all'], sim_cat, model_err=False, poly=True, verbose=True)

# CAMIRA clusters; logms; use spec-z
topn_clusters['cam_s16a_photoz_logms'] = wlensing.gather_topn_dsigma_profiles(
    cam_s16a_photoz, s16a_rand, topn_bins, 'logms', mask=mask, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['cam_s16a_photoz_logms'] = scatter.compare_model_dsigma(
    topn_clusters['cam_s16a_photoz_logms'], sim_cat, model_err=False, poly=True, verbose=True)

# SDSS redMaPPer clusters; lambda; using spec-z
mask = redm_sdss['flag'] > 0

topn_clusters['redm_sdss_lambda'] = wlensing.gather_topn_dsigma_profiles(
    redm_sdss, s16a_rand, topn_bins[0:2], 'lambda_cluster_redm', mask=mask, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['redm_sdss_lambda'] = scatter.compare_model_dsigma(
    topn_clusters['redm_sdss_lambda'], sim_cat, model_err=False, poly=True, verbose=True)

topn_clusters['redm_sdss_lambda_all'] = wlensing.gather_topn_dsigma_profiles(
    redm_sdss, s16a_rand, topn_bins[0:2], 'lambda_cluster_redm', mask=None, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['redm_sdss_lambda_all'] = scatter.compare_model_dsigma(
    topn_clusters['redm_sdss_lambda_all'], sim_cat, model_err=False, poly=True, verbose=True)

# HSC redMaPPer clusters; lambda; using spec-z
mask = redm_hsc['flag'] > 0

topn_clusters['redm_hsc_lambda'] = wlensing.gather_topn_dsigma_profiles(
    redm_hsc, s16a_rand, topn_bins, 'lambda', mask=mask, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['redm_hsc_lambda'] = scatter.compare_model_dsigma(
    topn_clusters['redm_hsc_lambda'], sim_cat, model_err=False, poly=True, verbose=True)

topn_clusters['redm_hsc_lambda_all'] = wlensing.gather_topn_dsigma_profiles(
    redm_hsc, s16a_rand, topn_bins, 'lambda', mask=None, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['redm_hsc_lambda_all'] = scatter.compare_model_dsigma(
    topn_clusters['redm_hsc_lambda_all'], sim_cat, model_err=False, poly=True, verbose=True)

# HSC redMaPPer clusters; lambda; using spec-z
mask = redm_hsc_photoz['flag'] > 0

topn_clusters['redm_hsc_photoz_lambda'] = wlensing.gather_topn_dsigma_profiles(
    redm_hsc_photoz, s16a_rand, topn_bins, 'lambda', mask=mask, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['redm_hsc_photoz_lambda'] = scatter.compare_model_dsigma(
    topn_clusters['redm_hsc_photoz_lambda'], sim_cat, model_err=False, poly=True, verbose=True)

topn_clusters['redm_hsc_photoz_lambda_all'] = wlensing.gather_topn_dsigma_profiles(
    redm_hsc_photoz, s16a_rand, topn_bins, 'lambda', mask=None, n_rand=n_rand, n_boot=n_boot, 
    verbose=True, n_jobs=n_jobs)

topn_clusters_sum['redm_hsc_photoz_lambda_all'] = scatter.compare_model_dsigma(
    topn_clusters['redm_hsc_photoz_lambda_all'], sim_cat, model_err=False, poly=True, verbose=True)

pickle.dump(
    topn_clusters, open(os.path.join(TOPN_DIR, 'topn_clusters_cen.pkl'), "wb"))

pickle.dump(
    topn_clusters_sum, open(os.path.join(TOPN_DIR, 'topn_clusters_cen_sum.pkl'), "wb"))