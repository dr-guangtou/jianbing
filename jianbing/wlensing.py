#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to deal with DeltaSigma profile calculation."""

import copy
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

from astropy.table import Table

from dsigma.stacking import excess_surface_density
from dsigma.jackknife import add_continous_fields
from dsigma.jackknife import jackknife_field_centers, add_jackknife_fields


__all__ = ['dsigma_no_wsys', 'dsigma_jk_resample', 'dsigma_bootstrap', 'dsigma_with_mask',
           'stack_dsigma_profile', 'gather_topn_dsigma_profiles']


def dsigma_bootstrap(lens, rand, n_boot=100, n_jobs=None):
    """Bootstrap resampling for the DeltaSigma profiles."""
    index_l = [np.random.choice(
        np.arange(len(lens)), size=len(lens), replace=True) for ii in range(n_boot)]
    index_r = [np.random.choice(
        np.arange(len(rand)), size=len(rand), replace=True) for ii in range(n_boot)]

    def dsigma_bt_single(i):
        return dsigma_with_mask(lens, rand, index_l[i], index_r[i])

    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    samples = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(dsigma_bt_single)(i) for i in range(n_boot))

    return (n_boot / (n_boot - 1)) * np.cov(np.array(samples), rowvar=False, ddof=0)


def dsigma_jk_resample(lens, rand, n_jobs=None):
    """Jackknife resampling for DeltaSigma profiles."""
    jks = np.unique(lens['field_jk'])

    def dsigma_jk_single(f):
        return dsigma_no_wsys(lens[lens['field_jk'] != f], rand[rand['field_jk'] != f])

    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()

    samples = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(dsigma_jk_single)(f) for f in jks)

    return (len(jks) - 1) * np.cov(np.array(samples), rowvar=False, ddof=0)


def dsigma_no_wsys(lens, rand):
    """Compute the total lensing signal, including all corrections from
    precompute results.
    Parameters
    ----------
    lens : astropy.table.Table
        Precompute results for the lenses.
    rand : astropy.table.Table, optional
        Precompute results for random lenses.
    """
    # Denominator that combines the system weight and the lens-source weight
    w_den_l = lens['sum w_ls'].sum(axis=0)
    w_den_r = rand['sum w_ls'].sum(axis=0)

    # This is the raw DeltaSigma profile
    dsig_l = lens['sum w_ls e_t sigma_crit'].sum(axis=0) / w_den_l
    dsig_r = rand['sum w_ls e_t sigma_crit'].sum(axis=0) / w_den_r

    # Multiplicative shear bias
    m_factor_l = lens['sum w_ls m'].sum(axis=0) / w_den_l
    m_factor_r = rand['sum w_ls m'].sum(axis=0) / w_den_r

    # Shear responsitivity factor
    r_factor_l = lens['sum w_ls (1 - e_rms^2)'].sum(axis=0) / w_den_l
    r_factor_r = rand['sum w_ls (1 - e_rms^2)'].sum(axis=0) / w_den_r

    # Multiplicative selection bias
    m_sel_l = lens['sum w_ls A p(R_2=0.3)'].sum(axis=0) / w_den_l
    m_sel_r = rand['sum w_ls A p(R_2=0.3)'].sum(axis=0) / w_den_r

    # Photometric redshift bias
    f_bias_l = (
        lens['sum w_ls e_t sigma_crit f_bias'].sum(axis=0) /
        lens['sum w_ls e_t sigma_crit'].sum(axis=0))
    f_bias_r = (
        rand['sum w_ls e_t sigma_crit f_bias'].sum(axis=0) /
        rand['sum w_ls e_t sigma_crit'].sum(axis=0))

    dsig_l *= ((f_bias_l * (1. + m_sel_l)) / ((1. + m_factor_l) * (2. * r_factor_l)))
    dsig_r *= ((f_bias_r * (1. + m_sel_r)) / ((1. + m_factor_r) * (2. * r_factor_r)))

    return dsig_l - dsig_r


def dsigma_with_mask(lens, rand, mask_l, mask_r):
    """Compute the total lensing signal, including all corrections from
    precompute results.
    Parameters
    ----------
    lens : astropy.table.Table
        Precompute results for the lenses.
    rand : astropy.table.Table, optional
        Precompute results for random lenses.
    """
    # Denominator that combines the system weight and the lens-source weight
    w_den_l = lens['sum w_ls'][mask_l].sum(axis=0)
    w_den_r = rand['sum w_ls'][mask_r].sum(axis=0)

    # This is the raw DeltaSigma profile
    dsig_l = lens['sum w_ls e_t sigma_crit'][mask_l].sum(axis=0) / w_den_l
    dsig_r = rand['sum w_ls e_t sigma_crit'][mask_r].sum(axis=0) / w_den_r

    # Multiplicative shear bias
    m_factor_l = lens['sum w_ls m'][mask_l].sum(axis=0) / w_den_l
    m_factor_r = rand['sum w_ls m'][mask_r].sum(axis=0) / w_den_r

    # Shear responsitivity factor
    r_factor_l = lens['sum w_ls (1 - e_rms^2)'][mask_l].sum(axis=0) / w_den_l
    r_factor_r = rand['sum w_ls (1 - e_rms^2)'][mask_r].sum(axis=0) / w_den_r

    # Multiplicative selection bias
    m_sel_l = lens['sum w_ls A p(R_2=0.3)'][mask_l].sum(axis=0) / w_den_l
    m_sel_r = rand['sum w_ls A p(R_2=0.3)'][mask_r].sum(axis=0) / w_den_r

    # Photometric redshift bias
    f_bias_l = (
        lens['sum w_ls e_t sigma_crit f_bias'][mask_l].sum(axis=0) /
        lens['sum w_ls e_t sigma_crit'][mask_l].sum(axis=0))
    f_bias_r = (
        rand['sum w_ls e_t sigma_crit f_bias'][mask_r].sum(axis=0) /
        rand['sum w_ls e_t sigma_crit'][mask_r].sum(axis=0))

    dsig_l *= ((f_bias_l * (1. + m_sel_l)) / ((1. + m_factor_l) * (2. * r_factor_l)))
    dsig_r *= ((f_bias_r * (1. + m_sel_r)) / ((1. + m_factor_r) * (2. * r_factor_r)))

    return dsig_l - dsig_r


def stack_dsigma_profile(lens, rand, mask=None, n_rand=None, use_dsigma=False,
                         bootstrap=False, n_boot=500, jackknife=True, n_jobs=None, n_jk=45):
    """Get the DeltaSigma profile of a sample of lens."""
    # Check to see the setup for lens and random
    assert np.all(lens.meta['bins'] == rand.meta['bins'])
    assert lens.meta['H0'] == rand.meta['H0']
    assert lens.meta['Om0'] == rand.meta['Om0']
    assert (lens['n_s_tot'] > 0).sum() == len(lens)
    assert (rand['n_s_tot'] > 0).sum() == len(rand)

    # Apply the mask
    lens_use = lens if mask is None else lens[mask]

    # Randomly downsample the random objects if necessary
    if n_rand is not None and n_rand < len(rand):
        rand_use = Table(np.random.choice(rand, size=n_rand, replace=False))
        rand_use.meta = rand.meta
    else:
        rand_use = rand

    # Get the stacked lensing profiles
    if use_dsigma:
        # Configurations for calculating HSC
        kwargs = {'return_table': True, 'shear_bias_correction': True,
                  'shear_responsivity_correction': True,
                  'selection_bias_correction': True,
                  'boost_correction': False, 'random_subtraction': True,
                  'photo_z_dilution_correction': True,
                  'rotation': False, 'table_r': rand_use}

        result = excess_surface_density(lens_use, **kwargs)
    else:
        result = Table()
        result['ds'] = dsigma_no_wsys(lens_use, rand_use)

    if jackknife:
        if n_jk <= 5:
            raise Exception("Number of jackknife fields is too small, should >5")

        if len(lens_use) <= 5:
            print("Number of lenses < 5, cannot use Jackknife resampling")
            jackknife = False
        else:
            # Deal with situations with small sample
            if len(lens_use) <= n_jk - 5:
                n_jk = len(lens) - 5

        # Add consistent Jackknife fields to both the lens and random catalogs
        add_continous_fields(lens_use, distance_threshold=2)
        centers = jackknife_field_centers(lens_use, n_jk, weight='n_s_tot')
        add_jackknife_fields(lens_use, centers)
        add_jackknife_fields(rand_use, centers)

        # Estimate the covariance matrix using Jackknife resampling
        cov_jk = dsigma_jk_resample(lens_use, rand_use, n_jobs=n_jobs)

        result['ds_err_jk'] = np.sqrt(np.diag(cov_jk))
        result.meta['cov_jk'] = cov_jk
        result.meta['s2n_jk'] = np.sqrt(
            np.dot(result['ds'].T.dot(np.linalg.inv(cov_jk)), result['ds']))

    # Estimate the covariance matrix using Bootstrap resampling
    if bootstrap:
        cov_bt = dsigma_bootstrap(lens_use, rand_use, n_boot=n_boot, n_jobs=n_jobs)

        result['ds_err_bt'] = np.sqrt(np.diag(cov_bt))
        result.meta['cov_bt'] = cov_bt
        result.meta['s2n_bt'] = np.sqrt(
            np.dot(result['ds'].T.dot(np.linalg.inv(cov_bt)), result['ds']))

    return result


def gather_topn_dsigma_profiles(lens, rand, topn, col, n_rand=None, n_boot=100,
                                n_jk=45, mask=None, verbose=True, n_jobs=None,
                                vol_factor=None):
    """
    For a sample of lens, gather all the DeltaSigma profiles
    for a series of bins based on a property.
    """
    # Downsample the random objects if necessary
    if n_rand is not None and n_rand <= len(rand):
        rand_use = Table(np.random.choice(rand, size=n_rand, replace=False))
        rand_use.meta = rand.meta
    else:
        rand_use = rand

    # Masking the lens and check for infinite values
    if mask is not None:
        lens = lens[mask]

    # Remove the infinite or NaN values
    flag = np.isfinite(lens[col])
    lens = lens[flag]
    if verbose and (~flag).sum() > 0:
        print("! There are {:d} infinite values for {:s}".format((~flag).sum(), col))

    # Sort the lens catalog in descending order based on the chosen property
    if verbose:
        print("\n# Using column: {:s}".format(col))
    lens.sort(col)
    lens.reverse()

    dsig, dsig_err_jk, dsig_err_bt = [], [], []
    dsig_cov_jk, dsig_cov_bt = [], []
    dsig_s2n_jk, dsig_s2n_bt = [], []
    samples = []

    for ii, topn_row in enumerate(topn):
        index_low, index_upp = topn_row['index_low'], topn_row['index_upp']

        if vol_factor is None and ii >= 2:
            vol_factor = 0.90

        if vol_factor is not None:
            index_low = int(index_low * vol_factor)
            index_upp = int(index_upp * vol_factor)

        if index_low >= len(lens):
            print("! No useful object in this bin !")
            dsig.append(None)
            dsig_err_jk.append(None)
            dsig_err_bt.append(None)
            dsig_cov_jk.append(None)
            dsig_cov_bt.append(None)
            dsig_s2n_jk.append(np.nan)
            dsig_s2n_bt.append(np.nan)
            samples.append(None)
        else:
            if index_upp >= len(lens):
                index_upp = len(lens) - 1
                print("! Not enough objects in the lens catalog")

            if verbose:
                print("# Bin {:d}: {:5d} - {:5d}".format(
                    topn_row['bin_id'], index_low, index_upp))

            lens_use = lens[index_low: index_upp]

            result = stack_dsigma_profile(
                lens_use, rand_use, mask=None, use_dsigma=False, bootstrap=True,
                jackknife=True, n_boot=n_boot, n_jk=n_jk, n_jobs=n_jobs)

            dsig.append(result['ds'])
            dsig_err_jk.append(result['ds_err_jk'])
            dsig_err_bt.append(result['ds_err_bt'])
            dsig_cov_jk.append(result.meta['cov_jk'])
            dsig_cov_bt.append(result.meta['cov_bt'])
            dsig_s2n_jk.append(result.meta['s2n_jk'])
            dsig_s2n_bt.append(result.meta['s2n_bt'])
            samples.append(np.asarray(lens_use[col]))

    # Organize the result into a Table
    dsigma_topn = Table(np.asarray(copy.deepcopy(topn)))
    dsigma_topn['dsigma'] = dsig
    dsigma_topn['dsig_err_jk'] = dsig_err_jk
    dsigma_topn['dsig_err_bt'] = dsig_err_bt
    dsigma_topn['dsig_cov_jk'] = dsig_cov_jk
    dsigma_topn['dsig_cov_bt'] = dsig_cov_bt
    dsigma_topn['dsig_s2n_jk'] = dsig_s2n_jk
    dsigma_topn['dsig_s2n_bt'] = dsig_s2n_bt
    dsigma_topn['samples'] = samples

    # Organize the metadata
    dsigma_topn.meta['H0'] = rand.meta['H0']
    dsigma_topn.meta['Om0'] = rand.meta['Om0']
    dsigma_topn.meta['Ok0'] = rand.meta['Ok0']
    dsigma_topn.meta['comoving'] = rand.meta['comoving']
    dsigma_topn.meta['bins'] = rand.meta['bins']
    dsigma_topn.meta['r_min'] = rand.meta['bins'][:-1]
    dsigma_topn.meta['r_max'] = rand.meta['bins'][1:]
    dsigma_topn.meta['r_mpc'] = np.sqrt(
        dsigma_topn.meta['r_min'] * dsigma_topn.meta['r_max'])

    return dsigma_topn


def batch_dsigma_profiles(lens, rand, mask_list, n_rand=None, n_boot=100,
                          n_jk=45, verbose=True, n_jobs=None, bootstrap=True):
    """
    For a sample of lens, gather DeltaSigma profiles based on the mask list.
    """
    # Downsample the random objects if necessary
    if n_rand is not None and n_rand <= len(rand):
        rand_use = Table(np.random.choice(rand, size=n_rand, replace=False))
        rand_use.meta = rand.meta
    else:
        rand_use = rand

    dsig, dsig_err_jk, dsig_err_bt = [], [], []
    dsig_cov_jk, dsig_cov_bt = [], []
    dsig_s2n_jk, dsig_s2n_bt = [], []

    for ii, mask in enumerate(mask_list):
        if mask.sum() == 0:
            print("! No useful object in this bin !")
            dsig.append(None)
            dsig_err_jk.append(None)
            dsig_err_bt.append(None)
            dsig_cov_jk.append(None)
            dsig_cov_bt.append(None)
            dsig_s2n_jk.append(np.nan)
            dsig_s2n_bt.append(np.nan)
        else:
            if verbose:
                print("There are {:d} objects in sample {:d}".format(mask.sum(), ii + 1))

            lens_use = lens[mask]

            result = stack_dsigma_profile(
                lens_use, rand_use, mask=None, use_dsigma=False, bootstrap=bootstrap,
                jackknife=True, n_boot=n_boot, n_jk=n_jk, n_jobs=n_jobs)

            dsig.append(result['ds'])
            dsig_err_jk.append(result['ds_err_jk'])
            dsig_cov_jk.append(result.meta['cov_jk'])
            dsig_s2n_jk.append(result.meta['s2n_jk'])
            if bootstrap:
                dsig_err_bt.append(result['ds_err_bt'])
                dsig_cov_bt.append(result.meta['cov_bt'])
                dsig_s2n_bt.append(result.meta['s2n_bt'])

    # Organize the result into a Table
    dsigma_sum = Table()
    dsigma_sum['dsigma'] = dsig
    dsigma_sum['dsig_err_jk'] = dsig_err_jk
    dsigma_sum['dsig_cov_jk'] = dsig_cov_jk
    dsigma_sum['dsig_s2n_jk'] = dsig_s2n_jk
    if bootstrap:
        dsigma_sum['dsig_err_bt'] = dsig_err_bt
        dsigma_sum['dsig_cov_bt'] = dsig_cov_bt
        dsigma_sum['dsig_s2n_bt'] = dsig_s2n_bt
    dsigma_sum['masks'] = mask_list

    # Organize the metadata
    dsigma_sum.meta['H0'] = rand.meta['H0']
    dsigma_sum.meta['Om0'] = rand.meta['Om0']
    dsigma_sum.meta['Ok0'] = rand.meta['Ok0']
    dsigma_sum.meta['comoving'] = rand.meta['comoving']
    dsigma_sum.meta['bins'] = rand.meta['bins']
    dsigma_sum.meta['r_min'] = rand.meta['bins'][:-1]
    dsigma_sum.meta['r_max'] = rand.meta['bins'][1:]
    dsigma_sum.meta['r_mpc'] = np.sqrt(
        dsigma_sum.meta['r_min'] * dsigma_sum.meta['r_max'])

    return dsigma_sum
