#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to deal with scatter of halo mass estimates."""

import numpy as np

from scipy import interpolate
from astropy.table import Table, join

from . import utils

__all__ = ['compare_model_dsigma', 'get_scatter_summary', 'get_chi2_curve',
           'get_dsig_chi2', 'sigm_to_sigo', 'sigo_to_sigm']


def compare_model_dsigma(obs, sim, model_err=False, poly=False, poly_order=6,
                         verbose=False):
    """
    Compare the observed profiles with the ones in simulation to estimate
    the scatter of halo mass.
    """
    # Radial bins for the observed profiles
    rad = obs.meta['r_mpc']
    summary = []

    # Go through all the bins in the data
    for obs_bin in obs:
        bin_id = obs_bin['bin_id']
        if verbose:
            print("# Dealing with Bin: {:d}".format(bin_id))

        # Get the predicted DeltaSigma profiles in this bin
        sim_use = sim[sim['bin'] == bin_id - 1]
        sim_use.sort('scatter')

        sigma = np.asarray(sim_use['scatter'])

        # Go through all the predicted DeltaSigma profiles
        sum_jk = get_scatter_summary(
            sigma, rad, obs_bin, sim_use, cov_type='jk', poly_order=poly_order,
            poly=poly, model_err=model_err)
        sum_bt = get_scatter_summary(
            sigma, rad, obs_bin, sim_use, cov_type='bt', poly_order=poly_order,
            poly=poly, model_err=model_err)

        # Combine the dictionaries
        sum_bin = {**sum_jk, **sum_bt}
        sum_bin['sigma'] = sigma
        sum_bin['bin_id'] = bin_id
        summary.append(sum_bin)

    return join(obs, Table(summary), keys='bin_id')


def get_scatter_summary(sigma, rad, obs_bin, sim_use, cov_type='jk',
                        poly_order=6, poly=False, model_err=False):
    """Get the summary statistics of the scatters."""
    # Get the best-fit scatter value using polynomial fitting
    summary = {}
    summary['chi2_' + cov_type] = get_chi2_curve(
        rad, obs_bin, sim_use, cov_type=cov_type, model_err=model_err)

    if poly:
        sig_arr = np.linspace(0.1, np.max(sigma) + 0.1, 5000)
        sig_ply = np.poly1d(
            np.polyfit(sigma[1:], (summary['chi2_' + cov_type] / (len(rad) - 1))[1:],
                       poly_order))(sig_arr)
        summary['sig_poly_' + cov_type] = sig_arr[np.argmin(sig_ply)]
        summary['idx_poly_' + cov_type] = np.argmin(
            np.abs(summary['sig_poly_' + cov_type] - sigma))

    # Get best-fit scatter using cumulative distribution
    try:
        likelihood = np.exp(-0.5 * summary['chi2_' + cov_type])
        cum_curve = np.cumsum(likelihood / np.nansum(likelihood))
        cum_inter = interpolate.interp1d(cum_curve, sigma, kind='slinear')
        summary['sig_low_' + cov_type] = cum_inter(0.16)[()]
        summary['sig_upp_' + cov_type] = cum_inter(0.84)[()]
        summary['sig_med_' + cov_type] = cum_inter(0.50)[()]
        summary['sig_err_' + cov_type] = (
            (summary['sig_upp_' + cov_type] - summary['sig_low_' + cov_type])) / 2.
        summary['idx_med_' + cov_type] = np.argmin(
            np.abs(summary['sig_med_' + cov_type] - sigma))
    except ValueError:
        print("! Something is wrong with the scatter !")
        if poly:
            summary['sig_err_' + cov_type] = 0.05
            summary['sig_med_' + cov_type] = summary['sig_poly_' + cov_type]
            summary['sig_low_' + cov_type] = summary['sig_poly_' + cov_type] - 0.05
            summary['sig_upp_' + cov_type] = summary['sig_poly_' + cov_type] + 0.05
            summary['idx_med_' + cov_type] = summary['idx_poly_' + cov_type]
        else:
            summary['sig_err_' + cov_type], summary['sig_med_' + cov_type] = np.nan, np.nan
            summary['sig_low_' + cov_type], summary['sig_upp_' + cov_type] = np.nan, np.nan
            summary['idx_med_' + cov_type] = np.nan

    # Get the "best-fit" model profile
    summary['simulation'] = sim_use['simulation']
    sim_best = sim_use[summary['idx_med_' + cov_type]]
    summary['sim_med_' + cov_type] = sim_best['simulation']
    summary['dsigma_mod_' + cov_type] = utils.interp_dsig(
        rad, sim_best['r_mpc'], sim_best['dsig'])
    summary['dsigma_mod_low_' + cov_type] = utils.interp_dsig(
        rad, sim_best['r_mpc'], sim_best['dsig'] - sim_best['dsig_err'])
    summary['dsigma_mod_upp_' + cov_type] = utils.interp_dsig(
        rad, sim_best['r_mpc'], sim_best['dsig'] + sim_best['dsig_err'])

    # Get the model profile for zero scatter
    sim_dsig0 = sim_use[(sim_use['scatter'] >= 0.) & (sim_use['scatter'] <= 0.1)][0]
    summary['dsigma_sig0_' + cov_type] = utils.interp_dsig(
        rad, sim_dsig0['r_mpc'], sim_dsig0['dsig'])
    summary['dsigma_sig0_low_' + cov_type] = utils.interp_dsig(
        rad, sim_dsig0['r_mpc'], sim_dsig0['dsig'] - sim_dsig0['dsig_err'])
    summary['dsigma_sig0_upp_' + cov_type] = utils.interp_dsig(
        rad, sim_dsig0['r_mpc'], sim_dsig0['dsig'] + sim_dsig0['dsig_err'])

    return summary

def get_chi2_curve(rad, obs_bin, sim_use, cov_type='jk', model_err=False):

    """Get the chi2 curves"""
    if cov_type.strip() == 'jk':
        cov_col = 'dsig_cov_jk'
    elif cov_type.strip() == 'bt':
        cov_col = 'dsig_cov_bt'
    else:
        raise ValueError("! Wrong type of covariance matrix: [jk|bt]")

    return np.array([
        get_dsig_chi2(rad,
                      obs_bin['dsigma'],
                      obs_bin[cov_col],
                      sim['r_mpc'],
                      sim['dsig'],
                      sim['dsig_err'],
                      include_model_err=model_err,
                      cov=True) for sim in sim_use
    ])


def get_dsig_chi2(r_obs, dsig_obs, err_obs, r_mod, dsig_mod, err_mod,
                  include_model_err=False, cov=False):
    """Compute a likelihood for the model DSigma profile."""
    # Interpolate the model DSigma profile to the observed radial bins
    dsig_inter = utils.interp_dsig(r_obs, r_mod, dsig_mod)

    # Interpolate the error
    err_inter = interpolate.interp1d(
        r_mod, err_mod, fill_value='extrapolate')(r_obs)

    dsig_diff = (dsig_inter - dsig_obs)
    dif_square = dsig_diff ** 2

    if not cov:
        # Differences between the DSigma profile
        if include_model_err:
            err_square = err_obs ** 2 + err_inter ** 2
        else:
            err_square = err_obs ** 2

        # "Chi2" value
        chi2 = np.sum(dif_square / err_square)
    else:
        # Treat the err_obs as covariance matrix
        if include_model_err:
            err_matrix = np.diag(err_inter ** 2) + err_obs
            cov_inv = np.linalg.inv(err_matrix)
        else:
            cov_inv = np.linalg.inv(err_obs)

        chi2 = np.dot(dsig_diff, np.dot(cov_inv, dsig_diff))

    return chi2

def sigo_to_sigm(sigo, alpha=1.0, beta2=2.959):
    """Convert the scatter of observable to the scatter of halo mass."""
    return sigo / np.sqrt(beta2 * (sigo ** 2) + (alpha ** 2))

def sigm_to_sigo(sigm, alpha=1.0, beta=2.959):
    """
    Convert the scatter of halo mass to scatter of observable based on 
    the slope of the halo mass-observable relation and the 
    high-mass end curvature of the HMF.
    """
    return (alpha * sigm) / np.sqrt(1. - beta * sigm ** 2)