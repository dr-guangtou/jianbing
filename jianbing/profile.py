#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions related to 1-D profile."""

import numpy as np

from scipy import interpolate
from scipy.optimize import brent, curve_fit

from mpmath import gamma as Gamma
from mpmath import gammainc as GammaInc

import matplotlib.pyplot as plt


__all__ = ['RSMA_COMMON', 'RKPC_COMMON', 'extrapolate_mass', 'get_extra_masses',
           'test_mass_extrapolation', 'Sersic', 'b_n_exact']


RSMA_COMMON = np.arange(0.4, 4.2, 0.01)
RKPC_COMMON = RSMA_COMMON ** 4.0


def Sersic(R, n, I_e, r_e):
    """Compute intensity at radius r for a Sersic profile, given the specified
    """
    return I_e * np.exp(-1 * b_n_exact(n) * (pow(R / r_e, 1.0 / n) - 1.0))


def b_n_exact(n):
    """Exact calculation of the Sersic derived parameter b_n, via solution
    of the function
            Gamma(2n) = 2 gamma_inc(2n, b_n)
    where Gamma = Gamma function and gamma_inc = lower incomplete gamma function.
    If n is a list or Numpy array, the return value is a 1-d Numpy array
    """
    if n > 0.05:
        def myfunc(bn, n):
            return abs(float(2 * GammaInc(2*n, 0, bn) - Gamma(2*n)))

        if np.iterable(n):
            b = [brent(myfunc, (nn,)) for nn in n]
            b = np.array(b)
        else:
            b = brent(myfunc, (n,))
        return b
    else:
        return 0.0


def extrapolate_mass(gal_test, rmin=30.0, rmax=90.0, q_outskirt=1.0, order=1,
                     verbose=False, use_r50=False):
    """Extrapolate the surface mass density profile.
    """
    # Area within each radius
    logr_extra = np.linspace(1.6, 2.9, 100)
    rkpc_extra = 10.0 ** logr_extra
    area_extra = np.pi * rkpc_extra * (rkpc_extra * q_outskirt)

    # Radius mask
    if use_r50:
        r50 = gal_test['r50_100']
        rmin, rmax = rmin * r50, rmax * r50
        rad_mask = (RKPC_COMMON >= rmin) & (RKPC_COMMON <= rmax)
    else:
        rad_mask = (RKPC_COMMON >= rmin) & (RKPC_COMMON <= rmax)

    # Mask for mass density profile and CoG
    sbp_mask = np.isfinite(gal_test['sbp']) & rad_mask & (gal_test['sbp'] > 0)
    cog_mask = np.isfinite(gal_test['cog']) & rad_mask

    # Fit Sersic profile to extrapolate the 1-D profile
    val_norm = np.nanmedian(10.0 ** gal_test['sbp'])
    sbp_norm = (10.0 ** gal_test['sbp']) / val_norm

    # Initial parameter guesses; lower and upper limits
    ini = [2.5, np.nanmedian(sbp_norm), gal_test['r50_100']]

    # Fit 1-Serisc profile
    try:
        best_sersic, _ = curve_fit(
            Sersic, RKPC_COMMON[sbp_mask], sbp_norm[sbp_mask], p0=ini, method='lm')
        #print(best_sersic)

        ser_extra = Sersic(
            rkpc_extra, best_sersic[0], best_sersic[1], best_sersic[2])
        sbp_extra = np.log10(ser_extra * val_norm)

        # Extra mass in the outskirt using mass density profiles
        mass_extra = np.cumsum(
            (10.0 ** sbp_extra)[1:] * (area_extra[1:] - area_extra[:-1]))
        logm_extra_use = np.log10(
            mass_extra[rkpc_extra[1:] >= 100.0] + 10.0 ** gal_test['logm_100'])
    except Exception:
        print("# Sersic fitting failed for %d" % gal_test['object_id'])
        sbp_extra, logm_extra_use = None, None

    # Extrapolate the curve of growth profile
    if cog_mask.sum() <= 5:
        if verbose:
            print("# CoG: Not enough data points for %d" % gal_test['object_id'])
        cog_extra = None
    else:
        # Fit log-log linear relation to extrapolate the curve of growth
        cog_extra = np.poly1d(
            np.polyfit(
                np.log10(RKPC_COMMON[cog_mask]), gal_test['cog'][cog_mask], order))(logr_extra)

    return rkpc_extra, sbp_extra, cog_extra, logm_extra_use, rad_mask


def get_extra_masses(gal_test, rmin=20, rmax=90, r_extra=300.0, q_outskirt=0.8,
                     verbose=False, use_r50=False, order=1):
    """
    Estimate extrapolated stellar mass using surface brightness profile and the
    curve of growth.
    """
    if rmin is None or rmax is None:
        if gal_test['logm_100'] >= 11.4:
            rmin, rmax = 20, 90
        else:
            rmin, rmax = 20, 90

    rkpc_extra, _, cog_extra, logm_extra_use, _ = extrapolate_mass(
        gal_test, rmin=rmin, rmax=rmax, q_outskirt=q_outskirt,
        use_r50=use_r50, verbose=verbose, order=order)

    rkpc_extra_use = rkpc_extra[rkpc_extra >= 100.0]

    if cog_extra is not None:
        logm_extra_cog = float(interpolate.interp1d(
            np.log10(rkpc_extra), cog_extra)(np.log10(r_extra)))
    else:
        logm_extra_cog = gal_test['logm_max']

    if logm_extra_use is not None:
        logm_extra_sbp = float(interpolate.interp1d(
            np.log10(rkpc_extra_use), logm_extra_use)(np.log10(r_extra)))
    else:
        logm_extra_sbp = gal_test['logm_max']

    return logm_extra_cog, logm_extra_sbp


def test_mass_extrapolation(gal_test, rmin=20, rmax=90, order=1,
                            use_r50=False):
    """
    Test the extrapolation and visulize the extrapolated results.
    """
    print("# Redshift : %5.2f" % gal_test['z_best'])
    print("# R50 : %5.2f" % gal_test['r50_100'])
    print("# logM100 : %5.2f" % gal_test['logm_100'])
    print("# logMmax : %5.2f" % gal_test['logm_max'])

    rkpc_extra, sbp_extra, cog_extra, _, rad_mask = extrapolate_mass(
        gal_test, rmin=rmin, rmax=rmax, q_outskirt=0.9, use_r50=use_r50, order=order,
        verbose=True)

    print("\n# Extrapolated mass to 300 kpc:")
    print(get_extra_masses(gal_test, rmin=rmin, rmax=rmax, r_extra=200.0,
                           order=order, use_r50=use_r50, verbose=True))
    print("# Extrapolated mass to 500 kpc:")
    print(get_extra_masses(gal_test, rmin=rmin, rmax=rmax, r_extra=300.0,
                           order=order, use_r50=use_r50, verbose=True))

    fig = plt.figure(figsize=(9, 6))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.99, wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)

    ax1.grid(linewidth=2.0, alpha=0.4, linestyle='--')

    ax1.plot(np.log10(RKPC_COMMON), gal_test['sbp'], linewidth=4, alpha=0.9)

    ax1.plot(np.log10(RKPC_COMMON[rad_mask]), gal_test['sbp'][rad_mask],
             linewidth=3.5, alpha=0.9)

    ax1.plot(np.log10(rkpc_extra), sbp_extra, linewidth=3.0, linestyle='--', alpha=0.8)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    _ = ax1.set_xlim(0.25, np.log10(499))

    _ = ax1.set_xlabel(r'$\log (R/[\mathrm{kpc}])$', size=30)
    _ = ax1.set_ylabel(r'$\log ({\mu}_{\star}/[M_{\odot}\ \mathrm{kpc}^{-2}])$', size=30)

    fig = plt.figure(figsize=(9, 6))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.99, wspace=0.00, hspace=0.00)
    ax1 = fig.add_subplot(111)

    ax1.grid(linewidth=2.0, alpha=0.4, linestyle='--')

    ax1.plot(np.log10(RKPC_COMMON), gal_test['cog'], linewidth=4, alpha=0.9)

    ax1.plot(np.log10(RKPC_COMMON[rad_mask]), gal_test['cog'][rad_mask],
             linewidth=4, alpha=0.9)

    ax1.plot(np.log10(rkpc_extra), cog_extra, linewidth=3.0, linestyle='--', alpha=0.8)

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)

    _ = ax1.set_xlim(0.25, np.log10(499.0))
    _ = ax1.set_ylim(10.5, np.nanmax(gal_test['cog']) + 0.19)

    _ = ax1.set_xlabel(r'$\log (R/[\mathrm{kpc}])$', size=30)
    _ = ax1.set_ylabel(r'$\log (M_{\star, \mathrm{enclosed}}/[M_{\odot}])$', size=30)