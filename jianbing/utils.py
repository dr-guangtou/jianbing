#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions."""

import math

import numpy as np

from scipy import interpolate

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from fast_histogram import histogram1d

D2R = math.pi / 180.0
R2D = 180.0 / math.pi

__all__ = ["get_volume", "mass_function_list", "interp_dsig", "rebin_hist", "print_ra_dec", 
           "angular_distance", "r_phy_to_ang"]


def r_phy_to_ang(r_phy, redshift, cosmo=None, phy_unit='kpc', ang_unit='arcsec'):
    """
    Convert physical radius into angular size.
    """
    # Cosmology
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

    # Convert the physical size into an Astropy quantity
    if not isinstance(r_phy, u.quantity.Quantity):
        r_phy = r_phy * u.Unit(phy_unit)

    return (r_phy / cosmo.kpc_proper_per_arcmin(redshift)).to(u.Unit(ang_unit))


def angular_distance(ra_1, dec_1, ra_arr_2, dec_arr_2, radian=False):
    """Angular distance between coordinates.
    Based on calcDistanceAngle from gglens_dsigma_pz_hsc.py by Hironao Miyatake
    Parameters
    ----------
    ra_1, dec_1 : float, float
        RA, Dec of the first sets of coordinates; can be array.
    ra_arr_2, dec_arr_2 : numpy array, numpy array
        RA, Dec of the second sets of coordinates; can be array
    radian: boolen, option
        Whether the input and output are in radian unit. Default=False
    Return
    ------
        Angular distance in unit of arcsec
    """
    # Convert everything into radian if necessary, and make everything
    # float64 array
    if not radian:
        ra_1 = np.array(ra_1 * D2R, dtype=np.float64)
        dec_1 = np.array(dec_1 * D2R, dtype=np.float64)
        ra_2 = np.array(ra_arr_2 * D2R, dtype=np.float64)
        dec_2 = np.array(dec_arr_2 * D2R, dtype=np.float64)
    else:
        ra_1 = np.array(ra_1, dtype=np.float64)
        dec_1 = np.array(dec_1, dtype=np.float64)
        ra_2 = np.array(ra_arr_2, dtype=np.float64)
        dec_2 = np.array(dec_arr_2, dtype=np.float64)

    if radian:
        return np.arccos(
            np.cos(dec_1) * np.cos(dec_2) * np.cos(ra_1 - ra_2) +
            np.sin(dec_1) * np.sin(dec_2))

    return np.arccos(
        np.cos(dec_1) * np.cos(dec_2) * np.cos(ra_1 - ra_2) +
        np.sin(dec_1) * np.sin(dec_2)) * R2D * 3600.0


def get_volume(area, z_low, z_upp, h=0.7, om=0.3, verbose=True):
    """Estimate the volume """
    cosmo = FlatLambdaCDM(H0=h * 100.0, Om0=om)

    vol = ((cosmo.comoving_volume(z_upp) - cosmo.comoving_volume(z_low)) * (area / 41254.0)).value

    if verbose:
        print("# Volume between {} < z < {} is {} Mpc^3".format(z_low, z_upp, vol))

    return vol


def mass_function_list(mass, volume, nb, low, upp, scatter=0.2,
                       correction=1.0, nsample=100):
    """Estimate mass function with potential additional scatter."""
    bins = np.linspace(low, upp, nb + 1)

    if scatter == 0.:
        return [
            histogram1d(
                mass[np.floor(np.random.rand(len(mass)) * len(mass)).astype(int)],
                bins=nb, range=[low, upp]
            ) * correction / volume / (bins[1] - bins[0])
            for i in np.arange(nsample)]

    return [
        histogram1d(
            np.random.normal(mass, scale=scatter), bins=nb, range=[low, upp]
        ) * correction / volume / (bins[1] - bins[0])
        for i in np.arange(nsample)]


def interp_dsig(r_obs, r_mod, dsig_mod):
    """Interpolate the simulated profile to the observed radial bins"""
    return 10.0 ** interpolate.interp1d(
        r_mod, np.log10(dsig_mod), fill_value='extrapolate')(r_obs)


def rebin_hist(x1, y1, x2):
    """Rebin a histogram.
    """
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)

    # the fractional bin locations of the new bins in the old bins
    i_place = np.interp(x2, x1, np.arange(len(x1)))

    cum_sum = np.r_[[0], np.cumsum(y1)]

    # calculate bins where lower and upper bin edges span
    # greater than or equal to one original bin.
    # This is the contribution from the 'intact' bins (not including the
    # fractional start and end parts.
    whole_bins = np.floor(i_place[1:]) - np.ceil(i_place[:-1]) >= 1.
    start = cum_sum[np.ceil(i_place[:-1]).astype(int)]
    finish = cum_sum[np.floor(i_place[1:]).astype(int)]

    y2 = np.where(whole_bins, finish - start, 0.)

    bin_loc = np.clip(np.floor(i_place).astype(int), 0, len(y1) - 1)

    # fractional contribution for bins where the new bin edges are in the same
    # original bin.
    same_cell = np.floor(i_place[1:]) == np.floor(i_place[:-1])
    frac = i_place[1:] - i_place[:-1]
    contrib = (frac * y1[bin_loc[:-1]])
    y2 += np.where(same_cell, contrib, 0.)

    # fractional contribution for bins where the left and right bin edges are in
    # different original bins.
    different_cell = np.floor(i_place[1:]) > np.floor(i_place[:-1])
    frac_left = np.ceil(i_place[:-1]) - i_place[:-1]
    contrib = (frac_left * y1[bin_loc[:-1]])

    frac_right = i_place[1:] - np.floor(i_place[1:])
    contrib += (frac_right * y1[bin_loc[1:]])

    y2 += np.where(different_cell, contrib, 0.)

    return y2

def print_ra_dec(cat, ra='ra', dec='dec', name=None):
    for obj in cat:
        if name is None:
            print("{:9.5f} {:9.5f}".format(obj[ra], obj[dec]))
        else:
            print("{:s} {:9.5f} {:9.5f}".format(str(obj[name]), obj[ra], obj[dec]))
