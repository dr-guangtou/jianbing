#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Functions to prepare catalogs and pre-compute lensing signals'''

import itertools

import numpy as np

import healpy

import astropy.units as u
from astropy.table import Table, Column
from astropy.cosmology import FlatLambdaCDM

from dsigma import helpers
from dsigma.surveys import hsc
from dsigma.precompute import add_precompute_results, add_maximum_lens_redshift
from dsigma.jackknife import add_continous_fields
from dsigma.jackknife import jackknife_resampling
from dsigma.jackknife import jackknife_field_centers, add_jackknife_fields

from . import catalog

__all__ = ["prepare_random_catalog", "prepare_source_random", "prepare_lens_catalog", 
           "design_topn_bins", "filter_healpix_mask"]


def filter_healpix_mask(mask_hp, cat, ra='ra', dec='dec', verbose=True):
    """Filter a catalog through a Healpix mask.
    Parameters
    ----------
    mask_hp : healpy mask file
        healpy mask file
    cat : numpy array or astropy.table
        Catalog that includes the coordinate information
    ra : string
        Name of the column for R.A.
    dec : string
        Name of the column for Dec.
    verbose : boolen, optional
        Default: True
    Return
    ------
        Selected objects that are covered by the mask.
    """
    # Read the healpix mask
    mask = healpy.read_map(mask_hp, nest=True, dtype=np.bool)

    nside, hp_indices = healpy.get_nside(mask), np.where(mask)[0]
    phi, theta = np.radians(cat[ra]), np.radians(90. - cat[dec])
    hp_masked = healpy.ang2pix(nside, theta, phi, nest=True)
    select = np.in1d(hp_masked, hp_indices)

    if verbose:
        print("# %d/%d objects are selected by the mask" % (select.sum(), len(cat)))

    return cat[select]

def prepare_random_catalog(randoms, z='redshift', w_sys=None, size=250000,
                           z_low=0.18, z_upp=0.52):
    """Pre-compute the DeltaSigma profile for a downsampled random objects."""
    # Randomly select a small sample of random objects
    random_use = np.random.choice(randoms, size=size, replace=False)

    # Assign a uniform distribution of redshift
    random_use[z] = np.random.uniform(low=z_low, high=z_upp, size=size)
    random_use = Table(random_use)

    # Add a systematic weight function is necessary
    if w_sys is None:
        random_use.add_column(Column(name='w_sys', data=np.ones(size)))

    random_pre = helpers.dsigma_table(
        random_use, 'lens', ra='ra', dec='dec', z=z, field='field', w_sys='w_sys'
    )

    return random_pre

def prepare_source_random(srcs, rand, calib=None, photoz_cut='medium', dz_min=0.1,
                          cosmology=None, H0=70.0, Om0=0.3, comoving=False, n_jobs=4,
                          r_min=0.15, r_max=11, n_bins=11, output=None, n_random=500000,
                          verbose=True):
    """Prepare the lensing source, calibration, and random catalogs.

    Also precompute the DeltaSigma profiles for randoms if necessary, and define
    the cosmology model and the radial bins used for lensing profiles.
    """
    # Define cosmology
    if cosmology is None:
        cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

    # Define radial bins
    rp_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins)

    # Photo-z quality cuts
    if verbose:
        print("# Use the {:s} photo-z quality cut".format(photoz_cut))

    if photoz_cut == 'basic':
        photoz_mask = (srcs['frankenz_model_llmin'] < 6.)
    elif photoz_cut == 'medium':
        photoz_mask = (
            srcs['frankenz_model_llmin'] < 6.) & (srcs['frankenz_photoz_risk_best'] < 0.25)
    elif photoz_cut == 'strict':
        photoz_mask = (
            srcs['frankenz_model_llmin'] < 6.) & (srcs['frankenz_photoz_risk_best'] < 0.15)
    else:
        raise Exception("# Wrong photo-z quality cut type: [basic/medium/strict]")

    # Prepare the source catalog
    if verbose:
        print("\n# Prepare the lensing source catalog")
    srcs_use = helpers.dsigma_table(
        srcs[photoz_mask], 'source', survey='hsc', version='PDR2', field='field',
        z='frankenz_photoz_best', z_low='frankenz_photoz_err68_min',
        z_upp='frankenz_photoz_err68_max'
    )

    # Add maximum usable redshift for lenses
    srcs_use = add_maximum_lens_redshift(
        srcs_use, dz_min=dz_min, z_err_factor=0, apply_z_low=True)

    # Prepare the calibration catalog if necessary
    if calib is not None:
        if verbose:
            print("\n# Prepare the lensing calibration catalog")
        # Photo-z quality cuts
        if photoz_cut == 'basic':
            photoz_mask = (calib['frankenz_model_llmin'] < 6.)
        elif photoz_cut == 'medium':
            photoz_mask = (
                calib['frankenz_model_llmin'] < 6.) & (calib['frankenz_photoz_risk_best'] < 0.25)
        elif photoz_cut == 'strict':
            photoz_mask = (
                calib['frankenz_model_llmin'] < 6.) & (calib['frankenz_photoz_risk_best'] < 0.15)
        else:
            raise Exception("# Wrong photo-z quality cut type: [basic/medium/strict]")

        # Prepare the calibration catalog
        calib_use = helpers.dsigma_table(
            calib[photoz_mask], 'calibration', z='frankenz_photoz_best',
            z_true='z_true', z_low='frankenz_photoz_err68_min',
            w='w_source', w_sys='somw_cosmos_samplevaraince'
        )

        # Add maximum usable redshift for lenses
        calib_use = add_maximum_lens_redshift(
            calib_use, dz_min=dz_min, z_err_factor=0, apply_z_low=True)
    else:
        calib_use = None

    # Prepare the random catalogs
    if verbose:
        print("\n# Prepare the random object catalog")
    rand_use = prepare_random_catalog(rand, size=n_random)

    # Pre-compute the DeltaSigma profiles for random objects
    if verbose:
        print("\n# Pre-compute the DeltaSigma profiles for random objects")
    rand_pre = add_precompute_results(
        rand_use, srcs_use, rp_bins, table_c=calib_use,
        cosmology=cosmology, comoving=comoving, n_jobs=n_jobs)

    # Remove the ones with no useful lensing information
    rand_pre['n_s_tot'] = np.sum(rand_pre['sum 1'], axis=1)
    rand_pre = rand_pre[rand_pre['n_s_tot'] > 0]

    if output is not None:
        srcs_use.write(output, path='source', format='hdf5')
        calib_use.write(output, path='calib', format='hdf5', append=True)
        rand_pre.write(output, path='random', format='hdf5', append=True)
        return
    else:
        return {'cosmology': cosmology, 'rp_bins': rp_bins, 'source': srcs_use,
                'calib': calib_use, 'random': rand_pre}

def prepare_lens_catalog(cat, src, rp_bins=None, calib=None, z_min=0.19, z_max=0.52,
                         z='z', ra='ra', dec='dec', comoving=False, n_jobs=4,
                         field=None, w_sys=None, r_max_mpc=2.0, verbose=True, col_used=None,
                         cosmology=None):
    """Prepare the lens catalog:
        1. Select lenses in the right redshift range defined by `z_min` < z <= `z_max`.
        2. Match to the source catalog using the KDTree. Matching radius is defined as `r_max_mpc`.
        3. Prepare the catalog for pre-computation: adding `field` and lense weight if necessary.
    """
    # Generate a KDTree to match
    src_tree = catalog.catalog_to_kdtree(src, 'ra', 'dec')

    # Cosmology parameters
    if cosmology is None:
        cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)

    # Radial bins
    if rp_bins is None:
        rp_bins = np.logspace(np.log10(0.1), np.log10(20), 11)

    # Redshift cut
    cat_use = cat[(cat[z] > z_min) & (cat[z] <= z_max)]
    if len(cat_use) < 1:
        print("# No useful objects left after the redshift cut!")
        return
    if verbose:
        print("# {:d} / {:d} objects left after the redshift cut".format(len(cat_use), len(cat)))

    # Match to the source catalog
    # Maximum matching radius in deg
    r_max_deg = (cosmology.arcsec_per_kpc_proper(cat_use[z]) * (
        r_max_mpc * u.Mpc).to(u.kpc)).to(u.degree).value

    # Maximum matching radius in the 3-D Cartesian coordinates used by the KDTree
    r_max_3d = np.sqrt(2 - 2 * np.cos(np.deg2rad(r_max_deg)))

    # Get the KDTree of the lens catalog
    cat_kdtree = catalog.catalog_to_kdtree(cat_use, ra, dec)

    cat_index = list(itertools.chain(
        *src_tree.query_ball_tree(cat_kdtree, r=r_max_3d.max())))
    cat_use = cat_use[np.unique(np.asarray(cat_index))]

    if len(cat_use) < 1:
        print("# No useful objects left after the source catalog match!")
        return
    if verbose:
        print("# {:d} / {:d} objects left after the source catalog match!".format(
            len(cat_use), len(cat)))

    # Add continued fields
    cat_use.rename_column(ra, 'ra')
    cat_use.rename_column(dec, 'dec')

    if field is None:
        cat_pre = add_continous_fields(cat_use, n_samples=10000, distance_threshold=1.0)
        field = 'field'

    # Add a place holder for systematic weight if necessary
    if w_sys is None:
        w_sys = 1.0

    # Organize the columns need to be transfered
    if col_used is None:
        col_used = cat_pre.colnames

    for col in ['ra', 'dec', 'z', z, field, 'w_sys']:
        if col in col_used:
            col_used.remove(col)

    col_kwargs = {}
    for col in col_used:
        col_kwargs[col.lower()] = col

    # Get the catalog ready for dsigma
    cat_pre = helpers.dsigma_table(
        cat_pre, 'lens', ra='ra', dec='dec', z=z, field=field, w_sys=w_sys,
        **col_kwargs
    )

    # Pre-computation for the lenses
    cat_pre = add_precompute_results(
        cat_pre, src, rp_bins, table_c=calib, cosmology=cosmology,
        comoving=comoving, n_jobs=n_jobs)

    # Remove the ones with no useful lensing information
    cat_pre['n_s_tot'] = np.sum(cat_pre['sum 1'], axis=1)
    cat_pre = cat_pre[cat_pre['n_s_tot'] > 0]

    return cat_pre

def design_topn_bins(sample, edges, col='lambda', upper=None, verbose=True, **kwargs):
    """Design the number density bins."""

    # Make sure the edges of the bins are in descending order
    edges = np.sort(np.array(edges))[::-1]

    # Upper boundary of the parameter used
    if upper is None:
        upper = np.nanmax(sample[col])

    n_obj, n_cum, r, n, r_err = [], [], [], [], []
    for index, edge in enumerate(edges):
        if index == 0:
            n_bin, rho, rho_err = catalog.get_number_density(
                sample, edge, upper, property_col=col, **kwargs)
        else:
            n_bin, rho, rho_err = catalog.get_number_density(
                sample, edge, edges[index - 1], property_col=col, **kwargs)

        n_tot, rho_cum, _ = catalog.get_number_density(
            sample, edges[index], upper, **kwargs)

        if verbose:
            print("N_bin: {:5d}  N_tot: {:5d}  rho: {:8.4f}+/-{:6.4f}  rho_cum: {:8.4f}".format(
                n_bin, n_tot, rho, rho_err, rho_cum))

        n_obj.append(n_bin)
        n_cum.append(n_tot)
        r.append(rho * 1e-6)
        r_err.append(rho_err * 1e-6)
        n.append(rho_cum * 1e-6)

    topn = Table()
    topn['bin_id'] = np.arange(len(edges)) + 1
    topn['n_obj'] = n_obj
    topn['n_cum'] = n_cum
    topn['rho_bin'] = r
    topn['rho_bin_err'] = r_err
    topn['rho_cum'] = n
    topn['index_low'] = np.asarray([0] + list(topn['n_cum'][:-1]))
    topn['index_upp'] = np.asarray(n_cum) - 1

    return topn
