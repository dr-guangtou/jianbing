#!/usr/bin/env python
"""
Prepare the HSC source and random catalogs for lensing analysis
"""

import os
import argparse

from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

from jianbing import prepare
from jianbing import catalog

TOPN_DIR = '/tigress/sh19/work/topn'

def get_specz_cat(z_lim=1.2):
    """
    Get the upper limit of redshift.
    """
    specz_cat = Table.read(os.path.join(TOPN_DIR, 'specz', 's20a_specz_use.fits'))

    return specz_cat[specz_cat['specz_redshift'] <= z_lim]

def precompute_lens(lensing, z_min=0.19, z_max=0.52, njobs=4, output=None, comoving=False):
    """
    Pre-computing dsigma profiles for lenses.
    """
    # Lensing data using medium photo-z quality cut
    prepare_file = os.path.join(TOPN_DIR, 'prepare', lensing)

    # Source
    srcs = Table.read(prepare_file, path='source')

    # Calibration
    calib = Table.read(prepare_file, path='calib')

    # Random
    rand = Table.read(prepare_file, path='random')

    # Get the spec-z catalog
    specz_cat = get_specz_cat()

    # Output file
    if output is None:
        output = lensing.replace('.hdf5', '_precompute.hdf5')
    precompute = os.path.join(TOPN_DIR, 'precompute', output)

    # Define cosmology
    cosmology = FlatLambdaCDM(
        H0=rand.meta['H0'], Om0=rand.meta['Om0'])

    # Radial bins
    rp_bins = rand.meta['bins']

    # HSC Massive galaxies
    hsc_cat = Table.read(
        os.path.join(TOPN_DIR, 'sample', 's16a_massive_logm_11.2_bsm.fits'))

    hsc_pre = prepare.prepare_lens_catalog(
        hsc_cat, srcs, rp_bins=rp_bins, calib=calib,
        z_min=z_min, z_max=z_max, z='z_best', ra='ra', dec='dec', comoving=comoving,
        n_jobs=njobs, field=None, w_sys=None, r_max_mpc=2.0, verbose=True,
        col_used=None, cosmology=cosmology)

    hsc_pre.write(precompute, format='hdf5', path='hsc', serialize_meta=True,
                  append=True, overwrite=True)

    # SDSS redMaPPer
    redm_sdss_cat = Table.read(
        os.path.join(TOPN_DIR, 'sample', 'redmapper_sdss_cluster_use_bsm.fits'))

    redm_sdss_cat = catalog.catalog_update_specz(
        redm_sdss_cat, specz_cat, ra='ra_bcg_redm', dec='dec_bcg_redm',
        redshift='z_lambda_redm', replace=False)

    # Using photo-z
    redm_sdss_pre = prepare.prepare_lens_catalog(
        redm_sdss_cat, srcs, rp_bins=rp_bins, calib=calib,
        z_min=z_min, z_max=z_max, z='z_lambda_redm',
        ra='ra_bcg_redm', dec='dec_bcg_redm', comoving=comoving, n_jobs=njobs,
        field=None, w_sys=None, r_max_mpc=2.0, verbose=True, col_used=None,
        cosmology=cosmology)

    redm_sdss_pre.write(precompute, format='hdf5', path='redm_sdss', serialize_meta=True,
                        append=True, overwrite=True)

    # Using spec-z when available
    redm_sdss_specz_pre = prepare.prepare_lens_catalog(
        redm_sdss_cat, srcs, rp_bins=rp_bins, calib=calib,
        z_min=z_min, z_max=z_max, z='z_best', ra='ra_bcg_redm', dec='dec_bcg_redm',
        comoving=comoving, n_jobs=njobs,
        field=None, w_sys=None, r_max_mpc=2.0, verbose=True, col_used=None,
        cosmology=cosmology)

    redm_sdss_specz_pre.write(
        precompute, format='hdf5', path='redm_sdss_specz', serialize_meta=True,
        append=True, overwrite=True)

    # HSC redM
    redm_hsc_cat = Table.read(
        os.path.join(TOPN_DIR, 'sample', 'redmapper_hsc_s16a_cluster_bsm.fits'))
    redm_hsc_cat = catalog.catalog_update_specz(
        redm_hsc_cat, specz_cat, ra='ra', dec='dec', redshift='z', replace=False)

    redm_hsc_pre = prepare.prepare_lens_catalog(
        redm_hsc_cat, srcs, rp_bins=rp_bins, calib=calib,
        z_min=z_min, z_max=z_max, z='z', ra='ra', dec='dec', comoving=comoving,
        n_jobs=njobs, field=None, w_sys=None, r_max_mpc=2.0, verbose=True,
        col_used=None, cosmology=cosmology)

    redm_hsc_pre.write(
        precompute, format='hdf5', path='redm_hsc', serialize_meta=True,
        append=True, overwrite=True)

    redm_hsc_specz_pre = prepare.prepare_lens_catalog(
        redm_hsc_cat, srcs, rp_bins=rp_bins, calib=calib,
        z_min=z_min, z_max=z_max, z='z_best', ra='ra', dec='dec',
        comoving=comoving, n_jobs=njobs, field=None, w_sys=None, r_max_mpc=2.0,
        verbose=True, col_used=None, cosmology=cosmology)

    redm_hsc_specz_pre.write(
        precompute, format='hdf5', path='redm_hsc_specz', serialize_meta=True,
        append=True, overwrite=True)

    # CAMIRA S16A
    cam_s16a_cat = Table.read(
        os.path.join(TOPN_DIR, 'sample', 'camira_s16a_cluster_use_bsm.fits'))
    cam_s16a_cat = catalog.catalog_update_specz(
        cam_s16a_cat, specz_cat, ra='RA', dec='Dec', redshift='z_cl',
        replace=False)

    cam_s16a_pre = prepare.prepare_lens_catalog(
        cam_s16a_cat, srcs, rp_bins=rp_bins, calib=calib,
        z_min=z_min, z_max=z_max, z='z_cl', ra='RA', dec='Dec', comoving=comoving,
        n_jobs=njobs, field=None, w_sys=None, r_max_mpc=2.0, verbose=True, col_used=None,
        cosmology=cosmology)

    cam_s16a_pre.write(
        precompute, format='hdf5', path='cam_s16a', serialize_meta=True,
        append=True, overwrite=True)

    cam_s16a_specz_pre = prepare.prepare_lens_catalog(
        cam_s16a_cat, srcs, rp_bins=rp_bins, calib=calib,
        z_min=z_min, z_max=z_max, z='z_best', ra='RA', dec='Dec', comoving=comoving,
        n_jobs=njobs, field=None, w_sys=None, r_max_mpc=2.0, verbose=True, col_used=None,
        cosmology=cosmology)

    cam_s16a_specz_pre.write(
        precompute, format='hdf5', path='cam_s16a_specz', serialize_meta=True,
        append=True, overwrite=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--lensing', type=str, dest='lensing', default='s16a_weak_lensing_medium.hdf5')
    parser.add_argument(
        '-o', '--output', type=str, dest='output', default=None)
    parser.add_argument(
        '--z_min', type=float, dest='z_min', default=0.19)
    parser.add_argument(
        '--z_max', type=float, dest='z_max', default=0.52)
    parser.add_argument(
        '-z', '--specz', action="store_true", dest='specz', default=False)
    parser.add_argument(
        '--comoving', action="store_true", dest='comoving', default=False)
    parser.add_argument(
        '-j', '--njobs', type=int, help='Number of jobs run at the same time',
        dest='njobs', default=4)

    args = parser.parse_args()

    precompute_lens(
        args.lensing, z_min=args.z_min, z_max=args.z_max, njobs=args.njobs,
        output=args.output, comoving=args.comoving)
