#!/usr/bin/env python
"""
Prepare the HSC source and random catalogs for lensing analysis
"""

import os
import argparse

from astropy.table import Table

from jianbing import prepare


DATA_DIR = '/tigress/sh19/work/topn'

# S16A weak lensing source catalog
S16A_SRCS = Table.read(os.path.join(DATA_DIR, 'raw', 's16a_weak_lensing_source_frankenz.fits'))

# S16A random object catalog
S16A_RAND = Table.read(os.path.join(DATA_DIR, 'raw', 's16a_random_acturus_master.fits'))

# Photometric calibration catalog
S16A_CALIB = Table.read(os.path.join(DATA_DIR, 'raw', 'cosmos_photoz_calibration_reweighted.fits'))


def run_prepare(medium=False, basic=False, strict=False, comoving=False,
                larger=False, njobs=4, nrand=500000):
    """Prepare the source and random catalogs."""

    if medium:
        print("# Preparing the medium photo-z cut...")
        s16a_prepare_med = os.path.join(DATA_DIR, 'prepare', 's16a_weak_lensing_medium.hdf5')

        _ = prepare.prepare_source_random(
            S16A_SRCS, S16A_RAND, calib=S16A_CALIB, photoz_cut='medium', dz_min=0.1,
            cosmology=None, H0=70.0, Om0=0.3, comoving=False, n_jobs=njobs,
            r_min=0.1, r_max=20, n_bins=12, output=s16a_prepare_med, n_random=nrand,
            verbose=True)

    if strict:
        print("# Preparing the strict photo-z cut...")
        s16a_prepare_str = os.path.join(DATA_DIR, 'prepare', 's16a_weak_lensing_strict.hdf5')

        _ = prepare.prepare_source_random(
            S16A_SRCS, S16A_RAND, calib=S16A_CALIB, photoz_cut='strict', dz_min=0.12,
            cosmology=None, H0=70.0, Om0=0.3, comoving=False, n_jobs=njobs,
            r_min=0.1, r_max=20, n_bins=12, output=s16a_prepare_str, n_random=nrand,
            verbose=True)

    if basic:
        print("# Preparing the basic photo-z cut...")
        s16a_prepare_bsc = os.path.join(DATA_DIR, 'prepare', 's16a_weak_lensing_basic.hdf5')

        _ = prepare.prepare_source_random(
            S16A_SRCS, S16A_RAND, calib=S16A_CALIB, photoz_cut='basic', dz_min=0.10,
            cosmology=None, H0=70.0, Om0=0.3, comoving=False, n_jobs=njobs,
            r_min=0.1, r_max=20, n_bins=12, output=s16a_prepare_bsc, n_random=nrand,
            verbose=True)

    if comoving:
        print("# Preparing the comoving run")
        s16a_prepare_comoving = os.path.join(
            DATA_DIR, 'prepare', 's16a_weak_lensing_medium_comoving.hdf5')

        _ = prepare.prepare_source_random(
            S16A_SRCS, S16A_RAND, calib=S16A_CALIB, photoz_cut='medium', dz_min=0.10,
            cosmology=None, H0=70.0, Om0=0.3, comoving=True, n_jobs=njobs,
            r_min=0.1, r_max=20, n_bins=12, output=s16a_prepare_comoving, n_random=nrand,
            verbose=True)

    if larger:
        print("# Preparing the larger radial range...")
        s16a_prepare_out = os.path.join(
            DATA_DIR, 'prepare', 's16a_weak_lensing_medium_larger.hdf5')

        _ = prepare.prepare_source_random(
            S16A_SRCS, S16A_RAND, calib=S16A_CALIB, photoz_cut='medium', dz_min=0.1,
            cosmology=None, H0=70.0, Om0=0.3, comoving=False, n_jobs=njobs,
            r_min=0.1, r_max=35, n_bins=14, output=s16a_prepare_out, n_random=nrand,
            verbose=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--medium', action="store_true", dest='medium', default=False)
    parser.add_argument(
        '-b', '--basic', action="store_true", dest='basic', default=False)
    parser.add_argument(
        '-s', '--strict', action="store_true", dest='strict', default=False)
    parser.add_argument(
        '-c', '--comoving', action="store_true", dest='comoving', default=False)
    parser.add_argument(
        '-l', '--larger', action="store_true", dest='larger', default=False)
    parser.add_argument(
        '-j', '--njobs', type=int, help='Number of jobs run at the same time',
        dest='njobs', default=1)
    parser.add_argument(
        '-r', '--nrand', type=int, help='Number of random objects',
        dest='nrand', default=500000)

    args = parser.parse_args()

    _ = run_prepare(medium=args.medium, basic=args.basic, strict=args.strict,
                    comoving=args.comoving, larger=args.larger, njobs=args.njobs,
                    nrand=args.nrand)
