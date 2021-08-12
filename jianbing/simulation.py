#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to handle data from MDPL2 or SMDPL simulations."""

import os

import numpy as np
import pandas as pd

import halotools.sim_manager

__all__ = ["reduce_hlist_size", "downsample_particles"]

# Useful columns form the Rockstar halo catalog
HLIST_USED_DEFAULT = {
    "id": (1, "i8"), # id of halo
    "pid": (5, "i8"), # least massive parent (direct parent) halo ID
    "upid": (6, "i8"), # least massive parent (direct parent) halo ID
    "Mvir": (10, "f8"), # Msun/h
    "Rvir": (11, "f8"), # kpc/h
    "rs": (12, "f8"), # scale radius kpc/h
    "halo_x": (17, "f8"), # halo position x Mpc/h
    "halo_y": (18, "f8"), # halo position y Mpc/h
    "halo_z": (19, "f8"), # halo position z Mpc/h
    "vx": (20, "f8"), # halo position x Mpc/h
    "vy": (21, "f8"), # halo position y Mpc/h
    "vz": (22, "f8"), # halo position z Mpc/h
    "M200b": (39, "f8"), # Msun/h
    "M500c": (41, "f8"), # Msun/h
    "Mpeak": (60, "f8"), # Msun/h
    "scale_half_mass": (63, "f8"), # scale factor at which we could to 0.5 * mpeak
    "scale_last_mm": (15, "f8"), # scale factor at last MM
    "vmax_mpeak": (74, "f8"), # vmax at the scale where mpeak was reached
}

def reduce_hlist_size(hlist, col_used=HLIST_USED_DEFAULT, logmh_lim=None, mh_type="Mpeak"):
    '''Reduce the number of columns of the Rockstar catalog.

    Parameters
    ----------
    hlist: `string`
        Path to the Rockstar halo catalog. Typical name is `hlist_[scale_factor].list`.
    col_used: `dict`, optional
        A dictionary that summarizes the useful columns. Default: HLIST_USED_DEFAULT
        The format should be: "column_name": (index_in_hlist, "data_type").
        Please see the header of the Rockstar catalog as reference.
    logmvir_lim: `float`, optional
        Minimum Mvir halo mass cut to reduce the number of halo. Default: None

    Returns
    -------
    saved: `bool`
        description


    Examples
    --------

    > simulation.reduce_hlist_size('hlist_0.73330.list', logmvir_lim=13.0)

    Notes
    -----

    '''
    # Use halotools to handle the reading of large files.
    hlist_reader = halotools.sim_manager.TabularAsciiReader(hlist, col_used)
    reduced_catalog = hlist_reader.read_ascii()

    # New file name
    hlist_pre, _ = os.path.splitext(hlist)
    hlist_pre += '_reduced'

    # Halo mass cut
    if logmh_lim is not None:
        reduced_catalog = reduced_catalog[reduced_catalog[mh_type] > (10.0 ** logmh_lim)]
        hlist_pre += '_log{:s}_{:4.1f}'.format(mh_type.lower().strip(), logmh_lim)

    # Save a new file
    np.save(hlist_pre + '.npy', reduced_catalog)


def downsample_particles(ptbl_file, n_million, seed=95064, csv=False, verbose=True):
    """Down-sample the partile files from the DM simulation."""
    if not os.path.isfile(ptbl_file):
        raise IOError("# Can not find the particle table : %s" % ptbl_file)
    ptbl_pre, ptbl_ext = os.path.splitext(ptbl_file)

    # Reduce the number of colunms and save as a numpy array
    ptbl_out = ptbl_pre + "_downsample_%.1fm.npy" % n_million
    if verbose:
        print("# Save the downsampled catalog to : %s" % ptbl_out)

    # Data format for output
    particle_table_dtype = [
        ("x", "float64"), ("y", "float64"), ("z", "float64")]

    if csv or ptbl_ext == '.csv':
        use_csv = True
    else:
        use_csv = False

    # Read the data
    chunksize = 1000000
    ptbl_pchunks = pd.read_csv(
        ptbl_file, usecols=[0, 1, 2], delim_whitespace=use_csv,
        names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'id'],
        dtype=particle_table_dtype, index_col=False,
        chunksize=chunksize)

    ptbl_pdframe = pd.concat(ptbl_pchunks)
    ptbl_array = ptbl_pdframe.values.ravel().view(dtype=particle_table_dtype)

    # Downsample
    np.random.seed(seed)
    ptbl_downsample = np.random.choice(ptbl_array, int(n_million * 1e6), replace=False)

    # Save the result
    np.save(ptbl_out, ptbl_downsample)
