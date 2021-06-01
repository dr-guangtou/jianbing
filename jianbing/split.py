#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to deal sample split test."""

import copy

import numpy as np

from scipy.spatial import cKDTree
from scipy.optimize import curve_fit

from astropy.table import Column, vstack

from . import wlensing
from . import visual

__all__ = ["get_mask_strait_line", "strait_line", "get_mask_rank_split", "sample_split_test", 
           "dsig_compare_matched_sample"]


def strait_line(x, A, B):
    """Simple strait line model."""
    return A * x + B

def get_mask_strait_line(x_arr, y_arr, mask=None, sigma=1.5, return_result=True):
    """Separte the sample using scaling relation."""
    if mask is None:
        mask = np.isfinite(x_arr)

    popt, pcov = curve_fit(strait_line, x_arr[mask], y_arr[mask])
    perr = np.sqrt(np.diag(pcov))

    slope, inter = popt[0], popt[1]
    slope_err, inter_err = perr[0], perr[1]

    mask_upp = (
        (y_arr > x_arr * slope + inter + sigma * inter_err) & mask
    )

    mask_low = (
        (y_arr < x_arr * slope + inter - sigma * inter_err) & mask
    )

    if return_result:
        result = {"slope": slope, "inter": inter,
                  "slope_err": slope_err, "inter_err": inter_err,
                  "sigma": sigma}
        return [mask_low, mask_upp], result
    return [mask_low, mask_upp]


def get_mask_rank_split(cat, X_col, Y_col, n_bins=5, n_sample=2, X_min=None, X_max=None,
                        X_bins=None, select=1, return_data=False, mask=None):
    """Split sample into N_sample with fixed distribution in X, but different
    rank orders in Y."""
    if mask is not None:
        data = copy.deepcopy(cat[mask])
    else:
        data = copy.deepcopy(cat)

    if isinstance(X_col, str):
        X = data[X_col]
    else:
        if mask is None:
            X = X_col
        else:
            X = X_col[mask]
        data.add_column(Column(data=X, name='X'))

    if isinstance(Y_col, str):
        data.rename_column(Y_col, 'Y')
    else:
        if mask is None:
            data.add_column(Column(data=Y_col, name='Y'))
        else:
            data.add_column(Column(data=Y_col[mask], name='Y'))

    X_len = len(X)
    if X_bins is None:
        if X_min is None:
            X_min = np.nanmin(X)
        if X_max is None:
            X_max = np.nanmax(X)

        msg = '# Sample size should be much larger than number of bins in X'
        assert X_len > (2 * n_bins), msg
        X_bins = np.linspace(X_min * 0.95, X_max * 1.05, (n_bins + 1))
    else:
        n_bins = (len(X_bins) - 1)

    # Place holder for sample ID
    data.add_column(Column(data=(np.arange(X_len) * 0), name='sample_id'))
    data.add_column(Column(data=np.arange(X_len), name='index_ori'))

    # Create index array for object in each bin
    X_idxbins = np.digitize(X, X_bins, right=True)

    bin_list = []
    for ii in range(n_bins):
        subbin = data[X_idxbins == (ii + 1)]
        if isinstance(Y_col, str):
            subbin.sort(Y_col)
        else:
            subbin.sort('Y')

        subbin_len = len(subbin)
        subbin_size = int(np.ceil(subbin_len / n_sample))

        idx_start, idx_end = 0, subbin_size
        for jj in range(n_sample):
            if idx_end > subbin_len:
                idx_end = subbin_len
            subbin['sample_id'][idx_start:idx_end] = (jj + 1)
            idx_start = idx_end
            idx_end += subbin_size

        bin_list.append(subbin)

    new_data = vstack(bin_list)
    new_data.sort('index_ori')
    new_data.meta = cat.meta

    if n_sample <= select:
        raise ValueError("n_sample needs to be larger than select!")
    mask_1 = (new_data['sample_id'] <= select)
    mask_2 = (new_data['sample_id'] > (n_sample - select))

    if return_data:
        return [mask_1, mask_2], new_data
    return [mask_1, mask_2]


def sample_split_test(cat, x_arr, y_arr, rand, mask=None, n_rand=150000, n_boot=0,
                      bootstrap=False, sigma=1.5, n_bins=30, n_sample=5, select=2,
                      n_jk=45, plot=True, rank=True, input_masks=None, **plot_kwargs):
    """Sample split DeltaSigma test."""
    if isinstance(x_arr, str) and isinstance(y_arr, str):
        x_arr = cat[x_arr]
        y_arr = cat[y_arr]

    # Using the best-fit scaling relation
    if input_masks is None:
        mask_list, line_result = get_mask_strait_line(x_arr, y_arr, mask=mask, sigma=sigma)
    else:
        mask_list, line_result = input_masks, None
        if mask is not None:
            mask_list = [(m & mask) for m in mask_list]
        rank = False

    dsig_line = wlensing.batch_dsigma_profiles(
        cat, rand, mask_list, n_rand=n_rand, n_boot=n_boot, bootstrap=bootstrap,
        n_jk=n_jk, verbose=True, n_jobs=None)

    if plot:
        _ = visual.show_split_result(
            dsig_line, x_arr, y_arr, mask_list, mask=mask, line_result=line_result, **plot_kwargs)

    # Rank order splitting
    if rank:
        mask_list, data_new = get_mask_rank_split(
            cat, x_arr, y_arr, n_bins=n_bins, n_sample=n_sample, select=select,
            mask=mask, return_data=True)

        dsig_rank = wlensing.batch_dsigma_profiles(
            data_new, rand, mask_list, n_rand=n_rand, n_boot=n_boot, bootstrap=bootstrap,
            n_jk=n_jk, verbose=True, n_jobs=None)

        if plot:
            _ = visual.show_split_result(
                dsig_rank, x_arr[mask], y_arr[mask], mask_list, mask=None, **plot_kwargs)

        return dsig_line, dsig_rank
    return dsig_line


def dsig_compare_matched_sample(sample, target, col_1, col_2, leaf_size=9, query_size=2,
                                unique=True):
    """Compare the DSigma profiles of two samples after matching them."""
    data_ref = np.c_[
        np.asarray(sample[col_1]), np.asarray(sample[col_2])]
    data_tar = np.c_[
        np.asarray(target[col_1]), np.asarray(target[col_2])]

    tree = cKDTree(data_ref, leafsize=leaf_size)
    _, index = tree.query(data_tar, k=query_size)

    index_match = index.flatten()

    if unique:
        match = sample[np.unique(index_match)]
    else:
        match = sample[index_match]

    target.add_column(Column(
        data=(['target'] * len(target)), name='type'))
    match.add_column(Column(
        data=(['match'] * len(match)), name='type'))

    sample_new = vstack([target, match])
    sample_new.meta = sample.meta

    return index_match, sample_new
