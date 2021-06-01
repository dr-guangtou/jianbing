#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to deal sample split test."""

import numpy as np
from astropy.table import vstack

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles

from palettable.colorbrewer.qualitative import Set2_3

from . import wlensing

__all__ = ['compare_two_samples', 'compare_three_samples', 'prepare_set', 'plot_venn_diagram',
           'gather_lensing_profiles', 'compare_lensing_profile']


def compare_two_samples(data, rand, n_rand=200000, n_jk=40, n_boot=200, rdsig=False, common=True,
                        x_off=0.05, y_factor=None, get_dsigma=True, single=False):
    """Compare the lensing properties using venn diagram of three samples."""
    # Prepare the dataset
    data = prepare_set(data)

    # Getting the DeltaSigma profiles
    if get_dsigma:
        (dsig_a, dsig_b, dsig_c,
         lens_a, lens_b, lens_c) = gather_lensing_profiles(
             data, rand, n_rand=n_rand, n_jk=n_jk, n_boot=n_boot,
             common=common, return_lens=True)
        data['dsig_a'], data['lens_a'] = dsig_a, lens_a
        data['dsig_b'], data['lens_b'] = dsig_b, lens_b
        data['dsig_c'], data['lens_c'] = dsig_c, lens_c
    else:
        dsig_a, dsig_b = data['dsig_a'], data['dsig_b']
        dsig_c = data['dsig_c']

    # Determine the Y-range
    if dsig_c is not None:
        ds_arr = list(vstack([dsig_a, dsig_b, dsig_c], metadata_conflicts='silent')['ds'])
        r_mpc = np.stack(
            [data['sample_1']['r_mpc'], data['sample_2']['r_mpc'],
             data['sample_1']['r_mpc']]).flatten()
    else:
        ds_arr = list(vstack([dsig_a, dsig_b], metadata_conflicts='silent')['ds'])
        r_mpc = np.stack(
            [data['sample_1']['r_mpc'], data['sample_2']['r_mpc']]).flatten()

    if rdsig:
        y_min, y_max = np.min(r_mpc * ds_arr), np.max(r_mpc * ds_arr)
    else:
        y_min, y_max = np.min(ds_arr), np.max(ds_arr)
    y_min = 0.15 if y_min <= 0. else y_min

    if y_factor is None:
        if rdsig:
            y_factor = [0.10, 1.5]
        else:
            y_factor = [0.30, 1.8]

    y_lim = [y_min * y_factor[0], y_max * y_factor[1]]

    # Making a figure
    if single:
        fig = plt.figure(figsize=(8, 7))
        ax1 = fig.add_axes([0.14, 0.125, 0.85, 0.865])
        if not rdsig:
            ax2 = ax1.inset_axes([0.05, 0.30, 0.35, 0.26])
            x_loc = 0.72
        else:
            ax2 = ax1.inset_axes([0.64, 0.77, 0.28, 0.26])
            x_loc = 0.36
        ax1 = compare_lensing_profile(
            data, dsig_a, dsig_b, dsig_c, dsig_d=None, ax=ax1,
            rdsig=rdsig, x_off=x_off, y_lim=y_lim,
            x_loc=x_loc, y_loc=0.92, fontsize=20)
        ax2 = plot_venn_diagram(data, alpha=0.5, ax=ax2, fontsize=15)
    else:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_axes([0.08, 0., 0.40, 1.0])
        ax2 = fig.add_axes([0.50, 0.16, 0.475, 0.80])
        ax1 = plot_venn_diagram(data, alpha=0.5, ax=ax1)
        ax2 = compare_lensing_profile(
            data, dsig_a, dsig_b, dsig_c, dsig_d=None, ax=ax2,
            rdsig=rdsig, x_off=x_off, y_lim=y_lim)

    return data, fig


def compare_three_samples(data, rand, n_rand=200000, n_jk=40, n_boot=200, rdsig=False, common=True,
                          x_off=0.05, y_factor=None, get_dsigma=True, single=False):
    """Compare the lensing properties using venn diagram of three samples."""
    # Prepare the dataset
    data = prepare_set(data)

    # Getting the DeltaSigma profiles
    if get_dsigma:
        (dsig_a, dsig_b, dsig_c, dsig_d,
         lens_a, lens_b, lens_c, lens_d) = gather_lensing_profiles(
             data, rand, n_rand=n_rand, n_jk=n_jk, n_boot=n_boot,
             common=common, return_lens=True)
        data['dsig_a'], data['lens_a'] = dsig_a, lens_a
        data['dsig_b'], data['lens_b'] = dsig_b, lens_b
        data['dsig_c'], data['lens_c'] = dsig_c, lens_c
        data['dsig_d'], data['lens_d'] = dsig_d, lens_d
    else:
        dsig_a, dsig_b = data['dsig_a'], data['dsig_b']
        dsig_c, dsig_d = data['dsig_c'], data['dsig_d']

    # Determine the Y-range
    if dsig_d is not None:
        ds_arr = list(vstack([dsig_a, dsig_b, dsig_c, dsig_d], metadata_conflicts='silent')['ds'])
        r_mpc = np.stack(
            [data['sample_1']['r_mpc'], data['sample_2']['r_mpc'],
             data['sample_3']['r_mpc'], data['sample_1']['r_mpc']]).flatten()
    else:
        ds_arr = list(vstack([dsig_a, dsig_b, dsig_c], metadata_conflicts='silent')['ds'])
        r_mpc = np.stack(
            [data['sample_1']['r_mpc'], data['sample_2']['r_mpc'],
             data['sample_3']['r_mpc']]).flatten()

    if rdsig:
        y_min, y_max = np.min(r_mpc * ds_arr), np.max(r_mpc * ds_arr)
    else:
        y_min, y_max = np.min(ds_arr), np.max(ds_arr)
    y_min = 0.9 if y_min <= 0. else y_min

    if y_factor is None:
        if rdsig:
            y_factor = [0.10, 1.5]
        else:
            y_factor = [0.30, 1.8]

    y_lim = [y_min * y_factor[0], y_max * y_factor[1]]

    # Making a figure
    if single:
        fig = plt.figure(figsize=(8, 7))
        ax1 = fig.add_axes([0.14, 0.12, 0.85, 0.875])
        if not rdsig:
            ax2 = ax1.inset_axes([0.05, 0.30, 0.35, 0.26])
        else:
            ax2 = ax1.inset_axes([0.64, 0.77, 0.28, 0.26])
        ax1 = compare_lensing_profile(
            data, dsig_a, dsig_b, dsig_c, dsig_d=None, ax=ax1,
            rdsig=rdsig, x_off=x_off, y_lim=y_lim,
            x_loc=0.36, y_loc=0.92, fontsize=20)
        ax2 = plot_venn_diagram(data, alpha=0.5, ax=ax2, fontsize=15)
    else:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_axes([0.025, 0.0, 0.36, 1.1])
        ax2 = fig.add_axes([0.52, 0.16, 0.475, 0.80])
        ax1 = plot_venn_diagram(data, alpha=0.5, ax=ax1, x_loc=-0.03)
        ax2 = compare_lensing_profile(
            data, dsig_a, dsig_b, dsig_c, dsig_d=dsig_d, ax=ax2,
            rdsig=rdsig, x_off=x_off, y_lim=y_lim, x_loc=0.7)

    return data, fig


def prepare_set(data):
    """Prepare the samples."""
    sample_1 = data['sample_1']
    cat_1 = sample_1['data']
    cat_1 = cat_1[(cat_1['z'] >= data['z_low']) & (cat_1['z'] < data['z_upp'])]
    if data['use_all_1']:
        use_1 = cat_1
    else:
        cat_1.sort(sample_1['column'])
        cat_1.reverse()
        use_1 = cat_1[data['index_low']: data['index_upp']]
    set_1 = set(use_1['index'])

    sample_2 = data['sample_2']
    cat_2 = sample_2['data']
    cat_2 = cat_2[(cat_2['z'] >= data['z_low']) & (cat_2['z'] < data['z_upp'])]
    if data['use_all_2']:
        use_2 = cat_2
    else:
        cat_2.sort(sample_2['column'])
        cat_2.reverse()
        use_2 = cat_2[data['index_low']: data['index_upp']]
    set_2 = set(use_2['index'])

    data['sample_1']['r_mpc'] = np.sqrt(
        cat_1.meta['rp_bins'][:-1] * cat_1.meta['rp_bins'][1:])
    data['sample_2']['r_mpc'] = np.sqrt(
        cat_2.meta['rp_bins'][:-1] * cat_2.meta['rp_bins'][1:])

    data['sample_1']['use'] = use_1
    data['sample_1']['set'] = set_1

    data['sample_2']['use'] = use_2
    data['sample_2']['set'] = set_2

    if data['sample_3'] is not None:
        sample_3 = data['sample_3']
        cat_3 = sample_3['data']
        cat_3 = cat_3[(cat_3['z'] >= data['z_low']) & (cat_3['z'] < data['z_upp'])]
        if data['use_all_3']:
            use_3 = cat_3
        else:
            cat_3.sort(sample_3['column'])
            cat_3.reverse()
            use_3 = cat_3[data['index_low']: data['index_upp']]
        set_3 = set(use_3['index'])

        data['sample_3']['r_mpc'] = np.sqrt(
            cat_3.meta['rp_bins'][:-1] * cat_3.meta['rp_bins'][1:])

        data['sample_3']['use'] = use_3
        data['sample_3']['set'] = set_3

    return data


def plot_venn_diagram(data, ax=None, fontsize=25, alpha=0.6, x_loc=0.01, y_loc=-0.1):
    """Plot a Venn diagram of the three samples of galaxies."""
    # Making the plot
    if ax is None:
        fig = plt.figure(figsize=(9, 6))
        fig.subplots_adjust(
            left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0, hspace=0)
        ax = fig.add_subplot(111)
        use_ax = False
    else:
        use_ax = True

    set_1 = data['sample_1']['set']
    set_2 = data['sample_2']['set']
    if data['sample_3'] is None:
        set_list = [set_1, set_2]
        set_3 = None
        name_list = (
            data['sample_1']['name'], data['sample_2']['name'])
        color_list = (
            data['sample_1']['color'], data['sample_2']['color'])
    else:
        set_3 = data['sample_3']['set']
        set_list = [set_1, set_2, set_3]
        name_list = (
            data['sample_1']['name'], data['sample_2']['name'], data['sample_3']['name'])
        color_list = (
            data['sample_1']['color'], data['sample_2']['color'], data['sample_3']['color'])

    if set_3 is not None:
        vd = venn3(
            set_list, set_labels=name_list, set_colors=color_list, alpha=alpha, ax=ax)
        vdc = venn3_circles(set_list, linestyle='-', linewidth=3, color='grey', ax=ax)
    else:
        vd = venn2(
            set_list, set_labels=name_list, set_colors=color_list, alpha=alpha, ax=ax)
        vdc = venn2_circles(set_list, linestyle='-', linewidth=3, color='grey', ax=ax)

    vdc[1].set_ls("-.")
    if set_3 is not None:
        vdc[2].set_ls(":")

    for text in vd.set_labels:
        text.set_fontsize(fontsize)
    for text in vd.subset_labels:
        text.set_fontsize(fontsize)

    if set_3 is not None:
        _ = ax.text(
            x_loc, y_loc, r'${:3.1f} < z < {:3.1f}$'.format(data['z_low'], data['z_upp']),
            transform=ax.transAxes, fontsize=fontsize)
        _ = ax.text(
            x_loc, y_loc + 0.11,
            r'$\rm Top\ [{:d}:{:d}]$'.format(data['index_low'], data['index_upp']),
            transform=ax.transAxes, fontsize=fontsize)

    if use_ax:
        return ax
    return fig


def compare_lensing_profile(data, dsig_a, dsig_b, dsig_c, dsig_d=None, ax=None,
                            rdsig=False, x_off=0.02, y_lim=None, fontsize=25,
                            x_loc=None, y_loc=None):
    """Compare the lensing profiles from different parts of the venn diagram."""
    # Show the plot
    if ax is None:
        fig = plt.figure(figsize=(6.5, 6))
        fig.subplots_adjust(
            left=0.13, right=0.99, bottom=0.11, top=0.99)
        ax = fig.add_subplot(111)
        use_ax = False
    else:
        use_ax = True

    ax.set_xscale("log", nonpositive='clip')
    if not rdsig:
        ax.set_yscale("log", nonpositive='clip')

    # Profile 1
    try:
        err_a = dsig_a['ds_err_jk']
    except KeyError:
        err_a = dsig_a['ds_err_bt']
    if rdsig:
        y_arr = data['sample_1']['r_mpc'] * dsig_a['ds']
        y_err = data['sample_1']['r_mpc'] * err_a
    else:
        y_arr = dsig_a['ds']
        y_err = err_a
    ax.errorbar(
        data['sample_1']['r_mpc'], y_arr, yerr=y_err,
        ecolor=data['sample_1']['color'], color=data['sample_1']['color'],
        alpha=1.0, capsize=4, capthick=2.0, elinewidth=2.0,
        label='__no_label__', fmt='o', zorder=0)
    ax.scatter(
        data['sample_1']['r_mpc'], y_arr, marker=data['sample_1']['marker'],
        s=data['sample_1']['msize'], alpha=0.9,
        facecolor=data['sample_1']['color'], edgecolor='w',
        label=data['sample_1']['name'], linewidth=1.5)

    # Profile 2
    try:
        err_b = dsig_b['ds_err_jk']
    except KeyError:
        err_b = dsig_b['ds_err_bt']

    if rdsig:
        y_arr = data['sample_2']['r_mpc'] * dsig_b['ds']
        y_err = data['sample_2']['r_mpc'] * err_b
    else:
        y_arr = dsig_b['ds']
        y_err = err_b
    ax.errorbar(
        data['sample_2']['r_mpc'] * (1. + x_off),
        y_arr, yerr=y_err,
        ecolor=data['sample_2']['color'], color=data['sample_2']['color'],
        alpha=1.0, capsize=4, capthick=2.0, elinewidth=2.0,
        label='__no_label__', fmt='o', zorder=0)
    ax.scatter(
        data['sample_2']['r_mpc'] * (1. + x_off),
        y_arr, marker=data['sample_2']['marker'],
        s=data['sample_2']['msize'], alpha=0.9,
        facecolor=data['sample_2']['color'], edgecolor='w',
        label=data['sample_2']['name'], linewidth=1.5)

    # Profile 3
    if data['sample_3'] is not None:
        try:
            err_c = dsig_c['ds_err_jk']
        except KeyError:
            err_c = dsig_c['ds_err_bt']

        if rdsig:
            y_arr = data['sample_3']['r_mpc'] * dsig_c['ds']
            y_err = data['sample_3']['r_mpc'] * err_c
        else:
            y_arr = dsig_c['ds']
            y_err = err_c
        ax.errorbar(
            data['sample_3']['r_mpc'] / (1. + x_off),
            y_arr, yerr=y_err,
            ecolor=data['sample_3']['color'], color=data['sample_3']['color'],
            alpha=1.0, capsize=4, capthick=2.0, elinewidth=2.0,
            label='__no_label__', fmt='o', zorder=0)
        ax.scatter(
            data['sample_3']['r_mpc'] / (1. + x_off),
            y_arr, marker=data['sample_3']['marker'],
            s=data['sample_3']['msize'], alpha=0.9,
            facecolor=data['sample_3']['color'], edgecolor='w',
            label=data['sample_3']['name'], linewidth=1.5)
    elif dsig_c is not None:
        try:
            err_c = dsig_c['ds_err_jk']
        except KeyError:
            err_c = dsig_c['ds_err_bt']
        if rdsig:
            y_arr = data['sample_1']['r_mpc'] * dsig_c['ds']
            y_err = data['sample_1']['r_mpc'] * err_c
        else:
            y_arr = dsig_c['ds']
            y_err = err_c
        ax.errorbar(
            data['sample_1']['r_mpc'] / (1. + x_off),
            y_arr, yerr=y_err, ecolor='grey', color='grey',
            alpha=0.8, capsize=4, capthick=2.0, elinewidth=2.0,
            label='__no_label__', fmt='o', zorder=0)
        ax.scatter(
            data['sample_1']['r_mpc'] / (1. + x_off),
            y_arr, marker='P', s=180, alpha=0.9,
            facecolor='grey', edgecolor='w', label=r'$\rm Both$', linewidth=2.0)

    # Profile 4: Common Clusters
    if dsig_d is not None:
        try:
            err_d = dsig_d['ds_err_jk']
        except KeyError:
            err_d = dsig_d['ds_err_bt']

        if rdsig:
            y_arr = data['sample_1']['r_mpc'] * dsig_d['ds']
            y_err = data['sample_1']['r_mpc'] * err_d
        else:
            y_arr = dsig_d['ds']
            y_err = err_d
        ax.errorbar(
            data['sample_1']['r_mpc'] / ((1. + x_off) ** 2),
            y_arr, yerr=y_err, ecolor='grey', color='grey',
            alpha=0.8, capsize=4, capthick=2.0, elinewidth=2.0,
            label='__no_label__', fmt='o', zorder=0)
        ax.scatter(
            data['sample_1']['r_mpc'] / ((1. + x_off) ** 2),
            y_arr, marker='P', s=180, alpha=0.9,
            facecolor='grey', edgecolor='w', label=r'$\rm Overlap$', linewidth=2.0)

    _ = ax.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=30)
    if rdsig:
        _ = ax.set_ylabel(
            r'$R \times \Delta\Sigma\ [{\rm Mpc}\ M_{\odot}/\mathrm{pc}^2]$', fontsize=30)
    else:
        _ = ax.set_ylabel(
            r'$\Delta\Sigma\ [{\rm Mpc}\ M_{\odot}/\mathrm{pc}^2]$', fontsize=30)

    if rdsig:
        if x_loc is None:
            x_loc = 0.06
        if y_loc is None:
            y_loc = 0.9
        ax.legend(loc='upper left', fontsize=18)
    else:
        if x_loc is None:
            x_loc = 0.60
        if y_loc is None:
            y_loc = 0.9
        ax.legend(loc='lower left', fontsize=18)

    _ = ax.text(
        x_loc, y_loc, r'$\rm {:3.1f} < z < {:3.1f}$'.format(data['z_low'], data['z_upp']),
        transform=ax.transAxes, fontsize=fontsize)
    _ = ax.text(
        x_loc, y_loc - 0.1, r'$\rm Top\ [{:d}:{:d}]$'.format(data['index_low'], data['index_upp']),
        transform=ax.transAxes, fontsize=fontsize)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    if use_ax:
        return ax
    return fig


def gather_lensing_profiles(data, rand, n_rand=150000, n_jk=40, n_boot=200,
                            common=True, return_lens=False):
    """Get the lensing profiles from different parts of the venn diagram."""
    if data['sample_3'] is not None:
        # Galaxies only in Sample 1
        index_a = list(
            (data['sample_1']['set'] - data['sample_2']['set']) - data['sample_3']['set'])
        use_a = data['sample_1']['use']
        lens_a = vstack([use_a[use_a['index'] == idx] for idx in index_a],
                        metadata_conflicts='silent')
        lens_a.meta = data['sample_1']['data'].meta

        if len(lens_a) <= n_jk + 1:
            dsig_a = wlensing.stack_dsigma_profile(
                lens_a, rand, n_rand=n_rand, bootstrap=True, jackknife=False, n_boot=n_boot)
        else:
            dsig_a = wlensing.stack_dsigma_profile(
                lens_a, rand, n_rand=n_rand, bootstrap=False, n_jk=n_jk)

        # Galaxies only in Sample 2
        index_b = list(
            (data['sample_2']['set'] - data['sample_1']['set']) - data['sample_3']['set'])
        use_b = data['sample_2']['use']
        lens_b = vstack([use_b[use_b['index'] == idx] for idx in index_b],
                        metadata_conflicts='silent')
        lens_b.meta = data['sample_1']['data'].meta

        if len(lens_b) <= n_jk + 1:
            dsig_b = wlensing.stack_dsigma_profile(
                lens_b, rand, n_rand=n_rand, bootstrap=True, jackknife=False, n_boot=n_boot)
        else:
            dsig_b = wlensing.stack_dsigma_profile(
                lens_b, rand, n_rand=n_rand, bootstrap=False, n_jk=n_jk)

        # Galaxies only in Sample 3
        index_c = list(
            (data['sample_3']['set'] - data['sample_2']['set']) - data['sample_1']['set'])
        use_c = data['sample_3']['use']
        lens_c = vstack([use_c[use_c['index'] == idx] for idx in index_c],
                        metadata_conflicts='silent')
        lens_c.meta = data['sample_1']['data'].meta

        if len(lens_c) <= n_jk + 1:
            dsig_c = wlensing.stack_dsigma_profile(
                lens_c, rand, n_rand=n_rand, bootstrap=True, jackknife=False, n_boot=n_boot)
        else:
            dsig_c = wlensing.stack_dsigma_profile(
                lens_c, rand, n_rand=n_rand, bootstrap=False, n_jk=n_jk)

        # Galxies in either Sample 1, 2, or 3
        if common:
            index_d_1 = list(
                (data['sample_1']['set'] & data['sample_2']['set']) |
                (data['sample_1']['set'] & data['sample_3']['set']))
            use_d_1 = data['sample_1']['use']
            lens_d_1 = vstack([use_d_1[use_d_1['index'] == idx] for idx in index_d_1])

            index_d_2 = list(
                (data['sample_2']['set'] & data['sample_3']['set']) -
                data['sample_1']['set'])
            use_d_2 = data['sample_2']['use']
            lens_d_2 = vstack([use_d_2[use_d_2['index'] == idx] for idx in index_d_2])

            lens_d = vstack([lens_d_1, lens_d_2])
            lens_d.meta = data['sample_1']['data'].meta

            if len(lens_d) <= n_jk + 1:
                dsig_d = wlensing.stack_dsigma_profile(
                    lens_d, rand, n_rand=n_rand, bootstrap=True, jackknife=False, n_boot=500)
            else:
                dsig_d = wlensing.stack_dsigma_profile(
                    lens_d, rand, n_rand=n_rand, bootstrap=False, n_jk=n_jk)
        else:
            lens_d, dsig_d = None, None

        if return_lens:
            return dsig_a, dsig_b, dsig_c, dsig_d, lens_a, lens_b, lens_c, lens_d
        return dsig_a, dsig_b, dsig_c, dsig_d
    else:
        # Galaxies only in Sample 1
        index_a = list(data['sample_1']['set'] - data['sample_2']['set'])
        use_a = data['sample_1']['use']
        lens_a = vstack([use_a[use_a['index'] == idx] for idx in index_a],
                        metadata_conflicts='silent')
        lens_a.meta = data['sample_1']['data'].meta

        if len(lens_a) <= n_jk + 1:
            dsig_a = wlensing.stack_dsigma_profile(
                lens_a, rand, n_rand=n_rand, bootstrap=True, jackknife=False, n_boot=500)
        else:
            dsig_a = wlensing.stack_dsigma_profile(
                lens_a, rand, n_rand=n_rand, bootstrap=False, n_jk=n_jk)

        # Galaxies only in Sample 2
        index_b = list(data['sample_2']['set'] - data['sample_1']['set'])
        use_b = data['sample_2']['use']
        lens_b = vstack([use_b[use_b['index'] == idx] for idx in index_b],
                        metadata_conflicts='silent')
        lens_b.meta = data['sample_1']['data'].meta

        if len(lens_b) <= n_jk + 1:
            dsig_b = wlensing.stack_dsigma_profile(
                lens_b, rand, n_rand=n_rand, bootstrap=True, jackknife=False, n_boot=500)
        else:
            dsig_b = wlensing.stack_dsigma_profile(
                lens_b, rand, n_rand=n_rand, bootstrap=False, n_jk=n_jk)

        if common:
            # Galaxies in Sample 1 & 2
            index_c = list(data['sample_1']['set'] & data['sample_2']['set'])
            use_c = data['sample_1']['use']
            lens_c = vstack([use_c[use_c['index'] == idx] for idx in index_c],
                            metadata_conflicts='silent')
            lens_c.meta = data['sample_1']['data'].meta

            if len(lens_c) <= n_jk + 1:
                dsig_c = wlensing.stack_dsigma_profile(
                    lens_c, rand, n_rand=n_rand, bootstrap=True, jackknife=False, n_boot=500)
            else:
                dsig_c = wlensing.stack_dsigma_profile(
                    lens_c, rand, n_rand=n_rand, bootstrap=False, n_jk=n_jk)
        else:
            lens_c, dsig_c = None, None

        if return_lens:
            return dsig_a, dsig_b, dsig_c, lens_a, lens_b, lens_c
        return dsig_a, dsig_b, dsig_c
