#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions related to visualization."""

from sys import platform
from itertools import cycle

import numpy as np

from scipy import interpolate
from scipy.stats import binned_statistic_2d
from scipy.ndimage.filters import gaussian_filter

import palettable

if platform == "linux" or platform == 'linux2':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import catalog

__all__ = ['sum_plot_topn', 'summary_plot_dsig', 'summary_plot_r_dsig',
           'sum_plot_chi2_curve', 'sample_over_mass_plane', 'compare_dsigma_profiles',
           'show_split_result']

color_bins = palettable.colorbrewer.qualitative.Set1_4.mpl_colors
marker_bins = ['o', 's', 'h', '8', '+']
msize_bins = [100, 100, 130, 110, 150]


def sum_plot_topn(sum_tab, label, note=None, ref_tab=None, cov_type='jk', show_bin=True):
    """Make a summary plot of the TopN result."""
    n_col, n_bins = 3, len(sum_tab)
    left, right = 0.06, 0.99
    bottom, top = 0.05, 0.965
    x_space = 0.07
    x_size = (right - left - x_space * 1.05) / n_col
    y_size = (top - bottom) / n_bins

    r_mpc = sum_tab.meta['r_mpc']

    fig = plt.figure(figsize=(n_col * 6, n_bins * 5))

    for ii, sum_bin in enumerate(sum_tab):
        bin_id = sum_bin['bin_id']
        # Setup the three columns
        ax1 = fig.add_axes([left, top - y_size * bin_id, x_size, y_size])
        ax2 = fig.add_axes([left + x_space + x_size, top - y_size * bin_id, x_size, y_size])
        ax3 = fig.add_axes([left + x_space + x_size * 2.02, top - y_size * bin_id, x_size, y_size])

        if ref_tab is None:
            ref_dsigma, ref_sig = None, None
        else:
            ref_dsigma = ref_tab[ii]
            ref_sig = ref_tab[ii]['sig_med_' + cov_type]

        # Subplot title
        if ii == 0:
            ax1.set_title(label, fontsize=38, pad=18)
            if note is not None:
                ax2.set_title(note, fontsize=38, pad=18)
            ax3.set_title(r'${\rm Reduced}\ \chi^2$', fontsize=38, pad=18)

        ax1 = summary_plot_dsig(
            ii, sum_bin, r_mpc, ax=ax1, cov_type=cov_type, label=None,
            xlabel=(ii == len(sum_tab) - 1), ylabel=True, legend=True, show_bin=show_bin,
            ref_dsigma=ref_dsigma)

        ax2 = summary_plot_r_dsig(
            ii, sum_bin, r_mpc, ax=ax2, cov_type=cov_type, label=None,
            xlabel=(ii == len(sum_tab) - 1), ylabel=True, legend=False, show_bin=False,
            ref_dsigma=ref_dsigma)

        ax3 = sum_plot_chi2_curve(
            ii, sum_bin, r_mpc, ax=ax3, cov_type=cov_type,
            ref_sig=ref_sig, xlabel=(ii == len(sum_tab) - 1), ylabel=False,
            show_bin=False)

    return fig


def summary_plot_dsig(bin_num, sum_bin, r_mpc, ax=None, cov_type='bt', label=None,
                      xlabel=True, ylabel=True, legend=True, show_bin=True,
                      ref_dsigma=None):
    """Plot the DeltaSigma profile."""
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        fig.subplots_adjust(
            left=0.16, bottom=0.13, right=0.99, top=0.99, wspace=None, hspace=None)
        ax = fig.add_subplot(111)

    ax.grid(linestyle='--', linewidth=2, alpha=0.4, zorder=0)
    ax.set_xscale("log", nonpositive='clip')
    ax.set_yscale("log", nonpositive='clip')

    if sum_bin['sim_med_' + cov_type] == 'mdpl2':
        error_factor = 5.
    elif sum_bin['sim_med_' + cov_type] == 'smdpl':
        error_factor = 2.
    else:
        raise ValueError('! Wrong Simulation: [mdpl2 | smdpl]')

    # Sigma=0.0 profile
    err_low = sum_bin['dsigma_sig0_' + cov_type] - sum_bin['dsigma_sig0_low_' + cov_type]
    err_upp = sum_bin['dsigma_sig0_upp_' + cov_type] - sum_bin['dsigma_sig0_' + cov_type]
    ax.fill_between(
        r_mpc, sum_bin['dsigma_sig0_' + cov_type] - err_low * error_factor,
        sum_bin['dsigma_sig0_' + cov_type] + err_upp * error_factor,
        alpha=0.3, edgecolor='none', linewidth=1.0, 
        label=r'$\sigma_{\mathcal{M}|\mathcal{O}}=0.00$',
        facecolor='teal', rasterized=True)

    # Best-fit profile
    err_low = sum_bin['dsigma_mod_' + cov_type] - sum_bin['dsigma_mod_low_' + cov_type]
    err_upp = sum_bin['dsigma_mod_upp_' + cov_type] - sum_bin['dsigma_mod_' + cov_type]
    ax.fill_between(
        r_mpc, sum_bin['dsigma_mod_' + cov_type] - err_low * error_factor,
        sum_bin['dsigma_mod_' + cov_type] + err_low * error_factor,
        alpha=0.3, edgecolor='none', linewidth=1.0,
        label=(r'$\sigma_{\mathcal{M}|\mathcal{O}}=\ $' + 
               r'${:4.2f}$'.format(sum_bin['sig_med_' + cov_type])),
        facecolor='grey', linestyle='--', rasterized=True)

    # Data profile
    ax.errorbar(r_mpc, sum_bin['dsigma'], yerr=sum_bin['dsig_err_' + cov_type],
                ecolor=color_bins[bin_num], color=color_bins[bin_num], alpha=0.7,
                capsize=4, capthick=2.0, elinewidth=2.0, label='__no_label__',
                fmt='o', zorder=0)
    ax.scatter(r_mpc, sum_bin['dsigma'],
               s=180, alpha=0.8, facecolor=color_bins[bin_num], edgecolor='k',
               linewidth=2.0, label=label)

    # Reference profile
    if ref_dsigma is not None:
        ax.errorbar(ref_dsigma.meta['r_mpc'], ref_dsigma['dsigma'], yerr=ref_dsigma['dsig_err_' + cov_type],
                    ecolor='grey', color='grey', alpha=0.5,
                    capsize=4, capthick=2.0, elinewidth=2.0, label='__no_label__',
                    fmt='o', zorder=0)
        ax.scatter(
            ref_dsigma.meta['r_mpc'], ref_dsigma['dsigma'], marker='X', s=160, linewidth=2.0,
            facecolor='none', edgecolor='k', alpha=0.7, label=r'$\rm Reference$')

    if legend:
        ax.legend(loc='lower left', fontsize=24, handletextpad=0.2, borderpad=0.2)

    y_min = np.min(sum_bin['dsigma_mod_' + cov_type] - err_low * error_factor) * 0.5
    y_max = np.max(sum_bin['dsigma_sig0_' + cov_type] + err_low * error_factor) * 1.2
    ax.set_ylim(y_min, y_max)

    if show_bin:
        _ = ax.text(0.74, 0.87, r'$\rm Bin\ {:1d}$'.format(bin_num + 1), fontsize=35,
                    transform=ax.transAxes)

    if xlabel:
        _ = ax.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=30)
    else:
        _ = ax.set_xticklabels([])
    if ylabel:
        _ = ax.set_ylabel(r'$\Delta\Sigma\ [M_{\odot}/\mathrm{pc}^2]$', fontsize=30)
    else:
        _ = ax.set_yticklabels([])

    if ax is None:
        return fig
    return ax


def summary_plot_r_dsig(bin_num, sum_bin, r_mpc, ax=None, cov_type='bt', label=None,
                        xlabel=True, ylabel=True, legend=True, show_bin=True,
                        ref_dsigma=None):
    """Plot the R x DeltaSigma profile."""
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        fig.subplots_adjust(
            left=0.16, bottom=0.13, right=0.99, top=0.99, wspace=None, hspace=None)
        ax = fig.add_subplot(111)

    ax.grid(linestyle='--', linewidth=2, alpha=0.4, zorder=0)
    ax.set_xscale("log", nonpositive='clip')

    if sum_bin['sim_med_' + cov_type] == 'mdpl2':
        error_factor = 5.
    elif sum_bin['sim_med_' + cov_type] == 'smdpl':
        # TODO: Need to check this
        error_factor = 2.
    else:
        raise ValueError('! Wrong Simulation: [mdpl2 | smdpl]')

    # Sigma=0.0 profile
    err_low = sum_bin['dsigma_sig0_' + cov_type] - sum_bin['dsigma_sig0_low_' + cov_type]
    err_upp = sum_bin['dsigma_sig0_upp_' + cov_type] - sum_bin['dsigma_sig0_' + cov_type]
    ax.fill_between(
        r_mpc,
        r_mpc * (sum_bin['dsigma_sig0_' + cov_type] - err_low * error_factor),
        r_mpc * (sum_bin['dsigma_sig0_' + cov_type] + err_upp * error_factor),
        alpha=0.3, edgecolor='none', linewidth=1.0, label=r'$\sigma=0.0$',
        facecolor='teal', rasterized=True)

    # Best-fit profile
    err_low = sum_bin['dsigma_mod_' + cov_type] - sum_bin['dsigma_mod_low_' + cov_type]
    err_upp = sum_bin['dsigma_mod_upp_' + cov_type] - sum_bin['dsigma_mod_' + cov_type]
    ax.fill_between(
        r_mpc,
        r_mpc * (sum_bin['dsigma_mod_' + cov_type] - err_low * error_factor),
        r_mpc * (sum_bin['dsigma_mod_' + cov_type] + err_low * error_factor),
        alpha=0.3, edgecolor='none', linewidth=1.0,
        label=r'$\sigma={:4.2f}$'.format(sum_bin['sig_med_' + cov_type]),
        facecolor='grey', linestyle='--', rasterized=True)

    # Data profile
    ax.errorbar(r_mpc, r_mpc * sum_bin['dsigma'],
                yerr=(r_mpc * sum_bin['dsig_err_' + cov_type]),
                ecolor=color_bins[bin_num], color=color_bins[bin_num], alpha=0.7,
                capsize=4, capthick=2.0, elinewidth=2.0, label='__no_label__',
                fmt='o', zorder=0)
    ax.scatter(r_mpc, r_mpc * sum_bin['dsigma'],
               s=180, alpha=0.8, facecolor=color_bins[bin_num], edgecolor='k',
               linewidth=2.0, label=label)

    # Reference profile
    if ref_dsigma is not None:
        r_mpc_ref = ref_dsigma.meta['r_mpc']
        ax.errorbar(r_mpc_ref, r_mpc_ref * ref_dsigma['dsigma'],
                    yerr=(r_mpc_ref * ref_dsigma['dsig_err_' + cov_type]),
                    ecolor='grey', color='grey', alpha=0.5,
                    capsize=4, capthick=2.0, elinewidth=2.0, label='__no_label__',
                    fmt='o', zorder=0)
        ax.scatter(
            r_mpc_ref, r_mpc_ref * ref_dsigma['dsigma'], marker='X', s=160, linewidth=2.0,
            facecolor='none', edgecolor='k', alpha=0.7, label=r'$\rm Reference$')

    if legend:
        ax.legend(loc='lower left', fontsize=22)

    y_min = np.min(r_mpc * sum_bin['dsigma_mod_low_' + cov_type]) * 0.5
    y_max = np.max(r_mpc * sum_bin['dsigma_sig0_upp_' + cov_type]) * 1.05
    ax.set_ylim(y_min, y_max)

    if show_bin:
        _ = ax.text(0.74, 0.87, r'$\rm Bin\ {:1d}$'.format(bin_num + 1), fontsize=35,
                    transform=ax.transAxes)

    if xlabel:
        _ = ax.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=30)
    else:
        _ = ax.set_xticklabels([])
    if ylabel:
        _ = ax.set_ylabel(
            r'$R \times \Delta\Sigma\ [{\rm Mpc}\ M_{\odot}/\mathrm{pc}^2]$', fontsize=30)
    else:
        _ = ax.set_yticklabels([])

    if ax is None:
        return fig
    return ax


def sum_plot_chi2_curve(bin_num, sum_bin, r_mpc, ax=None, cov_type='bt', label=None,
                        xlabel=True, ylabel=True, show_bin=True, ref_sig=None):
    """Plot the chi2 curve."""
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        fig.subplots_adjust(
            left=0.165, bottom=0.13, right=0.995, top=0.99, wspace=None, hspace=None)
        ax = fig.add_subplot(111)

    ax.axhline(1.0, linewidth=3.0, alpha=.4, c='k')

    # Reduced chi2 curves
    rchi2 = sum_bin['chi2_' + cov_type] / (len(sum_bin['dsigma']) - 1)

    # Best-fit scatter and its uncertainty
    ax.axvline(sum_bin['sig_med_' + cov_type], linewidth=2.0, alpha=0.4,
               linestyle='--', color='k')
    ax.fill_between(
        [sum_bin['sig_low_' + cov_type], sum_bin['sig_upp_' + cov_type]],
        [0, 0], [np.max(rchi2) * 1.2, np.max(rchi2) * 1.2],
        color=color_bins[bin_num], alpha=0.2)

    if ref_sig is not None:
        ax.axvline(ref_sig, linewidth=3.0, alpha=0.5, linestyle='-.', color='k')

    # Reduced chi2 curves
    sims = sum_bin['simulation']
    markers = cycle(['o', 's', 'h', '8', '+'])
    for sim in np.unique(sims):
        mask = sims == sim
        ax.scatter(
            sum_bin['sigma'][mask], rchi2[mask], marker=next(markers),
            s=60, alpha=0.8, facecolor=color_bins[bin_num], edgecolor='grey',
            linewidth=1.0, label=label)

    ax.scatter(sum_bin['sigma'][sum_bin['idx_med_' + cov_type]],
               rchi2[sum_bin['idx_med_' + cov_type]], marker='o',
               s=100, alpha=1.0, facecolor=color_bins[bin_num], edgecolor='k',
               linewidth=1.0, label=r'__no_label__')

    ax.set_xlim(0.00, np.max(sum_bin['sigma']) * 1.09)
    ax.set_ylim(0.01, np.max(rchi2) * 1.19)

    sig_best = sum_bin['sig_med_' + cov_type]
    sig_upp = sum_bin['sig_upp_' + cov_type]
    sig_low = sum_bin['sig_low_' + cov_type]
    if sig_best <= 0.65:
        _ = ax.text(
            sig_best + 0.05, np.max(rchi2) * 0.95,
            r'$\sigma={:4.2f}^{{+{:4.2f}}}_{{-{:4.2f}}}$'.format(
                sig_best, sig_upp - sig_best, sig_best - sig_low), fontsize=25)
    else:
        _ = ax.text(
            sig_best - 0.45, np.max(rchi2) * 0.95,
            r'$\sigma={:4.2f}^{{+{:4.2f}}}_{{-{:4.2f}}}$'.format(
                sig_best, sig_upp - sig_best, sig_best - sig_low), fontsize=25)

    if show_bin:
        _ = ax.text(0.07, 0.87, r'$\rm Bin\ {:1d}$'.format(bin_num + 1), fontsize=35,
                    transform=ax.transAxes)

    if xlabel:
        _ = ax.set_xlabel(r'$\sigma_{\mathcal{M} | \mathcal{O}}$', fontsize=30)
    else:
        _ = ax.set_xticklabels([])
    if ylabel:
        _ = ax.set_ylabel(r'$\rm Reduced\ \chi^2$', fontsize=30)
    else:
        _ = ax.set_yticklabels([])

    if ax is None:
        return fig
    return ax


def sample_over_mass_plane(data, mask=None, cmap='OrRd', xlim=None, ylim=None,
                           mass_x='logm_100', mass_y='logm_10', mass_z='logmh_vir_plane',
                           mvir_countour=False, count_contour=False,
                           bins=(35, 40), mask_list=None, hist=False, scatters=False,
                           c_levels=[5, 50, 100, 300], figsize=(7, 7),
                           z_levels=[13.2, 13.5, 13.8, 14.0, 14.2], fontsize=30,
                           label_list=None, marker_list=None, color_list=None,
                           xlabel=r'$\log M_{\star,\ \mathrm{100,\ kpc}}$',
                           ylabel=r'$\log M_{\star,\ \mathrm{10,\ kpc}}$',
                           zlabel=r'$\log M_{\rm Vir}$', ax=None):
    """Making plot"""
    # Making a plot
    if ax is None:
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(left=0.17, bottom=0.13, right=0.99, top=0.99, wspace=None, hspace=None)
        ax = fig.add_subplot(111)

    ax.grid(linestyle='--', linewidth=2, alpha=0.4, zorder=0)
    ax.axvline(11.50, linestyle='-.', linewidth=4.0, c='grey', alpha=0.7)

    # Color density plot that indicates the change of Mvir
    mask_finite = (np.isfinite(data[mass_x]) & np.isfinite(data[mass_y]) &
                   np.isfinite(data[mass_z]))
    if mask is not None:
        mask_use = mask_finite & mask
    else:
        mask_use = mask_finite

    if mask_use.sum() != len(data):
        print("# {:d}/{:d} objects are shown on the figure".format(mask_use.sum(), len(data)))

    x_arr = data[mask_use][mass_x]
    y_arr = data[mask_use][mass_y]
    z_arr = data[mask_use][mass_z]

    z_stats, x_edges, y_edges, _ = binned_statistic_2d(
        x_arr, y_arr, z_arr, np.nanmean, bins=bins)
    z_counts, _, _, _ = binned_statistic_2d(
        x_arr, y_arr, z_arr, 'count', bins=bins)

    HM = ax.imshow(z_stats.T, origin='lower', cmap=cmap,
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   vmin=12.5, vmax=14.80, aspect='auto', interpolation='nearest',
                   alpha=0.4)

    # Contours for iso-Mvir curves
    if count_contour:
        CT2 = ax.contour(x_edges[:-1], y_edges[:-1], gaussian_filter(z_counts.T, 0.1),
                         5, linewidths=2.0, linestyles='dashed', colors='grey', alpha=0.5,
                         levels=c_levels, extend='neither')

    if mvir_countour:
        CT = ax.contour(x_edges[:-1], y_edges[:-1], gaussian_filter(z_stats.T, 0.1),
                        4, linewidths=1.0, colors='k', alpha=0.5,
                        levels=z_levels, extend='neither')
        _ = ax.clabel(CT, inline=1, fontsize=20, colors='k', alpha=0.8, fmt=r'$%4.1f$')

    # Subsamples
    if mask_list is not None:
        proxy_list = []
        for j, mask in enumerate(mask_list):
            if color_list is None:
                color = palettable.cartocolors.qualitative.Bold_10_r.mpl_colors[j]
            else:
                color = color_list[j]

            if label_list is None:
                lab = '__no_label__'
            else:
                lab = label_list[j]

            if marker_list is None:
                marker_cycle = cycle(('p', 's', 'h', 'H', 'P', '+'))
                marker = next(marker_cycle)
            else:
                marker = marker_list[j]

            proxy_list.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.6))

            mask_bin = mask & mask_use
            if mask_bin.sum() != mask.sum():
                print("# {:d}/{:d} objects are shown for bin {:d}".format(
                    mask_bin.sum(), mask.sum(), j + 1))

            if scatters:
                ax.scatter(data[mask_bin][mass_x], data[mask_bin][mass_y], marker=marker,
                           facecolor=color, edgecolor='none', alpha=0.6, s=30, label=lab)

            sub_counts, x_edges, y_edges, _ = binned_statistic_2d(
                data[mask_bin][mass_x], data[mask_bin][mass_y], data[mask_bin][mass_z],
                'count', bins=(15, 15))

            if mask_bin.sum() < 100:
                _ = ax.contour(x_edges[:-1], y_edges[:-1], gaussian_filter(sub_counts.T, 1.5),
                               3, linewidths=4.0, colors=color, alpha=0.8,
                               extend='neither')
            else:
                _ = ax.contour(x_edges[:-1], y_edges[:-1], gaussian_filter(sub_counts.T, 1.5),
                               2, linewidths=4.0, colors=color, alpha=0.8,
                               extend='neither')

        # Show contours in legend
        if label_list is not None:
            ax.legend(proxy_list, label_list, loc='lower right', fontsize=25)

    # Histogram of halo mass
    if hist:
        # Colorbar
        cax = fig.add_axes([0.60, 0.46, 0.35, 0.04])
        cbar = plt.colorbar(HM, cax=cax, orientation='horizontal')
        cbar.solids.set_edgecolor("face")

        hax = fig.add_axes([0.60, 0.28, 0.35, 0.18])
        hax.grid(alpha=0.5, linestyle='--', linewidth=2.0)
        _ = hax.hist(z_arr, density=True, bins=8, histtype='stepfilled', facecolor='gray',
                     alpha=0.7)
        hax.set_yscale("log", nonposy='clip')
        hax.yaxis.set_ticklabels([])
        hax.set_xlabel(zlabel, fontsize=28)

    if xlim:
        _ = ax.set_xlim(xlim)
    if ylim:
        _ = ax.set_ylim(ylim)

    _ = ax.set_xlabel(xlabel, fontsize=fontsize)
    _ = ax.set_ylabel(ylabel, fontsize=fontsize)

    if ax is None:
        return fig
    return ax

def compare_dsigma_profiles(dsig_ref, dsig_cmp, sim_dsig, sim_mhalo, sig_type='bt',
                            compare_to_model=True, use_ref_range=False,
                            label_ref=r'$\rm Ref$', label_cmp=r'$\rm Test$',
                            sub_ref=r'{\rm Ref}', sub_cmp=r'{\rm Test}',
                            cmap_list=None, color_bins=None, show_stats=True,
                            marker_ref='o', msize_ref=150,
                            marker_cmp='P', msize_cmp=180, ratio_range=(0.01, 1.59),
                            show_best_ref=False, show_best_cmp=True, logmh_range=None):
    """Compare the Dsigma profiles."""

    def get_dsig_ratio(obs, ref, mod=None):
        """"""
        obs_rand = np.random.normal(
            loc=obs['dsigma'][0], scale=obs['dsig_err_{:s}'.format(sig_type)][0])

        if mod is not None:
            ref_rand = np.random.normal(
                loc=mod['dsig'], scale=(mod['dsig_err'] * err_factor))

            ref_inter = 10.0 ** (
                interpolate.interp1d(
                    mod['r_mpc'], np.log10(ref_rand), fill_value='extrapolate')(r_mpc_obs)
            )
            return obs_rand / ref_inter
        else:
            ref_rand = np.random.normal(
                loc=ref['dsigma'][0], scale=obs['dsig_err_{:s}'.format(sig_type)][0])
            return obs_rand / ref_rand

    chi2_cmp = np.min(dsig_cmp['chi2_jk'], axis=1)
    chi2_ref = np.min(dsig_ref['chi2_jk'], axis=1)
    sig_cmp = dsig_cmp['sig_med_jk']
    sig_ref = dsig_ref['sig_med_jk']

    # Color maps and bins
    if cmap_list is None:
        cmap_list = [
            palettable.colorbrewer.sequential.Blues_7_r,
            palettable.colorbrewer.sequential.OrRd_7_r,
            palettable.colorbrewer.sequential.YlGn_7_r,
            palettable.colorbrewer.sequential.Purples_7_r]

    if color_bins is None:
        color_bins = ["#377eb8", "#e41a1c", "#1b9e77", "#984ea3"]

    # Radius bin of the observed DSigma profiles
    r_mpc_obs = dsig_ref.meta['r_mpc']

    # ---- Start the figure ---- #
    # Setup the figure
    n_col, n_bins = 3, len(dsig_ref)
    fig_y = int(4 * n_bins + 2)
    left, right = 0.07, 0.98
    if n_bins == 4:
        bottom, top = 0.054, 0.96
    elif n_bins == 3:
        bottom, top = 0.07, 0.94
    elif n_bins == 2:
        bottom, top = 0.11, 0.92

    x_space = 0.08
    x_size = (right - left - x_space * 1.05) / n_col
    y_size = (top - bottom) / n_bins

    fig = plt.figure(figsize=(16, fig_y))

    for bin_id in np.arange(len(dsig_ref)) + 1:
        # Setup the three columns
        ax1 = fig.add_axes([left, top - y_size * bin_id, x_size, y_size])
        ax2 = fig.add_axes([left + x_space + x_size, top - y_size * bin_id, x_size, y_size])
        ax3 = fig.add_axes([left + x_space + x_size * 2.04, top - y_size * bin_id, x_size, y_size])

        # Subplot title
        if bin_id == 1:
            ax1.set_title(r'$R \times \Delta\Sigma\ \rm Profile$', fontsize=38, pad=18)
            ax2.set_title(r'$\Delta\Sigma_{:s}/\Delta\Sigma_{:s}$'.format(
                sub_cmp, sub_ref), fontsize=38, pad=18)
            ax3.set_title(r'$M_{\rm vir}\ \rm Distribution$', fontsize=38, pad=18)

        # Color map
        cmap, color = cmap_list[bin_id - 1], color_bins[bin_id - 1]

        # MDPL halo mass information for this bin
        sim_dsig_bin = sim_dsig[sim_dsig['bin'] == bin_id - 1]
        sim_mhalo_bin = sim_mhalo[sim_mhalo['number_density_bin'] == bin_id - 1]

        # DSigma result for this bin
        dsig_ref_bin = dsig_ref[dsig_ref['bin_id'] == bin_id]
        dsig_cmp_bin = dsig_cmp[dsig_cmp['bin_id'] == bin_id]

        # Best fit DSigma profiles
        dsig_ref_best = sim_dsig_bin[
            np.argmin(
                np.abs(sim_dsig_bin['scatter'] - dsig_ref_bin['sig_med_{:s}'.format(sig_type)]))]
        dsig_cmp_best = sim_dsig_bin[
            np.argmin(
                np.abs(sim_dsig_bin['scatter'] - dsig_cmp_bin['sig_med_{:s}'.format(sig_type)]))]

        if dsig_ref_bin['sig_med_{:s}'.format(sig_type)] < 0.6:
            err_factor = 5.
        else:
            err_factor = 4.

        # Interpolated the reference model profile
        ref_model_inter = 10.0 ** (
            interpolate.interp1d(
                dsig_ref_best['r_mpc'], np.log10(dsig_ref_best['dsig']),
                fill_value='extrapolate')(r_mpc_obs)
        )

        if compare_to_model:
            ratio_sample = [
                get_dsig_ratio(
                    dsig_cmp_bin, dsig_ref_bin, mod=dsig_ref_best) for i in np.arange(2000)]
            ratio_cmp = dsig_cmp_bin['dsigma'][0] / ref_model_inter
        else:
            ratio_sample = [
                get_dsig_ratio(dsig_cmp_bin, dsig_ref_bin, mod=None) for i in np.arange(2000)]
            ratio_cmp = dsig_cmp_bin['dsigma'][0] / dsig_ref_bin['dsigma'][0]

        ratio_cmp_err_low = ratio_cmp - np.nanpercentile(ratio_sample, 16, axis=0)
        ratio_cmp_err_upp = np.nanpercentile(ratio_sample, 84, axis=0) - ratio_cmp

        # Halo mass distribution from model
        if bin_id == 1:
            n_bins = 8
        elif bin_id == 2:
            n_bins = 4
        else:
            n_bins = 3

        mvir_true, hist_true, mvir_avg_true = catalog.rebin_mhalo_hist(
            sim_mhalo_bin, bin_id - 1, 0.0, n_bin=n_bins)

        mvir_ref, hist_ref, mvir_avg_ref = catalog.rebin_mhalo_hist(
            sim_mhalo_bin, bin_id - 1, dsig_ref_bin['sig_med_{:s}'.format(sig_type)], n_bin=20)
        mvir_cmp, hist_cmp, mvir_avg_cmp = catalog.rebin_mhalo_hist(
            sim_mhalo_bin, bin_id - 1, dsig_cmp_bin['sig_med_{:s}'.format(sig_type)], n_bin=20)

        # ----- Plot 1: R x DSigma plot ----- #
        ax1.set_xscale("log", nonpositive='clip')

        # MDPL: Best-fit
        if show_best_cmp:
            ax1.fill_between(
                dsig_cmp_best['r_mpc'],
                dsig_cmp_best['r_mpc'] * (
                    dsig_cmp_best['dsig'] - dsig_cmp_best['dsig_err'] * err_factor),
                dsig_cmp_best['r_mpc'] * (
                    dsig_cmp_best['dsig'] + dsig_cmp_best['dsig_err'] * err_factor),
                alpha=0.2, edgecolor='grey', linewidth=2.0,
                label=r'__no_label__', facecolor='grey', linestyle='-', rasterized=True)

        if show_best_ref:
            ax1.fill_between(
                dsig_ref_best['r_mpc'],
                dsig_ref_best['r_mpc'] * (
                    dsig_ref_best['dsig'] - dsig_ref_best['dsig_err'] * err_factor),
                dsig_ref_best['r_mpc'] * (
                    dsig_ref_best['dsig'] + dsig_ref_best['dsig_err'] * err_factor),
                alpha=0.15, edgecolor='grey', linewidth=2.0,
                label=r'__no_label__', facecolor='grey', linestyle='--', rasterized=True)

        # Reference DSigma profile
        ax1.errorbar(
            r_mpc_obs,
            r_mpc_obs * dsig_ref_bin['dsigma'][0],
            yerr=(r_mpc_obs * dsig_ref_bin['dsig_err_{:s}'.format(sig_type)][0]),
            ecolor=cmap.mpl_colormap(0.5), color=cmap.mpl_colormap(0.5), alpha=0.9, capsize=4,
            capthick=2.5, elinewidth=2.5, label='__no_label__', fmt='o', zorder=0)
        ax1.scatter(
            r_mpc_obs,
            r_mpc_obs * dsig_ref_bin['dsigma'][0],
            s=msize_ref, alpha=0.9, facecolor='w', edgecolor=cmap.mpl_colormap(0.5), 
            marker=marker_ref, linewidth=2.5, label=label_ref)

        # DSigma profiles to compare with
        ax1.errorbar(
            r_mpc_obs * 1.01,
            r_mpc_obs * dsig_cmp_bin['dsigma'][0],
            yerr=(r_mpc_obs * dsig_cmp_bin['dsig_err_{:s}'.format(sig_type)][0]),
            ecolor=color, color='w', alpha=0.9, capsize=4, capthick=2.5, elinewidth=2.5,
            label='__no_label__', fmt='o', zorder=0)
        ax1.scatter(
            r_mpc_obs * 1.01,
            r_mpc_obs * dsig_cmp_bin['dsigma'][0],
            s=msize_cmp, alpha=0.9, facecolor=color, edgecolor='w', marker=marker_cmp,
            linewidth=3.0, label=label_cmp)

        if not use_ref_range:
            ax1.set_ylim(1.0, np.max(dsig_cmp_best['r_mpc'] * dsig_cmp_best['dsig']) * 1.45)
        else:
            ax1.set_ylim(1.0, np.max(dsig_ref_best['r_mpc'] * dsig_ref_best['dsig']) * 1.45)

        # Bin ID
        _ = ax1.text(
            0.08, 0.83, r'$\rm Bin\ {:1d}$'.format(bin_id), fontsize=35, transform=ax1.transAxes)

        if bin_id == 4:
            _ = ax1.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=30)
            ax1.legend(loc='best', fontsize=20)
        else:
            ax1.set_xticklabels([])
        _ = ax1.set_ylabel(r'$R \times \Delta\Sigma\ [10^{6}\ M_{\odot}/\mathrm{pc}]$', fontsize=31)


        # ----- Plot 2: Ratio of DSigma plot ----- #
        ax2.set_xscale("log", nonpositive='clip')

        ax2.axhline(
            1.0, linewidth=3.0, alpha=0.5, color='k', linestyle='--', label='__no_label__', )

        # Uncertainty of the model
        ax2.fill_between(
            dsig_ref_best['r_mpc'],
            1.0 - (dsig_ref_best['dsig_err'] * err_factor / dsig_ref_best['dsig']),
            1.0 + (dsig_ref_best['dsig_err'] * err_factor / dsig_ref_best['dsig']),
            alpha=0.2, edgecolor='none', linewidth=1.0, label='__no_label__',
            facecolor='grey', rasterized=True)

        ax2.errorbar(
            r_mpc_obs, ratio_cmp, yerr=[ratio_cmp_err_low, ratio_cmp_err_upp],
            ecolor=cmap.mpl_colormap(0.5), color='w', alpha=0.8, capsize=4, capthick=2.5,
            elinewidth=3.0, label='__no_label__', fmt='o', zorder=0)
        ax2.scatter(
            r_mpc_obs, ratio_cmp,
            s=msize_cmp, alpha=0.9, facecolor='w', edgecolor=cmap.mpl_colormap(0.5),
            marker=marker_cmp, linewidth=3.0, label='__no_label__')

        if show_stats:
            print("Mean ratio in Bin {:d}: {:5.3f}+/-{:5.3f}".format(
                bin_id, np.mean(ratio_cmp[r_mpc_obs <= 3.0]),
                np.mean((ratio_cmp_err_upp)[r_mpc_obs <= 3.0])
            ))

        ax2.set_ylim(ratio_range)

        if bin_id == 4:
            _ = ax2.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=30)
        else:
            ax2.set_xticklabels([])
        _ = ax2.set_ylabel(r'$\Delta\Sigma_{:s}/\Delta\Sigma_{:s}$'.format(sub_cmp, sub_ref), 
                           fontsize=31)

        if show_stats:
            if np.mean(ratio_cmp[0:-3]) <= 1:
                ax2.text(0.07, 0.90, r'$\sigma_{:s}={:4.2f}$'.format(sub_ref, sig_ref[bin_id - 1]),
                        fontsize=23, transform=ax2.transAxes)
                ax2.text(0.07, 0.80, r'$\sigma_{:s}={:4.2f}$'.format(sub_cmp, sig_cmp[bin_id - 1]),
                        fontsize=23, transform=ax2.transAxes)

                ax2.text(0.54, 0.90, r'$\chi^2={:4.2f}$'.format(chi2_ref[bin_id - 1]),
                        fontsize=23, transform=ax2.transAxes)
                ax2.text(0.54, 0.80, r'$\chi^2={:4.2f}$'.format(chi2_cmp[bin_id - 1]),
                        fontsize=23, transform=ax2.transAxes)
            else:
                ax2.text(0.07, 0.16, r'$\sigma_{:s}={:4.2f}$'.format(sub_ref, sig_ref[bin_id - 1]),
                        fontsize=23, transform=ax2.transAxes)
                ax2.text(0.07, 0.06, r'$\sigma_{:s}={:4.2f}$'.format(sub_cmp, sig_cmp[bin_id - 1]),
                        fontsize=23, transform=ax2.transAxes)

                ax2.text(0.54, 0.16, r'$\chi^2={:4.2f}$'.format(chi2_ref[bin_id - 1]),
                        fontsize=23, transform=ax2.transAxes)
                ax2.text(0.54, 0.06, r'$\chi^2={:4.2f}$'.format(chi2_cmp[bin_id - 1]),
                        fontsize=23, transform=ax2.transAxes)


        # ----- Plot 3: Halo mass distribution plot ----- #

        # Histogram for sigma = 0.0
        ax3.fill_between(mvir_true, hist_true / hist_true.sum() / 1.7, color='grey',
                         step="pre", alpha=0.3, label=r'$\sigma_{\mathcal{M}|\mathcal{O}}=0.0$', zorder=0)
        ax3.axvline(mvir_avg_true, color='k', alpha=0.7, linewidth=4.0, linestyle='--')

        # Halo mass distribution for the reference sample
        ax3.fill_between(mvir_ref, hist_ref / hist_ref.sum(), color=color,
                         step="pre", alpha=0.5, label=label_ref, zorder=1)
        ax3.axvline(mvir_avg_ref, color=color, alpha=0.8, linewidth=4.0, linestyle='-.')

        # Halo mass distribution for the comparison sample
        ax3.fill_between(mvir_cmp, hist_cmp / hist_cmp.sum(), edgecolor=color, facecolor='none',
                         step="pre", alpha=0.8, linewidth=5, label=label_cmp, zorder=2)
        ax3.axvline(mvir_avg_cmp, color=color, alpha=0.8, linewidth=4.0, linestyle=':')

        if show_stats:
            print("Mean Mvir: {:5.3f} v.s. {:5.3f}".format(mvir_avg_ref, mvir_avg_cmp))
            print("Difference of Mvir in Bin {:d}: {:5.3f}".format(
                bin_id, mvir_avg_ref - mvir_avg_cmp))

        if logmh_range is None:
            ax3.set_xlim(11.85, 15.35)
        else:
            ax3.set_xlim(logmh_range)

        if bin_id == 1:
            ax3.legend(loc='best', fontsize=20)

        ax3.axhline(0.0, linewidth=3.0, c='grey', alpha=0.7)
        ax3.set_yticklabels([])
        if bin_id == 4:
            _ = ax3.set_xlabel(r'$\log(M_{\rm vir}/M_{\odot})\ [\rm dex]$', fontsize=30)
        else:
            ax3.set_xticklabels([])

        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(30)

        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(30)

        for tick in ax3.xaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        for tick in ax3.yaxis.get_major_ticks():
            tick.label.set_fontsize(30)

    return fig

def show_split_result(dsig, x_arr, y_arr, mask_list, mask=None, line_result=None,
                      x_label=r'$\rm X$', y_label=r'$\rm Y$', info_1=None, info_2=None,
                      legend_1=None, legend_2=None):
    """Visualize the split test results."""
    fig = plt.figure(figsize=(16.0, 5))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.99,
                        wspace=0.3, hspace=0.0)

    if legend_1 is None:
        legend_1 = r'$\rm Lower$'
    if legend_2 is None:
        legend_2 = r'$\rm Upper$'

    # ---- 2D Plot ---- #
    ax1 = fig.add_subplot(131)
    if mask is None:
        mask = np.isfinite(x_arr)
    ax1.scatter(
        x_arr[mask], y_arr[mask], s=5, alpha=0.8, facecolor='grey')

    if line_result is not None:
        x = np.linspace(np.min(x_arr[mask]), np.max(x_arr[mask]), 10)
        slope, inter = line_result['slope'], line_result['inter']
        inter_err = line_result['inter_err']
        sigma = line_result['sigma']
        ax1.plot(x, slope * x + inter + sigma * inter_err, c='k', linestyle='--', linewidth=2)
        ax1.plot(x, slope * x + inter - sigma * inter_err, c='k', linestyle='--', linewidth=2)

    ax1.scatter(
        x_arr[mask & mask_list[0]], y_arr[mask & mask_list[0]], facecolor='steelblue',
        s=6, alpha=0.8, label=legend_1)

    ax1.scatter(
        x_arr[mask & mask_list[1]], y_arr[mask & mask_list[1]], facecolor='orangered',
        s=7, alpha=0.8, label=legend_2, marker='h')

    _ = ax1.set_xlim(
        np.min(x_arr[mask]) * 0.98, np.max(x_arr[mask]) * 1.02)
    _ = ax1.set_ylim(
        np.min(y_arr[mask]) * 0.98, np.max(y_arr[mask]) * 1.02)

    ax1.set_xlabel(x_label, fontsize=30)
    ax1.set_ylabel(y_label, fontsize=30)

    ax1.legend(loc='best', fontsize=20)


    # ---- DeltaSigma ---- #
    ax2 = fig.add_subplot(132)
    ax2.set_xscale("log", nonpositive='clip')
    ax2.set_yscale("log", nonpositive='clip')

    # Lower sample
    ax2.errorbar(
        dsig.meta['r_mpc'], dsig[0]['dsigma'], yerr=dsig[0]['dsig_err_jk'],
        ecolor='steelblue', color='steelblue', alpha=0.5, capsize=4, capthick=2.0, elinewidth=2.0,
        label='__no_label__', fmt='o', zorder=0)
    ax2.scatter(
        dsig.meta['r_mpc'], dsig[0]['dsigma'], s=70, alpha=0.8,
        facecolor='steelblue', edgecolor='k', label=legend_1, linewidth=2.5)

    # Higher sample
    ax2.errorbar(
        dsig.meta['r_mpc'], dsig[1]['dsigma'], yerr=dsig[1]['dsig_err_jk'],
        ecolor='orangered', color='orangered', alpha=0.5, capsize=4, capthick=2.0, elinewidth=2.0,
        label='__no_label__', fmt='h', zorder=0)
    ax2.scatter(
        dsig.meta['r_mpc'], dsig[1]['dsigma'], s=85, alpha=0.8, marker='h',
        facecolor='orangered', edgecolor='k', label=legend_2, linewidth=2.5)

    _ = ax2.set_xlim(
        np.min(dsig.meta['r_mpc']) * 0.5, np.max(dsig.meta['r_mpc']) * 2.0)

    y = np.hstack([dsig[0]['dsigma'], dsig[1]['dsigma']])
    y_min = np.min(y) * 0.6
    y_min = 0.01 if y_min <= 0 else y_min
    _ = ax2.set_ylim(y_min, np.max(y) * 1.9)

    _ = ax2.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=30)
    _ = ax2.set_ylabel(
        r'$\Delta\Sigma\ [{\rm Mpc}\ M_{\odot}/\mathrm{pc}^2]$', fontsize=30)

    if info_1 is not None:
        _ = ax2.text(0.45, 0.90, info_1, fontsize=22, transform=ax2.transAxes)
    if info_2 is not None:
        _ = ax2.text(0.45, 0.80, info_2, fontsize=22, transform=ax2.transAxes)
    ax2.legend(loc='lower left', fontsize=20)

    # ---- R x DeltaSigma ---- #
    ax3 = fig.add_subplot(133)
    ax3.set_xscale("log", nonpositive='clip')

    # Lower sample
    ax3.errorbar(
        dsig.meta['r_mpc'], dsig.meta['r_mpc'] * dsig[0]['dsigma'],
        yerr=(dsig.meta['r_mpc'] * dsig[0]['dsig_err_jk']),
        ecolor='steelblue', color='steelblue', alpha=0.5, capsize=4, capthick=2.0, elinewidth=2.0,
        label='__no_label__', fmt='o')
    ax3.scatter(
        dsig.meta['r_mpc'], dsig.meta['r_mpc'] * dsig[0]['dsigma'], s=70, alpha=0.8,
        facecolor='steelblue', edgecolor='k', label=r'$\rm HSC$', linewidth=2.5)

    # Upper sample
    ax3.errorbar(
        dsig.meta['r_mpc'], dsig.meta['r_mpc'] * dsig[1]['dsigma'],
        yerr=(dsig.meta['r_mpc'] * dsig[1]['dsig_err_jk']),
        ecolor='orangered', color='orangered', alpha=0.5, capsize=4, capthick=2.0, elinewidth=2.0,
        label='__no_label__', fmt='o')
    ax3.scatter(
        dsig.meta['r_mpc'], dsig.meta['r_mpc'] * dsig[1]['dsigma'], s=85, alpha=0.8,
        marker='h', facecolor='orangered', edgecolor='k', label=r'$\rm DES$', linewidth=2.5)

    _ = ax3.set_xlim(
        np.min(dsig.meta['r_mpc']) * 0.5, np.max(dsig.meta['r_mpc']) * 2.0)

    y = np.hstack(
        [dsig.meta['r_mpc'] * dsig[0]['dsigma'], dsig.meta['r_mpc'] * dsig[1]['dsigma']])
    y_min = np.min(y) * 0.3
    y_min = 0.1 if y_min <= 0 else y_min
    _ = ax3.set_ylim(y_min, np.max(y) * 1.2)

    _ = ax3.set_xlabel(r'$R\ [\mathrm{Mpc}]$', fontsize=30)
    _ = ax3.set_ylabel(
        r'$R \times \Delta\Sigma\ [{\rm Mpc}\ M_{\odot}/\mathrm{pc}^2]$', fontsize=30)

    return fig
