#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to deal with cutout pictures."""

import random
import itertools

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from . import utils

__all__ = ['catalog_radius_match', 'visualize_cluster_circle', 'get_cluster_members',
           'get_galaxies_on_image', 'match_and_show_clusters', 'match_and_show_galaxies']


def catalog_radius_match(cat, ra, dec, rad_arcsec, ra_col='ra', dec_col='dec',
                         z_col='z', z_low=None, z_upp=None):
    """Return the objects in the catalog matched to a coordinate within a radius."""
    if isinstance(rad_arcsec, u.quantity.Quantity):
        rad_arcsec = rad_arcsec.value

    z_mask = np.isfinite(cat[ra_col])
    if z_low is not None:
        z_mask = z_mask & (cat[z_col] >= z_low)
    if z_upp is not None:
        z_mask = z_mask & (cat[z_col] < z_upp)

    if z_mask.sum() <= 0:
        # No useful cluster
        return None

    flag = utils.angular_distance(
        ra, dec, np.asarray(cat[ra_col]), np.asarray(cat[dec_col])) < rad_arcsec
    flag = flag & z_mask

    if flag.sum() == 0:
        return None
    return cat[flag]


def visualize_cluster_circle(cluster, wcs, ax=None, ra_col='ra', dec_col='dec', z_col='z',
                             r_col=None, rad=0.5, pix=0.168, use_mpc=True,
                             color='orangered', marker='x', marker_size=150, linewidth=3,
                             alpha=0.6, linestyle='--', show_redshift=True, show_center=True):
    """Show a cluster using a circle."""
    cen_coord = SkyCoord(cluster[ra_col], cluster[dec_col], unit="deg", frame='icrs')
    cen_x, cen_y = cen_coord.to_pixel(wcs)

    if r_col is not None:
        rad = cluster[r_col]
    if use_mpc:
        rad *= 1e3
    rad_pix = utils.r_phy_to_ang(rad, cluster[z_col]).value / pix

    if ax is None:
        return cen_x, cen_y, rad_pix

    if show_center:
        ax.scatter(cen_x, cen_y, marker=marker, linewidth=linewidth, s=marker_size,
                   alpha=alpha, color=color)
    cluster_circle = plt.Circle(
        (cen_x, cen_y), rad_pix, color=color, linewidth=linewidth, linestyle=linestyle,
        alpha=0.6, fill=False)
    _ = ax.add_artist(cluster_circle)

    if show_redshift:
        angles = np.linspace(0, 360, 15)
        random.shuffle(angles)
        img_w, img_h = wcs.array_shape
        for ang in angles:
            if ang >= 0 and ang < 180:
                y_off, v_align = -8, 'top'
            else:
                y_off, v_align = 8, 'bottom'
            text_x = cen_x - rad_pix * np.cos(ang) + 10
            text_y = cen_y - rad_pix * np.sin(ang) + y_off

            if text_x > 10 and text_x < (img_w - 50) and text_y > 10 and text_y < (img_h - 50):
                ax.text(text_x, text_y,
                        r"$z={:4.2f}$".format(cluster[z_col]), horizontalalignment='left',
                        verticalalignment=v_align, fontsize=20, color=color)
                break

    return ax


def get_cluster_members(cluster, members, id_col='id', ra_col='ra', dec_col='dec',
                        just_coord=True, wcs=None):
    """Return a catalog of member galaxies of a given cluster."""
    mems = members[members[id_col] == cluster[id_col]]

    if len(mems) <= 0:
        return None

    radec = np.asarray(
        [[obj[ra_col], obj[dec_col]] for obj in mems], dtype=np.float_)

    if wcs is not None:
        pixs = wcs.wcs_world2pix(radec, 0)
        x_arr, y_arr = pixs[:, 0], pixs[:, 1]
    else:
        x_arr, y_arr = radec[:, 0], radec[:, 1]

    if just_coord:
        return x_arr, y_arr
    return mems, x_arr, y_arr


def get_galaxies_on_image(cat, wcs, ra_col='ra', dec_col='dec', z_col='z', z_low=None, z_upp=None,
                          prop=None, prop_low=None, prop_upp=None, just_coord=False):
    """Return a catalog of objects on the image."""
    # Redshift cut
    z_flag = np.isfinite(cat[ra_col])
    if z_low is not None:
        z_flag = z_flag & (cat[z_col] >= z_low)
    if z_upp is not None:
        z_flag = z_flag & (cat[z_col] < z_upp)

    if prop is not None:
        p_flag = np.isfinite(cat[prop])
        if prop_low is not None:
            p_flag = p_flag & (cat[prop] >= prop_low)
        if prop_upp is not None:
            p_flag = p_flag & (cat[prop] <= prop_upp)
        z_flag = z_flag & p_flag

    # (RA, Dec) cut
    img_w, img_h = wcs.array_shape
    ra_max, dec_min = wcs.all_pix2world(0, 0, 0)
    ra_min, dec_max = wcs.all_pix2world(img_w, img_h, 0)
    mask = (
        (cat[ra_col] > ra_min) & (cat[ra_col] < ra_max) &
        (cat[dec_col] > dec_min) & (cat[dec_col] < dec_max)
    ) & z_flag

    if mask.sum() <= 0:
        return None

    use = cat[mask]
    radec = np.asarray(
        [[obj[ra_col], obj[dec_col]] for obj in use], dtype=np.float_)
    pixs = wcs.wcs_world2pix(radec, 0)
    if just_coord:
        return pixs[:, 0], pixs[:, 1]
    return use, pixs[:, 0], pixs[:, 1]


def match_and_show_clusters(cluster_dict, wcs, ax=None, pix=0.168, z_low=None, z_upp=None,
                            r_default=0.5, show_members=True):
    """Match and show clusters on the image."""
    # Get the image central coordinate and image size in arcsec
    img_w, img_h = wcs.array_shape
    ra_cen, dec_cen = wcs.all_pix2world(img_w / 2, img_h / 2, 0)
    size_arcsec = (img_w * pix) / 2.0 * 1.5

    # Color cycle
    ls_cycle = itertools.cycle(
        ['--', '-.', ':', '-', (0, (3, 5, 1, 5, 1, 5))])

    # Matched clusters
    cluster_matched = catalog_radius_match(
        cluster_dict['cluster'], ra_cen, dec_cen, size_arcsec, z_low=z_low, z_upp=z_upp,
        ra_col=cluster_dict['ra_cen'], dec_col=cluster_dict['dec_cen'],
        z_col=cluster_dict['z'])

    if cluster_matched is not None and len(cluster_matched) >= 1:
        members_matched = []
        for ls, cluster in zip(ls_cycle, cluster_matched):
            # Visualize a cluster using a circle
            if ax is not None:
                ax = visualize_cluster_circle(
                    cluster, wcs, ax=ax,
                    ra_col=cluster_dict['ra_cen'], dec_col=cluster_dict['dec_cen'],
                    z_col=cluster_dict['z'], r_col=cluster_dict['radius'], use_mpc=True,
                    show_center=cluster_dict['show_center'], color=cluster_dict['color'],
                    pix=pix, rad=r_default, marker=cluster_dict['cen_marker'],
                    marker_size=cluster_dict['cen_msize'],
                    linewidth=cluster_dict['cen_lw'], alpha=cluster_dict['cen_alpha'],
                    linestyle=ls, show_redshift=cluster_dict['show_redshift']
                )

            # Show the matched member galaxies
            mems, mem_x, mem_y = get_cluster_members(
                cluster, cluster_dict['members'], id_col=cluster_dict['id'],
                ra_col=cluster_dict['ra_mem'], dec_col=cluster_dict['dec_mem'],
                just_coord=False, wcs=wcs)
            members_matched.append(mems)

            if show_members and len(mems) >= 0 and ax is not None:
                ax.scatter(mem_x, mem_y, s=cluster_dict['mem_msize'],
                           marker=cluster_dict['mem_marker'],
                           facecolor='None', edgecolor=cluster_dict['color'], 
                           alpha=cluster_dict['mem_alpha'],
                           linewidth=cluster_dict['mem_lw'])
    else:
        members_matched = None

    if ax is None:
        return cluster_matched, members_matched
    return ax, cluster_matched, members_matched


def match_and_show_galaxies(galaxies, wcs, ax=None, ra_col='ra', dec_col='dec',
                            z_col='z_best', z_low=None, z_upp=None, cax=None,
                            prop='MSTAR', prop_low=None, prop_upp=None, s_max=1500,
                            prop_range=None, cmap=None, marker='8', alpha=0.9, 
                            linewidth=3.0, show_redshift=False, show_prop=False, y_off=15):
    """Match and show galaxies from a catalog on the image."""
    # Colormap
    if cmap is None:
        from palettable.scientific.sequential import Nuuk_10
        cmap = Nuuk_10.mpl_colormap

    gals, gal_x, gal_y = get_galaxies_on_image(
        galaxies, wcs, ra_col=ra_col, dec_col=dec_col, z_col=z_col, z_low=z_low, z_upp=z_upp,
        prop=prop, prop_low=prop_low, prop_upp=prop_upp
    )

    if ax is None:
        return gals, gal_x, gal_y

    if len(gals) >= 0.:
        # Marker size
        z_max = np.max(gals[z_col])
        msize = (s_max * (z_max + 0.1 - gals[z_col]))

        if prop is not None:
            # Marker color
            color_p = gals[prop]
            if prop_range is None:
                prop_range = [np.nanmin(color_p), np.nanmax(color_p)]
            prop_norm = (color_p - prop_range[0]) / (prop_range[1] - prop_range[0])
            prop_norm = np.where(prop_norm < 0, 0, prop_norm)
            prop_norm = np.where(prop_norm > 1, 1, prop_norm)
            ax.scatter(
                gal_x, gal_y, s=msize, marker=marker, facecolor='None', 
                edgecolor=cmap(prop_norm), alpha=alpha, linewidth=linewidth)
        else:
            ax.scatter(
                gal_x, gal_y, s=msize, marker=marker, facecolor='None', 
                edgecolor='w', alpha=alpha, linewidth=linewidth)

        # Redshift legend
        _ = ax.scatter(0.035, 0.10, marker=marker, s=(s_max * (z_max + 0.1 - 0.2)),
                       transform=ax.transAxes,
                       facecolor='None', edgecolor='w', linewidth=linewidth)
        _ = ax.text(0.07, 0.10, r'$z=0.2$', fontsize=20, transform=ax.transAxes,
                    horizontalalignment='left', verticalalignment='center', color='w')
        _ = ax.scatter(0.035, 0.14, marker=marker, s=(s_max * (z_max + 0.1 - 0.5)),
                       transform=ax.transAxes,
                       facecolor='None', edgecolor='w', linewidth=linewidth)
        _ = ax.text(0.07, 0.14, r'$z=0.5$', fontsize=20, transform=ax.transAxes,
                    horizontalalignment='left', verticalalignment='center', color='w')

        # Color bar
        if prop is not None:
            if cax is None:
                cax = inset_axes(ax, width="25%", height="2%", loc=9)
                #cax = ax.inset_axes([0.04, 0.95, 0.25, 0.02], transform=ax.transAxes)
            norm = mpl.colors.Normalize(vmin=prop_range[0], vmax=prop_range[1])
            cbar = mpl.colorbar.ColorbarBase(
                cax, cmap=cmap, norm=norm, orientation='horizontal')
            cbar.ax.tick_params(labelsize=16)
            cbar.ax.xaxis.set_tick_params(color='grey', which='major')
            cbar.ax.xaxis.set_tick_params(color='grey', which='minor')
            cbar.outline.set_edgecolor('grey')
            _ = plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')

        if show_redshift:
            _ = [ax.text(x, y + y_off, "{:4.2f}".format(g[z_col]), fontsize=12, color='w',
                         horizontalalignment='center', verticalalignment='bottom')
                 for g, x, y in zip(gals, gal_x, gal_y)]

        if show_prop:
            _ = [ax.text(x, y - y_off, "{:4.1f}".format(g[prop]), fontsize=12, color='w',
                         horizontalalignment='center', verticalalignment='top')
                 for g, x, y in zip(gals, gal_x, gal_y)]

    return ax, gals, gal_x, gal_y
