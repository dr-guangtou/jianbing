#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to deal with catalog."""

import os
import copy
import distutils.spawn

from shutil import copyfile

import numpy as np

from scipy.spatial import cKDTree
from astropy.table import Table, Column, vstack

from dsigma import helpers

from . import hsc
from . import utils

__all__ = ["get_number_density", "get_mask_within_ranks", "get_source_hist2d",
           "catalog_to_kdtree", "catalog_update_specz", "add_outer_mass",
           "show_summary_table", "get_mask_list", "rebin_mhalo_hist", "regulate_colname",
           "filter_through_bright_star_mask", "rank_splitting_sample", "prepare_catalog_for_acf"]


def regulate_colname(catalog, suffix=None):
    """
    Put suffix to the column names of the table.

    Parameters
    ----------

    catalog : astropy.table
        The input astropy table.
    suffix : string
        The string for suffix.
    """
    for col in catalog.colnames:
        col_new = col.strip().lower() + '_' + suffix if suffix else col.strip().lower()
        catalog.rename_column(col, col_new)

    return catalog


def get_number_density(sample, property_low, property_upp, z_low=0.19, z_upp=0.52,
                       z_col='z', property_col='lambda', area=hsc.A_S16A):
    """Return the number density of objects."""
    z_cut = (sample[z_col] >= z_low) & (sample[z_col] < z_upp)
    property_cut = (sample[property_col] >= property_low) & (sample[property_col] < property_upp)
    n_objects = len(sample[property_cut & z_cut])

    rho_objects = n_objects / utils.get_volume(area, z_low, z_upp, verbose=False)
    rho_err = np.sqrt(n_objects) / utils.get_volume(area, z_low, z_upp, verbose=False)

    return n_objects, rho_objects * 1E6, rho_err * 1E6


def get_mask_within_ranks(data, rank_low, rank_upp, master_mask=None, nan_replace=-999,
                          verbose=False, return_minmax=False):
    """Mask for objects ranking between N1 and N2."""
    if np.any(~np.isfinite(data)):
        data = np.where(~np.isfinite(data), nan_replace, data)

    if master_mask is None:
        data_use = data
    else:
        data_use = data[master_mask]

    idx_sorted = np.argsort(data_use)[::-1]

    max_val = data_use[idx_sorted[rank_low - 1]]
    min_val = data_use[idx_sorted[rank_upp - 1]]
    if verbose:
        print("Min - Max values: {:.3f} - {:.3f}".format(min_val, max_val))

    if master_mask is not None:
        if return_minmax:
            return (data >= min_val) & (data <= max_val) & master_mask, min_val, max_val
        else:
            return (data >= min_val) & (data <= max_val) & master_mask

    if return_minmax:
        return (data >= min_val) & (data <= max_val), min_val, max_val
    else:
        return (data >= min_val) & (data <= max_val)


def get_mask_list(data, ranks_low, ranks_upp, master_mask=None, sub_mask=None,
                  volume_factor=1.0, **kwargs):
    """Get the list of masks for each bin."""
    mask_list = [get_mask_within_ranks(
        data, r_low, r_upp, master_mask=master_mask, **kwargs) for (r_low, r_upp) in zip(
            ranks_low, ranks_upp)]

    if sub_mask:
        return [mask & sub_mask for mask in mask_list]
    return mask_list


def get_source_hist2d(cat, field='field', bin_size=0.05, verbose=True):
    """Generate 2-D histogram for the source galaxies.
    """
     # List of unique fields
    field_list = np.unique(cat[field])

    field_hist2d = {}

    for field in field_list:
        if verbose:
            print("Dealing with field {:d}".format(field))
        reg_mask = (cat['field'] == field)
        ra, dec = cat['ira'][reg_mask], cat['idec'][reg_mask]

        ra_range, dec_range = ra.max() - ra.min(), dec.max() - dec.min()
        ra_bins, dec_bins = int(ra_range / bin_size), int(dec_range / bin_size)
        aspect = dec_range / ra_range
        extent = (ra.min() - bin_size, ra.max() + bin_size,
                  dec.min() - bin_size, dec.max() + bin_size)

        density, ra_edges, dec_edges = np.histogram2d(
            ra, dec, bins=(ra_bins, dec_bins))

        field_hist2d[field] = {
            'density': density, 'ra_edges': ra_edges, 'dec_edges': dec_edges,
            'aspect': aspect, 'extent': extent,
            'ra_bins': ra_bins, 'dec_bins': dec_bins,
            'ra_range': ra_range, 'dec_range': dec_range
        }

    return field_hist2d


def catalog_to_kdtree(table, ra, dec):
    """Create a KDTree structure using the coordinates in the table."""
    x, y, z = helpers.spherical_to_cartesian(table[ra], table[dec])
    return cKDTree(np.column_stack([x, y, z]))


def catalog_update_specz(cat, specz, redshift='z', ra='ra', dec='dec', r_match=1.0,
                         replace=False, z_new='z_best', verbose=True):
    """Match the catalog to a spec-z catalog and update the best available redshift.
    """
    # Built a KDTree for the spec-z catalog
    specz_tree = catalog_to_kdtree(specz, 'specz_ra', 'specz_dec')

    # Convert the RA, Dec in the catalog to X, Y, Z in Cartesian coordinates
    x, y, z = helpers.spherical_to_cartesian(cat[ra], cat[dec])
    cat_xyz = np.column_stack([x, y, z])

    # Query for the nearest neighbour
    dist, index = specz_tree.query(cat_xyz, k=1)

    # Convert the matching radius into 3-D Cartesian distance
    r_max_3d = np.sqrt(2 - 2 * np.cos(np.deg2rad(r_match / 3600.0)))

    # Get the matched spec-z
    z_match = np.asarray(specz[index]['specz_redshift'])

    # Only keep the ones with distance smaller than the matching radius
    z_match[dist > r_max_3d] = np.nan
    if verbose:
        print("# {:d} objects with matched spec-z".format((dist <= r_max_3d).sum()))

    # Update the catalog
    cat_new = copy.deepcopy(cat)
    cat_new['z_spec'] = z_match
    z_best = copy.deepcopy(np.asarray(cat[redshift]))
    z_best[np.isfinite(z_match)] = z_match[np.isfinite(z_match)]

    if replace:
        cat_new.remove_column(redshift)
        cat_new['z_old'] = cat[redshift]
        cat_new[redshift] = z_best
    else:
        cat_new[z_new] = z_best

    return cat_new


def add_outer_mass(data, mass_inn, mass_out, name=None):
    """Add outer stellar mass to the catalog."""
    if name is None:
        name = 'logm_{:s}_{:s}'.format(
            mass_inn.split('_')[-1], mass_out.split('_')[-1])

    mout = np.where(np.isfinite(data[mass_out]), data[mass_out], 1e-10)
    minn = np.where(np.isfinite(data[mass_inn]), data[mass_inn], 1e-10)
    m_diff = 10.0 ** mout - 10.0 ** minn
    m_diff = np.where(m_diff <= 0, 1e-10, m_diff)

    data[name] = np.log10(m_diff)

    return data


def _get_bin_sum(bin_result):
    """
    Get the [min, max] of the sample; best-fit scatter and error.
    """
    min_val = bin_result['samples'].min()
    max_val = bin_result['samples'].max()
    sig = bin_result['sig_med_bt']
    err = bin_result['sig_err_bt']

    return min_val, max_val, sig, err


def show_summary_table(summary, keys=None, separation_line='-' * 95,
                       show_header=True, print_table=True, no_boundary=False):
    """Show a summary table of the result.
    """
    if keys is None:
        if type(summary) is dict:
            keys = list(summary.keys())
        elif type(summary) is Table:
            keys = summary.colnames
        else:
            raise TypeError("Astropy table or dictionary")

    sum_tab = []

    header = " " * 32 + (" " * 13).join(
        ["Bin {:1d}". format(ii + 1) for ii in np.arange(len(summary[keys[0]]))])
    sum_tab.append(sum_tab)
    if print_table and show_header:
        print(header)

    for key in keys:
        boundry_str = ''
        scatter_str = ''

        for bin_col in summary[key]:
            val_0, val_1, sig, err = _get_bin_sum(bin_col)
            boundry_str += " [{:6.2f}, {:6.2f}] ".format(val_0, val_1)
            scatter_str += "     {:4.2f}+/-{:4.2f}  ".format(sig, err)
        sum_tab.append(boundry_str)
        sum_tab.append(scatter_str)
        if print_table:
            print(separation_line)
            if no_boundary:
                print("{:22s} {:s}".format(key, scatter_str))
            else:
                print("{:24s} {:s}".format(key, boundry_str))
                print("{:22s} {:s}".format('' * 20, scatter_str))
    print(separation_line)

    return sum_tab

def rebin_mhalo_hist(sim_cat, bin_id, scatter, n_bin=10,
                     use_edges=True, min_mh=None, max_mh=None):
    """Rebin the halo mass histogram."""
    bin_center = lambda arr: (arr[1: ] + arr[: -1]) / 2.0

    sim_bin = sim_cat[sim_cat['number_density_bin'] == bin_id]
    sim_use = sim_bin[np.argmin(np.abs(sim_bin['scatter'] - scatter))]
    edges, hist = sim_use['edge'], sim_use['hist']
    cen = bin_center(edges)

    # Get the mean halo mass
    avg_mhalo = np.average(cen, weights=hist)

    # Define the new histogram bins
    min_mh = np.min(edges) if min_mh is None else min_mh
    max_mh = np.max(edges) if max_mh is None else max_mh
    edges_new = np.linspace(min_mh, max_mh, n_bin + 1)

    cen_new = bin_center(edges_new)
    hist_new = utils.rebin_hist(cen, hist, edges_new)

    if use_edges:
        return (np.concatenate([edges_new, [edges_new[-1]]]),
                np.concatenate([[0], hist_new, [0]]), avg_mhalo)

    return cen_new, hist_new, avg_mhalo


def filter_through_bright_star_mask(catalog, mask_dir, reg_prefix='new_S18Amask',
                                    filters='grizy', filter_type='outside',
                                    ra='ra', dec='dec', output_suffix='bsm'):
    """Filter the catalog through the .reg files of the bright star masks."""
    # Make the sure venice is installed
    venice = distutils.spawn.find_executable("venice")
    assert venice, "Venice is not installed!"

    # Get the .reg files for the bright star mask
    reg_files = [
        os.path.join(mask_dir, reg_prefix + '_' + band + '.reg') for band in filters]

    # Output catalog
    output_catalogs = [catalog.replace('.fits', '_bsm_' + band + '.fits') for band in filters]

    output_final = catalog.replace('.fits', '_%s.fits' % output_suffix)

    # Generate the commands
    for ii, reg_mask in enumerate(reg_files):
        if ii == 0:
            venice_command = (
                venice + ' -m ' + reg_mask + ' -f ' + filter_type + ' -cat ' + catalog +
                ' -xcol ' + ra + ' -ycol ' + dec + ' -o ' + output_catalogs[0]
            )
        else:
            venice_command = (
                venice + ' -m ' + reg_mask + ' -f ' + filter_type + ' -cat ' +
                output_catalogs[ii - 1] + ' -xcol ' + ra + ' -ycol ' + dec +
                ' -o ' + output_catalogs[ii]
            )
        # Execute the command
        _ = os.system(venice_command)

    # Copy the last catalog to the final name
    if not os.path.isfile(output_catalogs[-1]):
        raise Exception("# Something is wrong with the Venice!")
    else:
        _ = copyfile(output_catalogs[-1], output_final)

    # Delete the intermediate catalogs
    for output in output_catalogs:
        try:
            os.remove(output)
        except OSError:
            pass

    return Table.read(output_final)


def rank_splitting_sample(cat, X_col, Y_col, n_bins=5, n_sample=2,
                          X_min=None, X_max=None, X_bins=None,
                          id_each_bin=True):
    """Split sample into N_sample with fixed distribution in X, but different
    rank orders in Y.

    Parameters:
    -----------
    cat : astropy.table
        Table for input catalog
    X_col : string
        Name of the column for parameter that should have fixed distribution
    Y_col : string
        Name of the column for parameter that need to be split
    n_bins : int
        Number of bins in X
    n_sample : int
        Number of bins in Y
    X_min : float
        Minimum value of X for the binning
    X_max: float
        Maximum value of X for the binning
    X_bins : array
        Edges of X bins, provided by the users
        Usefull for irregular binnings

    Return
    ------

    """
    data = copy.deepcopy(cat)

    X = data[X_col]
    X_len = len(X)
    if X_bins is None:
        if X_min is None:
            X_min = np.nanmin(X)
        if X_max is None:
            X_max = np.nanmax(X)

        msg = '# Sample size should be much larger than number of bins in X'
        assert X_len > (2 * n_bins), msg

        X_bins = np.linspace(X_min, X_max, (n_bins + 1))
    else:
        n_bins = (len(X_bins) - 1)

    # Place holder for sample ID
    data.add_column(Column(data=(np.arange(X_len) * 0),
                           name='sample_id'))
    data.add_column(Column(data=np.arange(X_len), name='index_ori'))

    # Create index array for object in each bin
    X_idxbins = np.digitize(X, X_bins, right=True)

    bin_list = []
    for ii in range(n_bins):
        subbin = data[X_idxbins == (ii + 1)]
        subbin.sort(Y_col)

        subbin_len = len(subbin)
        subbin_size = int(np.ceil(subbin_len / n_sample))

        idx_start, idx_end = 0, subbin_size
        for jj in range(n_sample):
            if idx_end > subbin_len:
                idx_end = subbin_len
            if id_each_bin:
                subbin['sample_id'][idx_start:idx_end] = ((jj + 1) +
                                                          (ii * n_sample))
            else:
                subbin['sample_id'][idx_start:idx_end] = (jj + 1)
            idx_start = idx_end
            idx_end += subbin_size

        bin_list.append(subbin)

    new_data = vstack(bin_list)

    return new_data

def prepare_catalog_for_acf(catalog, logm='logm_max', index='object_id', ra='ra', dec='dec',
                            redshift='z_best', min_logm=None):
    """Prepare the HSC catalog for awesome cluster finder."""
    # Make a copy of the file
    cat_use = copy.deepcopy(catalog)

    # Add a Mstar column. Notice that this is not the logM*, but just M*
    cat_use.add_column(Column(data=(10.0 ** cat_use[logm]), name='Mstar'))

    # Normalize some column names
    if index != 'id':
        cat_use.rename_column(index, 'id')
    if ra != 'ra':
        cat_use.renmae_column(ra, 'ra')
    if dec != 'dec':
        cat_use.renmae_column(dec, 'dec')
    if redshift != 'z':
        cat_use.rename_column(redshift, 'z')

    # Make a mass cut if necessary
    if min_logm:
        cat_use = cat_use[cat_use[logm] >= min_logm]
        print("# Keep {} galaxies with {} >= {}".format(len(cat_use), logm, min_logm))

    # Only keep the useful columns
    return cat_use['id', 'ra', 'dec', 'z', 'Mstar'].as_array()
