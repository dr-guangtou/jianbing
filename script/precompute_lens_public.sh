#!/bin/sh

# Using medium photo-z quality cut
python3 precompute_lens_public.py -l s16a_weak_lensing_medium.hdf5 -o topn_public_s16a_medium_precompute.hdf5 --z_min=0.19 --z_max=0.52 -j 12

# Using basic photo-z quality cut 
python3 precompute_lens_public.py -l s16a_weak_lensing_basic.hdf5 -o topn_public_s16a_basic_precompute.hdf5 --z_min=0.19 --z_max=0.52 -j 12

# Using strict photo-z quality cut
python3 precompute_lens_public.py -l s16a_weak_lensing_strict.hdf5 -o topn_public_s16a_strict_precompute.hdf5 --z_min=0.19 --z_max=0.52 -j 12

# Using medium photo-z cut with comoving coordinates
python3 precompute_lens_public.py -l s16a_weak_lensing_medium_comoving.hdf5 -o topn_public_s16a_medium_comoving_precompute.hdf5 --z_min=0.19 --z_max=0.52 -j 12

# Using medium photo-z quality cut and larger radial range
python3 precompute_lens_public.py -l s16a_weak_lensing_medium_larger.hdf5 -o topn_public_s16a_medium_larger_precompute.hdf5 --z_min=0.19 --z_max=0.52 -j 12
