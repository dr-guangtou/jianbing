#!/bin/sh

# Medium photo-z cut
python3 prepare_source_random.py -m -j 12

# Basic photo-z cut
python3 prepare_source_random.py -b -j 12

# Strict photo-z cut
python3 prepare_source_random.py -s -j 12

# Medium photo-z cut with comoving coordinates
python3 prepare_source_random.py -c -j 12

# Medium photo-z cut with larger radial range 
python3 prepare_source_random.py -l -j 12
