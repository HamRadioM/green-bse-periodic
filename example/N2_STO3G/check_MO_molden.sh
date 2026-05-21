#!/bin/bash 

export SCRIPTDIR=../../script

python -u $SCRIPTDIR/ph_excitation_MO.py \
       --bse_file ./bse_singlet.h5 \
       --coords_file ./N2_geom.dat \
       --molden_path molden \
       --basis sto-3g \
       --excitation_num 5
       
