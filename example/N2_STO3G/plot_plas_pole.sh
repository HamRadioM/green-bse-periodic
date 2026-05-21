#!/bin/bash 

export SCRIPTDIR=../../script
export IRDIR=irgrid

python -u $SCRIPTDIR/plasmon_pole_fit_curve.py \
       --bse_file ./bse_singlet.h5 \
       --ir_grid $IRDIR/1e5.h5  \
       --exc_idx 0
       
