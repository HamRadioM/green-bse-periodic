#!/bin/bash 

export SCRIPTDIR=../../script
export IRDIR=irgrid

python -u $SCRIPTDIR/plot_spectrum.py \
       --bse_file ./bse_singlet.h5 \
       --plot_file exc_spectrum.pdf \
       --eta 0.1 --w_max 15
       
