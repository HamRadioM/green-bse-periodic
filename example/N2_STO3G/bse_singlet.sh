#!/bin/bash 

export SCRIPTDIR=../../script
export IRDIR=./irgrid

python -u $SCRIPTDIR/solveCasida_main.py \
       --type singlet --beta 1000 \
       --qpac 1 --iter -1 --iter_W -1 \
       --calc_pi 1 --monitor 1 --n_jobs 8 \
       --sim ./scGW_sim.h5 \
       --int_path ./df_hf_int/ \
       --ir_file $IRDIR/1e5.h5 \
       --input ./mean_field_input.h5 \
       --output ./bse_singlet.h5 
       
