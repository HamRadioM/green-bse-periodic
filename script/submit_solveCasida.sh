#!/bin/bash 
#SBATCH -p debug
#SBATCH -N 1 
#SBATCH -c 32
#SBATCH -J Nitrogen_bse
#SBATCH -t 0:30:00 
#SBATCH -o solveCasida_output.o%j 


export SCRIPTDIR=/your/code/directory/green-bse/script
export INPUTDIR=/your/input/directory
export SIMDIR=/your/input/directory
export IRDIR=/your/irgrid/directory


# Run `python $SCRIPTDIR/solveCasida_main.py -h` to see all available options.


python $SCRIPTDIR/solveCasida_main.py --type singlet \
       --calc_pi 1 --qpac 1 --monitor 1 --n_jobs -1 \
       --iter -1 --iter_W -1 --beta 1000 \
       --sim $SIMDIR/sim.h5 \
       --int_path $INPUTDIR/df_hf_int/ \
       --input $INPUTDIR/input.h5 \
       --ir_file $IRDIR/1e5_136.h5 \
       --output $SIMDIR/bse_singlet.h5 
