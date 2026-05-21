#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Entry-point script for periodic BSE@scGW calculations (solids).         #
#    Mirrors solveCasida_main.py but uses PeriodicBSESolver.                 #
#                                                                             #
#    Usage example:                                                           #
#      python solveCasida_periodic.py                                         #
#          --type singlet --qpac 1 --calc_pi 1 --monitor 1 --n_jobs -1       #
#          --beta 1000 --iter -1 --iter_W -1                                 #
#          --input input.h5 --sim sim.h5                                     #
#          --int_path df_hf_int/ --ir_file 1e5_136.h5                        #
#          --output bse_periodic_singlet.h5                                  #
#                                                                             #
#    All arguments are identical to solveCasida_main.py.                     #
#    The number of k-points is read automatically from input.h5.             #
#                                                                             #

import sys
import os

# Allow importing from green_bse/ when running from the script/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'green_bse'))

from bse import BSEConfig, create_argument_parser
from bse_periodic import PeriodicBSESolver


def main():
    parser = create_argument_parser()
    parser.description = (
        "Periodic BSE@scGW Casida solver for crystalline solids. "
        "K-point structure is read automatically from input.h5."
    )
    args = parser.parse_args()
    config = BSEConfig.from_args(args)

    solver = PeriodicBSESolver(config)
    solver.run()


if __name__ == "__main__":
    main()
