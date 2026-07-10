#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Job script for the periodic, finite-Q two-step active-space BSE          #
#    spectrum (BSE@scGW) of crystalline solids.  Quasiparticle energies are   #
#    generated on the fly from the periodic scGW output; no bse_singlet.h5    #
#    is required.                                                             #
#                                                                             #
#    Example                                                                  #
#    -------                                                                  #
#      python solvePeriodicTwoStepBSE_main.py \                               #
#          --input mean_field_input.h5 --sim scGW_sim.h5 \                    #
#          --int_path df_hf_int/ --pi_file p_iw_tilde_q0.h5 \                 #
#          --ir_file /path/to/ir_grid.h5 --q_idx 0 --channel singlet          #
#                                                                             #
#    Run with -h for the full list of options.                               #
#                                                                             #

import sys
from pathlib import Path

this_dir = Path(__file__).resolve().parent
sys.path.append(str(this_dir / "../green_bse"))

from periodic_twostep_bse import (create_argument_parser,
                                  PeriodicTwoStepBSEConfig,
                                  PeriodicTwoStepBSESolver)


def main():
    """Parse arguments, build the configuration, and run the periodic solver."""
    parser = create_argument_parser()
    args = parser.parse_args()

    config = PeriodicTwoStepBSEConfig.from_args(args)

    solver = PeriodicTwoStepBSESolver(config)
    solver.run()


if __name__ == "__main__":
    main()
