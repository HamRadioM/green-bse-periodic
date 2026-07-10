#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Job script for the two-step active-space BSE spectrum (BSE@scGW).        #
#    Quasiparticle energies are generated on the fly from the scGW output;    #
#    no bse_singlet.h5 is required.                                           #
#                                                                             #

import sys
from pathlib import Path

this_dir = Path(__file__).resolve().parent
sys.path.append(str(this_dir / "../green_bse"))

from twostep_bse import create_argument_parser, TwoStepBSEConfig, TwoStepBSESolver


def main():
    """
    Parse arguments, build the configuration, and run the two-step BSE solver.
    Run `python solveTwoStepBSE_main.py -h` for the available options.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    config = TwoStepBSEConfig.from_args(args)

    solver = TwoStepBSESolver(config)
    solver.run()


if __name__ == "__main__":
    main()
