#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Entry-point script for finite-q periodic BSE@scGW calculations.          #
#    Mirrors solveCasida_periodic.py but uses FiniteQBSESolver.               #
#                                                                             #
#    Usage example:                                                           #
#      python script/solveCasida_finite_q.py \                               #
#          --type singlet --beta 1000 --qpac 0 \                             #
#          --q_idx 1 \                                                        #
#          --input    example/H2_periodic_4k/mean_field_input.h5 \           #
#          --sim      example/H2_periodic_4k/sim.h5 \                        #
#          --int_path example/H2_periodic_4k/df_hf_int_fq/ \                 #
#          --ir_file  example/H2_periodic_4k/irgrid/1e5.h5 \                 #
#          --pi_file  example/H2_periodic_4k/p_iw_tilde_q1.h5 \             #
#          --output   example/H2_periodic_4k/bse_fq1_singlet.h5              #
#                                                                             #
#    All standard arguments from bse.py are supported.                       #
#    --q_idx selects the momentum transfer: 0 = optical limit, 1,2,... = fq. #
#                                                                             #

import sys
import os

# Allow importing from green_bse/ when running from the script/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'green_bse'))

from bse import BSEConfig, create_argument_parser
from bse_finite_q import FiniteQBSESolver


def main():
    parser = create_argument_parser()
    parser.description = (
        "Finite-q periodic BSE@scGW Casida solver for crystalline solids. "
        "Uses proper off-diagonal GDF integrals <phi_mu(k)|V_Q|phi_nu(k+q)>. "
        "K-point structure is read automatically from input.h5. "
        "q=0 recovers the optical-limit result of solveCasida_periodic.py."
    )
    parser.add_argument(
        '--q_idx', type=int, default=0,
        help=(
            "Index of the q-vector in the BvK k-mesh. "
            "q_idx=0 → q=0 (optical limit, same as solveCasida_periodic.py). "
            "q_idx=1,2,... → finite momentum transfer q=kpts[q_idx]. "
            "Default: 0."
        )
    )
    args   = parser.parse_args()
    config = BSEConfig.from_args(args)
    solver = FiniteQBSESolver(config, q_idx=args.q_idx)
    solver.run()


if __name__ == '__main__':
    main()
