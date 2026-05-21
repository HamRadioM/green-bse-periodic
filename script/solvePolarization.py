#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    CLI entry point for the standalone polarization solver.                  #
#                                                                             #
#    Computes P̃_{QQ'}(q, iΩ) — the density-density transferring screened     #
#    polarizability in the density-fitting basis — and writes it to HDF5      #
#    in a format directly consumable by the finite-q BSE solver via           #
#    --pi_file.                                                               #
#                                                                             #
#    Example usage                                                            #
#    -------------                                                            #
#    # Optical limit (q = 0)                                                  #
#    python solvePolarization.py \                                            #
#        --input  ./mean_field_input.h5 \                                     #
#        --sim    ./scGW_sim.h5         \                                     #
#        --int_path ./df_hf_int_fq/     \                                     #
#        --ir_file  ./irgrid/1e5.h5     \                                     #
#        --beta   1000                  \                                     #
#        --q_idx  0                     \                                     #
#        --output ./p_iw_tilde.h5                                             #
#                                                                             #
#    # Finite-q (q_idx = 2)                                                   #
#    python solvePolarization.py \                                            #
#        --input  ./mean_field_input.h5 \                                     #
#        --sim    ./scGW_sim.h5         \                                     #
#        --int_path ./df_hf_int_fq/     \                                     #
#        --ir_file  ./irgrid/1e5.h5     \                                     #
#        --beta   1000                  \                                     #
#        --q_idx  2                     \                                     #
#        --output ./p_iw_tilde_q2.h5                                          #
#                                                                             #

import os
import sys

# Allow running from the script directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BSE_DIR    = os.path.join(_SCRIPT_DIR, '..', 'green_bse')
sys.path.insert(0, _BSE_DIR)

import argparse
from polarization import PolarizationConfig, PolarizationSolver


def create_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            'Compute the density-fitting screened polarizability P̃(q, iΩ) '
            'from a Green-mbpt scGW calculation.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required I/O ---
    p.add_argument(
        '--input', required=True, metavar='FILE',
        help='Mean-field input HDF5 (mean_field_input.h5).'
    )
    p.add_argument(
        '--sim', required=True, metavar='FILE',
        help='scGW simulation HDF5 (scGW_sim.h5).  G(τ) is read from here.'
    )
    p.add_argument(
        '--int_path', required=True, metavar='DIR',
        help='Directory containing VQ_0.h5 (and VQ_q{idx}.h5 for q≠0).'
    )
    p.add_argument(
        '--ir_file', required=True, metavar='FILE',
        help='Intermediate-representation grid HDF5 (e.g. 1e5.h5).'
    )
    p.add_argument(
        '--output', required=True, metavar='FILE',
        help='Output HDF5 path for P̃(q, iΩ).  '
             'Pass this file to solveCasida_finite_q.py via --pi_file.'
    )

    # --- Physics ---
    p.add_argument(
        '--beta', type=float, default=1000.0, metavar='FLOAT',
        help='Inverse temperature β (a.u.).'
    )
    p.add_argument(
        '--q_idx', type=int, default=0, metavar='INT',
        help='Index of q-vector in the BvK k-mesh.  '
             '0 = optical limit (q=Γ); >0 = finite momentum transfer.'
    )
    p.add_argument(
        '--iter', type=int, default=-1, metavar='INT',
        dest='iteration',
        help='Which scGW iteration to read G(τ) from.  -1 = last available.'
    )

    return p


def main():
    parser = create_argument_parser()
    args   = parser.parse_args()

    config = PolarizationConfig(
        input_file  = args.input,
        sim_file    = args.sim,
        int_path    = args.int_path,
        ir_file     = args.ir_file,
        beta        = args.beta,
        q_idx       = args.q_idx,
        iteration   = args.iteration,
        output_file = args.output,
    )

    solver = PolarizationSolver(config)
    solver.run()


if __name__ == '__main__':
    main()
