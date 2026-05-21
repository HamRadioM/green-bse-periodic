#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.   #
#                                                                             #
#    General BSE solver: computes P⁰ and P_BSE on the Matsubara axis,        #
#    analytically continues each via per-eigenmode plasmon-pole fitting,      #
#    and evaluates the real-axis spectral function A(q,ω).                    #
#                                                                             #
#    Why eigenmode fitting?                                                   #
#    ----------------------                                                   #
#    P(iΩ) in the density-fitting (DF) basis is NOT diagonal: the ratio      #
#    ||off-diagonal|| / ||diagonal|| is typically 0.5–1.0, so fitting        #
#    raw DF diagonals gives poor single-pole approximations.                  #
#    P(iΩ) is also low-rank (rank = n_val × n_con × nk in DF space).         #
#    The correct approach:                                                     #
#      1. Symmetrise and diagonalise P(Ω=0) → eigenvectors U  (fixed)        #
#      2. Rotate P(iΩ) into the eigenbasis: P̃(iΩ) = U† P(iΩ) U             #
#      3. Fit a plasmon pole to each diagonal P̃_{mm}(iΩ)                    #
#      4. A_m(ω) = −Im P̃_{mm}(ω+iη)/π  ;  A(ω) = Σ_m A_m(ω)              #
#    The trace (= physical spectrum) is basis-independent:                    #
#      Σ_m A_m(ω) = −Im Tr P(ω+iη)/π                                        #
#                                                                             #
#    All results are written to a single HDF5 file organised by q-index.     #
#                                                                             #
#    Usage                                                                    #
#    -----                                                                    #
#    # q = 0 only                                                             #
#    python script/solve_bse.py --dir example/H2_correct_fq_8x1x1k          #
#                                                                             #
#    # All q-points detected automatically                                    #
#    python script/solve_bse.py --dir example/H2_correct_fq_8x1x1k \        #
#        --all_q                                                              #
#                                                                             #
#    # Explicit subset with singlet BSE                                       #
#    python script/solve_bse.py --dir example/hBN_sto-3g_4x4x1k \           #
#        --q_idx 0 1 2 3 --kappa 1.0                                         #
#                                                                             #
#    Output HDF5 layout                                                       #
#    -------------------                                                      #
#    /params/...                       run parameters                         #
#    /omega                (npts,)    real-axis grid (Ha)                     #
#    /q_indices            (nq,)      q-indices computed                      #
#    /q{n}/q_frac          (3,)       fractional q vector                    #
#    /q{n}/Omega           (niw,)     Matsubara grid (Ha)                    #
#    --- raw DF-basis data (diagnostics) ---                                  #
#    /q{n}/P0_iw_diag      (niw, NQ)  Re P⁰_{QQ}(iΩ)                        #
#    /q{n}/Pbse_iw_diag    (niw, NQ)  Re P_BSE_{QQ}(iΩ)                     #
#    --- eigenmode basis (used for AC) ---                                    #
#    /q{n}/P0_eigvals      (NQ,)      eigenvalues of P⁰(Ω=0), sorted desc.  #
#    /q{n}/P0_eigvecs      (NQ, NQ)   columns = eigenvectors U               #
#    /q{n}/P0_eig_iw       (niw, NQ)  diag of U†P⁰(iΩ)U  — what is fitted  #
#    /q{n}/P0_eig_fit      (niw, NQ)  plasmon-pole model per eigenmode       #
#    /q{n}/Pbse_eigvals    (NQ,)                                              #
#    /q{n}/Pbse_eigvecs    (NQ, NQ)                                           #
#    /q{n}/Pbse_eig_iw     (niw, NQ)                                         #
#    /q{n}/Pbse_eig_fit    (niw, NQ)                                         #
#    --- spectral functions (sum of eigenmodes = trace) ---                   #
#    /q{n}/A_P0            (npts, NQ) per-eigenmode spectral function        #
#    /q{n}/A_Pbse          (npts, NQ)                                         #
#    /q{n}/P0_peak_eV      scalar     peak of Σ_m A_m(ω)  (eV)              #
#    /q{n}/Pbse_peak_eV    scalar                                             #
#    /q{n}/pp/P0_Finf      (NQ,)      plasmon-pole parameters per eigenmode  #
#    /q{n}/pp/P0_S         (NQ,)                                             #
#    /q{n}/pp/P0_wp        (NQ,)                                             #
#    /q{n}/pp/Pbse_Finf    (NQ,)                                             #
#    /q{n}/pp/Pbse_S       (NQ,)                                             #
#    /q{n}/pp/Pbse_wp      (NQ,)                                             #

import argparse
import glob
import os
import sys
import tempfile

import h5py
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BSE_DIR    = os.path.join(_SCRIPT_DIR, '..', 'green_bse')
sys.path.insert(0, _BSE_DIR)

from polarization          import PolarizationConfig, PolarizationSolver
from bse_kernel_qq         import BSEKernelConfig, BSEKernelQQ, read_polarization_h5
from analytic_continuation import fit_ac, omega_grid, spectral_peak_eV

AU2EV = 27.211386245981


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available_q_indices(int_fq_path: str) -> list:
    """Return sorted list of q-indices with a VQ_q{n}.h5 file present."""
    indices = []
    for p in glob.glob(os.path.join(int_fq_path, 'VQ_q*.h5')):
        name = os.path.basename(p)
        try:
            indices.append(int(name[4:-3]))   # strip 'VQ_q' and '.h5'
        except ValueError:
            pass
    return sorted(indices)


def _q_frac(mf_h5: str, q_idx: int) -> np.ndarray:
    """Fractional q vector for a given q-index from mean_field_input.h5."""
    with h5py.File(mf_h5, 'r') as f:
        ksc = f['grid/k_mesh_scaled'][()]
    return ksc[q_idx]


def _int_path(example_dir: str) -> str:
    """Prefer df_hf_int_fq (has all q including q=0 copy); fall back to df_hf_int."""
    fq = os.path.join(example_dir, 'df_hf_int_fq')
    return fq if os.path.isdir(fq) else os.path.join(example_dir, 'df_hf_int')


# AC functions are in green_bse/analytic_continuation.py (fit_ac, omega_grid, spectral_peak_eV)


# ---------------------------------------------------------------------------
# Single-q solver
# ---------------------------------------------------------------------------

def solve_q(q_idx: int, example_dir: str,
            beta: float, omega: np.ndarray, eta: float,
            kappa: float,
            tmpdir: str) -> dict:
    """
    Compute P⁰ and P_BSE at momentum transfer q_idx.

    Returns a dict with Matsubara data, fit parameters, and per-diagonal
    spectral functions ready to be written to the output HDF5.
    """
    input_h5 = os.path.join(example_dir, 'mean_field_input.h5')
    sim_h5   = os.path.join(example_dir, 'sim.h5')
    int_dir  = _int_path(example_dir)
    ir_file  = os.path.join(example_dir, 'irgrid', '1e5.h5')

    pol_out = os.path.join(tmpdir, f'pol_q{q_idx}.h5')
    xi_out  = os.path.join(tmpdir, f'xi_q{q_idx}.h5')

    # --- P⁰ via PolarizationSolver ---
    PolarizationSolver(PolarizationConfig(
        input_file  = input_h5,
        sim_file    = sim_h5,
        int_path    = int_dir,
        ir_file     = ir_file,
        beta        = beta,
        q_idx       = q_idx,
        iteration   = -1,
        output_file = pol_out,
    )).run()

    pol   = read_polarization_h5(pol_out)
    P0_iw = pol['P0_iw']           # (niw, ns, 1, NQ, NQ)

    # --- P_BSE via BSEKernelQQ ---
    BSEKernelQQ(BSEKernelConfig(
        pol_files   = [pol_out],
        output_file = xi_out,
        kappa       = kappa,
    )).run()

    # New layout: /P_bse (Nq=1, niw, NQ, NQ) → take q-slot 0
    with h5py.File(xi_out, 'r') as f:
        P_bse = f['P_bse'][0].astype(np.complex128)   # (niw, NQ, NQ)

    Omega = omega_grid(ir_file, beta)
    NQ    = P0_iw.shape[-1]
    idx_Q = np.arange(NQ)

    # fit_ac expects (niw, ns, 1, NQ, NQ); add back spin and q-slot dims
    P_bse_5d = P_bse[:, np.newaxis, np.newaxis, :, :]   # (niw, 1, 1, NQ, NQ)

    # --- Eigenmode AC: diagonalise at Ω=0, fit per eigenmode ---
    (A_P0,   P0_fit_eig,   P0_eigvals,   P0_U,   P0_eig_iw,
     P0_Finf,   P0_S,   P0_wp)   = fit_ac(P0_iw, Omega, omega, eta)

    (A_Pbse, Pbse_fit_eig, Pbse_eigvals, Pbse_U, Pbse_eig_iw,
     Pbse_Finf, Pbse_S, Pbse_wp) = fit_ac(P_bse_5d, Omega, omega, eta)

    return dict(
        q_frac          = _q_frac(input_h5, q_idx),
        Omega           = Omega,
        # Raw DF-basis diagonals (for diagnostics)
        P0_iw_diag      = P0_iw[:, 0, 0, idx_Q, idx_Q].real,   # (niw, NQ)
        Pbse_iw_diag    = P_bse[:, idx_Q, idx_Q].real,          # (niw, NQ)
        # Eigenmode basis
        P0_eigvals      = P0_eigvals,       # (NQ,)
        P0_eigvecs      = P0_U,             # (NQ, NQ)
        P0_eig_iw       = P0_eig_iw,        # (niw, NQ)  — what is fitted
        P0_eig_fit      = P0_fit_eig,       # (niw, NQ)  — plasmon-pole model
        Pbse_eigvals    = Pbse_eigvals,
        Pbse_eigvecs    = Pbse_U,
        Pbse_eig_iw     = Pbse_eig_iw,
        Pbse_eig_fit    = Pbse_fit_eig,
        # Spectral functions (per eigenmode; sum = trace)
        A_P0            = A_P0,             # (npts, NQ)
        A_Pbse          = A_Pbse,
        P0_peak_eV      = spectral_peak_eV(A_P0,   omega),
        Pbse_peak_eV    = spectral_peak_eV(A_Pbse, omega),
        # Plasmon-pole parameters per eigenmode
        P0_Finf         = P0_Finf,
        P0_S            = P0_S,
        P0_wp           = P0_wp,
        Pbse_Finf       = Pbse_Finf,
        Pbse_S          = Pbse_S,
        Pbse_wp         = Pbse_wp,
    )


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_results(output_file: str, results_by_q: dict,
                  omega: np.ndarray, q_indices: list,
                  params: dict):
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('omega',     data=omega.astype(np.float64))
        f.create_dataset('q_indices', data=np.array(q_indices, dtype=np.int64))

        pg = f.create_group('params')
        for k, v in params.items():
            pg.create_dataset(k, data=v)

        for q_idx, res in results_by_q.items():
            g = f.create_group(f'q{q_idx}')
            g.create_dataset('q_frac',        data=res['q_frac'].astype(np.float64))
            g.create_dataset('Omega',          data=res['Omega'].astype(np.float64))

            # Raw DF-basis diagonals (diagnostics)
            g.create_dataset('P0_iw_diag',    data=res['P0_iw_diag'].astype(np.float64))
            g.create_dataset('Pbse_iw_diag',  data=res['Pbse_iw_diag'].astype(np.float64))

            # Eigenmode basis
            g.create_dataset('P0_eigvals',    data=res['P0_eigvals'].astype(np.float64))
            g.create_dataset('P0_eigvecs',    data=res['P0_eigvecs'].astype(np.float64))
            g.create_dataset('P0_eig_iw',     data=res['P0_eig_iw'].astype(np.float64))
            g.create_dataset('P0_eig_fit',    data=res['P0_eig_fit'].astype(np.float64))
            g.create_dataset('Pbse_eigvals',  data=res['Pbse_eigvals'].astype(np.float64))
            g.create_dataset('Pbse_eigvecs',  data=res['Pbse_eigvecs'].astype(np.float64))
            g.create_dataset('Pbse_eig_iw',   data=res['Pbse_eig_iw'].astype(np.float64))
            g.create_dataset('Pbse_eig_fit',  data=res['Pbse_eig_fit'].astype(np.float64))

            # Per-eigenmode spectral functions; sum over axis=1 = trace spectrum
            g.create_dataset('A_P0',          data=res['A_P0'].astype(np.float64))
            g.create_dataset('A_Pbse',        data=res['A_Pbse'].astype(np.float64))
            g.create_dataset('P0_peak_eV',    data=float(res['P0_peak_eV']))
            g.create_dataset('Pbse_peak_eV',  data=float(res['Pbse_peak_eV']))

            pp = g.create_group('pp')
            pp.create_dataset('P0_Finf',   data=res['P0_Finf'].astype(np.float64))
            pp.create_dataset('P0_S',      data=res['P0_S'].astype(np.float64))
            pp.create_dataset('P0_wp',     data=res['P0_wp'].astype(np.float64))
            pp.create_dataset('Pbse_Finf', data=res['Pbse_Finf'].astype(np.float64))
            pp.create_dataset('Pbse_S',    data=res['Pbse_S'].astype(np.float64))
            pp.create_dataset('Pbse_wp',   data=res['Pbse_wp'].astype(np.float64))

    print(f'\nResults written → {output_file}')


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(q_indices: list, results_by_q: dict, params: dict):
    sep = '-' * 65
    print(f'\n{sep}')
    print(f'  BSE solver summary   kappa={params["kappa"]:.2f}')
    print(f'  beta={params["beta"]:.0f} Ha⁻¹   eta={params["eta"]*AU2EV:.3f} eV')
    print(sep)
    print(f'  {"q_idx":>6}  {"q_frac":^22}  '
          f'{"P0 peak (eV)":>14}  {"PBSE peak (eV)":>14}')
    print(sep)
    for q_idx in q_indices:
        res = results_by_q[q_idx]
        qf  = res['q_frac']
        print(f'  {q_idx:>6}  ({qf[0]:+.4f},{qf[1]:+.4f},{qf[2]:+.4f})  '
              f'{res["P0_peak_eV"]:>14.3f}  {res["Pbse_peak_eV"]:>14.3f}')
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Compute P⁰ and P_BSE at one or more q-points, fit per-diagonal '
            'plasmon poles, and write all results (including per-diagonal '
            'spectral functions) to HDF5.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O ---
    parser.add_argument('--dir', required=True, metavar='DIR',
                        help='Example directory (must contain mean_field_input.h5, '
                             'sim.h5, irgrid/1e5.h5, df_hf_int_fq/).')
    parser.add_argument('--output', default='bse_results.h5', metavar='FILE',
                        help='Output HDF5 file.')

    # --- q-point selection ---
    qgrp = parser.add_mutually_exclusive_group()
    qgrp.add_argument('--q_idx', type=int, nargs='+', metavar='N',
                      help='One or more q-indices to compute (space-separated).')
    qgrp.add_argument('--all_q', action='store_true',
                      help='Auto-detect and compute all available q-indices '
                           '(including q=0).')

    # --- Physics ---
    parser.add_argument('--kappa', type=float, default=1.0,
                        help='Exchange mixing: 1.0=singlet BSE (default), 0.0=triplet.')
    parser.add_argument('--beta', type=float, default=1000.0,
                        help='Inverse temperature (Ha⁻¹).')
    parser.add_argument('--eta', type=float, default=0.005,
                        help='Lorentzian broadening for real-axis (Ha).')
    parser.add_argument('--omega_max', type=float, default=1.5,
                        help='Upper bound of real-axis frequency grid (Ha).')
    parser.add_argument('--npts', type=int, default=2000,
                        help='Number of real-axis frequency points.')

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.dir):
        sys.exit(f'ERROR: directory not found: {args.dir}')
    for fname in ('mean_field_input.h5', 'sim.h5',
                  os.path.join('irgrid', '1e5.h5')):
        if not os.path.exists(os.path.join(args.dir, fname)):
            sys.exit(f'ERROR: required file missing: {os.path.join(args.dir, fname)}')

    # Determine q-indices
    int_fq = os.path.join(args.dir, 'df_hf_int_fq')
    if args.all_q:
        fq_indices = _available_q_indices(int_fq) if os.path.isdir(int_fq) else []
        q_indices  = [0] + fq_indices if 0 not in fq_indices else fq_indices
        if not q_indices:
            sys.exit(f'ERROR: no VQ files found in {int_fq}')
    elif args.q_idx is not None:
        q_indices = args.q_idx
    else:
        q_indices = [0]

    omega = np.linspace(1e-4, args.omega_max, args.npts)

    params = dict(
        beta      = args.beta,
        eta       = args.eta,
        omega_max = args.omega_max,
        npts      = args.npts,
        kappa     = args.kappa,
    )

    print(f'Directory  : {args.dir}')
    print(f'q-indices  : {q_indices}')
    print(f'kappa      : {args.kappa}')
    print(f'beta       : {args.beta} Ha⁻¹   eta={args.eta*AU2EV:.3f} eV   '
          f'omega_max={args.omega_max*AU2EV:.1f} eV')

    results_by_q = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for q_idx in q_indices:
            print(f'\n{"="*60}')
            print(f'  q_idx = {q_idx}')
            print(f'{"="*60}')
            res = solve_q(
                q_idx       = q_idx,
                example_dir = args.dir,
                beta        = args.beta,
                omega       = omega,
                eta         = args.eta,
                kappa       = args.kappa,
                tmpdir      = tmpdir,
            )
            results_by_q[q_idx] = res
            NQ = res['A_P0'].shape[1]
            # Count significant eigenmodes (non-zero A contribution)
            n_sig_p0   = int((res['P0_wp']   > 0).sum())
            n_sig_pbse = int((res['Pbse_wp'] > 0).sum())
            ev_p0   = res['P0_eigvals']
            ev_pbse = res['Pbse_eigvals']
            print(f'  NQ = {NQ}  |  P⁰ significant modes = {n_sig_p0}'
                  f'  (top eigenvals: {np.round(ev_p0[:4], 4)})')
            print(f'           |  P_BSE significant modes = {n_sig_pbse}'
                  f'  (top eigenvals: {np.round(ev_pbse[:4], 4)})')
            print(f'  P⁰   spectral peak (Σ_m A_m) = {res["P0_peak_eV"]:.3f} eV')
            print(f'  P_BSE spectral peak (Σ_m A_m) = {res["Pbse_peak_eV"]:.3f} eV')

    _print_summary(q_indices, results_by_q, params)
    write_results(args.output, results_by_q, omega, q_indices, params)


if __name__ == '__main__':
    main()
