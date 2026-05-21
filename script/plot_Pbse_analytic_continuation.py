#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.   #
#                                                                             #
#    Analytic continuation of P⁰ and P_BSE for the H chain (8k) example.    #
#                                                                             #
#    For each requested q-point the script:                                  #
#      1. Runs PolarizationSolver → P⁰(iΩ), W(iΩ).                          #
#      2. Runs BSEKernelQQ (full exchange at finite q, head-only at q=0)     #
#         → Ξ(iΩ), P_BSE(iΩ).                                               #
#      3. Fits Tr[P(iΩ)] of both P⁰ and P_BSE to a single plasmon-pole      #
#         model on the imaginary axis.                                        #
#      4. Evaluates on the real axis and plots:                               #
#           Left  – imaginary axis: Tr[P(iΩ)] data + plasmon-pole fit        #
#           Right – real axis:      spectral function –Im Tr[P(ω+iη)]/π      #
#                                                                             #
#    Usage                                                                    #
#    -----                                                                    #
#    python script/plot_Pbse_analytic_continuation.py                        #
#        [--example_dir PATH]                                                 #
#        [--q_indices 0 1 2]                                                  #
#        [--beta 1000]                                                        #
#        [--eta 0.05]                                                         #
#        [--omega_max 2.0]    # in Hartree                                    #
#        [--npts 2000]                                                        #
#        [--output  P_bse_continuation.pdf]                                   #

import os
import sys
import argparse
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BSE_DIR    = os.path.join(_SCRIPT_DIR, '..', 'green_bse')
_ROOT_DIR   = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, _BSE_DIR)

from polarization   import PolarizationConfig, PolarizationSolver
from bse_kernel_qq  import (
    read_polarization_h5,
    BSEKernelConfig,
    BSEKernelQQ,
)
from plasPole import fit_plasmon_pole, plasmon_model

AU2EV = 27.211386245981


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_omega_grid(ir_file: str, beta: float) -> np.ndarray:
    """Return bosonic Matsubara frequencies Ω_n = 2nπ/β from an IR HDF5."""
    import h5py
    with h5py.File(ir_file, 'r') as f:
        ngrid = f['/bose/ngrid'][()]
    return 2.0 * ngrid * np.pi / beta


def _trace_iw(P_iw: np.ndarray) -> np.ndarray:
    """Tr_Q[P_{QQ}(iΩ)] for each Matsubara frequency.  Returns (niw,) real.

    P_iw : (niw, ns, 1, NQ, NQ)
    """
    niw = P_iw.shape[0]
    return np.array([np.trace(P_iw[iw, 0, 0]).real for iw in range(niw)])


def _fit_trace(Omega: np.ndarray, trace: np.ndarray):
    """
    Fit the imaginary-axis trace to a single plasmon-pole model.

    Uses Fdata[-1] as Finf estimate (high-Ω limit).  Returns dict with
    Finf, S, wp, residual_norm and the full fitted curve on Omega.
    """
    niw  = len(Omega)
    Finf = trace[-1]
    F0   = trace[niw // 2]
    fit  = fit_plasmon_pole(Omega, trace, F0=F0, Finf=Finf)
    z    = 1j * Omega
    curve = plasmon_model(z, fit['Finf'], fit['S'], fit['wp']).real
    return fit, curve


def _eval_real_axis(fit: dict, omega_real: np.ndarray, eta: float) -> np.ndarray:
    """
    Evaluate the plasmon-pole model for the trace on the real axis.

    Returns P_trace(ω+iη) as a complex (nomega,) array.
    """
    Finf = fit['Finf']
    S    = fit['S']
    wp   = fit['wp']
    z    = omega_real + 1j * eta
    return Finf + 2.0 * wp * S / (wp**2 - z**2)


def _spectral(P_real_trace: np.ndarray) -> np.ndarray:
    """A(ω) = –Im P(ω+iη) / π  (spectral function of the trace)."""
    return -P_real_trace.imag / np.pi


# ---------------------------------------------------------------------------
# Per-q pipeline
# ---------------------------------------------------------------------------

def _q_label(example_dir: str, q_idx: int) -> str:
    """Return a human-readable label for q_idx, e.g. 'q=0 (Γ)' or 'q=1 [0.25, 0, 0]'."""
    try:
        import h5py
        with h5py.File(os.path.join(example_dir, 'mean_field_input.h5'), 'r') as f:
            kmesh = f['grid/k_mesh_scaled'][()]    # (nk, 3) in fractional coords
        if q_idx == 0:
            return 'q=0 (Γ, optical)'
        qvec = kmesh[q_idx]
        return f'q={q_idx}  [{qvec[0]:.3f}, {qvec[1]:.3f}, {qvec[2]:.3f}]'
    except Exception:
        return f'q-idx={q_idx}'


def run_one_q(q_idx: int, example_dir: str, beta: float,
              omega_real: np.ndarray, eta: float,
              tmpdir: str) -> dict:
    """
    Run PolarizationSolver + BSEKernelQQ for one q-point, fit traces.

    Returns a dict with:
        Omega           (niw,) Matsubara grid
        P0_trace        (niw,) imaginary-axis trace of P⁰
        Pbse_trace      (niw,) imaginary-axis trace of P_BSE
        P0_fit_curve    (niw,) plasmon-pole fit on imaginary axis
        Pbse_fit_curve  (niw,)
        P0_fit          fit parameter dict
        Pbse_fit
        A_P0            (nomega,) spectral function of P⁰ on real axis
        A_Pbse          (nomega,) spectral function of P_BSE on real axis
    """
    input_h5 = os.path.join(example_dir, 'mean_field_input.h5')
    sim_h5   = os.path.join(example_dir, 'sim.h5')
    int_path = os.path.join(example_dir, 'df_hf_int_fq')
    ir_file  = os.path.join(example_dir, 'irgrid', '1e5.h5')

    pol_out = os.path.join(tmpdir, f'pol_q{q_idx}.h5')
    xi_out  = os.path.join(tmpdir, f'xi_q{q_idx}.h5')

    # ---- Step 1: polarization ----
    print(f"\n[q={q_idx}] Running PolarizationSolver …")
    PolarizationSolver(PolarizationConfig(
        input_file  = input_h5,
        sim_file    = sim_h5,
        int_path    = int_path,
        ir_file     = ir_file,
        beta        = beta,
        q_idx       = q_idx,
        iteration   = -1,
        output_file = pol_out,
    )).run()

    pol   = read_polarization_h5(pol_out)
    P0_iw = pol['P0_iw']
    W_iw  = pol['W_iw']

    # ---- Step 2: BSE kernel + Dyson ----
    print(f"[q={q_idx}] Running BSEKernelQQ …")
    BSEKernelQQ(BSEKernelConfig(
        pol_files   = [pol_out],
        output_file = xi_out,
        kappa       = 1.0,       # singlet; use 0.0 for triplet
    )).run()

    import h5py
    # New layout: /P_bse (Nq=1, niw, NQ, NQ) → take q-slot 0
    with h5py.File(xi_out, 'r') as f:
        P_bse = f['P_bse'][0].astype(np.complex128)   # (niw, NQ, NQ)

    # ---- Step 3: Matsubara grid and traces ----
    Omega      = _load_omega_grid(ir_file, beta)
    P0_trace   = _trace_iw(P0_iw)                     # P0_iw still 5D from PolarizationSolver
    niw        = P_bse.shape[0]
    Pbse_trace = np.array([np.trace(P_bse[iw]).real for iw in range(niw)])

    # ---- Step 4: plasmon-pole fits on imaginary axis ----
    print(f"[q={q_idx}] Fitting traces to plasmon-pole model …")
    P0_fit,   P0_fit_curve   = _fit_trace(Omega, P0_trace)
    Pbse_fit, Pbse_fit_curve = _fit_trace(Omega, Pbse_trace)

    print(f"  P⁰   fit: wp={P0_fit['wp']*AU2EV:.3f} eV, "
          f"S={P0_fit['S']:.4f}, residual={P0_fit['residual_norm']:.3e}")
    print(f"  P_BSE fit: wp={Pbse_fit['wp']*AU2EV:.3f} eV, "
          f"S={Pbse_fit['S']:.4f}, residual={Pbse_fit['residual_norm']:.3e}")

    # ---- Step 5: real-axis evaluation ----
    A_P0   = _spectral(_eval_real_axis(P0_fit,   omega_real, eta))
    A_Pbse = _spectral(_eval_real_axis(Pbse_fit, omega_real, eta))

    return dict(
        Omega           = Omega,
        P0_trace        = P0_trace,
        Pbse_trace      = Pbse_trace,
        P0_fit_curve    = P0_fit_curve,
        Pbse_fit_curve  = Pbse_fit_curve,
        P0_fit          = P0_fit,
        Pbse_fit        = Pbse_fit,
        A_P0            = A_P0,
        A_Pbse          = A_Pbse,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

C_P0   = '#4e88c7'   # blue – bare polarization / RPA
C_PBSE = '#e05c5c'   # red  – BSE-level polarization


def plot_results(results: dict, q_indices: list, omega_real: np.ndarray,
                 eta: float, output: str, example_dir: str = ''):
    """
    Two-column figure:
      Left  column: imaginary axis Tr[P(iΩ)] + plasmon-pole fit
      Right column: real-axis spectral function –Im Tr[P(ω+iη)]/π
    One row per q-point.
    """
    nq    = len(q_indices)
    fig, axes = plt.subplots(nq, 2, figsize=(7.0, 2.6 * nq),
                             squeeze=False)
    omega_eV = omega_real * AU2EV
    eta_eV   = eta * AU2EV

    for row, q_idx in enumerate(q_indices):
        res = results[q_idx]
        Omega = res['Omega']

        # ---- left: imaginary axis ----
        ax = axes[row, 0]
        Omega_eV = Omega * AU2EV

        ax.plot(Omega_eV, res['P0_trace'],
                'o', color=C_P0,   ms=2.5, label=r'$\mathrm{Tr}\,P^0(i\Omega)$')
        ax.plot(Omega_eV, res['P0_fit_curve'],
                '-', color=C_P0,   lw=1.2, label='PP fit')

        ax.plot(Omega_eV, res['Pbse_trace'],
                's', color=C_PBSE, ms=2.5, label=r'$\mathrm{Tr}\,P_\mathrm{BSE}(i\Omega)$')
        ax.plot(Omega_eV, res['Pbse_fit_curve'],
                '-', color=C_PBSE, lw=1.2, label='PP fit')

        # Focus on the physically relevant window: a few times the larger pole
        wp_max = max(res['P0_fit']['wp'], res['Pbse_fit']['wp']) * AU2EV
        xlim   = min(max(wp_max * 4, 10.0), 200.0)   # cap at 200 eV
        ax.set_xlim(-xlim, xlim)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xlabel(r'$i\Omega_n$ (eV)')
        ax.set_ylabel(r'$\mathrm{Tr}\,P(i\Omega_n)$')
        ax.legend(fontsize=7, frameon=False, loc='lower right')
        q_label = _q_label(example_dir, q_idx)
        ax.set_title(q_label, fontsize=9)

        # annotation: pole positions
        wp0_eV   = res['P0_fit']['wp']   * AU2EV
        wpBSE_eV = res['Pbse_fit']['wp'] * AU2EV
        ax.text(0.03, 0.05,
                fr"$\omega_p^{{P^0}}={wp0_eV:.2f}$ eV"
                "\n"
                fr"$\omega_p^{{P_\mathrm{{BSE}}}}={wpBSE_eV:.2f}$ eV",
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', lw=0.8))

        # ---- right: real axis ----
        ax2 = axes[row, 1]
        ax2.plot(omega_eV, res['A_P0'],
                 '-', color=C_P0,   lw=1.4, label=r'$-\mathrm{Im}\,\mathrm{Tr}\,P^0(\omega)/\pi$')
        ax2.fill_between(omega_eV, res['A_P0'],
                         alpha=0.2, color=C_P0)

        ax2.plot(omega_eV, res['A_Pbse'],
                 '-', color=C_PBSE, lw=1.4, label=r'$-\mathrm{Im}\,\mathrm{Tr}\,P_\mathrm{BSE}(\omega)/\pi$')
        ax2.fill_between(omega_eV, res['A_Pbse'],
                         alpha=0.2, color=C_PBSE)

        ax2.axhline(0, color='k', lw=0.5, ls='--')
        ax2.set_xlabel(r'$\omega$ (eV)')
        ax2.set_ylabel(r'$-\mathrm{Im}\,\mathrm{Tr}\,P(\omega)/\pi$')
        ax2.legend(fontsize=7, frameon=False, loc='upper right')
        ax2.set_title(f'{q_label}   η={eta_eV:.3f} eV', fontsize=9)

        # mark pole positions with vertical lines
        ax2.axvline(wp0_eV,   color=C_P0,   lw=0.8, ls='--', alpha=0.7)
        ax2.axvline(wpBSE_eV, color=C_PBSE, lw=0.8, ls='--', alpha=0.7)

    fig.suptitle('H chain (8k)  —  $P^0$ vs $P_\\mathrm{BSE}$  analytic continuation',
                 fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analytic continuation of P⁰ and P_BSE for the H chain (8k).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--example_dir', default=None,
        help='Path to H2_fq_8k example directory.')
    parser.add_argument('--q_indices', type=int, nargs='+', default=[0, 1, 2],
        help='q-point indices to compute (e.g. 0 1 2).')
    parser.add_argument('--beta',     type=float, default=1000.0,
        help='Inverse temperature β (a.u.).')
    parser.add_argument('--eta',      type=float, default=0.05,
        help='Lorentzian broadening η for real-axis evaluation (a.u.).')
    parser.add_argument('--omega_max', type=float, default=2.0,
        help='Maximum real-axis frequency for the plot (a.u.).')
    parser.add_argument('--npts',     type=int,   default=2000,
        help='Number of points on the real-axis frequency grid.')
    parser.add_argument('--output',   default='P_bse_continuation.pdf',
        help='Output figure filename.')
    args = parser.parse_args()

    example_dir = args.example_dir or os.path.join(
        _ROOT_DIR, 'example', 'H2_fq_8k')

    # Verify example directory
    for name in ['mean_field_input.h5', 'sim.h5', 'df_hf_int_fq', 'irgrid']:
        path = os.path.join(example_dir, name)
        if not os.path.exists(path):
            print(f"ERROR: required path not found: {path}")
            sys.exit(1)

    omega_real = np.linspace(1e-4, args.omega_max, args.npts)

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for q_idx in args.q_indices:
            results[q_idx] = run_one_q(
                q_idx       = q_idx,
                example_dir = example_dir,
                beta        = args.beta,
                omega_real  = omega_real,
                eta         = args.eta,
                tmpdir      = tmpdir,
            )

    plot_results(results, args.q_indices, omega_real, args.eta, args.output,
                 example_dir=example_dir)


if __name__ == '__main__':
    main()
