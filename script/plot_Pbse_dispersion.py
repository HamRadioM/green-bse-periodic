#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.   #
#                                                                             #
#    P⁰ vs P_BSE analytic continuation — optical detail + q-dispersion       #
#    heatmap.  Uses eigenmode diagonalisation for the AC step.                #
#                                                                             #
#    Layout                                                                   #
#    ------                                                                   #
#    Row 0 (2 panels): q = 0 detail                                           #
#        Left  – imaginary axis:  Σ_m P̃⁰_{mm}(iΩ) data + fitted model       #
#        Right – real axis:       Σ_m A^⁰_m(ω)  vs  Σ_m A^BSE_m(ω)         #
#    Row 1 (2 panels): heatmap of spectral function over finite-q             #
#        Left  – P⁰   A(q, ω) = Σ_m A^⁰_m(q, ω)                            #
#        Right – P_BSE A(q, ω) = Σ_m A^BSE_m(q, ω)                          #
#                                                                             #
#    Usage                                                                    #
#    -----                                                                    #
#    python script/plot_Pbse_dispersion.py \                                  #
#        --q0_dir  example/H2_correct_fq_8x1x1k \                            #
#        --fq_dir  example/H2_correct_fq_8x1x1k \                            #
#        [--beta 1000] [--eta 0.005] [--omega_max 1.5] [--npts 2000] \       #
#        [--kappa 1.0]  \                                   #
#        [--output dispersion.pdf]                                             #
#        [--save_q0 dispersion_q0.h5]   # default: <output stem>_q0.h5       #
#                                                                             #
#    q=0 HDF5 layout  (written to --save_q0)                                 #
#    -----------------------------------------                                #
#    /Omega              (niw,)       Matsubara grid (Ha)                     #
#    /omega              (npts,)      real-axis grid (Ha)                     #
#    /P0_iw              (niw,NQ,NQ)  bare polarization P⁰(iΩ) [complex]     #
#    /Xi_iw              (niw,NQ,NQ)  BSE kernel Ξ(iΩ) = 2U − W(iΩ)         #
#    /P_bse              (niw,NQ,NQ)  BSE polarizability P_BSE(iΩ)           #
#    /P0_eigvals         (NQ,)        eigenvalues of P⁰(Ω=0), |·| desc.      #
#    /P0_eigvecs         (NQ,NQ)      eigenvectors (columns)                  #
#    /P0_eig_iw          (niw,NQ)     diag(U†P⁰U)(iΩ)  — fitted data         #
#    /P0_eig_fit         (niw,NQ)     plasmon-pole model                      #
#    /Pbse_eigvals, /Pbse_eigvecs, /Pbse_eig_iw, /Pbse_eig_fit  (same)       #
#    /pp/P0_Finf         (NQ,)        plasmon-pole parameters per eigenmode   #
#    /pp/P0_S            (NQ,)                                                #
#    /pp/P0_wp           (NQ,)                                                #
#    /pp/Pbse_{Finf,S,wp}(NQ,)       (same for P_BSE)                        #
#    /A_P0               (npts,NQ)    spectral function per eigenmode         #
#    /A_Pbse             (npts,NQ)                                            #

import glob
import os
import sys
import argparse
import tempfile

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BSE_DIR    = os.path.join(_SCRIPT_DIR, '..', 'green_bse')
_ROOT_DIR   = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, _BSE_DIR)

from polarization          import PolarizationConfig, PolarizationSolver
from bse_kernel_qq         import BSEKernelConfig, BSEKernelQQ, read_polarization_h5
from analytic_continuation import fit_ac, omega_grid, spectral_peak_eV

AU2EV = 27.211386245981


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _available_q_indices(fq_int_path: str) -> list:
    files = glob.glob(os.path.join(fq_int_path, 'VQ_q*.h5'))
    indices = []
    for p in files:
        try:
            indices.append(int(os.path.basename(p)[4:-3]))
        except ValueError:
            pass
    return sorted(indices)


def _kmesh_qx(mf_h5: str) -> np.ndarray:
    with h5py.File(mf_h5, 'r') as f:
        return f['grid/k_mesh_scaled'][()][:, 0]


def _int_path(example_dir: str) -> str:
    fq = os.path.join(example_dir, 'df_hf_int_fq')
    return fq if os.path.isdir(fq) else os.path.join(example_dir, 'df_hf_int')


def _fq_save_path(q0_path: str, q_idx: int) -> str:
    """Derive the per-q save path from the q=0 save path."""
    if q0_path.endswith('_q0.h5'):
        return q0_path[:-len('_q0.h5')] + f'_q{q_idx}.h5'
    stem, _ = os.path.splitext(q0_path)
    return f'{stem}_q{q_idx}.h5'


# ---------------------------------------------------------------------------
# Per-q computation
# ---------------------------------------------------------------------------

def run_q(q_idx: int, example_dir: str,
          beta: float, omega: np.ndarray, eta: float,
          kappa: float, tmpdir: str,
          keep_raw: bool = False) -> dict:
    """
    Run PolarizationSolver + BSEKernelQQ for one q-point, then apply the
    eigenmode analytic continuation.

    Parameters
    ----------
    keep_raw : bool
        If True, include the full raw matrices P0_iw, Xi_iw, P_bse in the
        returned dict (needed for saving intermediate data at q=0).

    Returns
    -------
    dict with keys:
      Omega          (niw,)     Matsubara grid
      P0_eig_iw      (niw, NQ)  raw eigenmode traces — data points plotted
      Pbse_eig_iw    (niw, NQ)
      P0_eig_fit     (niw, NQ)  plasmon-pole model per eigenmode
      Pbse_eig_fit   (niw, NQ)
      P0_eigvals     (NQ,)      eigenvalues at Ω=0
      P0_eigvecs     (NQ, NQ)   eigenvectors (columns)
      Pbse_eigvals   (NQ,)
      Pbse_eigvecs   (NQ, NQ)
      A_P0           (npts, NQ) per-eigenmode spectral function
      A_Pbse         (npts, NQ)
      P0_peak_eV     float      peak of Σ_m A^P0_m
      Pbse_peak_eV   float
      # only when keep_raw=True:
      P0_iw          (niw, NQ, NQ)  bare polarization (spin-squeezed, real part)
      Xi_iw          (niw, NQ, NQ)  BSE kernel Ξ = 2U − W
      P_bse          (niw, NQ, NQ)  BSE polarizability (spin-squeezed, real part)
    """
    input_h5 = os.path.join(example_dir, 'mean_field_input.h5')
    sim_h5   = os.path.join(example_dir, 'sim.h5')
    int_dir  = _int_path(example_dir)
    ir_file  = os.path.join(example_dir, 'irgrid', '1e5.h5')

    pol_out = os.path.join(tmpdir, f'pol_q{q_idx}.h5')
    xi_out  = os.path.join(tmpdir, f'xi_q{q_idx}.h5')

    # --- P⁰ ---
    PolarizationSolver(PolarizationConfig(
        input_file=input_h5, sim_file=sim_h5,
        int_path=int_dir,    ir_file=ir_file,
        beta=beta, q_idx=q_idx, iteration=-1,
        output_file=pol_out,
    )).run()

    pol   = read_polarization_h5(pol_out)
    P0_iw = pol['P0_iw']                    # (niw, ns, 1, NQ, NQ)

    # --- P_BSE ---
    BSEKernelQQ(BSEKernelConfig(
        pol_files=[pol_out], output_file=xi_out,
        kappa=kappa,
    )).run()

    # New layout: /Xi_iw and /P_bse have shape (Nq=1, niw, NQ, NQ) → take q-slot 0
    with h5py.File(xi_out, 'r') as f:
        P_bse = f['P_bse'][0].astype(np.complex128)   # (niw, NQ, NQ)
        Xi_iw = f['Xi_iw'][0].astype(np.complex128)   # (niw, NQ, NQ)

    Omega = omega_grid(ir_file, beta)

    # fit_ac expects (niw, ns, 1, NQ, NQ); add back spin and q-slot dims
    P_bse_5d = P_bse[:, np.newaxis, np.newaxis, :, :]   # (niw, 1, 1, NQ, NQ)

    # --- Eigenmode AC ---
    (A_P0,   P0_fit,   P0_ev,   P0_U,   P0_eig_iw,
     P0_Finf, P0_S, P0_wp) = fit_ac(P0_iw, Omega, omega, eta)
    (A_Pbse, Pbse_fit, Pbse_ev, Pbse_U, Pbse_eig_iw,
     Pbse_Finf, Pbse_S, Pbse_wp) = fit_ac(P_bse_5d, Omega, omega, eta)

    result = dict(
        Omega          = Omega,
        P0_eig_iw      = P0_eig_iw,      # (niw, NQ) — what was fitted
        Pbse_eig_iw    = Pbse_eig_iw,
        P0_eig_fit     = P0_fit,          # (niw, NQ) — plasmon-pole model
        Pbse_eig_fit   = Pbse_fit,
        P0_eigvals     = P0_ev,
        P0_eigvecs     = P0_U,            # (NQ, NQ)
        Pbse_eigvals   = Pbse_ev,
        Pbse_eigvecs   = Pbse_U,
        A_P0           = A_P0,            # (npts, NQ)
        A_Pbse         = A_Pbse,
        P0_peak_eV     = spectral_peak_eV(A_P0,   omega),
        Pbse_peak_eV   = spectral_peak_eV(A_Pbse, omega),
        # plasmon-pole parameters per eigenmode
        P0_Finf=P0_Finf,   P0_S=P0_S,   P0_wp=P0_wp,
        Pbse_Finf=Pbse_Finf, Pbse_S=Pbse_S, Pbse_wp=Pbse_wp,
    )

    if keep_raw:
        # P0_iw is still (niw, ns, 1, NQ, NQ) from PolarizationSolver; squeeze it
        def _sq5(arr):
            if arr.ndim == 5 and arr.shape[2] == 1:
                return arr[:, 0, 0, :, :]
            elif arr.ndim == 5 and arr.shape[-1] == 1:
                return arr[:, 0, :, :, 0]
            return arr
        result['P0_iw_raw']  = _sq5(P0_iw)   # (niw, NQ, NQ) complex
        result['Xi_iw']      = Xi_iw          # (niw, NQ, NQ) complex  [already squeezed]
        result['P_bse_raw']  = P_bse          # (niw, NQ, NQ) complex  [already squeezed]

    return result


# ---------------------------------------------------------------------------
# Save intermediate q=0 data to HDF5
# ---------------------------------------------------------------------------

def save_q_data(res: dict, q_idx: int, omega: np.ndarray, path: str):
    """
    Write all intermediate quantities for one q-point to an HDF5 file.

    Layout
    ------
    /Omega              (niw,)       Matsubara grid (Ha)
    /omega              (npts,)      real-axis grid (Ha)
    --- raw frequency-dependent matrices (spin-squeezed, complex) ---
    /P0_iw              (niw, NQ, NQ)  bare polarization bubble P⁰(iΩ)
    /Xi_iw              (niw, NQ, NQ)  BSE kernel Ξ(iΩ) = 2U − W(iΩ)
    /P_bse              (niw, NQ, NQ)  BSE polarizability P_BSE(iΩ)
    --- eigenmode rotation (from diagonalising P(Ω=0)) ---
    /P0_eigvals         (NQ,)          eigenvalues of P⁰(Ω=0), |·| descending
    /P0_eigvecs         (NQ, NQ)       columns = eigenvectors U
    /Pbse_eigvals       (NQ,)
    /Pbse_eigvecs       (NQ, NQ)
    --- eigenmode projections and fits ---
    /P0_eig_iw          (niw, NQ)      diag(U†P⁰U)(iΩ)  — data that was fitted
    /P0_eig_fit         (niw, NQ)      plasmon-pole model per eigenmode
    /Pbse_eig_iw        (niw, NQ)
    /Pbse_eig_fit       (niw, NQ)
    --- plasmon-pole parameters ---
    /pp/P0_Finf         (NQ,)
    /pp/P0_S            (NQ,)
    /pp/P0_wp           (NQ,)          pole frequency (Ha)
    /pp/Pbse_Finf       (NQ,)
    /pp/Pbse_S          (NQ,)
    /pp/Pbse_wp         (NQ,)
    --- real-axis spectral functions ---
    /A_P0               (npts, NQ)     per-eigenmode: A_m = −Im P̃_mm(ω+iη)/π
    /A_Pbse             (npts, NQ)
    """
    with h5py.File(path, 'w') as f:
        f.attrs['description'] = (
            f'q_idx={q_idx} intermediate quantities from plot_Pbse_dispersion')
        f.attrs['q_idx'] = q_idx

        f.create_dataset('Omega',       data=res['Omega'])
        f.create_dataset('omega',       data=omega)

        # Raw matrices (present when keep_raw=True, which is always the case now)
        if 'P0_iw_raw' in res:
            f.create_dataset('P0_iw',   data=res['P0_iw_raw'])
            f.create_dataset('Xi_iw',   data=res['Xi_iw'])
            f.create_dataset('P_bse',   data=res['P_bse_raw'])

        # Eigenmode rotation
        f.create_dataset('P0_eigvals',  data=res['P0_eigvals'])
        f.create_dataset('P0_eigvecs',  data=res['P0_eigvecs'])
        f.create_dataset('Pbse_eigvals',data=res['Pbse_eigvals'])
        f.create_dataset('Pbse_eigvecs',data=res['Pbse_eigvecs'])

        # Projections and fits
        f.create_dataset('P0_eig_iw',   data=res['P0_eig_iw'])
        f.create_dataset('P0_eig_fit',  data=res['P0_eig_fit'])
        f.create_dataset('Pbse_eig_iw', data=res['Pbse_eig_iw'])
        f.create_dataset('Pbse_eig_fit',data=res['Pbse_eig_fit'])

        # Plasmon-pole parameters
        pp = f.require_group('pp')
        pp.create_dataset('P0_Finf',    data=res['P0_Finf'])
        pp.create_dataset('P0_S',       data=res['P0_S'])
        pp.create_dataset('P0_wp',      data=res['P0_wp'])
        pp.create_dataset('Pbse_Finf',  data=res['Pbse_Finf'])
        pp.create_dataset('Pbse_S',     data=res['Pbse_S'])
        pp.create_dataset('Pbse_wp',    data=res['Pbse_wp'])

        # Spectral functions
        f.create_dataset('A_P0',        data=res['A_P0'])
        f.create_dataset('A_Pbse',      data=res['A_Pbse'])

    print(f'\nq_idx={q_idx} data saved → {path}')
    print(f'  /Omega          {res["Omega"].shape}')
    if 'P0_iw_raw' in res:
        print(f'  /P0_iw          {res["P0_iw_raw"].shape}  (niw, NQ, NQ) complex')
        print(f'  /Xi_iw          {res["Xi_iw"].shape}  (niw, NQ, NQ) complex')
        print(f'  /P_bse          {res["P_bse_raw"].shape}  (niw, NQ, NQ) complex')
    print(f'  /P0_eig_iw      {res["P0_eig_iw"].shape}  (niw, NQ)')
    print(f'  /A_P0           {res["A_P0"].shape}  (npts, NQ)')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

C_P0   = '#4e88c7'
C_PBSE = '#e05c5c'


def _plot_q0_detail(axes_row, res0, omega, eta):
    """Fill the two axes for the q = 0 detail row.
    
    Uses finite_q results when q=0, summing over all eigenmodes.
    """
    ax_iw, ax_re = axes_row
    Omega_eV = res0['Omega'] * AU2EV
    omega_eV = omega * AU2EV

    # Sums over eigenmodes (= trace, basis-independent)
    P0_tr      = res0['P0_eig_iw'].sum(axis=1)
    Pbse_tr    = res0['Pbse_eig_iw'].sum(axis=1)
    P0_fit_tr  = res0['P0_eig_fit'].sum(axis=1)
    Pbse_fit_tr= res0['Pbse_eig_fit'].sum(axis=1)
    A_P0_sum   = res0['A_P0'].sum(axis=1)
    A_Pbse_sum = res0['A_Pbse'].sum(axis=1)

    p0_peak   = res0['P0_peak_eV']
    pbse_peak = res0['Pbse_peak_eV']

    # ── imaginary axis ──────────────────────────────────────────────────────
    ax_iw.plot(Omega_eV, P0_tr,       'o', ms=2.5, color=C_P0,
               label=r'$\mathrm{Tr}\,P^0$  (data)')
    ax_iw.plot(Omega_eV, P0_fit_tr,   '-', lw=1.2, color=C_P0,
               label='PP fit')
    ax_iw.plot(Omega_eV, Pbse_tr,     'x', ms=2.5, color=C_PBSE,
               label=r'$\mathrm{Tr}\,P_\mathrm{BSE}$  (data)')
    ax_iw.plot(Omega_eV, Pbse_fit_tr, '-', lw=1.2, color=C_PBSE,
               label='PP fit')

    # Count significant eigenmodes
    n_p0   = int((res0['P0_eigvals']   != 0).sum())
    n_pbse = int((res0['Pbse_eigvals'] != 0).sum())
    n_sig_p0   = int((res0['A_P0'].max(axis=0)   > 1e-8).sum())
    n_sig_pbse = int((res0['A_Pbse'].max(axis=0) > 1e-8).sum())

    xlim = min(max(p0_peak * 4, 10.0), 200.0)
    ax_iw.set_xlim(-xlim, xlim)
    ax_iw.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax_iw.set_xlabel(r'$i\Omega_n$ (eV)')
    ax_iw.set_ylabel(r'$\mathrm{Tr}\,P(i\Omega_n) = \sum_m \tilde{P}_{mm}(i\Omega_n)$')
    ax_iw.legend(fontsize=7, frameon=False, loc='lower right')
    ax_iw.set_title('q = Γ  (optical)  —  imaginary axis', fontsize=9)

    ax_iw.text(0.03, 0.06,
               fr"$\omega^{{P^0}}_\mathrm{{peak}}={p0_peak:.2f}$ eV"
               f"  ({n_sig_p0} modes)\n"
               fr"$\omega^{{P_\mathrm{{BSE}}}}_\mathrm{{peak}}={pbse_peak:.2f}$ eV"
               f"  ({n_sig_pbse} modes)",
               transform=ax_iw.transAxes, fontsize=7,
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='grey', lw=0.7))

    # ── real axis ────────────────────────────────────────────────────────────
    ax_re.plot(omega_eV, A_P0_sum,
               '-', lw=1.5, color=C_P0,   label=r'$P^0$')
    ax_re.fill_between(omega_eV, A_P0_sum,   alpha=0.18, color=C_P0)
    ax_re.plot(omega_eV, A_Pbse_sum,
               '-', lw=1.5, color=C_PBSE, label=r'$P_\mathrm{BSE}$')
    ax_re.fill_between(omega_eV, A_Pbse_sum, alpha=0.18, color=C_PBSE)

    ax_re.axvline(p0_peak,   color=C_P0,   lw=0.8, ls='--', alpha=0.7)
    ax_re.axvline(pbse_peak, color=C_PBSE, lw=0.8, ls='--', alpha=0.7)
    ax_re.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax_re.set_xlabel(r'$\omega$ (eV)')
    ax_re.set_ylabel(r'$\sum_m A_m(\omega) = -\mathrm{Im}\,\mathrm{Tr}\,P(\omega+i\eta)/\pi$')
    ax_re.legend(fontsize=8, frameon=False)
    ax_re.set_title(
        f'q = Γ — real axis  (η = {eta * AU2EV:.3f} eV)', fontsize=9)


def _plot_heatmap(ax, qx_vals, omega, A_matrix, title,
                  cmap='inferno', ymin=None, ymax=None):
    """
    Heatmap: x = q_x (fractional), y = ω (eV), colour = per-q normalised A.

    A_matrix : (nq, nomega)
    """
    omega_eV = omega * AU2EV

    dq = qx_vals[1] - qx_vals[0] if len(qx_vals) > 1 else 0.125
    q_edges = np.append(qx_vals - dq / 2, qx_vals[-1] + dq / 2)
    dom     = omega_eV[1] - omega_eV[0]
    w_edges = np.append(omega_eV - dom / 2, omega_eV[-1] + dom / 2)

    A_plot  = np.maximum(A_matrix, 0.0)
    row_max = A_plot.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    A_norm  = A_plot / row_max

    col = ax.pcolormesh(q_edges, w_edges, A_norm.T,
                        cmap=cmap, shading='flat', vmin=0, vmax=1)
    plt.colorbar(col, ax=ax, label='A / max  (per-q normalised)')
    ax.set_xlabel(r'$q_x$ (fractional)')
    ax.set_ylabel(r'$\omega$ (eV)')
    ax.set_title(title, fontsize=9)
    ax.set_xlim(q_edges[0], q_edges[-1])
    if ymin is not None:
        ax.set_ylim(ymin, ymax)


def make_figure(res0, fq_results, fq_qx, omega, eta, output,
                ymin=None, ymax=None):
    nomega = len(omega)
    nq     = len(fq_qx)

    A_P0_mat   = np.zeros((nq, nomega))
    A_Pbse_mat = np.zeros((nq, nomega))
    for i, res in enumerate(fq_results):
        A_P0_mat[i]   = res['A_P0'].sum(axis=1)
        A_Pbse_mat[i] = res['A_Pbse'].sum(axis=1)

    fig = plt.figure(figsize=(9, 7))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)

    ax_iw  = fig.add_subplot(gs[0, 0])
    ax_re  = fig.add_subplot(gs[0, 1])
    ax_hm0 = fig.add_subplot(gs[1, 0])
    ax_hmB = fig.add_subplot(gs[1, 1])

    _plot_q0_detail([ax_iw, ax_re], res0, omega, eta)
    _plot_heatmap(ax_hm0, fq_qx, omega, A_P0_mat,
                  title=r'$P^0$  —  $\sum_m A^0_m(q,\omega)$',
                  cmap='Blues', ymin=ymin, ymax=ymax)
    _plot_heatmap(ax_hmB, fq_qx, omega, A_Pbse_mat,
                  title=r'$P_\mathrm{BSE}$  —  $\sum_m A^\mathrm{BSE}_m(q,\omega)$',
                  cmap='Reds', ymin=ymin, ymax=ymax)

    fig.suptitle('P⁰ vs P_BSE: eigenmode AC + q-dispersion', fontsize=10, y=1.01)
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f'\nFigure saved → {output}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='P⁰ vs P_BSE: eigenmode AC, q=0 detail + dispersion heatmap.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--q0_dir', default=None,
        help='Directory for the q=0 calculation (mean_field_input.h5 etc.).')
    parser.add_argument('--fq_dir', default=None,
        help='Directory containing df_hf_int_fq/ for finite-q VQ files.')
    parser.add_argument('--beta',      type=float, default=1000.0)
    parser.add_argument('--eta',       type=float, default=0.005,
        help='Lorentzian broadening (Ha).')
    parser.add_argument('--omega_max', type=float, default=1.5,
        help='Real-axis upper limit (Ha).')
    parser.add_argument('--omega_min', type=float, default=0.0,
        help='Real-axis lower limit (Ha).')
    parser.add_argument('--npts',      type=int,   default=2000)
    parser.add_argument('--kappa',     type=float, default=1.0,
        help='Exchange mixing: 1.0 = singlet BSE, 0.0 = direct-only (triplet).')
    parser.add_argument('--full_exchange', action='store_true',
        help='Use full exchange kernel (for finite-q / non-optical).')
    parser.add_argument('--ymin',  type=float, default=None,
        help='Lower ω bound for heatmaps (eV). Default: auto.')
    parser.add_argument('--ymax',  type=float, default=None,
        help='Upper ω bound for heatmaps (eV). Default: auto.')
    parser.add_argument('--output', default='dispersion.pdf')
    parser.add_argument('--save_q0', default=None, metavar='FILE.h5',
        help='HDF5 file to save q=0 intermediate quantities '
             '(P0_iw, Xi_iw, P_bse, eigenmodes, AC results). '
             'Default: replace the output extension with _q0.h5. '
             'Finite-q files are saved alongside as _q{idx}.h5.')
    args = parser.parse_args()

    q0_dir = args.q0_dir or os.path.join(_ROOT_DIR, 'example', 'H2_correct_fq_8x1x1k')
    fq_dir = args.fq_dir or q0_dir

    for label, path in [('q0_dir', q0_dir), ('fq_dir', fq_dir)]:
        if not os.path.isdir(path):
            print(f'ERROR: {label} not found: {path}')
            sys.exit(1)

    # Derive save path for q=0 data
    save_q0_path = args.save_q0
    if save_q0_path is None:
        base, _ = os.path.splitext(args.output)
        save_q0_path = base + '_q0.h5'

    fq_int     = os.path.join(fq_dir, 'df_hf_int_fq')
    fq_indices = _available_q_indices(fq_int)
    if not fq_indices:
        print(f'ERROR: no VQ_q*.h5 found in {fq_int}')
        sys.exit(1)
    print(f'Finite-q indices: {fq_indices}')

    fq_qx_all = _kmesh_qx(os.path.join(fq_dir, 'mean_field_input.h5'))
    fq_qx     = np.array([fq_qx_all[i] for i in fq_indices])
    omega     = np.linspace(args.omega_min, args.omega_max, args.npts)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── q = 0 ──────────────────────────────────────────────────────────
        print(f'\n{"="*60}\nComputing q=0  from {q0_dir}\n{"="*60}')
        res0 = run_q(0, fq_dir, args.beta, omega, args.eta,
                     args.kappa, tmpdir,
                     keep_raw=True)           # keep P0_iw, Xi_iw, P_bse
        ev0  = res0['P0_eigvals']
        nsig = int((res0['A_P0'].max(axis=0) > 1e-8).sum())
        print(f'  P0 eigenvalues (top 4): {np.round(ev0[:4], 4)}'
              f'  significant modes: {nsig}')
        print(f'  P⁰   peak = {res0["P0_peak_eV"]:.3f} eV')
        print(f'  P_BSE peak = {res0["Pbse_peak_eV"]:.3f} eV')

        # Save q=0 intermediate quantities
        save_q_data(res0, 0, omega, save_q0_path)

        # ── finite q ───────────────────────────────────────────────────────
        fq_results = []
        for q_idx in fq_indices:
            print(f'\n{"="*60}\nComputing q={q_idx}  (qx={fq_qx_all[q_idx]:.4f})'
                  f'  from {fq_dir}\n{"="*60}')
            res = run_q(q_idx, fq_dir, args.beta, omega, args.eta,
                        args.kappa, tmpdir, keep_raw=True)
            fq_results.append(res)
            fq_path = _fq_save_path(save_q0_path, q_idx)
            save_q_data(res, q_idx, omega, fq_path)
            nsig_p0   = int((res['A_P0'].max(axis=0)   > 1e-8).sum())
            nsig_pbse = int((res['A_Pbse'].max(axis=0) > 1e-8).sum())
            print(f'  P⁰   peak = {res["P0_peak_eV"]:.3f} eV'
                  f'  ({nsig_p0} modes)')
            print(f'  P_BSE peak = {res["Pbse_peak_eV"]:.3f} eV'
                  f'  ({nsig_pbse} modes)')

    make_figure(res0, fq_results, fq_qx, omega, args.eta, args.output,
                ymin=args.ymin, ymax=args.ymax)


if __name__ == '__main__':
    main()
