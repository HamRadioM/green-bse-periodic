#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.   #
#                                                                             #
#    Five-panel figure for a hexagonal 2D material along K-Γ-M:              #
#                                                                             #
#      Panel 1 (top-left)   – Electronic band structure                      #
#      Panel 2 (top-center) – P⁰   spectral heatmap  A(q, ω)                #
#      Panel 3 (top-right)  – P_BSE spectral heatmap  A(q, ω)               #
#      Panel 4 (bot-left)   – Tr P⁰(iΩ) vs Tr P_BSE(iΩ) at q=0             #
#      Panel 5 (bot-right)  – Σ_Q A_Q(ω) for P⁰ and P_BSE at q=0           #
#                                                                             #
#    High-symmetry points (fractional, hexagonal convention):                 #
#      K = (2/3, 1/3, 0)   corner of hexagonal BZ                           #
#      Γ = (0,   0,   0)   zone centre                                       #
#      M = (1/2, 1/2, 0)   edge midpoint                                     #
#                                                                             #
#    Usage                                                                    #
#    -----                                                                    #
#    python script/plot_hBN_kgm.py \                                         #
#        --mf   example/hBN_sto-3g_4x4x1k/mean_field_input.h5 \             #
#        --bse  bse_results.h5 \                                              #
#        [--sim example/hBN_sto-3g_4x4x1k/sim.h5]  \                        #
#        [--omega_min 0] [--omega_max 60]           \   # eV                 #
#        [--emin -10]    [--emax  30]               \   # eV, band window    #
#        [--tol 0.05]                               \   # fractional tol.    #
#        [--output hBN_kgm.pdf]                                               #

import argparse
import os
import sys

import h5py
import numpy as np
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

AU2EV = 27.211386245981

# High-symmetry points in fractional (hexagonal PySCF convention)
# k_cart = k_frac @ B  where rows of B are reciprocal lattice vectors
_K_FRAC = np.array([2/3, 1/3, 0.0])
_G_FRAC = np.array([0.0, 0.0, 0.0])
_M_FRAC = np.array([0.5, 0.5, 0.0])

_PATH_FRAC   = [_K_FRAC, _G_FRAC, _M_FRAC]
_PATH_LABELS = ['K', 'Γ', 'M']


# ---------------------------------------------------------------------------
# Reciprocal lattice inference
# ---------------------------------------------------------------------------

def recip_matrix(mf_h5: str) -> np.ndarray:
    """
    Infer the 3×3 reciprocal lattice matrix B from stored k-points, where
    k_cart = k_frac @ B  (rows of B are b1, b2, b3).

    Only the 2×2 in-plane block is fitted (z is vacuum and trivially zero
    for a 2D material); the z-z element is left as 1.
    """
    with h5py.File(mf_h5, 'r') as f:
        kf = f['grid/k_mesh_scaled'][()]   # (nk, 3)
        kc = f['grid/k_mesh'][()]           # (nk, 3)

    B2, _, _, _ = np.linalg.lstsq(kf[:, :2], kc[:, :2], rcond=None)
    B = np.zeros((3, 3))
    B[:2, :2] = B2
    B[2, 2]   = 1.0   # placeholder; z components are always 0 for 2D
    return B


# ---------------------------------------------------------------------------
# Path projection
# ---------------------------------------------------------------------------

def build_path_cart(path_frac, B):
    """Convert list of fractional path vertices to Cartesian."""
    return [np.array(p) @ B for p in path_frac]


def path_arc_lengths(path_cart):
    """Cumulative arc lengths at each vertex of a piecewise path."""
    arcs = [0.0]
    for A, C in zip(path_cart[:-1], path_cart[1:]):
        arcs.append(arcs[-1] + np.linalg.norm(C - A))
    return np.array(arcs)


def project_on_path(k_frac, B, path_frac, tol_frac=0.05):
    """
    Project k-points onto a piecewise linear path.

    Parameters
    ----------
    k_frac     : (nk, 3) fractional coordinates
    B          : (3, 3)  reciprocal lattice matrix
    path_frac  : list of fractional vertex arrays
    tol_frac   : tolerance in fractional distance for membership

    Returns
    -------
    s    : (nk,)  arc-length along path (Bohr⁻¹); np.nan if not on path
    mask : (nk,)  bool — True if within tol of the path
    """
    path_cart = build_path_cart(path_frac, B)
    arc       = path_arc_lengths(path_cart)
    k_cart    = k_frac @ B

    # Tolerance: convert fractional to Cartesian using max lattice vector norm
    b_norms   = np.linalg.norm(B[:2, :2], axis=1)
    tol_cart  = tol_frac * float(np.max(b_norms))

    nk   = k_frac.shape[0]
    s    = np.full(nk, np.nan)
    dist = np.full(nk, np.inf)

    for ik in range(nk):
        p = k_cart[ik]
        for iseg, (A, C) in enumerate(zip(path_cart[:-1], path_cart[1:])):
            AC  = C - A
            seg = np.linalg.norm(AC)
            if seg < 1e-12:
                continue
            t = np.dot(p - A, AC) / seg**2
            t = float(np.clip(t, 0.0, 1.0))
            proj = A + t * AC
            d    = np.linalg.norm(p - proj)
            if d < dist[ik]:
                dist[ik] = d
                s[ik]    = arc[iseg] + t * seg

    mask = dist <= tol_cart
    return s, mask


# ---------------------------------------------------------------------------
# Band structure
# ---------------------------------------------------------------------------

def read_bands(mf_h5: str, sim_h5: str = None):
    """
    Diagonalise F(k) at every k-point.

    Returns
    -------
    k_frac  : (nk, 3)
    eps_all : (nk, nao)   in Hartree
    nel     : int
    mu      : float       chemical potential (Ha); midgap if sim not given
    """
    with h5py.File(mf_h5, 'r') as f:
        Fk_raw = f['HF/Fock-k'][()]   # (ns, nk, nao, nao, 2)
        Sk_raw = f['HF/S-k'][()]
        k_frac = f['grid/k_mesh_scaled'][()]
        nel    = int(f['params/nel_cell'][()])
        nao    = int(f['params/nao'][()])

    ns, nk = Fk_raw.shape[:2]
    Fk = Fk_raw[0, :, :, :, 0] + 1j * Fk_raw[0, :, :, :, 1]
    Sk = Sk_raw[0, :, :, :, 0] + 1j * Sk_raw[0, :, :, :, 1]

    eps_all = np.empty((nk, nao))
    for ik in range(nk):
        eps_all[ik], _ = eigh(Fk[ik], Sk[ik])

    nfilled = nel // 2
    if sim_h5 and os.path.isfile(sim_h5):
        with h5py.File(sim_h5, 'r') as f:
            it = int(f['iter'][()])
            mu = float(f[f'iter{it}/mu'][()])
    else:
        homo = eps_all[:, nfilled - 1].max()
        lumo = eps_all[:, nfilled].min()
        mu   = 0.5 * (homo + lumo)

    return k_frac, eps_all, nel, mu


# ---------------------------------------------------------------------------
# BSE results reader
# ---------------------------------------------------------------------------

def read_bse(bse_h5: str):
    """
    Load all per-q results from a bse_results.h5 written by solve_bse.py.

    Returns
    -------
    omega      : (npts,)  real-axis grid (Ha)
    q_indices  : list[int]
    results    : dict  q_idx → sub-dict with keys
                   q_frac          (3,)
                   Omega           (niw,)
                   P0_iw_diag      (niw, NQ)
                   Pbse_iw_diag    (niw, NQ)
                   A_P0            (npts, NQ)
                   A_Pbse          (npts, NQ)
    """
    with h5py.File(bse_h5, 'r') as f:
        omega     = f['omega'][()]
        q_indices = list(f['q_indices'][()])
        results   = {}
        for qi in q_indices:
            g = f[f'q{qi}']
            results[qi] = dict(
                q_frac        = g['q_frac'][()],
                Omega         = g['Omega'][()],
                P0_iw_diag    = g['P0_iw_diag'][()],
                Pbse_iw_diag  = g['Pbse_iw_diag'][()],
                A_P0          = g['A_P0'][()],
                A_Pbse        = g['A_Pbse'][()],
            )
    return omega, q_indices, results


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

C_P0   = '#4e88c7'
C_PBSE = '#e05c5c'
C_VAL  = '#4e88c7'
C_CON  = '#e05c5c'


def _xticks(path_cart, arc, labels):
    """Return (tick_positions, tick_labels) for the path x-axis."""
    return arc, labels


def _vlines_on_ax(ax, arc):
    """Draw vertical dashed lines at each path vertex."""
    for s in arc:
        ax.axvline(s, color='k', lw=0.6, ls='--', alpha=0.5)


def plot_bands(ax, k_frac, eps_all, nel, mu, B,
               path_frac, path_labels,
               emin_eV, emax_eV, tol_frac):
    """Panel 1: electronic band structure along path."""
    nfilled = nel // 2
    nao     = eps_all.shape[1]

    s, mask = project_on_path(k_frac, B, path_frac, tol_frac)

    if mask.sum() == 0:
        ax.text(0.5, 0.5, 'No k-points found\non path\n(decrease --tol?)',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        return

    path_cart = build_path_cart(path_frac, B)
    arc       = path_arc_lengths(path_cart)

    s_sel   = s[mask]
    eps_sel = (eps_all[mask] - mu) * AU2EV    # (n_sel, nao) shifted to eV

    order   = np.argsort(s_sel)
    s_sel   = s_sel[order]
    eps_sel = eps_sel[order]

    for ib in range(nao):
        color = C_VAL if ib < nfilled else C_CON
        ax.plot(s_sel, eps_sel[:, ib], 'o-', color=color,
                ms=4, lw=1.2, alpha=0.85)

    ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.6)
    _vlines_on_ax(ax, arc)

    ax.set_xlim(arc[0], arc[-1])
    ax.set_ylim(emin_eV, emax_eV)
    ax.set_xticks(arc)
    ax.set_xticklabels(path_labels)
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Band structure', fontsize=9)

    handles = [
        Line2D([0], [0], color=C_VAL, lw=1.5, label='Valence'),
        Line2D([0], [0], color=C_CON, lw=1.5, label='Conduction'),
    ]
    ax.legend(handles=handles, fontsize=7, frameon=False)


def plot_spectral_heatmap(ax, q_frac_list, A_list, omega_eV,
                          B, path_frac, path_labels,
                          omega_min_eV, omega_max_eV, tol_frac,
                          title, cmap='inferno'):
    """
    Panels 2 & 3: heatmap of Σ_Q A_Q(q, ω) vs (arc-length, ω).

    q_frac_list : list of (3,) fractional q-vectors
    A_list      : list of (npts, NQ) arrays
    """
    path_cart = build_path_cart(path_frac, B)
    arc       = path_arc_lengths(path_cart)

    # Project each q-point onto the path
    q_frac_arr = np.array(q_frac_list)             # (nq, 3)
    s, mask    = project_on_path(q_frac_arr, B, path_frac, tol_frac)

    on_path = np.where(mask)[0]
    if len(on_path) == 0:
        ax.text(0.5, 0.5, 'No q-points on path', ha='center', va='center',
                transform=ax.transAxes, fontsize=8)
        return

    # Sort by arc-length
    order   = np.argsort(s[on_path])
    on_path = on_path[order]
    s_sel   = s[on_path]                            # (nq_sel,)

    # Sum over Q to get total spectral weight per q
    A_sel = np.array([A_list[i].sum(axis=1) for i in on_path])  # (nq_sel, npts)

    # Trim to requested ω window
    omask  = (omega_eV >= omega_min_eV) & (omega_eV <= omega_max_eV)
    om_sel = omega_eV[omask]
    A_sel  = A_sel[:, omask]

    # Per-q row normalisation so weak dispersive features are visible
    row_max = A_sel.max(axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    A_norm  = np.clip(A_sel / row_max, 0, None)

    # Build pixel edges for pcolormesh
    if len(s_sel) > 1:
        ds  = np.diff(s_sel)
        sq  = np.concatenate([[s_sel[0] - ds[0]/2],
                               s_sel[:-1] + ds/2,
                               [s_sel[-1] + ds[-1]/2]])
    else:
        sq = np.array([s_sel[0] - 0.02, s_sel[0] + 0.02])

    dom  = (om_sel[-1] - om_sel[0]) / (len(om_sel) - 1)
    we   = np.concatenate([[om_sel[0] - dom/2],
                            om_sel[:-1] + dom/2,
                            [om_sel[-1] + dom/2]])

    col = ax.pcolormesh(sq, we, A_norm.T, cmap=cmap, shading='flat',
                        vmin=0, vmax=1)
    plt.colorbar(col, ax=ax, label='A / max  (per-q norm.)', pad=0.02)

    _vlines_on_ax(ax, arc)
    ax.set_xlim(arc[0], arc[-1])
    ax.set_ylim(omega_min_eV, omega_max_eV)
    ax.set_xticks(arc)
    ax.set_xticklabels(path_labels)
    ax.set_ylabel(r'$\omega$ (eV)')
    ax.set_title(title, fontsize=9)


def plot_q0_iw(ax, res0, beta=None):
    """Panel 4: imaginary-axis traces Tr P(iΩ) at q=0."""
    Omega_eV = res0['Omega'] * AU2EV
    P0_tr    = res0['P0_iw_diag'].sum(axis=1)
    Pbse_tr  = res0['Pbse_iw_diag'].sum(axis=1)

    ax.plot(Omega_eV, P0_tr,   'o',  ms=2.5, color=C_P0,   label=r'$\mathrm{Tr}\,P^0$')
    ax.plot(Omega_eV, Pbse_tr, 's',  ms=2.5, color=C_PBSE, label=r'$\mathrm{Tr}\,P_\mathrm{BSE}$')

    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel(r'$i\Omega_n$ (eV)')
    ax.set_ylabel(r'$\mathrm{Tr}\,P(i\Omega_n)$')
    ax.set_title(r'Imaginary axis  ($q=\Gamma$)', fontsize=9)

    # Zoom to the physically interesting low-frequency range
    peak_idx = np.argmax(np.abs(P0_tr - P0_tr[-1]))
    xlim = max(Omega_eV[peak_idx] * 4, 20.0)
    ax.set_xlim(-xlim, xlim)

    ax.legend(fontsize=7, frameon=False, loc='lower right')


def plot_q0_spectral(ax, res0, omega_eV, omega_min_eV, omega_max_eV):
    """Panel 5: real-axis spectral function Σ_Q A_Q(ω) at q=0."""
    A_P0_sum   = res0['A_P0'].sum(axis=1)
    A_Pbse_sum = res0['A_Pbse'].sum(axis=1)

    omask    = (omega_eV >= omega_min_eV) & (omega_eV <= omega_max_eV)
    om_sel   = omega_eV[omask]

    ax.plot(om_sel, A_P0_sum[omask],
            '-', lw=1.5, color=C_P0,   label=r'$P^0$')
    ax.fill_between(om_sel, A_P0_sum[omask], alpha=0.2, color=C_P0)

    ax.plot(om_sel, A_Pbse_sum[omask],
            '-', lw=1.5, color=C_PBSE, label=r'$P_\mathrm{BSE}$')
    ax.fill_between(om_sel, A_Pbse_sum[omask], alpha=0.2, color=C_PBSE)

    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.set_xlabel(r'$\omega$ (eV)')
    ax.set_ylabel(r'$\sum_Q A_Q(\omega)$  (Ha$^{-1}$)')
    ax.set_title(r'Real-axis spectral function  ($q=\Gamma$)', fontsize=9)
    ax.set_xlim(omega_min_eV, omega_max_eV)
    ax.legend(fontsize=8, frameon=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='K-Γ-M band structure + BSE spectral functions for hBN.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mf',   required=True, metavar='FILE',
                        help='mean_field_input.h5')
    parser.add_argument('--bse',  required=True, metavar='FILE',
                        help='bse_results.h5  (output of solve_bse.py)')
    parser.add_argument('--sim',  default=None,  metavar='FILE',
                        help='sim.h5  (optional; used to read μ for bands).')
    parser.add_argument('--output', default='hBN_kgm.pdf')

    # Energy windows
    parser.add_argument('--omega_min', type=float, default=0.0,
                        help='Min ω for spectral heatmaps (eV).')
    parser.add_argument('--omega_max', type=float, default=60.0,
                        help='Max ω for spectral heatmaps (eV).')
    parser.add_argument('--emin', type=float, default=-10.0,
                        help='Min energy for band structure (eV).')
    parser.add_argument('--emax', type=float, default=30.0,
                        help='Max energy for band structure (eV).')

    # Path tolerance
    parser.add_argument('--tol', type=float, default=0.05,
                        help='Fractional-coordinate tolerance for K-Γ-M membership.')

    # High-symmetry point overrides (fractional, space-separated floats)
    parser.add_argument('--K', type=float, nargs=3,
                        default=[2/3, 1/3, 0.0],
                        metavar=('f1', 'f2', 'f3'),
                        help='K point in fractional coords.')
    parser.add_argument('--M', type=float, nargs=3,
                        default=[0.5, 0.5, 0.0],
                        metavar=('f1', 'f2', 'f3'),
                        help='M point in fractional coords.')

    args = parser.parse_args()

    for f in (args.mf, args.bse):
        if not os.path.isfile(f):
            sys.exit(f'ERROR: file not found: {f}')

    path_frac   = [np.array(args.K), _G_FRAC, np.array(args.M)]
    path_labels = ['K', 'Γ', 'M']

    # --- Read data ---
    B                       = recip_matrix(args.mf)
    k_frac, eps_all, nel, mu = read_bands(args.mf, args.sim)
    omega, q_indices, bse   = read_bse(args.bse)
    omega_eV                = omega * AU2EV

    # Gather per-q arrays for heatmap helpers
    q_frac_list = [bse[qi]['q_frac'] for qi in q_indices]
    A_P0_list   = [bse[qi]['A_P0']   for qi in q_indices]
    A_Pbse_list = [bse[qi]['A_Pbse'] for qi in q_indices]

    # q=0 results for panels 4 & 5
    if 0 not in bse:
        sys.exit('ERROR: q=0 not found in bse_results.h5. '
                 'Re-run solve_bse.py with q_idx=0.')
    res0 = bse[0]

    # High-symmetry arc lengths for x-tick positioning
    path_cart = build_path_cart(path_frac, B)
    arc       = path_arc_lengths(path_cart)

    # Print path geometry
    print(f'K-Γ-M path (Cartesian, Bohr⁻¹):')
    for label, pt, s in zip(path_labels, path_cart, arc):
        print(f'  {label} = ({pt[0]:+.4f}, {pt[1]:+.4f})   s = {s:.4f}')
    print(f'Total path length: {arc[-1]:.4f} Bohr⁻¹')

    s_k, mask_k = project_on_path(k_frac, B, path_frac, args.tol)
    print(f'\nBand structure: {mask_k.sum()}/{len(k_frac)} k-points on path (tol={args.tol})')

    q_frac_arr = np.array(q_frac_list)
    s_q, mask_q = project_on_path(q_frac_arr, B, path_frac, args.tol)
    on_q = np.where(mask_q)[0]
    print(f'BSE heatmap: {mask_q.sum()}/{len(q_indices)} q-points on path')
    for i in on_q:
        qi = q_indices[i]
        qf = q_frac_list[i]
        print(f'  q_idx={qi}  frac=({qf[0]:.4f},{qf[1]:.4f})  s={s_q[i]:.4f}')

    # --- Figure ---
    fig = plt.figure(figsize=(14, 9))
    outer_gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                                  height_ratios=[1.2, 1.0])
    top_gs   = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0],
                                                wspace=0.38)
    bot_gs   = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1],
                                                wspace=0.35)

    ax_bands = fig.add_subplot(top_gs[0])
    ax_P0hm  = fig.add_subplot(top_gs[1])
    ax_Pbhm  = fig.add_subplot(top_gs[2])
    ax_iw    = fig.add_subplot(bot_gs[0])
    ax_re    = fig.add_subplot(bot_gs[1])

    # Panel 1 – bands
    plot_bands(ax_bands, k_frac, eps_all, nel, mu, B,
               path_frac, path_labels,
               emin_eV=args.emin, emax_eV=args.emax,
               tol_frac=args.tol)

    # Panel 2 – P⁰ heatmap
    plot_spectral_heatmap(
        ax_P0hm, q_frac_list, A_P0_list, omega_eV,
        B, path_frac, path_labels,
        omega_min_eV=args.omega_min, omega_max_eV=args.omega_max,
        tol_frac=args.tol,
        title=r'$P^0$  spectral function  $\sum_Q A_Q(q,\omega)$',
        cmap='Blues',
    )

    # Panel 3 – P_BSE heatmap
    plot_spectral_heatmap(
        ax_Pbhm, q_frac_list, A_Pbse_list, omega_eV,
        B, path_frac, path_labels,
        omega_min_eV=args.omega_min, omega_max_eV=args.omega_max,
        tol_frac=args.tol,
        title=r'$P_\mathrm{BSE}$  spectral function  $\sum_Q A_Q(q,\omega)$',
        cmap='Reds',
    )

    # Panel 4 – imaginary-axis traces at q=0
    plot_q0_iw(ax_iw, res0)

    # Panel 5 – real-axis spectral function at q=0
    plot_q0_spectral(ax_re, res0, omega_eV,
                     omega_min_eV=args.omega_min,
                     omega_max_eV=args.omega_max)

    fig.suptitle('hBN along K-Γ-M: bands + BSE spectral functions', fontsize=11)
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f'\nFigure saved → {args.output}')


if __name__ == '__main__':
    main()
