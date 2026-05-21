#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.   #
#                                                                             #
#    Step-by-step construction and visualisation of the BSE kernel            #
#    Ξ_{QQ'}(q, iΩ) = 2U_{QQ'} − W_{QQ'}(q, iΩ)                            #
#    using the H2_correct_fq_8k example.                                      #
#                                                                             #
#    Run from the project root:                                               #
#        python script/inspect_xi_kernel.py                                  #
#        python script/inspect_xi_kernel.py --q_idx 2 --save figs/           #
#                                                                             #
#    Figures produced:                                                         #
#        1. Pipeline overview — matrix heatmaps at Ω=0                       #
#        2. Frequency dependence — Frobenius norms vs iΩ                     #
#        3. W transformation — MO vs AO index-swap comparison                #
#        4. Kernel construction — 2U, W, Ξ; eigenvalues; Dyson check         #
#                                                                             #

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import eigh, inv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, os.path.join(ROOT_DIR, 'green_bse'))

from polarization import (
    _read_VQ, IR_factory, compute_P0_tau, symmetrize_P0,
    dyson_equation, transform_W_via_mo, transform_W_via_ao, compute_U_qq,
)
from bse_kernel_qq import bse_dyson_iw, _squeeze_iw

AU2EV = 27.211386245981


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def frob(arr_iw: np.ndarray) -> np.ndarray:
    """Frobenius norm at each frequency: arr_iw (niw, NQ, NQ) → (niw,)."""
    return np.array([np.linalg.norm(arr_iw[iw]) for iw in range(arr_iw.shape[0])])


def diag_trace(arr_iw: np.ndarray) -> np.ndarray:
    """Real part of the matrix trace at each frequency: (niw, NQ, NQ) → (niw,)."""
    return arr_iw.diagonal(axis1=-2, axis2=-1).real.sum(axis=-1)


def heatmap(ax, mat, title, cmap='RdBu_r', label='Re'):
    """Plot real part of a (NQ, NQ) matrix on ax with a centred colormap."""
    data  = mat.real
    vmax  = np.max(np.abs(data))
    vmax  = max(vmax, 1e-12)
    norm  = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im    = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Q\'', fontsize=7); ax.set_ylabel('Q', fontsize=7)
    ax.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    return im


def static(arr_iw: np.ndarray) -> np.ndarray:
    """Extract the Ω=0 slice (niw, NQ, NQ) → (NQ, NQ)."""
    return arr_iw[arr_iw.shape[0] // 2]


# ─────────────────────────────────────────────────────────────────────────────
# Step-by-step computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(example_dir: str, q_idx: int, beta: float, ir_file: str) -> dict:
    """
    Load raw inputs and compute every intermediate in the Ξ pipeline.

    Steps
    -----
    1.  Load G(τ), VQ_{k,k+q}^{AO}, C_k (from Fock diag.), kq_map
    2.  P⁰(q, τ)  = bare bubble
    3.  P⁰(q, iΩ) = IR Fourier transform τ → iΩ
    4.  P̃(q, iΩ)  = (I − P⁰)⁻¹ P⁰   (Dyson, RPA screening)
    5.  W_qq(q, iΩ) = I + P̃           (direct screened Coulomb in QQ' basis)
    6a. W_mo(q, iΩ) = index swap via MO-pair basis
    6b. W_ao(q, iΩ) = index swap directly in AO-pair basis
    7.  U_qq        = I_{NQ}            (exchange bare Coulomb, unity in QQ')
    8.  Ξ(q, iΩ)   = 2U − W_mo        (BSE kernel)
    9.  P_BSE(q, iΩ) = (I − P⁰ Ξ)⁻¹ P⁰  (BSE Dyson)
    """
    int_path = os.path.join(example_dir, 'df_hf_int_fq') + os.sep

    # ── Load MO coefficients (Fock diagonalisation) ──────────────────────────
    print("[1/9] Loading mean-field input, G(τ), VQ …")
    with h5py.File(os.path.join(example_dir, 'mean_field_input.h5'), 'r') as f:
        Fk_raw = f['HF/Fock-k'][()]     # (ns, nk, nao, nao, 2) float64
        nao    = int(f['params/nao'][()])
        nk     = int(f['params/nk'][()])
    # Re + i·Im, spin channel 0
    Fk  = Fk_raw[0, :, :, :, 0] + 1j * Fk_raw[0, :, :, :, 1]  # (nk, nao, nao)
    C_k = np.zeros((nk, nao, nao), dtype=np.complex128)
    for k in range(nk):
        Fk_sym = 0.5 * (Fk[k] + Fk[k].conj().T)
        _, C_k[k] = eigh(Fk_sym)

    # ── Load G(τ) ─────────────────────────────────────────────────────────────
    with h5py.File(os.path.join(example_dir, 'sim.h5'), 'r') as f:
        it    = int(f['iter'][()])
        G_tau = f[f'iter{it}/G_tau/data'][()].view(complex)  # (ntau, ns, nk, nao, nao)
    print(f"     G_tau {G_tau.shape}  (iter={it})")

    # ── Load VQ and set up kq_map ─────────────────────────────────────────────
    vq_file  = int_path + ('VQ_0.h5' if q_idx == 0 else f'VQ_q{q_idx}.h5')
    VQ_kq_ao = _read_VQ(vq_file)                   # (nk, NQ, nao, nao)
    NQ       = VQ_kq_ao.shape[1]
    kq_map   = (np.arange(nk) + q_idx) % nk
    print(f"     VQ {VQ_kq_ao.shape}  NQ={NQ}  kq_map={kq_map.tolist()}")

    # ── IR grid & bosonic Matsubara frequencies ───────────────────────────────
    ir = IR_factory(beta, ir_file)
    with h5py.File(ir_file, 'r') as f:
        ngrid = f['bose/ngrid'][()]
    Omega = 2.0 * np.pi * ngrid / beta      # (niw,) bosonic Matsubara freqs (Ha)

    # ── Step 2: P⁰(q, τ) ─────────────────────────────────────────────────────
    print("[2/9] P⁰(q, τ) = bare bubble …")
    P0_tau = symmetrize_P0(compute_P0_tau(G_tau, VQ_kq_ao, kq_map))
    print(f"     P0_tau {P0_tau.shape}")

    # ── Step 3: P⁰(q, iΩ) ────────────────────────────────────────────────────
    print("[3/9] FT τ → iΩ …")
    P0_iw5 = ir.tauf_to_wb(P0_tau)          # (niw, ns, 1, NQ, NQ)
    P0_iw  = _squeeze_iw(P0_iw5)            # (niw, NQ, NQ)
    niw    = P0_iw.shape[0]
    print(f"     P0_iw {P0_iw.shape}")

    # ── Step 4: P̃(q, iΩ) ─────────────────────────────────────────────────────
    print("[4/9] Dyson: P̃ = (I − P⁰)⁻¹ P⁰ …")
    Pt_iw5 = dyson_equation(P0_iw5)         # (niw, ns, 1, NQ, NQ)
    Pt_iw  = _squeeze_iw(Pt_iw5)            # (niw, NQ, NQ)

    # ── Step 5: W_qq = I + P̃  (direct screened Coulomb) ─────────────────────
    print("[5/9] W_qq = I + P̃ …")
    I_NQ    = np.eye(NQ, dtype=np.complex128)
    Wqq_iw5 = Pt_iw5.copy()
    for iw in range(niw):
        Wqq_iw5[iw, 0, 0] += I_NQ
    Wqq_iw = _squeeze_iw(Wqq_iw5)           # (niw, NQ, NQ)

    # ── Step 6a: W_mo = index swap via MO-pair basis ──────────────────────────
    print("[6a/9] W_mo: QQ' → MO-pair basis → swap (m,n,m',n')→(m,m',n',n) → QQ' …")
    W_mo = _squeeze_iw(transform_W_via_mo(Wqq_iw5, VQ_kq_ao, C_k, kq_map))

    # ── Step 6b: W_ao = index swap directly in AO-pair basis ─────────────────
    print("[6b/9] W_ao: QQ' → AO-pair basis → swap (a,b,a',b')→(a,a',b',b) → QQ' …")
    W_ao = _squeeze_iw(transform_W_via_ao(Wqq_iw5, VQ_kq_ao))

    diff_max = np.max(np.abs(W_mo - W_ao))
    print(f"     max|W_mo − W_ao| = {diff_max:.3e}")

    # ── Step 7: U_qq = I  (exchange bare Coulomb) ─────────────────────────────
    print("[7/9] U_qq = I_{NQ} …")
    U_qq = compute_U_qq(NQ)                  # (NQ, NQ)

    # ── Step 8: Ξ(q, iΩ) = 2U − W_mo ────────────────────────────────────────
    print("[8/9] Ξ = 2U − W_mo …")
    Xi_iw = 2.0 * U_qq[np.newaxis] - W_mo   # (niw, NQ, NQ)

    # ── Step 9: P_BSE from BSE Dyson equation ─────────────────────────────────
    print("[9/9] BSE Dyson: P_BSE = (I − P⁰ Ξ)⁻¹ P⁰ …")
    P_bse = bse_dyson_iw(P0_iw, Xi_iw)      # (niw, NQ, NQ)

    # ── Dyson residual ─────────────────────────────────────────────────────────
    dyson_res = np.array([
        np.linalg.norm((I_NQ - P0_iw[iw] @ Xi_iw[iw]) @ P_bse[iw] - P0_iw[iw])
        for iw in range(niw)
    ])
    print(f"     Dyson residual max = {dyson_res.max():.2e}  (should be ~machine ε)")

    return dict(
        NQ=NQ, nk=nk, nao=nao, q_idx=q_idx,
        Omega=Omega,
        P0_tau=P0_tau,
        P0_iw=P0_iw,
        Pt_iw=Pt_iw,
        Wqq_iw=Wqq_iw,
        W_mo=W_mo,
        W_ao=W_ao,
        U_qq=U_qq,
        Xi_iw=Xi_iw,
        P_bse=P_bse,
        dyson_res=dyson_res,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Pipeline overview: matrix heatmaps at Ω=0
# ─────────────────────────────────────────────────────────────────────────────

def fig_heatmaps(d: dict, save_dir: str):
    """2×4 grid of Re(M) heatmaps at Ω=0 for every intermediate."""
    panels = [
        (d['P0_iw'],  r"$P^0(i\Omega{=}0)$"),
        (d['Pt_iw'],  r"$\tilde{P}(i\Omega{=}0)$"),
        (d['Wqq_iw'], r"$W_{QQ'}^{\rm direct}(i\Omega{=}0)$"),
        (d['W_mo'],   r"$W_{\rm MO}(i\Omega{=}0)$"),
        (d['W_ao'],   r"$W_{\rm AO}(i\Omega{=}0)$"),
        (d['U_qq'][np.newaxis], r"$U_{QQ'}$ (static)"),
        (d['Xi_iw'],  r"$\Xi(i\Omega{=}0) = 2U - W_{\rm MO}$"),
        (d['P_bse'],  r"$P_{\rm BSE}(i\Omega{=}0)$"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle(
        f"BSE Kernel Ξ: Pipeline Overview  "
        f"(H₂, nk={d['nk']}, NQ={d['NQ']}, q_idx={d['q_idx']})",
        fontsize=11, fontweight='bold'
    )

    for ax, (arr, title) in zip(axes.flat, panels):
        mat = static(arr) if arr.ndim == 3 else arr
        heatmap(ax, mat, title)

    fig.tight_layout()
    _save(fig, save_dir, 'fig1_pipeline_heatmaps.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Frequency dependence
# ─────────────────────────────────────────────────────────────────────────────

def fig_frequency(d: dict, save_dir: str):
    """Frobenius norms and diagonal traces vs iΩ for all frequency-dependent quantities."""
    Omega  = d['Omega']
    OmEV   = Omega * AU2EV    # convert to eV for x-axis

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Frequency Dependence  (q_idx={d['q_idx']})",
        fontsize=11, fontweight='bold'
    )

    # — Left: Frobenius norms ———————————————————————————————————————————————
    ax = axes[0]
    curves = [
        (d['P0_iw'],  r"$\|P^0(i\Omega)\|_F$",       'C0',   '-'),
        (d['Pt_iw'],  r"$\|\tilde{P}(i\Omega)\|_F$",  'C1',   '-'),
        (d['Wqq_iw'], r"$\|W_{QQ'}^{\rm dir}(i\Omega)\|_F$", 'C2', '--'),
        (d['W_mo'],   r"$\|W_{\rm MO}(i\Omega)\|_F$", 'C3',   '-'),
        (d['W_ao'],   r"$\|W_{\rm AO}(i\Omega)\|_F$", 'C4',   ':'),
        (d['Xi_iw'],  r"$\|\Xi(i\Omega)\|_F$",         'C5',   '-'),
        (d['P_bse'],  r"$\|P_{\rm BSE}(i\Omega)\|_F$",'C6',   '-'),
    ]
    for arr, label, color, ls in curves:
        ax.plot(OmEV, frob(arr), label=label, color=color, ls=ls, lw=1.5)
    ax.set_xlabel(r'$i\Omega_n$ (eV)', fontsize=10)
    ax.set_ylabel(r'$\|\cdot\|_F$', fontsize=10)
    ax.set_title('Frobenius norms', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(OmEV[0], OmEV[-1])
    ax.axvline(0, color='k', lw=0.5, ls='--')

    # — Right: Diagonal traces (Re) ──────────────────────────────────────────
    ax = axes[1]
    trace_curves = [
        (d['P0_iw'],  r"${\rm Re}\,{\rm Tr}\,P^0(i\Omega)$",       'C0', '-'),
        (d['Pt_iw'],  r"${\rm Re}\,{\rm Tr}\,\tilde{P}(i\Omega)$",  'C1', '-'),
        (d['Wqq_iw'], r"${\rm Re}\,{\rm Tr}\,W_{QQ'}^{\rm dir}$",   'C2', '--'),
        (d['W_mo'],   r"${\rm Re}\,{\rm Tr}\,W_{\rm MO}$",           'C3', '-'),
        (d['Xi_iw'],  r"${\rm Re}\,{\rm Tr}\,\Xi$",                  'C5', '-'),
        (d['P_bse'],  r"${\rm Re}\,{\rm Tr}\,P_{\rm BSE}$",          'C6', '-'),
    ]
    for arr, label, color, ls in trace_curves:
        ax.plot(OmEV, diag_trace(arr), label=label, color=color, ls=ls, lw=1.5)
    ax.set_xlabel(r'$i\Omega_n$ (eV)', fontsize=10)
    ax.set_ylabel(r'${\rm Re}\,{\rm Tr}$', fontsize=10)
    ax.set_title('Diagonal traces (real part)', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(OmEV[0], OmEV[-1])
    ax.axvline(0, color='k', lw=0.5, ls='--')

    fig.tight_layout()
    _save(fig, save_dir, 'fig2_frequency_dependence.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — W transformation: MO vs AO comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_mo_vs_ao(d: dict, save_dir: str):
    """Side-by-side comparison of the two W index-swap routes."""
    W_mo = d['W_mo']
    W_ao = d['W_ao']
    diff = W_mo - W_ao
    Omega = d['Omega']
    OmEV  = Omega * AU2EV

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"W Index-Swap: MO vs AO Comparison  (q_idx={d['q_idx']})",
        fontsize=11, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # — Row 1: heatmaps at Ω=0 ───────────────────────────────────────────────
    ax_mo  = fig.add_subplot(gs[0, 0])
    ax_ao  = fig.add_subplot(gs[0, 1])
    ax_dif = fig.add_subplot(gs[0, 2])

    heatmap(ax_mo,  static(W_mo),        r"${\rm Re}\,W_{\rm MO}(i\Omega{=}0)$")
    heatmap(ax_ao,  static(W_ao),        r"${\rm Re}\,W_{\rm AO}(i\Omega{=}0)$")

    # difference: absolute value, different colormap
    diff0 = np.abs(static(diff))
    im = ax_dif.imshow(diff0, cmap='hot_r', aspect='auto')
    ax_dif.set_title(r"$|W_{\rm MO} - W_{\rm AO}|$ at $\Omega{=}0$", fontsize=9)
    ax_dif.set_xlabel("Q'", fontsize=7); ax_dif.set_ylabel("Q", fontsize=7)
    ax_dif.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax_dif, fraction=0.046, pad=0.04)

    # — Row 2: norm difference and diagonal difference vs iΩ ─────────────────
    ax_norm = fig.add_subplot(gs[1, :2])
    ax_diag = fig.add_subplot(gs[1, 2])

    diff_norms = frob(diff)
    ax_norm.semilogy(OmEV, diff_norms + 1e-20, color='C3', lw=1.8)
    ax_norm.axvline(0, color='k', lw=0.5, ls='--')
    ax_norm.set_xlabel(r'$i\Omega_n$ (eV)', fontsize=10)
    ax_norm.set_ylabel(r'$\|W_{\rm MO}(i\Omega) - W_{\rm AO}(i\Omega)\|_F$', fontsize=10)
    ax_norm.set_title('Difference norm vs frequency', fontsize=10)
    ax_norm.set_xlim(OmEV[0], OmEV[-1])

    # diagonal of MO vs AO at Ω=0 (bar chart over Q index)
    nQ = d['NQ']
    diag_mo0 = np.diag(static(W_mo)).real
    diag_ao0 = np.diag(static(W_ao)).real
    x = np.arange(nQ)
    w = 0.4
    ax_diag.bar(x - w/2, diag_mo0, width=w, label=r'$W_{\rm MO}$', color='C3', alpha=0.7)
    ax_diag.bar(x + w/2, diag_ao0, width=w, label=r'$W_{\rm AO}$', color='C4', alpha=0.7)
    ax_diag.set_xlabel('Q index', fontsize=9)
    ax_diag.set_ylabel(r'${\rm Re}\,W_{QQ}(i\Omega{=}0)$', fontsize=9)
    ax_diag.set_title('Diagonal at Ω=0', fontsize=10)
    ax_diag.legend(fontsize=8)
    ax_diag.tick_params(labelsize=7)

    _save(fig, save_dir, 'fig3_mo_vs_ao.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Kernel Ξ construction and verification
# ─────────────────────────────────────────────────────────────────────────────

def fig_kernel(d: dict, save_dir: str):
    """Show 2U, W, Ξ=2U-W at Ω=0; eigenvalue spectra; Dyson residual."""
    NQ    = d['NQ']
    Omega = d['Omega']
    OmEV  = Omega * AU2EV
    i0    = len(Omega) // 2   # Ω=0 index

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        fr"Kernel $\Xi = 2U - W$ Construction  (q_idx={d['q_idx']})",
        fontsize=11, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # — Row 1: heatmaps of 2U, W_mo, Ξ at Ω=0, and Im(Ξ) ───────────────────
    ax_2U  = fig.add_subplot(gs[0, 0])
    ax_W   = fig.add_subplot(gs[0, 1])
    ax_Xi  = fig.add_subplot(gs[0, 2])
    ax_XiI = fig.add_subplot(gs[0, 3])

    heatmap(ax_2U, 2.0 * d['U_qq'],      r"$2U_{QQ'}$")
    heatmap(ax_W,  static(d['W_mo']),     r"${\rm Re}\,W_{\rm MO}(i\Omega{=}0)$")
    heatmap(ax_Xi, static(d['Xi_iw']),    r"${\rm Re}\,\Xi(i\Omega{=}0)$")

    # Imaginary part of Ξ at Ω=0 (should be small for real systems)
    Xi0_im = d['Xi_iw'][i0].imag
    vmax_i = max(np.max(np.abs(Xi0_im)), 1e-14)
    im = ax_XiI.imshow(Xi0_im, cmap='RdBu_r',
                        vmin=-vmax_i, vmax=vmax_i, aspect='auto')
    ax_XiI.set_title(r"${\rm Im}\,\Xi(i\Omega{=}0)$", fontsize=9)
    ax_XiI.set_xlabel("Q'", fontsize=7); ax_XiI.set_ylabel("Q", fontsize=7)
    ax_XiI.tick_params(labelsize=6)
    plt.colorbar(im, ax=ax_XiI, fraction=0.046, pad=0.04, label='Im')

    # — Row 2: eigenvalue spectra, frequency-resolved eigvals, Dyson residual
    ax_eig  = fig.add_subplot(gs[1, 0])
    ax_eigw = fig.add_subplot(gs[1, 1:3])
    ax_dys  = fig.add_subplot(gs[1, 3])

    # Sorted eigenvalues at Ω=0
    eigs_P0  = np.sort(np.linalg.eigvalsh(d['P0_iw'][i0].real))
    eigs_W   = np.sort(np.linalg.eigvalsh(d['W_mo'][i0].real))
    eigs_Xi  = np.sort(np.linalg.eigvalsh(d['Xi_iw'][i0].real))
    eigs_Pb  = np.sort(np.linalg.eigvalsh(d['P_bse'][i0].real))
    idx = np.arange(NQ)
    ax_eig.plot(idx, eigs_P0, 'o-', label=r'$P^0$',          ms=4, lw=1.2, color='C0')
    ax_eig.plot(idx, eigs_W,  's-', label=r'$W_{\rm MO}$',   ms=4, lw=1.2, color='C3')
    ax_eig.plot(idx, eigs_Xi, '^-', label=r'$\Xi$',           ms=4, lw=1.2, color='C5')
    ax_eig.plot(idx, eigs_Pb, 'D-', label=r'$P_{\rm BSE}$',  ms=4, lw=1.2, color='C6')
    ax_eig.axhline(0, color='k', lw=0.5, ls='--')
    ax_eig.set_xlabel('Eigenvalue index', fontsize=9)
    ax_eig.set_ylabel('Eigenvalue (real)', fontsize=9)
    ax_eig.set_title(r'Spectra at $\Omega=0$', fontsize=10)
    ax_eig.legend(fontsize=7)
    ax_eig.tick_params(labelsize=7)

    # Frequency-resolved: largest eigenvalue of Ξ(iΩ) vs iΩ
    # and Frobenius norm of Ξ vs iΩ
    eig_max_Xi = np.array([np.linalg.eigvalsh(d['Xi_iw'][iw].real).max()
                           for iw in range(len(Omega))])
    eig_min_Xi = np.array([np.linalg.eigvalsh(d['Xi_iw'][iw].real).min()
                           for iw in range(len(Omega))])
    ax_eigw.plot(OmEV, eig_max_Xi, 'C5-', lw=1.5, label=r'$\lambda_{\max}[\Xi(i\Omega)]$')
    ax_eigw.plot(OmEV, eig_min_Xi, 'C5--', lw=1.5, label=r'$\lambda_{\min}[\Xi(i\Omega)]$')
    ax_eigw.plot(OmEV, frob(d['Xi_iw']), 'C5:', lw=1.5, label=r'$\|\Xi(i\Omega)\|_F$')
    ax_eigw.axvline(0, color='k', lw=0.5, ls='--')
    ax_eigw.axhline(0, color='k', lw=0.5, ls='--')
    ax_eigw.set_xlabel(r'$i\Omega_n$ (eV)', fontsize=10)
    ax_eigw.set_title(r'$\Xi(i\Omega_n)$ — eigenvalue bounds and norm', fontsize=10)
    ax_eigw.legend(fontsize=8)
    ax_eigw.set_xlim(OmEV[0], OmEV[-1])
    ax_eigw.tick_params(labelsize=7)

    # Dyson residual ||(I - P⁰Ξ) P_BSE - P⁰||_F per frequency
    ax_dys.semilogy(OmEV, d['dyson_res'] + 1e-20, 'C6-o', ms=3, lw=1.5)
    ax_dys.axvline(0, color='k', lw=0.5, ls='--')
    ax_dys.set_xlabel(r'$i\Omega_n$ (eV)', fontsize=10)
    ax_dys.set_ylabel(r'$\|(I - P^0\Xi)P_{\rm BSE} - P^0\|_F$', fontsize=9)
    ax_dys.set_title('Dyson residual (should be ~ε_mach)', fontsize=10)
    ax_dys.set_xlim(OmEV[0], OmEV[-1])
    ax_dys.tick_params(labelsize=7)

    _save(fig, save_dir, 'fig4_kernel_construction.pdf')


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, save_dir: str, name: str):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Step-by-step BSE kernel Ξ construction with plots.'
    )
    p.add_argument('--example_dir', default=None,
                   help='Path to example directory. '
                        'Defaults to example/H2_correct_fq_8k.')
    p.add_argument('--q_idx', type=int, default=1,
                   help='q-point index (default: 1).')
    p.add_argument('--beta', type=float, default=1000.0,
                   help='Inverse temperature β in a.u. (default: 1000).')
    p.add_argument('--save', default=None, metavar='DIR',
                   help='Directory to save PDF figures. If omitted, show interactively.')
    return p.parse_args()


def main():
    args = parse_args()

    example_dir = args.example_dir or os.path.join(ROOT_DIR, 'example', 'H2_correct_fq_8k')
    ir_file     = os.path.join(example_dir, 'irgrid', '1e5.h5')

    for path in [example_dir, ir_file]:
        if not os.path.exists(path):
            print(f"ERROR: not found: {path}")
            sys.exit(1)

    print("=" * 70)
    print("BSE KERNEL Ξ — STEP-BY-STEP INSPECTION")
    print(f"  example : {example_dir}")
    print(f"  q_idx   : {args.q_idx}")
    print(f"  β       : {args.beta} a.u.")
    print("=" * 70)

    d = compute_all(example_dir, args.q_idx, args.beta, ir_file)

    print("\nPlotting …")
    fig_heatmaps(d, args.save)
    fig_frequency(d, args.save)
    fig_mo_vs_ao(d, args.save)
    fig_kernel(d, args.save)

    if args.save:
        print(f"\nAll figures saved to {args.save}/")
    else:
        plt.show()


if __name__ == '__main__':
    main()
