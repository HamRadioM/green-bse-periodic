#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    2-D hexagonal  crystal of BN (one BN/cell #
#    STO-3G, 6x6x1) -- a band insulator (large sigma/sigma* gap, narrow bands). #
#    Along the high-symmetry path Gamma-M-K-Gamma, plots:                      #
#                                                                             #
#      (top)    Wannier-interpolated QP band structure eps_v(k), eps_c(k)      #
#               (HF bands shown dashed for contrast; exact mesh eigenvalues    #
#               overlaid as dots to confirm node-exactness).                   #
#      (bottom) Singlet excitonic dispersion A(Q,omega), obtained by Wannier-  #
#               interpolating the two-step polarization P_bse(Q, iOmega) over  #
#               the exciton momentum Q and continuing Tr P to the real axis    #
#               (smooth -- no per-Q discrete pole detection).  QP particle-    #
#               hole continuum edges overlaid.                                 #
#                                                                             #
#    Run:  python test_band_and_exciton_wannier.py                             #
#                                                                             #

import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import h5py
import matplotlib.pyplot as plt

from green_mbtools.pesto import winter

sys.path.append(str(Path(__file__).resolve().parent / "../../green_bse"))

import periodic_twostep_bse as P
import periodic_active_bse as pab
import periodic_integrals as pint
import wannier_exciton as wx

AU2EV = 27.211386245981

INPUT, SIM = "mean_field_input.h5", "scGW_sim_beta_1000.h5"
INTP = "df_int/"                              # bare-Coulomb exchange integrals
IR = "../H2_STO3G_grid/irgrid/1e5.h5"           # IR grid is system-independent
BETA = 1000.0
CHANNEL = "singlet"
ETA = 0.20                  # spectral broadening (eV)
W_MAX = 1.5                  # pole window (a.u.)

NK1 = 6                     # k-mesh is NK1 x NK1 x 1
NPTS = 60                   # interpolated points per high-symmetry segment

# hexagonal high-symmetry corners in fractional (scaled) reciprocal coords
# (Setyawan-Curtarolo HEX convention; all land on the 6x6 mesh).
GAMMA = np.array([0.0, 0.0, 0.0])
M = np.array([0.5, 0.0, 0.0])
K = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
CORNERS = [(r"$\Gamma$", GAMMA), ("M", M), ("K", K), (r"$\Gamma$", GAMMA)]


def build_qpath(corners, npts):
    """Dense fractional Q-path; returns (frac (N,3), segment-boundary indices)."""
    segs, bounds = [], [0]
    for a, b in zip(corners[:-1], corners[1:]):
        segs.append(np.linspace(a[1], b[1], npts, endpoint=False))
        bounds.append(bounds[-1] + npts)
    segs.append(corners[-1][1][None, :])
    bounds[-1] += 1
    return np.vstack(segs), bounds


def on_mesh(frac, tol=1e-9):
    """True if a fractional point lands on the NK1 x NK1 mesh."""
    return (abs(frac[0] * NK1 - round(frac[0] * NK1)) < tol and
            abs(frac[1] * NK1 - round(frac[1] * NK1)) < tol)


def kidx(frac):
    """Mesh index of a fractional k that lands on the NK1 x NK1 grid."""
    i = int(round(frac[0] * NK1)) % NK1
    j = int(round(frac[1] * NK1)) % NK1
    return i * NK1 + j


def bands_from_H(Hk, Sk):
    """Generalised eigenvalues of (H(k), S(k)) for every k -> (nk, nao)."""
    return np.array([sla.eigh(Hk[k], Sk[k], eigvals_only=True)
                     for k in range(Hk.shape[0])])


def wannier_bands(solver, kfrac, path_frac):
    """Wannier-interpolated QP and HF bands (eV) along path_frac -> (nP, nao) each."""
    s = solver
    Sk = s.results["rSk"]                                   # (1, nk, nao, nao)
    Fk = s.results["rFk"]                                   # (1, nk, nao, nao) static Fock
    C = s.vexMO
    eps = s.mo_energy                                       # (nk, nao) a.u.
    # QP Hamiltonian in AO basis: H = S C diag(eps) C^dag S
    SC = np.einsum("kmn,knp->kmp", Sk[0], C[0])
    H_qp = np.einsum("kmp,kp,knp->kmn", SC, eps, SC.conj())[None]

    S_i = winter.interpolate(Sk,   kfrac, path_frac, dim=2, hermi=True)[0]
    Hqp_i = winter.interpolate(H_qp, kfrac, path_frac, dim=2, hermi=True)[0]
    Hhf_i = winter.interpolate(Fk,   kfrac, path_frac, dim=2, hermi=True)[0]
    return (bands_from_H(Hqp_i, S_i) * AU2EV,
            bands_from_H(Hhf_i, S_i) * AU2EV)


def qp_continuum(solver, kfrac, path_frac):
    """Frontier QP particle-hole continuum edges (eV) along the dense Q-path.

    eps_c(k+Q) - eps_v(k), min/max over mesh k.  eps_v(k) is exact on the mesh;
    eps_c(k+Q) (off-mesh) comes from the same Wannier interpolation of the QP
    Hamiltonian used for the bands, so the continuum is densified consistently.
    """
    s = solver
    Sk = s.results["rSk"]
    C = s.vexMO
    eps = s.mo_energy
    nao = eps.shape[1]
    SC = np.einsum("kmn,knp->kmp", Sk[0], C[0])
    H_qp = np.einsum("kmp,kp,knp->kmn", SC, eps, SC.conj())[None]

    targets = (kfrac[None, :, :] + path_frac[:, None, :]).reshape(-1, 3)
    Hc = winter.interpolate(H_qp, kfrac, targets, dim=2, hermi=True)[0]
    Sc = winter.interpolate(Sk,   kfrac, targets, dim=2, hermi=True)[0]
    eps_kq = bands_from_H(Hc, Sc).reshape(len(path_frac), s.nk, nao)

    ev = eps[:, s.occ - 1]
    trans = (eps_kq[:, :, s.occ] - ev[None, :]) * AU2EV
    return trans.min(axis=1), trans.max(axis=1)


def main():
    cfg = P.PeriodicTwoStepBSEConfig(
        input_file=INPUT, sim_file=SIM, int_path=INTP, ir_file=IR, method="plaspole",
        beta=BETA, q_idx=0, screening="qresolved", active_mode="series")
    s = P.PeriodicTwoStepBSESolver(cfg)
    s.load_input_data()
    s.solve_quasiparticles()
    s.build_intermediates()

    stored = pint.load_stored_integrals(INTP)
    sym = pint.load_kpair_symmetry(INPUT)
    ao, av = list(range(s.occ)), list(range(s.occ, s.nao))

    # ---- dense path + Cartesian x-spacing ----
    path_frac, bounds = build_qpath(CORNERS, NPTS)
    with h5py.File(INPUT, "r") as f:
        kfrac = f["symmetry/k/mesh_scaled"][:]
        kcart = f["symmetry/k/mesh"][:]
    B, *_ = np.linalg.lstsq(kfrac, kcart, rcond=None)          # frac @ B = cart
    pc = path_frac @ B
    xp = np.concatenate(([0.0],
                         np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=1))))
    tick_x = [xp[b if b < len(xp) else -1] for b in bounds]
    tick_lab = [CORNERS[0][0]] + [c[0] for c in CORNERS[1:]]

    # ---- (a) Wannier band structure ----
    print("Wannier-interpolating QP / HF band structure ...")
    qp_path, hf_path = wannier_bands(s, kfrac, path_frac)
    # exact mesh eigenvalues at on-mesh path points (node-exactness check)
    mesh_x, mesh_qp = [], []
    for n, fr in enumerate(path_frac):
        if on_mesh(fr):
            mesh_x.append(xp[n]); mesh_qp.append(s.mo_energy[kidx(fr)] * AU2EV)
    mesh_x = np.array(mesh_x); mesh_qp = np.array(mesh_qp)   # (Nm, nao)

    # ---- (b) exciton dispersion: stream Tr P over Q, continue to real axis ----
    # Memory-light: only the scalar Tr P(Q, iOmega) is interpolated to the dense
    # path (trace commutes with the linear Wannier interpolation), so the full
    # (NQ x NQ) polarization is never stored on the path -- avoids the
    # O(nP*niw*NQ^2) blow-up that OOMs for larger bases (large NQ).
    cont_lo, cont_hi = qp_continuum(s, kfrac, path_frac)
    emin, emax = cont_lo.min() - 2.0, cont_hi.max() + 2.0
    egrid = np.linspace(emin, emax, 700)
    print("Streaming Tr P over the %d-point Q-mesh -> %d path points ..."
          % (s.nk, len(path_frac)))
    A, node_err = wx.spectral_map_over_path(
        lambda Q: wx.P_at_Q(s, stored, sym, Q, ao, av, CHANNEL),
        range(s.nk), kfrac, path_frac, s.omega, egrid, eta_eV=ETA, dim=2)
    print("Tr P interpolation node-exactness (max |dTrP|): %.2e" % node_err)

    # ======================= figure =======================
    fig = plt.figure(figsize=(4.6, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.04],
                          height_ratios=[1.0, 1.25], hspace=0.13, wspace=0.02)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])

    # ---- (a) band structure (valence: band < occ, conduction: band >= occ) ----
    nband = qp_path.shape[1]
    for b in range(nband):
        valence = b < s.occ
        col = "#1f77b4" if valence else "#d62728"
        lab = None
        if b == 0:
            lab = "scGW(QP) valence"
        elif b == s.occ:
            lab = "scGW(QP) conduction"
        ax1.plot(xp, qp_path[:, b], "-", color=col, lw=1.5, label=lab)
        ax1.plot(xp, hf_path[:, b], "--", color=col, lw=0.9, alpha=0.5,
                 label=("HF" if b == 0 else None))
        ax1.scatter(mesh_x, mesh_qp[:, b], s=12, facecolors="none",
                    edgecolors=col, lw=0.9, zorder=3)
    ax1.set_ylabel("Band energy (eV)")
    ax1.set_title("2-D hexagonal BN crystal (STO-3G, 6x6)\n"
                  "Wannier-interpolated bands")
    ax1.legend(fontsize=6.5, ncol=2, loc="best")
    ax1.set_ylim(-10, 10)
    ax1.grid(alpha=0.2)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ---- (b) exciton dispersion ----
    Xg, Yg = np.meshgrid(xp, egrid)
    pcm = ax2.pcolormesh(Xg, Yg, A, shading="gouraud", cmap="inferno",
                         vmax=np.percentile(A[A > 0], 99) if np.any(A > 0) else 1.0)
    ax2.fill_between(xp, cont_lo, cont_hi, color="white", alpha=0.10, zorder=1)
    ax2.plot(xp, cont_lo, color="white", lw=1.0, ls="--", alpha=0.7, label="QP continuum")
    ax2.plot(xp, cont_hi, color="white", lw=1.0, ls="--", alpha=0.7)
    ax2.set_ylabel(r"Excitation energy $\omega$ (eV)")
    ax2.set_title(r"Singlet excitonic dispersion $A(Q,\omega)$")
    ax2.set_ylim(emin, emax)
    ax2.legend(fontsize=7, loc="upper center")
    fig.colorbar(pcm, cax=cax).set_label(r"$A(Q,\omega)$ singlet")

    # high-symmetry ticks (shared x-axis)
    ax2.set_xlim(xp[0], xp[-1])
    ax2.set_xticks(tick_x)
    ax2.set_xticklabels(tick_lab)
    for tx in tick_x:
        ax1.axvline(tx, color="0.7", lw=0.6, zorder=0)
        ax2.axvline(tx, color="white", lw=0.5, alpha=0.4, zorder=2)
    ax2.set_xlabel("crystal momentum  (k for bands, Q for excitons)")

    for ext in ("pdf", "png"):
        fig.savefig("band_and_exciton_wannier." + ext, dpi=150, bbox_inches="tight")
    print("Saved -> band_and_exciton_wannier.pdf / .png")


if __name__ == "__main__":
    main()
