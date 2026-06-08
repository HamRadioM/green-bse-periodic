#!/usr/bin/env python3
"""
Plot heatmaps of W_act (active-space screened Coulomb) at the central
Matsubara frequency for several active-space choices on N2/STO-3G.

W_act is reshaped to (n_act^2, n_act^2) with row/column index = MO pair (p,q).
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import inv as spinv

sys.path.insert(0, "/Users/wenming/green-bse-periodic/green-bse-periodic-bse-solver/green_bse")

import casidaEq as casida
import gwtool
from irFT import IR_factory
from polarization import eval_W_MO_active, eval_screened_W_active, eval_VPiSWPiV, eval_P_dressed as eval_Pi0_dressed

# ---------------------------------------------------------------------------
# Paths & parameters
# ---------------------------------------------------------------------------
INPUT_H5 = "mean_field_input.h5"
SIM_H5   = "scGW_sim.h5"
INT_PATH = "df_hf_int/"
IR_FILE  = "irgrid/1e5.h5"
BETA     = 1000.0
ITER     = -1

# Active spaces to compare: (label, list of MO indices)
# N2 STO-3G: occ=7 (indices 0-6), virt=3 (indices 7-9)
ACTIVE_SPACES = [
    ("HOMO+LUMO\n[6,7]",            [6, 7]),
    ("HOMO-1→LUMO+1\n[5,6,7,8]",   [5, 6, 7, 8]),
    ("HOMO-1→LUMO+1\n[4,5,6,7,8]",   [4, 5, 6, 7, 8]),
    ("HOMO-2→LUMO+2\n[5,6,7,8,9]", [5, 6, 7, 8, 9]),
    # ("All valence\n[2..9]",          list(range(2, 10))),
]

# ---------------------------------------------------------------------------
# Load common data (done once)
# ---------------------------------------------------------------------------
print("Reading system data...")

with h5py.File(INPUT_H5, "r") as f:
    nao       = int(f["/params/nao"][()])
    mo_coeff  = f["/HF/mo_coeff"][()]
    mo_energy = f["/HF/mo_energy"][()]

with h5py.File(INT_PATH + "VQ_0.h5", "r") as f:
    V_raw = f["/0"][()]

NQ = V_raw.shape[1]
VQ_ao = np.zeros((1, NQ, nao, nao), dtype=np.complex128)
for idx in range(nao):
    VQ_ao[0, :, :, idx] = V_raw[0, :, :, idx*2] + 1j * V_raw[0, :, :, idx*2+1]
VQ_mo = casida.VQ_ao2mo(VQ_ao, mo_coeff)

# Build tildeP_iw (Q-space dressed polarizability, shared across all active spaces)
print("Computing P0_tilde and tildeP_iw...")
P0_tilde = gwtool.eval_P0_tilde_Q(ITER, nao, NQ, int_path=INT_PATH, sim_h5=SIM_H5)
P0_tilde = gwtool.symmetrize_P0(P0_tilde)

ir       = IR_factory(BETA, IR_FILE)
P0_iw    = ir.tauf_to_wb(P0_tilde)
niw      = P0_iw.shape[0]
iw_mid   = niw // 2

P_tilde_iw = np.zeros_like(P0_iw)
I_NQ = np.eye(NQ, dtype=np.complex128)
for iw in range(niw):
    p0 = P0_iw[iw, 0, 0]
    P_tilde_iw[iw, 0, 0] = spinv(I_NQ - p0) @ p0
P_tilde_iw = 0.5 * (P_tilde_iw + P_tilde_iw.conj().transpose(0, 1, 2, 4, 3))
tildeP_iw  = P_tilde_iw[:, :, 0, :, :][:, :, :, :, np.newaxis]

# ---------------------------------------------------------------------------
# Compute Pi0_iw (shared tau->freq FT, spin/k squeezed)
# ---------------------------------------------------------------------------
print("Computing Pi0_MO and VPiSWPiV for all active spaces...")

# ---------------------------------------------------------------------------
# Compute W_act and VPiSWPiV for each active space
# ---------------------------------------------------------------------------
panels = []   # (label, mo_idx, W_mat, VPiSWPiV_mat)

# Read G_tau once for Pi0 (eval_Pi0_MO_active re-reads internally, so cache it)
import h5py as _h5
with _h5.File(SIM_H5, "r") as _f:
    _it = _f["iter"][()]
G_tau_raw = _h5.File(SIM_H5, "r")[f"iter{_it}/G_tau/data"][()].view(complex)
# shape (ntau, ns, nk, nao, nao)
mo_coeff_full = mo_coeff   # (nao, nmo)

for label, mo_idx in ACTIVE_SPACES:
    n_act = len(mo_idx)
    n2    = n_act ** 2
    print(f"  Active space {mo_idx} (n_act={n_act})...")

    # --- W_act ---
    W_act = eval_W_MO_active(VQ_mo, tildeP_iw, mo_idx)
    W_mat = W_act[iw_mid].reshape(n2, n2).real

    # --- Pi0 in active MO space (tau), then FT ---
    Pi0_tau = gwtool.eval_Pi0_MO_active(ITER, mo_idx,
                                         input_h5=INPUT_H5,
                                         sim_h5=SIM_H5)
    # Symmetrize second half
    ntau = Pi0_tau.shape[0]
    for t in range(ntau // 2):
        Pi0_tau[ntau - t - 1, :, :] = Pi0_tau[t, :, :]
    Pi0_iw = ir.tauf_to_wb(Pi0_tau[:, 0, 0])   # (niw, n_act, n_act, n_act, n_act)

    # --- screened_W ---
    screened_W = eval_screened_W_active(W_act, Pi0_iw)

    # --- VQ_act slice ---
    VQ_act = VQ_mo[0, :, :, :][:, mo_idx, :][:, :, mo_idx]   # (NQ, n_act, n_act)

    # --- V Pi SW Pi V -> (niw, NQ, NQ) ---
    result    = eval_VPiSWPiV(VQ_act, Pi0_iw, screened_W)
    result_mat = result[iw_mid].real   # (NQ, NQ)

    # --- Pi0_dressed in active MO space, projected to Q-space: V Pi0_dr V† ---
    Pi0_dr    = eval_Pi0_dressed(Pi0_iw)                         # (niw, n_act,n_act,n_act,n_act)
    V_flat    = VQ_act.reshape(NQ, n_act**2)                     # (NQ, n_act²)
    Pi0_dr_mat = Pi0_dr[iw_mid].reshape(n_act**2, n_act**2)
    Pi0_dr_Q  = (V_flat @ Pi0_dr_mat @ V_flat.conj().T).real    # (NQ, NQ)

    diff_mat  = Pi0_dr_Q - result_mat                            # (NQ, NQ), Q-space diff

    # --- Back-project diff to MO-pair space: V† diff V ---
    diff_MO   = (V_flat.conj().T @ diff_mat @ V_flat).real       # (n_act², n_act²)

    # --- RPA-dressed: P_input = Pi0 - P_ph, then P_input[I - 2 P_input]^{-1} ---
    # Back-project VΠ SW ΠV to MO-pair space: V† result V
    P_ph_MO = np.zeros((niw, n2, n2), dtype=np.complex128)
    for iw in range(niw):
        P_ph_MO[iw] = V_flat.conj().T @ result[iw] @ V_flat

    Pi0_iw_mat = Pi0_iw.reshape(niw, n2, n2)          # (niw, n2, n2)
    P_input_iw  = Pi0_iw_mat - P_ph_MO                 # (niw, n2, n2)
    P_rpa_iw    = eval_Pi0_dressed(P_input_iw)         # (niw, n2, n2)

    # Project P_RPA to Q-space for display: V P_rpa V†
    P_rpa_Q = (V_flat @ P_rpa_iw[iw_mid] @ V_flat.conj().T).real   # (NQ, NQ)

    # Diff between RPA-dressed (with p-h correction) and plain P_dressed (without)
    rpa_corr_Q = P_rpa_Q - Pi0_dr_Q                    # (NQ, NQ)

    # Back-project RPA result to MO-pair for display
    P_rpa_MO = P_rpa_iw[iw_mid].real                   # (n2, n2)

    panels.append((label, mo_idx, W_mat, result_mat, Pi0_dr_Q, diff_mat, diff_MO,
                   P_rpa_MO, P_rpa_Q, rpa_corr_Q))

    # Print the diff_MO element for the HOMO→LUMO (MO 6→7) pair
    if 6 in mo_idx and 7 in mo_idx:
        loc6 = list(mo_idx).index(6)
        loc7 = list(mo_idx).index(7)
        flat_67 = loc6 * n_act + loc7
        flat_76 = loc7 * n_act + loc6
        val = diff_MO[flat_67, flat_76]
        print(f"    diff_MO[(6,7),(7,6)]            = {val:.6e}  (n_act={n_act})")
        val_rpa = P_rpa_MO[flat_67, flat_76]
        print(f"    P_RPA_MO[(6,7),(7,6)]           = {val_rpa:.6e}  (n_act={n_act})")
    else:
        print(f"    MO 6 or 7 not in active space {mo_idx}, skipping.")

# ---------------------------------------------------------------------------
# Plot: 6 rows × n_panels columns
#   Row 0: W_act                              (MO-pair space, n_act²×n_act²)
#   Row 1: VΠ SW ΠV                           (Q-space, NQ×NQ)
#   Row 2: P⁰[I−2P⁰]⁻¹ − VΠ SW ΠV          (Q-space difference)
#   Row 3: V† diff V                          (back-projected to MO-pair space)
#   Row 4: (P⁰−P_ph)[I−2(P⁰−P_ph)]⁻¹       (RPA-dressed, MO-pair space)
#   Row 5: P_RPA − P⁰[I−2P⁰]⁻¹  (Q-space)  (correction from p-h term)
# ---------------------------------------------------------------------------
n_panels = len(panels)
fig, axes = plt.subplots(6, n_panels, figsize=(3 * n_panels, 18))

row_titles = [
    "W_act  (MO-pair space)",
    "VΠ SW ΠV  (Q-space)",
    "P⁰[I−2P⁰]⁻¹ − VΠ SW ΠV  (Q-space)",
    "V† diff V  (MO-pair space)",
    "(P⁰−P_ph)[I−2(P⁰−P_ph)]⁻¹  (MO-pair)",
    "P_RPA − P⁰[I−2P⁰]⁻¹  (Q-space)",
]

def _plot_Q(ax, mat, title_extra=""):
    vmax = np.max(np.abs(mat)) or 1e-30
    im = ax.imshow(mat.real, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_step = max(1, NQ // 8)
    ticks = list(range(0, NQ, tick_step))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_yticklabels(ticks, fontsize=7)
    ax.set_xlabel("Q'", fontsize=8); ax.set_ylabel("Q", fontsize=8)
    ax.set_title(f"||·||_F={np.linalg.norm(mat):.3e}{title_extra}", fontsize=8)

def _plot_MO(ax, mat, mo_idx, title_extra=""):
    n_act = len(mo_idx)
    n2    = n_act ** 2
    pairs = [f"{p},{q}" for p in mo_idx for q in mo_idx]
    vmax  = np.max(np.abs(mat)) or 1e-30
    im = ax.imshow(mat.real, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n2)); ax.set_yticks(range(n2))
    ax.set_xticklabels(pairs, rotation=90, fontsize=6)
    ax.set_yticklabels(pairs, fontsize=6)
    for i in range(1, n_act):
        ax.axhline(i * n_act - 0.5, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(i * n_act - 0.5, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("MO pair (r,s)", fontsize=8)
    ax.set_ylabel("MO pair (p,q)", fontsize=8)
    ax.set_title(f"||·||_F={np.linalg.norm(mat):.3e}{title_extra}", fontsize=8)

for col, (label, mo_idx, W_mat, VP_mat, Pi0_dr_Q, diff_mat, diff_MO,
          P_rpa_MO, P_rpa_Q, rpa_corr_Q) in enumerate(panels):
    n_act = len(mo_idx)

    # ---- Row 0: W_act ----
    _plot_MO(axes[0, col], W_mat, mo_idx, title_extra=f"\n{label}  n_act={n_act}")

    # ---- Row 1: VΠ SW ΠV ----
    _plot_Q(axes[1, col], VP_mat)

    # ---- Row 2: Q-space difference ----
    _plot_Q(axes[2, col], diff_mat,
            title_extra=f"\nmax|Δ|={np.max(np.abs(diff_mat)):.2e}")

    # ---- Row 3: MO-pair back-projection of diff ----
    _plot_MO(axes[3, col], diff_MO, mo_idx,
             title_extra=f"\nmax|Δ|={np.max(np.abs(diff_MO)):.2e}")

    # ---- Row 4: RPA-dressed P with p-h correction (MO-pair space) ----
    _plot_MO(axes[4, col], P_rpa_MO, mo_idx,
             title_extra=f"\nmax|·|={np.max(np.abs(P_rpa_MO)):.2e}")

    # ---- Row 5: P_RPA − P_dressed (Q-space) ----
    _plot_Q(axes[5, col], rpa_corr_Q,
            title_extra=f"\nmax|Δ|={np.max(np.abs(rpa_corr_Q)):.2e}")

# Row labels on the left
for row, title in enumerate(row_titles):
    axes[row, 0].annotate(title, xy=(0, 0.5),
                          xytext=(-axes[row, 0].yaxis.labelpad - 65, 0),
                          xycoords=axes[row, 0].yaxis.label,
                          textcoords="offset points",
                          ha="right", va="center", fontsize=9,
                          rotation=90, fontweight="bold")

fig.suptitle(
    "N2 / STO-3G  –  active-space polarization analysis  (Re, central Matsubara freq.)",
    fontsize=11, y=1.01)
plt.tight_layout()
outfile = "W_active_space_heatmap.pdf"
plt.savefig(outfile, bbox_inches="tight")
print(f"\nSaved → {outfile}")
plt.show()
