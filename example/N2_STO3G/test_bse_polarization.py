#!/usr/bin/env python3
"""
BSE polarization test on N2/STO-3G.

P^ph  = V (I + Π W)^{-1} Π V†          [Q-space, NQ×NQ per frequency]
P^BSE = (I - P^ph)^{-1} P^ph            [Dyson equation in Q-space]

where Π = Pi0 (independent-particle polarizability, active MO-pair space)
      W  = screened Coulomb (active MO-pair space)
      V  = VQ slice for active MOs, flattened to (NQ, n_act²)
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.linalg import inv as spinv

sys.path.insert(0, "/Users/wenming/green-bse-periodic/green-bse-periodic-bse-solver/green_bse")

import casidaEq as casida
import gwtool
from irFT import IR_factory
from polarization import eval_W_MO_active, eval_Pph, eval_PBSE

# ---------------------------------------------------------------------------
# Paths & parameters
# ---------------------------------------------------------------------------
INPUT_H5 = "mean_field_input.h5"
SIM_H5   = "scGW_sim.h5"
INT_PATH = "df_hf_int/"
IR_FILE  = "irgrid/1e5.h5"
BETA     = 1000.0
ITER     = -1

ACTIVE_SPACES = [
    ("HOMO+LUMO\n[6,7]",            [6, 7]),
    ("HOMO-1→LUMO+1\n[5,6,7,8]",   [5, 6, 7, 8]),
    ("HOMO-2→LUMO+2\n[4,5,6,7,8,9]", [4, 5, 6, 7, 8, 9]),
    ("HOMO-4→LUMO+2\n[2,...,9]", [2, 3, 4, 5, 6, 7, 8, 9]),
    ("All orbital  \n", list(range(0, 10))),
]

# ---------------------------------------------------------------------------
# Load common data
# ---------------------------------------------------------------------------
print("Reading system data...")

with h5py.File(INPUT_H5, "r") as f:
    nao      = int(f["/params/nao"][()])
    mo_coeff = f["/HF/mo_coeff"][()]

with h5py.File(INT_PATH + "VQ_0.h5", "r") as f:
    V_raw = f["/0"][()]

NQ = V_raw.shape[1]
VQ_ao = np.zeros((1, NQ, nao, nao), dtype=np.complex128)
for idx in range(nao):
    VQ_ao[0, :, :, idx] = V_raw[0, :, :, idx*2] + 1j * V_raw[0, :, :, idx*2+1]
VQ_mo = casida.VQ_ao2mo(VQ_ao, mo_coeff)   # (1, NQ, nmo, nmo)

# Build tildeP_iw (full Q-space dressed polarizability via Dyson)
print("Building P_tilde in Q-space...")
P0_tilde = gwtool.eval_P0_tilde_Q(ITER, nao, NQ, int_path=INT_PATH, sim_h5=SIM_H5)
P0_tilde = gwtool.symmetrize_P0(P0_tilde)

ir    = IR_factory(BETA, IR_FILE)
P0_iw = ir.tauf_to_wb(P0_tilde)            # (niw, 1, 1, NQ, NQ)
niw   = P0_iw.shape[0]
iw_mid = niw // 2

I_NQ = np.eye(NQ, dtype=np.complex128)
tildeP_raw = np.zeros_like(P0_iw)
for iw in range(niw):
    p0 = P0_iw[iw, 0, 0]
    tildeP_raw[iw, 0, 0] = spinv(I_NQ - p0) @ p0
tildeP_raw = 0.5 * (tildeP_raw + tildeP_raw.conj().transpose(0, 1, 2, 4, 3))
tildeP_iw  = tildeP_raw[:, :, 0, :, :][:, :, :, :, np.newaxis]   # (niw,1,NQ,NQ,1)


# ---------------------------------------------------------------------------
# Loop over active spaces
# ---------------------------------------------------------------------------
panels = []   # (label, mo_idx, Pph_mat, Pbse_mat)

for label, mo_idx in ACTIVE_SPACES:
    n_act = len(mo_idx)
    n2    = n_act ** 2
    print(f"\nActive space {mo_idx}  (n_act={n_act})")

    # Pi0 in active MO space (tau → freq)
    # eval_Pi0_MO_active returns (ntau, ns, nk, n_act, n_act, n_act, n_act)
    Pi0_tau = gwtool.eval_Pi0_MO_active(ITER, mo_idx, input_h5=INPUT_H5, sim_h5=SIM_H5)
    ntau = Pi0_tau.shape[0]
    for t in range(ntau // 2):
        Pi0_tau[ntau - t - 1] = Pi0_tau[t]

    # FT: collapse (ns, nk) into a batch dimension, transform, then restore
    ns_loc = Pi0_tau.shape[1]
    nk_loc = Pi0_tau.shape[2]
    Pi0_tau_flat = Pi0_tau.reshape(ntau, ns_loc * nk_loc, n_act, n_act, n_act, n_act)
    Pi0_iw_flat  = ir.tauf_to_wb(Pi0_tau_flat)
    Pi0_iw_k     = Pi0_iw_flat.reshape(niw, ns_loc, nk_loc, n2, n2)

    # W in active MO space — per k-point: (niw, nk, n_act, n_act, n_act, n_act)
    W_act_k = eval_W_MO_active(VQ_mo, tildeP_iw, mo_idx)
    W_mat_k = W_act_k.reshape(niw, nk_loc, n2, n2)      # (niw, nk, n2, n2)

    # V projection matrix — (nk, NQ, n_act, n_act); nk=1 for this molecular example
    VQ_act_k = VQ_mo[:, :, mo_idx, :][:, :, :, mo_idx]  # (nk, NQ, n_act, n_act)
    V_flat = VQ_act_k[0].reshape(NQ, n2)                 # (NQ, n2), for back-projection

    # P^ph: (niw, ns, nk, NQ, NQ) — k/spin resolved, no averaging yet
    # W passed as (niw, n2, n2): use k=0 slice for nk=1; for multi-k pass full W_mat_k
    P_ph_k = eval_Pph(VQ_act_k, Pi0_iw_k, W_mat_k[:, 0])   # (niw, ns, nk, NQ, NQ)

    # k-sum and spin-sum → full Q-space P^ph for the Dyson equation
    P_ph_full = P_ph_k.sum(axis=(1, 2)) / nk_loc           # (niw, NQ, NQ)

    P_bse = eval_PBSE(P_ph_full)                            # (niw, NQ, NQ)

    Pph_mid  = P_ph_full[iw_mid].real
    Pbse_mid = P_bse[iw_mid].real

    panels.append((label, mo_idx, Pph_mid, Pbse_mid))

    # Print HOMO→LUMO element back-projected to MO-pair space
    if 6 in mo_idx and 7 in mo_idx:
        loc6 = list(mo_idx).index(6)
        loc7 = list(mo_idx).index(7)
        flat_67 = loc6 * n_act + loc7
        flat_76 = loc7 * n_act + loc6
        Pph_MO  = (V_flat.conj().T @ P_ph_full[iw_mid] @ V_flat).real
        Pbse_MO = (V_flat.conj().T @ P_bse[iw_mid]     @ V_flat).real
        print(f"  V P^ph  V [(6,7),(7,6)] = {Pph_MO[flat_67, flat_76]:.6e}")
        print(f"  V P^BSE V [(6,7),(7,6)] = {Pbse_MO[flat_67, flat_76]:.6e}")

# ---------------------------------------------------------------------------
# Plot: 2 rows × n_panels columns
#   Row 0: P^ph   (Q-space)
#   Row 1: P^BSE  (Q-space)
# ---------------------------------------------------------------------------
n_panels = len(panels)
fig, axes = plt.subplots(2, n_panels, figsize=(3.2 * n_panels, 7))

row_titles = ["P^ph  (Q-space)", "P^BSE  (Q-space)"]


def _plot_Q(ax, mat, title=""):
    vmax = np.max(np.abs(mat)) or 1e-30
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    tick_step = max(1, NQ // 8)
    ticks = list(range(0, NQ, tick_step))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=7)
    ax.set_yticklabels(ticks, fontsize=7)
    ax.set_xlabel("Q'", fontsize=8); ax.set_ylabel("Q", fontsize=8)
    ax.set_title(title, fontsize=8)


for col, (label, mo_idx, Pph_mat, Pbse_mat) in enumerate(panels):
    n_act = len(mo_idx)
    col_title = f"{label}  n_act={n_act}"
    _plot_Q(axes[0, col], Pph_mat,
            title=f"{col_title}\n||·||_F={np.linalg.norm(Pph_mat):.3e}")
    _plot_Q(axes[1, col], Pbse_mat,
            title=f"||·||_F={np.linalg.norm(Pbse_mat):.3e}")

for row, title in enumerate(row_titles):
    axes[row, 0].annotate(title, xy=(0, 0.5),
                          xytext=(-axes[row, 0].yaxis.labelpad - 60, 0),
                          xycoords=axes[row, 0].yaxis.label,
                          textcoords="offset points",
                          ha="right", va="center", fontsize=9,
                          rotation=90, fontweight="bold")

fig.suptitle(
    "N2 / STO-3G  –  P^ph and P^BSE at central Matsubara frequency (Re)",
    fontsize=11, y=1.01)
plt.tight_layout()
outfile = "bse_polarization.pdf"
plt.savefig(outfile, bbox_inches="tight")
print(f"\nSaved → {outfile}")
plt.show()
