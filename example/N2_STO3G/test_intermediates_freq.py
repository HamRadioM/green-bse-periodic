#!/usr/bin/env python3
"""
Frequency-resolved intermediate matrices for several active spaces.

At every bosonic Matsubara frequency iΩ_n this script computes:

  P^(0)(iΩ)    — full bare polarizability in Q-space  (eval_P0_tilde_Q)
  P̃(iΩ)        — screened polarizability via Dyson    (dyson_equation)
  Π_act(iΩ)    — active-space bubble in MO-pair space (eval_Pi0_MO_active)
  W_act(iΩ)    — screened Coulomb in MO-pair space    (eval_W_MO_active)
  P_ph(iΩ)     — particle-hole kernel in Q-space      (eval_Pph)
  P_BSE(iΩ)    — BSE polarizability via Dyson         (eval_PBSE)

Results are saved to an HDF5 file and Frobenius-norm frequency plots are
produced for all active spaces side-by-side.
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/wenming/green-bse-periodic/green-bse-periodic-bse-solver/green_bse")

import casidaEq as casida
import contract as ct
import gwtool
from irFT import IR_factory
from polarization import (
    eval_W_MO_active,
    eval_Pph,
    eval_PBSE,
    dyson_equation,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
INPUT_H5 = "mean_field_input.h5"
SIM_H5   = "scGW_sim.h5"
INT_PATH = "df_hf_int/"
IR_FILE  = "irgrid/1e5.h5"
BETA     = 1000.0
ITER     = -1
OUT_H5   = "intermediates_freq.h5"

ACTIVE_SPACES = [
    [6, 7],
    [5, 6, 7, 8],
    [4, 5, 6, 7, 8, 9],
    list(range(0, 10)),
]

# ---------------------------------------------------------------------------
# System data
# ---------------------------------------------------------------------------
print("Reading system data...")
with h5py.File(INPUT_H5, "r") as f:
    nao       = int(f["/params/nao"][()])
    mo_coeff  = f["/HF/mo_coeff"][()]
    mo_energy = f["/HF/mo_energy"][()]
    S_raw     = f["/HF/S-k"][()].view(complex)
    S_raw     = S_raw.reshape(S_raw.shape[:-1])   # (..., nao, nao)
S = S_raw.reshape(-1, nao, nao)[0]               # (nao, nao) — first k/spin block

VQ_ao = ct.readVQ(INT_PATH + "VQ_0.h5")          # (1, NQ, nao, nao)
NQ    = VQ_ao.shape[1]
VQ_mo = casida.VQ_ao2mo(VQ_ao, mo_coeff, S=S)    # (1, NQ, nmo, nmo)

occ = int(round(sum(1 for e in mo_energy if e < 0)))
print(f"  nao={nao}, NQ={NQ}, HOMO={occ-1}, LUMO={occ}")

# ---------------------------------------------------------------------------
# Bosonic Matsubara grid
# ---------------------------------------------------------------------------
ir = IR_factory(BETA, IR_FILE)
with h5py.File(IR_FILE, "r") as f:
    ngrid = f["/bose/ngrid"][()]
omega_n = 2 * np.pi * ngrid / BETA    # (niw,)  bosonic Matsubara frequencies

# ---------------------------------------------------------------------------
# Full-system polarizabilities (shared across all active spaces)
# ---------------------------------------------------------------------------
print("\nBuilding full-system P^(0) and P̃ ...")
P0_tilde = gwtool.eval_P0_tilde_Q(ITER, nao, NQ, int_path=INT_PATH, sim_h5=SIM_H5)
P0_tilde = gwtool.symmetrize_P0(P0_tilde)

P0_iw  = ir.tauf_to_wb(P0_tilde)                                        # (niw, ns, 1, NQ, NQ)
niw    = P0_iw.shape[0]
ns     = P0_iw.shape[1]

tildeP_raw = dyson_equation(P0_iw)                                       # (niw, ns, 1, NQ, NQ)
tildeP_iw  = tildeP_raw[:, 0, 0, :, :][:, np.newaxis, :, :, np.newaxis] # (niw, 1, NQ, NQ, 1)

print(f"  niw={niw}, ns={ns},  ω range=[{omega_n.min():.3f}, {omega_n.max():.3f}] Ha")

# ---------------------------------------------------------------------------
# Frobenius-norm helper
# ---------------------------------------------------------------------------
def frob_iw(arr):
    """Frobenius norm at each frequency for an array with leading iw axis."""
    return np.array([np.linalg.norm(arr[iw]) for iw in range(arr.shape[0])])


# ---------------------------------------------------------------------------
# Loop over active spaces
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(ACTIVE_SPACES), 1,
                         figsize=(8, 3 * len(ACTIVE_SPACES)), sharex=True)
if len(ACTIVE_SPACES) == 1:
    axes = [axes]

with h5py.File(OUT_H5, "w") as fout:
    fout.attrs["BETA"]     = BETA
    fout.attrs["IR_FILE"]  = IR_FILE
    fout.attrs["INPUT_H5"] = INPUT_H5
    fout.attrs["SIM_H5"]   = SIM_H5
    fout["omega_n"]        = omega_n

    # Store full-system quantities once
    grp0 = fout.create_group("full_system")
    grp0["P0_iw"]      = P0_iw[:, 0, 0]       # (niw, NQ, NQ)
    grp0["tildeP_iw"]  = tildeP_raw[:, 0, 0]  # (niw, NQ, NQ)

    for ax, mo_idx in zip(axes, ACTIVE_SPACES):
        n_act  = len(mo_idx)
        n2     = n_act ** 2
        label  = str(mo_idx)
        key    = f"act_{'_'.join(map(str, mo_idx))}"
        print(f"\n{'='*60}")
        print(f"Active space {mo_idx}  (n_act={n_act})")

        # --- Π_act(iΩ) in MO-pair space ---
        Pi0_tau = gwtool.eval_Pi0_MO_active(ITER, mo_idx,
                                             input_h5=INPUT_H5, sim_h5=SIM_H5)
        ntau   = Pi0_tau.shape[0]
        ns_loc = Pi0_tau.shape[1]
        nk_loc = Pi0_tau.shape[2]

        # Complete the second half of the tau axis by symmetry
        for t in range(ntau // 2):
            Pi0_tau[ntau - t - 1] = Pi0_tau[t]

        Pi0_tau_flat = Pi0_tau.reshape(ntau, ns_loc * nk_loc,
                                       n_act, n_act, n_act, n_act)
        Pi0_iw_k     = ir.tauf_to_wb(Pi0_tau_flat).reshape(
                            niw, ns_loc, nk_loc, n_act, n_act, n_act, n_act)

        # Spin-summed Π_act: multiply by ns to match spin factor in P0_tilde_Q
        Pi_iw = (Pi0_iw_k[:, 0, 0] * ns).reshape(niw, n2, n2)   # (niw, n2, n2)

        # --- W_act(iΩ) ---
        W_act_k = eval_W_MO_active(VQ_mo, tildeP_iw, mo_idx)   # (niw, 1, n_act^4)
        W_iw    = W_act_k[:, 0].reshape(niw, n2, n2)            # (niw, n2, n2)

        # --- P_ph(iΩ) in Q-space ---
        VQ_act  = VQ_mo[0][:, mo_idx, :][:, :, mo_idx]   # (NQ, n_act, n_act)
        VQ_flat = VQ_act.reshape(NQ, n2)                  # (NQ, n2)

        # eval_Pph expects (nk, NQ, n2), (niw, ns, nk, n2, n2), (niw, n2, n2)
        P_ph_iw_full = eval_Pph(
            VQ_flat[np.newaxis],                    # (1, NQ, n2)
            Pi_iw[:, np.newaxis, np.newaxis],       # (niw, 1, 1, n2, n2)
            W_iw,                                   # (niw, n2, n2)
        )                                           # → (niw, 1, 1, NQ, NQ)
        P_ph_iw = P_ph_iw_full[:, 0, 0]            # (niw, NQ, NQ)

        # --- P_BSE(iΩ) ---
        P_bse_iw = eval_PBSE(P_ph_iw)              # (niw, NQ, NQ)

        # --- P_ph^(0)(iΩ) = V Π V† ---
        P_ph0_iw = np.einsum('Qm,wmn,Pn->wQP',
                              VQ_flat.conj(), Pi_iw, VQ_flat,
                              optimize=True)        # (niw, NQ, NQ)

        # --- Diagnostics at iΩ=0 ---
        iw0 = int(np.argmin(np.abs(omega_n)))
        print(f"  iΩ=0 index={iw0}  (Ω={omega_n[iw0]:.4f} Ha)")
        for name, mat in [("P^(0)_AO",  P0_iw[iw0, 0, 0]),
                          ("P̃",         tildeP_raw[iw0, 0, 0]),
                          ("Π_act",     Pi_iw[iw0]),
                          ("W_act",     W_iw[iw0]),
                          ("P_ph^(0)",  P_ph0_iw[iw0]),
                          ("P_ph",      P_ph_iw[iw0]),
                          ("P_BSE",     P_bse_iw[iw0])]:
            print(f"    {name:12s}  ‖·‖_F = {np.linalg.norm(mat):.4e}")

        # --- Save to HDF5 ---
        grp = fout.create_group(key)
        grp.attrs["mo_idx"] = mo_idx
        grp["Pi_iw"]        = Pi_iw         # (niw, n2, n2)
        grp["W_iw"]         = W_iw          # (niw, n2, n2)
        grp["P_ph0_iw"]     = P_ph0_iw      # (niw, NQ, NQ)
        grp["P_ph_iw"]      = P_ph_iw       # (niw, NQ, NQ)
        grp["P_bse_iw"]     = P_bse_iw      # (niw, NQ, NQ)

        # --- Frequency plot ---
        ax.semilogy(omega_n, frob_iw(P0_iw[:, 0, 0]),   label="P⁰_AO",    lw=1.5)
        ax.semilogy(omega_n, frob_iw(tildeP_raw[:,0,0]), label="P̃",        lw=1.5, ls="--")
        ax.semilogy(omega_n, frob_iw(Pi_iw),             label="Π_act",    lw=1.5)
        ax.semilogy(omega_n, frob_iw(W_iw),              label="W_act",    lw=1.5)
        ax.semilogy(omega_n, frob_iw(P_ph0_iw),          label="P_ph⁰",   lw=1.5, ls=":")
        ax.semilogy(omega_n, frob_iw(P_ph_iw),           label="P_ph",     lw=1.5)
        ax.semilogy(omega_n, frob_iw(P_bse_iw),          label="P_BSE",    lw=1.5, ls="-.")
        ax.set_title(f"Active space {label}", fontsize=10)
        ax.set_ylabel("‖·‖_F", fontsize=9)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(True, which="both", ls=":", alpha=0.4)

axes[-1].set_xlabel("iΩ_n  (Ha)", fontsize=10)
fig.suptitle(f"Intermediate matrices vs frequency  (β={BETA})", fontsize=12)
fig.tight_layout()
outpdf = "intermediates_freq.pdf"
fig.savefig(outpdf, bbox_inches="tight")
print(f"\nSaved plot  → {outpdf}")
print(f"Saved data  → {OUT_H5}")
plt.show()
