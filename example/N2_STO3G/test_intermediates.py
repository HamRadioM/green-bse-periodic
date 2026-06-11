#!/usr/bin/env python3
"""
Careful comparison of intermediate matrices at iΩ=0 for several active spaces.

Four panels per active space:
  1. P^(0) from AO  — full bare polarizability in Q-space from eval_P0_tilde_Q
  2. P_ph^(0)       — V_flat Π_act V_flat†  (active bubble back-projected to Q)
  3. W_act          — screened Coulomb in active MO-pair space
  4. Π_act          — independent-particle polarizability in active MO-pair space

P^(0) from AO and P_ph^(0) should agree for full active space (up to spin factor).
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
from polarization import eval_W_MO_active

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
INPUT_H5   = "mean_field_input.h5"
SIM_H5     = "scGW_sim.h5"
INT_PATH   = "df_hf_int/"
IR_FILE    = "irgrid/1e5.h5"
BETA       = 1000.0
ITER       = -1

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

with h5py.File(INT_PATH + "VQ_0.h5", "r") as f:
    V_raw = f["/0"][()]

NQ = V_raw.shape[1]
VQ_ao = np.zeros((1, NQ, nao, nao), dtype=np.complex128)
for idx in range(nao):
    VQ_ao[0, :, :, idx] = V_raw[0, :, :, idx*2] + 1j * V_raw[0, :, :, idx*2+1]
VQ_mo = casida.VQ_ao2mo(VQ_ao, mo_coeff, S=S)   # (1, NQ, nmo, nmo)

occ = int(round(sum(1 for e in mo_energy if e < 0)))
print(f"  nao={nao}, NQ={NQ}, HOMO={occ-1}, LUMO={occ}")

# ---------------------------------------------------------------------------
# P^(0) fully from AO  —  eval_P0_tilde_Q  (includes spin factor 2)
# ---------------------------------------------------------------------------
print("\nBuilding P^(0) from AO...")
P0_tilde = gwtool.eval_P0_tilde_Q(ITER, nao, NQ, int_path=INT_PATH, sim_h5=SIM_H5)
P0_tilde = gwtool.symmetrize_P0(P0_tilde)

ir    = IR_factory(BETA, IR_FILE)
P0_iw = ir.tauf_to_wb(P0_tilde)       # (niw, 1, 1, NQ, NQ)
niw   = P0_iw.shape[0]
ns    = P0_iw.shape[1]                 # number of spin channels in G_tau

I_NQ = np.eye(NQ, dtype=np.complex128)
tildeP_raw = np.zeros_like(P0_iw)
for iw in range(niw):
    p0 = P0_iw[iw, 0, 0]
    tildeP_raw[iw, 0, 0] = spinv(I_NQ - p0) @ p0
tildeP_raw = 0.5 * (tildeP_raw + tildeP_raw.conj().transpose(0, 1, 2, 4, 3))
tildeP_iw  = tildeP_raw[:, :, 0, :, :][:, :, :, :, np.newaxis]   # (niw,1,NQ,NQ,1)

_, wsample_bose, *_ = ir.read_IR_matrices(IR_FILE, BETA, ptype='bose')
omega_n = wsample_bose.real
iw0     = np.argmin(np.abs(omega_n))
print(f"  niw={niw}, ns={ns}, iΩ=0 index={iw0} (Ω={omega_n[iw0]:.4f})")

# P^(0) from AO at iΩ=0  — shape (NQ, NQ), real part
P0_AO = P0_iw[iw0, 0, 0].real

# ---------------------------------------------------------------------------
# Loop over active spaces: compute Π_act, W_act, P_ph^(0) at iΩ=0
# ---------------------------------------------------------------------------
n_spaces = len(ACTIVE_SPACES)
ROWS = ["P$^{(0)}$ from AO", "P$_{\\rm ph}^{(0)}$", "P$_{\\rm ph}$", "W$_{\\rm act}$", "$\\Pi_{\\rm act}$"]
n_rows = len(ROWS)

fig, axes = plt.subplots(n_rows, n_spaces, figsize=(2.5*n_spaces, 2*n_rows))

for col, mo_idx in enumerate(ACTIVE_SPACES):
    n_act = len(mo_idx)
    n2    = n_act ** 2
    label = str(mo_idx)
    print(f"\n{'='*60}")
    print(f"Active space {mo_idx}  (n_act={n_act})")

    # --- Π_act ---
    Pi0_tau  = gwtool.eval_Pi0_MO_active(ITER, mo_idx, input_h5=INPUT_H5, sim_h5=SIM_H5)
    ntau     = Pi0_tau.shape[0]
    ns_loc   = Pi0_tau.shape[1]
    nk_loc   = Pi0_tau.shape[2]
    for t in range(ntau // 2):
        Pi0_tau[ntau - t - 1] = Pi0_tau[t].transpose(0, 1, 3, 2, 5, 4)

    Pi0_tau_flat = Pi0_tau.reshape(ntau, ns_loc * nk_loc, n_act, n_act, n_act, n_act)
    Pi0_iw_flat  = ir.tauf_to_wb(Pi0_tau_flat)
    Pi0_iw_k     = Pi0_iw_flat.reshape(niw, ns_loc, nk_loc, n_act, n_act, n_act, n_act)

    # Π_act at iΩ=0: explicit (spin=0, k=0) — shape (n_act, n_act, n_act, n_act)
    # Multiply by ns for spin degeneracy, matching the factor-2 in eval_P0_tilde_Q
    Pi_iw0_4d = Pi0_iw_k[iw0, 0, 0].real * ns              # (n_act, n_act, n_act, n_act)

    # --- W_act at iΩ=0 ---
    W_act_k = eval_W_MO_active(VQ_mo, tildeP_iw, mo_idx)    # (niw, nk, n_act^4)
    W_iw0   = W_act_k[iw0, 0].reshape(n_act, n_act, n_act, n_act).real

    # --- P_ph^(0) via einsum: no reshape, indices are explicit ---
    #   P[Q1,Q2] = Σ_{ijkl} VQ_mo*[Q1,i,j] Π[i,j,k,l] VQ_mo[Q2,k,l]
    #
    # Extract k=0 VQ in active space without mixing scalar and fancy indexing
    # in the same []: first apply scalar k-index, then fancy MO-index.
    VQ_act   = VQ_mo[0][:, mo_idx, :][:, :, mo_idx]          # (NQ, n_act, n_act)
    VQ_flat  = VQ_act.reshape(NQ, n2)                         # (NQ, n_act²)

    P_ph0  = np.einsum('Qij,ijkl,Pkl->QP',
                        VQ_act.conj(), Pi_iw0_4d, VQ_act,
                        optimize=True).real                   # (NQ, NQ)

    # --- P_ph = V (I + ΠW)^{-1} Π V†  at iΩ=0, in Q-space ---
    Pi_mat  = Pi_iw0_4d.reshape(n2, n2)
    W_mat   = W_iw0.reshape(n2, n2)
    I_n2    = np.eye(n2)
    A       = I_n2 + Pi_mat @ W_mat                          # (n2, n2)
    x       = np.linalg.solve(A, Pi_mat)                     # (I+ΠW)^{-1} Π
    P_ph    = (VQ_flat @ x @ VQ_flat.conj().T).real          # (NQ, NQ)

    # For display of W and Π as 2D matrices
    Pi_iw0 = Pi_mat
    W_iw0  = W_mat

    # --- Print diagnostics ---
    print(f"  P^(0) AO  diag: min={np.diag(P0_AO).min():.4e}  max={np.diag(P0_AO).max():.4e}")
    print(f"  P_ph^(0)  diag: min={np.diag(P_ph0).min():.4e}  max={np.diag(P_ph0).max():.4e}")
    print(f"  P_ph      diag: min={np.diag(P_ph).min():.4e}  max={np.diag(P_ph).max():.4e}")
    if n_act == nao:
        ratio = np.diag(P_ph0) / (np.diag(P0_AO) + 1e-30)
        print(f"  P_ph^(0)/P^(0)_AO diag (full space, should be ≈1): {ratio[:8]}")
    print(f"  W_act     diag: min={np.diag(W_iw0).min():.4e}  max={np.diag(W_iw0).max():.4e}")
    print(f"  Π_act     diag: min={np.diag(Pi_iw0).min():.4e}  max={np.diag(Pi_iw0).max():.4e}")

    # --- Heatmaps ---
    mats  = [P0_AO, P_ph0, P_ph,  W_iw0, Pi_iw0]
    sizes = [NQ,    NQ,    NQ,    n2,    n2   ]

    for row, (mat, title_base, sz) in enumerate(zip(mats, ROWS, sizes)):
        ax   = axes[row, col]
        vmax = np.abs(mat).max()
        if vmax == 0:
            vmax = 1.0
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{title_base}\n{label}  ({sz}×{sz})", fontsize=8)
        ax.set_xlabel("col index", fontsize=7)
        ax.set_ylabel("row index", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Label MO-pair axes for small active spaces
        if sz == n2 and n_act <= 4:
            pair_labels = [f"({mo_idx[p]},{mo_idx[q]})"
                           for p in range(n_act) for q in range(n_act)]
            ax.set_xticks(range(n2))
            ax.set_xticklabels(pair_labels, fontsize=5, rotation=90)
            ax.set_yticks(range(n2))
            ax.set_yticklabels(pair_labels, fontsize=5)

fig.suptitle("Intermediate matrices at iΩ=0\n"
             "Rows: P⁰_AO | P_ph⁰ | P_ph | W_act | Π_act", fontsize=11)
fig.tight_layout()
outfile = "intermediates_comparison.pdf"
fig.savefig(outfile, bbox_inches="tight")
print(f"\nSaved → {outfile}")
plt.show()
