#!/usr/bin/env python3
"""
Compute P^BSE(iΩ) for all bosonic Matsubara frequencies on N2/STO-3G and
plot the diagonal elements as a function of iΩ.

Pipeline (active space [5, 6, 7, 8]):
  1. Build tildeP_iw  (Q-space Dyson, full system)
  2. Pi0_MO_active    (tau → freq, active MO space, per k-point)
  3. W_MO_active      (per k-point)
  4. P^ph             (per-k, then k-summed)
  5. P^BSE            (Dyson in Q-space)
  6. Plot diagonal of P^BSE vs iΩ
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
INPUT_H5     = "mean_field_input.h5"
SIM_H5       = "scGW_sim.h5"
INT_PATH     = "df_hf_int/"
IR_FILE      = "irgrid/1e5.h5"
BETA         = 1000.0
ITER         = -1
ACTIVE_MOS   = [5, 6, 7, 8]   # HOMO-1, HOMO, LUMO, LUMO+1

# ---------------------------------------------------------------------------
# Load system data
# ---------------------------------------------------------------------------
print("Reading system data...")
with h5py.File(INPUT_H5, "r") as f:
    nao      = int(f["/params/nao"][()])
    mo_coeff = f["/HF/mo_coeff"][()]
    S_raw    = f["/HF/S-k"][()].view(complex)
    S_raw    = S_raw.reshape(S_raw.shape[:-1])   # (..., nao, nao)
S = S_raw.reshape(-1, nao, nao)[0]              # (nao, nao) — first k/spin block

with h5py.File(INT_PATH + "VQ_0.h5", "r") as f:
    V_raw = f["/0"][()]

NQ = V_raw.shape[1]
VQ_ao = np.zeros((1, NQ, nao, nao), dtype=np.complex128)
for idx in range(nao):
    VQ_ao[0, :, :, idx] = V_raw[0, :, :, idx*2] + 1j * V_raw[0, :, :, idx*2+1]
VQ_mo = casida.VQ_ao2mo(VQ_ao, mo_coeff, S=S)   # (1, NQ, nmo, nmo)  [nk=1 molecular]

# ---------------------------------------------------------------------------
# Build tildeP_iw (full Q-space Dyson)
# ---------------------------------------------------------------------------
print("Building tildeP_iw...")
P0_tilde = gwtool.eval_P0_tilde_Q(ITER, nao, NQ, int_path=INT_PATH, sim_h5=SIM_H5)
P0_tilde = gwtool.symmetrize_P0(P0_tilde)

ir     = IR_factory(BETA, IR_FILE)
P0_iw  = ir.tauf_to_wb(P0_tilde)           # (niw, 1, 1, NQ, NQ)
niw    = P0_iw.shape[0]

I_NQ = np.eye(NQ, dtype=np.complex128)
tildeP_raw = np.zeros_like(P0_iw)
for iw in range(niw):
    p0 = P0_iw[iw, 0, 0]
    tildeP_raw[iw, 0, 0] = spinv(I_NQ - p0) @ p0
tildeP_raw = 0.5 * (tildeP_raw + tildeP_raw.conj().transpose(0, 1, 2, 4, 3))
tildeP_iw  = tildeP_raw[:, :, 0, :, :][:, :, :, :, np.newaxis]   # (niw,1,NQ,NQ,1)

# ---------------------------------------------------------------------------
# Bosonic Matsubara frequencies from the IR grid
# ---------------------------------------------------------------------------
_, wsample_bose, *_ = ir.read_IR_matrices(IR_FILE, BETA, ptype='bose')
omega_n = wsample_bose.real    # iΩ_n values (real axis of imaginary frequencies)
assert len(omega_n) == niw, f"Frequency grid mismatch: {len(omega_n)} vs {niw}"
print(f"  niw={niw},  Ω range: [{omega_n.min():.2f}, {omega_n.max():.2f}]  (a.u.)")

# ---------------------------------------------------------------------------
# Pi0 in active MO space: (ntau, ns, nk, ...) → (niw, ns, nk, n_act^4)
# ---------------------------------------------------------------------------
mo_idx = ACTIVE_MOS
n_act  = len(mo_idx)
n2     = n_act ** 2
print(f"\nActive space {mo_idx}  (n_act={n_act})")

Pi0_tau = gwtool.eval_Pi0_MO_active(ITER, mo_idx, input_h5=INPUT_H5, sim_h5=SIM_H5)
ntau   = Pi0_tau.shape[0]
ns_loc = Pi0_tau.shape[1]
nk_loc = Pi0_tau.shape[2]

# symmetrise second half
for t in range(ntau // 2):
    Pi0_tau[ntau - t - 1] = Pi0_tau[t]

# FT: collapse (ns, nk) into a batch dimension, transform, then restore
Pi0_tau_flat = Pi0_tau.reshape(ntau, ns_loc * nk_loc, n_act, n_act, n_act, n_act)
Pi0_iw_flat  = ir.tauf_to_wb(Pi0_tau_flat)    # (niw, ns*nk, n_act, n_act, n_act, n_act)
Pi0_iw_k     = Pi0_iw_flat.reshape(niw, ns_loc, nk_loc, n_act, n_act, n_act, n_act)

# ---------------------------------------------------------------------------
# W in active MO space (niw, nk, n_act^4) and P^ph / P^BSE
# ---------------------------------------------------------------------------
W_act_k  = eval_W_MO_active(VQ_mo, tildeP_iw, mo_idx)     # (niw, nk, n_act^4)
W_mat_k  = W_act_k.reshape(niw, nk_loc, n2, n2)

VQ_act_k = VQ_mo[:, :, mo_idx, :][:, :, :, mo_idx]        # (nk, NQ, n_act, n_act)

# P^ph: (niw, ns, nk, NQ, NQ) — unaveraged
P_ph_k   = eval_Pph(VQ_act_k, Pi0_iw_k, W_mat_k[:, 0])

# k-sum + spin-sum → (niw, NQ, NQ)
P_ph     = P_ph_k.sum(axis=(1, 2)) / nk_loc

# P^BSE via Dyson: (niw, NQ, NQ)
P_bse    = eval_PBSE(P_ph)
print(f"  P_bse shape: {P_bse.shape}")

# ---------------------------------------------------------------------------
# Plot: diagonal of P^BSE vs iΩ
# ---------------------------------------------------------------------------
diag_bse = np.array([np.diag(P_bse[iw].real) for iw in range(niw)])   # (niw, NQ)
diag_ph  = np.array([np.diag(P_ph[iw].real)  for iw in range(niw)])   # (niw, NQ)

# Select a representative subset of diagonal indices to plot
n_show = min(8, NQ)
stride = max(1, NQ // n_show)
idx_show = list(range(0, NQ, stride))[:n_show]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for ax, diag, label in zip(axes, [diag_ph, diag_bse], ["P^ph", "P^BSE"]):
    for qi in idx_show:
        ax.plot(omega_n, diag[:, qi], label=f"Q={qi}", lw=1.2)
    ax.set_xlabel(r"$i\Omega_n$  (a.u.)", fontsize=11)
    ax.set_ylabel(r"Re $P_{QQ}(i\Omega_n)$", fontsize=11)
    ax.set_title(f"{label}  diagonal — active space {mo_idx}", fontsize=10)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xscale("symlog", linthresh=1.0)

plt.tight_layout()
outfile = "Pbse_frequency.pdf"
plt.savefig(outfile, bbox_inches="tight")
print(f"\nSaved → {outfile}")
plt.show()

# ---------------------------------------------------------------------------
# Print summary: diagonal at Ω=0 (iw_mid)
# ---------------------------------------------------------------------------
iw_mid = niw // 2
print(f"\nDiagonal of P^ph  at Ω≈0 (iw={iw_mid}, Ω={omega_n[iw_mid]:.3f}):")
print("  " + "  ".join(f"Q={qi}: {diag_ph[iw_mid, qi]:.4e}" for qi in idx_show))
print(f"\nDiagonal of P^BSE at Ω≈0 (iw={iw_mid}, Ω={omega_n[iw_mid]:.3f}):")
print("  " + "  ".join(f"Q={qi}: {diag_bse[iw_mid, qi]:.4e}" for qi in idx_show))
