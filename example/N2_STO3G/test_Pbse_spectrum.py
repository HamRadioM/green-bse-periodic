#!/usr/bin/env python3
"""
BSE spectrum via plasmon-pole fitting on N2/STO-3G.

Pipeline:
  1. Full active space (all 10 MOs)
  2. P^ph and P^BSE at all bosonic Matsubara frequencies
  3. Take the diagonal P^BSE[Q,Q](iΩ) — each Q-channel is a bosonic response
  4. Fit each diagonal element with a single plasmon-pole model (positive Ω half)
  5. Plot:
       (a) diagonal curves + model fits on the imaginary-frequency axis
       (b) broadened absorption spectrum on the real-frequency axis
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
from plasPole import fit_plasmon_pole, plasmon_model

AU2EV = 27.211386245981

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
INPUT_H5   = "mean_field_input.h5"
SIM_H5     = "scGW_sim.h5"
INT_PATH   = "df_hf_int/"
IR_FILE    = "irgrid/1e5.h5"
BETA       = 1000.0
ITER       = -1
ACTIVE_MOS = list(range(2,10))     # all 10 MOs — full active space

ETA        = 0.001   # Lorentzian broadening (a.u.)
OMEGA_MAX  = 1.5    # real-frequency plot range (a.u.)
N_OMEGA    = 2000   # real-frequency grid points

# ---------------------------------------------------------------------------
# Load common data
# ---------------------------------------------------------------------------
print("Reading system data...")
with h5py.File(INPUT_H5, "r") as f:
    nao      = int(f["/params/nao"][()])
    mo_coeff = f["/HF/mo_coeff"][()]
    mo_energy = f["/HF/mo_energy"][()]
    S_raw    = f["/HF/S-k"][()].view(complex)
    S_raw    = S_raw.reshape(S_raw.shape[:-1])   # (..., nao, nao)
S = S_raw.reshape(-1, nao, nao)[0]              # (nao, nao) — first k/spin block

with h5py.File(INT_PATH + "VQ_0.h5", "r") as f:
    V_raw = f["/0"][()]

NQ = V_raw.shape[1]
VQ_ao = np.zeros((1, NQ, nao, nao), dtype=np.complex128)
for idx in range(nao):
    VQ_ao[0, :, :, idx] = V_raw[0, :, :, idx*2] + 1j * V_raw[0, :, :, idx*2+1]
VQ_mo = casida.VQ_ao2mo(VQ_ao, mo_coeff, S=S)   # (1, NQ, nmo, nmo)

occ  = int(round(sum(1 for e in mo_energy if e < 0)))   # occupied count from neg energies
print(f"  nao={nao}, NQ={NQ},  HOMO index={occ-1},  LUMO index={occ}")
print(f"  Active space: MOs {ACTIVE_MOS}  (n_act={len(ACTIVE_MOS)})")

# ---------------------------------------------------------------------------
# Build tildeP_iw (full Q-space Dyson)
# ---------------------------------------------------------------------------
print("\nBuilding tildeP_iw...")
P0_tilde = gwtool.eval_P0_tilde_Q(ITER, nao, NQ, int_path=INT_PATH, sim_h5=SIM_H5)
P0_tilde = gwtool.symmetrize_P0(P0_tilde)

ir    = IR_factory(BETA, IR_FILE)
P0_iw = ir.tauf_to_wb(P0_tilde)
niw   = P0_iw.shape[0]

I_NQ = np.eye(NQ, dtype=np.complex128)
tildeP_raw = np.zeros_like(P0_iw)
for iw in range(niw):
    p0 = P0_iw[iw, 0, 0]
    tildeP_raw[iw, 0, 0] = spinv(I_NQ - p0) @ p0
tildeP_raw = 0.5 * (tildeP_raw + tildeP_raw.conj().transpose(0, 1, 2, 4, 3))
tildeP_iw  = tildeP_raw[:, :, 0, :, :][:, :, :, :, np.newaxis]

# ---------------------------------------------------------------------------
# Bosonic Matsubara frequency grid
# ---------------------------------------------------------------------------
_, wsample_bose, *_ = ir.read_IR_matrices(IR_FILE, BETA, ptype='bose')
omega_n = wsample_bose.real    # shape (niw,), symmetric around 0

# Use only the positive-frequency half for fitting (response is even)
pos_mask = omega_n > 0
omega_pos = omega_n[pos_mask]          # (n_pos,)
print(f"  niw={niw},  positive-freq points: {pos_mask.sum()}")

# ---------------------------------------------------------------------------
# Pi0 in active MO space → FT
# ---------------------------------------------------------------------------
mo_idx = ACTIVE_MOS
n_act  = len(mo_idx)
n2     = n_act ** 2
print(f"\nComputing Pi0 (active MOs {mo_idx})...")

Pi0_tau = gwtool.eval_Pi0_MO_active(ITER, mo_idx, input_h5=INPUT_H5, sim_h5=SIM_H5)
ntau   = Pi0_tau.shape[0]
ns_loc = Pi0_tau.shape[1]
nk_loc = Pi0_tau.shape[2]

for t in range(ntau // 2):
    Pi0_tau[ntau - t - 1] = Pi0_tau[t]

Pi0_tau_flat = Pi0_tau.reshape(ntau, ns_loc * nk_loc, n_act, n_act, n_act, n_act)
Pi0_iw_flat  = ir.tauf_to_wb(Pi0_tau_flat)
Pi0_iw_k     = Pi0_iw_flat.reshape(niw, ns_loc, nk_loc, n_act, n_act, n_act, n_act)

# ---------------------------------------------------------------------------
# W_MO_active and P^BSE
# ---------------------------------------------------------------------------
print("Computing W_MO_active and P^BSE...")
W_act_k  = eval_W_MO_active(VQ_mo, tildeP_iw, mo_idx)
W_mat_k  = W_act_k.reshape(niw, nk_loc, n2, n2)

VQ_act_k = VQ_mo[:, :, mo_idx, :][:, :, :, mo_idx]   # (nk, NQ, n_act, n_act)

P_ph_k   = eval_Pph(VQ_act_k, Pi0_iw_k, W_mat_k[:, 0])  # (niw, ns, nk, NQ, NQ)
P_ph     = P_ph_k.sum(axis=(1, 2)) / nk_loc              # (niw, NQ, NQ)
P_bse    = eval_PBSE(P_ph)                                # (niw, NQ, NQ)
print(f"  P_bse shape: {P_bse.shape}")

# ---------------------------------------------------------------------------
# Take diagonal of P^BSE: P_BSE[iw, Q, Q]  →  shape (niw, NQ)
# Use the positive-frequency half for fitting
# ---------------------------------------------------------------------------
diag_bse_all = np.array([np.diag(P_bse[iw].real) for iw in range(niw)])  # (niw, NQ)
diag_bse_pos = diag_bse_all[pos_mask]                                      # (n_pos, NQ)

iw_mid = np.argmin(np.abs(omega_n))   # index closest to Ω=0

# Sort Q-channels by |diagonal at Ω≈0| for display
sort_idx = np.argsort(np.abs(diag_bse_all[iw_mid]))[::-1]

# ---------------------------------------------------------------------------
# Plasmon-pole fit for each Q-channel diagonal
# ---------------------------------------------------------------------------
print("Fitting plasmon-pole model to each P_BSE diagonal element...")
wp_arr   = np.full(NQ, np.nan)
S_arr    = np.zeros(NQ)
Finf_arr = np.zeros(NQ)
res_arr  = np.full(NQ, np.nan)

for Q in range(NQ):
    curve = diag_bse_pos[:, Q]
    Finf  = float(curve[-1])
    F0    = float(curve[0])
    if abs(F0 - Finf) < 1e-12:
        continue
    try:
        fit = fit_plasmon_pole(omega_pos, curve, F0=F0, Finf=Finf)
        wp_arr[Q]   = fit['wp']
        S_arr[Q]    = fit['S']
        Finf_arr[Q] = fit['Finf']
        res_arr[Q]  = fit['residual_norm']
    except Exception as e:
        print(f"  Q={Q} fit failed: {e}")

top5 = np.argsort(np.abs(S_arr))[::-1][:5]
print(f"\nTop-5 Q-channels by |S|:")
for Q in top5:
    print(f"  Q={Q:3d}:  wp = {wp_arr[Q]:.4f} a.u. = {wp_arr[Q]*AU2EV:.3f} eV"
          f"   S = {S_arr[Q]:.4e}   res = {res_arr[Q]:.2e}")

# Q-channels with the smallest (lowest-energy) pole positions
valid_mask = ~np.isnan(wp_arr) & (np.abs(S_arr) > 1e-10)
low_wp_idx = np.argsort(np.where(valid_mask, wp_arr, np.inf))[:8]
print(f"\nQ-channels with lowest wp:")
for Q in low_wp_idx:
    print(f"  Q={Q:3d}:  wp = {wp_arr[Q]:.4f} a.u. = {wp_arr[Q]*AU2EV:.3f} eV"
          f"   S = {S_arr[Q]:.4e}   res = {res_arr[Q]:.2e}")

# ---------------------------------------------------------------------------
# Plot A: diagonal P_BSE(iΩ) + plasmon-pole fits — lowest-wp Q-channels
# ---------------------------------------------------------------------------
n_show = min(8, NQ)
z_iw   = 1j * omega_pos
cmap   = plt.get_cmap("tab10")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for i, Q in enumerate(low_wp_idx):
    col = cmap(i % 10)
    ax.plot(omega_pos, diag_bse_pos[:, Q], color=col,
            lw=1.5, label=f"Q={Q}  wp={wp_arr[Q]*AU2EV:.2f} eV")
    if not np.isnan(wp_arr[Q]):
        fit_curve = plasmon_model(z_iw, Finf_arr[Q], S_arr[Q], wp_arr[Q]).real
        ax.plot(omega_pos, fit_curve, color=col, lw=1.0, ls="--", alpha=0.7)

ax.set_xlabel(r"$\Omega_n$  (a.u.)", fontsize=11)
ax.set_ylabel(r"$P^\mathrm{BSE}_{QQ}(i\Omega_n)$", fontsize=11)
ax.set_title(f"P^BSE diagonal — all {n_act} MOs  (solid: data, dashed: fit)",
             fontsize=10)
ax.set_xscale("log")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.legend(fontsize=7, ncol=2)

# ---------------------------------------------------------------------------
# Plot B: absorption spectrum
#   A(ω) = Σ_Q  S_Q · η / π / [(ω - wp_Q)² + η²]
# ---------------------------------------------------------------------------
ax = axes[1]
omega_real = np.linspace(0.0, OMEGA_MAX, N_OMEGA)
spectrum   = np.zeros(N_OMEGA)

for Q in range(NQ):
    if np.isnan(wp_arr[Q]) or wp_arr[Q] < 1e-6 or abs(S_arr[Q]) < 1e-10:
        continue
    spectrum += S_arr[Q] * ETA / ((omega_real - wp_arr[Q])**2 + ETA**2) / np.pi

ax.plot(omega_real, spectrum, color="steelblue", lw=1.5)

for i, Q in enumerate(low_wp_idx):
    if np.isnan(wp_arr[Q]):
        continue
    ax.axvline(wp_arr[Q], color=cmap(i % 10), lw=1.0, ls="--", alpha=0.8,
               label=f"Q={Q}: {wp_arr[Q]:.3f} au = {wp_arr[Q]*AU2EV:.2f} eV")

ax.set_xlabel(r"$\omega$  (a.u.)", fontsize=11)
ax.set_ylabel(r"$A(\omega) = \sum_Q S_Q\,L(\omega{-}\omega_p^Q)$", fontsize=11)
ax.set_title(f"BSE absorption spectrum  (η={ETA:.3f} a.u., active: all {n_act} MOs)",
             fontsize=10)
ax.legend(fontsize=7)
ax.set_xlim(0, OMEGA_MAX)
ax.set_ylim(bottom=0)

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim()[0] * AU2EV, ax.get_xlim()[1] * AU2EV)
ax2.set_xlabel("eV", fontsize=10)

plt.tight_layout()
outfile = "Pbse_spectrum.pdf"
plt.savefig(outfile, bbox_inches="tight")
print(f"\nSaved → {outfile}")
plt.show()
