#!/usr/bin/env python3
"""
BSE spectrum via plasmon-pole fitting on N2/STO-3G.

Pipeline:
  1. Full active space (all 10 MOs)
  2. P^ph and P^BSE at all bosonic Matsubara frequencies
  3. Diagonalise P^BSE at each iΩ → eigenvalue series per mode
  4. Fit each mode with a single plasmon-pole model (positive Ω half only)
  5. Plot:
       (a) eigenvalue curves + model fits on the imaginary-frequency axis
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
ACTIVE_MOS = list(range(10))     # all 10 MOs — full active space

ETA        = 0.01   # Lorentzian broadening (a.u.)
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

with h5py.File(INT_PATH + "VQ_0.h5", "r") as f:
    V_raw = f["/0"][()]

NQ = V_raw.shape[1]
VQ_ao = np.zeros((1, NQ, nao, nao), dtype=np.complex128)
for idx in range(nao):
    VQ_ao[0, :, :, idx] = V_raw[0, :, :, idx*2] + 1j * V_raw[0, :, :, idx*2+1]
VQ_mo = casida.VQ_ao2mo(VQ_ao, mo_coeff)   # (1, NQ, nmo, nmo)

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
# Diagonalise P^BSE at each frequency → eigenvalue series (niw, NQ)
# ---------------------------------------------------------------------------
print("Diagonalising P^BSE at each frequency...")
eigs_all = np.array([np.linalg.eigvalsh(P_bse[iw].real) for iw in range(niw)])
# eigvalsh sorts ascending; at Ω≈0 the most-negative eigenvalue = dominant mode
# Shape: (niw, NQ)

eigs_pos = eigs_all[pos_mask]    # (n_pos, NQ)
n_pos    = eigs_pos.shape[0]

# Sort modes by |eigenvalue| at Ω=0 (iw_mid) — largest first
iw_mid  = np.argmin(np.abs(omega_n))   # index closest to Ω=0
sort_idx = np.argsort(np.abs(eigs_all[iw_mid]))[::-1]
eigs_all_sorted = eigs_all[:, sort_idx]
eigs_pos_sorted = eigs_pos[:, sort_idx[:NQ]]

# ---------------------------------------------------------------------------
# Plasmon-pole fit for each mode
# ---------------------------------------------------------------------------
print("Fitting plasmon-pole models...")
wp_list    = []
S_list     = []
Finf_list  = []
res_list   = []

for mode in range(NQ):
    curve = eigs_pos_sorted[:, mode]
    Finf  = float(curve[-1])   # high-Ω limit
    F0    = float(curve[0])    # Ω≈0 value
    try:
        fit = fit_plasmon_pole(omega_pos, curve, F0=F0, Finf=Finf)
        wp_list.append(fit['wp'])
        S_list.append(fit['S'])
        Finf_list.append(fit['Finf'])
        res_list.append(fit['residual_norm'])
    except Exception as e:
        print(f"  Mode {mode} fit failed: {e}")
        wp_list.append(np.nan)
        S_list.append(0.0)
        Finf_list.append(Finf)
        res_list.append(np.nan)

wp_arr  = np.array(wp_list)
S_arr   = np.array(S_list)
Finf_arr = np.array(Finf_list)

print(f"\nTop-5 modes by |S|:")
top5 = np.argsort(np.abs(S_arr))[::-1][:5]
for i in top5:
    print(f"  mode {i:3d}:  wp = {wp_arr[i]:.4f} a.u. = {wp_arr[i]*AU2EV:.3f} eV"
          f"   S = {S_arr[i]:.4e}   res = {res_list[i]:.2e}")

# ---------------------------------------------------------------------------
# Plot A: eigenvalue curves + plasmon-pole fits (imaginary axis)
# ---------------------------------------------------------------------------
n_show = min(8, NQ)
z_iw   = 1j * omega_pos

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
cmap = plt.get_cmap("tab10")
for mi in range(n_show):
    col = cmap(mi % 10)
    ax.plot(omega_pos, eigs_pos_sorted[:, mi], color=col,
            lw=1.5, label=f"mode {mi}")
    if not np.isnan(wp_arr[mi]):
        fit_curve = plasmon_model(z_iw, Finf_arr[mi], S_arr[mi], wp_arr[mi]).real
        ax.plot(omega_pos, fit_curve, color=col, lw=1.0, ls="--", alpha=0.7)

ax.set_xlabel(r"$\Omega_n$  (a.u.)", fontsize=11)
ax.set_ylabel(r"Eigenvalue of $P^\mathrm{BSE}(i\Omega_n)$", fontsize=11)
ax.set_title(f"P^BSE eigenvalues — all {n_act} MOs  (solid: data, dashed: fit)",
             fontsize=10)
ax.set_xscale("log")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.legend(fontsize=7, ncol=2)

# ---------------------------------------------------------------------------
# Plot B: absorption spectrum — Lorentzian peaks at pole positions
# ---------------------------------------------------------------------------
ax = axes[1]
omega_real = np.linspace(0.0, OMEGA_MAX, N_OMEGA)
spectrum   = np.zeros(N_OMEGA)

for mi in range(NQ):
    if np.isnan(wp_arr[mi]) or wp_arr[mi] < 1e-6:
        continue
    # Spectral function from analytic continuation of plasmon pole:
    # -Im P(ω+iη) ∝ S * η / ((ω-wp)^2 + η^2)  (Lorentzian at ω=wp)
    # Only include modes with non-negligible weight
    if abs(S_arr[mi]) < 1e-8:
        continue
    spectrum += S_arr[mi] * ETA / ((omega_real - wp_arr[mi])**2 + ETA**2) / np.pi

ax.plot(omega_real, spectrum, color="steelblue", lw=1.5)
ax.plot(omega_real * AU2EV, spectrum, color="steelblue", lw=0.0)   # invisible; for twin x

# Mark individual pole positions (top modes by |S|)
for i in top5:
    if np.isnan(wp_arr[i]):
        continue
    ax.axvline(wp_arr[i], color=cmap(list(top5).index(i) % 10),
               lw=1.0, ls="--", alpha=0.7,
               label=f"wp={wp_arr[i]:.3f} au={wp_arr[i]*AU2EV:.2f} eV")

ax.set_xlabel(r"$\omega$  (a.u.)", fontsize=11)
ax.set_ylabel(r"$A(\omega)$  (arb. units)", fontsize=11)
ax.set_title(f"BSE absorption spectrum  (η={ETA:.3f} a.u., active: all {n_act} MOs)",
             fontsize=10)
ax.legend(fontsize=7)
ax.set_xlim(0, OMEGA_MAX)
ax.set_ylim(bottom=0)

# Secondary x-axis in eV
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim()[0] * AU2EV, ax.get_xlim()[1] * AU2EV)
ax2.set_xlabel("eV", fontsize=10)

plt.tight_layout()
outfile = "Pbse_spectrum.pdf"
plt.savefig(outfile, bbox_inches="tight")
print(f"\nSaved → {outfile}")
plt.show()
