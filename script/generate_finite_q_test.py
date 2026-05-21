#!/usr/bin/env python3
"""
generate_finite_q_test.py
=========================
Generate synthetic periodic-system input files for testing the finite-q BSE
solver.  Extends generate_periodic_test.py by adding:

  1. Off-diagonal GDF integrals  ⟨φ_μ(k)|V_Q|φ_ν(k+q)⟩  → VQ_q{idx}.h5
  2. Finite-q screened polarization P̃(q, iΩ)            → p_iw_tilde_q{idx}.h5

System: 1-D hydrogen chain  H2 ... H2 ... H2 (unit cell = H-H, STO-3G,
        k-mesh Nk x 1 x 1) — identical to generate_periodic_test.py.

Output directory layout::

  <outdir>/
    mean_field_input.h5          — k-RHF Fock / overlap / core matrices
    sim.h5                       — G(τ) and synthetic Σ(τ)
    irgrid/1e5.h5                — IR grid (copied)
    df_hf_int/                   — same-k integrals (q=0 BSE)
      VQ_0.h5
      meta.h5
    df_hf_int_fq/                — finite-q integral directory
      VQ_0.h5                    — same-k (copy of df_hf_int/VQ_0.h5)
      VQ_q{idx}.h5               — off-diagonal k → k+q integrals
      meta.h5                    — copy of df_hf_int/meta.h5
    p_iw_tilde.h5                — q=0  P̃(iΩ)
    p_iw_tilde_q{idx}.h5        — q≠0  P̃(q, iΩ)
    bse_periodic_singlet.sh      — q=0 run script
    bse_fq{idx}_singlet.sh       — finite-q run script

Usage
-----
  python script/generate_finite_q_test.py --nk 4 --q_idx 1 \\
      --outdir example/H2_periodic_4k

Then run the finite-q BSE solver:
  cd example/H2_periodic_4k
  bash bse_fq1_singlet.sh

Notes
-----
* --q_idx must be in [0, nk).  q_idx=0 reproduces the q=0 optical-limit result.
* For q_idx=0 the off-diagonal integrals equal the same-k integrals; no second
  GDF extraction is needed.
* P̃(q, iΩ) is computed using the proper off-diagonal transition density
  (k, k+q) via compute_tildeP_iw_fq — this differs from the q=0 formula
  used in generate_periodic_test.py.
"""

import argparse
import os
import sys
import shutil
import json

import h5py
import numpy as np

try:
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc import scf as pbcscf
    from pyscf.pbc import df  as pbcdf
except ImportError:
    raise SystemExit("PySCF not found.  Install with:  pip install pyscf")


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from generate_periodic_test.py so this script
# is self-contained and can be run independently)
# ---------------------------------------------------------------------------

def _to_real_pairs(arr):
    """complex128 → float64 with extra trailing dim of 2 (green-mbpt convention)."""
    out = np.zeros(arr.shape + (2,), dtype=np.float64)
    out[..., 0] = arr.real
    out[..., 1] = arr.imag
    return out


def _vq_to_storage(VQ_ao):
    """
    VQ_ao : (nk, NQ, nao, nao) complex128
    Returns (nk, NQ, nao, 2*nao) float64 — interleaved real/imag along last axis.
    This is what readVQ / readH5 with .view(complex) expects.
    """
    nk, NQ, nao, _ = VQ_ao.shape
    out = np.zeros((nk, NQ, nao, 2 * nao), dtype=np.float64)
    out[:, :, :, 0::2] = VQ_ao.real
    out[:, :, :, 1::2] = VQ_ao.imag
    return out


def _get_tau_grid(ir_file, beta=1000.0):
    """Return fermionic tau mesh from the IR file (nts = n_fermi_x + 2)."""
    with h5py.File(ir_file, 'r') as f:
        xgrid = f['fermi/xgrid'][()]

    tau_mesh = np.zeros(xgrid.shape[0] + 2)
    tau_mesh[0]    = 0.0
    tau_mesh[1:-1] = (xgrid + 1) * beta / 2.0
    tau_mesh[-1]   = beta

    ntau = tau_mesh.shape[0]
    return ntau, xgrid, tau_mesh


# ---------------------------------------------------------------------------
# Mean-field: k-point RHF
# ---------------------------------------------------------------------------

def run_krhf(nk, basis='sto-3g', bond_length=1.4, lattice_const=None, verbose=0):
    """
    Run k-point RHF on a 1-D H₂ chain.

    Returns
    -------
    cell, kmf, mydf : PySCF objects
    """
    if lattice_const is None:
        lattice_const = 2 * bond_length + 1.5

    cell = pbcgto.Cell()
    cell.atom   = f'H 0 0 0; H 0 0 {bond_length}'
    cell.basis  = basis
    cell.a      = np.diag([20.0, 20.0, lattice_const])
    cell.unit   = 'B'
    cell.verbose = verbose
    cell.build()

    kpts = cell.make_kpts([nk, 1, 1])

    mydf = pbcdf.GDF(cell, kpts)
    mydf.build()

    kmf = pbcscf.KRHF(cell, kpts)
    kmf.with_df = mydf
    kmf.init_guess = 'atom'
    kmf.kernel()

    return cell, kmf, mydf


# ---------------------------------------------------------------------------
# Extract VQ integrals
# ---------------------------------------------------------------------------

def extract_VQ(cell, mydf, kpts):
    """
    Extract same-k 3-centre DF integrals V_{Q,μν}^k for each k.

    Returns
    -------
    VQ_ao : (nk, NQ, nao, nao) complex128
    """
    nk  = len(kpts)
    nao = cell.nao_nr()
    NQ  = mydf.get_naoaux()
    VQ_ao = np.zeros((nk, NQ, nao, nao), dtype=np.complex128)
    for ik, kpt in enumerate(kpts):
        col = 0
        kpti_kptj = np.array([kpt, kpt])
        for Lpq in mydf.sr_loop(kpti_kptj=kpti_kptj, compact=False):
            Lpq  = Lpq[0] + Lpq[1] * 1j
            naux = Lpq.shape[0]
            VQ_ao[ik, col:col + naux, :, :] = Lpq.reshape(naux, nao, nao)
            col += naux
    return VQ_ao


def extract_VQ_kq(cell, mydf, kpts, q_idx):
    """
    Extract off-diagonal GDF integrals ⟨φ_μ(k)|V_Q|φ_ν(k+q)⟩ for all k.

    For each k-point, the integral is computed with kpti=kpts[k] and
    kptj=kpts[(k + q_idx) % nk].  This yields the proper off-diagonal
    (k, k+q) tensor required by the finite-q BSE.

    Parameters
    ----------
    cell   : PySCF Cell object
    mydf   : GDF density-fitting object (already built)
    kpts   : ndarray, shape (nk, 3)
    q_idx  : int  index of q in the BvK k-mesh (0 returns same-k integrals)

    Returns
    -------
    VQ_kq : ndarray, shape (nk, NQ, nao, nao) complex128
    """
    nk  = len(kpts)
    nao = cell.nao_nr()
    NQ  = mydf.get_naoaux()
    VQ_kq = np.zeros((nk, NQ, nao, nao), dtype=np.complex128)

    for ik in range(nk):
        kq            = (ik + q_idx) % nk
        kpti_kptj     = np.array([kpts[ik], kpts[kq]])
        col = 0
        for Lpq in mydf.sr_loop(kpti_kptj=kpti_kptj, compact=False):
            Lpq  = Lpq[0] + Lpq[1] * 1j
            naux = Lpq.shape[0]
            VQ_kq[ik, col:col + naux] = Lpq.reshape(naux, nao, nao)
            col += naux

    return VQ_kq


# ---------------------------------------------------------------------------
# Synthetic self-energy
# ---------------------------------------------------------------------------

def build_synthetic_sim(kmf, tau_mesh, beta=1000.0):
    """
    Build a minimal sim.h5 dataset (self-energy + Green's function).

    Returns a dict with all iter1 / iter22 datasets.
    """
    nk   = len(kmf.kpts)
    nao  = kmf.cell.nao_nr()
    ns   = 1
    ntau = len(tau_mesh)

    hcore_k = kmf.get_hcore()
    vhf_k   = kmf.get_veff()

    Sigma1 = np.zeros((ns, nk, nao, nao), dtype=np.complex128)
    for ik in range(nk):
        Sigma1[0, ik] = vhf_k[ik]

    decay     = np.exp(-np.abs(tau_mesh - beta / 2) * 0.005)
    Sigma_tau = np.zeros((ntau, ns, nk, nao, nao), dtype=np.complex128)
    for itau in range(ntau):
        for ik in range(nk):
            Sigma_tau[itau, 0, ik] = Sigma1[0, ik] * decay[itau] * 0.01

    mo_energy = np.array(kmf.mo_energy)
    mu = float(np.mean([mo_energy[ik, nao // 2 - 1] + mo_energy[ik, nao // 2]
                        for ik in range(nk)]) / 2)

    G_tau = np.zeros((ntau, ns, nk, nao, nao), dtype=np.complex128)
    for itau in range(ntau):
        tau = tau_mesh[itau]
        for ik in range(nk):
            eps = mo_energy[ik] - mu
            n_F    = 1.0 / (np.exp(np.clip(beta * eps, -500.0, 500.0)) + 1.0)
            exp_bm = np.exp(np.clip((beta - tau) * eps, -500.0, 500.0))
            exp_mt = np.exp(np.clip(-tau * eps,          -500.0, 500.0))
            g_diag = -(n_F * exp_bm + (1.0 - n_F) * (-exp_mt))
            G_tau[itau, 0, ik] = np.diag(g_diag)

    return {
        'Sigma1':    Sigma1,
        'Sigma_tau': Sigma_tau,
        'G_tau':     G_tau,
        'mu':        mu,
        'tau_mesh':  tau_mesh,
    }


# ---------------------------------------------------------------------------
# Polarization computation
# ---------------------------------------------------------------------------

def compute_tildeP_iw(G_tau, VQ_ao, beta, ir_file):
    """
    Compute the screened polarization P̃(iΩ) for q=0 using the BvK k-sum.

    Parameters
    ----------
    G_tau  : (ntau, ns, nk, nao, nao) complex128
    VQ_ao  : (nk, NQ, nao, nao)       complex128  same-k integrals
    beta   : float
    ir_file: str

    Returns
    -------
    P_iw : (niw, ns, 1, NQ, NQ) complex128
    """
    bse_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'green_bse')
    sys.path.insert(0, bse_dir)
    from irFT import IR_factory
    from scipy.linalg import inv

    ntau, ns, nk, nao, _ = G_tau.shape
    NQ = VQ_ao.shape[1]

    ir = IR_factory(beta, ir_file)
    assert ntau == ir.nts, (
        f"G_tau ntau={ntau} does not match IR grid nts={ir.nts}."
    )

    P0 = np.zeros((ntau, ns, 1, NQ, NQ), dtype=np.complex128)

    for t in range(ntau // 2):
        tt = ntau - t - 1
        for s in range(ns):
            P_block = np.zeros((NQ, NQ), dtype=np.complex128)
            for k in range(nk):
                G1 = G_tau[tt, s, k]
                G2 = G_tau[t,  s, k]
                V  = VQ_ao[k]

                A   = np.tensordot(V.conj(), G1, axes=([2], [0]))
                B   = np.tensordot(G2, V, axes=([1], [1])).transpose(0, 2, 1)
                A2d = A.reshape(NQ, nao * nao)
                B2d = B.reshape(nao * nao, NQ)
                P_block -= (2.0 / nk) * (A2d @ B2d)

            P0[t, s, 0] = P_block

    for t in range(ntau // 2):
        tt = ntau - t - 1
        for s in range(ns):
            H = 0.5 * (P0[t, s, 0] + P0[t, s, 0].conj().T)
            P0[t,  s, 0] = H
            P0[tt, s, 0] = H

    P0_iw = ir.tauf_to_wb(P0)
    niw   = P0_iw.shape[0]

    P_iw = np.zeros_like(P0_iw)
    for iw in range(niw):
        for s in range(ns):
            I_NQ = np.eye(NQ, dtype=np.complex128)
            temp = inv(I_NQ - P0_iw[iw, s, 0])
            P_iw[iw, s, 0] = temp @ P0_iw[iw, s, 0]

    P_iw = 0.5 * (P_iw + P_iw.conj().transpose(0, 1, 2, 4, 3))
    return P_iw


def compute_tildeP_iw_fq(G_tau, VQ_kq_ao, kq_map, beta, ir_file):
    """
    Compute P̃(q, iΩ) using proper off-diagonal (k, k+q) transition integrals.

    The polarization bubble at finite q is:

        P0[Q1,Q2](τ) = -(2/Nk) Σ_k Σ_{ac}
            [Σ_d V_kq*[Q1,a,d] G_{k+q}(β-τ)[d,c]] ·
            [Σ_b G_k(τ)[a,b] V_kq[Q2,b,c]]

    which follows gwtool.eval_P0_tilde_Q with the k-sum included explicitly.

    After computing P0(τ), a bosonic Fourier transform (τ→iΩ) and the Dyson
    equation P̃ = (I − P0)^{-1} P0 give the screened polarization.

    Parameters
    ----------
    G_tau    : (ntau, ns, nk, nao, nao) complex128
    VQ_kq_ao : (nk, NQ, nao, nao)       complex128  off-diagonal (k, k+q) integrals
    kq_map   : ndarray, shape (nk,)      kq_map[k] = index of k+q
    beta     : float
    ir_file  : str

    Returns
    -------
    P_iw : (niw, ns, 1, NQ, NQ) complex128
    """
    bse_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'green_bse')
    sys.path.insert(0, bse_dir)
    from irFT import IR_factory
    from scipy.linalg import inv

    ntau, ns, nk, nao, _ = G_tau.shape
    NQ = VQ_kq_ao.shape[1]

    ir = IR_factory(beta, ir_file)
    assert ntau == ir.nts, (
        f"G_tau ntau={ntau} does not match IR grid nts={ir.nts}."
    )

    # P0[Q1,Q2](τ) at finite q
    P0 = np.zeros((ntau, ns, 1, NQ, NQ), dtype=np.complex128)

    for t in range(ntau // 2):
        tt = ntau - t - 1
        for s in range(ns):
            P_block = np.zeros((NQ, NQ), dtype=np.complex128)
            for ik in range(nk):
                ikq = kq_map[ik]
                G_k_tau  = G_tau[t,  s, ik]    # G_k(τ),    shape (nao, nao)
                G_kq_btm = G_tau[tt, s, ikq]   # G_{k+q}(β-τ), shape (nao, nao)
                V_kq     = VQ_kq_ao[ik]         # (NQ, nao, nao)

                # X1[Q1, a, c] = Σ_d V_kq*[Q1, a, d] G_{k+q}(β-τ)[d, c]
                # tensordot(V.conj(), G, axes=([2], [0])) → shape (NQ, nao, nao)
                X1  = np.tensordot(V_kq.conj(), G_kq_btm, axes=([2], [0]))

                # B[a, c, Q2] = Σ_b G_k(τ)[a, b] V_kq[Q2, b, c]
                # tensordot(G, V, axes=([1], [1])) → shape (nao, nao, NQ)
                # then transpose to (nao, NQ, nao) is NOT what we want —
                # we need (a, c, Q2), so transpose(0, 2, 1) gives (nao, nao, NQ)
                Bmat = np.tensordot(G_k_tau, V_kq, axes=([1], [1])).transpose(0, 2, 1)

                # P0[Q1, Q2] -= (2/Nk) Σ_{a,c} X1[Q1,a,c] B[a,c,Q2]
                X1_2d  = X1.reshape(NQ, nao * nao)
                B_2d   = Bmat.reshape(nao * nao, NQ)
                P_block -= (2.0 / nk) * (X1_2d @ B_2d)

            P0[t, s, 0] = P_block

    # Bosonic symmetry: P0(τ) = P0(β−τ)
    for t in range(ntau // 2):
        tt = ntau - t - 1
        for s in range(ns):
            H = 0.5 * (P0[t, s, 0] + P0[t, s, 0].conj().T)
            P0[t,  s, 0] = H
            P0[tt, s, 0] = H

    # τ → bosonic Matsubara transform
    P0_iw = ir.tauf_to_wb(P0)
    niw   = P0_iw.shape[0]

    # Dyson: P̃ = (I − P0)^{-1} P0
    P_iw = np.zeros_like(P0_iw)
    for iw in range(niw):
        for s in range(ns):
            I_NQ = np.eye(NQ, dtype=np.complex128)
            temp = inv(I_NQ - P0_iw[iw, s, 0])
            P_iw[iw, s, 0] = temp @ P0_iw[iw, s, 0]

    # Final Hermitization
    P_iw = 0.5 * (P_iw + P_iw.conj().transpose(0, 1, 2, 4, 3))
    return P_iw


# ---------------------------------------------------------------------------
# Write functions
# ---------------------------------------------------------------------------

def write_mean_field_input(path, kmf):
    """Write mean_field_input.h5 in green-mbpt format."""
    cell = kmf.cell
    nk   = len(kmf.kpts)
    nao  = cell.nao_nr()
    ns   = 1

    hcore_k = kmf.get_hcore()
    fock_k  = np.array(kmf.get_fock())
    ovlp_k  = np.array(kmf.get_ovlp())

    def _prep(arr_nk):
        arr = np.array(arr_nk, dtype=np.complex128)
        out = np.zeros((ns, nk, nao, nao, 2), dtype=np.float64)
        out[0, :, :, :, 0] = arr.real
        out[0, :, :, :, 1] = arr.imag
        return out

    nel_cell    = int(cell.nelectron)
    NQ          = kmf.with_df.get_naoaux()
    kpts_scaled = cell.get_scaled_kpts(kmf.kpts)
    k_idx       = np.arange(nk, dtype=np.int64)

    def basis_name(c):
        b = c.basis
        return b if isinstance(b, str) else 'custom'

    with h5py.File(path, 'w') as f:
        f.create_dataset('HF/Fock-k',   data=_prep(fock_k))
        f.create_dataset('HF/H-k',      data=_prep(hcore_k))
        f.create_dataset('HF/S-k',      data=_prep(ovlp_k))
        f.create_dataset('HF/Energy',   data=float(kmf.e_tot))
        f.create_dataset('HF/Energy_nuc', data=float(cell.energy_nuc()))
        f.create_dataset('HF/madelung', data=0.0)
        f.create_dataset('HF/Nk',       data=np.int64(nk))
        f.create_dataset('HF/nk',       data=np.int64(nk))
        f.create_dataset('HF/mo_energy', data=np.array(kmf.mo_energy[0], dtype=np.float64))
        f.create_dataset('HF/mo_coeff',  data=np.array(kmf.mo_coeff[0].real, dtype=np.float64))

        f.create_dataset('params/nao',      data=np.int64(nao))
        f.create_dataset('params/nel_cell', data=np.int64(nel_cell))
        f.create_dataset('params/nk',       data=np.int64(nk))
        f.create_dataset('params/ns',       data=np.int64(ns))
        f.create_dataset('params/nso',      data=np.int64(nao))
        f.create_dataset('params/NQ',       data=np.int64(NQ))

        f.create_dataset('grid/nk',               data=np.int64(nk))
        f.create_dataset('grid/ink',              data=np.int64(nk))
        f.create_dataset('grid/k_mesh',           data=kmf.kpts.astype(np.float64))
        f.create_dataset('grid/k_mesh_scaled',    data=kpts_scaled.astype(np.float64))
        f.create_dataset('grid/index',            data=k_idx)
        f.create_dataset('grid/ir_list',          data=k_idx)
        f.create_dataset('grid/weight',           data=np.ones(nk) / nk)
        f.create_dataset('grid/conj_list',        data=k_idx)
        f.create_dataset('grid/conj_pairs_list',  data=k_idx)
        f.create_dataset('grid/trans_pairs_list', data=k_idx)
        kpair_idx = np.column_stack([k_idx, k_idx])
        f.create_dataset('grid/kpair_idx',        data=kpair_idx.astype(np.int64))
        f.create_dataset('grid/kpair_irre_list',  data=k_idx)
        f.create_dataset('grid/num_kpair_stored', data=np.int64(nk))

        nat = cell.natm
        nao_per_atom = nao // nat
        last_ao = np.array([(i + 1) * nao_per_atom for i in range(nat)], dtype=np.int64)
        Zs      = np.array([cell.atom_charge(i) for i in range(nat)], dtype=np.int32)
        f.create_dataset('mulliken/Zs',     data=Zs)
        f.create_dataset('mulliken/last_ao', data=last_ao)

        geom_str = json.dumps({'atom': str(cell.atom), 'basis': basis_name(cell)})
        f.create_dataset('Cell', data=geom_str.encode())

    print(f"  Wrote {path}")


def write_sim(path, sim_data, iter_num=22):
    """Write sim.h5 in green-mbpt format."""
    Sigma1    = sim_data['Sigma1']
    Sigma_tau = sim_data['Sigma_tau']
    G_tau     = sim_data['G_tau']
    mu        = float(sim_data['mu'])
    tau_mesh  = sim_data['tau_mesh']

    with h5py.File(path, 'w') as f:
        f.create_dataset('iter', data=np.int64(iter_num))
        for it in [1, iter_num]:
            grp = f'iter{it}'
            f.create_dataset(f'{grp}/mu',           data=mu)
            f.create_dataset(f'{grp}/Sigma1',       data=Sigma1.astype(np.complex128))
            f.create_dataset(f'{grp}/Energy_1b',    data=0.0)
            f.create_dataset(f'{grp}/Energy_2b',    data=0.0)
            f.create_dataset(f'{grp}/Energy_HF',    data=0.0)
            f.create_dataset(f'{grp}/Selfenergy/data', data=Sigma_tau.astype(np.complex128))
            f.create_dataset(f'{grp}/Selfenergy/mesh', data=tau_mesh.astype(np.float64))
            f.create_dataset(f'{grp}/G_tau/data',   data=G_tau.astype(np.complex128))
            f.create_dataset(f'{grp}/G_tau/mesh',   data=tau_mesh.astype(np.float64))

    print(f"  Wrote {path}")


def write_VQ(int_path, VQ_ao):
    """Write df_hf_int/VQ_0.h5 and meta.h5 (same-k integrals)."""
    os.makedirs(int_path, exist_ok=True)
    nk = VQ_ao.shape[0]

    vq_path   = os.path.join(int_path, 'VQ_0.h5')
    meta_path = os.path.join(int_path, 'meta.h5')

    VQ_stored = _vq_to_storage(VQ_ao)

    with h5py.File(vq_path, 'w') as f:
        f.create_dataset('0', data=VQ_stored)
    print(f"  Wrote {vq_path}")

    with h5py.File(meta_path, 'w') as f:
        f.create_dataset('chunk_size',    data=np.int64(nk))
        f.create_dataset('chunk_indices', data=np.arange(nk, dtype=np.int64))
    print(f"  Wrote {meta_path}")


def write_VQ_kq(int_path, VQ_kq, q_idx):
    """
    Write the off-diagonal GDF integrals to df_hf_int_fq/VQ_q{q_idx}.h5.

    The file contains dataset '0' with shape (nk, NQ, nao, 2*nao) float64
    (interleaved real/imag), matching the convention in contract.readVQ().

    Parameters
    ----------
    int_path : str   path to the df_hf_int_fq/ directory
    VQ_kq    : ndarray, shape (nk, NQ, nao, nao) complex128
    q_idx    : int
    """
    os.makedirs(int_path, exist_ok=True)
    path      = os.path.join(int_path, f'VQ_q{q_idx}.h5')
    VQ_stored = _vq_to_storage(VQ_kq)

    with h5py.File(path, 'w') as f:
        f.create_dataset('0', data=VQ_stored)
    print(f"  Wrote {path}")


def write_tildeP_iw(path, P_iw, iter_num=22):
    """Write P̃(iΩ) to path in the format expected by contract.readPtilde()."""
    with h5py.File(path, 'w') as f:
        f.create_dataset('iter', data=np.int64(iter_num))
        f.create_dataset(f'iter{iter_num}/P_iw_tilde',
                         data=P_iw.astype(np.complex128))
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Generate synthetic periodic finite-q BSE inputs from a 1-D H₂ chain. '
            'Produces proper off-diagonal GDF integrals and finite-q P̃(q,iΩ).'
        )
    )
    parser.add_argument('--nk',        type=int,   default=4,
                        help='Number of k-points along the chain (default: 4)')
    parser.add_argument('--q_idx',     type=int,   default=1,
                        help='Index of q-vector in the BvK k-mesh (default: 1)')
    parser.add_argument('--bond',      type=float, default=1.4,
                        help='H-H bond length in Bohr (default: 1.4)')
    parser.add_argument('--lattice',   type=float, default=None,
                        help='Lattice constant in Bohr (default: 2*bond+1.5)')
    parser.add_argument('--basis',     type=str,   default='sto-3g',
                        help='Basis set (default: sto-3g)')
    parser.add_argument('--outdir',    type=str,   default=None,
                        help='Output directory (default: example/H2_periodic_<nk>k)')
    parser.add_argument('--ir_source', type=str,
                        default='example/N2_STO3G/irgrid/1e5.h5',
                        help='IR grid file to copy into output directory')
    args = parser.parse_args()

    nk     = args.nk
    q_idx  = args.q_idx
    outdir = args.outdir or f'example/H2_periodic_{nk}k'

    if q_idx < 0 or q_idx >= nk:
        raise ValueError(f"q_idx={q_idx} must be in [0, {nk})")

    os.makedirs(outdir, exist_ok=True)

    # --- IR grid ---
    ir_dir  = os.path.join(outdir, 'irgrid')
    os.makedirs(ir_dir, exist_ok=True)
    ir_dest = os.path.join(ir_dir, '1e5.h5')
    if os.path.exists(args.ir_source):
        shutil.copy2(args.ir_source, ir_dest)
        print(f"  Copied IR grid → {ir_dest}")
    else:
        raise FileNotFoundError(
            f"IR source not found: {args.ir_source}\n"
            "Run from the project root or adjust --ir_source."
        )

    ntau, _xgrid, tau_mesh = _get_tau_grid(ir_dest)

    # --- Mean-field ---
    print(f"\n--- Running k-RHF on 1-D H₂ chain: nk={nk}, basis={args.basis} ---")
    cell, kmf, mydf = run_krhf(
        nk=nk, basis=args.basis,
        bond_length=args.bond, lattice_const=args.lattice,
        verbose=3
    )

    nao = cell.nao_nr()
    nel = int(cell.nelectron)
    print(f"\nSystem: nao={nao}, nel={nel}, nk={nk}")
    print(f"HF total energy: {kmf.e_tot:.6f} Ha")
    for ik in range(nk):
        print(f"  k[{ik}] MO energies (Ha): {np.array(kmf.mo_energy[ik])}")

    # --- Same-k VQ ---
    print(f"\n--- Extracting same-k VQ integrals (nk={nk}) ---")
    VQ_ao = extract_VQ(cell, mydf, kmf.kpts)
    print(f"  VQ_kk shape: {VQ_ao.shape}  (nk, NQ, nao, nao)")

    # --- Off-diagonal VQ_kq ---
    if q_idx == 0:
        print(f"\n--- q_idx=0: off-diagonal integrals equal same-k integrals ---")
        VQ_kq = VQ_ao
    else:
        print(f"\n--- Extracting off-diagonal VQ_kq integrals (q_idx={q_idx}) ---")
        VQ_kq = extract_VQ_kq(cell, mydf, kmf.kpts, q_idx)
        print(f"  VQ_kq shape: {VQ_kq.shape}  (nk, NQ, nao, nao)")

    # --- Synthetic self-energy & Green's function ---
    print(f"\n--- Building synthetic self-energy (ntau={ntau}) ---")
    sim_data = build_synthetic_sim(kmf, tau_mesh, beta=1000.0)
    print(f"  Chemical potential μ = {sim_data['mu']:.6f} Ha")

    # --- q=0 polarization ---
    print(f"\n--- Computing P̃(iΩ) for q=0 ---")
    P_iw_q0 = compute_tildeP_iw(
        sim_data['G_tau'], VQ_ao, beta=1000.0, ir_file=ir_dest
    )
    print(f"  P̃_q0(iΩ) shape: {P_iw_q0.shape}")

    # --- finite-q polarization ---
    kq_map = (np.arange(nk) + q_idx) % nk
    if q_idx == 0:
        print(f"\n--- q_idx=0: P̃_fq = P̃_q0 ---")
        P_iw_fq = P_iw_q0
    else:
        print(f"\n--- Computing P̃(q={q_idx}, iΩ) using off-diagonal integrals ---")
        P_iw_fq = compute_tildeP_iw_fq(
            sim_data['G_tau'], VQ_kq, kq_map, beta=1000.0, ir_file=ir_dest
        )
        print(f"  P̃_fq(iΩ) shape: {P_iw_fq.shape}")

    # --- Write files ---
    print(f"\n--- Writing output files to {outdir}/ ---")

    # Standard mean-field input and sim
    write_mean_field_input(os.path.join(outdir, 'mean_field_input.h5'), kmf)
    write_sim(os.path.join(outdir, 'sim.h5'), sim_data)

    # q=0 integral directory (standard periodic BSE)
    int_path_q0 = os.path.join(outdir, 'df_hf_int')
    write_VQ(int_path_q0, VQ_ao)

    # Finite-q integral directory
    int_path_fq = os.path.join(outdir, 'df_hf_int_fq')
    os.makedirs(int_path_fq, exist_ok=True)

    # Same-k copy → VQ_0.h5
    shutil.copy2(os.path.join(int_path_q0, 'VQ_0.h5'),
                 os.path.join(int_path_fq, 'VQ_0.h5'))
    print(f"  Copied VQ_0.h5 → {int_path_fq}/VQ_0.h5")

    # Same meta.h5
    shutil.copy2(os.path.join(int_path_q0, 'meta.h5'),
                 os.path.join(int_path_fq, 'meta.h5'))
    print(f"  Copied meta.h5 → {int_path_fq}/meta.h5")

    # Off-diagonal VQ_q{idx}.h5
    write_VQ_kq(int_path_fq, VQ_kq, q_idx)

    # Polarization files
    write_tildeP_iw(os.path.join(outdir, 'p_iw_tilde.h5'), P_iw_q0)
    write_tildeP_iw(os.path.join(outdir, f'p_iw_tilde_q{q_idx}.h5'), P_iw_fq)

    # --- q=0 run script ---
    run_sh_q0 = os.path.join(outdir, 'bse_periodic_singlet.sh')
    with open(run_sh_q0, 'w') as f:
        f.write(f"""#!/bin/bash
# Auto-generated by generate_finite_q_test.py
# System: 1-D H2 chain, nk={nk}, basis={args.basis}
# q=0 optical-limit BSE (standard periodic solver)

export SCRIPTDIR=../../script
export IRDIR=./irgrid

python -u $SCRIPTDIR/solveCasida_periodic.py \\
       --type     singlet       \\
       --beta     1000          \\
       --qpac     0             \\
       --iter     -1            \\
       --iter_W   -1            \\
       --calc_pi  0             \\
       --monitor  1             \\
       --n_jobs   -1            \\
       --input    ./mean_field_input.h5   \\
       --sim      ./sim.h5               \\
       --int_path ./df_hf_int/           \\
       --ir_file  $IRDIR/1e5.h5          \\
       --pi_file  ./p_iw_tilde.h5        \\
       --output   ./bse_periodic_singlet.h5
""")
    os.chmod(run_sh_q0, 0o755)
    print(f"  Wrote run script → {run_sh_q0}")

    # --- finite-q run script ---
    run_sh_fq = os.path.join(outdir, f'bse_fq{q_idx}_singlet.sh')
    with open(run_sh_fq, 'w') as f:
        f.write(f"""#!/bin/bash
# Auto-generated by generate_finite_q_test.py
# System: 1-D H2 chain, nk={nk}, basis={args.basis}
# Finite-q BSE at q_idx={q_idx}  (kq_map[k] = (k + {q_idx}) % {nk})

export SCRIPTDIR=../../script
export IRDIR=./irgrid

python -u $SCRIPTDIR/solveCasida_finite_q.py \\
       --type     singlet       \\
       --beta     1000          \\
       --qpac     0             \\
       --iter     -1            \\
       --iter_W   -1            \\
       --calc_pi  0             \\
       --monitor  1             \\
       --n_jobs   -1            \\
       --q_idx    {q_idx}            \\
       --input    ./mean_field_input.h5       \\
       --sim      ./sim.h5                   \\
       --int_path ./df_hf_int_fq/            \\
       --ir_file  $IRDIR/1e5.h5              \\
       --pi_file  ./p_iw_tilde_q{q_idx}.h5   \\
       --output   ./bse_fq{q_idx}_singlet.h5
""")
    os.chmod(run_sh_fq, 0o755)
    print(f"  Wrote run script → {run_sh_fq}")

    print(f"""
=== Done ===
Generated files in {outdir}/

To run the q=0 (optical limit) BSE solver:
  cd {outdir}
  bash bse_periodic_singlet.sh

To run the finite-q BSE solver (q_idx={q_idx}):
  cd {outdir}
  bash bse_fq{q_idx}_singlet.sh

Or from the project root:
  python script/solveCasida_finite_q.py \\
      --type singlet --beta 1000 --qpac 0 --calc_pi 0 --n_jobs -1 \\
      --q_idx    {q_idx} \\
      --input    {outdir}/mean_field_input.h5 \\
      --sim      {outdir}/sim.h5 \\
      --int_path {outdir}/df_hf_int_fq/ \\
      --ir_file  {outdir}/irgrid/1e5.h5 \\
      --pi_file  {outdir}/p_iw_tilde_q{q_idx}.h5 \\
      --output   {outdir}/bse_fq{q_idx}_singlet.h5
""")


if __name__ == '__main__':
    main()
