#!/usr/bin/env python3
"""
generate_periodic_test.py
=========================
Generate synthetic periodic-system input files for testing the periodic BSE solver.

System: 1-D hydrogen chain  H2 ... H2 ... H2 (unit cell = H-H, STO-3G, k-mesh Nk x 1 x 1)

Produces the HDF5 layout expected by PeriodicBSESolver:
  <outdir>/mean_field_input.h5   — k-RHF Fock/overlap/core matrices
  <outdir>/sim.h5                — G(τ) and synthetic Σ(τ)
  <outdir>/df_hf_int/VQ_0.h5    — GDF 3-centre integrals V_Q^k
  <outdir>/df_hf_int/meta.h5    — chunk index metadata
  <outdir>/p_iw_tilde.h5        — precomputed P̃(iΩ) for q=0
  <outdir>/irgrid/1e5.h5        — IR grid (copied from N2_STO3G example)
  <outdir>/bse_periodic_singlet.sh  — ready-to-run script (--calc_pi 0)

Why --calc_pi 0?
  The built-in on-the-fly P0 computation (gwtool.readVQFromMeta) assumes a
  k-independent VQ format used by the molecular code.  For nk>1, this
  generator precomputes P̃(iΩ) directly with the correct k-sum and stores it
  in p_iw_tilde.h5; the solver then reads it with --calc_pi 0 --pi_file.

Usage
-----
  python script/generate_periodic_test.py --nk 2 --outdir example/H2_periodic_2k
  python script/generate_periodic_test.py --nk 4 --outdir example/H2_periodic_4k

Then run the BSE solver (or just: cd <outdir> && bash bse_periodic_singlet.sh):
  python script/solveCasida_periodic.py \\
      --type singlet --beta 1000 --qpac 0 --calc_pi 0 --monitor 1 --n_jobs -1 \\
      --input  <outdir>/mean_field_input.h5 \\
      --sim    <outdir>/sim.h5 \\
      --int_path <outdir>/df_hf_int/ \\
      --ir_file  <outdir>/irgrid/1e5.h5 \\
      --pi_file  <outdir>/p_iw_tilde.h5 \\
      --output   <outdir>/bse_periodic_singlet.h5

Notes
-----
* --qpac 0 is recommended for this synthetic case because the self-energy is
  approximated as the HF exchange (iter=1 → G0W0 starting point).  Use
  --qpac 1 to apply the Padé QP correction anyway.
* The VQ integrals are written as a single file VQ_0.h5 with dataset '/0'
  of shape (nk, NQ, nao, nao) stored as float64 complex pairs, matching the
  convention in contract.readVQ().
* The self-energy stored in sim.h5 is the bare HF exchange (frequency-
  independent Fock minus core), so the "scGW" result at iter=1 is equivalent
  to a GW@HF starting point without self-consistency.
"""

import argparse
import os
import shutil
import json

import h5py
import numpy as np

try:
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc import scf as pbcscf
    from pyscf.pbc import df  as pbcdf
except ImportError:
    raise SystemExit(
        "PySCF not found.  Install with:  pip install pyscf"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_real_pairs(arr):
    """
    Store a complex128 array as float64 with an extra trailing dimension of 2.
    This is the green-mbpt HDF5 convention for complex data.
    arr : ndarray, shape (..., m, n) complex128
    returns ndarray, shape (..., m, n, 2) float64
    """
    out = np.zeros(arr.shape + (2,), dtype=np.float64)
    out[..., 0] = arr.real
    out[..., 1] = arr.imag
    return out


def _vq_to_storage(VQ_ao):
    """
    VQ_ao : (nk, NQ, nao, nao) complex128
    Returns (nk, NQ, nao, 2*nao) float64  — interleaved real/imag along last axis.
    This is what readVQ / readH5 with .view(complex) expects.
    """
    nk, NQ, nao, _ = VQ_ao.shape
    out = np.zeros((nk, NQ, nao, 2 * nao), dtype=np.float64)
    out[:, :, :, 0::2] = VQ_ao.real
    out[:, :, :, 1::2] = VQ_ao.imag
    return out


def _get_tau_grid(ir_file, beta=1000.0):
    """
    Return the fermionic tau mesh (nts points) from the IR file.

    IR_factory uses the *fermionic* xgrid for G(τ): it maps x∈[-1,1] to
    τ∈[0,β] and prepends τ=0 and appends τ=β, giving nts = n_fermi_x + 2.
    The bosonic xgrid has a different size and must NOT be used here.
    """
    with h5py.File(ir_file, 'r') as f:
        xgrid = f['fermi/xgrid'][()]   # shape (n_x,) in [-1, 1]

    # Reproduce the tau_mesh built by new_read_IR_matrices / legacy_read_IR_matrices
    tau_mesh = np.zeros(xgrid.shape[0] + 2)
    tau_mesh[0]    = 0.0
    tau_mesh[1:-1] = (xgrid + 1) * beta / 2.0
    tau_mesh[-1]   = beta

    ntau = tau_mesh.shape[0]   # = n_x + 2  (e.g. 144 for 1e5.h5 fermi grid)
    return ntau, xgrid, tau_mesh


# ---------------------------------------------------------------------------
# Mean-field: k-point RHF with density fitting
# ---------------------------------------------------------------------------

def run_krhf(nk, basis='sto-3g', bond_length=1.4, lattice_const=None, verbose=0):
    """
    Run k-point RHF on a 1-D H₂ chain.

    H-H bond length  : bond_length (Bohr)
    Lattice constant : lattice_const (Bohr); defaults to 2 * bond_length + 0.5
    k-mesh           : [nk, 1, 1]

    Returns
    -------
    cell, kmf, mydf : PySCF objects
    """
    if lattice_const is None:
        lattice_const = 2 * bond_length + 1.5   # ~4.3 Bohr for bond_length=1.4

    cell = pbcgto.Cell()
    cell.atom   = f'H 0 0 0; H 0 0 {bond_length}'
    cell.basis  = basis
    cell.a      = np.diag([20.0, 20.0, lattice_const])  # large vacuum in x,y
    cell.unit   = 'B'
    cell.verbose = verbose
    cell.build()

    kpts = cell.make_kpts([nk, 1, 1])

    # Gaussian density fitting
    mydf = pbcdf.GDF(cell, kpts)
    mydf.build()

    kmf = pbcscf.KRHF(cell, kpts)
    kmf.with_df = mydf
    kmf.init_guess = 'atom'   # avoid minao which is broken with scipy>=1.11
    kmf.kernel()

    return cell, kmf, mydf


# ---------------------------------------------------------------------------
# Extract VQ integrals
# ---------------------------------------------------------------------------

def extract_VQ(cell, mydf, kpts):
    """
    Extract 3-centre density-fitting integrals V_{Q,μν}^k for each k-point.

    Returns VQ_ao : (nk, NQ, nao, nao) complex128
    """
    nk  = len(kpts)
    nao = cell.nao_nr()
    NQ  = mydf.get_naoaux()

    VQ_ao = np.zeros((nk, NQ, nao, nao), dtype=np.complex128)

    for ik, kpt in enumerate(kpts):
        # GDF stores integrals in (nao_pair, NQ) or (NQ, nao, nao) format;
        # sr_loop yields (NQ_chunk, nao, nao) blocks.
        col = 0
        kpti_kptj = np.array([kpt, kpt])
        for Lpq in mydf.sr_loop(kpti_kptj=kpti_kptj, compact=False):
            Lpq   = Lpq[0] + Lpq[1] * 1j    # real + imag parts from sr_loop
            naux  = Lpq.shape[0]
            VQ_ao[ik, col:col + naux, :, :] = Lpq.reshape(naux, nao, nao)
            col  += naux

    return VQ_ao


# ---------------------------------------------------------------------------
# Build synthetic self-energy
# ---------------------------------------------------------------------------

def build_synthetic_sim(kmf, tau_mesh, beta=1000.0):
    """
    Build a minimal sim.h5 dataset.

    Self-energy Σ(iτ): we use the HF exchange (static), which corresponds
    to the G0W0 starting point.  The frequency-dependent part is set to a
    small smooth function consistent with the IR grid so that the Padé
    analytic continuation has no numerical issues.

    tau_mesh : (ntau,) float64 — fermionic tau grid including endpoints 0 and β

    Returns a dict with all iter1 / iter22 datasets.
    """
    nk   = len(kmf.kpts)
    nao  = kmf.cell.nao_nr()
    ns   = 1   # spin-restricted
    ntau = len(tau_mesh)

    # Static self-energy = Fock - core Hamiltonian (exchange only)
    # Shape: (ns, nk, nao, nao)
    hcore_k = kmf.get_hcore()          # (nk, nao, nao)
    vhf_k   = kmf.get_veff()           # (nk, nao, nao) — HF potential = J + K

    Sigma1 = np.zeros((ns, nk, nao, nao), dtype=np.complex128)
    for ik in range(nk):
        Sigma1[0, ik] = vhf_k[ik]     # exchange-correlation part of Fock

    # Frequency-dependent self-energy: make it small and smooth.
    # We use a simple Lorentzian decay in τ: Σ(τ) = Σ1 * exp(-|τ-β/2| * 0.005)
    # Shape: (ntau, ns, nk, nao, nao)
    decay    = np.exp(-np.abs(tau_mesh - beta / 2) * 0.005)   # slow decay

    Sigma_tau = np.zeros((ntau, ns, nk, nao, nao), dtype=np.complex128)
    for itau in range(ntau):
        for ik in range(nk):
            Sigma_tau[itau, 0, ik] = Sigma1[0, ik] * decay[itau] * 0.01

    # Non-interacting Green's function G0(τ) from HF eigenvalues
    mo_energy = np.array(kmf.mo_energy)   # (nk, nao)
    mu        = float(np.mean([mo_energy[ik, nao // 2 - 1] +
                                mo_energy[ik, nao // 2]
                                for ik in range(nk)]) / 2)

    G_tau = np.zeros((ntau, ns, nk, nao, nao), dtype=np.complex128)
    for itau in range(ntau):
        tau = tau_mesh[itau]
        for ik in range(nk):
            eps = mo_energy[ik] - mu
            # G(τ) = -[n_F e^{+(β-τ)ε} + (1-n_F)(-e^{-τε})]  (diagonal MO)
            # Clip exponent arguments to avoid overflow (β~1000, |ε|~1 Ha)
            n_F    = 1.0 / (np.exp(np.clip(beta * eps, -500.0, 500.0)) + 1.0)
            exp_bm = np.exp(np.clip((beta - tau) * eps,  -500.0, 500.0))
            exp_mt = np.exp(np.clip(-tau * eps,           -500.0, 500.0))
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
# Periodic P0 / P_tilde computation (bypasses gwtool.readVQFromMeta)
# ---------------------------------------------------------------------------

def compute_tildeP_iw(G_tau, VQ_ao, beta, ir_file):
    """
    Compute the screened polarization P̃(iΩ) for q=0 using the BvK k-sum.

    Parameters
    ----------
    G_tau  : (ntau, ns, nk, nao, nao) complex128
    VQ_ao  : (nk, NQ, nao, nao)      complex128
    beta   : inverse temperature (float)
    ir_file: path to IR grid HDF5

    Returns
    -------
    P_iw : (niw, ns, 1, NQ, NQ) complex128 — q=0 screened polarization
    """
    import sys
    bse_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'green_bse')
    sys.path.insert(0, bse_dir)
    from irFT import IR_factory
    from scipy.linalg import inv

    ntau, ns, nk, nao, _ = G_tau.shape
    NQ = VQ_ao.shape[1]

    ir = IR_factory(beta, ir_file)
    assert ntau == ir.nts, (
        f"G_tau ntau={ntau} does not match IR grid nts={ir.nts}. "
        "Did you use the correct tau mesh?"
    )

    # --- Independent-particle polarization P0(τ) ---
    # P0[Q1,Q2](t) = -(2/Nk) Σ_k Σ_{a,c} V*_k[Q1,a,c'] G_k(τ')[c',a'] V_k[Q2,a',c]
    # Following gwtool.eval_P0_tilde_Q:
    #   A[Q1,a,c] = Σ_d V*_k[Q1,a,d] G_k(τ')[d,c]
    #   B[a,c,Q2] = Σ_b G_k(τ)[a,b] V_k[Q2,b,c]
    #   P0[Q1,Q2] -= (2/Nk) Σ_{a,c} A[Q1,a,c] B[a,c,Q2]
    P0 = np.zeros((ntau, ns, 1, NQ, NQ), dtype=np.complex128)

    for t in range(ntau // 2):
        tt = ntau - t - 1
        for s in range(ns):
            P_block = np.zeros((NQ, NQ), dtype=np.complex128)
            for k in range(nk):
                G1 = G_tau[tt, s, k]   # (nao, nao), τ' = β-τ
                G2 = G_tau[t,  s, k]   # (nao, nao), τ
                V  = VQ_ao[k]           # (NQ, nao, nao)

                # A[Q1,a,c] = Σ_d V*[Q1,a,d] G1[d,c]  →  shape (NQ, nao, nao)
                A = np.tensordot(V.conj(), G1, axes=([2], [0]))
                # B[a,c,Q2] = Σ_b G2[a,b] V[Q2,b,c]   →  shape (nao, nao, NQ)
                B = np.tensordot(G2, V, axes=([1], [1])).transpose(0, 2, 1)
                # P0[Q1,Q2] -= (2/Nk) Σ_{a,c} A[Q1,a,c] B[a,c,Q2]
                # einsum 'Qac,acR->QR'
                A2d = A.reshape(NQ, nao * nao)
                B2d = B.reshape(nao * nao, NQ)
                P_block -= (2.0 / nk) * (A2d @ B2d)

            P0[t, s, 0] = P_block

    # Bosonic symmetry: P0(τ) = P0(β−τ); Hermitize each slice
    for t in range(ntau // 2):
        tt = ntau - t - 1
        for s in range(ns):
            H = 0.5 * (P0[t, s, 0] + P0[t, s, 0].conj().T)
            P0[t,  s, 0] = H
            P0[tt, s, 0] = H

    # --- Transform τ → bosonic Matsubara grid ---
    P0_iw = ir.tauf_to_wb(P0)    # (niw, ns, 1, NQ, NQ)
    niw = P0_iw.shape[0]

    # --- Dyson equation: P̃ = (I − P0)^{-1} P0 ---
    P_iw = np.zeros_like(P0_iw)
    for iw in range(niw):
        for s in range(ns):
            I_NQ = np.eye(NQ, dtype=np.complex128)
            temp = inv(I_NQ - P0_iw[iw, s, 0])
            P_iw[iw, s, 0] = temp @ P0_iw[iw, s, 0]

    # Final Hermitization
    P_iw = 0.5 * (P_iw + P_iw.conj().transpose(0, 1, 2, 4, 3))

    return P_iw


def write_tildeP_iw(path, P_iw, iter_num=22):
    """
    Write P̃(iΩ) to path in the format expected by contract.readPtilde().

    Layout: /iter = iter_num (int64),  /iter{N}/P_iw_tilde (complex128).
    The file is read via readH5 which does .view(complex), so storing as
    complex128 is correct (view is a no-op for complex data).
    """
    with h5py.File(path, 'w') as f:
        f.create_dataset('iter', data=np.int64(iter_num))
        f.create_dataset(f'iter{iter_num}/P_iw_tilde',
                         data=P_iw.astype(np.complex128))
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Write HDF5 files
# ---------------------------------------------------------------------------

def write_mean_field_input(path, kmf):
    """Write mean_field_input.h5 in green-mbpt format."""
    cell = kmf.cell
    nk   = len(kmf.kpts)
    nao  = cell.nao_nr()
    ns   = 1

    hcore_k = kmf.get_hcore()                           # (nk, nao, nao)
    fock_k  = np.array(kmf.get_fock())                  # (nk, nao, nao)
    ovlp_k  = np.array(kmf.get_ovlp())                  # (nk, nao, nao)

    # green-mbpt stores as (ns, nk, nao, nao, 2) float64
    def _prep(arr_nk):
        # arr_nk: (nk, nao, nao) complex
        arr = np.array(arr_nk, dtype=np.complex128)
        out = np.zeros((ns, nk, nao, nao, 2), dtype=np.float64)
        out[0, :, :, :, 0] = arr.real
        out[0, :, :, :, 1] = arr.imag
        return out

    nel_cell = int(cell.nelectron)
    NQ       = kmf.with_df.get_naoaux()

    # k-mesh metadata
    kpts_scaled = cell.get_scaled_kpts(kmf.kpts)   # (nk, 3)
    k_idx = np.arange(nk, dtype=np.int64)

    with h5py.File(path, 'w') as f:
        f.create_dataset('HF/Fock-k',   data=_prep(fock_k))
        f.create_dataset('HF/H-k',      data=_prep(hcore_k))
        f.create_dataset('HF/S-k',      data=_prep(ovlp_k))
        f.create_dataset('HF/Energy',   data=float(kmf.e_tot))
        f.create_dataset('HF/Energy_nuc', data=float(cell.energy_nuc()))
        f.create_dataset('HF/madelung', data=0.0)
        f.create_dataset('HF/Nk',       data=np.int64(nk))
        f.create_dataset('HF/nk',       data=np.int64(nk))

        # MO data at Gamma point (for back-compatibility with molecular tools)
        mo_e_k0 = np.array(kmf.mo_energy[0], dtype=np.float64)
        mo_c_k0 = np.array(kmf.mo_coeff[0].real, dtype=np.float64)
        f.create_dataset('HF/mo_energy', data=mo_e_k0)
        f.create_dataset('HF/mo_coeff',  data=mo_c_k0)

        f.create_dataset('params/nao',      data=np.int64(nao))
        f.create_dataset('params/nel_cell', data=np.int64(nel_cell))
        f.create_dataset('params/nk',       data=np.int64(nk))
        f.create_dataset('params/ns',       data=np.int64(ns))
        f.create_dataset('params/nso',      data=np.int64(nao))
        f.create_dataset('params/NQ',       data=np.int64(NQ))

        # k-grid
        f.create_dataset('grid/nk',               data=np.int64(nk))
        f.create_dataset('grid/ink',               data=np.int64(nk))
        f.create_dataset('grid/k_mesh',           data=kmf.kpts.astype(np.float64))
        f.create_dataset('grid/k_mesh_scaled',    data=kpts_scaled.astype(np.float64))
        f.create_dataset('grid/index',            data=k_idx)
        f.create_dataset('grid/ir_list',          data=k_idx)
        f.create_dataset('grid/weight',           data=np.ones(nk)/nk)
        f.create_dataset('grid/conj_list',        data=k_idx)
        f.create_dataset('grid/conj_pairs_list',  data=k_idx)
        f.create_dataset('grid/trans_pairs_list', data=k_idx)
        kpair_idx = np.column_stack([k_idx, k_idx])
        f.create_dataset('grid/kpair_idx',        data=kpair_idx.astype(np.int64))
        f.create_dataset('grid/kpair_irre_list',  data=k_idx)
        f.create_dataset('grid/num_kpair_stored', data=np.int64(nk))

        # Mulliken block boundaries (one block per atom)
        nat = cell.natm
        nao_per_atom = nao // nat
        last_ao = np.array([(i + 1) * nao_per_atom for i in range(nat)],
                           dtype=np.int64)
        Zs = np.array([cell.atom_charge(i) for i in range(nat)], dtype=np.int32)
        f.create_dataset('mulliken/Zs',    data=Zs)
        f.create_dataset('mulliken/last_ao', data=last_ao)

        # Store geometry as JSON string (mirrors N2_STO3G format)
        geom_str = json.dumps({'atom': str(cell.atom), 'basis': basis_name(cell)})
        f.create_dataset('Cell', data=geom_str.encode())

    print(f"  Wrote {path}")


def basis_name(cell):
    """Return a string for the basis name."""
    b = cell.basis
    return b if isinstance(b, str) else 'custom'


def write_sim(path, sim_data, iter_num=22):
    """Write sim.h5 in green-mbpt format."""
    Sigma1    = sim_data['Sigma1']     # (ns, nk, nao, nao)
    Sigma_tau = sim_data['Sigma_tau']  # (ntau, ns, nk, nao, nao)
    G_tau     = sim_data['G_tau']      # (ntau, ns, nk, nao, nao)
    mu        = float(sim_data['mu'])
    tau_mesh  = sim_data['tau_mesh']   # (ntau,)

    def _c2f(arr):
        """complex128 → stored as-is (green-mbpt reads with .view(complex))."""
        return arr.astype(np.complex128)

    with h5py.File(path, 'w') as f:
        f.create_dataset('iter', data=np.int64(iter_num))

        for it in [1, iter_num]:
            grp = f'iter{it}'
            f.create_dataset(f'{grp}/mu',          data=mu)
            f.create_dataset(f'{grp}/Sigma1',      data=_c2f(Sigma1))
            f.create_dataset(f'{grp}/Energy_1b',   data=0.0)
            f.create_dataset(f'{grp}/Energy_2b',   data=0.0)
            f.create_dataset(f'{grp}/Energy_HF',   data=0.0)
            f.create_dataset(f'{grp}/Selfenergy/data', data=_c2f(Sigma_tau))
            f.create_dataset(f'{grp}/Selfenergy/mesh', data=tau_mesh.astype(np.float64))
            f.create_dataset(f'{grp}/G_tau/data',  data=_c2f(G_tau))
            f.create_dataset(f'{grp}/G_tau/mesh',  data=tau_mesh.astype(np.float64))

    print(f"  Wrote {path}")


def write_VQ(int_path, VQ_ao):
    """Write df_hf_int/VQ_0.h5 and meta.h5."""
    os.makedirs(int_path, exist_ok=True)

    vq_path   = os.path.join(int_path, 'VQ_0.h5')
    meta_path = os.path.join(int_path, 'meta.h5')

    nk = VQ_ao.shape[0]
    VQ_stored = _vq_to_storage(VQ_ao)   # (nk, NQ, nao, 2*nao) float64

    with h5py.File(vq_path, 'w') as f:
        f.create_dataset('0', data=VQ_stored)
    print(f"  Wrote {vq_path}")

    with h5py.File(meta_path, 'w') as f:
        f.create_dataset('chunk_size',    data=np.int64(nk))
        f.create_dataset('chunk_indices', data=np.arange(nk, dtype=np.int64))
    print(f"  Wrote {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic periodic BSE inputs from a 1-D H₂ chain.'
    )
    parser.add_argument('--nk',         type=int,   default=2,
                        help='Number of k-points along the chain (default: 2)')
    parser.add_argument('--bond',       type=float, default=1.4,
                        help='H-H bond length in Bohr (default: 1.4)')
    parser.add_argument('--lattice',    type=float, default=None,
                        help='Lattice constant in Bohr (default: 2*bond+1.5)')
    parser.add_argument('--basis',      type=str,   default='sto-3g',
                        help='Basis set (default: sto-3g)')
    parser.add_argument('--outdir',     type=str,   default=None,
                        help='Output directory (default: example/H2_periodic_<nk>k)')
    parser.add_argument('--ir_source',  type=str,
                        default='example/N2_STO3G/irgrid/1e5.h5',
                        help='IR grid file to copy into output directory')
    args = parser.parse_args()

    nk     = args.nk
    outdir = args.outdir or f'example/H2_periodic_{nk}k'
    os.makedirs(outdir, exist_ok=True)

    # Copy IR grid
    ir_dir = os.path.join(outdir, 'irgrid')
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

    print(f"\n--- Extracting VQ integrals (nk={nk}) ---")
    VQ_ao = extract_VQ(cell, mydf, kmf.kpts)
    print(f"  VQ shape: {VQ_ao.shape}  (nk, NQ, nao, nao)")

    print(f"\n--- Building synthetic self-energy (ntau={ntau}) ---")
    sim_data = build_synthetic_sim(kmf, tau_mesh, beta=1000.0)
    print(f"  Chemical potential μ = {sim_data['mu']:.6f} Ha")

    print(f"\n--- Computing P̃(iΩ) (periodic, q=0) ---")
    print("  (This may take a moment for ntau=144 and nk>1 ...)")
    P_iw = compute_tildeP_iw(
        sim_data['G_tau'], VQ_ao,
        beta=1000.0, ir_file=ir_dest
    )
    print(f"  P̃(iΩ) shape: {P_iw.shape}  (niw, ns, 1, NQ, NQ)")

    print(f"\n--- Writing output files to {outdir}/ ---")
    write_mean_field_input(os.path.join(outdir, 'mean_field_input.h5'), kmf)
    write_sim(os.path.join(outdir, 'sim.h5'), sim_data)
    write_VQ(os.path.join(outdir, 'df_hf_int'), VQ_ao)
    pi_path = os.path.join(outdir, 'p_iw_tilde.h5')
    write_tildeP_iw(pi_path, P_iw)

    # Write a ready-to-use run script
    run_sh = os.path.join(outdir, 'bse_periodic_singlet.sh')
    with open(run_sh, 'w') as f:
        f.write(f"""#!/bin/bash
# Auto-generated by generate_periodic_test.py
# System: 1-D H2 chain, nk={nk}, basis={args.basis}
# Uses precomputed P̃(iΩ) to avoid the molecular readVQFromMeta path.

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
    os.chmod(run_sh, 0o755)
    print(f"  Wrote run script → {run_sh}")

    print(f"""
=== Done ===
To run the periodic BSE solver:

  cd {outdir}
  bash bse_periodic_singlet.sh

Or from the project root:

  python script/solveCasida_periodic.py \\
      --type singlet --beta 1000 --qpac 0 --calc_pi 0 --n_jobs -1 \\
      --input    {outdir}/mean_field_input.h5 \\
      --sim      {outdir}/sim.h5 \\
      --int_path {outdir}/df_hf_int/ \\
      --ir_file  {outdir}/irgrid/1e5.h5 \\
      --pi_file  {outdir}/p_iw_tilde.h5 \\
      --output   {outdir}/bse_periodic_singlet.h5
""")


if __name__ == '__main__':
    main()
