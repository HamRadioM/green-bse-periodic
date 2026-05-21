"""
gen_utils.py
============
Shared utility functions for BSE synthetic-data generators.
Used by generate_H2chain_correct.py, generate_hBN.py, etc.
"""

import os
import json

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _vq_to_storage(VQ):
    """(nk, NQ, nao, nao) complex → (nk, NQ, nao, 2*nao) float interleaved."""
    nk, NQ, nao, _ = VQ.shape
    out = np.zeros((nk, NQ, nao, 2 * nao), dtype=np.float64)
    out[:, :, :, 0::2] = VQ.real
    out[:, :, :, 1::2] = VQ.imag
    return out


def _get_tau_grid(ir_file, beta=1000.0):
    with h5py.File(ir_file, 'r') as f:
        xgrid = f['fermi/xgrid'][()]
    tau = np.zeros(xgrid.shape[0] + 2)
    tau[0]    = 0.0
    tau[1:-1] = (xgrid + 1) * beta / 2.0
    tau[-1]   = beta
    return len(tau), xgrid, tau


def build_kq_map(cell, kpts, tol=1e-10):
    """
    Build mapping:

        kq_map[ik, iq] = jk

    such that

        kpts[jk] = kpts[ik] + kpts[iq] + G

    modulo reciprocal lattice vectors.
    """
    nk = len(kpts)

    scaled = cell.get_scaled_kpts(kpts)
    kq_map = np.empty((nk, nk), dtype=np.int32)

    for ik in range(nk):
        for iq in range(nk):

            target = scaled[ik] + scaled[iq]
            target -= np.floor(target)

            found = False

            for jk in range(nk):

                diff = target - scaled[jk]
                diff -= np.round(diff)

                if np.linalg.norm(diff) < tol:
                    kq_map[ik, iq] = jk
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    f"Could not find k+q match for ik={ik}, iq={iq}"
                )

    return kq_map

# ---------------------------------------------------------------------------
# VQ extraction
# ---------------------------------------------------------------------------

def extract_VQ(cell, mydf, kpts):
    """Same-k density-fitting integrals: (nk, NQ, nao, nao) complex."""
    nk  = len(kpts)
    nao = cell.nao_nr()
    NQ  = mydf.get_naoaux()
    VQ  = np.zeros((nk, NQ, nao, nao), dtype=np.complex128)
    for ik, kpt in enumerate(kpts):
        col = 0
        for Lpq in mydf.sr_loop(kpti_kptj=np.array([kpt, kpt]), compact=False):
            Lpq  = Lpq[0] + 1j * Lpq[1]
            naux = Lpq.shape[0]
            VQ[ik, col:col + naux] = Lpq.reshape(naux, nao, nao)
            col += naux
    return VQ


def extract_VQ_kq(cell, mydf, kpts, q_idx, kq_map):
    """Off-diagonal (k, k+q) density-fitting integrals: (nk, NQ, nao, nao)."""
    nk  = len(kpts)
    nao = cell.nao_nr()
    NQ  = mydf.get_naoaux()
    VQ  = np.zeros((nk, NQ, nao, nao), dtype=np.complex128)
    print(f"Extracting VQ for q_idx={q_idx} (q={cell.get_scaled_kpts([kpts[q_idx]])[0]})...")
    print(f"  kpts shape: {kpts.shape}, VQ shape: {VQ.shape}")
    for ik in range(nk):
        ikq = kq_map[ik, q_idx]
        col = 0  # Debug print to verify k and k+q points
        print(f"  ik={ik} (k={cell.get_scaled_kpts([kpts[ik]])[0]}), "
              f"  ikq={ikq} (k+q={cell.get_scaled_kpts([kpts[ikq]])[0]})")
        for Lpq in mydf.sr_loop(kpti_kptj=np.array([kpts[ik], kpts[ikq]]),
                                 compact=False):
            Lpq  = Lpq[0] + 1j * Lpq[1]
            naux = Lpq.shape[0]
            VQ[ik, col:col + naux] = Lpq.reshape(naux, nao, nao)
            col += naux
    return VQ


# ---------------------------------------------------------------------------
# Green's function from k-RHF (numerically stable at low T)
# ---------------------------------------------------------------------------

def build_synthetic_sim(kmf, tau_mesh, beta=1000.0):
    """
    Build a physically correct non-interacting G(τ) from k-RHF eigenvalues.

    The numerically stable formula for orbital i at k-point k is:

        g_i(τ) = -(1 - n_F(ε_i)) * exp(-τ * ε_i)        0 < τ ≤ β/2
        g_i(τ) = -(1 - n_F(ε_i)) * exp(-(τ-β) * ε_i)    β/2 < τ < β

    with ε_i = ε_i^k - μ.
    """
    nk   = len(kmf.kpts)
    nao  = kmf.cell.nao_nr()
    ns   = 1
    ntau = len(tau_mesh)

    mo_energy  = np.array(kmf.mo_energy)   # (nk, nao)
    nel_cell   = int(kmf.cell.nelectron)
    nk_        = mo_energy.shape[0]
    nfilled    = nk_ * (nel_cell // 2)
    all_eps    = np.sort(mo_energy.ravel())
    mu = 0.5 * (all_eps[nfilled - 1] + all_eps[nfilled])

    vhf_k  = kmf.get_veff()
    Sigma1 = np.zeros((ns, nk, nao, nao), dtype=np.complex128)
    for ik in range(nk):
        Sigma1[0, ik] = vhf_k[ik]

    G_tau = np.zeros((ntau, ns, nk, nao, nao), dtype=np.complex128)
    for ik in range(nk):
        eps = mo_energy[ik] - mu
        C = np.array(kmf.mo_coeff[ik], dtype=np.complex128)
        for itau in range(ntau):
            tau  = tau_mesh[itau]
            g    = np.empty_like(eps)
            for ib, ep in enumerate(eps):
                if ep >= 0:
                    g[ib] = -np.exp(-np.log1p(np.exp(-beta * ep)) - tau * ep)
                else:
                    g[ib] = -np.exp(-np.log1p(np.exp(beta * ep)) + (beta - tau) * ep)
            G_tau[itau, 0, ik] = C @ np.diag(g) @ C.conj().T

    decay     = np.exp(-np.abs(tau_mesh - beta / 2) * 0.005)
    Sigma_tau = np.zeros((ntau, ns, nk, nao, nao), dtype=np.complex128)
    for itau in range(ntau):
        for ik in range(nk):
            Sigma_tau[itau, 0, ik] = Sigma1[0, ik] * decay[itau] * 0.01

    return {'Sigma1': Sigma1, 'Sigma_tau': Sigma_tau,
            'G_tau': G_tau, 'mu': mu, 'tau_mesh': tau_mesh}


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_mean_field_input(path, kmf):
    cell  = kmf.cell
    nk    = len(kmf.kpts)
    nao   = cell.nao_nr()
    ns    = 1
    NQ    = kmf.with_df.get_naoaux()
    ksc   = cell.get_scaled_kpts(kmf.kpts)
    k_idx = np.arange(nk, dtype=np.int64)

    def _prep(arr):
        arr = np.array(arr, dtype=np.complex128)
        out = np.zeros((ns, nk, nao, nao, 2), dtype=np.float64)
        out[0, :, :, :, 0] = arr.real
        out[0, :, :, :, 1] = arr.imag
        return out

    with h5py.File(path, 'w') as f:
        f.create_dataset('HF/Fock-k',    data=_prep(np.array(kmf.get_fock())))
        f.create_dataset('HF/H-k',       data=_prep(np.array(kmf.get_hcore())))
        f.create_dataset('HF/S-k',       data=_prep(np.array(kmf.get_ovlp())))
        f.create_dataset('HF/Energy',    data=float(kmf.e_tot))
        f.create_dataset('HF/Energy_nuc',data=float(cell.energy_nuc()))
        f.create_dataset('HF/madelung',  data=0.0)
        f.create_dataset('HF/Nk',        data=np.int64(nk))
        f.create_dataset('HF/nk',        data=np.int64(nk))
        f.create_dataset('HF/mo_energy', data=np.array(kmf.mo_energy[0], dtype=np.float64))
        f.create_dataset('HF/mo_coeff',  data=np.array(kmf.mo_coeff[0].real, dtype=np.float64))

        f.create_dataset('params/nao',      data=np.int64(nao))
        f.create_dataset('params/nel_cell', data=np.int64(int(cell.nelectron)))
        f.create_dataset('params/nk',       data=np.int64(nk))
        f.create_dataset('params/ns',       data=np.int64(ns))
        f.create_dataset('params/nso',      data=np.int64(nao))
        f.create_dataset('params/NQ',       data=np.int64(NQ))

        f.create_dataset('grid/nk',               data=np.int64(nk))
        f.create_dataset('grid/ink',              data=np.int64(nk))
        f.create_dataset('grid/k_mesh',           data=kmf.kpts.astype(np.float64))
        f.create_dataset('grid/k_mesh_scaled',    data=ksc.astype(np.float64))
        f.create_dataset('grid/index',            data=k_idx)
        f.create_dataset('grid/ir_list',          data=k_idx)
        f.create_dataset('grid/weight',           data=np.ones(nk) / nk)
        f.create_dataset('grid/conj_list',        data=k_idx)
        f.create_dataset('grid/conj_pairs_list',  data=k_idx)
        f.create_dataset('grid/trans_pairs_list', data=k_idx)
        f.create_dataset('grid/kpair_idx',
                         data=np.column_stack([k_idx, k_idx]).astype(np.int64))
        f.create_dataset('grid/kpair_irre_list',  data=k_idx)
        f.create_dataset('grid/num_kpair_stored', data=np.int64(nk))
        f.create_dataset('grid/kq_map',  data=build_kq_map(cell, kmf.kpts))

        nat = cell.natm
        nao_per = nao // nat
        f.create_dataset('mulliken/Zs',
                         data=np.array([cell.atom_charge(i) for i in range(nat)],
                                       dtype=np.int32))
        f.create_dataset('mulliken/last_ao',
                         data=np.array([(i+1)*nao_per for i in range(nat)],
                                       dtype=np.int64))
        b = cell.basis
        f.create_dataset('Cell',
                         data=json.dumps({'atom': str(cell.atom),
                                          'basis': b if isinstance(b, str) else 'custom'
                                          }).encode())
    print(f'  Wrote {path}')


def write_sim(path, sim, iter_num=22):
    with h5py.File(path, 'w') as f:
        f.create_dataset('iter', data=np.int64(iter_num))
        for it in [1, iter_num]:
            g = f'iter{it}'
            f.create_dataset(f'{g}/mu',              data=float(sim['mu']))
            f.create_dataset(f'{g}/Sigma1',          data=sim['Sigma1'].astype(np.complex128))
            f.create_dataset(f'{g}/Energy_1b',       data=0.0)
            f.create_dataset(f'{g}/Energy_2b',       data=0.0)
            f.create_dataset(f'{g}/Energy_HF',       data=0.0)
            f.create_dataset(f'{g}/Selfenergy/data', data=sim['Sigma_tau'].astype(np.complex128))
            f.create_dataset(f'{g}/Selfenergy/mesh', data=sim['tau_mesh'].astype(np.float64))
            f.create_dataset(f'{g}/G_tau/data',      data=sim['G_tau'].astype(np.complex128))
            f.create_dataset(f'{g}/G_tau/mesh',      data=sim['tau_mesh'].astype(np.float64))
    print(f'  Wrote {path}')


def write_VQ(int_path, VQ, fname='VQ_0.h5'):
    os.makedirs(int_path, exist_ok=True)
    nk = VQ.shape[0]
    with h5py.File(os.path.join(int_path, fname), 'w') as f:
        f.create_dataset('0', data=_vq_to_storage(VQ))
    print(f'  Wrote {os.path.join(int_path, fname)}')
    meta = os.path.join(int_path, 'meta.h5')
    if not os.path.exists(meta):
        with h5py.File(meta, 'w') as f:
            f.create_dataset('chunk_size',    data=np.int64(nk))
            f.create_dataset('chunk_indices', data=np.arange(nk, dtype=np.int64))
        print(f'  Wrote {meta}')
