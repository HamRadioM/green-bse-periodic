#!/usr/bin/env python3
"""
generate_Hchain_atomic.py
=========================
Generate test inputs for the equidistant hydrogen ATOMIC chain:
one H atom per unit cell, STO-3G basis, k-mesh Nk × 1 × 1.

Unlike the H2 (dimerized) chain, this system is a half-filled 1D metal.
Its non-interacting polarizability P⁰(q, ω) is the 1D Lindhard function,
which has a clear q-dispersion with a logarithmic divergence at ω = v_F q
and a 2k_F Kohn anomaly.  This makes it ideal for testing the finite-q
BSE heatmap.

Output directory layout
-----------------------
  <outdir>/
    mean_field_input.h5
    sim.h5
    irgrid/1e5.h5
    df_hf_int/
      VQ_0.h5
      meta.h5
    df_hf_int_fq/
      VQ_0.h5   (copy)
      VQ_q1.h5 … VQ_q{nk-1}.h5
      meta.h5

Usage
-----
  python script/generate_Hchain_atomic.py --nk 8 \\
      --outdir example/H_atomic_fq_8k

  python script/generate_Hchain_atomic.py --nk 16 \\
      --outdir example/H_atomic_fq_16k

Then plot the dispersion:
  python script/plot_Pbse_dispersion.py \\
      --q0_dir example/H_atomic_fq_16k \\
      --fq_dir example/H_atomic_fq_8k  \\
      --output H_atomic_dispersion.pdf
"""

import argparse
import os
import sys
import shutil

import h5py
import numpy as np

try:
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc import scf as pbcscf
    from pyscf.pbc import df  as pbcdf
except ImportError:
    raise SystemExit("PySCF not found.  Install with:  pip install pyscf")


# ---------------------------------------------------------------------------
# Storage helpers  (same conventions as generate_finite_q_test.py)
# ---------------------------------------------------------------------------

def _vq_to_storage(VQ_ao):
    """(nk, NQ, nao, nao) complex → (nk, NQ, nao, 2*nao) float  interleaved."""
    nk, NQ, nao, _ = VQ_ao.shape
    out = np.zeros((nk, NQ, nao, 2 * nao), dtype=np.float64)
    out[:, :, :, 0::2] = VQ_ao.real
    out[:, :, :, 1::2] = VQ_ao.imag
    return out


def _get_tau_grid(ir_file, beta=1000.0):
    with h5py.File(ir_file, 'r') as f:
        xgrid = f['fermi/xgrid'][()]
    tau_mesh = np.zeros(xgrid.shape[0] + 2)
    tau_mesh[0]    = 0.0
    tau_mesh[1:-1] = (xgrid + 1) * beta / 2.0
    tau_mesh[-1]   = beta
    return len(tau_mesh), xgrid, tau_mesh


# ---------------------------------------------------------------------------
# Mean-field: equidistant H atomic chain
# ---------------------------------------------------------------------------

def run_krhf_atomic(nk, lattice_const=3.0, basis='sto-3g', verbose=0):
    """
    k-RHF on a 1-D equidistant H chain: 1 H atom per unit cell.

    The chain runs along x so that make_kpts([nk,1,1]) samples the correct
    chain BZ and k_mesh_scaled[:,0] gives the fractional q-coordinates.

    Parameters
    ----------
    nk            : number of k-points
    lattice_const : H–H spacing in Bohr (default 3.0)
    basis         : PySCF basis string
    verbose       : PySCF verbosity level

    Returns
    -------
    cell, kmf, mydf
    """
    cell = pbcgto.Cell()
    cell.atom    = 'H 0 0 0'          # single H at origin; chain along x
    cell.basis   = basis
    cell.a       = np.diag([lattice_const, 20.0, 20.0])   # chain along x
    cell.unit    = 'B'
    cell.verbose = verbose
    cell.build()

    kpts = cell.make_kpts([nk, 1, 1])  # nk points along x (chain BZ)

    mydf = pbcdf.GDF(cell, kpts)
    mydf.build()

    kmf = pbcscf.KRHF(cell, kpts)
    kmf.with_df = mydf
    kmf.init_guess = 'atom'
    kmf.kernel()

    return cell, kmf, mydf


# ---------------------------------------------------------------------------
# Extract VQ integrals (identical logic to generate_finite_q_test.py)
# ---------------------------------------------------------------------------

def extract_VQ(cell, mydf, kpts):
    """Same-k integrals: (nk, NQ, nao, nao)."""
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


def extract_VQ_kq(cell, mydf, kpts, q_idx):
    """Off-diagonal integrals ⟨μ(k)|V_Q|ν(k+q)⟩: (nk, NQ, nao, nao)."""
    nk  = len(kpts)
    nao = cell.nao_nr()
    NQ  = mydf.get_naoaux()
    VQ  = np.zeros((nk, NQ, nao, nao), dtype=np.complex128)
    for ik in range(nk):
        ikq = (ik + q_idx) % nk
        col = 0
        for Lpq in mydf.sr_loop(kpti_kptj=np.array([kpts[ik], kpts[ikq]]),
                                 compact=False):
            Lpq  = Lpq[0] + 1j * Lpq[1]
            naux = Lpq.shape[0]
            VQ[ik, col:col + naux] = Lpq.reshape(naux, nao, nao)
            col += naux
    return VQ


# ---------------------------------------------------------------------------
# Synthetic Green's function from k-RHF
# ---------------------------------------------------------------------------

def build_synthetic_sim(kmf, tau_mesh, beta=1000.0):
    nk   = len(kmf.kpts)
    nao  = kmf.cell.nao_nr()
    ns   = 1
    ntau = len(tau_mesh)

    vhf_k  = kmf.get_veff()
    Sigma1 = np.zeros((ns, nk, nao, nao), dtype=np.complex128)
    for ik in range(nk):
        Sigma1[0, ik] = vhf_k[ik]

    decay     = np.exp(-np.abs(tau_mesh - beta / 2) * 0.005)
    Sigma_tau = np.zeros((ntau, ns, nk, nao, nao), dtype=np.complex128)
    for itau in range(ntau):
        for ik in range(nk):
            Sigma_tau[itau, 0, ik] = Sigma1[0, ik] * decay[itau] * 0.01

    mo_energy  = np.array(kmf.mo_energy)
    nel_cell   = int(kmf.cell.nelectron)
    nk_        = mo_energy.shape[0]
    nfilled    = nk_ * (nel_cell // 2)   # total filled orbitals across BZ
    sorted_eps = np.sort(mo_energy.ravel())
    if 0 < nfilled < len(sorted_eps):
        mu = 0.5 * (sorted_eps[nfilled - 1] + sorted_eps[nfilled])
    else:
        mu = float(np.mean(sorted_eps))

    G_tau = np.zeros((ntau, ns, nk, nao, nao), dtype=np.complex128)
    for itau in range(ntau):
        tau = tau_mesh[itau]
        for ik in range(nk):
            eps    = mo_energy[ik] - mu
            g_diag = np.empty_like(eps)
            for ib, ep in enumerate(eps):
                if ep >= 0:
                    g_diag[ib] = -np.exp(-np.log1p(np.exp(-beta * ep)) - tau * ep)
                else:
                    g_diag[ib] = -np.exp(-np.log1p(np.exp(beta * ep)) + (beta - tau) * ep)
            G_tau[itau, 0, ik] = np.diag(g_diag)

    return {'Sigma1': Sigma1, 'Sigma_tau': Sigma_tau,
            'G_tau': G_tau, 'mu': mu, 'tau_mesh': tau_mesh}


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def write_mean_field_input(path, kmf):
    import json
    cell = kmf.cell
    nk   = len(kmf.kpts)
    nao  = cell.nao_nr()
    ns   = 1
    NQ   = kmf.with_df.get_naoaux()
    kpts_scaled = cell.get_scaled_kpts(kmf.kpts)
    k_idx       = np.arange(nk, dtype=np.int64)

    def _prep(arr_nk):
        arr = np.array(arr_nk, dtype=np.complex128)
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
        f.create_dataset('grid/k_mesh_scaled',    data=kpts_scaled.astype(np.float64))
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

        nat = cell.natm
        nao_per_atom = nao // nat
        f.create_dataset('mulliken/Zs',
                         data=np.array([cell.atom_charge(i) for i in range(nat)],
                                       dtype=np.int32))
        f.create_dataset('mulliken/last_ao',
                         data=np.array([(i+1)*nao_per_atom for i in range(nat)],
                                       dtype=np.int64))

        b = cell.basis
        bname = b if isinstance(b, str) else 'custom'
        f.create_dataset('Cell',
                         data=json.dumps({'atom': str(cell.atom),
                                          'basis': bname}).encode())
    print(f'  Wrote {path}')


def write_sim(path, sim_data, iter_num=22):
    with h5py.File(path, 'w') as f:
        f.create_dataset('iter', data=np.int64(iter_num))
        for it in [1, iter_num]:
            g = f'iter{it}'
            f.create_dataset(f'{g}/mu',                data=float(sim_data['mu']))
            f.create_dataset(f'{g}/Sigma1',            data=sim_data['Sigma1'].astype(np.complex128))
            f.create_dataset(f'{g}/Energy_1b',         data=0.0)
            f.create_dataset(f'{g}/Energy_2b',         data=0.0)
            f.create_dataset(f'{g}/Energy_HF',         data=0.0)
            f.create_dataset(f'{g}/Selfenergy/data',   data=sim_data['Sigma_tau'].astype(np.complex128))
            f.create_dataset(f'{g}/Selfenergy/mesh',   data=sim_data['tau_mesh'].astype(np.float64))
            f.create_dataset(f'{g}/G_tau/data',        data=sim_data['G_tau'].astype(np.complex128))
            f.create_dataset(f'{g}/G_tau/mesh',        data=sim_data['tau_mesh'].astype(np.float64))
    print(f'  Wrote {path}')


def write_VQ(int_path, VQ_ao, fname='VQ_0.h5'):
    os.makedirs(int_path, exist_ok=True)
    nk   = VQ_ao.shape[0]
    path = os.path.join(int_path, fname)
    with h5py.File(path, 'w') as f:
        f.create_dataset('0', data=_vq_to_storage(VQ_ao))
    print(f'  Wrote {path}')
    meta = os.path.join(int_path, 'meta.h5')
    if not os.path.exists(meta):
        with h5py.File(meta, 'w') as f:
            f.create_dataset('chunk_size',    data=np.int64(nk))
            f.create_dataset('chunk_indices', data=np.arange(nk, dtype=np.int64))
        print(f'  Wrote {meta}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate equidistant H atomic chain inputs for BSE finite-q.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nk',       type=int,   default=8,
                        help='Number of k-points.')
    parser.add_argument('--lattice',  type=float, default=3.0,
                        help='H–H lattice constant (Bohr).')
    parser.add_argument('--basis',    type=str,   default='sto-3g')
    parser.add_argument('--beta',     type=float, default=1000.0,
                        help='Inverse temperature (a.u.).')
    parser.add_argument('--outdir',   type=str,   default=None,
                        help='Output directory (default: example/H_atomic_fq_{nk}k).')
    parser.add_argument('--ir_source', type=str,
                        default='example/N2_STO3G/irgrid/1e5.h5',
                        help='IR grid file to copy.')
    args = parser.parse_args()

    nk     = args.nk
    outdir = args.outdir or f'example/H_atomic_fq_{nk}k'
    os.makedirs(outdir, exist_ok=True)

    # IR grid
    ir_dir  = os.path.join(outdir, 'irgrid')
    os.makedirs(ir_dir, exist_ok=True)
    ir_dest = os.path.join(ir_dir, '1e5.h5')
    if not os.path.exists(args.ir_source):
        raise FileNotFoundError(f'IR source not found: {args.ir_source}')
    shutil.copy2(args.ir_source, ir_dest)
    print(f'  Copied IR grid → {ir_dest}')

    _, _, tau_mesh = _get_tau_grid(ir_dest, beta=args.beta)

    # Mean-field
    print(f'\n--- k-RHF on equidistant H chain: nk={nk}, a={args.lattice} Bohr ---')
    cell, kmf, mydf = run_krhf_atomic(
        nk=nk, lattice_const=args.lattice, basis=args.basis, verbose=3)

    nao = cell.nao_nr()
    print(f'\nSystem: nao={nao} (1 H/cell), nk={nk}')
    print(f'HF total energy: {kmf.e_tot:.6f} Ha')
    ksc = cell.get_scaled_kpts(kmf.kpts)
    for ik in range(nk):
        print(f'  k[{ik}] (kx={ksc[ik,0]:.3f})  ε = {np.array(kmf.mo_energy[ik])} Ha')

    # Same-k VQ
    print(f'\n--- Extracting same-k VQ (nk={nk}) ---')
    VQ_kk = extract_VQ(cell, mydf, kmf.kpts)
    print(f'  VQ_kk shape: {VQ_kk.shape}')

    # Synthetic sim
    print(f'\n--- Building synthetic Green\'s function ---')
    sim_data = build_synthetic_sim(kmf, tau_mesh, beta=args.beta)
    print(f'  μ = {sim_data["mu"]:.6f} Ha')

    # Write standard files
    print(f'\n--- Writing to {outdir}/ ---')
    write_mean_field_input(os.path.join(outdir, 'mean_field_input.h5'), kmf)
    write_sim(os.path.join(outdir, 'sim.h5'), sim_data)

    int_q0 = os.path.join(outdir, 'df_hf_int')
    write_VQ(int_q0, VQ_kk, fname='VQ_0.h5')

    int_fq = os.path.join(outdir, 'df_hf_int_fq')
    os.makedirs(int_fq, exist_ok=True)
    shutil.copy2(os.path.join(int_q0, 'VQ_0.h5'),  os.path.join(int_fq, 'VQ_0.h5'))
    shutil.copy2(os.path.join(int_q0, 'meta.h5'),  os.path.join(int_fq, 'meta.h5'))
    print(f'  Copied VQ_0.h5, meta.h5 → {int_fq}/')

    # All finite-q integrals  q_idx = 1 … nk-1
    for q_idx in range(1, nk):
        print(f'\n--- Extracting VQ_q{q_idx}  (q_x = {q_idx}/{nk}) ---')
        VQ_kq = extract_VQ_kq(cell, mydf, kmf.kpts, q_idx)
        write_VQ(int_fq, VQ_kq, fname=f'VQ_q{q_idx}.h5')

    print(f'''
=== Done ===
Generated files in {outdir}/

To plot P⁰ and P_BSE dispersion:
  python script/plot_Pbse_dispersion.py \\
      --q0_dir {outdir} \\
      --fq_dir {outdir} \\
      --output H_atomic_dispersion.pdf
''')


if __name__ == '__main__':
    main()
