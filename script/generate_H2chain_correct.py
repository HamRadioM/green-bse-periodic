#!/usr/bin/env python3
"""
generate_H2chain_correct.py
===========================
Generate test inputs for a 1-D H₂ molecular chain with the chain axis
along x, so that make_kpts([nk,1,1]) correctly samples the 1D Brillouin
zone.  All q-indices (0 … nk-1) are produced in a single run.

The existing H2_fq_8k example was generated with the chain along z but
the k-mesh along x, meaning all k-points were at k_z=0 (the Γ-point of
the chain BZ).  That is why P⁰ showed no q-dispersion.

Here the cell is oriented as:
  H at (0, 0, 0)  and  H at (bond, 0, 0)   ← along x
  a = diag([lattice, 20, 20])               ← chain along x
  kpts = make_kpts([nk, 1, 1])             ← samples the chain BZ ✓

Tuning lattice_const controls band dispersion:
  - large lattice  (> 4 Bohr): well-separated H₂ molecules,
    narrow bands → flat P⁰(q)
  - small lattice  (~ 2.5 Bohr): strongly hybridised chain,
    wide bands → clear q-dependent P⁰(q, ω)

Usage
-----
  # Default: 8x1x1 k-mesh, lattice=3.0, bond=1.4, sto-3g
  python script/generate_H2chain_correct.py --nkx 8

  # 3D k-mesh (e.g. for a 2D or 3D material)
  python script/generate_H2chain_correct.py --nkx 8 --nky 4 --nkz 4

  # Better basis
  python script/generate_H2chain_correct.py --nkx 16 --lattice 6.0 \\
      --basis cc-pvdz --outdir example/H2_ccpvdz_fq_16k

Then plot:
  python script/plot_Pbse_dispersion.py \\
      --q0_dir example/H2_ccpvdz_fq_16k \\
      --fq_dir example/H2_ccpvdz_fq_16k \\
      --output H2_ccpvdz_dispersion.pdf
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

sys.path.insert(0, os.path.dirname(__file__))
from gen_utils import (
    _vq_to_storage, _get_tau_grid, build_kq_map,
    extract_VQ, extract_VQ_kq,
    build_synthetic_sim,
    write_mean_field_input, write_sim, write_VQ,
)


# ---------------------------------------------------------------------------
# Mean-field: H₂ chain along x
# ---------------------------------------------------------------------------

def run_krhf(kmesh, bond=1.4, lattice=3.0, basis='sto-3g', verbose=0):
    """
    k-RHF on a 1-D H₂ chain with the chain running along x.

    Unit cell: H at (0,0,0) and H at (bond,0,0).
    Lattice: a = diag([lattice, 20, 20]).
    k-mesh: make_kpts(kmesh) where kmesh = [nkx, nky, nkz].

    Parameters
    ----------
    kmesh   : [nkx, nky, nkz] k-point grid
    bond    : H-H intramolecular bond length (Bohr)
    lattice : unit cell length along x (Bohr); must be > bond
    """
    if lattice <= bond:
        raise ValueError(f'lattice ({lattice}) must exceed bond ({bond})')

    cell = pbcgto.Cell()
    cell.atom    = f'H 0 0 0; H {bond} 0 0'   # H₂ along x
    cell.basis   = basis
    cell.a       = np.diag([lattice, 20.0, 20.0])
    cell.unit    = 'B'
    cell.verbose = verbose
    cell.build()

    kpts = cell.make_kpts(kmesh)

    mydf = pbcdf.GDF(cell, kpts)
    mydf.build()

    kmf = pbcscf.KRHF(cell, kpts)
    kmf.with_df = mydf
    kmf.init_guess = 'atom'
    kmf.kernel()

    return cell, kmf, mydf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate H₂ chain (axis along x) inputs for BSE finite-q.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nkx',     type=int,   default=8,
                        help='k-points along x (chain axis).')
    parser.add_argument('--nky',     type=int,   default=1,
                        help='k-points along y.')
    parser.add_argument('--nkz',     type=int,   default=1,
                        help='k-points along z.')
    parser.add_argument('--bond',    type=float, default=1.4,
                        help='H-H intramolecular bond length (Bohr).')
    parser.add_argument('--lattice', type=float, default=3.0,
                        help='Unit cell length along x (Bohr); must exceed --bond.')
    parser.add_argument('--basis',   type=str,   default='sto-3g')
    parser.add_argument('--beta',    type=float, default=1000.0)
    parser.add_argument('--outdir',  type=str,   default=None)
    parser.add_argument('--ir_source', type=str,
                        default='example/N2_STO3G/irgrid/1e5.h5')
    args = parser.parse_args()

    kmesh  = [args.nkx, args.nky, args.nkz]
    nk     = args.nkx * args.nky * args.nkz
    outdir = args.outdir or f'example/H2_correct_fq_{args.nkx}x{args.nky}x{args.nkz}k'
    os.makedirs(outdir, exist_ok=True)

    # IR grid
    ir_dir  = os.path.join(outdir, 'irgrid')
    os.makedirs(ir_dir, exist_ok=True)
    ir_dest = os.path.join(ir_dir, '1e5.h5')
    if not os.path.exists(args.ir_source):
        raise FileNotFoundError(f'IR source not found: {args.ir_source}')
    shutil.copy2(args.ir_source, ir_dest)

    _, _, tau_mesh = _get_tau_grid(ir_dest, beta=args.beta)

    # Mean-field
    print(f'\n--- k-RHF on H₂ chain along x: kmesh={kmesh}, '
          f'bond={args.bond}, lattice={args.lattice} Bohr ---')
    cell, kmf, mydf = run_krhf(
        kmesh=kmesh, bond=args.bond, lattice=args.lattice,
        basis=args.basis, verbose=3)

    nao = cell.nao_nr()
    ksc = cell.get_scaled_kpts(kmf.kpts)
    print(f'\nSystem: nao={nao}/cell (2 H/cell), nk={nk} ({kmesh})')
    print(f'HF total energy: {kmf.e_tot:.6f} Ha')
    print(f'inter-molecular gap: {args.lattice - args.bond:.2f} Bohr')
    for ik in range(nk):
        eps = np.array(kmf.mo_energy[ik])
        print(f'  k[{ik}] (k={ksc[ik]})  ε = {eps} Ha')

    # Same-k VQ
    print(f'\n--- Extracting same-k VQ ---')
    # VQ_kk = extract_VQ(cell, mydf, kmf.kpts)
    # print(f'  shape: {VQ_kk.shape}  (nk, NQ, nao, nao)')

    # Green's function
    print(f'\n--- Building G(τ) ---')
    sim = build_synthetic_sim(kmf, tau_mesh, beta=args.beta)
    print(f'  μ = {sim["mu"]:.6f} Ha')

    # Quick sanity: Fermi occupancies at k=Γ from MO eigenvalues
    nel_cell = int(cell.nelectron)
    nfilled_per_k = nel_cell // 2
    eps0 = np.array(kmf.mo_energy[0]) - sim['mu']
    nF0  = 1.0 / (np.exp(np.clip(args.beta * eps0, -500.0, 500.0)) + 1.0)
    print(f'  k=Γ nF (should be 1 for filled, 0 for empty): {np.round(nF0, 3)}'
          f'  ({nfilled_per_k} filled)')

    # Write standard files
    print(f'\n--- Writing output to {outdir}/ ---')
    write_mean_field_input(os.path.join(outdir, 'mean_field_input.h5'), kmf)
    write_sim(os.path.join(outdir, 'sim.h5'), sim)

    int_q0 = os.path.join(outdir, 'df_hf_int')
    # write_VQ(int_q0, VQ_kk, 'VQ_0.h5')

    int_fq = os.path.join(outdir, 'df_hf_int_fq')
    os.makedirs(int_fq, exist_ok=True)
    # shutil.copy2(os.path.join(int_q0, 'VQ_0.h5'),  os.path.join(int_fq, 'VQ_0.h5'))
    # shutil.copy2(os.path.join(int_q0, 'meta.h5'),  os.path.join(int_fq, 'meta.h5'))

    # All finite-q
    kq_map = build_kq_map(cell, kmf.kpts)
    for q_idx in range(0, nk):
        print(f'\n--- VQ_q{q_idx}  (q = {ksc[q_idx]}) ---')
        VQ_kq = extract_VQ_kq(cell, mydf, kmf.kpts, q_idx, kq_map=kq_map)
        write_VQ(int_fq, VQ_kq, f'VQ_q{q_idx}.h5')

    print(f'''
=== Done ===
Files in {outdir}/

Plot dispersion:
  python script/plot_Pbse_dispersion.py \\
      --q0_dir {outdir} \\
      --fq_dir {outdir} \\
      --output H2_correct_dispersion.pdf
''')


if __name__ == '__main__':
    main()
