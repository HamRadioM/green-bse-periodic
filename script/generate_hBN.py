#!/usr/bin/env python3
"""
generate_hBN.py
===============
Generate test inputs for hexagonal boron nitride (hBN) monolayer using
PySCF periodic k-RHF + Gaussian density fitting.

Unit cell (Bohr):
  a1 = (a, 0, 0)
  a2 = (a/2, a√3/2, 0)
  a3 = (0, 0, vacuum)          ← out-of-plane (2D slab geometry)
  B at (0, 0, 0)
  N at fractional (1/3, 1/3, 0) = Cartesian (a/2, a√3/6, 0)
  B-N bond = a/√3

Default lattice constant: a = 4.733 Bohr (2.504 Å).
nel_cell = 12 (B: 5e, N: 7e); STO-3G → nao = 10, nfilled = 6.

Usage
-----
  # 4×4×1 k-mesh (default)
  python script/generate_hBN.py

  # Custom k-mesh and basis
  python script/generate_hBN.py --nkx 6 --nky 6 \\
      --basis sto-3g --outdir example/hBN_sto3g_6x6k

Then plot dispersion:
  python script/plot_Pbse_dispersion.py \\
      --q0_dir example/hBN_sto3g_4x4k \\
      --fq_dir example/hBN_sto3g_4x4k \\
      --output hBN_dispersion.pdf

Plot band structure:
  python script/plot_bandstructure.py \\
      --input example/hBN_sto3g_4x4k/mean_field_input.h5 \\
      --sim   example/hBN_sto3g_4x4k/sim.h5 \\
      --output hBN_bands.pdf
"""

import argparse
import os
import shutil
import sys

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

try:
    from pyscf.pbc import gto as pbcgto
    from pyscf.pbc import scf as pbcscf
    from pyscf.pbc import df  as pbcdf
except ImportError:
    raise SystemExit("PySCF not found.  Install with:  pip install pyscf")

from gen_utils import (
    _get_tau_grid,
    extract_VQ,
    extract_VQ_kq,
    build_synthetic_sim,
    write_mean_field_input,
    write_sim,
    write_VQ,
)


# ---------------------------------------------------------------------------
# hBN mean-field
# ---------------------------------------------------------------------------

def run_krhf_hBN(kmesh, a=4.733, vacuum=20.0, basis='sto-3g', verbose=0):
    """
    k-RHF on an hBN monolayer in the xy-plane.

    Lattice vectors (Bohr):
      a1 = (a,   0,       0)
      a2 = (a/2, a√3/2,   0)
      a3 = (0,   0,   vacuum)

    Basis positions:
      B at (0,    0,         0)
      N at (a/2,  a√3/6,     0)   [fractional (1/3, 1/3, 0)]

    Parameters
    ----------
    kmesh  : [nkx, nky, nkz]  k-point grid (use nkz=1 for 2D)
    a      : in-plane lattice constant (Bohr); default 4.733 ~ 2.504 Å
    vacuum : out-of-plane cell size (Bohr); default 20
    basis  : GTO basis set string; default 'sto-3g'
    verbose: PySCF verbosity level
    """
    a1 = np.array([a,       0.0,              0.0])
    a2 = np.array([a / 2.0, a * np.sqrt(3) / 2.0, 0.0])
    a3 = np.array([0.0,     0.0,              vacuum])

    N_pos = np.array([a / 2.0, a * np.sqrt(3) / 6.0, 0.0])
    bn_bond = np.linalg.norm(N_pos)   # = a / sqrt(3)

    cell = pbcgto.Cell()
    cell.atom    = f'B 0 0 0; N {N_pos[0]:.8f} {N_pos[1]:.8f} 0'
    cell.a       = np.array([a1, a2, a3])
    cell.basis   = basis
    cell.unit    = 'B'
    cell.verbose = verbose
    cell.build()

    print(f'  B-N bond = {bn_bond:.4f} Bohr = {bn_bond * 0.529177:.4f} Å  '
          f'(lit. ≈ 1.446 Å)')
    print(f'  nel_cell = {cell.nelectron},  nao = {cell.nao_nr()}')

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
        description='Generate hBN monolayer inputs for BSE finite-q.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--nkx',     type=int,   default=4,
                        help='k-points along a1 (in-plane).')
    parser.add_argument('--nky',     type=int,   default=4,
                        help='k-points along a2 (in-plane).')
    parser.add_argument('--nkz',     type=int,   default=1,
                        help='k-points along a3 (out-of-plane; use 1 for 2D).')
    parser.add_argument('--a',       type=float, default=4.733,
                        help='In-plane lattice constant (Bohr).')
    parser.add_argument('--vacuum',  type=float, default=20.0,
                        help='Out-of-plane cell size (Bohr).')
    parser.add_argument('--basis',   type=str,   default='sto-3g')
    parser.add_argument('--beta',    type=float, default=1000.0,
                        help='Inverse temperature (1/Ha).')
    parser.add_argument('--outdir',  type=str,   default=None)
    parser.add_argument('--ir_source', type=str,
                        default='example/N2_STO3G/irgrid/1e5.h5',
                        help='Path to the IR grid file to copy.')
    args = parser.parse_args()

    kmesh  = [args.nkx, args.nky, args.nkz]
    nk     = args.nkx * args.nky * args.nkz
    outdir = args.outdir or (
        f'example/hBN_{args.basis}_{args.nkx}x{args.nky}x{args.nkz}k'
    )
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
    print(f'\n--- k-RHF on hBN: kmesh={kmesh}, '
          f'a={args.a:.4f} Bohr, vacuum={args.vacuum:.1f} Bohr ---')
    cell, kmf, mydf = run_krhf_hBN(
        kmesh=kmesh, a=args.a, vacuum=args.vacuum,
        basis=args.basis, verbose=3)

    nao = cell.nao_nr()
    ksc = cell.get_scaled_kpts(kmf.kpts)
    nel_cell      = int(cell.nelectron)
    nfilled_per_k = nel_cell // 2

    print(f'\nSystem: nao={nao}/cell (B+N), nk={nk} ({kmesh})')
    print(f'HF total energy: {kmf.e_tot:.6f} Ha')
    for ik in range(nk):
        eps = np.array(kmf.mo_energy[ik])
        print(f'  k[{ik}] (k={ksc[ik]})  ε = {eps} Ha')

    # Same-k VQ
    print(f'\n--- Extracting same-k VQ ---')
    VQ_kk = extract_VQ(cell, mydf, kmf.kpts)
    print(f'  shape: {VQ_kk.shape}  (nk, NQ, nao, nao)')

    # Green's function
    print(f'\n--- Building G(τ) ---')
    sim = build_synthetic_sim(kmf, tau_mesh, beta=args.beta)
    print(f'  μ = {sim["mu"]:.6f} Ha')

    # Sanity: Fermi occupancies at first k-point
    eps0 = np.array(kmf.mo_energy[0]) - sim['mu']
    nF0  = 1.0 / (np.exp(np.clip(args.beta * eps0, -500.0, 500.0)) + 1.0)
    print(f'  k[0] nF (1=filled, 0=empty): {np.round(nF0, 3)}'
          f'  ({nfilled_per_k} filled expected)')

    # Write output
    print(f'\n--- Writing output to {outdir}/ ---')
    write_mean_field_input(os.path.join(outdir, 'mean_field_input.h5'), kmf)
    write_sim(os.path.join(outdir, 'sim.h5'), sim)

    int_q0 = os.path.join(outdir, 'df_hf_int')
    write_VQ(int_q0, VQ_kk, 'VQ_0.h5')

    int_fq = os.path.join(outdir, 'df_hf_int_fq')
    os.makedirs(int_fq, exist_ok=True)
    shutil.copy2(os.path.join(int_q0, 'VQ_0.h5'), os.path.join(int_fq, 'VQ_0.h5'))
    shutil.copy2(os.path.join(int_q0, 'meta.h5'), os.path.join(int_fq, 'meta.h5'))

    for q_idx in range(1, nk):
        print(f'\n--- VQ_q{q_idx}  (q = {ksc[q_idx]}) ---')
        VQ_kq = extract_VQ_kq(cell, mydf, kmf.kpts, q_idx)
        write_VQ(int_fq, VQ_kq, f'VQ_q{q_idx}.h5')

    print(f'''
=== Done ===
Files in {outdir}/

Plot dispersion:
  python script/plot_Pbse_dispersion.py \\
      --q0_dir {outdir} \\
      --fq_dir {outdir} \\
      --output hBN_dispersion.pdf

Plot band structure:
  python script/plot_bandstructure.py \\
      --input {outdir}/mean_field_input.h5 \\
      --sim   {outdir}/sim.h5 \\
      --output hBN_bands.pdf
''')


if __name__ == '__main__':
    main()
