#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.   #
#                                                                             #
#    Plot the HF electronic band structure from a mean_field_input.h5 file.  #
#                                                                             #
#    Usage                                                                    #
#    -----                                                                    #
#    python script/plot_bandstructure.py \                                   #
#        --input  example/H2_correct_fq_8k/mean_field_input.h5 \             #
#        [--sim   example/H2_correct_fq_8k/sim.h5]              \            #
#        [--output bands.pdf]                                                 #

import os
import sys
import argparse

import numpy as np
import h5py
from scipy.linalg import eigh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AU2EV = 27.211386245981


def read_bands(mf_h5: str):
    """Diagonalise F(k) at every k-point; return fractional k_x and eigenvalues."""
    with h5py.File(mf_h5, 'r') as f:
        # Fock and overlap: (ns, nk, nao, nao, 2)  last axis = real/imag
        Fk_raw = f['HF/Fock-k'][()]
        Sk_raw = f['HF/S-k'][()]
        ksc    = f['grid/k_mesh_scaled'][()]      # (nk, 3)
        nel    = int(f['params/nel_cell'][()])
        nao    = int(f['params/nao'][()])

    ns, nk = Fk_raw.shape[:2]

    # Reconstruct complex matrices
    Fk = Fk_raw[0, :, :, :, 0] + 1j * Fk_raw[0, :, :, :, 1]  # (nk, nao, nao)
    Sk = Sk_raw[0, :, :, :, 0] + 1j * Sk_raw[0, :, :, :, 1]

    eps_all = np.empty((nk, nao))
    for ik in range(nk):
        # Generalised eigenvalue: F|n> = ε S|n>
        eps_all[ik], _ = eigh(Fk[ik], Sk[ik])

    kx = ksc[:, 0]   # fractional k_x component
    return kx, eps_all, nel


def read_mu(sim_h5):
    """Read the chemical potential from a sim.h5 file (last iteration)."""
    if sim_h5 is None or not os.path.isfile(sim_h5):
        return None
    with h5py.File(sim_h5, 'r') as f:
        it = int(f['iter'][()])
        return float(f[f'iter{it}/mu'][()])


def plot_bands(kx, eps_all, nel, mu, output):
    nk, nao = eps_all.shape
    nfilled  = nel // 2   # filled bands per spin

    # Sort k-points by kx for a clean x-axis
    order = np.argsort(kx)
    kx_s  = kx[order]
    eps_s = eps_all[order] * AU2EV   # (nk, nao) in eV

    # Reference energy
    if mu is not None:
        E_ref = mu * AU2EV
        ref_label = r'$\mu$ (sim)'
    else:
        HOMO = eps_all[:, nfilled - 1].max() * AU2EV
        LUMO = eps_all[:, nfilled].min()     * AU2EV
        E_ref = 0.5 * (HOMO + LUMO)
        ref_label = 'midgap'

    eps_s -= E_ref   # shift so reference is at 0

    HOMO_shift = eps_all[:, nfilled - 1].max() * AU2EV - E_ref
    LUMO_shift = eps_all[:, nfilled].min()     * AU2EV - E_ref
    gap = LUMO_shift - HOMO_shift

    fig, ax = plt.subplots(figsize=(4, 3))

    # Band colours: valence blue, conduction red
    for ib in range(nao):
        color = '#4e88c7' if ib < nfilled else '#e05c5c'
        lw    = 1.6
        ax.plot(kx_s, eps_s[:, ib], color=color, lw=lw)

    # Fermi level line
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6, label=f'{ref_label} (0 eV)')

    # HOMO / LUMO markers
    ax.axhline(HOMO_shift, color='#4e88c7', lw=0.7, ls=':', alpha=0.8)
    ax.axhline(LUMO_shift, color='#e05c5c', lw=0.7, ls=':', alpha=0.8)

    # Gap annotation
    ax.annotate('', xy=(0.97, LUMO_shift), xytext=(0.97, HOMO_shift),
                xycoords=('axes fraction', 'data'),
                textcoords=('axes fraction', 'data'),
                arrowprops=dict(arrowstyle='<->', color='grey', lw=1.0))
    ax.text(1.05, 0.5 * (HOMO_shift + LUMO_shift),
            f'{gap:.2f} eV', ha='left', va='center', fontsize=7.5,
            color='grey', transform=ax.get_yaxis_transform())

    ax.set_xlabel(r'$k_x$ (fractional)')
    ax.set_ylabel(r'$\varepsilon_n(k)$ (eV, ref. = midgap)')
    ax.set_xlim(kx_s[0] - 0.02, kx_s[-1] + 0.12)

    # y-axis: show a window around the gap unless bands are very wide
    y_lo = max(eps_s.min(), HOMO_shift - 3 * gap)
    y_hi = min(eps_s.max(), LUMO_shift + 3 * gap)
    ax.set_ylim(y_lo, y_hi)

    # Legend patches
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='#4e88c7', lw=1.6, label='Valence'),
        Line2D([0], [0], color='#e05c5c', lw=1.6, label='Conduction'),
        Line2D([0], [0], color='k',       lw=0.8, ls='--', label=ref_label),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc='upper right')
    ax.set_title(
        f'HF band structure — {nk} k-points, {nao} bands/k\n'
        f'Fundamental gap = {gap:.3f} eV',
        fontsize=9)

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f'Figure saved → {output}')

    # Console summary
    print(f'\nBand summary (relative to {ref_label}):')
    print(f'  HOMO  = {HOMO_shift:+.4f} eV')
    print(f'  LUMO  = {LUMO_shift:+.4f} eV')
    print(f'  Gap   = {gap:.4f} eV')
    for ib in range(nao):
        kind = 'val' if ib < nfilled else 'con'
        bmin = eps_s[:, ib].min()
        bmax = eps_s[:, ib].max()
        print(f'  band {ib} ({kind}): [{bmin:+.3f}, {bmax:+.3f}] eV  '
              f'(width {bmax-bmin:.3f} eV)')


def main():
    p = argparse.ArgumentParser(
        description='Plot HF band structure from mean_field_input.h5.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--input',  required=True,  metavar='FILE',
                   help='mean_field_input.h5')
    p.add_argument('--sim',    default=None,   metavar='FILE',
                   help='sim.h5 (optional; used to read μ).')
    p.add_argument('--output', default='bands.pdf', metavar='FILE',
                   help='Output figure path.')
    args = p.parse_args()

    kx, eps_all, nel = read_bands(args.input)
    mu = read_mu(args.sim)
    plot_bands(kx, eps_all, nel, mu, args.output)


if __name__ == '__main__':
    main()
