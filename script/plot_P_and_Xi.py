import argparse
import glob
import os
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt

AU2EV = 27.211386245981


def discover_q_files(reference: str) -> list:
    """
    Given one q-file (e.g. 'dispersion_q0.h5'), return all siblings sorted
    by q-index: [(0, 'dispersion_q0.h5'), (1, 'dispersion_q1.h5'), ...]

    If the filename does not match the _q{n}.h5 pattern, just return the
    single file with index 0.
    """
    m = re.match(r'^(.+_q)\d+(\.h5)$', reference)
    if not m:
        return [(0, reference)]

    stem_prefix = m.group(1)   # e.g. 'dispersion_q'
    suffix      = m.group(2)   # '.h5'
    pattern     = f'{stem_prefix}*{suffix}'

    found = []
    for fp in glob.glob(pattern):
        m2 = re.search(r'_q(\d+)\.h5$', fp)
        if m2:
            found.append((int(m2.group(1)), fp))

    if not found:
        return [(0, reference)]

    return sorted(found, key=lambda x: x[0])


def load_file(filepath):
    data = {}
    with h5py.File(filepath, 'r') as f:
        print("Keys in file:", list(f.keys()))
        if '/P0_iw' in f:
            data['P'] = f['/P0_iw'][:]
            print(f"  P shape: {data['P'].shape}")
        if '/Xi_iw' in f:
            data['Xi'] = f['/Xi_iw'][:]
            print(f"  Xi shape: {data['Xi'].shape}")
        if '/Omega' in f:
            data['Omega'] = f['/Omega'][:]
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Plot P and Xi traces, plus Re Tr Xi(Ω=0) dispersion vs q. '
                    'Sibling _q*.h5 files are discovered automatically.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('reference', nargs='?', default='H2_16k_dispersion_q0.h5',
                        help='Any one of the per-q HDF5 files. All _q*.h5 siblings '
                             'in the same directory are loaded automatically.')
    parser.add_argument('--qx', default=None,
                        help='Comma-separated q_x fractional coordinates, one per '
                             'discovered file (overrides q-index on the x-axis).')
    parser.add_argument('--output', default='P_and_Xi_traces.png')
    args = parser.parse_args()

    q_pairs = discover_q_files(args.reference)   # [(q_idx, path), ...]
    q_indices = [qi for qi, _ in q_pairs]
    filepaths = [fp for _, fp in q_pairs]

    print(f'\nDiscovered {len(filepaths)} q-file(s):')
    for qi, fp in q_pairs:
        print(f'  q_idx={qi}  {fp}')

    if args.qx is not None:
        qx_vals = np.array([float(x) for x in args.qx.split(',')])
        if len(qx_vals) != len(filepaths):
            parser.error(f'--qx has {len(qx_vals)} values but {len(filepaths)} '
                         f'files were discovered')
        qx_label = r'$q_x$ (fractional)'
    else:
        qx_vals  = np.array(q_indices, dtype=float)
        qx_label = 'q index'

    all_data = [load_file(fp) for fp in filepaths]
    first    = all_data[0]

    cmap   = plt.cm.viridis
    colors = [cmap(i / max(len(filepaths) - 1, 1)) for i in range(len(filepaths))]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # ── Panel 0: Re Tr P(iΩ) from q=0 file ──────────────────────────────────
    if 'P' in first:
        P = first['P']
        if P.ndim > 2:
            xvals = first['Omega'] * AU2EV if 'Omega' in first else np.arange(P.shape[0])
            P_tr  = np.array([np.trace(P[i]).real for i in range(P.shape[0])])
            axes[0].plot(xvals, P_tr, color='#4e88c7')
        axes[0].axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
        axes[0].set_title(r'Re Tr $P^0(i\Omega_n)$  (q = q₀)')
        axes[0].set_xlabel(r'$i\Omega_n$ (eV)' if 'Omega' in first else 'iΩ index')
        axes[0].set_ylabel(r'Re Tr $P^0$')

    # ── Panel 1: Re Tr Xi(iΩ) — one curve per q ──────────────────────────────
    for qi_idx, (qx, d) in enumerate(zip(qx_vals, all_data)):
        if 'Xi' not in d:
            continue
        Xi       = d['Xi']
        xvals    = d['Omega'] * AU2EV if 'Omega' in d else np.arange(Xi.shape[0])
        Xi_tr    = np.array([np.trace(Xi[i]).real for i in range(Xi.shape[0])])
        label    = (f'$q_x={qx:.3f}$' if args.qx is not None
                    else f'q{q_indices[qi_idx]}')
        axes[1].plot(xvals, Xi_tr, color=colors[qi_idx], label=label)

    axes[1].axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    axes[1].set_title(r'Re Tr $\Xi(i\Omega_n)$  — all q')
    axes[1].set_xlabel(r'$i\Omega_n$ (eV)' if 'Omega' in first else 'iΩ index')
    axes[1].set_ylabel(r'Re Tr $\Xi$')
    if len(filepaths) > 1:
        axes[1].legend(fontsize=7, frameon=False)

    # ── Panel 2: Xi heatmap at Ω=0 (q=0 file) ────────────────────────────────
    if 'Xi' in first and first['Xi'].ndim > 2:
        Xi0 = first['Xi']
        i0  = Xi0.shape[0] // 2
        print("Xi diagonal at Ω=0:", np.diag(Xi0[i0].real))
        im = axes[2].imshow(Xi0[i0].real, cmap='viridis', aspect='auto')
        axes[2].set_title(r'Re $\Xi_{QQ^\prime}(\Omega=0)$  (q = q₀)')
        axes[2].set_xlabel("Q'")
        axes[2].set_ylabel('Q')
        plt.colorbar(im, ax=axes[2])

    # ── Panel 3: Re Tr Xi(Ω=0) vs q ──────────────────────────────────────────
    xi_static = []
    for d in all_data:
        if 'Xi' in d:
            i0 = d['Xi'].shape[0] // 2
            xi_static.append(np.trace(d['Xi'][i0]).real)
        else:
            xi_static.append(np.nan)

    axes[3].plot(qx_vals, xi_static, 'o-', color='#e05c5c', ms=6)
    for qx, val in zip(qx_vals, xi_static):
        axes[3].annotate(f'{val:.2f}', (qx, val),
                         textcoords='offset points', xytext=(4, 4), fontsize=7)
    axes[3].axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    axes[3].set_title(r'Re Tr $\Xi(\Omega=0)$ vs $\vec{q}$')
    axes[3].set_xlabel(qx_label)
    axes[3].set_ylabel(r'Re Tr $\Xi(\Omega=0)$')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved → {args.output}')


if __name__ == '__main__':
    main()
