#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.   #
#                                                                             #
#    BSE kernel Ξ_{QQ'}(q, iΩ) evaluated in the density-fitting aux. basis.  #
#                                                                             #
#    Steps (following "Finite-q finite-temperature BSE" notes):              #
#                                                                             #
#      1. Read P⁰_{QQ'}(q, iΩ) and W_{QQ'}(q, iΩ) from PolarizationSolver. #
#         One HDF5 file per q-point; loop over all q.                        #
#      2. Exchange kernel U_{QQ'}(q) = κ · δ_{QQ'}  (frequency-independent) #
#         κ=1 (singlet, default);  κ=0 reserved for triplet.                 #
#      3. BSE kernel Ξ_{QQ'}(q, iΩ) = 2U_{QQ'} − W_{QQ'}(q, iΩ).           #
#      4. BSE Dyson: P_BSE(q,iΩ) = [I − P⁰(q,iΩ)·Ξ(q,iΩ)]⁻¹ P⁰(q,iΩ).    #
#      5. Stack results over q: shape (Nq, niw, NQ, NQ).                    #
#                                                                             #
#    Output shapes:                                                           #
#      Xi_iw  : (Nq, niw, NQ, NQ)                                           #
#      P_bse  : (Nq, niw, NQ, NQ)                                           #
#      P0_iw  : (Nq, niw, NQ, NQ)                                           #
#                                                                             #

import numpy as np
import h5py
from scipy.linalg import inv
from dataclasses import dataclass, field
from typing import Optional, List

from green_bse import polarization


# ---------------------------------------------------------------------------
# I/O helper: read P⁰ and W from PolarizationSolver HDF5 output
# ---------------------------------------------------------------------------

def read_polarization_h5(filename: str) -> dict:
    """
    Read the output of PolarizationSolver into a plain dict.

    Expected datasets (written by polarization.py):
      /iter{N}/P0_iw       (niw, ns, 1, NQ, NQ) complex128
      /iter{N}/P_iw_tilde  (niw, ns, 1, NQ, NQ) complex128
      /iter{N}/W_iw        (niw, ns, 1, NQ, NQ) complex128   exchange-contracted W
      /iter{N}/W_qq_iw     (niw, ns, 1, NQ, NQ) complex128   W = I + P̃ (DF basis,
                            fallback if W_iw not present)
      /iter{N}/U_qq        (NQ, NQ)              complex128   exchange bare Coulomb
                            (fallback: identity if not present in older files)

    Returns
    -------
    dict with keys: 'P0_iw', 'P_tilde_iw', 'W_iw', 'U_qq', 'iter', 'q_idx', 'NQ'
    """
    with h5py.File(filename, 'r') as f:
        it    = int(f['iter'][()])
        q_idx = int(f['q_idx'][()])
        NQ    = int(f['params/NQ'][()])
        grp   = f[f'iter{it}']
        P0    = grp['P0_iw'][()].astype(np.complex128)
        Pt    = grp['P_iw_tilde'][()].astype(np.complex128)
        # Prefer W_iw (exchange-contracted); fall back to W_qq_iw (DF basis)
        wkey  = 'W_iw' if 'W_iw' in grp else 'W_qq_iw'
        W     = grp[wkey][()].astype(np.complex128)
        # U_qq: exchange-contracted bare Coulomb (q-dependent, static).
        # Fall back to identity for files produced before this feature was added.
        if 'U_qq' in grp:
            U = grp['U_qq'][()].astype(np.complex128)
        else:
            U = np.eye(NQ, dtype=np.complex128)
    return {'P0_iw': P0, 'P_tilde_iw': Pt, 'W_iw': W, 'U_qq': U,
            'iter': it, 'q_idx': q_idx, 'NQ': NQ}


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _extract_static(arr_iw: np.ndarray) -> np.ndarray:
    """
    Extract the static (Ω=0) slice from a bosonic Matsubara array.

    Handles the storage layout (niw, ns, 1, NQ, NQ) produced by polarization.py.

    Returns
    -------
    arr_0 : (NQ, NQ) complex128
    """
    niw = arr_iw.shape[0]
    if arr_iw.ndim == 5 and arr_iw.shape[2] == 1:
        return arr_iw[niw // 2, 0, 0, :, :].copy()
    elif arr_iw.ndim == 5 and arr_iw.shape[-1] == 1:
        return arr_iw[niw // 2, 0, :, :, 0].copy()
    else:
        NQ = arr_iw.shape[-1]
        return arr_iw[niw // 2].reshape(NQ, NQ).copy()


def _squeeze_iw(arr_iw: np.ndarray) -> np.ndarray:
    """
    Collapse the spin (ns) and q-slot dimensions.

    (niw, ns, 1, NQ, NQ)  →  (niw, NQ, NQ)
    """
    if arr_iw.ndim == 5 and arr_iw.shape[2] == 1:
        return arr_iw[:, 0, 0, :, :]
    elif arr_iw.ndim == 5 and arr_iw.shape[-1] == 1:
        return arr_iw[:, 0, :, :, 0]
    else:
        niw = arr_iw.shape[0]
        NQ  = arr_iw.shape[-1]
        return arr_iw.reshape(niw, NQ, NQ)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

# def exchange_kernel(VQ_kq_ao: np.array, NQ: int, q_idx: int, kappa: float = 1.0) -> np.ndarray:
#     """
#     Static fallback exchange kernel: U_{QQ'} = κ · δ_{QQ'}.

#     This is the q→∞ / plane-wave limit in which the exchange-contracted bare
#     Coulomb U_{QQ'}(q) reduces to κ · I_{NQ}.  Use it only when the proper
#     q-dependent U_qq from PolarizationSolver is unavailable (e.g. in tests).

#     For production runs, U_qq(q) is computed by ``polarization.compute_U_qq``
#     and stored in the polarization HDF5 as ``/iter{N}/U_qq``.  It is read
#     automatically by ``read_polarization_h5``.

#     Parameters
#     ----------
#     NQ    : int    number of auxiliary basis functions
#     kappa : float  1.0 = singlet (default), 0.0 = triplet (no exchange)

#     Returns
#     -------
#     U : (NQ, NQ) complex128
#     """
#     # return kappa * np.eye(NQ, dtype=np.complex128)
    
#     return polarization.compute_U_qq()




def bse_kernel_iw(U: np.ndarray, W_iw: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """
    Compute the dynamic BSE kernel Ξ_{QQ'}(q, iΩ) = 2U − W(q, iΩ).

    Parameters
    ----------
    U     : (NQ, NQ) complex128
        Exchange-contracted bare Coulomb from PolarizationSolver.
    W_iw  : (niw, ns, 1, NQ, NQ) or (niw, NQ, NQ)  complex128
        Exchange-contracted screened Coulomb from PolarizationSolver.
    Returns
    -------
    Xi_iw : (niw, NQ, NQ) complex128
    """
    
    W_all = _squeeze_iw(W_iw) if W_iw.ndim == 5 else W_iw   # (niw, NQ, NQ)
    NQ    = W_all.shape[-1]

    return 2.0 * U[np.newaxis] - W_all                         # (niw, NQ, NQ)
    # return U[np.newaxis]                         # (niw, NQ, NQ)


def bse_dyson_iw(P0_iw: np.ndarray,
                 Xi_iw: np.ndarray) -> np.ndarray:
    """
    Solve the BSE Dyson equation in the QQ' auxiliary basis at every iΩ.

        P_BSE(iΩ) = [I − P⁰(iΩ) · Ξ(iΩ)]⁻¹ · P⁰(iΩ)

    Parameters
    ----------
    P0_iw : (niw, ns, 1, NQ, NQ) or (niw, NQ, NQ)  complex128
        Bare polarization at all Matsubara frequencies.
    Xi_iw : (niw, NQ, NQ) complex128
        BSE kernel, output of bse_kernel_iw.

    Returns
    -------
    P_bse : (niw, NQ, NQ) complex128
    """
    P0_all = _squeeze_iw(P0_iw) if P0_iw.ndim == 5 else P0_iw  # (niw, NQ, NQ)
    niw, NQ, _ = P0_all.shape
    I_NQ  = np.eye(NQ, dtype=np.complex128)
    P_bse = np.empty_like(P0_all)

    for iw in range(niw):
        P0  = P0_all[iw]               # (NQ, NQ)
        Xi  = Xi_iw[iw]                # (NQ, NQ)
        lhs = I_NQ - P0 @ Xi           # I − P⁰(iΩ) Ξ(iΩ)
        P_bse[iw] = inv(lhs) @ P0

    return P_bse                       # (niw, NQ, NQ)


# ---------------------------------------------------------------------------
# Convenience wrapper: run the full Ξ pipeline over all q-points
# ---------------------------------------------------------------------------

@dataclass
class BSEKernelConfig:
    """Configuration for BSEKernelQQ."""
    pol_files   : List[str]         # HDF5 output from PolarizationSolver,
                                    #   one file per q-point, ordered by q_idx
    output_file : str  = "bse_kernel_qq.h5"
    kappa       : float = 1.0       # 1.0 = singlet (default); 0.0 = triplet


class BSEKernelQQ:
    """
    Compute the BSE kernel Ξ_{QQ'}(q, iΩ) and P_BSE(q, iΩ) for all q-points.

    Pipeline (per q-point)
    ----------------------
    1. Read P⁰(q,iΩ) and W(q,iΩ) from PolarizationSolver HDF5 output.
    2. Build U_{QQ'} = κ · δ_{QQ'}  (frequency-independent).
    3. Ξ(q,iΩ) = 2U − W(q,iΩ).
    4. BSE Dyson: P_BSE(q,iΩ) = [I − P⁰(q,iΩ)·Ξ(q,iΩ)]⁻¹ P⁰(q,iΩ).
    5. Stack over q → shapes (Nq, niw, NQ, NQ).

    Output HDF5 layout
    ------------------
    /Nq                              int
    /NQ                              int
    /q_indices    (Nq,)              int     q-point indices from pol files
    /kappa                           float
    /Xi_iw        (Nq, niw, NQ, NQ) complex128
    /P_bse        (Nq, niw, NQ, NQ) complex128
    /P0_iw        (Nq, niw, NQ, NQ) complex128
    """

    def __init__(self, config: BSEKernelConfig):
        self.cfg = config

    def _print_header(self):
        print("=" * 80)
        print("BSE KERNEL  Ξ_{QQ'}(q, iΩ)  —  dynamic auxiliary-basis formulation")
        print("=" * 80)
        print(f"  Number of q-points : {len(self.cfg.pol_files)}")
        print(f"  Polarization files : {self.cfg.pol_files[0]}")
        for pf in self.cfg.pol_files[1:]:
            print(f"                       {pf}")
        print(f"  Output file        : {self.cfg.output_file}")
        print(f"  κ (exchange)       : {self.cfg.kappa}  "
              f"({'singlet' if self.cfg.kappa == 1.0 else 'triplet' if self.cfg.kappa == 0.0 else 'custom'})")
        print("=" * 80)

    def run(self) -> dict:
        """
        Execute the full pipeline over all q-points.

        Returns
        -------
        dict with keys:
            'Xi_iw'     : (Nq, niw, NQ, NQ) complex128
            'P_bse'     : (Nq, niw, NQ, NQ) complex128
            'P0_iw'     : (Nq, niw, NQ, NQ) complex128
            'q_indices' : (Nq,) int
        """
        self._print_header()

        Nq = len(self.cfg.pol_files)
        Xi_list  = []
        Pbse_list = []
        P0_list   = []
        q_indices = []

        for iq, pol_file in enumerate(self.cfg.pol_files):
            print(f"\n[q {iq+1}/{Nq}]  {pol_file}")

            # Step 1 – read
            pol   = read_polarization_h5(pol_file)
            NQ    = pol['NQ']
            niw   = pol['P0_iw'].shape[0]
            q_idx = pol['q_idx']
            print(f"  NQ={NQ}, niw={niw}, q_idx={q_idx}")

            P0_raw = pol['P0_iw']    # (niw, ns, 1, NQ, NQ)
            W_raw  = pol['W_iw']     # (niw, ns, 1, NQ, NQ)
            U_raw  = pol['U_qq']     # (NQ, NQ)

            # Step 2+3 – Ξ(iΩ) = 2U − W(iΩ)
            Xi_iw = bse_kernel_iw(U_raw, W_raw)  # (niw, NQ, NQ)

            # Diagnostics at Ω=0
            i0 = niw // 2
            eigs_w  = np.linalg.eigvalsh((_squeeze_iw(W_raw)[i0]).real)
            eigs_xi = np.linalg.eigvalsh(Xi_iw[i0].real)
            print(f"  W(Ω=0) eigvals:  min={eigs_w[0]:.4f}, max={eigs_w[-1]:.4f}")
            print(f"  Ξ(Ω=0) eigvals:  min={eigs_xi[0]:.4f}, max={eigs_xi[-1]:.4f}  "
                  f"||Ξ||_F={np.linalg.norm(Xi_iw[i0]):.4f}")
            max_dXi = max(np.linalg.norm(Xi_iw[iw] - Xi_iw[i0]) for iw in range(niw))
            print(f"  max ||Ξ(iΩ)−Ξ(0)||_F = {max_dXi:.4e}  (freq. variation)")

            # Step 4 – BSE Dyson
            P_bse = bse_dyson_iw(P0_raw, Xi_iw)   # (niw, NQ, NQ)
            P0    = _squeeze_iw(P0_raw)             # (niw, NQ, NQ)
            print(f"  ||P_BSE(Ω=0)||_F  = {np.linalg.norm(P_bse[i0]):.6f}")
            print(f"  ||P⁰(Ω=0)||_F     = {np.linalg.norm(P0[i0]):.6f}")

            Xi_list.append(Xi_iw)
            Pbse_list.append(P_bse)
            P0_list.append(P0)
            q_indices.append(q_idx)

        # Stack over q
        Xi_all  = np.stack(Xi_list,  axis=0)   # (Nq, niw, NQ, NQ)
        Pbse_all = np.stack(Pbse_list, axis=0) # (Nq, niw, NQ, NQ)
        P0_all  = np.stack(P0_list,  axis=0)   # (Nq, niw, NQ, NQ)
        q_arr   = np.array(q_indices, dtype=np.int64)

        print(f"\nStacked results:")
        print(f"  Xi_iw : {Xi_all.shape}")
        print(f"  P_bse : {Pbse_all.shape}")
        print(f"  P0_iw : {P0_all.shape}")

        self._write_output(Xi_all, Pbse_all, P0_all, q_arr, NQ)

        return {'Xi_iw': Xi_all, 'P_bse': Pbse_all,
                'P0_iw': P0_all, 'q_indices': q_arr}

    def _write_output(self,
                      Xi_all: np.ndarray, Pbse_all: np.ndarray,
                      P0_all: np.ndarray, q_arr: np.ndarray,
                      NQ: int):
        path = self.cfg.output_file
        Nq, niw, _, _ = Xi_all.shape
        print(f"\nWriting output to {path} …")
        with h5py.File(path, 'w') as f:
            f.create_dataset('Nq',        data=np.int64(Nq))
            f.create_dataset('NQ',        data=np.int64(NQ))
            f.create_dataset('niw',       data=np.int64(niw))
            f.create_dataset('kappa',     data=np.float64(self.cfg.kappa))
            f.create_dataset('q_indices', data=q_arr)
            f.create_dataset('Xi_iw',     data=Xi_all.astype(np.complex128))
            f.create_dataset('P_bse',     data=Pbse_all.astype(np.complex128))
            f.create_dataset('P0_iw',     data=P0_all.astype(np.complex128))
        print(f"  /Xi_iw     : {Xi_all.shape}  (Nq, niw, NQ, NQ)")
        print(f"  /P_bse     : {Pbse_all.shape}  (Nq, niw, NQ, NQ)")
        print(f"  /P0_iw     : {P0_all.shape}  (Nq, niw, NQ, NQ)")
        print(f"  /q_indices : {q_arr}")
        print("Done.")
