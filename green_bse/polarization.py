#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Standalone polarization solver for P̃_{QQ'}(q, iΩ).                      #
#                                                                             #
#    Computes the density–density transferring polarizability in the          #
#    density-fitting (auxiliary) basis:                                       #
#                                                                             #
#      P⁰_{QQ'}(q, τ) = -(2/Nk) Σ_{k,s} Σ_{ac}                             #
#                            [Σ_d V*_{kq}[Q,a,d] G_{k+q}(β-τ)[d,c]]         #
#                          × [Σ_b G_k(τ)[a,b] V_{kq}[Q',b,c]]               #
#                                                                             #
#    After bosonic FT (τ → iΩ) and the Dyson equation                        #
#                                                                             #
#      P̃(q, iΩ) = [I − P⁰(q, iΩ)]⁻¹ P⁰(q, iΩ)                             #
#                                                                             #
#    the screened Coulomb W(q, iΩ) = I + P̃(q, iΩ) is also written.         #
#                                                                             #
#    Reference:                                                               #
#    "Finite-q finite-temperature BSE" notes (2025).                         #
#                                                                             #
#    Usage:                                                                   #
#      solver = PolarizationSolver(config)                                   #
#      solver.run()                                                           #
#                                                                             #

import os
import sys
import time
import numpy as np
import h5py
from scipy.linalg import inv, eigh
from dataclasses import dataclass, field
from typing import Optional

from irFT import IR_factory

AU2EV = 27.211386245981


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PolarizationConfig:
    """All parameters needed by PolarizationSolver."""

    input_file  : str                  # mean_field_input.h5
    sim_file    : str                  # scGW_sim.h5
    int_path    : str                  # directory with VQ_0.h5 / VQ_q{idx}.h5
    ir_file     : str                  # IR grid HDF5
    beta        : float                # inverse temperature (a.u.)
    q_idx       : int  = 0            # q-point index (0 = optical limit)
    iteration   : int  = -1           # sim iteration to use; -1 = last
    n_jobs      : int  = 1            # reserved for future parallelism
    output_file : str  = "p_iw_tilde.h5"

    def __post_init__(self):
        if not self.int_path.endswith('/'):
            self.int_path += '/'


# ---------------------------------------------------------------------------
# Helper: read VQ from HDF5
# ---------------------------------------------------------------------------

def _read_VQ(path: str) -> np.ndarray:
    """
    Read density-fitting integrals from path.

    Storage convention (from generate_{periodic,finite_q}_test.py):
      dataset '/0', float64 with last dim ×2 → .view(complex) → complex128.

    Returns
    -------
    VQ : (nk, NQ, nao, nao) complex128
    """
    with h5py.File(path, 'r') as f:
        raw = f['/0'][()]
    return raw.view(np.complex128)


# ---------------------------------------------------------------------------
# Core computation: P⁰(q, τ)
# ---------------------------------------------------------------------------

def compute_P0_tau(G_tau: np.ndarray,
                   VQ_kq_ao: np.ndarray,
                   kq_map: np.ndarray) -> np.ndarray:
    """
    Compute the bare polarization bubble P⁰(q, τ) in the DF basis.

    Physics
    -------
    From the finite-q BSE notes (imaginary-time axis):

        P⁰_{QQ'}(q, τ) = −4 Σ_{m,n,k} G_{n,k}(τ) G_{m,k+q}(−τ)
                            × V^{m,k+q}_{n,k}(Q) [V^{m,k+q}_{n,k}(Q')]∗

    Using G(−τ) = −G(β−τ) (fermionic anti-periodicity) and contracting in
    the DF / AO basis:

        X₁[Q, a, c] = Σ_d V∗_{kq}[Q, a, d]  G_{k+q}(β−τ)[d, c]
        B [a, c, Q'] = Σ_b G_k(τ)[a, b]      V_{kq}[Q', b, c]
        P⁰[Q, Q']   −= (2/Nk) Σ_{a,c} X₁[Q, a, c] B[a, c, Q']

    The factor 2 per spin (ns = 1 spin-restricted) matches the original
    gwtool.eval_P0_tilde_Q convention.

    Parameters
    ----------
    G_tau    : (ntau, ns, nk, nao, nao) complex128  AO Green's function
    VQ_kq_ao : (nk, NQ, nao, nao)       complex128  off-diagonal DF integrals
    kq_map   : (nk,) int                            kq_map[k] = (k + q_idx) % nk

    Returns
    -------
    P0_tau : (ntau, ns, 1, NQ, NQ) complex128
        Bare polarization bubble.  The '1' axis is the q slot (one q per call).
    """
    ntau, ns, nk, nao, _ = G_tau.shape
    NQ = VQ_kq_ao.shape[1]

    P0 = np.zeros((ntau, ns, 1, NQ, NQ), dtype=np.complex128)

    for t in range(ntau // 2):
        tt = ntau - t - 1          # β − τ index
        for s in range(ns):
            P_block = np.zeros((NQ, NQ), dtype=np.complex128)
            for ik in range(nk):
                ikq      = kq_map[ik]
                G_k_t    = G_tau[tt,  s, ik]   # G_k(τ)
                G_kq_btm = G_tau[t, s, ikq]  # G_{k+q}(β−τ)
                # G_kq_btm = G_tau[tt, s, ik]  # G_{k+q}(β−τ)
                V        = VQ_kq_ao[ik]        # (NQ, nao, nao)

                # X1[Q, a, c] = Σ_d V∗[Q, a, d] G_{k+q}(β−τ)[d, c]
                X1 = np.tensordot(V.conj(), G_kq_btm, axes=([2], [0]))   # (NQ, nao, nao)

                # B[a, c, Q'] = Σ_b G_k(τ)[a, b] V[Q', b, c]
                B  = np.tensordot(G_k_t, V, axes=([1], [1])).transpose(0, 2, 1)  # (nao, nao, NQ)

                P_block -= (1.0 / nk) * (X1.reshape(NQ, nao * nao) @ B.reshape(nao * nao, NQ))

            P0[t, s, 0] = P_block

    return P0


def symmetrize_P0(P0: np.ndarray) -> np.ndarray:
    """
    Enforce bosonic tau symmetry and Hermiticity on P⁰(τ).

    Bosonic symmetry: P⁰(τ) = P⁰(β − τ)
    Hermiticity:       P⁰(τ) = [P⁰(τ)]†

    Only the first half (t < ntau//2) was filled by compute_P0_tau;
    this function Hermitises the first half and mirrors it to the second.

    Parameters
    ----------
    P0 : (ntau, ns, 1, NQ, NQ) complex128  (modified in-place and returned)

    Returns
    -------
    P0 : same array, symmetrised
    """
    ntau = P0.shape[0]
    ns   = P0.shape[1]
    for t in range(ntau // 2):
        tt = ntau - t - 1
        for s in range(ns):
            H = 0.5 * (P0[t, s, 0] + P0[t, s, 0].conj().T)
            P0[t,  s, 0] = H
            P0[tt, s, 0] = H
    return P0


# ---------------------------------------------------------------------------
# Dyson equation: P̃ = (I − P⁰)⁻¹ P⁰
# ---------------------------------------------------------------------------

def dyson_equation(P0_iw: np.ndarray) -> np.ndarray:
    """
    Solve the Dyson equation for the screened polarization.

        P̃(q, iΩ) = [I − P⁰(q, iΩ)]⁻¹ P⁰(q, iΩ)

    Parameters
    ----------
    P0_iw : (niw, ns, 1, NQ, NQ) complex128

    Returns
    -------
    P_iw : (niw, ns, 1, NQ, NQ) complex128
        Hermitised screened polarization.
    """
    niw, ns, _, NQ, _ = P0_iw.shape
    P_iw = np.zeros_like(P0_iw)
    I_NQ = np.eye(NQ, dtype=np.complex128)

    for iw in range(niw):
        for s in range(ns):
            p0 = P0_iw[iw, s, 0]
            P_iw[iw, s, 0] = inv(I_NQ - p0) @ p0

    # Final Hermitisation
    P_iw = 0.5 * (P_iw + P_iw.conj().transpose(0, 1, 2, 4, 3))
    return P_iw


# ---------------------------------------------------------------------------
# W index-swap transformation — two equivalent implementations for comparison
# ---------------------------------------------------------------------------

def transform_W_via_mo(W_qq_iw: np.ndarray,
                       VQ_kq_ao: np.ndarray,
                       C_k: np.ndarray,
                       kq_map: np.ndarray) -> np.ndarray:
    """
    Transform W_{QQ'}(iΩ) by going through the MO-pair basis:

      (i)  V_{Q,mn}(k) = Σ_{ab} C*_{k,am} V_{Q,ab}(k,k+q) C_{k+q,bn}
           W_{mn,m'n'}  = Σ_{QQ'} V*_{Q,mn}(k) W_{QQ'} V_{Q',m'n'}(k)
      (ii) Swap MO indices: (m, n, m', n') → (m, m', n', n)
      (iii) W'_{QQ'} += Σ_{mm',n'n} V*_{Q,mm'}(k) W^{swap}_{mm',n'n} V_{Q',n'n}(k)

    Parameters
    ----------
    W_qq_iw  : (niw, ns, 1, NQ, NQ)  complex128
    VQ_kq_ao : (nk, NQ, nao, nao)    complex128   AO-basis DF integrals V(k, k+q)
    C_k      : (nk, nao, nao)         complex128   MO coefficients at each k
    kq_map   : (nk,) int                           kq_map[k] = index of k+q

    Returns
    -------
    W_iw : (niw, ns, 1, NQ, NQ)  complex128
    """
    niw, ns, _, NQ, _ = W_qq_iw.shape
    nk, _, nao, _    = VQ_kq_ao.shape
    npairs = nao * nao

    # V_{Q,mn}(k) = C†_k @ V_{Q,ab}(k,k+q) @ C_{k+q}
    V_flat = np.empty((nk, NQ, npairs), dtype=np.complex128)
    for k in range(nk):
        tmp    = VQ_kq_ao[k] @ C_k[k]                         # (NQ, nao, nao)
        V_mo_k = np.einsum('am,Qan->Qmn', C_k[k].conj(), tmp)  # (NQ, nao, nao)
        V_flat[k] = V_mo_k.reshape(NQ, npairs)

    W_iw = np.zeros((niw, ns, 1, NQ, NQ), dtype=np.complex128)

    for iw in range(niw):
        for s in range(ns):
            W_qq = W_qq_iw[iw, s, 0]       # (NQ, NQ)
            for k in range(nk):
                Vk = V_flat[k]              # (NQ, npairs)

                # (i) Expand to MO-pair basis
                W_mo_flat = Vk.conj().T @ W_qq @ Vk  # (npairs, npairs)

                # (ii) Swap (m, n, m', n') → (m, m', n', n)
                W_swap_flat = (W_mo_flat
                               .reshape(nao, nao, nao, nao)
                               .transpose(0, 2, 3, 1)
                               .reshape(npairs, npairs))

                # (iii) Contract back to QQ'
                W_iw[iw, s, 0] += Vk @ W_swap_flat @ Vk.conj().T
                # W_iw[iw, s, 0] += evaluate_B_stable(W_swap_flat, Vk, NQ)

    return W_iw


def transform_W_via_ao(W_qq_iw: np.ndarray,
                       VQ_kq_ao: np.ndarray) -> np.ndarray:
    """
    Compute the exchange BSE kernel K^x_{QQ'} entirely within the NQ-dimensional
    DF space, avoiding the expand-swap-project route that suffers from subspace
    mismatch.

    The expand-then-swap approach:
        W_ao = V† W V   (rank NQ, lives in V's row space)
        W_swap = permute(W_ao)   (same rank NQ, but now in a DIFFERENT subspace)
        K^x = V W_swap V†   ← most of W_swap is outside V's column space → large error

    Parameters
    ----------
    W_qq_iw  : (niw, ns, 1, NQ, NQ)  complex128
    VQ_kq_ao : (nk, NQ, nao, nao)    complex128   AO-basis DF integrals V(k, k+q)

    Returns
    -------
    W_iw : (niw, ns, 1, NQ, NQ)  complex128
    """
    niw, ns, _, NQ, _ = W_qq_iw.shape
    nk, _, nao, _     = VQ_kq_ao.shape
    npairs = nao * nao

    V_flat = VQ_kq_ao.reshape(nk, NQ, npairs)

    # Precompute L_k = V_k.conj() @ V_k.conj().T  for each k  (NQ × NQ)
    # L = np.stack([V_flat[k].conj() @ V_flat[k].conj().T for k in range(nk)])

    W_iw = np.zeros((niw, ns, 1, NQ, NQ), dtype=np.complex128)

    for iw in range(niw):
        for s in range(ns):
            for k in range(nk):   
                W_qq = W_qq_iw[iw, s, 0]       # (NQ, NQ)
                W_ao_flat = V_flat[k].conj().T @ W_qq @ V_flat[k]   # (NQ, npairs, nk)
                W_swap_flat = (W_ao_flat
                               .reshape(nao, nao, nao, nao)
                               .transpose(0, 2, 3, 1)
                               .reshape(npairs, npairs))
                W_iw[iw, s, 0] += V_flat[k] @ W_swap_flat @ V_flat[k].conj().T
            if iw == niw//2 and s == 0 and k == 0:  # Debug print for the first frequency and spin
                print(f"Debug: iw={iw}, s={s}, k={k}, "
                        f"||W_qq||={np.linalg.norm(W_qq):.3e}, "
                        f"||W_ao_flat||={np.linalg.norm(W_ao_flat):.3e}, "
                        f"||W_swap_flat||={np.linalg.norm(W_swap_flat):.3e}, "
                        f"partial ||W_iw||={np.linalg.norm(W_iw[iw, s, 0]):.3e}")
                    
    W_iw /= nk  # Average over k  
    
    return W_iw





# ---------------------------------------------------------------------------
# Exchange bare Coulomb U_{QQ'}(q) — identity in QQ' basis
# ---------------------------------------------------------------------------

def compute_U_qq(VQ_kq_ao) -> np.ndarray:
    """
    Exchange bare Coulomb U_{QQ'}(q) — unity (identity) in the QQ' basis.

    Parameters
    ----------
    NQ : int   Number of DF auxiliary functions.

    Returns
    -------
    U_qq : (NQ, NQ) complex128
    """
    nk, NQ, nao, _ = VQ_kq_ao.shape
    # npairs = nao * nao
    # V_flat = VQ_kq_ao.reshape(nk, NQ, npairs)
    # U_qq = np.zeros((NQ, NQ), dtype=np.complex128)
    
    # for k in range(nk):   
    #     U_flat = V_flat[k].conj().T @ np.eye(NQ, dtype=np.complex128) @ V_flat[k]  # (npairs, npairs)
    #     U_qq  += V_flat[k] @ U_flat @ V_flat[k].conj().T  # (NQ, NQ)
    
    # U_qq /= nk  # Average over k
    
    U_qq = np.eye(NQ, dtype=np.complex128)  # Identity in the QQ' basis
     
    return U_qq


# ---------------------------------------------------------------------------
# PolarizationSolver
# ---------------------------------------------------------------------------

class PolarizationSolver:
    """
    Standalone solver for P̃_{QQ'}(q, iΩ) — density-density transferring
    screened polarizability in the density-fitting basis.

    Pipeline
    --------
    1. Read G(τ)     from sim_file   (iter = config.iteration)
    2. Read VQ_kq_ao from int_path   (VQ_0.h5 for q=0; VQ_q{idx}.h5 otherwise)
    3. Compute P⁰(q, τ)  via the k-sum bubble
    4. Symmetrise P⁰(τ)
    5. FT P⁰(τ) → P⁰(q, iΩ)  using bosonic IR transform
    6. Dyson:  P̃(q, iΩ) = [I − P⁰]⁻¹ P⁰
    7. Write output HDF5

    Output datasets (all complex128)
    ---------------------------------
    /iter                       — iteration number (int64)
    /q_idx                      — q-point index    (int64)
    /params/beta                — β (float64)
    /params/nk                  — number of k-points
    /params/NQ                  — number of DF functions
    /iter{N}/P0_tau             — bare P in tau,       shape (ntau, ns, 1, NQ, NQ)
    /iter{N}/P0_iw              — bare P in freq,      shape (niw,  ns, 1, NQ, NQ)
    /iter{N}/P_iw_tilde         — screened P,          shape (niw,  ns, 1, NQ, NQ)
    /iter{N}/W_qq_iw            — W = I + P̃ (direct), shape (niw,  ns, 1, NQ, NQ)
    /iter{N}/W_iw               — W after MO-basis index swap (exchange-contracted),
                                  shape (niw, ns, 1, NQ, NQ); used by BSEKernelQQ.
    """

    def __init__(self, config: PolarizationConfig):
        self.config = config
        self.results = {}

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def print_header(self):
        q_label = "q=0 (optical limit)" if self.config.q_idx == 0 \
                  else f"q_idx={self.config.q_idx}"
        print("=" * 90)
        print("POLARIZATION SOLVER  P̃_{QQ'}(q, iΩ)")
        print("=" * 90)
        print(f"  Input (MF)         : {self.config.input_file}")
        print(f"  Sim (scGW)         : {self.config.sim_file}")
        print(f"  Int path           : {self.config.int_path}")
        print(f"  IR file            : {self.config.ir_file}")
        print(f"  β (a.u.)           : {self.config.beta}")
        print(f"  q-vector           : {q_label}")
        print(f"  Iteration          : {self.config.iteration} "
              f"(-1 = last)")
        print(f"  Output             : {self.config.output_file}")
        print("=" * 90)

    # ------------------------------------------------------------------
    # Read input
    # ------------------------------------------------------------------

    def read_input(self):
        """
        Read nk, nao from mean_field_input.h5 and diagonalise the Fock matrix
        at every (spin, k) to obtain the full MO coefficient matrix.

        Sets
        ----
        self.nao : int
        self.nk  : int
        self.C_k : (nk, nao, nao) complex128
            Columns are MO eigenvectors sorted by ascending eigenvalue (spin 0).
        """
        print("\n[1/5] Reading input file …")
        with h5py.File(self.config.input_file, 'r') as f:
            rFk      = f["/HF/Fock-k"][()].view(complex)
            rFk      = rFk.reshape(rFk.shape[:-1])   # (ns, nk, nao, nao)
            self.nao = int(f["/params/nao"][()])
            self.nk  = rFk.shape[1]
            self.kq_map = f["/grid/kq_map"][()]

        ns = rFk.shape[0]
        # Diagonalise Fock at each (s=0, k) — spin-restricted: use spin index 0
        C_k = np.zeros((self.nk, self.nao, self.nao), dtype=np.complex128)
        for k in range(self.nk):
            Fk = rFk[0, k]
            _, C = eigh(0.5 * (Fk + Fk.conj().T))   # symmetrise for safety
            C_k[k] = C                               # columns sorted by eigenvalue

        self.C_k = C_k
        print(f"       nao={self.nao}, nk={self.nk}")

    def read_G_tau(self):
        """
        Read G(τ) from sim_file.

        Returns
        -------
        G_tau : (ntau, ns, nk, nao, nao) complex128
        iter_used : int
        """
        print("\n[2/5] Reading G(τ) from sim file …")
        with h5py.File(self.config.sim_file, 'r') as f:
            if self.config.iteration == -1:
                it = f["iter"][()]
            else:
                it = self.config.iteration
            tag = f"/iter{it}/G_tau/data"
            G_data = f[tag][()].view(complex)

        print(f"       iter={it}, G_tau shape={G_data.shape}")
        return G_data, int(it)

    def read_VQ(self, q_idx: int) -> np.ndarray:
        """
        Load VQ_kq_ao from the int_path directory.

        For q_idx == 0: reads VQ_0.h5 (same-k integrals).
        For q_idx  > 0: reads VQ_q{q_idx}.h5 (off-diagonal integrals).

        Returns
        -------
        VQ : (nk, NQ, nao, nao) complex128
        """
        if q_idx == 0:
            path = self.config.int_path + "VQ_0.h5"
        else:
            path = self.config.int_path + f"VQ_q{q_idx}.h5"
        print(f"\n[3/5] Reading VQ from {path} …")
        VQ = _read_VQ(path)
        print(f"       VQ shape={VQ.shape}")
        return VQ

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full polarization pipeline."""
        self.print_header()

        # --- (1) input dimensions ---
        self.read_input()

        # --- (2) G(τ) ---
        G_tau, it = self.read_G_tau()
        ntau, ns, nk_g, nao_g, _ = G_tau.shape

        if nk_g != self.nk:
            print(f"  WARNING: G_tau has {nk_g} k-points; "
                  f"input.h5 has {self.nk}.  Using {nk_g}.")
            self.nk = nk_g

        # --- (3) VQ ---
        VQ_kq_ao = self.read_VQ(self.config.q_idx)
        kq_neg = -self.config.q_idx % self.nk  # Map negative q_idx to valid index
        VQ_neg_kq_ao = self.read_VQ(kq_neg)  # for negative q if needed
        NQ = VQ_kq_ao.shape[1]

        # kq_map
        kq_map = self.kq_map
        # print(f"       kq_map = {kq_map.tolist()}")

        # --- IR factory ---
        ir = IR_factory(self.config.beta, self.config.ir_file)
        assert ntau == ir.nts, (
            f"G_tau ntau={ntau} does not match IR grid nts={ir.nts}.  "
            "Check the tau grid used to build sim_file."
        )

        # --- (4) P⁰(q, τ) ---
        print(f"\n[4/5] Computing P⁰(q, τ) "
              f"[nk={self.nk}, nao={nao_g}, NQ={NQ}, ntau={ntau}, ns={ns}] …")
        t0 = time.time()
        P0_tau = compute_P0_tau(G_tau, VQ_kq_ao, kq_map[:,self.config.q_idx])
        P0_tau_neg_q = compute_P0_tau(G_tau, VQ_neg_kq_ao, kq_map[:,kq_neg])
        
        print(f"       done in {time.time()-t0:.2f} s")

        P0_tau = symmetrize_P0(P0_tau)
        P0_tau_neg_q = symmetrize_P0(P0_tau_neg_q)

        # --- (5) FT: P⁰(τ) → P⁰(iΩ) ---
        print("\n[5/5] Fourier transform τ → iΩ  (bosonic IR) …")
        t1 = time.time()
        P0_iw = ir.tauf_to_wb(P0_tau)
        P0_iw_neg_q = ir.tauf_to_wb(P0_tau_neg_q)
        print(f"       P0_iw shape={P0_iw.shape}, done in {time.time()-t1:.2f} s")

        # --- (6) Dyson equation ---
        print("       Solving Dyson equation P̃ = (I−P⁰)⁻¹ P⁰ …")
        t2 = time.time()
        P_tilde_iw = dyson_equation(P0_iw)
        P_tilde_iw_neg_q = dyson_equation(P0_iw_neg_q)
        print(f"       done in {time.time()-t2:.2f} s")

        # --- (7) Screened Coulomb W ---
        #
        # (i)   W_{QQ'}(iΩ) = I + P̃_{QQ'}(iΩ)  (DF basis)
        niw   = P_tilde_iw_neg_q.shape[0]
        I_eye = np.eye(NQ, dtype=np.complex128)
        W_qq_iw = P_tilde_iw_neg_q.copy()
        for iw in range(niw):
            for s in range(ns):
                W_qq_iw[iw, s, 0] += I_eye

        print("       Transforming W (AO route): QQ' → AO → swap → QQ' …")
        t3b = time.time()
        W_iw_ao = transform_W_via_ao(W_qq_iw, VQ_kq_ao)
        print(f"       W_iw_ao shape={W_iw_ao.shape}, done in {time.time()-t3b:.2f} s")

        W_iw = W_iw_ao  # pick one for downstream use
        # W_iw = W_iw_ao  # pick one for downstream use
        # W_iw = W_iw_att  # pick one for downstream use

        # (iv) Exchange bare Coulomb U_{QQ'}(q) — identity in QQ' basis.
        print("       Computing exchange bare Coulomb U_{QQ'}(q) …")
        t5 = time.time()
        U_qq = compute_U_qq(VQ_kq_ao)
        print(f"       U_qq shape={U_qq.shape}, done in {time.time()-t5:.2f} s")

        # --- Diagnostics ---
        self._print_diagnostics(P0_iw, P_tilde_iw, W_qq_iw, W_iw, U_qq)

        # --- Write output ---
        self.write_output(it, P0_tau, P0_iw, P_tilde_iw, W_qq_iw, W_iw, U_qq)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_diagnostics(self, P0_iw, P_tilde_iw, W_qq_iw, W_iw, U_qq):
        """Print key statistics for quick sanity checks."""
        niw = P0_iw.shape[0]
        print("\n" + "=" * 90)
        print("POLARIZATION DIAGNOSTICS")
        print("-" * 90)

        # Ω=0 sits at index niw//2 in the symmetric IR bosonic grid
        i0   = niw // 2
        P0_0 = P0_iw[i0, 0, 0]
        Pt_0 = P_tilde_iw[i0, 0, 0]
        W_0  = W_qq_iw[i0, 0, 0]
        W_x0 = W_iw[i0, 0, 0]

        eigs_P0 = np.sort(np.linalg.eigvalsh(P0_0.real))
        eigs_Pt = np.sort(np.linalg.eigvalsh(Pt_0.real))
        eigs_U  = np.sort(np.linalg.eigvalsh(U_qq.real))

        print(f"  NQ={P0_0.shape[0]}, niw={niw}, nao={self.nao}, nk={self.nk}")
        print(f"  P⁰(Ω=0) eigenvalues (real): "
              f"min={eigs_P0[0]:.4f}, max={eigs_P0[-1]:.4f}")
        print(f"  P̃ (Ω=0) eigenvalues (real): "
              f"min={eigs_Pt[0]:.4f}, max={eigs_Pt[-1]:.4f}")
        print(f"  W_QQ' (Ω=0) diagonal (real, first 4): {np.diag(W_0.real)[:4]}")
        print(f"  W_iw  (Ω=0) diagonal (real, first 4): {np.diag(W_x0.real)[:4]}"
              f"  [AO-route exchange-contracted]")
        print(f"  U_QQ' diagonal    (real, first 4): {np.diag(U_qq.real)[:4]}"
              f"  [eigenvals: min={eigs_U[0]:.4f}, max={eigs_U[-1]:.4f}]")
        print(f"  ||P⁰(Ω=0)||_F   = {np.linalg.norm(P0_0):.6f}")
        print(f"  ||P̃ (Ω=0)||_F   = {np.linalg.norm(Pt_0):.6f}")
        print(f"  ||U_QQ'  ||_F    = {np.linalg.norm(U_qq):.6f}")
        print("=" * 90 + "\n")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------

    def write_output(self, it: int,
                     P0_tau   : np.ndarray,
                     P0_iw    : np.ndarray,
                     P_tilde  : np.ndarray,
                     W_qq_iw  : np.ndarray,
                     W_iw     : np.ndarray,
                     U_qq     : np.ndarray):
        """
        Write all results to output HDF5.

        Layout
        ------
        /iter                        int64
        /q_idx                       int64
        /params/beta                 float64
        /params/nk                   int64
        /params/NQ                   int64
        /iter{N}/P0_tau              complex128, (ntau, ns, 1, NQ, NQ)
        /iter{N}/P0_iw               complex128, (niw,  ns, 1, NQ, NQ)
        /iter{N}/P_iw_tilde          complex128, (niw,  ns, 1, NQ, NQ)
        /iter{N}/W_qq_iw             complex128, (niw,  ns, 1, NQ, NQ)
            Direct screened Coulomb:  W_{QQ'}(iΩ) = I + P̃_{QQ'}(iΩ)
        /iter{N}/W_iw                complex128, (niw,  ns, 1, NQ, NQ)
            Exchange-contracted W via AO route (for comparison).
        /iter{N}/W_attract           complex128, (niw,  ns, 1, NQ, NQ)
            Attractive kernel:  W^{attract}_{QP} = Σ_k V^{hh}†_k W V^{pp}_k
            using same-k MO integrals at k (holes) and k+q (particles).
        /iter{N}/Xi_iw               complex128, (niw,  ns, 1, NQ, NQ)
            BSE kernel:  Ξ_{QP}(iΩ) = 2U_{QP} − W^{attract}_{QP}(iΩ)
        """
        path = self.config.output_file
        NQ   = P_tilde.shape[-1]

        print(f"Writing output to {path} …")
        with h5py.File(path, 'w') as f:
            f.create_dataset('iter',           data=np.int64(it))
            f.create_dataset('q_idx',          data=np.int64(self.config.q_idx))
            f.create_dataset('params/beta',    data=np.float64(self.config.beta))
            f.create_dataset('params/nk',      data=np.int64(self.nk))
            f.create_dataset('params/NQ',      data=np.int64(NQ))

            grp = f.require_group(f'iter{it}')
            grp.create_dataset('P0_tau',     data=P0_tau.astype(np.complex128))
            grp.create_dataset('P0_iw',      data=P0_iw.astype(np.complex128))
            grp.create_dataset('P_iw_tilde', data=P_tilde.astype(np.complex128))
            grp.create_dataset('W_qq_iw',    data=W_qq_iw.astype(np.complex128))
            grp.create_dataset('W_iw',       data=W_iw.astype(np.complex128))
            grp.create_dataset('U_qq',       data=U_qq.astype(np.complex128))

        print(f"  Wrote {path}")
        print(f"    /iter{it}/P0_tau     : {P0_tau.shape}")
        print(f"    /iter{it}/P0_iw      : {P0_iw.shape}")
        print(f"    /iter{it}/P_iw_tilde : {P_tilde.shape}")
        print(f"    /iter{it}/W_qq_iw    : {W_qq_iw.shape}  (direct, I + P̃)")
        print(f"    /iter{it}/W_iw       : {W_iw.shape}  (AO exchange-contracted)")
        print(f"    /iter{it}/U_qq       : {U_qq.shape}  (exchange bare Coulomb)")


# ---------------------------------------------------------------------------
# Active-space screened Coulomb W in the MO basis
# ---------------------------------------------------------------------------

def eval_W_MO_active(VQ, Pi_iw, active_mo_indices):
    """
    Calculate the screened Coulomb interaction W in the active MO space.

    Mirrors the W construction in casidaEq.py but restricts all MO indices
    to the supplied active-space window instead of the full occ/virt split.

    The formula at each frequency point is:

        W[p,q,r,s] = sum_Q  V_act[Q,p,q] * V_act[Q,r,s]                (bare)
                   + sum_{Q,P} V_act[Q,p,q] * Pi[Q,P] * V_act[P,r,s]  (screened)

    where p,q,r,s ∈ active_mo_indices and Pi is the dressed polarizability
    in the auxiliary Q-basis (unchanged from the full calculation).

    Parameters
    ----------
    VQ : ndarray, shape (1, NQ, nmo, nmo)
        Two-electron integrals in the MO basis (output of casida.VQ_ao2mo).
    Pi_iw : ndarray, shape (niw, 1, NQ, NQ, 1)
        Dressed polarizability in the auxiliary Q-basis on the Matsubara
        frequency grid (tildeP_iw from BSESolver.prepare_interaction_matrices).
    active_mo_indices : array-like of int
        MO indices defining the active space.

    Returns
    -------
    W_act : ndarray, complex128, shape (niw, n_act, n_act, n_act, n_act)
        W_act[iw, p, q, r, s] is the screened Coulomb interaction at
        frequency iw with all indices in the active space.
    """
    active_mo_indices = np.asarray(active_mo_indices)
    n_act = len(active_mo_indices)
    niw   = Pi_iw.shape[0]

    # Restrict VQ to the active MO subspace: (NQ, n_act, n_act)
    VQ_act = VQ[0, :, :, :][:, active_mo_indices, :][:, :, active_mo_indices]

    # Bare Coulomb term (frequency-independent): V[Q,p,q]*V[Q,r,s]
    U_bare = np.einsum('qij,qkl->ijkl', VQ_act, VQ_act, optimize=True)

    W_act = np.zeros((niw, n_act, n_act, n_act, n_act), dtype=np.complex128)

    for iw in range(niw):
        Pi = Pi_iw[iw, 0, :, :, 0]          # (NQ, NQ)
        # Pi * V_act -> (NQ, n_act, n_act)
        PV = np.einsum('qp,pkl->qkl', Pi, VQ_act, optimize=True)
        # V_act * PV -> (n_act, n_act, n_act, n_act)
        VPV = np.einsum('qij,qkl->ijkl', VQ_act, PV, optimize=True)
        W_act[iw] = (U_bare + VPV).transpose(0,2,3,1)

    return W_act


def eval_screened_W_active(W_act, Pi_iw):
    """
    Compute the BSE kernel  [I - (W Pi)^2]^{-1} W  in the active MO space.

    W and Pi are treated as matrices over MO-pair indices:
        M_{(p,q),(r,s)}  ↔  M[p, q, r, s]
    so the matrix product (WPi) contracts the (r,s) legs of W against the
    (p,q) legs of Pi.

    At each frequency point:
        WPi    = W  @ Pi              (n_act^2 × n_act^2 matrix product)
        result = (I - WPi @ WPi)^{-1} W

    Parameters
    ----------
    W_act : ndarray, complex128, shape (niw, n_act, n_act, n_act, n_act)
        Screened Coulomb interaction in the active MO space (from eval_W_MO_active).
    Pi_iw : ndarray, complex128, shape (niw, n_act, n_act, n_act, n_act)
        Independent-particle polarizability in the active MO space on the
        Matsubara frequency grid.

    Returns
    -------
    result : ndarray, complex128, shape (niw, n_act, n_act, n_act, n_act)
        [I - (W Pi)^2]^{-1} W evaluated at each frequency point.
    """
    niw   = W_act.shape[0]
    n_act = W_act.shape[1]
    n2    = n_act * n_act
    I     = np.eye(n2, dtype=np.complex128)

    result = np.zeros_like(W_act)

    for iw in range(niw):
        W_mat  = W_act[iw].reshape(n2, n2)
        Pi_mat = Pi_iw[iw].reshape(n2, n2)

        WPi  = W_mat @ Pi_mat
        WPi2 = WPi @ WPi                          # (W Pi)^2

        result[iw] = np.linalg.solve(I - WPi2, W_mat).reshape(n_act, n_act, n_act, n_act)

    return result


def eval_VPiSWPiV(VQ_act, Pi_iw, screened_W):
    """
    Compute the contraction  V Pi (screened_W) Pi V  in the auxiliary Q-basis.

    All MO-pair indices are treated as flat vectors so the expression becomes
    a sequence of matrix multiplications:

        result[Q, Q'] = V[Q, :] @ Pi @ SW @ Pi @ V[Q', :].conj()

    where V[Q, :] is V_act[Q] flattened over (p, q),
    Pi and SW are (n_act^2 x n_act^2) matrices over MO-pair indices.

    Parameters
    ----------
    VQ_act : ndarray, complex128, shape (NQ, n_act, n_act)
        Density-fitting integrals restricted to the active MO space.
        Obtain by slicing VQ[0, :, active_mo_indices, :][:, :, active_mo_indices].
    Pi_iw : ndarray, complex128, shape (niw, n_act, n_act, n_act, n_act)
        Independent-particle polarizability in the active MO space on the
        Matsubara frequency grid.
    screened_W : ndarray, complex128, shape (niw, n_act, n_act, n_act, n_act)
        Output of eval_screened_W_active: [I - (W Pi)^2]^{-1} W.

    Returns
    -------
    result : ndarray, complex128, shape (niw, NQ, NQ)
        V Pi SW Pi V evaluated at each frequency point.
    """
    niw   = Pi_iw.shape[0]
    n_act = Pi_iw.shape[1]
    n2    = n_act * n_act
    NQ    = VQ_act.shape[0]

    # Flatten V over MO-pair index: (NQ, n_act^2)
    V_flat = VQ_act.reshape(NQ, n2)

    result = np.zeros((niw, NQ, NQ), dtype=np.complex128)

    for iw in range(niw):
        Pi_mat = Pi_iw[iw].reshape(n2, n2)
        SW_mat = screened_W[iw].reshape(n2, n2)

        # V @ Pi: (NQ, n2) @ (n2, n2) -> (NQ, n2)
        VPi = V_flat @ Pi_mat
        # VPi @ SW: (NQ, n2) @ (n2, n2) -> (NQ, n2)
        VPiSW = VPi @ SW_mat
        # VPiSW @ Pi: (NQ, n2) @ (n2, n2) -> (NQ, n2)
        VPiSWPi = VPiSW @ Pi_mat
        # VPiSWPi @ V^†: (NQ, n2) @ (n2, NQ) -> (NQ, NQ)
        result[iw] = VPiSWPi @ V_flat.conj().T

    return result


def eval_Pph(VQ_act_k, Pi_iw_k, W_iw):
    """
    Particle-hole attraction kernel P^ph in the Q-space auxiliary basis.

        P^ph(q, iω)[Q,Q'] = (1/Nk) Σ_{s,k}
            V_k[Q,:] @ (I + Π_{s,k}(iω) W(iω))^{-1} @ Π_{s,k}(iω) @ V_k†[:,Q']

    The k-sum is averaged (factor 1/Nk).  The spin sum is a plain sum — for
    spin-restricted calculations (ns=1) multiply the output by 2 externally
    if spin degeneracy is required.

    Conventions for shapes
    ----------------------
    VQ_act_k : (..., nk, NQ, n_act, n_act) or (nk, NQ, n2)
        DF integrals for each k-point restricted to the active MO pairs.
        For the molecular Gamma-only case pass shape (1, NQ, n2) or
        simply the 2D (NQ, n2) array — it will be promoted to nk=1 internally.
    Pi_iw_k  : (niw, ns, nk, n_act, n_act, n_act, n_act) or (niw, ns, nk, n2, n2)
        Per-(iω, spin, k) independent-particle polarizability in active MO-pair
        space on the bosonic Matsubara grid.  For the molecular Gamma-only case
        pass shape (niw, 1, 1, n2, n2) or the squeezed (niw, n2, n2) —
        the latter will be promoted to ns=1, nk=1 internally.
    W_iw     : (niw, n_act, n_act, n_act, n_act) or (niw, n2, n2)
        k-averaged screened Coulomb in active MO-pair space (output of
        eval_W_MO_active with the screened full-system polarizability).

    Returns
    -------
    P_ph : (niw, NQ, NQ) complex128
    """
    # --- normalise V ---
    if VQ_act_k.ndim == 2:                          # (NQ, n2) molecular shorthand
        VQ_act_k = VQ_act_k[np.newaxis]             # → (1, NQ, n2)
    nk = VQ_act_k.shape[0]
    NQ = VQ_act_k.shape[1]
    n2 = int(np.prod(VQ_act_k.shape[2:]))
    V  = VQ_act_k.reshape(nk, NQ, n2)              # (nk, NQ, n2)

    # --- normalise Pi ---
    if Pi_iw_k.ndim == 3:                           # (niw, n2, n2) molecular shorthand
        Pi_iw_k = Pi_iw_k[:, np.newaxis, np.newaxis]   # → (niw, 1, 1, n2, n2)
    niw = Pi_iw_k.shape[0]
    ns  = Pi_iw_k.shape[1]
    Pi  = Pi_iw_k.reshape(niw, ns, nk, n2, n2)     # (niw, ns, nk, n2, n2)

    # --- normalise W ---
    W = W_iw.reshape(niw, n2, n2)                  # (niw, n2, n2)

    I_n   = np.eye(n2, dtype=np.complex128)
    P_ph  = np.zeros((niw, NQ, NQ), dtype=np.complex128)

    for iw in range(niw):
        W_mat = W[iw]                               # (n2, n2)
        for s in range(ns):
            for ik in range(nk):
                Pi_k = Pi[iw, s, ik]               # (n2, n2)
                Vk   = V[ik]                        # (NQ, n2)
                A    = I_n + Pi_k @ W_mat           # (I + Π_k W)
                # Solve A x = Π_k V_k†  so that  V_k @ x = V_k (I+ΠW)^{-1} Π V_k†
                x    = np.linalg.solve(A, Pi_k @ Vk.conj().T)   # (n2, NQ)
                P_ph[iw] += Vk @ x                 # (NQ, NQ)

    P_ph /= nk          # k-average (spin sum is kept as-is)
    return P_ph


def eval_PBSE(P_ph_iw):
    """
    BSE polarizability via the Dyson equation in the Q-space auxiliary basis.

        P^BSE(iω) = (I - P^ph(iω))^{-1} P^ph(iω)

    This is a Dyson-type resummation of the particle-hole ladder diagrams
    encoded in P^ph.

    Parameters
    ----------
    P_ph_iw : (niw, NQ, NQ) complex128
        Particle-hole kernel from eval_Pph.

    Returns
    -------
    P_bse : (niw, NQ, NQ) complex128
    """
    niw, NQ, _ = P_ph_iw.shape
    I_q  = np.eye(NQ, dtype=np.complex128)
    P_bse = np.zeros_like(P_ph_iw)
    for iw in range(niw):
        Ph = P_ph_iw[iw]
        P_bse[iw] = np.linalg.solve(I_q - Ph, Ph)
    return P_bse


def eval_P_dressed(P0_iw):
    """
    Compute the dressed polarizability  P^{(0)} [I - 2 P^{(0)}]^{-1}
    in the active MO space.

    Treating MO-pair indices as a flat matrix index (same convention as
    eval_screened_W_active and eval_VPiSWPiV), the expression at each
    frequency point is:

        result = Pi0 @ (I - 2 Pi0)^{-1}

    implemented as the transposed linear solve

        (I - 2 Pi0)^T result^T = Pi0^T

    for numerical stability (avoids explicitly forming the inverse).

    Parameters
    ----------
    Pi0_iw : ndarray, complex128, shape (niw, n_act, n_act, n_act, n_act)
        Independent-particle polarizability in the active MO space on the
        Matsubara frequency grid.

    Returns
    -------
    result : ndarray, complex128, shape (niw, n_act, n_act, n_act, n_act)
        P^{(0)} [I - 2 P^{(0)}]^{-1} at each frequency point.
    """
    niw  = P0_iw.shape[0]
    orig_shape = P0_iw.shape[1:]        # whatever trailing shape Pi has

    # Flatten to 2D matrix per frequency: (niw, m, m)
    m    = int(np.prod(orig_shape) ** 0.5)  # m = n_aux or n_act²
    P_2d = P0_iw.reshape(niw, m, m)
    I    = np.eye(m, dtype=np.complex128)

    result_2d = np.zeros((niw, m, m), dtype=np.complex128)

    for iw in range(niw):
        P = P_2d[iw]
        A = I - 2.0 * P                # (I - 2 P^{(0)})
        # result = P @ A^{-1}  ⟺  result^T = A^{-T} P^T
        # Use lstsq for robustness when A is near-singular
        result_2d[iw] = np.linalg.lstsq(A.T, P.T, rcond=None)[0].T

    return result_2d.reshape(P0_iw.shape)
