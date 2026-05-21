#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Finite-q periodic BSE Casida equations.                                  #
#    Fork of casidaEq_periodic.py using proper off-diagonal GDF integrals     #
#    ⟨φ_μ(k)|V_Q|φ_ν(k+q)⟩ instead of the approximate same-k transform.     #
#                                                                             #
#    Physics summary:                                                          #
#      - Two VQ tensors are needed:                                            #
#          VQ_kk_ao[k,Q,μ,ν]  same-k integrals  (from VQ_0.h5)               #
#          VQ_kq_ao[k,Q,μ,ν]  off-diagonal k→k+q (from VQ_q{idx}.h5)        #
#      - After MO transform:                                                   #
#          VQ_oo[k]  ← C_k†[:occ] @ VQ_kk[k] @ C_k[:occ]   (hole-hole)     #
#          VQ_vv[k+q]← C_kq†[nocc:] @ VQ_kk[kq] @ C_kq[nocc:] (ptcl-ptcl) #
#          VQ_ia[k]  ← C_k†[:occ] @ VQ_kq[k] @ C_kq[nocc:]  (transition)   #
#      - At q≠0 the long-range exchange term is ZERO (κ=0).                   #
#      - At q=0 it is κ=2 (singlet) or κ=0 (triplet) as in the optical limit. #
#                                                                             #
#    Flat pair index: ρ = k * nocc * nvirt + i * nvirt + a                    #
#    kq_map[k] = (k + q_idx) % nk   (cyclic shift on the k-mesh)             #
#                                                                             #

import numpy as np
import scipy.linalg as LA
from joblib import Parallel, delayed
from scipy.linalg import matrix_balance

from casidaEq import concatAB, fix_phase, solveMO


# ---------------------------------------------------------------------------
# MO-basis transformations
# ---------------------------------------------------------------------------

def VQ_ao2mo_kk(VQ_kk_ao, vexMO):
    """
    Transform same-k GDF integrals from AO to MO basis at each k-point.

    V_{mn,Q}(k) = C_k^† V_{μν,Q}(k) C_k

    This is the standard same-k transform, identical to VQ_ao2mo_k in
    casidaEq_periodic.py.  The result is used for the hole-hole and
    particle-particle sectors (W_A) via the occ/virt slices.

    Parameters
    ----------
    VQ_kk_ao : ndarray, shape (nk, NQ, nao, nao)  complex128
        Same-k AO integrals ⟨φ_μ(k)|V_Q|φ_ν(k)⟩.
    vexMO : ndarray, shape (ns, nk, nao, nao)  complex128
        MO coefficient matrices from solveMO; spin index 0 is used.

    Returns
    -------
    VQ_mo_kk : ndarray, shape (nk, NQ, nao, nao)  complex128
        Same-k Coulomb integrals in MO basis.
    """
    nk, nQ, nao, _ = VQ_kk_ao.shape
    VQ_mo_kk = np.zeros_like(VQ_kk_ao, dtype=np.complex128)
    for ik in range(nk):
        C = vexMO[0, ik, :, :]          # (nao, nao) at k-point ik
        for iQ in range(nQ):
            VQ_mo_kk[ik, iQ] = C.conj().T @ VQ_kk_ao[ik, iQ] @ C
    return VQ_mo_kk


def VQ_ao2mo_kq_proper(VQ_kq_ao, vexMO, kq_map):
    """
    Transform off-diagonal GDF integrals from AO to MO basis using proper
    hole (k) and particle (k+q) MO coefficients.

    VQ_ia[k,Q,i,a] = C_k†[:occ] @ VQ_kq_ao[k,Q] @ C_{k+q}[:,nocc:]

    The ⟨φ_μ(k)|V_Q|φ_ν(k+q)⟩ integrals are read from VQ_q{idx}.h5 and
    already encode the correct (k, k+q) pair.  This function applies the
    MO rotation to produce the transition-density elements.

    Only the [:nocc, nocc:] block (occ at k, virt at k+q) is physically
    meaningful; the full (nao, nao) tensor is returned for convenience.

    Parameters
    ----------
    VQ_kq_ao : ndarray, shape (nk, NQ, nao, nao)  complex128
        Off-diagonal AO integrals ⟨φ_μ(k)|V_Q|φ_ν(k+q)⟩ for all k.
    vexMO : ndarray, shape (ns, nk, nao, nao)  complex128
        MO coefficient matrices; spin index 0 is used.
    kq_map : ndarray, shape (nk,)
        kq_map[k] = index of k+q in the k-mesh.

    Returns
    -------
    VQ_ia : ndarray, shape (nk, NQ, nao, nao)  complex128
        Off-diagonal MO tensor; VQ_ia[k,Q,:nocc,nocc:] is the transition
        density (occ at k, virt at k+q).
    """
    nk, nQ, nao, _ = VQ_kq_ao.shape
    VQ_ia = np.zeros_like(VQ_kq_ao, dtype=np.complex128)
    for ik in range(nk):
        ikq   = kq_map[ik]
        C_k   = vexMO[0, ik,  :, :]   # MO coefficients at k   (holes)
        C_kq  = vexMO[0, ikq, :, :]   # MO coefficients at k+q (particles)
        for iQ in range(nQ):
            VQ_ia[ik, iQ] = C_k.conj().T @ VQ_kq_ao[ik, iQ] @ C_kq
    return VQ_ia


# ---------------------------------------------------------------------------
# Kinetic energy
# ---------------------------------------------------------------------------

def diffEpsVec_fq(valsMO, nelec, kq_map):
    """
    Single-particle energy differences ε_{a,k+q} − ε_{i,k} as a 1-D vector.

    At q=0 (kq_map = identity) this equals the diagonal of diffEpsMat_k.

    Parameters
    ----------
    valsMO : ndarray, shape (ns, nk, nao)  QP energies; spin index 0 used.
    nelec  : int
    kq_map : ndarray, shape (nk,)

    Returns
    -------
    dEps : ndarray, shape (N,)  complex128
        N = nk * nocc * nvirt, with ρ = k*nocc*nvirt + i*nvirt + a.
    """
    nk    = valsMO.shape[1]
    nao   = valsMO.shape[2]
    nocc  = nelec // 2
    nvirt = nao - nocc
    N     = nk * nocc * nvirt

    dEps = np.zeros(N, dtype=np.complex128)
    for ik in range(nk):
        ikq          = kq_map[ik]
        eps_holes    = valsMO[0, ik,  :nocc]    # (nocc,)
        eps_particles = valsMO[0, ikq, nocc:]    # (nvirt,)
        for i in range(nocc):
            base = ik * nocc * nvirt + i * nvirt
            dEps[base:base + nvirt] = eps_particles - eps_holes[i]
    return dEps


# ---------------------------------------------------------------------------
# Static polarization extraction
# ---------------------------------------------------------------------------

def build_Pi_stat(tildeP_iw):
    """
    Extract the static (Ω=0) polarization from the frequency grid.

    The Ω=0 point corresponds to the middle of the bosonic Matsubara grid.

    Handles the two storage layouts:
      (niw, ns, NQ, NQ, 1)  — molecular / single-q layout
      (niw, ns, 1, NQ, NQ)  — alternative k-resolved layout

    Parameters
    ----------
    tildeP_iw : ndarray, shape (niw, ns, *, NQ, NQ) or (niw, ns, NQ, NQ, 1)
        Frequency-dependent screened polarization.

    Returns
    -------
    Pi_stat : ndarray, shape (NQ, NQ)  complex128
    """
    niw   = tildeP_iw.shape[0]
    shape = tildeP_iw.shape

    if tildeP_iw.ndim == 5 and shape[-1] == 1:
        # (niw, ns, NQ, NQ, 1)
        return tildeP_iw[niw // 2, 0, :, :, 0]
    elif tildeP_iw.ndim == 5 and shape[2] == 1:
        # (niw, ns, 1, NQ, NQ)
        return tildeP_iw[niw // 2, 0, 0, :, :]
    else:
        # Fallback: assume last two axes are NQ x NQ
        NQ = shape[-1]
        return tildeP_iw[niw // 2].reshape(NQ, NQ)


# ---------------------------------------------------------------------------
# Static screened-Coulomb matrices  (finite-q versions)
# ---------------------------------------------------------------------------

def build_W_A_fq(VQ_mo_kk, kq_map, Pi_stat, nocc, nvirt):
    """
    Screened attraction matrix for the A block at finite q.

    W_A[(k,i,a),(k,i',a')] = -(1/Nk) Σ_{QQ'} V_oo[k,Q,i,i'] W_eff[Q,Q'] V_vv[k+q,Q',a,a']

    The A block is diagonal in k (same k on both sides).  The hole-hole
    integrals V_oo come from VQ_mo_kk at k; the particle-particle integrals
    V_vv come from VQ_mo_kk at k+q (via kq_map), because the particle lives
    at k+q.

    W_eff = I + Pi_stat  (screened Coulomb kernel)

    Parameters
    ----------
    VQ_mo_kk : ndarray, shape (nk, NQ, nao, nao)
        Same-k MO-basis integrals (from VQ_ao2mo_kk).
    kq_map   : ndarray, shape (nk,)
    Pi_stat  : ndarray, shape (NQ, NQ)
    nocc, nvirt : int

    Returns
    -------
    contrib : ndarray, shape (N, N)  where N = nk * nocc * nvirt
        Ready-to-add negative-W_A matrix contribution.
    """
    nk, nQ = VQ_mo_kk.shape[:2]
    N = nk * nocc * nvirt

    # Hole sector: occ-occ at k
    V_oo      = VQ_mo_kk[:, :, :nocc, :nocc]                       # (nk, nQ, nocc, nocc)
    V_oo_flat = V_oo.transpose(0, 2, 3, 1).reshape(nk * nocc * nocc, nQ)
    # row = k * nocc^2 + i * nocc + j

    # Particle sector: virt-virt at k+q (index via kq_map)
    V_vv_kq   = np.stack([VQ_mo_kk[kq_map[ik], :, nocc:, nocc:]
                           for ik in range(nk)], axis=0)            # (nk, nQ, nvirt, nvirt)
    V_vvT_flat = V_vv_kq.transpose(0, 2, 3, 1).reshape(nk * nvirt * nvirt, nQ)
    # row = k * nvirt^2 + b * nvirt + a

    W_eff    = np.eye(nQ, dtype=np.complex128) + Pi_stat            # (nQ, nQ)
    PV       = W_eff @ V_vvT_flat.T                                  # (nQ, nk*nvirt^2)
    W_A_flat = (1.0 / nk) * (V_oo_flat @ PV)                        # (nk*nocc^2, nk*nvirt^2)

    W_A_full = W_A_flat.reshape(nk, nocc, nocc, nk, nvirt, nvirt)
    # Map to A:  A_contrib[k,i,a,k',j,b] = -W_A_full[k,i,j,k',b,a]
    # Source axes: (k=0, i=1, j=2, k'=3, b=4, a=5)
    # Target axes: (k=0, i=1, a=5, k'=3, j=2, b=4) → permutation (0,1,5,3,2,4)
    return -W_A_full.transpose(0, 1, 5, 3, 2, 4).reshape(N, N)


def build_W_B_fq(VQ_ia, Pi_stat, nocc, nvirt):
    """
    Screened attraction matrix for the B block using proper transition density.

    W_B[(k,i,a),(k,i',a')] = -(1/Nk) Σ_{QQ'} V_ia[k,Q,i,a] W_eff[Q,Q'] V_ia[k,Q',i',a']

    Both factors use VQ_ia, which encodes the proper off-diagonal (k→k+q)
    transition density from VQ_ao2mo_kq_proper.

    Parameters
    ----------
    VQ_ia   : ndarray, shape (nk, NQ, nao, nao)
        Off-diagonal MO tensor; occ@k, virt@k+q slice is meaningful.
    Pi_stat : ndarray, shape (NQ, NQ)
    nocc, nvirt : int

    Returns
    -------
    contrib : ndarray, shape (N, N)  where N = nk * nocc * nvirt
    """
    nk, nQ = VQ_ia.shape[:2]
    N      = nk * nocc * nvirt

    # V_ov_flat: (N, nQ)  row = k * nocc*nvirt + i * nvirt + a
    V_ov_flat = VQ_ia[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, nQ)

    W_eff    = np.eye(nQ, dtype=np.complex128) + Pi_stat
    PV       = W_eff @ V_ov_flat.T                                  # (nQ, N)
    W_B_flat = (1.0 / nk) * (V_ov_flat @ PV)                       # (N, N)

    W_B_full = W_B_flat.reshape(nk, nocc, nvirt, nk, nocc, nvirt)
    # Map to B:  B_contrib[k,i,a,k',j,b] = -W_B_full[k,i,b,k',j,a]
    # Source axes: (k=0, i=1, a=2, k'=3, j=4, b=5)
    # Target:  (k=0, i=1, a=5, k'=3, j=4, b=2) → permutation (0,1,5,3,4,2)
    return -W_B_full.transpose(0, 1, 5, 3, 4, 2).reshape(N, N)


# ---------------------------------------------------------------------------
# Static BSE Hamiltonian for finite-q
# ---------------------------------------------------------------------------

def solveHstatic_fq(Pi_stat, VQ_mo_kk, VQ_ia, valsMO, nelec, kq_map,
                    ex_type="singlet", tda=False, q_is_zero=False):
    """
    Build and diagonalize the static BSE Casida Hamiltonian at finite q.

    Key physics:
    - At q≠0: the long-range exchange term vanishes (κ=0) because the
      photon carries finite momentum and cannot excite the q=0 plasmon.
    - At q=0 (q_is_zero=True): exchange κ=2 (singlet) or κ=0 (triplet),
      identical to solveHstatic_k.
    - W_A uses VQ_mo_kk (same-k MO basis) with the kq_map to route the
      virt-virt integrals to k+q.
    - W_B uses VQ_ia (proper off-diagonal transition density).
    - The exchange contribution (if any) also uses VQ_ia.

    Hamiltonian structure:
      A = diag(dEps) + κ/Nk * U_exchange + W_A_contrib
      B = κ/Nk * U_B_exchange + W_B_contrib
      H = concatAB(A, B)   (2N × 2N non-Hermitian Casida matrix)

    Parameters
    ----------
    Pi_stat   : ndarray, shape (NQ, NQ)
    VQ_mo_kk  : ndarray, shape (nk, NQ, nao, nao)
        Same-k MO-basis integrals (from VQ_ao2mo_kk).
    VQ_ia     : ndarray, shape (nk, NQ, nao, nao)
        Off-diagonal MO tensor (from VQ_ao2mo_kq_proper).
    valsMO    : ndarray, shape (ns, nk, nao)
    nelec     : int
    kq_map    : ndarray, shape (nk,)
    ex_type   : str  "singlet" or "triplet"
    tda       : bool  Tamm-Dancoff approximation (B=0)
    q_is_zero : bool  If True use optical-limit exchange (κ≠0 for singlet).

    Returns
    -------
    effVals : ndarray, shape (2N,)
    effVex  : ndarray, shape (2N, 2N)
    H_stat  : ndarray, shape (2N, 2N)
    """
    nk, nQ, nao, _ = VQ_ia.shape
    nocc  = nelec // 2
    nvirt = nao - nocc
    N     = nk * nocc * nvirt

    # Exchange prefactor: 0 for finite q, 2 (singlet) or 0 (triplet) for q=0
    if q_is_zero:
        kappa = 2.0 if ex_type == "singlet" else 0.0
    else:
        kappa = 0.0

    print(f"  Finite-q BSE: Nk={nk}, nocc={nocc}, nvirt={nvirt}, "
          f"N={N}, 2N={2*N}, q_is_zero={q_is_zero}, kappa={kappa:.1f}")

    # Kinetic diagonal ε_{a,k+q} − ε_{i,k}
    dEps = diffEpsVec_fq(valsMO, nelec, kq_map)

    # Transition-density ov slice (shared by exchange and W_B)
    V_ov_flat = VQ_ia[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, nQ)

    # --- A block ---
    A = np.diag(dEps)
    if kappa != 0.0:
        A += (kappa / nk) * (V_ov_flat @ V_ov_flat.T)
    A += build_W_A_fq(VQ_mo_kk, kq_map, Pi_stat, nocc, nvirt)

    # --- B block ---
    if tda:
        B = np.zeros((N, N), dtype=np.complex128)
    else:
        B = np.zeros((N, N), dtype=np.complex128)
        if kappa != 0.0:
            # V_bj: VQ_ia[k,Q,nocc+b,j] reordered to flat row = k*N1 + j*nvirt + b
            V_bj_flat = VQ_ia[:, :, nocc:, :nocc].transpose(0, 3, 2, 1).reshape(N, nQ)
            B += (kappa / nk) * (V_ov_flat @ V_bj_flat.T)
        B += build_W_B_fq(VQ_ia, Pi_stat, nocc, nvirt)

    H_stat = concatAB(A, B)

    cond = np.linalg.cond(H_stat)
    print(f"  Solving non-Hermitian eigenvalue equation, cond = {cond:.4f}")

    H_balanced, scale = matrix_balance(H_stat)
    effVals, effVex   = LA.eig(H_balanced)
    effVex            = LA.solve(scale, effVex)

    # Fix sign ambiguity (same convention as casidaEq_periodic)
    idx = np.argmax(abs(effVex.real), axis=0)
    effVex[:, effVex[idx, np.arange(len(effVals))].real < 0] *= -1

    return effVals, effVex, H_stat


# ---------------------------------------------------------------------------
# Dynamic correction (diagonal approximation)
# ---------------------------------------------------------------------------

def _build_H_dyn_fq(Pi_iw_freq, effVex, VQ_mo_kk, VQ_ia,
                     valsMO, nelec, kq_map, nocc, nvirt,
                     Pi_stat=None, q_is_zero=False):
    """
    Build the dynamic BSE Hamiltonian at one Matsubara frequency and return
    the diagonal of V^{-1} H(iΩ) V.

    This is the finite-q analogue of _build_H_dyn_freq_k in
    casidaEq_periodic.py.  The polarization Pi_iw_freq (NQ×NQ) at the current
    frequency replaces Pi_stat in both the A and B blocks.

    Parameters
    ----------
    Pi_iw_freq : ndarray, shape (NQ, NQ)
        Screened polarization at frequency iΩ (already extracted from the
        full Pi array by the caller).
    effVex : ndarray, shape (2N, 2N)
        Static eigenvector matrix (from solveHstatic_fq).
    VQ_mo_kk : ndarray, shape (nk, NQ, nao, nao)
        Same-k MO-basis integrals.
    VQ_ia : ndarray, shape (nk, NQ, nao, nao)
        Off-diagonal MO transition-density tensor.
    valsMO : ndarray, shape (ns, nk, nao)
    nelec  : int
    kq_map : ndarray, shape (nk,)
    nocc, nvirt : int
    Pi_stat : ndarray or None
        Static polarization (Ω=0).  If provided, the static contribution is
        NOT subtracted here; the caller handles the net dynamic correction.
    q_is_zero : bool

    Returns
    -------
    h2p : ndarray, shape (2N,)
        Diagonal of V^{-1} H_dyn(iΩ) V.
    """
    nk, nQ = VQ_mo_kk.shape[:2]
    N = nk * nocc * nvirt

    if q_is_zero:
        # Rebuild exchange type consistently
        ex_type = "singlet"   # caller passes ex_type implicitly via kappa
        kappa = 2.0
    else:
        kappa = 0.0

    # Kinetic diagonal (frequency-independent)
    dEps = diffEpsVec_fq(valsMO, nelec, kq_map)

    # Transition-density ov slice
    V_ov_flat = VQ_ia[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, nQ)

    # --- A block at frequency iΩ ---
    A = np.diag(dEps)
    if kappa != 0.0:
        A += (kappa / nk) * (V_ov_flat @ V_ov_flat.T)
    A += build_W_A_fq(VQ_mo_kk, kq_map, Pi_iw_freq, nocc, nvirt)

    # --- B block at frequency iΩ ---
    B = np.zeros((N, N), dtype=np.complex128)
    if kappa != 0.0:
        V_bj_flat = VQ_ia[:, :, nocc:, :nocc].transpose(0, 3, 2, 1).reshape(N, nQ)
        B += (kappa / nk) * (V_ov_flat @ V_bj_flat.T)
    B += build_W_B_fq(VQ_ia, Pi_iw_freq, nocc, nvirt)

    H_dyn = concatAB(A, B)

    effVex_inv = LA.inv(effVex)
    return np.einsum('ij,jk,ki->i', effVex_inv, H_dyn, effVex)


def HDynDiagApprox_fq(tildeP_iw, effVex_static, VQ_mo_kk, VQ_ia,
                        valsMO, nelec, kq_map,
                        Pi_stat=None, q_is_zero=False, n_jobs=1):
    """
    Diagonal approximation to the frequency-dependent BSE Hamiltonian at all
    Matsubara frequencies for finite-q periodic systems.

    Mirrors HDynDiagApprox_k (casidaEq_periodic.py) but uses the proper
    off-diagonal transition density VQ_ia and the finite-q kinetic term.

    The adiabatic approximation (Eq. 36 of the manuscript) assumes that the
    static eigenvector matrix V also approximately diagonalizes H_eff(iΩ) at
    all bosonic Matsubara frequencies.  We therefore compute the diagonal

        H2p[iΩ, p] = [V^{-1} H_eff(iΩ) V]_{pp}

    at each iΩ without re-diagonalizing.

    Parameters
    ----------
    tildeP_iw : ndarray, shape (niw, ns, NQ, NQ, 1)
        Frequency-dependent screened polarization, after going through
        _extract_q0_polarization (or the analogous q-slice extraction).
        The (NQ, NQ) matrix at frequency iΩ is tildeP_iw[iΩ, 0, :, :, 0].
    effVex_static : ndarray, shape (2N, 2N)
        Eigenvectors from solveHstatic_fq (unsorted, matching H2p_inf).
    VQ_mo_kk : ndarray, shape (nk, NQ, nao, nao)
    VQ_ia    : ndarray, shape (nk, NQ, nao, nao)
    valsMO   : ndarray, shape (ns, nk, nao)
    nelec    : int
    kq_map   : ndarray, shape (nk,)
    Pi_stat  : ndarray or None, shape (NQ, NQ)
        Static polarization (Ω=0).  Optional; not used internally but kept
        for API symmetry with the periodic version.
    q_is_zero : bool
    n_jobs   : int  Joblib parallelism over frequency points.

    Returns
    -------
    H2p_dyn : ndarray, shape (niw, 2*N)
        Diagonal of V^{-1} H_eff(iΩ_n) V at all n, with bosonic symmetry
        H(iΩ) = H(-iΩ)* enforced.
    """
    nk, nQ, nao, _ = VQ_mo_kk.shape
    nocc  = nelec // 2
    nvirt = nao - nocc
    N     = nk * nocc * nvirt

    niw      = tildeP_iw.shape[0]
    niw_half = niw // 2 + 1

    def _get_Pi_at_iw(iw):
        """Extract (NQ, NQ) polarization at frequency index iw."""
        if tildeP_iw.ndim == 5 and tildeP_iw.shape[-1] == 1:
            return tildeP_iw[iw, 0, :, :, 0]
        elif tildeP_iw.ndim == 5 and tildeP_iw.shape[2] == 1:
            return tildeP_iw[iw, 0, 0, :, :]
        else:
            return tildeP_iw[iw].reshape(nQ, nQ)

    def _compute_one_freq(iw):
        Pi_iw_freq = _get_Pi_at_iw(iw)
        return _build_H_dyn_fq(
            Pi_iw_freq, effVex_static, VQ_mo_kk, VQ_ia,
            valsMO, nelec, kq_map, nocc, nvirt,
            Pi_stat=Pi_stat, q_is_zero=q_is_zero
        )

    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_compute_one_freq)(iw)
        for iw in range(niw_half)
    )

    H2p_dyn = np.zeros((niw, 2 * N), dtype=np.complex128)
    for iw, result in enumerate(results):
        H2p_dyn[iw] = result
    # Bosonic symmetry: H(iΩ) = H(-iΩ)*
    for iw, result in enumerate(results):
        H2p_dyn[niw - 1 - iw] = result

    # Symmetrise positive/negative frequency halves
    for iw in range(niw_half):
        H2p_dyn[iw]              = 0.5 * (H2p_dyn[iw] + H2p_dyn[niw - 1 - iw].conj())
        H2p_dyn[niw - 1 - iw]   = H2p_dyn[iw].conj()

    return H2p_dyn
