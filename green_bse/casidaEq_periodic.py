#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Periodic (k-point) extension of the BSE Casida equations.               #
#    Extends casidaEq.py to support multiple k-points for solids.            #
#                                                                             #
#    Theory reference: same as casidaEq.py (manuscript Sec. 2.4).            #
#    For periodic systems, particle-hole pairs carry a k-point index k,      #
#    so the BSE Hamiltonian has dimension 2 * Nk * nocc * nvirt.             #
#                                                                             #
#    Flat pair index: rho = k * nocc * nvirt + i * nvirt + a                 #
#    where k in [0, Nk), i in [0, nocc), a in [0, nvirt).                    #
#                                                                             #
#    q = 0 (optical limit, this file's original scope):                      #
#      A[k,i,a; k',j,b] = (eps_{k,c} - eps_{k,v}) delta + exchange - W_A   #
#      B[k,i,a; k',j,b] = exchange_B - W_B                                  #
#                                                                             #
#    Finite-q generalization (added):                                        #
#      Pair index S = (n=val@k, m=cond@k+q); kq_map[k] gives k+q index.    #
#      A[S,S'] = (eps_{m,k+q} - eps_{n,k}) delta + Df_S * (exchange - W_A) #
#      where Df_S = f(eps_{n,k}) - f(eps_{m,k+q})  (Fermi-Dirac weights).  #
#      For q=0 + T=0: Df=1 and reduces exactly to the original equations.   #
#                                                                             #
#    The k-normalization factor 1/Nk comes from the BvK cell volume.         #
#    For Nk=1 (molecular/Gamma-only) this reduces exactly to casidaEq.py.   #
#                                                                             #

import numpy as np
import h5py
import scipy.linalg as LA
from joblib import Parallel, delayed
from scipy.linalg import matrix_balance
from scipy.sparse.linalg import LinearOperator, eigsh

from casidaEq import concatAB, fix_phase, solveMO


# ---------------------------------------------------------------------------
# MO-basis transformation
# ---------------------------------------------------------------------------

def VQ_ao2mo_k(VQ, vexMO, rSk=None):
    """
    Transform the density-fitting Coulomb tensor from AO to MO basis at each k-point.
    Unlike the molecular VQ_ao2mo, each k-point uses its own MO coefficient matrix.

    Parameters
    ----------
    VQ : ndarray, shape (nk, nQ, nao, nao)
        Density-fitted Coulomb integrals in AO basis.
    vexMO : ndarray, shape (ns, nk, nao, nao)
        MO coefficient matrices from solveMO; spin index 0 is used.
    rSk : ndarray, shape (ns, nk, nao, nao), optional
        AO overlap matrices at each k-point.  When provided the transform uses
        SC_k = S_k @ C_k so that V_mo = (SC)† V_ao (SC).

    Returns
    -------
    VQ_mo : ndarray, shape (nk, nQ, nao, nao)
        Coulomb integrals in MO basis, V_{mn,Q}(k) = (SC_k)^† V_{ij,Q}(k) (SC_k).
    """
    nk, nQ, nao, _ = VQ.shape
    VQ_mo = np.zeros_like(VQ, dtype=np.complex128)
    for ik in range(nk):
        C  = vexMO[0, ik, :, :]
        SC = rSk[0, ik] @ C if rSk is not None else C
        for iQ in range(nQ):
            VQ_mo[ik, iQ] = SC.conj().T @ VQ[ik, iQ] @ SC
    return VQ_mo


# ---------------------------------------------------------------------------
# Kinetic energy matrix
# ---------------------------------------------------------------------------

def diffEpsMat_k(valsMO, nelec):
    """
    Build the diagonal QP energy-difference matrix for periodic BSE.

    The kinetic block is diagonal in all indices:
        Δε[k,i,a] = ε_{k, nocc+a} - ε_{k, i}

    Parameters
    ----------
    valsMO : ndarray, shape (ns, nk, nao)
        QP energies; spin index 0 is used.
    nelec : int
        Total number of electrons (spin-summed).

    Returns
    -------
    diffEps : ndarray, shape (N, N) where N = Nk * nocc * nvirt
        Diagonal matrix of energy differences.
    """
    nk = valsMO.shape[1]
    nao = valsMO.shape[2]
    nocc = nelec // 2
    nvirt = nao - nocc
    N = nk * nocc * nvirt
    diffEps = np.zeros((N, N), dtype=np.complex128)
    for ik in range(nk):
        for i in range(nocc):
            for a in range(nvirt):
                row = ik * nocc * nvirt + i * nvirt + a
                diffEps[row, row] = valsMO[0, ik, nocc + a] - valsMO[0, ik, i]
    return diffEps


# ---------------------------------------------------------------------------
# Static screened-Coulomb matrices
# ---------------------------------------------------------------------------

def _build_W_A(VQ_mo, Pi_stat):
    """
    Build the screened attraction matrix for the A block.

    W_A[k,i,j; k',a,b] = (1/Nk) * sum_{QQ'} V_{ij}^{k,Q} (delta+Pi)_{QQ'} V_{ab}^{k',Q'}

    where i,j are occupied and a,b are virtual.

    The A-block matrix element is:
        A[k,i,a; k',j,b] -= W_A[k,i,j; k',b,a]   <-- note b,a transposed

    Returns
    -------
    W_A_contrib : ndarray, shape (N, N)
        Contribution -W_A (with index transposition applied), ready to add to A.
    """
    nk, nQ, nao, _ = VQ_mo.shape
    nocc_virt_pairs = [(nocc, nao - nocc) for nocc in []]
    # Infer nocc from VQ shape: not possible here, caller must pass it.
    raise NotImplementedError("Use build_W_A_contrib instead.")


def build_W_A_contrib(VQ_mo, Pi_stat, nocc, nvirt):
    """
    Compute the -W_A contribution to the A block for all k-point pairs.

    W_A arises from the density-fitting decomposition of the screened Coulomb:
        W(iΩ) = U + sum_{QQ'} V_{ij}^Q [Π̃(iΩ)]_{QQ'} V_{ab}^{Q'}
    where i,j ∈ occ and a,b ∈ virt (cf. Eq. 29 and 32b of the manuscript).

    Enters the A matrix as:  A[k,i,a; k',j,b] -= W_A[k,i,j; k',b,a]
    (the b,a transposition follows the molecular convention in matEleXiStat).

    Parameters
    ----------
    VQ_mo : ndarray, shape (nk, nQ, nao, nao)
    Pi_stat : ndarray, shape (nQ, nQ)  -- q=0 static auxiliary-basis polarization
    nocc, nvirt : int

    Returns
    -------
    contrib : ndarray, shape (N, N) where N = nk * nocc * nvirt
        Ready-to-add negative-W_A matrix.
    """
    nk, nQ = VQ_mo.shape[:2]
    N = nk * nocc * nvirt
    N1 = nocc * nvirt

    # V_oo[k,Q,i,j] -> flat (nk*nocc*nocc, nQ) with order k fastest, then i, then j
    V_oo = VQ_mo[:, :, :nocc, :nocc]                              # (nk, nQ, nocc, nocc)
    V_oo_flat = V_oo.transpose(0, 2, 3, 1).reshape(nk * nocc * nocc, nQ)
    #  row index = k * nocc^2 + i * nocc + j

    # V_vv^T[k',Q',b,a] = VQ_mo[k',Q',nocc+b,nocc+a] -- transposed virt block
    V_vv = VQ_mo[:, :, nocc:, nocc:]                              # (nk, nQ, nvirt, nvirt)
    V_vvT = V_vv.transpose(0, 2, 3, 1)                            # (nk, nvirt, nvirt, nQ)
    # swap last two virt axes so that index order is (k', b, a, Q')
    V_vvT = V_vvT.transpose(0, 2, 1, 3)                           # (nk, a, b, Q') -- wait
    # We want V_vvT[k',Q',b,a] in flat form (nk*nvirt*nvirt, nQ) with (k',b,a) ordering
    # VQ_mo[k',Q',nocc+b,nocc+a] = V_vv[k',Q',b,a] -- this is already V_vv[k',Q',b,a]
    # but we swapped indices: V_vvT[k',b,a,Q'] = V_vv[k',Q',b,a]
    # We want flat row = k'*nvirt^2 + b*nvirt + a
    V_vvT_flat = V_vv.transpose(0, 2, 3, 1).reshape(nk * nvirt * nvirt, nQ)
    # row = k' * nvirt^2 + b * nvirt + a  (b runs slower, a runs faster)

    # Effective screened Coulomb kernel: (I + Pi_stat)
    W_eff = np.eye(nQ, dtype=np.complex128) + Pi_stat             # (nQ, nQ)

    # W_A_flat[k*nocc^2+i*nocc+j, k'*nvirt^2+b*nvirt+a] =
    #   (1/Nk) * V_oo[k,i,j] . W_eff . V_vvT[k',b,a]
    PV = W_eff @ V_vvT_flat.T                                      # (nQ, nk*nvirt^2)
    W_A_flat = (1.0 / nk) * (V_oo_flat @ PV)                      # (nk*nocc^2, nk*nvirt^2)

    # Reshape to (nk, nocc, nocc, nk, nvirt, nvirt) with axes (k,i,j,k',b,a)
    W_A_full = W_A_flat.reshape(nk, nocc, nocc, nk, nvirt, nvirt)

    # Map to A:  A_contrib[k,i,a,k',j,b] = -W_A_full[k,i,j,k',b,a]
    # Source axes: (k=0, i=1, j=2, k'=3, b=4, a=5)
    # Target axes: (k=0, i=1, a=5, k'=3, j=2, b=4)  -> permutation (0,1,5,3,2,4)
    contrib = -W_A_full.transpose(0, 1, 5, 3, 2, 4).reshape(N, N)
    return contrib


def build_W_B_contrib(VQ_mo, Pi_stat, nocc, nvirt):
    """
    Compute the -W_B contribution to the B block for all k-point pairs.

    W_B arises from the ov-ov density-fitting product (cf. Eq. 32c):
        W_B[k,i,a; k',j,b] = (1/Nk) sum_{QQ'} V_{ia}^{k,Q} (I+Pi)_{QQ'} V_{jb}^{k',Q'}

    Enters the B matrix as:  B[k,i,a; k',j,b] -= W_B[k,i,b; k',j,a]
    (the a<->b transposition follows the molecular convention in matEleBStat).

    Returns
    -------
    contrib : ndarray, shape (N, N)
        Ready-to-add negative-W_B matrix.
    """
    nk, nQ = VQ_mo.shape[:2]
    N = nk * nocc * nvirt

    # V_ov[k,Q,i,a] -> flat (N, nQ) with flat index = k*nocc*nvirt + i*nvirt + a
    V_ov = VQ_mo[:, :, :nocc, nocc:]                              # (nk, nQ, nocc, nvirt)
    V_ov_flat = V_ov.transpose(0, 2, 3, 1).reshape(N, nQ)
    #  row = k * nocc*nvirt + i * nvirt + a

    W_eff = np.eye(nQ, dtype=np.complex128) + Pi_stat

    # W_B_flat[k*N1+i*nvirt+a, k'*N1+j*nvirt+b] = (1/Nk) V_ov[k,i,a] . W_eff . V_ov[k',j,b]
    PV = W_eff @ V_ov_flat.T                                      # (nQ, N)
    W_B_flat = (1.0 / nk) * (V_ov_flat @ PV)                     # (N, N)

    # Reshape to (nk, nocc, nvirt, nk, nocc, nvirt) axes (k,i,a,k',j,b)
    N1 = nocc * nvirt
    W_B_full = W_B_flat.reshape(nk, nocc, nvirt, nk, nocc, nvirt)

    # Map to B:  B_contrib[k,i,a,k',j,b] = -W_B_full[k,i,b,k',j,a]
    # Source axes: (k=0, i=1, a=2, k'=3, j=4, b=5)
    # Target:  (k=0, i=1, a=5, k'=3, j=4, b=2)  -> permutation (0,1,5,3,4,2)
    contrib = -W_B_full.transpose(0, 1, 5, 3, 4, 2).reshape(N, N)
    return contrib


# ---------------------------------------------------------------------------
# Finite-q, finite-temperature extensions
# ---------------------------------------------------------------------------

def fermi_dirac(eps, beta, mu):
    """
    Fermi-Dirac occupation f(ε) = [exp(β(ε − μ)) + 1]^{−1}.

    Numerically stable: clamps β(ε−μ) to [−500, 500] to avoid overflow.
    Use a large beta (e.g. 1e6) to approximate the T = 0 step function.
    """
    x = np.clip(beta * (np.asarray(eps, dtype=float) - mu), -500.0, 500.0)
    return 1.0 / (np.exp(x) + 1.0)


def fermi_weights_kq(valsMO, nelec, beta, mu, kq_map):
    """
    Occupation-difference array Δf_{k,i,a} = f(ε_{i,k}) − f(ε_{a,k+q})
    for all particle-hole pairs in the periodic BSE.

    Δf → 1 at T = 0 for states well inside a gap.
    Δf < 1 near the Fermi level at finite T, suppressing excitations from
    partially occupied states.

    Parameters
    ----------
    valsMO : ndarray, shape (ns, nk, nao)   QP energies.
    nelec  : int
    beta   : float   inverse temperature (1/k_B T).
    mu     : float   chemical potential.
    kq_map : ndarray, shape (nk,)
        kq_map[k] = index of k+q.  Use np.arange(nk) for q = 0.

    Returns
    -------
    Df : ndarray, shape (N,)  N = nk * nocc * nvirt
    """
    nk    = valsMO.shape[1]
    nao   = valsMO.shape[2]
    nocc  = nelec // 2
    nvirt = nao - nocc
    N     = nk * nocc * nvirt

    Df = np.zeros(N, dtype=float)
    for ik in range(nk):
        ikq = kq_map[ik]
        eps_holes    = valsMO[0, ik,  :nocc].real          # (nocc,)
        eps_particles = valsMO[0, ikq, nocc:].real          # (nvirt,)
        f_h = fermi_dirac(eps_holes,    beta, mu)           # (nocc,)
        f_p = fermi_dirac(eps_particles, beta, mu)          # (nvirt,)
        for i in range(nocc):
            base = ik * nocc * nvirt + i * nvirt
            Df[base:base + nvirt] = f_h[i] - f_p           # broadcast over a
    return Df


def VQ_ao2mo_kq(VQ_ao, vexMO, kq_map):
    """
    AO-to-MO transform for the finite-q transition-density tensor.

    For each k-point the hole MOs at k are applied on the left and the
    particle MOs at k+q on the right:

        V^{k,k+q}_{Q,mn} = Σ_{μν} C^*_{μm}(k) · V^k_{Q,μν} · C_{νn}(k+q)

    The AO integrals VQ_ao[k] are used on the k-side of the integral.
    For q = 0 this is identical to VQ_ao2mo_k.

    Parameters
    ----------
    VQ_ao  : ndarray, shape (nk, nQ, nao, nao)
    vexMO  : ndarray, shape (ns, nk, nao, nao)
    kq_map : ndarray, shape (nk,)

    Returns
    -------
    VQ_mo_kq : ndarray, shape (nk, nQ, nao, nao)
        VQ_mo_kq[k, Q, i, a] = transition matrix element for hole i@k
        and particle a@(k+q).
    """
    nk, nQ, nao, _ = VQ_ao.shape
    VQ_mo_kq = np.zeros_like(VQ_ao, dtype=np.complex128)
    for ik in range(nk):
        ikq  = kq_map[ik]
        C_k  = vexMO[0, ik]    # MO coefficients at k   (hole)
        C_kq = vexMO[0, ikq]   # MO coefficients at k+q (particle)
        for iQ in range(nQ):
            VQ_mo_kq[ik, iQ] = C_k.conj().T @ VQ_ao[ik, iQ] @ C_kq
    return VQ_mo_kq


def diffEpsVec_kq(valsMO, nelec, kq_map):
    """
    Single-particle energy differences ε_{a,k+q} − ε_{i,k} as a 1-D vector.

    For q = 0 (kq_map = identity) this equals the diagonal of diffEpsMat_k.

    Returns
    -------
    dEps : ndarray, shape (N,)  complex128
    """
    nk    = valsMO.shape[1]
    nao   = valsMO.shape[2]
    nocc  = nelec // 2
    nvirt = nao - nocc
    N     = nk * nocc * nvirt

    dEps = np.zeros(N, dtype=np.complex128)
    for ik in range(nk):
        ikq = kq_map[ik]
        eps_holes    = valsMO[0, ik,  :nocc]    # (nocc,)
        eps_particles = valsMO[0, ikq, nocc:]    # (nvirt,)
        for i in range(nocc):
            base = ik * nocc * nvirt + i * nvirt
            dEps[base:base + nvirt] = eps_particles - eps_holes[i]
    return dEps


def build_W_A_contrib_kq(VQ_mo, kq_map, Pi_stat, nocc, nvirt):
    """
    W_A attraction for the A block at finite q.

    At finite q the hole lives at k and the particle at k+q, so:
      - occ-occ (hole-hole) integrals use the pure-k MO basis: VQ_mo[k]
      - virt-virt (particle-particle) integrals use VQ_mo[k+q] = VQ_mo[kq_map[k]]

    W_A[k,i,j; k',b,a] = (1/Nk) V^k_{ij} · W_eff · (V^{k'+q}_{ba})^T

    For q = 0 (kq_map = identity) this is identical to build_W_A_contrib.

    Parameters
    ----------
    VQ_mo  : ndarray, shape (nk, nQ, nao, nao)   pure k-point MO basis.
    kq_map : ndarray, shape (nk,)
    Pi_stat, nocc, nvirt : as in build_W_A_contrib.
    """
    nk, nQ = VQ_mo.shape[:2]
    N = nk * nocc * nvirt

    # Hole sector: occ-occ at k — same as before
    V_oo_flat = VQ_mo[:, :, :nocc, :nocc].transpose(0, 2, 3, 1).reshape(nk * nocc * nocc, nQ)

    # Particle sector: virt-virt at k+q (index via kq_map)
    V_vv_kq   = np.stack([VQ_mo[kq_map[ik], :, nocc:, nocc:] for ik in range(nk)], axis=0)
    V_vvT_flat = V_vv_kq.transpose(0, 2, 3, 1).reshape(nk * nvirt * nvirt, nQ)

    W_eff    = np.eye(nQ, dtype=np.complex128) + Pi_stat
    PV       = W_eff @ V_vvT_flat.T
    W_A_flat = (1.0 / nk) * (V_oo_flat @ PV)
    W_A_full = W_A_flat.reshape(nk, nocc, nocc, nk, nvirt, nvirt)
    return -W_A_full.transpose(0, 1, 5, 3, 2, 4).reshape(N, N)


def build_W_B_contrib_kq(VQ_mo_kq, Pi_stat, nocc, nvirt):
    """
    W_B attraction for the B block at finite q.

    Uses the finite-q transition-density VQ_mo_kq for both ov factors.
    For q = 0 this is identical to build_W_B_contrib.

    Parameters
    ----------
    VQ_mo_kq : ndarray, shape (nk, nQ, nao, nao)   transition-density MO basis.
    """
    nk, nQ = VQ_mo_kq.shape[:2]
    N      = nk * nocc * nvirt

    V_ov_flat = VQ_mo_kq[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, nQ)
    W_eff     = np.eye(nQ, dtype=np.complex128) + Pi_stat
    PV        = W_eff @ V_ov_flat.T
    W_B_flat  = (1.0 / nk) * (V_ov_flat @ PV)
    W_B_full  = W_B_flat.reshape(nk, nocc, nvirt, nk, nocc, nvirt)
    return -W_B_full.transpose(0, 1, 5, 3, 4, 2).reshape(N, N)


def solveHstatic_kq(Pi_stat, VQ_mo_kq, VQ_mo, valsMO, nelec, kq_map,
                    beta=None, mu=0.0, ex_type="singlet", tda=False):
    """
    Static BSE Casida Hamiltonian for finite-q and/or finite-temperature
    periodic systems.

    Generalization of solveHstatic_k that implements two extensions:

    1. **Finite momentum transfer q** via kq_map: the conduction band sits
       at k+q instead of k, so the kinetic term is ε_{m,k+q} − ε_{n,k}.
       VQ_mo_kq contains the transition densities with hole MOs at k and
       particle MOs at k+q (call VQ_ao2mo_kq first); VQ_mo is the pure
       k-point transform used for the W_A attraction (oo at k, vv at k+q).

    2. **Finite temperature** via Fermi-Dirac weights: the interaction kernel
       is multiplied row-wise by Δf_{k,i,a} = f(ε_{i,k}) − f(ε_{a,k+q}):

           H_{SS'} = Δε_S δ_{SS'} + Δf_S · [κ/Nk U_{SS'} − W_{A,SS'}]

       At T = 0, Δf = 1 and this reduces exactly to solveHstatic_k.

    Parameters
    ----------
    Pi_stat  : ndarray, shape (nQ, nQ)   static q = 0 polarization.
    VQ_mo_kq : ndarray, shape (nk, nQ, nao, nao)
        Finite-q transition-density in MO basis (call VQ_ao2mo_kq first).
        For q = 0 this equals VQ_mo (same C on both sides).
    VQ_mo    : ndarray, shape (nk, nQ, nao, nao)
        Pure k-point MO-basis Coulomb tensor (C_k on both sides at every k).
        Used for the W_A oo (hole) sector and, via kq_map, the vv (particle).
    valsMO   : ndarray, shape (ns, nk, nao)   QP energies.
    nelec    : int
    kq_map   : ndarray, shape (nk,)
        kq_map[k] = index of k+q.  Use np.arange(nk) for q = 0.
    beta     : float or None
        Inverse temperature.  None → T = 0 (Δf = 1 exactly).
    mu       : float   chemical potential (default 0).
    ex_type  : "singlet" or "triplet".
    tda      : bool   Tamm-Dancoff approximation (B = 0).

    Returns
    -------
    effVals : ndarray, shape (2N,)
    effVex  : ndarray, shape (2N, 2N)
    H_stat  : ndarray, shape (2N, 2N)

    See also
    --------
    solveHstatic_k_finiteT : convenience wrapper for the common q = 0 case.
    VQ_ao2mo_kq            : AO-to-MO transform for finite-q transition density.
    """
    nk, nQ, nao, _ = VQ_mo_kq.shape
    nocc  = nelec // 2
    nvirt = nao - nocc
    N     = nk * nocc * nvirt
    kappa = 2.0 if ex_type == "singlet" else 0.0

    beta_str = "∞ (T=0)" if beta is None else f"{beta:.1f}"
    print(f"  Periodic BSE (finite-q/T): Nk={nk}, nocc={nocc}, nvirt={nvirt}, "
          f"N={N}, 2N={2*N}, β={beta_str}")

    # --- Kinetic diagonal: ε_{a,k+q} − ε_{i,k} ---
    dEps = diffEpsVec_kq(valsMO, nelec, kq_map)

    # --- Fermi-Dirac occupation weights ---
    Df = (fermi_weights_kq(valsMO, nelec, beta, mu, kq_map)
          if beta is not None
          else np.ones(N, dtype=float))

    # --- Transition-density ov slice (shared by exchange and W_B) ---
    V_ov_flat = VQ_mo_kq[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, nQ)

    # --- A-block interaction kernel K_A = (κ/Nk) U + W_A  (W_A < 0) ---
    K_A = np.zeros((N, N), dtype=np.complex128)
    if kappa != 0.0:
        K_A += (kappa / nk) * (V_ov_flat @ V_ov_flat.T)
    K_A += build_W_A_contrib_kq(VQ_mo, kq_map, Pi_stat, nocc, nvirt)

    # A = diag(Δε) + diag(Δf) · K_A
    A = np.diag(dEps) + Df[:, None] * K_A

    # --- B block ---
    if tda:
        B = np.zeros((N, N), dtype=np.complex128)
    else:
        K_B = np.zeros((N, N), dtype=np.complex128)
        if kappa != 0.0:
            V_bj_flat = VQ_mo_kq[:, :, nocc:, :nocc].transpose(0, 3, 2, 1).reshape(N, nQ)
            K_B += (kappa / nk) * (V_ov_flat @ V_bj_flat.T)
        K_B += build_W_B_contrib_kq(VQ_mo_kq, Pi_stat, nocc, nvirt)
        B = Df[:, None] * K_B

    H_stat = concatAB(A, B)

    cond = np.linalg.cond(H_stat)
    print(f"  Solving non-Hermitian eigenvalue equation, cond = {cond:.4f}")

    H_balanced, scale = matrix_balance(H_stat)
    effVals, effVex   = LA.eig(H_balanced)
    effVex            = LA.solve(scale, effVex)

    idx = np.argmax(abs(effVex.real), axis=0)
    effVex[:, effVex[idx, np.arange(len(effVals))].real < 0] *= -1

    return effVals, effVex, H_stat


def solveHstatic_k_finiteT(Pi_stat, VQ_mo, valsMO, nelec, beta, mu=0.0,
                            ex_type="singlet", tda=False):
    """
    Finite-temperature BSE at q = 0.  Convenience wrapper around
    solveHstatic_kq with kq_map = identity (no momentum transfer).

    At beta → ∞ this is identical to solveHstatic_k.

    Parameters
    ----------
    Pi_stat, VQ_mo, valsMO, nelec, ex_type, tda : same as solveHstatic_k.
    beta : float   inverse temperature (must be provided; use 1e6 for T ≈ 0).
    mu   : float   chemical potential (default 0).
    """
    nk     = VQ_mo.shape[0]
    kq_map = np.arange(nk, dtype=int)
    # At q=0 the transition density equals the pure-k VQ (same MOs on both sides)
    return solveHstatic_kq(
        Pi_stat, VQ_mo, VQ_mo, valsMO, nelec, kq_map,
        beta=beta, mu=mu, ex_type=ex_type, tda=tda,
    )


# ---------------------------------------------------------------------------
# Static BSE Hamiltonian
# ---------------------------------------------------------------------------

def solveHstatic_k(Pi_stat, VQ, valsMO, nelec, ex_type="singlet", tda=False):
    """
    Build and diagonalize the static BSE Casida Hamiltonian for a periodic solid.

    This is the k-point generalization of casidaEq.solveHstatic.  The full
    Hamiltonian has dimension 2N x 2N with N = Nk * nocc * nvirt.

    Parameters
    ----------
    Pi_stat : ndarray, shape (nQ, nQ)
        Static (Ω=0) auxiliary-basis polarization at q=0 (from scGW).
    VQ : ndarray, shape (nk, nQ, nao, nao)
        Density-fitting Coulomb integrals in MO basis (call VQ_ao2mo_k first).
    valsMO : ndarray, shape (ns, nk, nao)
        QP energies; spin index 0 is used.
    nelec : int
        Total number of electrons.
    ex_type : str
        "singlet" (kappa=2) or "triplet" (kappa=0).
    tda : bool
        If True, apply Tamm-Dancoff approximation (B=0).

    Returns
    -------
    effVals : ndarray, shape (2N,)
        BSE eigenvalues (complex in general; positive real parts are excitations).
    effVex : ndarray, shape (2N, 2N)
        BSE eigenvectors.
    H_stat : ndarray, shape (2N, 2N)
        Full static BSE Hamiltonian (for bookkeeping).
    """
    nk = VQ.shape[0]
    nao = VQ.shape[2]
    nocc = nelec // 2
    nvirt = nao - nocc
    N = nk * nocc * nvirt
    kappa = 2.0 if ex_type == "singlet" else 0.0

    print(f"  Periodic BSE: Nk={nk}, nocc={nocc}, nvirt={nvirt}, N={N}, 2N={2*N}")

    # --- A block ---
    A = diffEpsMat_k(valsMO, nelec)                               # (N, N) kinetic

    # Exchange in A: κ/Nk * sum_Q V_ov[k,Q,i,a] V_ov[k',Q,j,b]
    if kappa != 0.0:
        V_ov_flat = VQ[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, VQ.shape[1])
        A += (kappa / nk) * (V_ov_flat @ V_ov_flat.T)

    # Attraction in A: -W_A (includes bare U + screened P̃ term)
    A += build_W_A_contrib(VQ, Pi_stat, nocc, nvirt)

    # --- B block ---
    if tda:
        B = np.zeros((N, N), dtype=np.complex128)
    else:
        B = np.zeros((N, N), dtype=np.complex128)

        # Exchange in B: κ/Nk * sum_Q V_ov[k,Q,i,a] V_vo[k',Q,b,j]
        # V_vo reindexed so flat row = k*nocc*nvirt + j*nvirt + b
        if kappa != 0.0:
            V_ov_flat = VQ[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, VQ.shape[1])
            # V_bj: VQ_mo[k',Q,nocc+b,j] reordered to (k',j,b,Q) -> flat row = k'*N1+j*nvirt+b
            V_bj_flat = VQ[:, :, nocc:, :nocc].transpose(0, 3, 2, 1).reshape(N, VQ.shape[1])
            B += (kappa / nk) * (V_ov_flat @ V_bj_flat.T)

        # Attraction in B: -W_B
        B += build_W_B_contrib(VQ, Pi_stat, nocc, nvirt)

    H_stat = concatAB(A, B)

    cond = np.linalg.cond(H_stat)
    print(f"  Solving non-Hermitian eigenvalue equation, cond = {cond:.4f}")

    H_balanced, scale = matrix_balance(H_stat)
    effVals, effVex = LA.eig(H_balanced)
    effVex = LA.solve(scale, effVex)

    # Fix sign ambiguity
    idx = np.argmax(abs(effVex.real), axis=0)
    effVex[:, effVex[idx, np.arange(len(effVals))].real < 0] *= -1

    return effVals, effVex, H_stat


# ---------------------------------------------------------------------------
# Davidson / iterative TDA diagonalization
# ---------------------------------------------------------------------------

def _build_A_matvec(VQ, Pi_stat, valsMO, nelec, ex_type):
    """
    Return a closure that applies the TDA A-block to a vector without ever
    forming the full N×N matrix.  Used as the matvec for Davidson (eigsh).

    The A-block is:  A = diag(Δε) + (κ/Nk) V_ov V_ov^T  -  W_A
    where W_A = -(1/Nk) V_oo (I + Pi) V_vv^T   (built explicitly once).

    Parameters
    ----------
    VQ : ndarray, shape (nk, nQ, nao, nao)   MO-basis Coulomb tensor.
    Pi_stat : ndarray, shape (nQ, nQ)         Static polarization at q=0.
    valsMO  : ndarray, shape (ns, nk, nao)    QP energies.
    nelec   : int
    ex_type : str   "singlet" or "triplet"

    Returns
    -------
    matvec : callable, shape (N,) -> (N,)
    N      : int   dimension of the TDA space
    diag_A : ndarray, shape (N,)  diagonal of A (used as Davidson preconditioner)
    """
    nk, nQ, nao, _ = VQ.shape
    nocc  = nelec // 2
    nvirt = nao - nocc
    N     = nk * nocc * nvirt
    kappa = 2.0 if ex_type == "singlet" else 0.0

    # Kinetic diagonal  Δε_{k,i,a} = ε_{k,a} - ε_{k,i}
    diag_A = np.array([
        valsMO[0, ik, nocc + a] - valsMO[0, ik, i]
        for ik in range(nk)
        for i  in range(nocc)
        for a  in range(nvirt)
    ], dtype=np.complex128)

    # Flat ov slices: shape (N, nQ)
    V_ov = VQ[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, nQ)   # (N, nQ)

    # Pre-build the W_A contribution matrix once: shape (N, N)
    # W_A[ρ,σ] = (1/Nk) * sum_Q  V_oo[ρ_oo,Q] * (I+Pi)_QP * V_vv[σ_vv,P]
    # (build_W_A_contrib already does this; we call it once and store)
    W_A = build_W_A_contrib(VQ, Pi_stat, nocc, nvirt)   # (N, N), stored once

    # Exchange term V_ov V_ov^T is a rank-nQ outer product; apply as two
    # sequential matrix-vector products to stay O(N·nQ) per matvec.
    exchange_prefactor = kappa / nk if kappa != 0.0 else 0.0

    def matvec(v):
        v = np.asarray(v, dtype=np.complex128)
        # Kinetic
        result = diag_A * v
        # Exchange: (κ/Nk) V_ov (V_ov^T v)  — O(N·nQ) instead of O(N²)
        if exchange_prefactor != 0.0:
            result += exchange_prefactor * (V_ov @ (V_ov.T @ v))
        # Attraction: W_A v  (dense, O(N²), but built once above)
        result += W_A @ v
        return result

    return matvec, N, diag_A.real


def solveHstatic_k_davidson(Pi_stat, VQ, valsMO, nelec,
                             ex_type="singlet", n_roots=10, tol=1e-8):
    """
    TDA-only iterative Davidson diagonalization of the periodic BSE A-block.

    Uses scipy's ARPACK-backed ``eigsh`` with a ``LinearOperator`` so the
    full N×N matrix is **never explicitly formed or stored**.  The exchange
    term is applied via two O(N·nQ) matrix-vector products; only the W_A
    attraction block (N×N dense) is precomputed and cached.

    This replaces the O(N³) dense ``LA.eig`` call in ``solveHstatic_k``
    when only the lowest few singlet/triplet excitation energies are needed.

    Parameters
    ----------
    Pi_stat : ndarray, shape (nQ, nQ)
        Static (Ω=0) auxiliary-basis polarization at q=0.
    VQ : ndarray, shape (nk, nQ, nao, nao)
        Density-fitting Coulomb integrals in MO basis (call VQ_ao2mo_k first).
    valsMO : ndarray, shape (ns, nk, nao)
        QP energies; spin index 0 is used.
    nelec : int
        Total number of electrons.
    ex_type : str
        "singlet" (κ=2) or "triplet" (κ=0).
    n_roots : int
        Number of lowest excitation energies to compute.
    tol : float
        Convergence tolerance passed to eigsh (ARPACK).

    Returns
    -------
    effVals : ndarray, shape (n_roots,)
        Lowest TDA excitation energies (real, in Hartree), sorted ascending.
    effVex  : ndarray, shape (N, n_roots)
        Corresponding TDA eigenvectors (X amplitudes only; Y=0 in TDA).
    A_diag  : ndarray, shape (N,)
        Diagonal of A, useful as a Davidson preconditioner or first-order
        correction when adding the B block perturbatively later.

    Notes
    -----
    * TDA only (B = 0).  For the full BSE use ``solveHstatic_k``.
    * The A-block is real-symmetric for real VQ (Γ-point / GTO basis), so
      ``eigsh`` is exact (not approximate).  For complex VQ (PW / k≠Γ) it
      is still Hermitian and eigsh handles it correctly.
    * Increase ``n_roots`` generously (e.g. 2× the number of peaks you need)
      because oscillator strengths can be tiny near the edge.
    """
    nk, nQ, nao, _ = VQ.shape
    nocc  = nelec // 2
    nvirt = nao - nocc
    N     = nk * nocc * nvirt

    print(f"  Periodic BSE (Davidson/TDA): Nk={nk}, nocc={nocc}, nvirt={nvirt}, "
          f"N={N}, n_roots={n_roots}")

    if n_roots >= N:
        raise ValueError(
            f"n_roots={n_roots} must be strictly less than N={N}. "
            "Use solveHstatic_k for a full diagonalization instead."
        )

    matvec, N, diag_A = _build_A_matvec(VQ, Pi_stat, valsMO, nelec, ex_type)

    # Wrap as a LinearOperator so eigsh never sees the matrix entries directly
    A_op = LinearOperator((N, N), matvec=matvec, dtype=np.complex128)

    # Initial guess: unit vectors along the n_roots smallest diagonal entries
    # (these are the dominant particle-hole pairs — much better than random)
    idx0  = np.argsort(diag_A)[:n_roots]
    v0    = np.zeros((N, n_roots), dtype=np.complex128)
    for col, row in enumerate(idx0):
        v0[row, col] = 1.0

    effVals, effVex = eigsh(A_op, k=n_roots, which='SA', tol=tol, v0=v0[:, 0])

    # eigsh returns ascending order by default; enforce it explicitly
    order   = np.argsort(effVals)
    effVals = effVals[order].real
    effVex  = effVex[:, order]

    print(f"  Davidson converged. Lowest excitation: {effVals[0]:.6f} Ha "
          f"({effVals[0]*27.2114:.4f} eV)")

    return effVals, effVex, diag_A


# ---------------------------------------------------------------------------
# Dynamic effective Hamiltonian (diagonal approximation)
# ---------------------------------------------------------------------------

def _build_H_dyn_freq_k(iw, Pi, VQ, nocc, nvirt, effVex_inv, effVex,
                         diffEps_diag, nelec, ex_type, U_ov, U_oo_vv):
    """
    Compute the diagonal of V^{-1} H_eff(iΩ_iw) V at a single frequency point.

    This is the k-point generalisation of casidaEq._process_hdyn_frequency.
    Precomputed U_ov and U_oo_vv (bare Coulomb terms) are reused to avoid
    redundant einsum contractions inside the frequency loop.

    Parameters
    ----------
    iw : int
        Frequency index into Pi.
    Pi : ndarray, shape (niw, ns, nk_q, nQ, nQ) or (niw, ns, nQ, nQ, nk_q)
        Frequency-dependent auxiliary-basis polarization at q=0.
        The q=0 component is extracted as Pi[iw, 0, 0, :, :].
    VQ : ndarray, shape (nk, nQ, nao, nao)
        Coulomb integrals in MO basis.
    nocc, nvirt : int
    effVex_inv : ndarray, shape (2N, 2N)  inverse of static eigenvector matrix
    effVex     : ndarray, shape (2N, 2N)  static eigenvector matrix
    diffEps_diag : ndarray, shape (N,)    diagonal of kinetic matrix
    nelec : int
    ex_type : str
    U_ov : ndarray, shape (N, N)   precomputed bare-Coulomb exchange for A
    U_oo_vv : ndarray, shape (N, N) precomputed bare occ-occ x virt-virt for A

    Returns
    -------
    h2p : ndarray, shape (2N,)
        Diagonal of V^{-1} H_eff(iΩ) V.
    """
    nk, nQ = VQ.shape[:2]
    N = nk * nocc * nvirt
    kappa = 2.0 if ex_type == "singlet" else 0.0

    # Extract q=0 polarization at this frequency
    # Pi shape from contract.py: (niw, ns, nQ, nQ, nk_q=1) or similar
    # Use the same convention as the molecular code: Pi[iw, 0, :, :, 0]
    if Pi.ndim == 5 and Pi.shape[-1] == 1:
        Pi_iw = Pi[iw, 0, :, :, 0]
    elif Pi.ndim == 5 and Pi.shape[2] > 1:
        # shape (niw, ns, nk, nQ, nQ): use q=0 component
        Pi_iw = Pi[iw, 0, 0, :, :]
    else:
        Pi_iw = Pi[iw].reshape(nQ, nQ)

    # --- A block at this frequency ---
    A = np.diag(diffEps_diag.astype(np.complex128))
    if kappa != 0.0:
        A += (kappa / nk) * U_ov
    A += build_W_A_contrib(VQ, Pi_iw, nocc, nvirt)

    # --- B block at this frequency ---
    V_ov_flat = VQ[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, nQ)
    B = build_W_B_contrib(VQ, Pi_iw, nocc, nvirt)
    if kappa != 0.0:
        V_bj_flat = VQ[:, :, nocc:, :nocc].transpose(0, 3, 2, 1).reshape(N, nQ)
        B += (kappa / nk) * (V_ov_flat @ V_bj_flat.T)

    H = concatAB(A, B)
    return np.einsum('ij,jk,ki->i', effVex_inv, H, effVex)


def HDynDiagApprox_k(Pi, effVex, VQ, valsMO, nelec, ex_type="singlet", n_jobs=-1):
    """
    Diagonal approximation to the frequency-dependent BSE Hamiltonian for periodic systems.

    Mirrors casidaEq.HDynDiagApprox but uses k-resolved VQ and the full
    Nk*nocc*nvirt pair space.

    The static eigenvector matrix effVex (from solveHstatic_k) diagonalizes
    H_stat.  We assume it also approximately diagonalizes H_eff(iΩ) at all
    frequencies (adiabatic approximation, Eq. 36 of the manuscript).

    Parameters
    ----------
    Pi : ndarray
        Frequency-dependent q=0 polarization; see _build_H_dyn_freq_k for shape.
    effVex : ndarray, shape (2N, 2N)
        Eigenvectors from solveHstatic_k.
    VQ : ndarray, shape (nk, nQ, nao, nao)
        Coulomb integrals in MO basis (already transformed).
    valsMO : ndarray, shape (ns, nk, nao)
    nelec : int
    ex_type : str
    n_jobs : int
        Joblib parallelism over frequency points.

    Returns
    -------
    H2p_Dyn : ndarray, shape (niw, 2N)
        Diagonal of V^{-1} H_eff(iΩ_n) V at all n (symmetrised).
    """
    nk, nQ, nao = VQ.shape[:3]
    nocc = nelec // 2
    nvirt = nao - nocc
    N = nk * nocc * nvirt
    kappa = 2.0 if ex_type == "singlet" else 0.0

    niw = Pi.shape[0]
    niw_half = niw // 2 + 1

    effVex_inv = LA.inv(effVex)

    # Precompute kinetic diagonal
    diffEps_diag = np.array([
        valsMO[0, ik, nocc + a] - valsMO[0, ik, i]
        for ik in range(nk)
        for i in range(nocc)
        for a in range(nvirt)
    ])

    # Precompute bare-Coulomb exchange (frequency-independent)
    V_ov_flat = VQ[:, :, :nocc, nocc:].transpose(0, 2, 3, 1).reshape(N, nQ)
    U_ov = (1.0 / nk) * (V_ov_flat @ V_ov_flat.T) if kappa != 0.0 else None

    # Bare occ-occ x virt-virt (for reference; not separately needed since build_W_A
    # uses I+Pi internally, but we keep for clarity)
    U_oo_vv = None  # absorbed into build_W_A_contrib via W_eff = I + Pi

    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_build_H_dyn_freq_k)(
            iw, Pi, VQ, nocc, nvirt, effVex_inv, effVex,
            diffEps_diag, nelec, ex_type, U_ov, U_oo_vv
        )
        for iw in range(niw_half)
    )

    H2p_Dyn = np.zeros((niw, 2 * N), dtype=np.complex128)
    for iw, result in enumerate(results):
        H2p_Dyn[iw] = result
    # Bosonic symmetry: H(iΩ) = H(-iΩ)*
    for iw, result in enumerate(results):
        H2p_Dyn[niw - 1 - iw] = result

    # Symmetrise across positive/negative frequencies
    for iw in range(niw_half):
        H2p_Dyn[iw] = 0.5 * (H2p_Dyn[iw] + H2p_Dyn[niw - 1 - iw].conj())
        H2p_Dyn[niw - 1 - iw] = H2p_Dyn[iw].conj()

    return H2p_Dyn


# ---------------------------------------------------------------------------
# Quasiparticle correction for periodic systems
# ---------------------------------------------------------------------------

def qp_energies_k(Sigma_iw, valsMO, beta, mu, ir_file):
    """
    Apply the QP approximation at each k-point using Pade interpolation.
    Thin wrapper around qp.padeSigma, which already loops over k.

    Parameters
    ----------
    Sigma_iw : ndarray, shape (niw, ns, nk, nao, nao)
        Self-energy in MO basis.
    valsMO : ndarray, shape (ns, nk, nao)
        Initial Fock/scGW eigenvalues.
    beta, mu : float
    ir_file : str

    Returns
    -------
    valsMO_qp : ndarray, shape (ns, nk, nao)
    """
    import qp
    return qp.padeSigma(Sigma_iw, valsMO, beta, mu, ir_file)
