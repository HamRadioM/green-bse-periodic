#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    QP-FREE two-step active-space BSE.                                       #
#                                                                             #
#    The standard two-step scheme (active_bse.py) builds the active-space     #
#    particle-hole ladder from SHARP quasiparticle poles                      #
#        G^{2p}(iW) = (iW - Delta - K)^{-1},   Delta_ia = eps_a - eps_i,      #
#    so it needs QP energies eps_p as input.  This module removes that:       #
#    NO orbital energies appear anywhere.  Instead                            #
#                                                                             #
#      1. the active frame is the set of NATURAL ORBITALS of the dressed      #
#         one-body density matrix  gamma = -G(beta^-)  (occupations n_p);     #
#      2. the active bare polarizability Pi^0(iW) is built directly from the  #
#         dressed imaginary-time G(tau) in that frame (eval_Pi0_MO_active     #
#         style) -- it carries the full spectral content of G;               #
#      3. the screened -W ladder is summed as a matrix DYSON equation in the  #
#         small active particle-hole space,                                   #
#             Pi^act = [I - Pi^0 K_W]^{-1} Pi^0,                              #
#         which reproduces the sharp-pole ladder when fed a QP G but needs no #
#         energies for a dressed G;                                          #
#      4. everything is projected to Q-space and combined with the dressed    #
#         full-space backbone P^(0) = eval_P0_tilde_Q exactly as before.      #
#                                                                             #
#    The only single-particle input is the dressed G(tau) itself (plus the    #
#    AO overlap S).  Scaling is unchanged: the O(n_act^6) Dyson solve is      #
#    confined to a small, fixed-size active space; the rest is NQ x NQ.       #
#                                                                             #
#    Conventions (verified to ~1e-10, see validate_qp_free_bse.py)            #
#    -----------                                                              #
#    * spin-summed FULL bubble: Pi^0 carries the "-2" spin factor, so the     #
#      singlet Q-space RPA uses spin_factor=1 (active_bse.qspace_rpa).        #
#    * V is transformed C-only (no overlap S): V_no = C^dag V_ao C, matching  #
#      active_bse; G is transformed with SC (eval_Pi0_MO_active).             #

import numpy as np
import h5py

import gwtool
import contract as ct
from active_bse import bubble_replacement, qspace_rpa


# ===========================================================================
# 1. Natural orbitals of the dressed one-body density matrix
# ===========================================================================
def natural_orbitals(G_tau, S):
    """
    Natural orbitals / occupations from the dressed density matrix
    gamma = -G(beta^-) (the last imaginary-time slice), solving the generalized
    eigenproblem  gamma S C = C n  in the S metric.

    Returns
    -------
    occ : (nao,)  occupations per spin, descending (n_p in [0,1]).
    C   : (nao, nao)  natural-orbital coefficients, S-orthonormal (C^dag S C = I).
    """
    nao = S.shape[0]
    gamma = -G_tau[-1, 0, 0].reshape(nao, nao)
    sval, svec = np.linalg.eigh(S)
    Shalf = (svec * np.sqrt(sval)) @ svec.conj().T
    Sihalf = (svec / np.sqrt(sval)) @ svec.conj().T
    occ, u = np.linalg.eigh(Shalf @ gamma @ Shalf)
    occ, u = occ[::-1], u[:, ::-1]
    C = Sihalf @ u
    return occ.real, C


def active_by_occupation(occ, tol=1e-3):
    """
    Active orbitals = those whose occupation is fractional (away from 0 and 1
    by more than `tol`), i.e. the correlated frontier orbitals.  The "activity"
    n_p (1 - n_p) peaks at the Fermi level.

    Returns the sorted active index array.
    """
    activity = occ * (1.0 - occ)
    return np.where(activity > tol)[0]


# ===========================================================================
# 2. Active bare polarizability Pi^0(iW) from the dressed G(tau)
# ===========================================================================
def pi0_active(G_tau, C, S, active, ir):
    """
    Spin-summed full active bare polarizability, from the dressed G(tau):

        Pi^0_{pq,rs}(tau) = -2 G^MO(beta-tau)_{pr} G^MO(tau)_{qs},
        G^MO = (SC_act)^dag G_AO (SC_act),

    tau-symmetrized and Fourier transformed to the bosonic Matsubara grid.

    Returns
    -------
    Pi0 : (niw, na*na, na*na) complex, composite index (p,q) x (r,s).
    """
    active = np.asarray(active)
    na = active.size
    nao = S.shape[0]
    ntau = G_tau.shape[0]
    SC = (S @ C)[:, active].T                       # (na, nao)

    Pi0t = np.zeros((ntau, na, na, na, na), dtype=np.complex128)
    for t in range(ntau // 2):
        tt = ntau - t - 1
        g1 = SC.conj() @ G_tau[tt, 0, 0].reshape(nao, nao) @ SC.T
        g2 = SC.conj() @ G_tau[t, 0, 0].reshape(nao, nao) @ SC.T
        Pi0t[t] -= 2.0 * np.einsum('pr,qs->pqrs', g1, g2, optimize=True)
    # tau symmetrization (mirror of gwtool.symmetrize_P0 on the pair index)
    for t in range(ntau // 2):
        tt = ntau - t - 1
        Pi0t[t] = 0.5 * (Pi0t[t] + Pi0t[t].conj().transpose(2, 3, 0, 1))
        Pi0t[tt] += Pi0t[t]

    Pi0w = ir.tauf_to_wb(Pi0t.reshape(ntau, na ** 4))
    return Pi0w.reshape(-1, na * na, na * na)


# ===========================================================================
# 3. Active DF integrals and screened -W kernel (full active pair space)
# ===========================================================================
def vno_active(VQ_ao, C, active):
    """Active MO DF integrals, C-only metric:  V_no[Q,p,q] = (C^dag V_ao C)_{pq}."""
    active = np.asarray(active)
    Ca = C[:, active]
    return np.einsum('mp,Qmn,nq->Qpq', Ca.conj(), VQ_ao[0], Ca, optimize=True)


def screened_kernel(V_no, M):
    """
    Screened-direct -W kernel in the full active pair space:

        K_{(p1 q1),(p2 q2)} = - sum_{QQ'} V[Q,p1,p2] M[Q,Q'] V[Q',q2,q1].

    With M = I + tildeP(0) the static screened Coulomb.  Restricted to an
    occ x virt block this equals active_bse.screened_direct_kernel.
    """
    na = V_no.shape[1]
    K4 = -np.einsum('Qab,QP,Pcd->adbc', V_no, M, V_no, optimize=True)  # [p1,q1,p2,q2]
    return K4.reshape(na * na, na * na)


# ===========================================================================
# 4. Active ladder by matrix Dyson + Q-space projection
# ===========================================================================
def ladder_dyson(Pi0, K):
    """Pi^act(iW) = [I - Pi^0(iW) K]^{-1} Pi^0(iW)  (per frequency)."""
    niw, n, _ = Pi0.shape
    I = np.eye(n, dtype=np.complex128)
    out = np.empty_like(Pi0)
    for iw in range(niw):
        out[iw] = np.linalg.solve(I - Pi0[iw] @ K, Pi0[iw])
    return out


def project_Q(V_no, Pi):
    """P_{QQ'}(iW) = sum V_no[Q,pq] Pi[(pq),(rs)] V_no*[Q',rs]."""
    NQ = V_no.shape[0]
    Vf = V_no.reshape(NQ, -1)                        # (NQ, na*na)
    return np.einsum('Qx,wxy,Ry->wQR', Vf, Pi, Vf.conj(), optimize=True)


# ===========================================================================
# High-level driver
# ===========================================================================
def two_step_bse_qp_free(G_tau, S, VQ_ao, tildeP_static, P0_backbone, ir,
                         active=None, occ_tol=1e-3, channel="singlet"):
    """
    QP-free two-step active-space BSE polarization in Q-space.

    Parameters
    ----------
    G_tau : (ntau, ns, nk, nao, nao)  dressed imaginary-time Green's function.
    S : (nao, nao)  AO overlap.
    VQ_ao : (1, NQ, nao, nao)  DF integrals in AO.
    tildeP_static : (NQ, NQ)  static dressed polarizability (M = I + tildeP).
    P0_backbone : (niw, NQ, NQ)  dressed full-space backbone (eval_P0_tilde_Q).
    ir : IR_factory.
    active : explicit active natural-orbital indices, or None to choose by
             occupation (active_by_occupation, threshold occ_tol).
    channel : "singlet" (apply +2v Q-space RPA, spin_factor=1) or "triplet".

    Returns
    -------
    P_irr, P_bse, info  where info = dict(occ=, active=, C=).
    """
    occ, C = natural_orbitals(G_tau, S)
    if active is None:
        active = active_by_occupation(occ, tol=occ_tol)

    V_no = vno_active(VQ_ao, C, active)
    M = np.eye(VQ_ao.shape[1], dtype=np.complex128) + tildeP_static
    K_W = screened_kernel(V_no, M)

    Pi0 = pi0_active(G_tau, C, S, active, ir)
    Pi_act = ladder_dyson(Pi0, K_W)

    P_act = project_Q(V_no, Pi_act)
    P_act0 = project_Q(V_no, Pi0)
    P_irr = bubble_replacement(P_act, P_act0, P0_backbone)

    info = dict(occ=occ, active=np.asarray(active), C=C)
    if channel == "triplet":
        return P_irr, P_irr, info
    P_bse = qspace_rpa(P_irr, spin_factor=1.0)
    return P_irr, P_bse, info
