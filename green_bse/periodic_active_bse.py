#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Periodic (k-point), finite-Q two-step active-space BSE in Q-space.       #
#                                                                             #
#    This is the crystalline generalization of active_bse.py.  The molecular  #
#    single-k machinery is lifted to Nk k-points and an arbitrary exciton     #
#    momentum Q_exc on the Born-von-Karman mesh, while keeping the SAME        #
#    three-step, RPA-scaling structure:                                       #
#                                                                             #
#      1.  P^act   = active-space screened -W ladder            (Q-space)     #
#      2.  P^irr   = P^act - P^act(0) + P^(0)                                 #
#      3.  P^BSE   = [I - f P^irr]^{-1} P^irr                   (Q-space RPA)  #
#                                                                             #
#    Pair (excitonic) index                                                   #
#    ----------------------                                                   #
#    An electron-hole pair is (i k -> a k+Q_exc): hole band i at crystal      #
#    momentum k, particle band a at k+Q_exc.  Pairs are flattened as          #
#                                                                             #
#        rho = k * (n_occ_act * n_virt_act) + i_loc * n_virt_act + a_loc      #
#                                                                             #
#    with k in [0, Nk), i_loc indexing the active occupied bands and a_loc    #
#    the active virtual bands.  kq_map[k] = index of k+Q_exc in the k-mesh.   #
#                                                                             #
#    Kernel conventions (must match the full periodic BSE in casidaEq_finite_q#
#    so the two-step is exact in the full-active limit)                       #
#    ----------------------------------------------------------------------   #
#    * Same-k MO integrals  V_mo_kk[k,Q,p,q] = (S_k C_k)^dagger V_Q(k,k) ...   #
#      supply the occ-occ and virt-virt blocks of the screened -W ladder.     #
#    * Off-diagonal MO integrals  V_ia[k,Q,i,a] = (S_k C_k)^dagger            #
#      V_Q(k,k+Q_exc) (S_{k+Q_exc} C_{k+Q_exc})  carry the transition density #
#      that projects the pair propagator into the auxiliary Q-space.          #
#    * Screening enters through  M = I + Pi_stat , Pi_stat the static (Omega=0)#
#      screened polarization at the exciton momentum Q_exc (read once).  This #
#      mirrors casidaEq_finite_q.build_W_A_fq (a single static W_eff), so the #
#      two-step and the full Casida BSE share the same direct kernel.         #
#    * The 1/Nk Born-von-Karman factor lives (a) explicitly on the -W ladder  #
#      and (b) as 1/sqrt(Nk) on each transition density, so the spin-singlet  #
#      +2v exchange supplied by the Q-space RPA carries the correct 2/Nk.     #
#                                                                             #
#    Exchange / channel                                                       #
#    ------------------                                                       #
#    The electron-hole exchange (+2v, bare Coulomb) is supplied by the Q-space #
#    RPA (spin_factor = 2) from the transition densities V_ia(k,k+Q_exc):      #
#    * singlet : exchange applied at EVERY exciton momentum Q_exc.  The        #
#      exchange is a smooth (continuous-in-Q) "local-field" object; it does    #
#      NOT switch off at finite Q.  Dropping it at Q != 0 (an earlier          #
#      convention, cf. casidaEq_finite_q's kappa=0) produces a spurious        #
#      discontinuity of the singlet branch at Gamma.                          #
#    * triplet : no exchange (spin algebra); excitations are the poles of      #
#      P^irr at all Q.                                                        #
#    Note on the optical limit: the only genuinely non-analytic piece is the   #
#    G=0 macroscopic "head" of v (depolarization / LO-TO).  Here the Coulomb   #
#    G=0 divergence is Ewald-regularized inside the DF integrals, so the       #
#    exchange is finite and continuous at every Q including Gamma; the         #
#    transverse-vs-longitudinal head distinction is a separate, smaller effect #
#    not resolved by this regularized formulation.                            #
#                                                                             #
#    Pole extraction and plotting reuse active_bse unchanged (they act on a   #
#    generic (niw, NQ, NQ) Q-space polarization).                             #

import numpy as np

import active_bse as abse
from active_bse import (qspace_rpa, bubble_replacement,
                        extract_excitations, spectral_function, plot_spectrum)

AU2EV = 27.211386245981


# ===========================================================================
# Pair-space building blocks (transition densities, energies, kernel)
# ===========================================================================
def transition_densities(VQ_ia, act_occ, act_virt, nk):
    """
    Q-space transition densities D[Q, rho] for the active electron-hole pairs.

        D[Q, (k,i,a)] = (1/sqrt(Nk)) V_ia[k, Q, i, a]

    The 1/sqrt(Nk) on each leg reproduces the 1/Nk Born-von-Karman weight of
    the polarization bubble  P0_{QQ'} = sum_rho D[Q,rho] g_rho D*[Q',rho].

    Parameters
    ----------
    VQ_ia : (nk, NQ, nao, nao) complex
        Off-diagonal MO integrals V_Q(k, k+Q_exc); the [i in occ, a in virt]
        block is the transition density (occ at k, virt at k+Q_exc).
    act_occ, act_virt : 1-D int arrays
        Absolute MO-band indices of the active occupied / virtual orbitals.
    nk : int

    Returns
    -------
    D : (NQ, n_pair) complex,  n_pair = nk * len(act_occ) * len(act_virt)
    """
    act_occ = np.asarray(act_occ)
    act_virt = np.asarray(act_virt)
    NQ = VQ_ia.shape[1]
    no, nv = act_occ.size, act_virt.size
    # (k, Q, i, a) -> (Q, k, i, a) -> (NQ, n_pair)
    block = VQ_ia[:, :, act_occ][:, :, :, act_virt]            # (nk, NQ, no, nv)
    D = block.transpose(1, 0, 2, 3).reshape(NQ, nk * no * nv)
    return D / np.sqrt(nk)


def pair_energies(mo_energy, kq_map, act_occ, act_virt):
    """
    Single-particle pair energies  Delta[(k,i,a)] = eps_{a, k+Q_exc} - eps_{i,k}.

    Parameters
    ----------
    mo_energy : (nk, nao) real  quasiparticle (or mean-field) energies per k.
    kq_map : (nk,) int          kq_map[k] = index of k+Q_exc in the k-mesh.
    act_occ, act_virt : 1-D int arrays of absolute band indices.

    Returns
    -------
    Delta : (n_pair,) complex   flattened as (k, i, a).
    """
    act_occ = np.asarray(act_occ)
    act_virt = np.asarray(act_virt)
    nk = mo_energy.shape[0]
    no, nv = act_occ.size, act_virt.size
    Delta = np.empty((nk, no, nv), dtype=np.complex128)
    for k in range(nk):
        eps_v = mo_energy[kq_map[k], act_virt]                 # (nv,)
        eps_o = mo_energy[k, act_occ]                          # (no,)
        Delta[k] = eps_v[None, :] - eps_o[:, None]
    return Delta.reshape(nk * no * nv)


def screened_direct_kernel_periodic(VQ_mo_kk, M, kq_map, act_occ, act_virt, nk):
    """
    Periodic Tamm-Dancoff screened-direct (-W) kernel in the active pair space.

        K_W[(k,i,a),(k',j,b)] =
            -(1/Nk) sum_{QQ'} V_oo[k, Q, i, j] M[Q,Q'] V_vv[k'+Q_exc, Q', b, a]

    with  V_oo = occ-occ block of the same-k MO integrals at k,
          V_vv = virt-virt block of the same-k MO integrals at k'+Q_exc,
          M    = I + Pi_stat  the static screened Coulomb in the aux basis.

    This is the active-space restriction of casidaEq_finite_q.build_W_A_fq, so
    in the full-active limit the two-step kernel coincides with the full
    periodic BSE direct kernel.

    Parameters
    ----------
    VQ_mo_kk : (nk, NQ, nao, nao) complex   same-k MO integrals.
    M : (NQ, NQ) complex                     I + Pi_stat.
    kq_map : (nk,) int
    act_occ, act_virt : 1-D int arrays of absolute band indices.
    nk : int

    Returns
    -------
    K_W : (n_pair, n_pair) complex,  n_pair = nk*len(act_occ)*len(act_virt).
    """
    act_occ = np.asarray(act_occ)
    act_virt = np.asarray(act_virt)
    no, nv = act_occ.size, act_virt.size
    n = nk * no * nv

    # V_oo[k, Q, i, j]  (occ-occ at k)
    Voo = VQ_mo_kk[:, :, act_occ][:, :, :, act_occ]                 # (nk, NQ, no, no)
    # V_vv[k'+Q_exc, Q', b, a]  (virt-virt at the particle momentum)
    Vvv_kq = VQ_mo_kk[kq_map][:, :, act_virt][:, :, :, act_virt]    # (nk, NQ, nv, nv)

    # K_W[k,i,a, K,j,b] = -(1/Nk) Voo[k,Q,i,j] M[Q,P] Vvv[K,P,b,a]
    K_W = -(1.0 / nk) * np.einsum('kQij,QP,KPba->kiaKjb',
                                  Voo, M, Vvv_kq, optimize=True)
    return K_W.reshape(n, n)


def screened_direct_kernel_qresolved(Voo_all, Vvv_all, M_q, kq_exc, kdiff,
                                     act_occ, act_virt, occ, nk):
    """
    Fully q-resolved Tamm-Dancoff screened-direct (-W) kernel.

        K_W[(k,i,a),(k',j,b)] =
            -(1/Nk) sum_{QQ'} V_oo[k,k', Q, i, j]
                              M^{q}[Q,Q'] V_vv[k'+Q_exc, k+Q_exc, Q', b, a]

    with the momentum transfer  q = k - k'  (NOT the exciton momentum Q_exc):
    the direct interaction screens the hole-hole density (i@k, j@k') and the
    particle-particle density (a@k+Q_exc, b@k'+Q_exc), both of which carry
    crystal momentum k - k'.  M^{q} = I + Pi(q) is the static screened Coulomb
    at that momentum.  This is the physically correct BSE direct kernel; the
    single-Pi variant screened_direct_kernel_periodic approximates M^{q} by a
    constant M(Q_exc).

    Parameters
    ----------
    Voo_all : (nk, nk, NQ, occ, occ)     MO occ-occ blocks (periodic_integrals).
    Vvv_all : (nk, nk, NQ, virt, virt)   MO virt-virt blocks.
    M_q : (nk, NQ, NQ)                   I + Pi_stat(q) for every q on the mesh.
    kq_exc : (nk,) int                   kq_exc[k] = index of k + Q_exc.
    kdiff : (nk, nk) int                 kdiff[k,k'] = index of k - k'.
    act_occ : absolute occ band indices (0..occ-1).
    act_virt : absolute virt band indices (occ..nao-1).
    occ : int                            (to map absolute virt -> local virt index).
    nk : int

    Returns
    -------
    K_W : (n_pair, n_pair) complex,  n_pair = nk*len(act_occ)*len(act_virt).
    """
    ao = np.asarray(act_occ)
    av = np.asarray(act_virt) - occ                       # local virt indices
    no, nv = ao.size, av.size
    n = nk * no * nv
    K = np.zeros((nk, no, nv, nk, no, nv), dtype=np.complex128)
    for k in range(nk):
        kq = kq_exc[k]
        for kp in range(nk):
            kpq = kq_exc[kp]
            Voo = Voo_all[k, kp][:, ao][:, :, ao]               # (NQ, no, no)  [i@k, j@k']
            Vvv = Vvv_all[kpq, kq][:, av][:, :, av]             # (NQ, nv, nv)  [b@k'q, a@kq]
            M = M_q[kdiff[k, kp]]
            K[k, :, :, kp, :, :] = -(1.0 / nk) * np.einsum(
                'Qij,QP,Pba->iajb', Voo, M, Vvv, optimize=True)
    return K.reshape(n, n)


# ===========================================================================
# Q-space polarization of the (active or full) pair propagator
# ===========================================================================
def eval_ladder_Q(D, Delta, omega, kernel=None):
    """
    Q-space polarization of a Tamm-Dancoff pair propagator:

        P(iOmega)_{QQ'} = sum_{rho,rho'} D[Q,rho]
                             [(iOmega I - A)^{-1}]_{rho,rho'} D*[Q',rho']

    with the pair Hamiltonian  A = diag(Delta) + kernel.

    * kernel = screened_direct_kernel_periodic(...)  -> P^act  (-W ladder)
    * kernel = None                                  -> bare bubble
                                                        (P^act(0) or P^(0))

    Parameters
    ----------
    D : (NQ, n_pair) complex      transition densities (1/sqrt(Nk) included).
    Delta : (n_pair,) complex     pair energies.
    omega : (niw,) real           bosonic Matsubara frequencies Omega_n.
    kernel : (n_pair, n_pair) complex or None.

    Returns
    -------
    P : (niw, NQ, NQ) complex.
    """
    NQ, n = D.shape
    niw = omega.shape[0]
    A = np.diag(Delta).astype(np.complex128)
    if kernel is not None:
        A = A + kernel
    Dh = D.conj().T                                        # (n, NQ)
    In = np.eye(n, dtype=np.complex128)
    P = np.zeros((niw, NQ, NQ), dtype=np.complex128)
    for iw in range(niw):
        G2p = np.linalg.solve(1j * omega[iw] * In - A, Dh)   # (n, NQ)
        P[iw] = D @ G2p
    return P


# ===========================================================================
# High-level driver
# ===========================================================================
def two_step_bse_periodic(VQ_mo_kk, VQ_ia, mo_energy, Pi_stat, kq_map,
                          occ, nao, act_occ, act_virt, omega,
                          channel="singlet", q_is_zero=True, K_W=None):
    """
    Periodic finite-Q two-step active-space BSE polarization in Q-space.

    Parameters
    ----------
    VQ_mo_kk : (nk, NQ, nao, nao) complex
        Same-k MO integrals (casidaEq_finite_q.VQ_ao2mo_kk).  Supplies the
        occ-occ / virt-virt blocks of the screened -W ladder.
    VQ_ia : (nk, NQ, nao, nao) complex
        Off-diagonal MO integrals V_Q(k, k+Q_exc)
        (casidaEq_finite_q.VQ_ao2mo_kq_proper).  The occ x virt block is the
        transition density.
    mo_energy : (nk, nao) real      quasiparticle energies per k-point.
    Pi_stat : (NQ, NQ) complex      static screened polarization at Q_exc.
    kq_map : (nk,) int              kq_map[k] = index of k+Q_exc.
    occ : int                       number of occupied bands per k.
    nao : int                       number of bands per k.
    act_occ, act_virt : 1-D int arrays of absolute band indices (the ladder
        space).  The full space is range(occ) x range(occ, nao) at every k.
    omega : (niw,) real             bosonic Matsubara grid.
    channel : "singlet" or "triplet".
        singlet -> the +2v exchange (Q-space RPA, spin_factor=2) is applied at
        EVERY exciton momentum Q_exc (it is a continuous-in-Q local-field term).
        triplet -> no exchange; the excitations are the poles of P^irr.
    q_is_zero : bool                DEPRECATED / ignored.  Kept only for call
        compatibility; the exchange is no longer gated on Q_exc = 0 (doing so
        made the singlet branch discontinuous at Gamma).  See the module header.
    K_W : (n_pair, n_pair) complex or None
        Precomputed active-space screened-direct kernel.  Pass the output of
        screened_direct_kernel_qresolved for the fully q-resolved W^{k-k'}.
        If None (default), the single-Pi(Q_exc) kernel
        screened_direct_kernel_periodic(VQ_mo_kk, I+Pi_stat, ...) is built here.

    Returns
    -------
    P_irr : (niw, NQ, NQ)   irreducible polarization (= the triplet result).
    P_bse : (niw, NQ, NQ)   BSE polarization with the +2v exchange (= the
                            singlet result), at every Q_exc.  For channel
                            "triplet" P_bse == P_irr.
    """
    nk = VQ_mo_kk.shape[0]
    full_occ = np.arange(occ)
    full_virt = np.arange(occ, nao)

    # Active-space -W ladder and its bare counterpart
    if K_W is None:
        M = np.eye(VQ_mo_kk.shape[1], dtype=np.complex128) + Pi_stat
        K_W = screened_direct_kernel_periodic(VQ_mo_kk, M, kq_map, act_occ, act_virt, nk)
    D_act = transition_densities(VQ_ia, act_occ, act_virt, nk)
    Del_act = pair_energies(mo_energy, kq_map, act_occ, act_virt)
    P_act = eval_ladder_Q(D_act, Del_act, omega, kernel=K_W)
    P_act0 = eval_ladder_Q(D_act, Del_act, omega, kernel=None)

    # Full-space bare bubble
    D_full = transition_densities(VQ_ia, full_occ, full_virt, nk)
    Del_full = pair_energies(mo_energy, kq_map, full_occ, full_virt)
    P0_full = eval_ladder_Q(D_full, Del_full, omega, kernel=None)

    P_irr = bubble_replacement(P_act, P_act0, P0_full)

    # +2v electron-hole exchange (singlet only) -- applied at EVERY Q_exc.
    if channel != "singlet":
        return P_irr, P_irr
    P_bse = qspace_rpa(P_irr, spin_factor=2.0)
    return P_irr, P_bse
