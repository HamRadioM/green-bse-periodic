#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2026 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Wannier interpolation of the two-step polarization P(Q, iOmega) over the # 
#    exciton momentum q, performed before the analytic continuation.          #
#                                                                             #
#    The full matrix P(Q, iOmega) is a smooth analytic function of q.         #
#    We interpolate each matrix element P[iw, A, B] over the q-mesh,          #
#    then re-extract the poles per interpolated Q with the usual continuation.#
#                                                                             #
#    Section 2b applies the same collect / interpolate / validate machinery   #
#    to the diagonal of VPV in the occ/virt MO space instead of P itself.     #
#                                                                             #

import numpy as np
import h5py

import periodic_active_bse as pabse
import periodic_integrals as pint
import casidaEq_finite_q as cfq

from green_mbtools.pesto import winter


# --------------------------------------------------------------------------- #
#  1. Assemble P(iOmega) at one exciton momentum on the BvK mesh               #
# --------------------------------------------------------------------------- #
def _P_and_VQia_at_q(solver, stored, sym, qidx, ao, av, channel):
    """P(iOmega) at mesh momentum qidx together with the V_Q(k,k+Q) MO integrals.

    Shared per-q construction behind P_at_q and VPV_diag_at_q, so the DF
    integrals V_Q(k, k+Q_exc) are built once when both objects are needed.
    """
    if solver.config.screening != "qresolved":
        raise NotImplementedError(
            "wannier_exciton currently supports screening='qresolved' only; "
            "the static mode pins Pi_stat to a single Q_exc and cannot sweep Q.")

    s = solver
    kq = s.kmaps["ksum"][:, qidx]                      # k + Q for every k
    rSk = s.results["rSk"]

    VQ_kq_ao = pint.build_VQ_offdiag(stored, sym, kq)               # V_Q(k, k+Q)
    VQ_ia    = cfq.VQ_ao2mo_kq_proper(VQ_kq_ao, s.vexMO, kq, rSk=rSk)  # -> MO o/v

    K_W = pabse.screened_direct_kernel_qresolved(
        s.Voo_all, s.Vvv_all, s.M_q, kq, s.kmaps["kdiff"],
        ao, av, s.occ, s.nk)

    P_irr, P_bse = pabse.two_step_bse_periodic(
        s.VQ_mo_kk, VQ_ia, s.mo_energy, None, kq, s.occ, s.nao, ao, av,
        s.omega, channel=channel, K_W=K_W)

    return (P_bse if channel == "singlet" else P_irr), VQ_ia


def P_at_q(solver, stored, sym, qidx, ao, av, channel):
    """Two-step polarization that feeds the continuation, at mesh momentum qidx.

    P_bse for the singlet channel (exchange applied at all Q), P_irr for the
    triplet channel (no exchange).

    Requires the q-resolved screening intermediates (config.screening ==
    "qresolved"): solver.{Voo_all, Vvv_all, M_q, VQ_mo_kk, mo_energy} and
    solver.kmaps, as built by PeriodicTwoStepBSESolver.build_intermediates().
    """
    return _P_and_VQia_at_q(solver, stored, sym, qidx, ao, av, channel)[0]


def collect_P_over_mesh(solver, stored, sym, ao, av, channel,
                        q_indices=None, verbose=True):
    """P(iOmega) at every (or a chosen subset of) exciton momentum on the mesh.

    Returns
    -------
    P_mesh : (nQ, niw, NQ, NQ) complex
    q_frac : (nQ, 3)  fractional coords of those Q (= the k-mesh, scaled units).
    """
    s = solver
    q_indices = list(range(s.nk)) if q_indices is None else list(q_indices)

    with h5py.File(s.config.input_file, "r") as f:
        kfrac = f["symmetry/k/mesh_scaled"][:]          # (nk, 3) fractional

    P_list, qf = [], []
    for n, Q in enumerate(q_indices):
        if verbose:
            print("  building P at Q-mesh point %3d/%d (idx=%d)"
                  % (n + 1, len(q_indices), Q))
        P_list.append(P_at_q(solver, stored, sym, Q, ao, av, channel))
        qf.append(kfrac[Q])

    return np.asarray(P_list), np.asarray(qf)


# --------------------------------------------------------------------------- #
#  2. Wannier interpolation of P over Q                                        #
# --------------------------------------------------------------------------- #
def interpolate_rect2d(obj_k, kmesh, kpts_inter, hermi=False):
    """winter.interpolate for a possibly RECTANGULAR 2-D k-mesh.

    In green_mbtools, winter.interpolate(dim=2) infers the mesh as sqrt(Nk) x sqrt(Nk). 
    This is an adaptation that works for a rectangular nk1 x nk2 mesh (e.g. 10x5). 

    Parameters mirror winter.interpolate: obj_k[ns, Nk, n, n], meshes in
    scaled (fractional) units.
    """
    from green_mbtools.pesto import ft

    km = np.asarray(kmesh)
    # fold to [0,1) BEFORE and AFTER rounding: a coordinate like 1-1e-12
    # otherwise rounds to 1.0 and fails to merge with 0.0
    nkx, nky, nkz = (np.unique(np.round(km[:, i] % 1.0, 8) % 1.0).size
                     for i in range(3))
    if nkx * nky * nkz != km.shape[0]:
        raise ValueError(
            "k-mesh is not a full rectangular grid: %d x %d x %d != %d points"
            % (nkx, nky, nkz, km.shape[0]))
    rmesh = ft.construct_rmesh(nkx, nky, nkz)

    _, frk = ft.compute_fourier_coefficients(km, rmesh)
    weights = [1] * km.shape[0]
    ns = obj_k.shape[0]
    obj_i = np.array([ft.k_to_real(frk, obj_k[s], weights) for s in range(ns)])

    fkr_int, _ = ft.compute_fourier_coefficients(np.asarray(kpts_inter), rmesh)
    obj_int = np.array([ft.real_to_k(fkr_int, obj_i[s]) for s in range(ns)])

    if hermi:
        obj_int = 0.5 * (obj_int + obj_int.conj().transpose(0, 1, 3, 2))
    return obj_int


def _interpolate(obj, q_mesh, q_target, dim, hermi=False):
    if dim == 2:
        return interpolate_rect2d(obj, q_mesh, q_target, hermi=hermi)
    return winter.interpolate(obj, q_mesh, q_target, dim=dim, hermi=hermi)


def interpolate_P_over_q(P_mesh, q_frac_mesh, q_frac_target, dim=2):
    """Band-limited Wannier interpolation of P(Q, iOmega) over Q.

    Parameters
    ----------
    P_mesh        : (nQ, niw, NQ, NQ) complex, P at the full BvK Q-mesh.
    q_frac_mesh   : (nQ, 3)  fractional coords of that mesh (scaled units).
    q_frac_target : (nP, 3)  fractional coords to interpolate to.
    dim           : 2 or 3   mesh dimensionality for the Wannier FT.

    Returns
    -------
    P_path : (nP, niw, NQ, NQ) complex.

    Notes
    -----
    * hermi=False is mandatory. P(iOmega) is NOT Hermitian for Omega != 0. 
      Only the static frequency frame is Hermitian.
      The downstream extractor (extract_excitations) re-Hermitizes P(Omega ~ 0) itself 
      when it defines the channels, so we leave the interpolated matrices untouched here.
    * winter.interpolate expects obj_k[ns, Nk, n, n]; we map the niw frequency
      axis onto the leading ns slot and the Q-mesh onto Nk.
    * Node-exactness: at q_frac_target points that coincide with mesh points,
      P_path reproduces P_mesh to ~1e-12.  Check this as a gauge sanity test
      (see validate_node_exactness).
    """
    # nQ, niw, NQ, _ = P_mesh.shape
    obj = np.ascontiguousarray(P_mesh.transpose(1, 0, 2, 3))      # (niw, nQ, NQ, NQ)
    obj_int = _interpolate(obj, np.asarray(q_frac_mesh),
                           np.asarray(q_frac_target),
                           dim, hermi=False)                      # (niw, nP, NQ, NQ)
    return np.ascontiguousarray(obj_int.transpose(1, 0, 2, 3))   # (nP, niw, NQ, NQ)


def validate_node_exactness(P_mesh, q_frac_mesh, dim=2):
    """Interpolate the mesh back onto itself; return the max abs deviation.

    A small number (~1e-12) confirms the Q-gauge of P is smooth enough for the
    Wannier real-space (J(R)) representation to be well localized.  A large
    number signals a phase-gauge problem in V^Q(k, k+Q) / the transition
    densities that must be fixed before trusting interpolated points.
    """
    P_back = interpolate_P_over_q(P_mesh, q_frac_mesh, q_frac_mesh, dim=dim)
    return float(np.max(np.abs(P_back - P_mesh)))


# --------------------------------------------------------------------------- #
#  2b. Same machinery for the diagonal of VPV in the occ/virt MO space         #
# --------------------------------------------------------------------------- #
def vpv_diag_from_P(P, VQ_ia, act_occ, act_virt):
    """Diagonal of V P V in the active occupied/virtual BAND-pair space.

        VPV[iw, i, a] = (1/Nk) sum_k sum_{QQ'}
                        V_ia[k,Q,i,a]^* P[iw,Q,Q'] V_ia[k,Q',i,a]

    the polarization-mediated part of the screened interaction W = v + v P v
    for the band pair (i, a), with the Born-von-Karman k-trace built in: this
    equals sum_k (D^dag P D)_{(k,i,a),(k,i,a)} with the transition densities D
    of transition_densities (whose 1/sqrt(Nk) legs supply the 1/Nk).  The
    diagonal is over (i, a) only -- for a single occupied and single virtual
    band it is ONE scalar function of (Q_exc, iOmega).  At nk=1 this matches
    the molecular casidaEq VPV convention.  Like P itself it is a smooth
    analytic function of the exciton momentum Q, so it Wannier-interpolates
    the same way.

    Hermiticity mirrors P: real on the static frame only, since
    P(iOmega)^dagger = P(-iOmega).

    Parameters
    ----------
    P : (niw, NQ, NQ) complex          two-step polarization at one Q_exc.
    VQ_ia : (nk, NQ, nao, nao) complex V_Q(k, k+Q_exc) MO integrals.
    act_occ, act_virt : 1-D int arrays of absolute band indices.

    Returns
    -------
    vpv : (niw, no, nv) complex.
    """
    ao = np.asarray(act_occ)
    av = np.asarray(act_virt)
    nk, NQ = VQ_ia.shape[0], VQ_ia.shape[1]
    no, nv = ao.size, av.size
    V = VQ_ia[:, :, ao][:, :, :, av]                       # (nk, NQ, no, nv)
    Vd = V.transpose(1, 0, 2, 3).reshape(NQ, nk * no * nv)  # (NQ, nk*no*nv)

    niw = P.shape[0]
    vpv = np.empty((niw, no, nv), dtype=np.complex128)
    for iw in range(niw):
        per_k = np.sum(Vd.conj() * (P[iw] @ Vd), axis=0)   # (nk*no*nv,)
        vpv[iw] = per_k.reshape(nk, no, nv).sum(axis=0) / nk
    return vpv


def vpv_static_matrix_from_P(P_stat, VQ_ia, act_occ, act_virt):
    """FULL k-traced pair-space VPV matrix at one frequency (static frame).

        M[(i,a),(j,b)] = (1/Nk) sum_k sum_{QQ'}
                         V_ia[k,Q,i,a]^* P_stat[Q,Q'] V_ia[k,Q',j,b]

    The diagonal reproduces vpv_diag_from_P at that frequency; the
    off-diagonal magnitudes quantify what the diagonal approximation
    discards (they are gauge-safe in magnitude: an MO phase multiplies
    whole rows/columns by a unit factor).

    Parameters
    ----------
    P_stat : (NQ, NQ) complex          P at one (usually Omega ~ 0) frequency.
    VQ_ia : (nk, NQ, nao, nao) complex V_Q(k, k+Q_exc) MO integrals.
    act_occ, act_virt : 1-D int arrays of absolute band indices.

    Returns
    -------
    M : (no*nv, no*nv) complex, pair index flattened as i*nv + a.
    """
    ao = np.asarray(act_occ)
    av = np.asarray(act_virt)
    nk, NQ = VQ_ia.shape[0], VQ_ia.shape[1]
    no, nv = ao.size, av.size
    V = VQ_ia[:, :, ao][:, :, :, av]                       # (nk, NQ, no, nv)
    Vd = V.transpose(1, 0, 2, 3).reshape(NQ, nk, no * nv)  # (NQ, nk, npair)
    PV = (P_stat @ Vd.reshape(NQ, -1)).reshape(NQ, nk, no * nv)
    return np.einsum("Qkr,Qks->rs", Vd.conj(), PV) / nk


def vpv_matrix_from_P(P, VQ_ia, act_occ, act_virt):
    """FULL k-traced pair-space VPV matrix at every frequency.

    Frequency-resolved generalization of vpv_static_matrix_from_P:
        M[iw, (i,a), (j,b)] = (1/Nk) sum_k sum_{QQ'}
                              V_ia[k,Q,i,a]^* P[iw,Q,Q'] V_ia[k,Q',j,b]
    Used to define eigen-channels: diagonalize M at the static frame and
    take u_r^dag M(iOmega) u_r as the (gauge-invariant) channel curves.

    Returns
    -------
    M : (niw, no*nv, no*nv) complex, pair index flattened as i*nv + a.
    """
    ao = np.asarray(act_occ)
    av = np.asarray(act_virt)
    nk, NQ = VQ_ia.shape[0], VQ_ia.shape[1]
    no, nv = ao.size, av.size
    V = VQ_ia[:, :, ao][:, :, :, av]                       # (nk, NQ, no, nv)
    Vd = V.transpose(1, 0, 2, 3).reshape(NQ, nk, no * nv)  # (NQ, nk, npair)

    niw = P.shape[0]
    M = np.empty((niw, no * nv, no * nv), dtype=np.complex128)
    for iw in range(niw):
        PV = (P[iw] @ Vd.reshape(NQ, -1)).reshape(NQ, nk, no * nv)
        M[iw] = np.einsum("Qkr,Qks->rs", Vd.conj(), PV) / nk
    return M


def offdiag_ratios(M):
    """Off-diagonal weight of a pair-space matrix: (max-ratio, Frobenius-ratio).

    max-ratio = max |offdiag| / max |diag|;  fro-ratio = ||off||_F / ||diag||_F.
    """
    d = np.abs(np.diag(M))
    off = np.abs(M - np.diag(np.diag(M)))
    return (float(off.max() / (d.max() or 1.0)),
            float(np.linalg.norm(off) / (np.linalg.norm(d) or 1.0)))


def VPV_diag_at_Q(solver, stored, sym, qidx, ao, av, channel):
    """Diagonal of VPV in the occ/virt band-pair space at mesh momentum qidx.

    Same per-Q construction as P_at_q (and the same q-resolved requirement),
    but the polarization is immediately sandwiched between the transition
    integrals V_Q(k, k+Q_exc) and only the k-traced (niw, no, nv) diagonal
    is kept.
    """
    P, VQ_ia = _P_and_VQia_at_q(solver, stored, sym, qidx, ao, av, channel)
    return vpv_diag_from_P(P, VQ_ia, ao, av)


def collect_VPV_diag_over_mesh(solver, stored, sym, ao, av, channel,
                               q_indices=None, verbose=True):
    """VPV diagonal at every (or a chosen subset of) exciton momentum on the mesh.

    The full (niw, NQ, NQ) polarization is built one mesh-Q at a time and
    contracted to its k-traced (niw, no, nv) diagonal on the spot, so peak
    memory never holds more than one dense P.

    Returns
    -------
    VPV_mesh : (nQ, niw, no, nv) complex
    q_frac   : (nQ, 3)  fractional coords of those Q (= the k-mesh, scaled units).
    """
    s = solver
    q_indices = list(range(s.nk)) if q_indices is None else list(q_indices)

    with h5py.File(s.config.input_file, "r") as f:
        kfrac = f["symmetry/k/mesh_scaled"][:]          # (nk, 3) fractional

    vpv_list, qf = [], []
    for n, Q in enumerate(q_indices):
        if verbose:
            print("  building VPV diag at Q-mesh point %3d/%d (idx=%d)"
                  % (n + 1, len(q_indices), Q))
        vpv_list.append(VPV_diag_at_Q(solver, stored, sym, Q, ao, av, channel))
        qf.append(kfrac[Q])

    return np.asarray(vpv_list), np.asarray(qf)


def wannier_decay(F_mesh, q_frac_mesh):
    """Real-space decay profile |J(R)| of a per-Q field -- basis quality metric.

    A representation is good for band-limited Q-interpolation exactly when its
    Wannier (real-space) coefficients J(R) = (1/nQ) sum_Q e^{i 2pi Q.R} F(Q)
    decay fast with |R|: slow decay means the field carries sharp Q-structure
    (band-label switches, moving poles) that the interpolant will ring on
    between mesh nodes.  Compare profiles of candidate representations (e.g.
    Tr P vs the o-v VPV diagonal vs a frozen-sandwich VPV) on the SAME mesh.

    Parameters
    ----------
    F_mesh      : (nQ, ...) complex   field(s) on the full BvK Q-mesh; all
                  trailing axes (frequencies, pairs, ...) are aggregated by a
                  max of |J| so the profile reflects the worst component.
    q_frac_mesh : (nQ, 3) fractional mesh coordinates.

    Returns
    -------
    Rlen : (nR,) real   |R| in lattice-translation units, sorted ascending.
           (Euclidean norm of the integer translations; for anisotropic
           lattices this mixes the two axes -- fine for relative comparison.)
    w    : (nR,) real   max |J(R)| over trailing axes, normalized to max 1.
    """
    from green_mbtools.pesto import ft

    km = np.asarray(q_frac_mesh)
    nkx, nky, nkz = (np.unique(np.round(km[:, i] % 1.0, 8) % 1.0).size
                     for i in range(3))
    rmesh = ft.construct_rmesh(nkx, nky, nkz)
    _, frk = ft.compute_fourier_coefficients(km, rmesh)

    F = np.asarray(F_mesh).reshape(km.shape[0], -1)       # (nQ, nflat)
    J = (frk @ F) / km.shape[0]                           # (nR, nflat)
    w = np.abs(J).max(axis=1)
    Rlen = np.linalg.norm(rmesh, axis=1)
    order = np.argsort(Rlen)
    return Rlen[order], w[order] / (w.max() or 1.0)


def collect_vpv_variants_over_mesh(solver, stored, sym, ao, av, channel,
                                   ref_idx=0, q_indices=None, verbose=True):
    """One mesh sweep, three interpolation representations of the same P(Q).

    For every mesh Q the two-step polarization is built once and contracted
    three ways:
      trP        : Tr P(Q, iOmega)            -- basis-free reference (the
                   smoothest scalar; what spectral_map_over_path uses).
      vpv        : o-v VPV diagonal            -- current convention
                   (vpv_diag_from_P with the Q-consistent V_Q(k, k+Q)).
      vpv_frozen : frozen-sandwich VPV diagonal -- SAME contraction but with
                   the transition integrals pinned to mesh momentum ref_idx
                   for every Q.  The weight matrix is then Q-independent, so
                   this field is exactly as smooth as P itself; comparing its
                   wannier_decay against vpv isolates how much Q-rotation of
                   the occupied/virtual basis costs.

    Returns
    -------
    out : dict with "trP" (nQ, niw), "vpv" and "vpv_frozen" (nQ, niw, no, nv).
    q_frac : (nQ, 3).
    """
    s = solver
    q_indices = list(range(s.nk)) if q_indices is None else list(q_indices)

    with h5py.File(s.config.input_file, "r") as f:
        kfrac = f["symmetry/k/mesh_scaled"][:]

    # reference (frozen) transition integrals
    _, VQ_ref = _P_and_VQia_at_q(solver, stored, sym, ref_idx, ao, av, channel)

    no, nv = len(ao), len(av)
    nQm, niw = len(q_indices), s.omega.shape[0]
    out = {"trP": np.empty((nQm, niw), dtype=np.complex128),
           "vpv": np.empty((nQm, niw, no, nv), dtype=np.complex128),
           "vpv_frozen": np.empty((nQm, niw, no, nv), dtype=np.complex128)}
    qf = []
    for n, Q in enumerate(q_indices):
        if verbose:
            print("  building P variants at Q-mesh point %3d/%d (idx=%d)"
                  % (n + 1, len(q_indices), Q))
        P, VQ_ia = _P_and_VQia_at_q(solver, stored, sym, Q, ao, av, channel)
        out["trP"][n] = np.einsum("wqq->w", P)
        out["vpv"][n] = vpv_diag_from_P(P, VQ_ia, ao, av)
        out["vpv_frozen"][n] = vpv_diag_from_P(P, VQ_ref, ao, av)
        qf.append(kfrac[Q])
    return out, np.asarray(qf)


def interpolate_VPV_diag_over_Q(VPV_mesh, q_frac_mesh, q_frac_target, dim=2):
    """Band-limited Wannier interpolation of the VPV diagonal over Q.

    Each scalar element VPV[iw, i, a] is interpolated over the Q-mesh
    exactly as the matrix elements of P are in interpolate_P_over_q (hermi=False
    for the same reason); the (iw, i, a) axes ride along on the winter ns
    slot as 1x1 matrices.

    Parameters
    ----------
    VPV_mesh      : (nQ, niw, no, nv) complex, from collect_VPV_diag_over_mesh.
    q_frac_mesh   : (nQ, 3)  fractional coords of that mesh (scaled units).
    q_frac_target : (nP, 3)  fractional coords to interpolate to.
    dim           : 2 or 3   mesh dimensionality for the Wannier FT.

    Returns
    -------
    VPV_path : (nP, niw, no, nv) complex.
    """
    nQ = VPV_mesh.shape[0]
    trail = VPV_mesh.shape[1:]
    obj = np.ascontiguousarray(
        VPV_mesh.reshape(nQ, -1).T)[:, :, None, None]     # (ns, nQ, 1, 1)
    obj_int = _interpolate(obj, np.asarray(q_frac_mesh),
                           np.asarray(q_frac_target),
                           dim, hermi=False)               # (ns, nP, 1, 1)
    nP = obj_int.shape[1]
    return np.ascontiguousarray(obj_int[:, :, 0, 0].T).reshape((nP,) + trail)


def validate_VPV_node_exactness(VPV_mesh, q_frac_mesh, dim=2):
    """Interpolate the VPV-diag mesh back onto itself; return the max abs deviation.

    Same gauge sanity test as validate_node_exactness, for the VPV diagonal.
    """
    back = interpolate_VPV_diag_over_Q(VPV_mesh, q_frac_mesh, q_frac_mesh, dim=dim)
    return float(np.max(np.abs(back - VPV_mesh)))


# --------------------------------------------------------------------------- #
#  3. Re-extract excitations on the interpolated path                          #
# --------------------------------------------------------------------------- #
def excitons_along_path(P_path, omega, **extract_kwargs):
    """Run extract_excitations on every interpolated Q point.

    Returns a list (len nP) of the usual [(energy_eV, strength), ...] lists.

    NOTE: discrete pole extraction is done *independently* per Q, so the pole
    set (and which pole is "lowest") can jitter between neighbouring Q even
    though P(Q) is smooth -- this is what makes a discrete-pole dispersion plot
    look ragged.  For a smooth heatmap use spectral_map_from_P instead; for
    clean band *lines* feed this output through track_bands.
    """
    return [pabse.extract_excitations(P_path[n], omega, **extract_kwargs)
            for n in range(P_path.shape[0])]


AU2EV = 27.211386245981


def aaa_continue_field(F, omega, egrid_eV, eta_eV=0.1, aaa_rtol=1e-10):
    """AAA analytic continuation of a smooth per-Q scalar field F(Q, iOmega).

    Generic version of the continuation inside spectral_map_from_P: each row
    F[n] (one path point) is continued from the Matsubara axis to
    omega + i*eta on the real axis.  Use it for any scalar built from the
    interpolated objects -- Tr P, the VPV diagonal (k-averaged or per pair), ...
    Because the input is smooth in Q (node-exact interpolation), the continued
    field is smooth in Q too.

    Parameters
    ----------
    F        : (nP, niw) complex   scalar field on the Matsubara grid.
    omega    : (niw,) real         bosonic Matsubara grid (a.u.).
    egrid_eV : (nE,) real          real-frequency grid (eV).
    eta_eV   : float               broadening (eV).

    Returns
    -------
    Fw : (nE, nP) complex   F(Q, omega + i*eta), same units as F (a.u.);
         columns where AAA fails are left as NaN.
    """
    try:
        from scipy.interpolate import AAA
    except Exception as exc:                       # pragma: no cover
        raise RuntimeError("aaa_continue_field needs scipy>=1.15 (AAA).") from exc

    z = 1j * omega
    w = egrid_eV / AU2EV + 1j * (eta_eV / AU2EV)
    Fw = np.full((w.size, F.shape[0]), np.nan + 0j, dtype=np.complex128)
    for n in range(F.shape[0]):
        try:
            Fw[:, n] = AAA(z, F[n], rtol=aaa_rtol)(w)
        except Exception:
            continue
    return Fw


def pp_continue_field(F, omega, egrid_eV, eta_eV=0.1):
    """One-plasmon-pole continuation of a smooth per-Q scalar field F(Q, iOmega).

    Drop-in alternative to aaa_continue_field: each row F[n] is fit to the
    single plasmon-pole model (plasPole.fit_plasmon_pole, F0/Finf pinned to
    the data endpoints)

        F(z) = Finf + 2 wp S / (wp^2 - z^2)

    and the model is evaluated at omega + i*eta.  The three fit parameters
    vary smoothly with the (smooth) interpolated data, so the continued map
    has none of AAA's branch-truncation seams -- but the model resolves
    exactly ONE pole pair.  Use it when the field is dominated by a single
    mode, and check fit["resid"] where it is not (large residual -> multi-
    mode field, prefer aaa_continue_field).

    The model is real on the Matsubara axis, so only Re F on the positive
    branch is fit; the odd-in-Omega imaginary part a finite-Q field carries
    (F(-iOmega) = conj F(iOmega)) is outside the model.

    Parameters
    ----------
    F        : (nP, niw) complex   scalar field on the Matsubara grid.
    omega    : (niw,) real         bosonic Matsubara grid (a.u.).
    egrid_eV : (nE,) real          real-frequency grid (eV).
    eta_eV   : float               broadening (eV).

    Returns
    -------
    Fw  : (nE, nP) complex   model at omega + i*eta, same units as F (a.u.);
          columns where the fit fails are left as NaN.
    fit : dict of (nP,) arrays -- "wp" (pole position, a.u.), "S" (strength),
          "Finf", and "resid" (relative fit residual of Re F).
    """
    from plasPole import fit_plasmon_pole, plasmon_model

    pos = omega > 0
    om_p = omega[pos]
    w = egrid_eV / AU2EV + 1j * (eta_eV / AU2EV)
    nP = F.shape[0]
    Fw = np.full((w.size, nP), np.nan + 0j, dtype=np.complex128)
    fit = {key: np.full(nP, np.nan) for key in ("wp", "S", "Finf", "resid")}

    for n in range(nP):
        fd = F[n, pos].real
        if abs(fd[0] - fd[-1]) < 1e-13:                # no pole weight
            continue
        try:
            p = fit_plasmon_pole(om_p, fd, F0=float(fd[0]), Finf=float(fd[-1]))
        except Exception:
            continue
        model = plasmon_model(1j * om_p, p["Finf"], p["S"], p["wp"]).real
        denom = np.linalg.norm(fd - fd[-1]) or 1.0
        fit["wp"][n], fit["S"][n], fit["Finf"][n] = p["wp"], p["S"], p["Finf"]
        fit["resid"][n] = np.linalg.norm(model - fd) / denom
        Fw[:, n] = plasmon_model(w, p["Finf"], p["S"], p["wp"])
    return Fw, fit


def aaa_continue_vpv_pairs(VPV_path, omega, egrid_eV, eta_eV=0.1,
                           aaa_rtol=1e-10, average=False):
    """AAA continuation of the VPV diagonal, one continuation per (i, a).

    Same pair-resolved strategy as pp_continue_vpv_pairs, with AAA in place of
    the one-pole model: each occupied/virtual band-pair channel
    VPV_path[:, :, i, a] is continued independently (aaa_continue_field) and
    the total DOS is the sum over pairs of the continued channels (pair-mean
    if average=True).  A single pair channel carries fewer active modes than
    the pair-summed field, so the per-Q rational fits are shorter and better
    conditioned than one AAA on the sum.

    Parameters
    ----------
    VPV_path : (nP, niw, no, nv) complex   interpolated VPV diagonal.
    omega    : (niw,) real                 bosonic Matsubara grid (a.u.).
    egrid_eV : (nE,) real                  real-frequency grid (eV).
    eta_eV   : float                       broadening (eV).
    average  : bool                        divide the pair-sum by no*nv.

    Returns
    -------
    Fw     : (nE, nP) complex  pair-summed continued field (a.u.).  Pairs
             whose AAA fit fails at a path point contribute 0 there.
    n_fail : (nP,) int         number of failed pair fits per path point.
    """
    nP, niw = VPV_path.shape[:2]
    Vp = VPV_path.reshape(nP, niw, -1)
    npair = Vp.shape[2]
    nE = np.asarray(egrid_eV).size
    Fw = np.zeros((nE, nP), dtype=np.complex128)
    n_fail = np.zeros(nP, dtype=int)

    for r in range(npair):
        Fr = aaa_continue_field(Vp[:, :, r], omega, egrid_eV,
                                eta_eV=eta_eV, aaa_rtol=aaa_rtol)
        bad = np.isnan(Fr.real).any(axis=0)
        Fr[:, bad] = 0.0
        n_fail += bad
        Fw += Fr
    if average:
        Fw /= npair
    return Fw, n_fail


def nevan_continue_vpv_pairs(VPV_path, ir, egrid_eV, eta_eV=0.1, average=False):
    """Nevanlinna continuation of the VPV diagonal, one continuation per (i,a).

    Each occupied/virtual band-pair channel is mapped to its auxiliary
    Herglotz function with bosonic_to_fermionic_aux and continued with
    green_mbtools analyt_cont.nevan_run; the bosonic spectrum is recovered
    per pair as A_B = A_F * tanh(beta w / 2) and summed over pairs
    (pair-mean if average=True).

    Notes
    -----
    * The SIGNED spectral weight is returned.  Nevanlinna guarantees
      A_F >= 0 only for input satisfying the Pick criterion; interpolated
      mid-segment data violates it slightly, so small negative pockets
      remain and serve as a built-in physicality diagnostic of the
      INTERPOLATION -- inspect them before clipping.
    * nevan_run spawns a multiprocessing pool: call this from a real script
      file (it breaks under heredoc/stdin execution).
    * Cost is ~0.2 s per (pair, path-point) column at prec=128.

    Parameters
    ----------
    VPV_path : (nP, niw_bose, no, nv) complex   interpolated VPV diagonal on
               the bosonic IR Matsubara frequencies.
    ir       : irFT.IR_factory   the solver's IR transformer (solver.ir).
    egrid_eV : (nE,) real   UNIFORM real-frequency grid (eV); nevan_run
               evaluates on linspace(egrid[0], egrid[-1], nE).
    eta_eV   : float        broadening (eV).
    average  : bool         divide the pair-sum by no*nv.

    Returns
    -------
    A_B : (nE, nP) real, pair-summed signed spectral weight, scaled by
          AU2EV to match the -Im/pi (eV) maps of the aaa/plaspole branches.
    """
    from green_mbtools.pesto import analyt_cont

    nP, niw = VPV_path.shape[:2]
    npair = int(np.prod(VPV_path.shape[2:]))
    F = VPV_path.reshape(nP, niw, npair).transpose(1, 0, 2)   # (niw_b, nP, npair)
    G_aux = bosonic_to_fermionic_aux(F, ir)                   # (niw_f, nP, npair)

    wn = ir.wsample
    pos = wn > 0
    egrid = np.asarray(egrid_eV, dtype=float) / AU2EV
    freqs, A_F = analyt_cont.nevan_run(
        G_aux[pos], wn[pos], n_real=egrid.size,
        w_min=float(egrid[0]), w_max=float(egrid[-1]), eta=eta_eV / AU2EV)

    shape = (-1,) + (1,) * (A_F.ndim - 1)
    A_B = A_F * np.tanh(ir.beta * freqs.reshape(shape) / 2.0)
    A_B = A_B.reshape(len(freqs), nP, npair).sum(axis=2)
    if average:
        A_B /= npair
    return A_B * AU2EV


def pp_continue_vpv_pairs(VPV_path, omega, egrid_eV, eta_eV=0.1, average=False):
    """One-plasmon-pole continuation of the VPV diagonal, one fit per (i, a).

    Each occupied/virtual band-pair channel VPV_path[:, :, i, a] is a
    (nP, niw) scalar field that gets its own independent single-pole fit
    (pp_continue_field); the total DOS is the sum over pairs of the continued
    models (pair-mean if average=True).  Each pair channel is far closer to
    single-mode than the pair-summed field, so the one-pole model applies per
    pair even when it would not to the sum -- a plasmon-pole analogue of the
    AAA map, seam-free because each pair's 3 fit parameters vary smoothly
    with the (smooth) interpolated data.

    Parameters
    ----------
    VPV_path : (nP, niw, no, nv) complex   interpolated VPV diagonal.
    omega    : (niw,) real                 bosonic Matsubara grid (a.u.).
    egrid_eV : (nE,) real                  real-frequency grid (eV).
    eta_eV   : float                       broadening (eV).
    average  : bool                        divide the pair-sum by no*nv.

    Returns
    -------
    Fw  : (nE, nP) complex   pair-summed continued field (a.u.).
          Pairs whose fit fails (or that carry no pole weight) contribute 0.
    fit : dict of (nP, no, nv) arrays -- per-pair "wp" (a.u.), "S",
          "Finf", "resid"; NaN where the fit was skipped or failed.
    """
    nP, niw = VPV_path.shape[:2]
    pair_shape = VPV_path.shape[2:]
    Vp = VPV_path.reshape(nP, niw, -1)
    npair = Vp.shape[2]

    nE = np.asarray(egrid_eV).size
    Fw = np.zeros((nE, nP), dtype=np.complex128)
    fit = {key: np.full((nP, npair), np.nan) for key in
           ("wp", "S", "Finf", "resid")}

    for r in range(npair):
        Fr, fr = pp_continue_field(Vp[:, :, r], omega, egrid_eV, eta_eV=eta_eV)
        good = ~np.isnan(Fr.real).any(axis=0)
        Fw[:, good] += Fr[:, good]
        for key in fit:
            fit[key][:, r] = fr[key]
    if average:
        Fw /= npair
    return Fw, {key: v.reshape((nP,) + pair_shape) for key, v in fit.items()}


def bosonic_to_fermionic_aux(F_wb, ir):
    """Map a bosonic response to an auxiliary FERMIONIC function (Herglotz).

    Method: F(iOmega) --(bosonic IR)--> F(tau) evaluated on the fermionic
    tau nodes --(declare anti-periodic, i.e. apply the fermionic transform)-->
    G_aux(iomega_n).  Implemented as ir.tau_to_w(ir.wb_to_tauf(F_wb)).

    Why this works: inserting the bosonic spectral representation
    F(tau) = int dw A_B(w) e^{-tau w} / (1 - e^{-beta w}) into the fermionic
    transform integral over (0, beta) turns the (1 - e^{-beta w}) denominator
    into (1 + e^{-beta w}), so

        G_aux(i w_n) = int dw  A_F(w) / (i w_n - w),
        A_F(w) = A_B(w) * coth(beta w / 2)   (odd x odd = EVEN, >= 0).

    G_aux is therefore a genuine Herglotz (fermionic-type) function and can be
    continued with Nevanlinna (green_mbtools analyt_cont.nevan_run), which
    enforces A_F >= 0 by construction.  Recover the bosonic spectrum as

        A_B(w) = A_F(w) * tanh(beta w / 2),

    which also suppresses continuation noise near w = 0.

    Parameters
    ----------
    F_wb : (niw_bose, ...) complex   bosonic response on the bosonic IR
           Matsubara sample frequencies (e.g. a VPV column or Tr P).
    ir   : irFT.IR_factory           the solver's IR transformer (solver.ir).

    Returns
    -------
    G_aux : (niw_fermi, ...) complex on the fermionic IR sample frequencies
            i*ir.wsample (ir.wsample = (2n+1) pi / beta values).
    """
    return ir.tau_to_w(ir.wb_to_tauf(F_wb))


def vpv_pair_poles(vpv, omega, method="aaa", w_min_eV=0.1, w_max_eV=None,
                   imag_tol=1e-2, strength_tol=1e-3, pp_resid_tol=0.05,
                   aaa_rtol=1e-10):
    """Discrete pole energies of the (i,a) channels of ONE VPV diagonal.

    Intended for the NON-interpolated mesh data: continue each band-pair
    channel of vpv (e.g. one mesh-Q entry of collect_VPV_diag_over_mesh)
    independently and pool the extracted pole energies -- the exact grid
    solutions to overlay on an interpolated heatmap as a validation marker.

    method "aaa"      : per-channel AAA; keeps near-real poles
                        (|Im p| < imag_tol*|Re p|) with residue strength above
                        strength_tol * (strongest pooled residue).
    method "plaspole" : per-channel one-pole fit; keeps wp of fits with
                        relative residual < pp_resid_tol.

    Parameters
    ----------
    vpv   : (niw, no, nv) complex   VPV diagonal at one exciton momentum.
    omega : (niw,) real             bosonic Matsubara grid (a.u.).
    w_min_eV, w_max_eV : energy window (eV); w_max_eV=None -> no upper cut.

    Returns
    -------
    (n_pole,) real, sorted pole energies in eV (pooled over pairs).
    """
    F = vpv.reshape(omega.shape[0], -1)
    ener, stren = [], []

    if method == "plaspole":
        from plasPole import fit_plasmon_pole, plasmon_model
        pos = omega > 0
        om_p = omega[pos]
        for r in range(F.shape[1]):
            fd = F[pos, r].real
            if abs(fd[0] - fd[-1]) < 1e-13:
                continue
            try:
                p = fit_plasmon_pole(om_p, fd, F0=float(fd[0]),
                                     Finf=float(fd[-1]))
            except Exception:
                continue
            model = plasmon_model(1j * om_p, p["Finf"], p["S"], p["wp"]).real
            denom = np.linalg.norm(fd - fd[-1]) or 1.0
            if np.linalg.norm(model - fd) / denom > pp_resid_tol:
                continue
            ener.append(p["wp"] * AU2EV)
            stren.append(abs(p["S"]))
    else:
        try:
            from scipy.interpolate import AAA
        except Exception as exc:                   # pragma: no cover
            raise RuntimeError("vpv_pair_poles needs scipy>=1.15 (AAA).") from exc
        z = 1j * omega
        for r in range(F.shape[1]):
            try:
                rat = AAA(z, F[:, r], rtol=aaa_rtol)
            except Exception:
                continue
            for p, res in zip(rat.poles(), rat.residues()):
                if p.real > 0 and abs(p.imag) < imag_tol * abs(p.real):
                    ener.append(p.real * AU2EV)
                    stren.append(abs(res))

    if not ener:
        return np.array([])
    ener, stren = np.asarray(ener), np.asarray(stren)
    # window FIRST, then the relative strength cut: normalizing against the
    # global maximum would let a giant out-of-window pole (e.g. the ~480 eV
    # head/core weight at Gamma) suppress every in-window pole
    win = ener > w_min_eV
    if w_max_eV is not None:
        win &= ener < w_max_eV
    ener, stren = ener[win], stren[win]
    if ener.size == 0:
        return np.array([])
    return np.sort(ener[stren > strength_tol * stren.max()])


def spectral_map_from_P(P_path, omega, egrid_eV, eta_eV=0.1, aaa_rtol=1e-10):
    """Smooth A(Q, omega) heatmap directly from the interpolated polarization.

    A(Q, omega) = -(1/pi) Im Tr P(Q, omega + i*eta), obtained by AAA-continuing
    the *smooth* scalar Tr P(Q, iOmega) to the real axis -- bypassing the
    per-Q discrete pole detection (and its threshold/assignment jitter)
    entirely.  Because Tr P is smooth in Q (the interpolation is node-exact),
    so is its continuation, giving a smooth spectral field.

    Parameters
    ----------
    P_path   : (nQ, niw, NQ, NQ) complex   interpolated polarization.
    omega    : (niw,) real                 bosonic Matsubara grid (a.u.).
    egrid_eV : (nE,) real                  real-frequency grid (eV).
    eta_eV   : float                       Lorentzian broadening (eV).

    Returns
    -------
    A : (nE, nQ) real, non-negative spectral weight.
    """
    try:
        from scipy.interpolate import AAA
    except Exception as exc:                       # pragma: no cover
        raise RuntimeError("spectral_map_from_P needs scipy>=1.15 (AAA).") from exc

    nQ = P_path.shape[0]
    z = 1j * omega
    w = egrid_eV / AU2EV                            # real-axis grid in a.u.
    eta = eta_eV / AU2EV
    A = np.zeros((w.size, nQ))
    for n in range(nQ):
        trP = np.einsum("wqq->w", P_path[n])        # Tr P(iOmega), smooth in Q
        try:
            r = AAA(z, trP, rtol=aaa_rtol)
            val = r(w + 1j * eta)
        except Exception:
            continue
        A[:, n] = np.maximum(-val.imag / np.pi, 0.0)
    return A


def spectral_map_over_path(P_at_q, q_indices, q_frac_mesh, q_frac_target,
                           omega, egrid_eV, eta_eV=0.1, dim=2,
                           aaa_rtol=1e-10, verbose=True, n_workers=None):
    """Memory-light A(Q, omega) on a dense Q-path -- never stores the full matrix.

    Same result as interpolate_P_over_q -> spectral_map_from_P, but exploits the
    fact that A only needs Tr P, and the trace commutes with the (linear) Wannier
    interpolation.  So only the SCALAR Tr P(Q, iOmega) is interpolated to the
    dense path; the full (NQ x NQ) polarization is built one mesh-Q at a time and
    immediately traced.  Peak memory is O(nQ*niw) + one (niw, NQ, NQ) block,
    instead of the O(nP * niw * NQ^2) dense P_path that OOMs for large NQ.

    Parameters
    ----------
    P_at_q   : callable, P_at_q(qidx) -> (niw, NQ, NQ) complex at mesh Q qidx.
    q_indices: iterable of mesh-Q indices to build (e.g. range(nk)).
    q_frac_mesh   : (nQ, 3) fractional coords of those mesh Q (scaled units).
    q_frac_target : (nP, 3) dense path fractional coords.
    omega    : (niw,) real    bosonic Matsubara grid (a.u.).
    egrid_eV : (nE,) real     real-frequency grid (eV).
    n_workers: int or None    build the independent mesh-Q polarizations with
        this many threads (None/1 = serial).  The heavy work per Q is LAPACK /
        BLAS, which releases the GIL, so threads scale without copying the
        solver state; cap BLAS threads per worker to avoid oversubscription.

    Returns
    -------
    A        : (nE, nP) real, non-negative spectral weight.
    node_err : float, node-exactness of the Tr P interpolation (max |dTrP|).
    """
    try:
        from scipy.interpolate import AAA
    except Exception as exc:                       # pragma: no cover
        raise RuntimeError("spectral_map_over_path needs scipy>=1.15 (AAA).") from exc

    q_indices = list(q_indices)
    nQ, niw = len(q_indices), omega.shape[0]
    trP_mesh = np.empty((nQ, niw), dtype=np.complex128)

    def _trP(Q):
        return np.einsum("wqq->w", P_at_q(Q))      # trace, then discard matrix

    if n_workers and n_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for n, trP in enumerate(ex.map(_trP, q_indices)):
                if verbose:
                    print("  Tr P at mesh Q %3d/%d" % (n + 1, nQ))
                trP_mesh[n] = trP
    else:
        for n, Q in enumerate(q_indices):
            if verbose:
                print("  Tr P at mesh Q %3d/%d" % (n + 1, nQ))
            trP_mesh[n] = _trP(Q)

    qm, qt = np.asarray(q_frac_mesh), np.asarray(q_frac_target)
    obj = np.ascontiguousarray(trP_mesh.T)[:, :, None, None]   # (niw, nQ, 1, 1)
    trP_path = _interpolate(obj, qm, qt, dim, hermi=False)[:, :, 0, 0].T
    trP_back = _interpolate(obj, qm, qm, dim, hermi=False)[:, :, 0, 0].T
    node_err = float(np.max(np.abs(trP_back - trP_mesh)))

    z = 1j * omega
    w = egrid_eV / AU2EV
    eta = eta_eV / AU2EV
    A = np.zeros((w.size, trP_path.shape[0]))
    for n in range(trP_path.shape[0]):
        try:
            r = AAA(z, trP_path[n], rtol=aaa_rtol)
            A[:, n] = np.maximum(-r(w + 1j * eta).imag / np.pi, 0.0)
        except Exception:
            continue
    return A, node_err
