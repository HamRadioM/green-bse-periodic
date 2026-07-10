#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
###############################################################################
#                                                                             #
#    wannier_exciton.py                                                        #
#    =================                                                         #
#    Wannier (band-limited Fourier) interpolation of the *frequency-domain*    #
#    two-step polarization P(Q, iOmega) over the exciton momentum Q, performed #
#    BEFORE the analytic continuation.                                         #
#                                                                             #
#    Why interpolate P and not the excitation energies E(Q)?                   #
#    ------------------------------------------------------------------------- #
#    E(Q) are eigenvalues (poles) of the two-step kernel.  Eigenvalues cross,  #
#    touch and reorder across the BZ, and band-limited interpolation of        #
#    eigenvalues mangles those crossings.  The full matrix P(Q, iOmega) is a   #
#    smooth analytic function of Q *through* the crossings (same reason        #
#    Wannier interpolation of H(k) works where interpolating eps_n(k) fails),  #
#    and it carries the entire spectrum at each Q, not just one isolated       #
#    branch.  We interpolate each matrix element P[iw, A, B] over the Q-mesh,   #
#    then re-extract the poles per interpolated Q with the usual continuation. #
#                                                                             #
#    CAVEAT (same as interpolating E(Q)): this is still interpolation of the   #
#    on-mesh data.  P(Q) is built from an internal BZ sum over mesh k of       #
#    V^Q(k, k+Q); evaluating at off-mesh Q would need V^Q at off-mesh k, which #
#    we do not have.  The interpolant is node-exact at the mesh Q and smooth   #
#    between -- it does NOT recover the physics of a genuinely denser k-mesh   #
#    (that needs a green-mbpt re-run).                                         #
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
def P_at_Q(solver, stored, sym, Qidx, ao, av, channel):
    """Two-step polarization that feeds the continuation, at mesh momentum Qidx.

    Mirrors the per-Q construction used in the band_and_exciton example, but
    returns the (niw, NQ, NQ) polarization itself instead of its poles:
    P_bse for the singlet channel (exchange applied at all Q), P_irr for the
    triplet channel.

    Requires the q-resolved screening intermediates (config.screening ==
    "qresolved"): solver.{Voo_all, Vvv_all, M_q, VQ_mo_kk, mo_energy} and
    solver.kmaps, as built by PeriodicTwoStepBSESolver.build_intermediates().
    """
    if solver.config.screening != "qresolved":
        raise NotImplementedError(
            "wannier_exciton currently supports screening='qresolved' only; "
            "the static mode pins Pi_stat to a single Q_exc and cannot sweep Q.")

    s = solver
    kq = s.kmaps["ksum"][:, Qidx]                      # k + Q for every k
    rSk = s.results["rSk"]

    VQ_kq_ao = pint.build_VQ_offdiag(stored, sym, kq)               # V_Q(k, k+Q)
    VQ_ia = cfq.VQ_ao2mo_kq_proper(VQ_kq_ao, s.vexMO, kq, rSk=rSk)  # -> MO o/v

    K_W = pabse.screened_direct_kernel_qresolved(
        s.Voo_all, s.Vvv_all, s.M_q, kq, s.kmaps["kdiff"],
        ao, av, s.occ, s.nk)

    P_irr, P_bse = pabse.two_step_bse_periodic(
        s.VQ_mo_kk, VQ_ia, s.mo_energy, None, kq, s.occ, s.nao, ao, av,
        s.omega, channel=channel, K_W=K_W)

    return P_bse if channel == "singlet" else P_irr


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
        P_list.append(P_at_Q(solver, stored, sym, Q, ao, av, channel))
        qf.append(kfrac[Q])

    return np.asarray(P_list), np.asarray(qf)


# --------------------------------------------------------------------------- #
#  2. Wannier interpolation of P over Q                                        #
# --------------------------------------------------------------------------- #
def interpolate_P_over_Q(P_mesh, q_frac_mesh, q_frac_target, dim=2):
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
    * hermi=False is mandatory.  P(iOmega) is NOT Hermitian for Omega != 0
      (it obeys P(iOmega)^dagger = P(-iOmega)); only the static frame is
      Hermitian.  The downstream extractor (extract_excitations) re-Hermitizes
      P(Omega ~ 0) itself when it defines the channels, so we leave the
      interpolated matrices untouched here.
    * winter.interpolate expects obj_k[ns, Nk, n, n]; we map the niw frequency
      axis onto the leading ns slot and the Q-mesh onto Nk.
    * Node-exactness: at q_frac_target points that coincide with mesh points,
      P_path reproduces P_mesh to ~1e-12.  Check this as a gauge sanity test
      (see validate_node_exactness).
    """
    nQ, niw, NQ, _ = P_mesh.shape
    obj = np.ascontiguousarray(P_mesh.transpose(1, 0, 2, 3))      # (niw, nQ, NQ, NQ)
    obj_int = winter.interpolate(obj, np.asarray(q_frac_mesh),
                                 np.asarray(q_frac_target),
                                 dim=dim, hermi=False)            # (niw, nP, NQ, NQ)
    return np.ascontiguousarray(obj_int.transpose(1, 0, 2, 3))   # (nP, niw, NQ, NQ)


def validate_node_exactness(P_mesh, q_frac_mesh, dim=2):
    """Interpolate the mesh back onto itself; return the max abs deviation.

    A small number (~1e-12) confirms the Q-gauge of P is smooth enough for the
    Wannier real-space (J(R)) representation to be well localized.  A large
    number signals a phase-gauge problem in V^Q(k, k+Q) / the transition
    densities that must be fixed before trusting interpolated points.
    """
    P_back = interpolate_P_over_Q(P_mesh, q_frac_mesh, q_frac_mesh, dim=dim)
    return float(np.max(np.abs(P_back - P_mesh)))


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


def spectral_map_over_path(P_at_Q, q_indices, q_frac_mesh, q_frac_target,
                           omega, egrid_eV, eta_eV=0.1, dim=2,
                           aaa_rtol=1e-10, verbose=True):
    """Memory-light A(Q, omega) on a dense Q-path -- never stores the full matrix.

    Same result as interpolate_P_over_Q -> spectral_map_from_P, but exploits the
    fact that A only needs Tr P, and the trace commutes with the (linear) Wannier
    interpolation.  So only the SCALAR Tr P(Q, iOmega) is interpolated to the
    dense path; the full (NQ x NQ) polarization is built one mesh-Q at a time and
    immediately traced.  Peak memory is O(nQ*niw) + one (niw, NQ, NQ) block,
    instead of the O(nP * niw * NQ^2) dense P_path that OOMs for large NQ.

    Parameters
    ----------
    P_at_Q   : callable, P_at_Q(Qidx) -> (niw, NQ, NQ) complex at mesh Q Qidx.
    q_indices: iterable of mesh-Q indices to build (e.g. range(nk)).
    q_frac_mesh   : (nQ, 3) fractional coords of those mesh Q (scaled units).
    q_frac_target : (nP, 3) dense path fractional coords.
    omega    : (niw,) real    bosonic Matsubara grid (a.u.).
    egrid_eV : (nE,) real     real-frequency grid (eV).

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
    for n, Q in enumerate(q_indices):
        if verbose:
            print("  Tr P at mesh Q %3d/%d" % (n + 1, nQ))
        trP_mesh[n] = np.einsum("wqq->w", P_at_Q(Q))   # trace, then discard matrix

    qm, qt = np.asarray(q_frac_mesh), np.asarray(q_frac_target)
    obj = np.ascontiguousarray(trP_mesh.T)[:, :, None, None]   # (niw, nQ, 1, 1)
    trP_path = winter.interpolate(obj, qm, qt, dim=dim, hermi=False)[:, :, 0, 0].T
    trP_back = winter.interpolate(obj, qm, qm, dim=dim, hermi=False)[:, :, 0, 0].T
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
