#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Two-step active-space BSE from the polarization (RPA-scaling).           #
#                                                                             #
#    Motivation                                                               #
#    ----------                                                               #
#    The full BSE costs O(niw · (occ·virt)^3) — the dense particle-hole       #
#    ladder.  But the screened-exchange (-W) binding that distinguishes the   #
#    BSE from the RPA is dominated by transitions near the Fermi level.  So   #
#    we keep the (expensive) -W ladder ONLY in a small, fixed-size active     #
#    space and treat the rest of the system at the RPA level:                 #
#                                                                             #
#      1.  P^act   = active-space screened -W ladder           (Q-space)      #
#      2.  P^irr   = P^act - P^act(0) + P^(0)                                 #
#             ("replace the bare active bubble with the full-space bubble")   #
#      3.  P^BSE   = [I - 2 P^irr]^{-1} P^irr                  (Q-space RPA)   #
#                                                                             #
#    Properties                                                               #
#    ----------                                                               #
#    * EXACT in the full-active limit: when the active space is the whole     #
#      orbital set, step 3 supplies the +2v exchange and the result is the    #
#      one-shot (Tamm-Dancoff) BSE -- the push-through identity               #
#          [I - 2 V chi V†]^{-1} V chi V† = V [I - 2 chi (V†V)]^{-1} chi V†    #
#      turns the Q-space RPA into the +2v exchange in the orbital-pair space. #
#    * Controlled approximation for a small active space (converges as the    #
#      active space grows).                                                   #
#    * SCALING: the O(N_act^3) ladder is confined to a fixed-size active      #
#      space; everything else is NQ x NQ (the full bubble and the RPA).       #
#      Overall O(niw · NQ^3) -- standard RPA scaling, no (occ·virt)^3 wall.   #
#                                                                             #
#    Conventions                                                              #
#    ----------                                                               #
#    * Tamm-Dancoff (resonant) ladder.  This is the version for which the     #
#      two-step scheme is exact in the full-active limit.  (The de-excitation #
#      / B-block coupling would enter the Q-space RPA with the wrong metric   #
#      sign and is left out -- the residual TDA-vs-BSE gap is ~0.1 eV.)       #
#    * V_Q is the density-fitting integral in the MO basis, transformed with  #
#      the MO coefficients ALONE (no overlap S):  V_mo = C† V_ao C            #
#      (see casidaEq.VQ_ao2mo with S=None).                                   #
#    * Excitation energies are the poles of the Q-space polarization,         #
#      obtained by AAA rational analytic continuation off the Matsubara axis. #
#                                                                             #
#    NOTE: molecular / single-k (Gamma) only for now.  The structure is       #
#    written so a k-point loop can wrap eval_active_ladder_Q / the RPA later. #

import numpy as np

try:
    from scipy.interpolate import AAA
    _HAVE_AAA = True
except Exception:                       # older scipy without AAA
    _HAVE_AAA = False

AU2EV = 27.211386245981


# ===========================================================================
# Kernel: screened direct -W in an active orbital space
# ===========================================================================
def screened_direct_kernel(VQ_mo, M, act_occ, act_virt):
    """
    Tamm-Dancoff screened-direct BSE kernel  K_W = -W  in the active ov space.

        W(ij,ab) = Σ_{Q,Q'} V[Q,i,j] M[Q,Q'] V[Q',a,b]          (i,j occ; a,b virt)
        K_W[(i,a),(j,b)] = - W(ij,ba)

    with  M = I + tildeP  the screened Coulomb in the auxiliary basis
    (tildeP = full-system dressed polarizability in Q-space).

    Parameters
    ----------
    VQ_mo : ndarray (NQ, nmo, nmo) complex
        Density-fitting integrals in the MO basis (no overlap S; see module doc).
    M : ndarray (NQ, NQ) complex
        I + tildeP(iΩ) at the chosen (static) frequency.
    act_occ, act_virt : array-like of int
        Active occupied / virtual MO indices.

    Returns
    -------
    K_W : ndarray (n_act, n_act) complex,  n_act = len(act_occ)*len(act_virt)
    """
    act_occ = np.asarray(act_occ)
    act_virt = np.asarray(act_virt)
    Voo = VQ_mo[:, act_occ][:, :, act_occ]          # (NQ, no, no)
    Vvv = VQ_mo[:, act_virt][:, :, act_virt]        # (NQ, nv, nv)
    n = act_occ.size * act_virt.size
    # W[i,j,b,a] = Σ_QQ' Voo[Q,i,j] M[Q,Q'] Vvv[Q',b,a];  K_W[(ia),(jb)] = -W[i,j,b,a]
    K_W = -np.einsum('Qij,QP,Pba->iajb', Voo, M, Vvv, optimize=True).reshape(n, n)
    return K_W


# ===========================================================================
# Active-space (or full) Tamm-Dancoff polarization in Q-space
# ===========================================================================
def eval_active_ladder_Q(VQ_mo, mo_energy, act_occ, act_virt, omega,
                         kernel=None, qp_weights=None):
    """
    Q-space polarization of a TDA particle-hole propagator:

        P(iΩ)_{QQ'} = Σ_{ia,jb} V[Q,i,a] [ (iΩ I - A)^{-1} ]_{ia,jb} V*[Q',j,b]

    with the (active) two-particle Hamiltonian  A = Δ + kernel,
    Δ_{ia} = ε_a - ε_i.

    * kernel = screened_direct_kernel(...)  -> the screened -W ladder  (P^act)
    * kernel = None                         -> the bare bubble         (P^act(0)
                                               or, with the full ov space, P^(0))

    Quasiparticle-weight renormalization (optional, beyond-QP)
    ----------------------------------------------------------
    The QP-pole residue of the bubble is Z_i Z_a (Z = quasiparticle weight).
    Writing z_{ia} = Z_i Z_a, this enters as sqrt(z) on the transition density
    and on each side of the kernel:

        A   ->  Δ + √z K √z ,        V_ov  ->  V_ov √z .

    Z < 1 reduces the spectral weight, weakening the e-h binding and shifting
    excitations up.  qp_weights = None (default) sets z = 1 (sharp QP poles).

    Parameters
    ----------
    VQ_mo : (NQ, nmo, nmo) complex
    mo_energy : (nmo,)  quasiparticle (or mean-field) energies
    act_occ, act_virt : active occ / virt MO indices
    omega : (niw,)  bosonic Matsubara frequencies Ω_n (real)
    kernel : (n_act, n_act) or None
    qp_weights : (nmo,) or None
        Per-orbital quasiparticle weights Z_p (see active_bse.quasiparticle_weights).

    Returns
    -------
    P : (niw, NQ, NQ) complex
    """
    act_occ = np.asarray(act_occ)
    act_virt = np.asarray(act_virt)
    NQ = VQ_mo.shape[0]
    no, nv = act_occ.size, act_virt.size
    n = no * nv
    niw = omega.shape[0]

    Delta = (mo_energy[act_virt][None, :] - mo_energy[act_occ][:, None]).reshape(n)
    Vov = VQ_mo[:, act_occ][:, :, act_virt].reshape(NQ, n)     # (NQ, n)

    sqz = None
    if qp_weights is not None:
        z = np.asarray(qp_weights)
        sqz = np.sqrt(z[act_occ][:, None] * z[act_virt][None, :]).reshape(n)

    A = np.diag(Delta).astype(np.complex128)
    if kernel is not None:
        A = A + (kernel if sqz is None else sqz[:, None] * kernel * sqz[None, :])
    if sqz is not None:
        Vov = Vov * sqz[None, :]

    P = np.zeros((niw, NQ, NQ), dtype=np.complex128)
    In = np.eye(n, dtype=np.complex128)
    for iw in range(niw):
        G2p = np.linalg.solve(1j * omega[iw] * In - A, Vov.conj().T)   # (n, NQ)
        P[iw] = Vov @ G2p
    return P


# ===========================================================================
# Full (resonant + antiresonant, spin-summed) active bubble in Q-space
# ===========================================================================
def eval_active_ladder_full(VQ_mo, mo_energy, act_occ, act_virt, omega,
                            kernel=None, qp_weights=None):
    """
    Spin-summed FULL particle-hole bubble in Q-space (resonant + antiresonant):

        P(iW)_{QQ'} = 2 sum_{ia} V[Q,ia] V*[Q',ia]
                          [ (iW I - A)^{-1}_{ia,jb}  -  (iW I + Delta)^{-1}_{ia} ]

    i.e. the resonant block carries the (optional) -W ladder kernel A = Delta + K,
    while the antiresonant block stays bare (Tamm-Dancoff binding: the kernel is
    applied to the resonant block only).  With kernel=None and the full ov space
    this is *exactly* the bubble that gwtool.eval_P0_tilde_Q produces from a
    sharp-pole G(tau) -- verified to 1e-13 (see example/.../validate_dressed_bubble.py),
    which is what makes a dressed-G backbone drop-in consistent.

    The leading factor 2 is the closed-shell spin sum; it matches the "2 *" in
    eval_P0_tilde_Q.  Because of it, the Q-space RPA that supplies the singlet
    +2v exchange uses spin_factor=1 here (the 2 is already inside P), not 2.

    Parameters mirror eval_active_ladder_Q.

    Returns
    -------
    P : (niw, NQ, NQ) complex
    """
    act_occ = np.asarray(act_occ)
    act_virt = np.asarray(act_virt)
    NQ = VQ_mo.shape[0]
    no, nv = act_occ.size, act_virt.size
    n = no * nv
    niw = omega.shape[0]

    Delta = (mo_energy[act_virt][None, :] - mo_energy[act_occ][:, None]).reshape(n)
    Vov = VQ_mo[:, act_occ][:, :, act_virt].reshape(NQ, n)

    sqz = None
    if qp_weights is not None:
        z = np.asarray(qp_weights)
        sqz = np.sqrt(z[act_occ][:, None] * z[act_virt][None, :]).reshape(n)

    A = np.diag(Delta).astype(np.complex128)
    if kernel is not None:
        A = A + (kernel if sqz is None else sqz[:, None] * kernel * sqz[None, :])
    Vres = Vov if sqz is None else Vov * sqz[None, :]

    P = np.zeros((niw, NQ, NQ), dtype=np.complex128)
    In = np.eye(n, dtype=np.complex128)
    for iw in range(niw):
        z = 1j * omega[iw]
        G_res = np.linalg.solve(z * In - A, Vres.conj().T)            # (n, NQ)
        anti = (Vres / (z + Delta)[None, :]) @ Vres.conj().T          # (NQ, NQ)
        P[iw] = 2.0 * (Vres @ G_res - anti)
    return P


# ===========================================================================
# Step 2: bubble replacement
# ===========================================================================
def bubble_replacement(P_act, P_act0, P0_full):
    """P^irr = P^act - P^act(0) + P^(0)  (replace the bare active bubble by the
    full-space bubble, keeping the active-space -W ladder corrections)."""
    return P_act - P_act0 + P0_full


# ===========================================================================
# Step 3: Q-space RPA  (supplies the +2v exchange -> singlet)
# ===========================================================================
def qspace_rpa(P_irr, spin_factor=2.0):
    """
    P^BSE(iΩ) = [I - f P^irr(iΩ)]^{-1} P^irr(iΩ),  f = spin_factor.

    f = 2 gives the spin-singlet (density) channel.  Skip this step entirely
    (use P^irr directly) for the spin-triplet, which has no +2v exchange.
    """
    niw, NQ, _ = P_irr.shape
    I = np.eye(NQ, dtype=np.complex128)
    P_bse = np.zeros_like(P_irr)
    for iw in range(niw):
        P_bse[iw] = np.linalg.solve(I - spin_factor * P_irr[iw], P_irr[iw])
    return P_bse


# ===========================================================================
# High-level driver
# ===========================================================================
def two_step_bse(VQ_mo, mo_energy, tildeP_static, occ, act_occ, act_virt,
                 omega, channel="singlet", qp_weights=None):
    """
    Two-step active-space BSE polarization in Q-space.

    Parameters
    ----------
    VQ_mo : (NQ, nmo, nmo) complex
        DF integrals in the MO basis (no overlap S).
    mo_energy : (nmo,)  quasiparticle energies.
    tildeP_static : (NQ, NQ) complex
        Full-system dressed polarizability in Q-space at the static point
        (iΩ = 0).  M = I + tildeP_static enters the screened W.
    occ : int
        Number of occupied orbitals (Fermi split).
    act_occ, act_virt : active occ / virt MO indices (the ladder space).
    omega : (niw,)  bosonic Matsubara grid.
    channel : "singlet" (apply the +2v RPA) or "triplet" (return P^irr).
    qp_weights : (nmo,) or None
        Optional per-orbital quasiparticle weights Z_p for the beyond-QP
        (Z-renormalized) variant; None gives the standard sharp-QP BSE.  The
        √(Z_i Z_a) factors propagate consistently into both the active -W
        ladder (P^act) and the bubbles, and -- through the transition densities
        carried by P^irr -- into the +2v RPA as well.  See
        active_bse.quasiparticle_weights.

    Returns
    -------
    P_irr : (niw, NQ, NQ)   irreducible polarization (= triplet polarization)
    P_bse : (niw, NQ, NQ)   BSE polarization (singlet); equals P_irr for triplet
    """
    nmo = VQ_mo.shape[1]
    full_occ = list(range(occ))
    full_virt = list(range(occ, nmo))
    M = np.eye(VQ_mo.shape[0], dtype=np.complex128) + tildeP_static

    K_W = screened_direct_kernel(VQ_mo, M, act_occ, act_virt)
    P_act  = eval_active_ladder_Q(VQ_mo, mo_energy, act_occ, act_virt, omega,
                                  kernel=K_W, qp_weights=qp_weights)
    P_act0 = eval_active_ladder_Q(VQ_mo, mo_energy, act_occ, act_virt, omega,
                                  kernel=None, qp_weights=qp_weights)
    P0_full = eval_active_ladder_Q(VQ_mo, mo_energy, full_occ, full_virt, omega,
                                   kernel=None, qp_weights=qp_weights)

    P_irr = bubble_replacement(P_act, P_act0, P0_full)
    if channel == "triplet":
        return P_irr, P_irr
    P_bse = qspace_rpa(P_irr, spin_factor=2.0)
    return P_irr, P_bse


def two_step_bse_full(VQ_mo, mo_energy, tildeP_static, occ, act_occ, act_virt,
                      omega, channel="singlet", qp_weights=None,
                      backbone=None):
    """
    Two-step active-space BSE in the FULL (resonant + antiresonant, spin-summed)
    convention -- the convention in which a *dressed* (beyond-QP) backbone bubble
    can be substituted consistently.

    Same three steps as two_step_bse, but every bubble is the spin-summed full
    bubble (eval_active_ladder_full), so:

      * the full-active limit targets the full-BSE (not the Tamm-Dancoff BSE);
      * the singlet RPA uses spin_factor=1 (the spin 2 is already inside P);
      * the triplet excitations are the poles of P^irr (a scalar prefactor does
        not move poles, so triplet energies are insensitive to the spin/exchange
        bookkeeping -- the cleanest probe of the backbone).

    backbone : None | (niw, NQ, NQ) complex
        The full-space P^(0) backbone.
        * None  -> sharp-QP backbone: eval_active_ladder_full over the full ov
                   space (the QP control, equals analytic_full_bubble).
        * array -> a precomputed dressed backbone, e.g.
                       ir.tauf_to_wb(symmetrize_P0(eval_P0_tilde_Q(...)))[:,0,0]
                   (the scGW dressed-G bubble; same convention, verified to 1e-13).
        Must be on the same omega grid and in the same (C-only, no-S) metric.

    Returns
    -------
    P_irr, P_bse : (niw, NQ, NQ) each.  P_bse == P_irr for the triplet channel.
    """
    nmo = VQ_mo.shape[1]
    full_occ = list(range(occ))
    full_virt = list(range(occ, nmo))
    M = np.eye(VQ_mo.shape[0], dtype=np.complex128) + tildeP_static

    K_W = screened_direct_kernel(VQ_mo, M, act_occ, act_virt)
    P_act  = eval_active_ladder_full(VQ_mo, mo_energy, act_occ, act_virt, omega,
                                     kernel=K_W, qp_weights=qp_weights)
    P_act0 = eval_active_ladder_full(VQ_mo, mo_energy, act_occ, act_virt, omega,
                                     kernel=None, qp_weights=qp_weights)
    if backbone is None:
        P0_full = eval_active_ladder_full(VQ_mo, mo_energy, full_occ, full_virt,
                                          omega, kernel=None, qp_weights=qp_weights)
    else:
        P0_full = np.asarray(backbone)

    P_irr = bubble_replacement(P_act, P_act0, P0_full)
    if channel == "triplet":
        return P_irr, P_irr
    P_bse = qspace_rpa(P_irr, spin_factor=1.0)
    return P_irr, P_bse


# ===========================================================================
# Excitation energies: poles of P(iΩ) by AAA analytic continuation
# ===========================================================================
def _channel_curves(P_iw, omega, eig_tol):
    """Diagonalise P(Ω≈0) and return (channel curves c[m](iΩ), eigenvalues)."""
    niw = omega.shape[0]
    iw0 = int(np.argmin(np.abs(omega)))
    P_stat = 0.5 * (P_iw[iw0] + P_iw[iw0].conj().T)
    ev, U = np.linalg.eigh(P_stat.real)
    keep = np.abs(ev) > eig_tol * np.abs(ev).max()
    curves = {}
    for m in np.where(keep)[0]:
        curves[m] = np.array([U[:, m].conj() @ P_iw[iw] @ U[:, m] for iw in range(niw)])
    return curves, ev


def extract_excitations(P_iw, omega, method="aaa", eig_tol=1e-4,
                        imag_tol=1e-2, strength_tol=1e-3, w_min=1e-3,
                        w_max=np.inf, cluster=0.05, n_keep=12, aaa_rtol=1e-10,
                        pp_resid_tol=1e-3):
    """
    Excitation energies = poles of P(iΩ).

    P is diagonalised at Ω≈0 to define fixed channels; each channel curve
    P̃_mm(iΩ) is analytically continued and its poles collected.

    method :
      "plaspole" : a single plasmon-pole fit per channel (fast, robust, but only
                   correct for a single-mode channel — see pp_resid_tol).
      "aaa"      : an AAA rational continuation per channel (resolves all modes).
      "hybrid"   : single plasmon-pole where the fit is clean
                   (relative residual < pp_resid_tol), AAA on the rest.  Default.

    Returns
    -------
    list of (energy_eV, strength) sorted by energy.
    """
    if method in ("aaa", "hybrid") and not _HAVE_AAA:
        import warnings
        warnings.warn(
            "scipy.interpolate.AAA unavailable (needs scipy >= 1.15); falling back "
            "to method='plaspole'.  Single-mode channels stay exact, but an "
            "excitation buried in a multi-mode channel may be missed.  Upgrade "
            "scipy (or run with a python that has it) for the full extractor.",
            RuntimeWarning)
        method = "plaspole"
    if method in ("plaspole", "hybrid"):
        from plasPole import fit_plasmon_pole, plasmon_model

    pos = omega > 0
    om_p = omega[pos]
    curves, _ = _channel_curves(P_iw, omega, eig_tol)
    z = 1j * omega

    def aaa_poles(c):
        out = []
        try:
            r = AAA(z, c, rtol=aaa_rtol)
        except Exception:
            return out
        for p, res in zip(r.poles(), r.residues()):
            if w_min < p.real < w_max and abs(p.imag) < imag_tol * abs(p.real):
                out.append((p.real * AU2EV, abs(res)))
        return out

    def pp_pole(c):
        """Single plasmon-pole fit; returns (energy, strength, rel_residual) or None."""
        fd = c[pos].real
        if abs(fd[0] - fd[-1]) < 1e-13:
            return None
        try:
            p = fit_plasmon_pole(om_p, fd, F0=float(fd[0]), Finf=float(fd[-1]))
        except Exception:
            return None
        model = plasmon_model(1j * om_p, p["Finf"], p["S"], p["wp"]).real
        denom = np.linalg.norm(fd - fd[-1]) or 1.0
        resid = np.linalg.norm(model - fd) / denom
        if not (w_min < p["wp"] < w_max):
            return None
        return (p["wp"] * AU2EV, abs(p["S"]), resid)

    raw = []
    for m, c in curves.items():
        if method == "aaa":
            raw += aaa_poles(c)
        elif method == "plaspole":
            r = pp_pole(c)
            if r is not None:
                raw.append((r[0], r[1]))
        else:  # hybrid
            r = pp_pole(c)
            if r is not None and r[2] < pp_resid_tol:
                raw.append((r[0], r[1]))          # clean single mode
            else:
                raw += aaa_poles(c)               # multi-mode -> AAA

    if not raw:
        return []
    smax = max(s for _, s in raw)
    raw = sorted((e, s) for e, s in raw if s > strength_tol * smax)
    out = []
    for e, s in raw:
        if out and abs(e - out[-1][0]) < cluster:
            e0, s0 = out[-1]
            out[-1] = ((e0 * s0 + e * s) / (s0 + s), s0 + s)
        else:
            out.append((e, s))
    return out[:n_keep]


# ===========================================================================
# Spectral function from the extracted poles
# ===========================================================================
def spectral_function(excitations, omega, eta=0.1):
    """
    Lorentzian-broadened spectral function from extract_excitations() output:

        A(ω) = Σ_n  S_n / π · η / [ (ω - E_n)^2 + η^2 ]

    Parameters
    ----------
    excitations : list of (energy_eV, strength)   (output of extract_excitations)
    omega : (npts,)  real-frequency grid (eV)
    eta : float      Lorentzian half-width (eV)

    Returns
    -------
    A : (npts,) float
    """
    omega = np.asarray(omega, dtype=float)
    A = np.zeros_like(omega)
    for e, s in excitations:
        A += s * eta / ((omega - e) ** 2 + eta ** 2) / np.pi
    return A


def plot_spectrum(excitations, ax=None, omega=None, eta=0.1, npts=2000,
                  pad=2.0, stems=False, label=None, **plot_kw):
    """
    Plot the spectral function A(ω) built from extract_excitations() output.

    Parameters
    ----------
    excitations : list of (energy_eV, strength)
    ax : matplotlib Axes, or None to create a new figure/axis
    omega : (npts,) eV grid, or None to auto-span [min(E)-pad, max(E)+pad]
    eta : Lorentzian half-width (eV)
    stems : if True, draw a vertical stem at each pole (height ∝ strength)
    label : curve label
    **plot_kw : forwarded to ax.plot

    Returns
    -------
    (omega, A) : the energy grid (eV) and the spectral function.
    """
    import matplotlib.pyplot as plt
    energies = [e for e, _ in excitations]
    if omega is None:
        lo = max(0.0, min(energies) - pad) if energies else 0.0
        hi = (max(energies) + pad) if energies else 1.0
        omega = np.linspace(lo, hi, npts)
    A = spectral_function(excitations, omega, eta)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    line, = ax.plot(omega, A, label=label, **plot_kw)
    if stems:
        smax = max((s for _, s in excitations), default=1.0) or 1.0
        h = A.max() if A.max() > 0 else 1.0
        for e, s in excitations:
            ax.vlines(e, 0, h * s / smax, color=line.get_color(), lw=0.8, alpha=0.4)
    ax.set_xlabel(r"$\omega$ (eV)")
    ax.set_ylabel(r"$A(\omega)$")
    return omega, A


# ===========================================================================
# Helper: quasiparticle weights Z (optional, for Z-renormalized variants)
# ===========================================================================
def quasiparticle_weights(sim_h5, vexMO, mo_energy, beta, ir_file):
    """
    QP weights  Z_p = 1 / (1 - dReΣ/dω)|_{ε_p}  from a scGW self-energy
    (Padé analytic continuation; reuses pyscf's Thiele Padé).

    Parameters
    ----------
    sim_h5 : path to the scGW sim file (with iterN/Selfenergy, iterN/mu).
    vexMO : (nmo, nmo)  MO coefficients used to rotate Σ to the MO basis.
    mo_energy : (nmo,)  QP energies at which to evaluate dΣ/dω.
    beta, ir_file : inverse temperature and IR grid file.
    """
    import h5py
    import contract as ct
    from pyscf.gw.gw_ac import AC_pade_thiele_diag, pade_thiele

    with h5py.File(sim_h5, "r") as f:
        last = max((k for k in f if k.startswith("iter") and k[4:].isdigit()),
                   key=lambda s: int(s[4:]))
        Sig = f[f"{last}/Selfenergy/data"][()]
        mu = float(f[f"{last}/mu"][()])
    nao = vexMO.shape[0]
    Sig_mo = np.einsum("ab,tbc,cd->tad", vexMO.conj().T, Sig[:, 0, 0], vexMO)
    Sig_diag = np.einsum("tii->ti", Sig_mo)[:, None, None, :]
    Sig_iw = ct.tau2omegaFTforG(Sig_diag, beta, ir_file)
    with h5py.File(ir_file, "r") as f:
        ng = f["/fermi/ngrid"][()]
    iwf = (2 * ng + 1) * np.pi / beta
    iwp = iwf[iwf > 0]
    Sig_pos = Sig_iw[iwf > 0][:, 0, 0, :]
    idx = np.arange(0, min(40, len(iwp) - 1))
    coeff, ofit = AC_pade_thiele_diag(Sig_pos[idx].T, 1j * np.tile(iwp[idx], (nao, 1)))
    coeff = np.asarray(coeff)
    Z = np.zeros(nao)
    h = 1e-3
    for p in range(nao):
        sig = lambda w: pade_thiele(np.array([w - mu]), ofit[p], coeff[:, p]).real[0]
        Z[p] = 1.0 / (1.0 - (sig(mo_energy[p] + h) - sig(mo_energy[p] - h)) / (2 * h))
    return Z
