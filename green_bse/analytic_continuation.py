#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.   #
#                                                                             #
#    Eigenmode analytic continuation for polarization matrices.               #
#                                                                             #
#    Why eigenmode fitting?                                                   #
#    ----------------------                                                   #
#    P(iΩ) in the density-fitting (DF) basis is not diagonal in general.     #
#    For a 2-band insulator with NQ=36 aux functions the ratio                #
#    ||off-diag|| / ||diag|| is typically 0.5–1.0.  Fitting raw DF            #
#    diagonals gives a poor single-pole description because each DF           #
#    diagonal mixes multiple physical modes.                                  #
#                                                                             #
#    P(iΩ) is also low-rank: rank = n_val × n_con × nk in DF space,          #
#    so only a small number of eigenmodes are nonzero.                        #
#                                                                             #
#    Correct approach (quasi-static diagonalisation):                         #
#      1. Symmetrise and diagonalise P(Ω=0)  → U  (fixed rotation)           #
#      2. Rotate P(iΩ) → P̃(iΩ) = U† P(iΩ) U,  extract diagonal             #
#      3. Fit a plasmon pole to each P̃_{mm}(iΩ) independently               #
#      4. A_m(ω) = −Im P̃_{mm}(ω+iη)/π                                       #
#                                                                             #
#    Trace invariance guarantees:                                             #
#      Σ_m A_m(ω) = −Im Tr P(ω+iη)/π   (basis-independent physical result)  #

import numpy as np
from scipy.linalg import eigh

from plasPole import fit_plasmon_pole, plasmon_model


# ---------------------------------------------------------------------------
# Step 1 – fixed rotation from P(Ω=0)
# ---------------------------------------------------------------------------

def eigenbasis(P_iw: np.ndarray):
    """
    Diagonalise P at Ω=0 to obtain a fixed rotation U.

    Parameters
    ----------
    P_iw : (niw, ns, 1, NQ, NQ)  complex polarization on the Matsubara axis

    Returns
    -------
    eigvals : (NQ,)    real eigenvalues of P(Ω=0), sorted by |value| descending
    U       : (NQ, NQ) columns are eigenvectors (unitary)
    """
    i0     = P_iw.shape[0] // 2          # Ω = 0 sits at the grid centre
    P_stat = P_iw[i0, 0, 0]             # (NQ, NQ) complex
    P_sym  = 0.5 * (P_stat + P_stat.conj().T)  # symmetrise for numerical safety
    eigvals, U = eigh(P_sym.real)        # real symmetric → real eigenvalues
    order   = np.argsort(-np.abs(eigvals))
    return eigvals[order], U[:, order]


# ---------------------------------------------------------------------------
# Step 2 – project all P(iΩ) onto the eigenmode basis
# ---------------------------------------------------------------------------

def rotate_to_eigbasis(P_iw: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Project P(iΩ) onto the eigenmode basis and return the diagonal.

    P̃(iΩ) = U† P(iΩ) U  →  real(diag P̃)
    The imaginary part of the diagonal is zero for a Hermitian P on the
    imaginary axis; taking the real part discards numerical rounding only.

    Parameters
    ----------
    P_iw : (niw, ns, 1, NQ, NQ)
    U    : (NQ, NQ)

    Returns
    -------
    P_eig_diag : (niw, NQ)  real
    """
    niw, _, _, NQ, _ = P_iw.shape
    P_eig_diag = np.empty((niw, NQ))
    Uh = U.conj().T
    for iw in range(niw):
        P_eig_diag[iw] = (Uh @ P_iw[iw, 0, 0] @ U).real.diagonal()
    return P_eig_diag


# ---------------------------------------------------------------------------
# Step 3 + 4 – plasmon-pole fit per eigenmode + real-axis spectral function
# ---------------------------------------------------------------------------

def fit_eigenmodes(P_eig_diag: np.ndarray, Omega: np.ndarray,
                   omega: np.ndarray, eta: float,
                   eigval_tol: float = 0.01):
    """
    Fit a plasmon pole to each eigenmode curve P̃_{mm}(iΩ) and evaluate
    A_m(ω) = −Im P̃_{mm}(ω+iη)/π on the real axis.

    Modes whose |P̃_{mm}(Ω=0)| < eigval_tol × max(|P̃_{mm}(Ω=0)|) are
    treated as numerical zeros (they arise from the low-rank structure of P)
    and are skipped — their A_m is left as zero.

    Parameters
    ----------
    P_eig_diag : (niw, NQ)  real diagonals in eigenmode basis
    Omega      : (niw,)     Matsubara frequencies (Ha)
    omega      : (npts,)    real-axis grid (Ha)
    eta        : float      Lorentzian broadening (Ha)
    eigval_tol : float      fraction of max eigenvalue below which modes
                            are considered zero (default 0.01 = 1 %)

    Returns
    -------
    A        : (npts, NQ)  per-eigenmode spectral function
    fit_eig  : (niw,  NQ)  plasmon-pole model values on iΩ axis
    Finf     : (NQ,)       high-frequency asymptote per eigenmode
    S        : (NQ,)       pole strength
    wp       : (NQ,)       pole frequency (Ha)
    """
    niw  = Omega.shape[0]
    NQ   = P_eig_diag.shape[1]
    npts = omega.shape[0]
    i0   = niw // 2
    z_iw = 1j * Omega
    z_re = omega + 1j * eta

    A        = np.zeros((npts, NQ))
    fit_eig  = np.zeros((niw,  NQ))
    Finf_arr = np.zeros(NQ)
    S_arr    = np.zeros(NQ)
    wp_arr   = np.zeros(NQ)

    static    = np.abs(P_eig_diag[i0])
    threshold = eigval_tol * static.max() if static.max() > 0 else 0.0

    for m in range(NQ):
        data = P_eig_diag[:, m]
        if static[m] < threshold:
            fit_eig[:, m] = data        # store raw (≈ 0) for diagnostic purposes
            continue

        Finf = data[-1]
        F0   = data[i0]
        try:
            fit = fit_plasmon_pole(Omega, data, F0=F0, Finf=Finf)
        except Exception:
            fit_eig[:, m] = data
            continue

        Finf_m, S_m, wp_m = fit['Finf'], fit['S'], fit['wp']
        Finf_arr[m] = Finf_m
        S_arr[m]    = S_m
        wp_arr[m]   = wp_m

        fit_eig[:, m] = plasmon_model(z_iw, Finf_m, S_m, wp_m).real
        P_re           = Finf_m + 2.0 * wp_m * S_m / (wp_m**2 - z_re**2)
        A[:, m]        = -P_re.imag / np.pi

    return A, fit_eig, Finf_arr, S_arr, wp_arr


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def fit_ac(P_iw: np.ndarray, Omega: np.ndarray,
           omega: np.ndarray, eta: float,
           eigval_tol: float = 0.01):
    """
    Full eigenmode analytic-continuation pipeline.

    Parameters
    ----------
    P_iw      : (niw, ns, 1, NQ, NQ)
    Omega     : (niw,)   Matsubara frequencies (Ha)
    omega     : (npts,)  real-axis grid (Ha)
    eta       : float    broadening (Ha)
    eigval_tol: float    eigenmode significance threshold (default 1 %)

    Returns
    -------
    A          : (npts, NQ)  per-eigenmode spectral function
                             A.sum(axis=1) = −Im Tr P(ω+iη)/π
    fit_eig    : (niw,  NQ)  fitted model on iΩ axis per eigenmode
    eigvals    : (NQ,)       eigenvalues at Ω=0 (|·| descending)
    U          : (NQ, NQ)    eigenvectors (columns)
    P_eig_diag : (niw, NQ)  raw eigenmode curves that were fitted
    Finf       : (NQ,)
    S          : (NQ,)
    wp         : (NQ,)       pole frequencies (Ha)
    """
    eigvals, U = eigenbasis(P_iw)
    P_eig_diag = rotate_to_eigbasis(P_iw, U)
    A, fit_eig, Finf, S, wp = fit_eigenmodes(P_eig_diag, Omega, omega, eta,
                                              eigval_tol=eigval_tol)
    return A, fit_eig, eigvals, U, P_eig_diag, Finf, S, wp


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def omega_grid(ir_file: str, beta: float) -> np.ndarray:
    """Read bosonic Matsubara frequencies from an IR grid file."""
    import h5py
    with h5py.File(ir_file, 'r') as f:
        ngrid = f['/bose/ngrid'][()]
    return 2.0 * ngrid * np.pi / beta


def spectral_peak_eV(A: np.ndarray, omega: np.ndarray) -> float:
    """Peak position (eV) of the mode-summed spectral function A (npts, NQ)."""
    AU2EV = 27.211386245981
    return float(omega[np.argmax(A.sum(axis=1))] * AU2EV)
