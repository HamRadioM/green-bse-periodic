#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Plasmon-pole model fitting for BSE calculations                          #
#    Fits G(iΩ) data to single/two plasmon-pole models.                       #
#                                                                             #

import numpy as np
import h5py
from scipy.optimize import least_squares, minimize_scalar


AU2EV = 27.211386245981  # Hartree to eV conversion factor

def plasmon_model(z, Finf, S, wp):
    """
    Single plasmon-pole model for a bosonic response function F(z).
    """
    return Finf + 2 * wp * S / (wp**2 - z**2)


def fit_plasmon_pole(Omega, Fdata, F0=None, Finf=None):
    """
    Fit F(iOmega) data to a plasmon-pole model function.
    The model is parameterized with Finf, S, and wp. 
    Finf is usually the high-frequency limit of F(iOmega), which can be fixed as zero.
    It represents a pair of poles at ± wp on the real axis, with strength S.
    """
    z = 1j * Omega
    
    # trapezoidal quadrature weights
    # because Omega is not uniformly spaced
    # https://en.wikipedia.org/wiki/Trapezoidal_rule
    
    dOmega = np.empty_like(Omega)
    dOmega[1:-1] = 0.5 * (Omega[2:] - Omega[:-2])
    dOmega[0] = Omega[1] - Omega[0]
    dOmega[-1] = Omega[-1] - Omega[-2]

    w = np.sqrt(dOmega)
    
    niw = len(Omega)

    def residuals(params):
        if Finf is None:
            Finf_fit, S, wp = params
        else:
            Finf_fit = Finf
            S, wp = params
        model_vals = plasmon_model(z, Finf_fit, S, wp)
        diff = Fdata - model_vals.real
        # Weighted residuals
        res = w * diff.real
        
        return res

    # Initial guesses
    if F0 is None and Finf is None:
        Finf0 = Fdata[-1].real
        S0 = (Fdata[niw//2].real - Finf0) * 1.0
        wp0 = abs(1/Fdata[niw//2].real)
        x0 = [Finf0, S0, wp0]
        bounds = ([-np.inf, -np.inf, 1e-8], [np.inf, np.inf, np.inf])
    else:
        S0 = (F0.real - Finf.real) * 1.0
        wp0 = abs(1/F0.real)
        x0 = [S0, wp0]
        bounds = ([-np.inf, 1e-8], [np.inf, np.inf])

    res = least_squares(residuals, x0, 
                        bounds=bounds,    
                        ftol=1e-8,
                        xtol=1e-8,
                        gtol=1e-8)
    
    if Finf is None:
        Finf_fit, S_fit, wp_fit = res.x
    else:
        Finf_fit = Finf
        S_fit, wp_fit = res.x

    return {
        "Finf": Finf_fit,
        "S": S_fit,
        "wp": wp_fit,
        "residual_norm": np.linalg.norm(res.fun)
    }


def fit_plasmon_pole_set_S(Omega, Fdata, F0=None, Finf=None):
    """
    Fit F(iOmega) data to a plasmon-pole model function.
    The model is parameterized  with Finf, S, and wp. 
    It represents a pair of poles at ± wp on the real axis, with strength S.
    This version sets S based on the sign of F0 - Finf, and only fits wp.
    """
    z = 1j * Omega
    w = 1.0
    
    F_sign = np.sign(F0 - Finf)
    S_fit = F_sign * 0.5  # Fixed S based on F0 and Finf
    Finf_fit = Finf
    def objective(wp):
        model = plasmon_model(z, Finf_fit, S_fit, wp)
        diff = w * (Fdata.real - model)
        # return np.linalg.norm(diff)
        return np.trapz(diff**2, Omega)
    
    wp_min = 1e-8
    wp_max = Omega.max()
    
    res = minimize_scalar(
        objective,
        bounds=(wp_min, wp_max),
        method="bounded"
    )
    
    wp_fit = res.x
    
    return {
        "Finf": Finf_fit,
        "S": S_fit,
        "wp": wp_fit,
        "residual_norm": res.fun
    }


def fit_G_update(Fdata, ir_file, beta=1000):
    """
    Load F(iOmega) data from BSE output file, fit to plasmon-pole model,
    and return the fitted parameters.
    Referred as "G_update" here, the two-particle response function G2p_update
    is rendered in the BSE algorithm.
    It is not the same as the single-particle Green's function G.
    """
    print("Fitting to single-plasmon-pole model.")
    with h5py.File(ir_file, 'r') as f:
        # use ngrid, which is the Matsubara frequencies integer.
        wgrid = f["/bose/ngrid"][()]
        wgrid = 2 * wgrid * np.pi / beta
    
    print(f"Fdata shape: {Fdata.shape}")
    Fdata = Fdata.real
    n_exc = Fdata.shape[1]
    niw = wgrid.shape[0]

    S_data = np.zeros(n_exc, dtype=np.float64)
    wpole_data = np.zeros(n_exc, dtype=np.float64)
    Finf_data = np.zeros(n_exc, dtype=np.float64)
    res_norm_data = np.zeros(n_exc, dtype=np.float64)

    for exc in range(n_exc):
        try:
            Finf = Fdata[-1, exc]
            F0 = Fdata[niw // 2, exc]
            fit_results = fit_plasmon_pole(wgrid, Fdata[:, exc], F0=F0, Finf=Finf)
            # fit_results = fit_plasmon_pole_set_S(wgrid, Fdata[:, exc], F0=F0, Finf=Finf)
            
            # Store results
            S_data[exc] = fit_results['S']
            wpole_data[exc] = fit_results['wp']
            Finf_data[exc] = fit_results['Finf']
            res_norm_data[exc] = fit_results['residual_norm']
        except Exception as e:
            print(f"Error fitting excitation {exc}: {e}")

    return wpole_data, S_data, Finf_data, res_norm_data

#                                                                             #
#    Two plasmon-pole fitting model.                                          #
#    It uses four parameters: S1, wp1, S2, wp2                                #
#    Which represent the strengths and frequencies of the two pair of poles.  #
#    Test functions: It is not in use currently.                              #
#                                                                             #

def two_plasmon_model(z, Finf, S1, wp1, S2, wp2):
    """Two plasmon-pole model for F(z)."""
    return Finf + 2 * wp1 * S1 / (wp1**2 - z**2) + 2 * wp2 * S2 / (wp2**2 - z**2)


def fit_two_plasmon_pole(Omega, Fdata, F0=None, Finf=None):
    """
    Fit F(iOmega) data to a two plasmon-pole model:
        F(z) = Finf + S1 / (wp1^2 - z^2) + S2 / (wp2^2 - z^2).
    """
    z = 1j * Omega
    # Weights: downweight high-frequency points
    # Frequencies near zero imaginaries (iΩ = 0) are more important
    # w = 1.0 / (1.0 + Omega**2)
    w = 1.0

    # Initial guesses for two poles
    wp1_0 = 0.01
    wp2_0 = 0.01

    def residuals(params):
        if Finf is None:
            Finf_fit, S1, wp1, S2, wp2 = params
        else:
            Finf_fit = Finf
            S1, wp1, S2, wp2 = params
        model_vals = two_plasmon_model(z, Finf_fit, S1, wp1, S2, wp2)
        diff = Fdata - model_vals.real
        # res = np.concatenate([(w * diff.real), (w * diff.imag)])
        res = np.trapz(w*diff**2, Omega)
        return res

    # Initial guesses
    if F0 is None and Finf is None:
        Finf0 = Fdata[-1].real
        S_total = (Fdata[0].real - Finf0) * 1.0
        S1_0 = S_total * 1.1
        S2_0 = S_total * (-0.1)
        x0 = [Finf0, S1_0, wp1_0, S2_0, wp2_0]
        bounds = ([-np.inf, -np.inf, 1e-8, -np.inf, 1e-8], 
                    [np.inf, np.inf, np.inf, np.inf, np.inf])
    else:
        F_sign = np.sign(F0 - Finf)
        S_total = (F0 - Finf) * 1.0
        S1_0 = S_total * 0.49
        S2_0 = S_total * 0.01
        x0 = [S1_0, wp1_0, S2_0, wp2_0]
        if F_sign >= 0:
            bounds = ([1e-8, 1e-8, -np.inf, 1e-8], 
                      [np.inf, np.inf, np.inf, np.inf])
            # bounds = ([1e-8, 1e-8, 1e-8, 1e-8], 
            #           [np.inf, np.inf, np.inf, np.inf])
        else:
            bounds = ([-np.inf, 1e-8, -np.inf, 1e-8], 
                      [-1e-8, np.inf, np.inf, np.inf])
            # bounds = ([-np.inf, 1e-8, -np.inf, 1e-8], 
            #           [-1e-8, np.inf, -1e-8, np.inf])
            
    res = least_squares(residuals, x0, bounds=bounds)

    if Finf is None:
        Finf_fit, S1_fit, wp1_fit, S2_fit, wp2_fit = res.x
    else:
        Finf_fit = Finf
        S1_fit, wp1_fit, S2_fit, wp2_fit = res.x

    return {
        "Finf": Finf_fit,
        "S1": S1_fit,
        "wp1": wp1_fit,
        "S2": S2_fit,
        "wp2": wp2_fit,
        "residual_norm": np.linalg.norm(res.fun)
    }


def fit_G_update_two_pole(Fdata, ir_file, beta=1000):
    """
    Load F(iOmega) data from BSE output file, fit to two-plasmon-pole model,
    and return fitted parameters.
    """
    
    print("Fitting to two-plasmon-pole model.")
    with h5py.File(ir_file, 'r') as f:
        wgrid = f["/bose/ngrid"][()]
        wgrid = 2 * wgrid * np.pi / beta
    
    print(f"Fdata shape: {Fdata.shape}")
    Fdata = Fdata.real
    n_exc = Fdata.shape[1]
    niw = wgrid.shape[0]

    S1_data = np.zeros(n_exc, dtype=np.float64)
    wpole1_data = np.zeros(n_exc, dtype=np.float64)
    S2_data = np.zeros(n_exc, dtype=np.float64)
    wpole2_data = np.zeros(n_exc, dtype=np.float64)
    Finf_data = np.zeros(n_exc, dtype=np.float64)
    res_norm_data = np.zeros(n_exc, dtype=np.float64)

    for exc in range(n_exc):
        try:
            Finf = Fdata[-1, exc]
            F0 = Fdata[niw // 2, exc]
            fit_results = fit_two_plasmon_pole(wgrid, Fdata[:, exc], F0=F0, Finf=Finf)
            
            # Store results
            S1 = fit_results['S1']
            wp1 = fit_results['wp1']
            S2 = fit_results['S2']
            wp2 = fit_results['wp2']
            
            # Keep the peak that has the same sign as F0 - Finf as the first peak
            # If both satisfy this condition, keep the larger peak as S1
            F0_Finf_sign = np.sign(F0 - Finf)
            S1_same_sign = np.sign(S1) == F0_Finf_sign
            S2_same_sign = np.sign(S2) == F0_Finf_sign
            
            if S1_same_sign and S2_same_sign:
                # Both have same sign, keep larger absolute value as S1
                if abs(S2) > abs(S1):
                    S1, S2 = S2, S1
                    wp1, wp2 = wp2, wp1
            elif S2_same_sign and not S1_same_sign:
                # Only S2 has correct sign, swap them
                S1, S2 = S2, S1
                wp1, wp2 = wp2, wp1
            # If only S1 has correct sign or neither does, keep as is
            
            S1_data[exc] = S1
            S2_data[exc] = S2
            wpole1_data[exc] = wp1
            wpole2_data[exc] = wp2
            Finf_data[exc] = fit_results['Finf']
            res_norm_data[exc] = fit_results['residual_norm']
        except Exception as e:
            print(f"Error fitting excitation {exc}: {e}")

    # Combine S1 and S2 into one array, same for wpole1 and wpole2
    S_data = np.stack([S1_data, S2_data], axis=1)
    wpole_data = np.stack([wpole1_data, wpole2_data], axis=1)

    return wpole_data, S_data, Finf_data, res_norm_data