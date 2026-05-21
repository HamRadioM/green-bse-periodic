#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Plot the plasmon-pole model fitted results for BSE calculations          #
#                                                                             #

import sys
from pathlib import Path

this_dir = Path(__file__).resolve().parent
sys.path.append(str(this_dir / "../green_bse"))

import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from plasPole import plasmon_model
from plasPole import two_plasmon_model

AU2EV = 27.211386245981  # Hartree to eV conversion factor

def readBSEh5(bse_path):
    """
    Read BSE data from HDF5 output file.
    """
    with h5py.File(bse_path, 'r') as f:
        pole_loc = f["/PoleFit/data"][()]
        pole_str = f["/SFit/data"][()]
        G2pUpdated = f["/G2pUpdated/data"][()]
        res_norm = f["/ResNorm/data"][()]
        # pole_inf = f["/InfExcVals/data"][()].real
    return G2pUpdated, pole_loc, pole_str, res_norm


def plot_G2p(G2pUpdated, tau_h5, pole_loc, pole_str, res_norm, exc_idx=0, beta=1000):
    """
    Plot the original G2pUpdated and the plasmon-pole model fitted curve for a given excitation index.
    """
    print("Excitation index:", exc_idx)
    ov2 = G2pUpdated.shape[1]
    print("G2pUpdated shape:", G2pUpdated.shape)
    true_exc_idx = ov2//2 + exc_idx 
    
    with h5py.File(tau_h5, 'r') as f:
        # legacy IR grid dataset name.
        # wgrid = f["/bose/wsample"][()]
        wgrid = f["/bose/ngrid"][()]
        wgrid = 2 * wgrid * np.pi / beta
        Omega = wgrid * 1j

    # Check dimensions and use appropriate model
    if pole_str.ndim == 2 and pole_str.shape[1] == 2:
        # Two-pole model
        plasPole_fit = two_plasmon_model(Omega, G2pUpdated[0,true_exc_idx], 
                                              pole_str[true_exc_idx,0], pole_loc[true_exc_idx,0],
                                              pole_str[true_exc_idx,1], pole_loc[true_exc_idx,1])
        print("Using two-pole model")
        print("  wp1  =", pole_loc[true_exc_idx, 0])
        print("  wp2  =", pole_loc[true_exc_idx, 1])
        print("  S1   =", pole_str[true_exc_idx, 0])
        print("  S2   =", pole_str[true_exc_idx, 1])
        w_pole = pole_loc[true_exc_idx,0]
        w_str = pole_str[true_exc_idx,0]
    else:
        # Single-pole model
        plasPole_fit = plasmon_model(Omega, G2pUpdated[0,true_exc_idx], 
                                     pole_str[true_exc_idx], pole_loc[true_exc_idx])
        print("Using single-pole model")
        print("  wp   =", pole_loc[true_exc_idx])
        print("  S    =", pole_str[true_exc_idx])
        w_pole = pole_loc[true_exc_idx]
        w_str = pole_str[true_exc_idx]

    F0 = G2pUpdated[len(wgrid)//2,true_exc_idx]
    w_stat = abs((1/F0).real)
    
    # plt.figure(figsize=(5,2))
    plt.figure(figsize=(3.5,1.75))
    plt.plot(wgrid, G2pUpdated[:,true_exc_idx].real, 'x', label='Original $F(iΩ_n)$', markersize=4, color='#4e88c7')
    plt.plot(wgrid, plasPole_fit.real, '-', label='Fitted $F^{\mathrm{mod}}(iΩ_n)$', color='#ec8f9c')    
    # plt.plot(wgrid, -w_stat/(wgrid**2+w_stat**2), '-', label='Static Limit')    
    plt.xlabel("$iΩ_n$ (a.u.)")
    plt.ylabel("$F(iΩ_n)$")
    # plt.legend(fontsize=11, frameon=False)
    plt.legend(fontsize=7, frameon=False, loc='lower left')
    plt.xlim(-4, 4)
    ymax = np.ceil(max(G2pUpdated[:,true_exc_idx].real.max(), plasPole_fit.real.max()) * 2) / 2
    ymin = np.floor(min(G2pUpdated[:,true_exc_idx].real.min(), plasPole_fit.real.min()) * 2) / 2
    padding = (ymax - ymin) * 0.1
    ymax += padding
    ymin -= padding
    plt.ylim(ymin, ymax)
    info_text = (
        f"$\\Omega^\\mathrm{{stat}}$ = {w_stat * AU2EV:.4f} eV\n"
        f"$\\Omega^\\mathrm{{dyn}}$ = {w_pole * AU2EV:.4f} eV\n"
        f"$\\Delta^\\mathrm{{res}}$ = {res_norm[true_exc_idx]:.4f}"
    )
    plt.text(0.60, 0.1, info_text, transform=plt.gca().transAxes,
             fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    plt.tight_layout()
    # plt.show()
    
    
def plot_G2p_fit_error(G2pUpdated, tau_h5, pole_loc, pole_str, res_norm, exc_idx=0, beta=1000):
    """
    Plot the fitting error of the plasmon-pole model for a given excitation index. 
    The fit error is defined as the difference between the original G2pUpdated 
    and the plasmon-pole model fitted curve.
    """
    print("Excitation index:", exc_idx)
    ov2 = G2pUpdated.shape[1]
    print("G2pUpdated shape:", G2pUpdated.shape)
    true_exc_idx = ov2//2 + exc_idx 
    
    with h5py.File(tau_h5, 'r') as f:
        # legacy IR grid dataset name.
        # wgrid = f["/bose/wsample"][()]
        wgrid = f["/bose/ngrid"][()]
        wgrid = 2 * wgrid * np.pi / beta
        Omega = wgrid * 1j

    # Check dimensions and use appropriate model
    if pole_str.ndim == 2 and pole_str.shape[1] == 2:
        plasPole_fit = two_plasmon_model(Omega, G2pUpdated[0,true_exc_idx], 
                                         pole_str[true_exc_idx,0], pole_loc[true_exc_idx,0],
                                         pole_str[true_exc_idx,1], pole_loc[true_exc_idx,1])
        print("Using two-pole model")
        print("  wp1  =", pole_loc[true_exc_idx, 0])
        print("  wp2  =", pole_loc[true_exc_idx, 1])
        print("  S1   =", pole_str[true_exc_idx, 0])
        print("  S2   =", pole_str[true_exc_idx, 1])
        w_pole = pole_loc[true_exc_idx,0]
        w_str = pole_str[true_exc_idx,0]
    else:
        # Single-pole model
        plasPole_fit = plasmon_model(Omega, G2pUpdated[0,true_exc_idx], 
                                     pole_str[true_exc_idx], pole_loc[true_exc_idx])
        print("Using single-pole model")
        print("  wp   =", pole_loc[true_exc_idx])
        print("  S    =", pole_str[true_exc_idx])
        w_pole = pole_loc[true_exc_idx]
        w_str = pole_str[true_exc_idx]

    F0 = G2pUpdated[len(wgrid)//2,true_exc_idx]
    w_stat = abs((1/F0).real)
    
    plt.figure(figsize=(4,2))
    
    plt.plot(wgrid, G2pUpdated[:,true_exc_idx].real - plasPole_fit.real,'-',label='Fit error', markersize=4)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  
    plt.xlabel("$iΩ$ (A.U.)")
    plt.ylabel("Fit Error")
    plt.legend(fontsize=11, frameon=False)
    plt.xlim(-10, 10)
    plt.annotate("$\\Delta^\\mathrm{{res}}$ = {:.4f}".format(res_norm[true_exc_idx]),
                 xy=(0.65, 0.1), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))
    plt.tight_layout()
    
    
def main():
    parser = argparse.ArgumentParser(
        description="Plot the plasmon-pole model fitted results for BSE calculations."
    )
    parser.add_argument(
        "--bse_file", type=str, default="bse.h5",
        help="Input of BSE file"
    )
    parser.add_argument(
        "--ir_grid", type=str, default="1e5_136.h5",
        help="Input of IR grid file"
    )
    parser.add_argument(
        "--plot_file", type=str, default="plasmon_pole.pdf",
        help="File name for saving the plot."
    )
    parser.add_argument(
        "--exc_idx", type=int, default=0,
        help="Index for the excitation to plot."
    )
    
    # Parse arguments
    args = parser.parse_args()
    bse_path   = args.bse_file
    plot_file  = args.plot_file
    tau_h5     = args.ir_grid
    exc_idx    = args.exc_idx

    G2pUpdated, pole_loc, pole_str, res_norm = readBSEh5(bse_path)

    plot_G2p(G2pUpdated, tau_h5, pole_loc, pole_str, res_norm, exc_idx=exc_idx, beta=1000)
    plt.savefig(plot_file, dpi=300)

    plot_G2p_fit_error(G2pUpdated, tau_h5, pole_loc, pole_str, res_norm, exc_idx=exc_idx, beta=1000)
    plt.savefig("fit_error_" + plot_file, dpi=300)

if __name__ == "__main__":
    main()
    