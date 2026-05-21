#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Plot the plasmon-pole model fitted spectrum for BSE calculations         #
#                                                                             #

import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt

AU2EV = 27.211386245981  # Hartree to eV conversion factor

def readBSEh5_dynamic(bse_path):
    """
    Read BSE dynamical solution data from HDF5 output file.
    """
    with h5py.File(bse_path, 'r') as f:
        pole_loc = f["/PoleFit/data"][()]
        pole_str = f["/SFit/data"][()]
    return pole_loc, pole_str


def readBSEh5_static(bse_path):
    """
    Read BSE static solution data from HDF5 output file.
    """
    with h5py.File(bse_path, 'r') as f:
        pole_loc = f["/StatExcVals/data"][()]
    return pole_loc


def plot_static_pole(pole_loc, eta, wmin=0, wmax=20, npts=1000):
    """
    Plot the static solution as poles.
    """
    w = np.linspace(wmin, wmax, npts)
    pole_str = 0.5 * np.ones_like(pole_loc.real)  # Arbitrary strength for static poles
    
    # single-pole Lorentzian spectral function
    A = np.zeros_like(w)
    for i in range(len(pole_loc)):
        if pole_str[i] > 0:
            A += pole_str[i] * eta / ((w - pole_loc[i].real)**2 + eta**2)
        elif pole_str[i] < 0:
            A += pole_str[i] * eta / ((w + pole_loc[i].real)**2 + eta**2)
    
    # Plot pole locations as vertical lines
    # y_max is the height of the first peak

    plt.plot(w, A, color='#ec8f9c', label='Stat. BSE@sc$GW$')
    plt.fill_between(w, A, alpha=0.3, color='#ec8f9c')
    
    # y_max = 0.25 * plt.ylim()[1]
    # for i in range(len(pole_loc)):
    #     plt.axvline(pole_loc[i], color='#ec8f9c', linestyle='-', alpha=0.5, ymin=0, ymax=y_max/plt.ylim()[1], label='Stat. BSE@sc$GW$' if i == 0 else None)
    

def plot_spectral_function(pole_loc, pole_str, eta, wmin=0, wmax=20, npts=1000):
    """
    Plot the spectral function based on the plasmon-pole model fitted results.
    """
    w = np.linspace(wmin, wmax, npts)

    # single-pole Lorentzian spectral function
    A = np.zeros_like(w)
    for i in range(len(pole_loc)):
        if pole_str[i] > 0:
            A += pole_str[i] * eta / ((w - pole_loc[i])**2 + eta**2)
        elif pole_str[i] < 0:
            A += pole_str[i] * eta / ((w + pole_loc[i])**2 + eta**2)
    
    plt.plot(w, A, color='#4e88c7', label='Dyn. BSE@sc$GW$')
    plt.fill_between(w, A, alpha=0.3, color='#4e88c7')
    
    # Plot pole locations as vertical lines
    # y_max is the height of the first peak
    # Do not plot the vertical lines for dynamic poles.
    # For dynamic poles, the spectral function already captures their contribution, 
    # and plotting vertical lines may be redundant and visually cluttered.
    
    # y_max = 0.25 * plt.ylim()[1]

    # for i in range(len(pole_loc)):
    #     if pole_str[i] > 0:
    #         plt.axvline(pole_loc[i], color='#ec8f9c', linestyle='-', alpha=0.5, ymin=0, ymax=y_max/plt.ylim()[1])
    #     elif pole_str[i] < 0:
    #         plt.axvline(-pole_loc[i], color='#ec8f9c', linestyle='-', alpha=0.5, ymin=0, ymax=y_max/plt.ylim()[1])

    plt.xlim(wmin, wmax)
    plt.ylim(plt.ylim()[1] * -0.01, plt.ylim()[1])
    # plt.title("BSE Plasmon-Pole Model Fitted Spectral Function")
    # plt.show()
    
    
def main():
    parser = argparse.ArgumentParser(
        description="Plot the plasmon-pole model fitted results for BSE calculations."
    )
    parser.add_argument(
        "--bse_file", type=str, default="bse.h5",
        help="Input of BSE file"
    )
    parser.add_argument(
        "--w_max", type=float, default=20,
        help="Maximum frequency for plotting (in units of eV)."
    )
    parser.add_argument(
        "--w_min", type=float, default=0,
        help="Minimum frequency for plotting (in units of eV)."
    )
    parser.add_argument(
        "--eta", type=float, default=1e-2,
        help="Broadening factor for plotting (in units of Hartree)."
    )
    parser.add_argument(
        "--plot_file", type=str, default="neutral_excitation_spectrum.pdf",
        help="File name for saving the plot."
    )

    # Parse arguments
    args = parser.parse_args()
    bse_path   = args.bse_file
    w_min      = args.w_min
    w_max      = args.w_max
    eta        = args.eta
    plot_file  = args.plot_file
    
    plt.figure(figsize=(3.5,1.75))
    
    pole_loc = readBSEh5_static(bse_path)
    pole_loc *= AU2EV 
    
    plot_static_pole(pole_loc, eta, wmin=w_min, wmax=w_max)

    pole_loc, pole_str = readBSEh5_dynamic(bse_path)
    pole_loc *= AU2EV 
    
    plot_spectral_function(pole_loc, pole_str, eta, wmax=w_max, wmin=w_min)
    
    plt.xlabel(r"$\Omega$ (eV)")
    plt.ylabel(r"$\rho_{A}$ (arb. unit)")
    plt.legend(fontsize=8, frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)

if __name__ == "__main__":
    main()
    