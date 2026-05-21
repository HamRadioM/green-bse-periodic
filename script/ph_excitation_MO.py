#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Analyze virtual-occupied orbital data from BSE calculations and output   #
#    results as molden files.                                                 #
#    The particle-hole pair MOs are stored in each molden file for each       #
#    excitation.                                                              #
#                                                                             #


import h5py
import numpy as np
import argparse
from pyscf import gto, tools
from pyscf import tools
import os


AU2EV = 27.211386245981  # Hartree to eV conversion factor

def get_ao_labels(mol):
    """
    Output AO labels for a given molecule coordinates. 
    """
    ao_labels = mol.ao_labels()
    return ao_labels

def examine_AO(file_path, type = 'occ'):
    """
    Examine the AO data for occupied or virtual orbitals from the BSE output file.
    """
    
    with h5py.File(file_path, 'r') as f:
        # Access specific datasets
        if ('/AOindices/' + type) in f:
            num_exc = f['/AOindices/' + type].shape[1]
            ao_data = f['/AOindices/' + type][:]
            polefit_data = f['/PoleFit/data'][:]

    return ao_data, polefit_data, num_exc


def write_molden(mol, occ_ao_data, virt_ao_data, ene, filename='orbital.molden'):
    """
    Write occupied and virtual molecular orbitals to a molden file.
    """
    
    # Ensure ao_data is 2D
    if occ_ao_data.ndim == 1:
        occ_ao_data = occ_ao_data.reshape(-1, 1)
    if virt_ao_data.ndim == 1:
        virt_ao_data = virt_ao_data.reshape(-1, 1)
    if ene.ndim == 0:
        ene = [ene]
    elif isinstance(ene, np.ndarray):
        ene = ene.tolist()
    # Take real part if complex
    if np.iscomplexobj(occ_ao_data):
        occ_ao_data = occ_ao_data.real
    if np.iscomplexobj(virt_ao_data):
        virt_ao_data = virt_ao_data.real
    
    # Write molden file
    with open(filename, 'w') as f:
        tools.molden.header(mol, f)
        tools.molden.orbital_coeff(mol, f, occ_ao_data)
        tools.molden.orbital_coeff(mol, f, virt_ao_data, ene=ene)
    
    print(f"Molden file written to {filename}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Examine AO data from BSE singlet HDF5 files')
    parser.add_argument('--bse_file', '-f', type=str, 
                        default='./N2_bse_singlet.h5',
                        help='Path to the BSE singlet HDF5 file')
    parser.add_argument('--molden_path', '-m', type=str, 
                        default='./',
                        help='Path to the output molden file')
    parser.add_argument('--coords_file', '-c', type=str,
                        default='../nitrogen_coords.dat', 
                        help='Path to the molecular coordinates file')
    parser.add_argument('--basis', '-b', type=str,
                        default='cc-pVDZ',
                        help='Basis set to use (default: cc-pVDZ)')
    parser.add_argument('--excitation_num', '-e', type=int,
                        default=1,
                        help='Excitation index to examine (default: 0)')
    parser.add_argument('--excitation_start', type=int,
                        default=0,
                        help='Excitation index to start with (the lowest excitation). \
                            Set this for inner shell excitations (default: 0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.molden_path, exist_ok=True)
    # Build molecule
    mol = gto.Mole()
    mol.build(atom=args.coords_file, basis=args.basis, verbose=1 if args.verbose else 0)
    Lz_ao = mol.intor('int1e_z')
    print(Lz_ao.shape)
    
    nel = mol.nelectron
    n_occ = nel // 2
    n_virt = mol.nao_nr() - n_occ
    
    # Get AO labels
    ao_labels = get_ao_labels(mol)
    
    # Examine AO data
    occ_ao_data, polefit_data, num_exc = examine_AO(args.bse_file, type='occ')
    virt_ao_data, polefit_data, num_exc = examine_AO(args.bse_file, type='virt')
    if polefit_data.ndim > 1:
        polefit_data = polefit_data[:, 0]
    
    print("=" * 60)
    print(f"BSE file: {args.bse_file}")
    print(f"Coordinates file: {args.coords_file}")
    
    print(f"Basis set: {args.basis}")
    print(f"Number of electrons: {nel}")
    print(f"Number of occupied orbitals: {n_occ}")
    print(f"Number of virtual orbitals: {n_virt}")
    print("=" * 60)
    
    for i in range(args.excitation_num):
        exc_idx = i + num_exc//2 + args.excitation_start
        print(f"\nExamining occupied orbitals for excitation index {i}")
        print(f"Plasmon-pole energy: {polefit_data[exc_idx] * AU2EV:.4f} eV ")
        write_molden(mol, occ_ao_data[:, exc_idx], virt_ao_data[:, exc_idx], polefit_data[exc_idx] * AU2EV,
                     filename=f"{args.molden_path}/{args.basis}_exc_{i + args.excitation_start}.molden")
        print("AO Labels and coefficients:")
        print("-" * 50)
        print(f"{'AO label':<12} {'Hole':>12} {'Particle':>12}")
        print("-" * 50)
        for i, label in enumerate(ao_labels):
            print(f"{label:12s} {occ_ao_data[i, exc_idx].real:12.3f} {virt_ao_data[i, exc_idx].real:12.3f}")
        
        
if __name__ == "__main__":
    main()
    