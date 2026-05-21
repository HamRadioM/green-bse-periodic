#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Functions for manipulating and contracting tensors.                      #
#                                                                             #


import h5py
from irFT import IR_factory
import gwtool as gw


def readH5(filename, dataset):
    # read .h5 file.
    fi = h5py.File(filename, 'r')  
    tensor = fi[dataset][()].view(complex)
    return tensor


def readVQ(path="df_hf_int/VQ_0.h5"):
    # Only for one block of VQ at the moment. 
    VQ = readH5(path, "/0")

    return VQ


def readPtilde(filename = "p_iw_tilde_q0.h5"):
    # Read P tilde from h5 files.
    fi = h5py.File(filename, 'r')  
    iter = fi["/iter"][()]  # get the number of iteration.
    Ptilde = readH5(filename, "/iter{}/P_iw_tilde".format(iter))

    return Ptilde


def getPtilde_init(nQ,beta=1000,
                   tau_h5="1e5_120.h5", 
                   int_path="df_hf_int/",
                   input_h5="input.h5"):
    # Initialize P tilde from the initial G.
    P0 = gw.eval_P0_tilde_Q_init(nQ, beta=beta, 
                                 int_path=int_path, input_h5=input_h5, tau_h5=tau_h5)
    P0 = gw.symmetrize_P0(P0)
    # It is on the tau axis. This is evaluated with the Dyson equation.
    Ptilde_init = gw.eval_P_tilde(P0, tau_h5=tau_h5)

    return Ptilde_init


def getPtilde(iter,nao,nQ,tau_h5="1e5_120.h5", int_path="df_hf_int/", sim_h5="sim.h5"):
    # Compute P tilde from G and V.
    P0 = gw.eval_P0_tilde_Q(iter, nao, nQ, int_path=int_path, sim_h5=sim_h5)
    P0 = gw.symmetrize_P0(P0)
    Ptilde = gw.eval_P_tilde(P0, tau_h5=tau_h5)
    # print("Ptilde shape:", Ptilde.shape)

    return Ptilde


def tau2omegaFT(tau, beta = 1000, tau_h5 = "/home/wenm/irgrids/" + "1e5_120.h5"):
    print("Performing Fourier transformation from img time to img freq...")
    fourier = IR_factory(beta, tau_h5)
    omega = fourier.tauf_to_wb(tau)
    
    return omega


def tau2omegaFTforG(tau, beta = 1000, tau_h5 = "/home/wenm/irgrids/" + "1e5_120.h5"):
    # For G or any Fermionic quantity.
    print("Performing Fourier transformation from img time to img freq...")
    fourier = IR_factory(beta, tau_h5)
    omega = fourier.tau_to_w(tau)
    
    return omega


def omega2tauFT(omega, beta = 1000, tau_h5 = "/home/wenm/irgrids/" + "1e5_120.h5"):
    print("Performing Fourier transformation from img freq to img time...")
    fourier = IR_factory(beta, tau_h5)
    tau = fourier.wb_to_tauf(omega)
    
    return tau


def omega2tauFTforG(omega, beta = 1000, tau_h5 = "/home/wenm/irgrids/" + "1e5_120.h5"):
    print("Performing Fourier transformation from img freq to img time...")
    fourier = IR_factory(beta, tau_h5)
    tau = fourier.w_to_tau(omega)
    
    return tau

