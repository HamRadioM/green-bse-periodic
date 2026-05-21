#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    The Casida equation functions for BSE.                                   #
#    Parallelized frequency loops for improved performance.                   #
#                                                                             #


import numpy as np
import h5py
import scipy.linalg as LA
from joblib import Parallel, delayed 
from scipy.linalg import matrix_balance


def matEleXiStat(VQ, W, nelec, type="normal"):
    """
    Calculate static matrix block A for BSE.
    A(ia,jb) = Eps(ia,jb) + U(ia,bj) - W(ij,ab)
    VQ is in the form of V_{Qij}, in OV basis.
    """
    occ = nelec // 2
    virt = VQ.shape[-1] - occ
    kappa = 1.0
    
    # i (occ), a (virt), j (occ), b (virt)
    Xi = np.zeros((occ, virt, occ, virt), dtype=np.complex128)
    if (type != "singlet") and (type != "triplet"):
        kappa = 1.0
        U = np.einsum('qia,qjb->iajb', VQ[0, :, :occ, occ:], VQ[0, :, :occ, occ:], optimize='optimal')
    elif type == "singlet":
        kappa = 2.0
        U = np.einsum('qia,qjb->iajb', VQ[0, :, :occ, occ:], VQ[0, :, :occ, occ:], optimize='optimal')
    elif type == "triplet":
        kappa = 0.0

    for a in range(virt):
        for j in range(occ):
            for b in range(virt):
                if type == "triplet":
                    # For triplet excitations, the exchange term is zero.
                    Xi[:, a, j, b] = - W[:, j, b, a]
                else:
                    Xi[:, a, j, b] = kappa * U[:, a, j, b] - W[:, j, b, a]
    
    return Xi


def matEleBStat(VQ, W, nelec, type="normal"):
    """
    Calculate static matrix block B for BSE.
    B(ia,jb) = U(ia,bj) - W(ib,aj)
    VQ is in the form of V_{Qij}, in OV basis.
    """
    occ = nelec // 2
    virt = VQ.shape[-1] - occ
    kappa = 1.0
    B = np.zeros((occ, virt, occ, virt), dtype=np.complex128)
    
    if (type != "singlet") and (type != "triplet"):
        kappa = 1.0
        U = np.einsum('qia,qjb->iajb', VQ[0, :, :occ, occ:], VQ[0, :, occ:, :occ], optimize='optimal')
    elif type == "singlet":
        kappa = 2.0
        U = np.einsum('qia,qjb->iajb', VQ[0, :, :occ, occ:], VQ[0, :, occ:, :occ], optimize='optimal')
    elif type == "triplet":
        kappa = 0.0
    
    # U shape is (occ,virt,virt,occ)
    # B shape is (occ,virt,occ,virt)
    for a in range(virt):
        for j in range(occ):
            for b in range(virt):
                if type == "triplet":
                    # For triplet excitations, the exchange term is zero.
                    B[:, a, j, b] = - W[:, b, j, a]
                else:
                    B[:, a, j, b] = kappa * U[:, a, b, j] - W[:, b, j, a]

    return B


def fix_phase(mo):
    """
    Fix the phase of the molecular orbitals so that 
    the largest component is real and positive.
    """
    for i in range(mo.shape[1]):
        k = np.argmax(np.abs(mo[:, i]))  # index of largest element
        phase = np.angle(mo[k, i])
        mo[:, i] *= np.exp(-1j * phase)  # rotate so largest component is real
    return mo


def solveMO(F, S, eigh_solver=LA.eigh, thr=1e-7):
    """
    Solve the generalized eigenvalue problem F C = S C E for each k-point and spin channel.
    F and S are in the shape of (ns, nk, nao, nao).
    """
    ns, nk, nao = F.shape[0:3]
    eiv_sk = np.zeros((ns, nk, nao))
    mo_coeff_sk = np.zeros((ns, nk, nao, nao), dtype=F.dtype)
    if S is None:
        S = np.array([[np.eye(nao)]*nk]*ns)
    for ss in range(ns):
        for k in range(nk):
            # eiv, mo = eigh_solver(F[ss, k], S[ss, k], thr)
            eiv, mo = eigh_solver(F[ss, k], S[ss, k])
            # Re-order
            idx = np.argmax(abs(mo.real), axis=0)
            mo[:, mo[idx, np.arange(len(eiv))].real < 0] *= -1
            mo = fix_phase(mo)
            nbands = eiv.shape[0]
            eiv_sk[ss, k, :nbands] = eiv
            mo_coeff_sk[ss, k, :, :nbands] = mo

    return eiv_sk, mo_coeff_sk


def VQ_ao2mo(VQ, mo_vecs):
    """
    Transform the V_{Qij} tensor from AO basis to MO basis.
    """
    # dimmension of VQ is (1, nQ, nao, nao)
    VQ_mo  = np.zeros(VQ.shape,dtype=np.complex128)  
    for ik in range(VQ.shape[0]):
        for iQ in range(VQ.shape[1]):
            # temp = M_ao[ik, iQ] @ C  -> shape (nao, nmo)
            temp = VQ[ik, iQ] @ mo_vecs
            # M_mo[ik, iQ] = C.T @ temp -> shape (nmo, nmo)
            VQ_mo[ik, iQ] = mo_vecs.conj().T @ temp

    return VQ_mo


def effVex2AO(effVex, moVex, nelec):
    """
    Transform effVex to AO basis, find out the occ and virt components.
    """
    nao = moVex.shape[-1]
    occ = nelec // 2
    virt = nao - occ
    
    # Separate the mo_vecs matrix into two blocks
    moVex_occ = moVex[:, :occ]    # size = nao * occ(ao)
    moVex_virt = moVex[:, occ:]   # size = nao * virt(ao)
    
    # mo2ao is of the shape (nao, 2*occ*virt)
    mo2ao_occ = np.zeros((nao, 2 * occ * virt), dtype=np.complex128)
    mo2ao_virt = np.zeros((nao, 2 * occ * virt), dtype=np.complex128)
    
    for block in range(2):
        sign = 1 if block == 0 else -1
        for i in range(0, occ):
            for a in range(0, virt):
                mo2ao_occ[:,  block * occ * virt + i * virt + a] = sign * moVex_occ[:, i]
                mo2ao_virt[:, block * occ * virt + i * virt + a] = sign * moVex_virt[:, a]
    
    # effVex is of the shape (2*occ*virt, num excitations)
    # effVex_ao is of the shape (nao, num excitations)
    effVex_occ_ao = mo2ao_occ @ effVex
    effVex_virt_ao = mo2ao_virt @ effVex
    
    # Renormalize effVex_ao[:, ex]
    for ex in range(effVex.shape[1]):
        norm = np.sqrt(np.sum(abs(effVex_occ_ao[:, ex])**2))
        if norm > 1e-10:
            effVex_occ_ao[:, ex] /= norm
        norm = np.sqrt(np.sum(abs(effVex_virt_ao[:, ex])**2))
        if norm > 1e-10:
            effVex_virt_ao[:, ex] /= norm
    
    # The electron is being promoted from occ to virt
    return effVex_occ_ao, effVex_virt_ao


def mo2ovStat(mat4PStat_mo):
    """
    This is an empty placeholder function because matrices are 
    already in OV basis after transformation.
    In case we need to do some reordering or reshaping, we can implement it here.
    """
    mat4PStat_ov = mat4PStat_mo[:]

    return mat4PStat_ov


def diffEpsMat(eigVals,nelec):
    """
    Calculate the difference in orbital energies for occupied and virtual orbitals.
    Return a 4D array of shape (n_occ, n_virt, n_occ, n_virt) used in building block A.
    """
    nao = len(eigVals)
    nocc = nelec // 2
    nvirt = nao - nocc
    diffEps = np.zeros((nocc,nvirt,nocc,nvirt),dtype=np.complex128)
    # ia,jb
    # i = j and a = b
    for i in range(nocc):
        for a in range(nvirt):
            diffEps[i,a,i,a] += eigVals[a+nocc] - eigVals[i]
        
    return diffEps


def concatAB(A,B):
    """
    Concatenate two matrices A and B into the effective Hamiltonian.
    Only for static solution.
    """
    block_dim = A.shape[-1]
    temp = np.zeros((2*block_dim,2*block_dim),dtype=np.complex128)
    temp[0:block_dim,0:block_dim] += A
    temp[block_dim:,0:block_dim] += -B.T.conj()
    temp[0:block_dim,block_dim:] += B
    temp[block_dim:,block_dim:] += -A.T.conj()

    return temp


def getDiffEps(valsMO,nelec):
    """
    Get the difference in orbital energies for occupied and virtual orbitals.
    """
    nao   = valsMO.shape[-1]
    occ   = nelec // 2
    virt  = nao - occ
    diffEps = diffEpsMat(valsMO[0,0,:],nelec)
    diffEps_ov = np.zeros([occ*virt,occ*virt],dtype=np.complex128)
    for i in range(occ):
        for a in range(occ,nao):
            # iajb -> ia,jb
            ia = i * virt + a - occ
            diffEps_ov[ia, ia] += diffEps[i, a, i, a] 

    return diffEps_ov


def HinfDiagApprox(effVex,VQ,valsMO,nelec,ex_type="singlet",tda=0):
    """
    Solve the Hamiltonian in the limit(iomega -> infinity). 
    Must be in MO basis already.
    return the diagonalized H_inf matrix.
    """
    occ  = nelec // 2
    virt = VQ.shape[-1] - occ
    diffEps_ov = mo2ovStat(diffEpsMat(valsMO[0,0,:],nelec)).reshape(occ*virt,occ*virt)
    nao  = VQ.shape[-1]
    virt = nao - occ
    # n_occ * n_occ * n_virt * n_virt
    U1   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    # n_occ * n_virt * n_occ * n_virt
    U2   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,:occ,occ:], optimize='optimal')
    
    # Diagonal approximation to H2p_Dyn at each frequency point.
    H2p_inf = np.zeros((2*occ*virt),dtype=np.complex128)

    W_inf   = U1
    A   = matEleXiStat(VQ,W_inf,nelec,ex_type)
    A   = diffEps_ov + A.reshape(occ*virt,occ*virt)
    if not tda:
        W_inf = U2
        B = matEleBStat(VQ,W_inf,nelec,ex_type)
    else:
        print("TDA approximation is enabled. Using TDA effective Hamiltonian.")
        # B block is zero in TDA.
        B = np.zeros((occ,virt,occ,virt),dtype=np.complex128)

    B   = B.reshape(occ*virt,occ*virt)
    H   = concatAB(A,B)
    # Not necessarily Hermitian.
    effVex_inv = LA.inv(effVex)
    
    H2p_inf += np.einsum('ij,jk,ki->i', effVex_inv, H, effVex)

    return H2p_inf


def HStatDiagApprox(Pi_stat,effVex,VQ,valsMO,nelec,ex_type="singlet",tda=0):
    """
    Solve the Hamiltonian in the limit (iomega -> infinity). 
    Must be in MO basis already.
    return the diagonalized H_stat matrix.
    """
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    diffEps_ov = mo2ovStat(diffEpsMat(valsMO[0,0,:],nelec)).reshape(occ*virt,occ*virt)
    # print("*****    Solving static effective Hamiltonian    *****")
    # Qkl: NQ * Nvirt * Nvirt
    PV  = np.einsum("qp,pkl->qkl", Pi_stat, VQ[0,:,occ:,occ:], optimize='optimal') 
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    U   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    W_stat  = VPV + U
    # iajb: Nocc * Nvirt * Nocc * Nvirt
    A_stat = matEleXiStat(VQ,W_stat,nelec,ex_type)
    
    if not tda:
        print("TDA approximation is disabled. Solving full effective Hamiltonian.")
        # Qkl: NQ * Nvirt * Nocc
        PV  = np.einsum("qp,pkl->qkl", Pi_stat, VQ[0,:,occ:,:occ], optimize='optimal') 
        # ijkl: Nocc * Nvirt * Nvirt * Nocc
        VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,occ:], PV, optimize='optimal')
        # U   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,occ:,:occ], optimize='optimal')
        # ijkl: Nocc * Nvirt * Nocc * Nvirt 
        U   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,:occ,occ:], optimize='optimal')
        W_stat  = VPV + U
        B_stat = matEleBStat(VQ,W_stat,nelec,ex_type)
    else:
        print("TDA approximation is enabled. Using TDA effective Hamiltonian.")
        # B block is zero in TDA.
        B_stat = np.zeros((occ,virt,occ,virt),dtype=np.complex128)

    A_stat  = diffEps_ov + A_stat.reshape(occ*virt,occ*virt)
    B_stat  = B_stat.reshape(occ*virt,occ*virt)
    
    H_stat  = concatAB(A_stat,B_stat)
    # check condition number.
    cond = np.linalg.cond(H_stat)
    print(f" Approximating the diag of static Hamiltonian, Condition number = {cond:10.4f}")
    # Diagonal approximation to H2p_Dyn at each frequency point.
    H2p_stat = np.zeros((2*occ*virt),dtype=np.complex128)
    effVex_inv = LA.inv(effVex)
    H2p_stat += np.einsum('ij,jk,ki->i', effVex_inv, H_stat, effVex)

    return H2p_stat


def initG2p_inv(H2p_inf,ir_file,beta=1000):
    """
    Initialize the diagonlaized two-particle response function
    G2p^(-1) from H2p at limit (iomega -> inf).
    """
    # legacy support
    # wgrid = h5py.File(ir_file,"r")["/bose/wsample"][()]
    # green-mbpt support
    wgrid = h5py.File(ir_file,"r")["/bose/ngrid"][()]
    wgrid = 2 * wgrid * np.pi / beta
    niw = len(wgrid)

    G2p_inv_init = np.zeros((niw,H2p_inf.shape[0]),dtype=np.complex128)
    for iw in range(niw):
        # G2p_inv_init[iw,:] = (1j * wgrid[iw] - H2p_inf)
        G2p_inv_init[iw,:] = (wgrid[iw]**2 + H2p_inf**2)/(-H2p_inf)

    return G2p_inv_init


def G2p_inv(H2p,ir_file,beta=1000):
    """
    Diagonalized G2p from H2p at all iomega.
    """
    # legacy support
    # wgrid = h5py.File(ir_file,"r")["/bose/wsample"][()]
    # green-mbpt support
    wgrid = h5py.File(ir_file,"r")["/bose/ngrid"][()]
    wgrid = 2 * wgrid * np.pi / beta
    niw = len(wgrid)

    G2p_inv = np.zeros((niw,H2p.shape[1]),dtype=np.complex128)
    for iw in range(niw):
        # G2p_inv[iw,:] = (1j * wgrid[iw] - H2p[iw,:])
        G2p_inv[iw,:] = (wgrid[iw]**2 + H2p[iw,:]**2)/(-H2p[iw,:])

    return G2p_inv


def G2p(H2p,ir_file,beta=1000):
    """
    Build the two-particle response function G2p from H2p at all iomega.
    """
    # legacy support
    # wgrid = h5py.File(ir_file,"r")["/bose/wsample"][()]
    # green-mbpt support
    wgrid = h5py.File(ir_file,"r")["/bose/ngrid"][()]
    wgrid = 2 * wgrid * np.pi / beta
    niw = len(wgrid)

    G2p = np.zeros((niw,H2p.shape[1]),dtype=np.complex128)
    for iw in range(niw):
        # G2p[iw,:] = 1 / (1j * wgrid[iw] - H2p[iw,:])
        G2p[iw,:] = -H2p[iw,:]/(wgrid[iw]**2 + H2p[iw,:]**2)

    return G2p


def _process_hdyn_frequency(iw, Pi, VQ, effVex_inv, effVex, occ, virt, diffEps_ov, nelec, ex_type, U1, U2):
    """
    Helper function for HDyn process threading at each single frequency point.
    """
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,occ:], optimize='optimal') 
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    W   = VPV + U1
    A   = matEleXiStat(VQ,W,nelec,ex_type)
    A   = diffEps_ov + A.reshape(occ*virt,occ*virt)
    # PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,:occ], optimize='optimal') 
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,:occ,occ:], optimize='optimal') 
    # ijkl: Nocc * Nvirt * Nocc * Nvirt
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,occ:], PV, optimize='optimal')
    W   = VPV + U2
    B   = matEleBStat(VQ,W,nelec,ex_type)
    B   = B.reshape(occ*virt,occ*virt)
    H   = concatAB(A,B)
    return np.einsum('ij,jk,ki->i', effVex_inv, H, effVex)


def HDynDiagApprox(Pi,effVex,VQ,valsMO,nelec,ex_type="singlet",n_jobs=-1):
    """
    Diagonal approximation to the dynamic part of the 
    effective Hamiltonian H2p_Dyn at each frequency point.
    Must be in MO basis already.
    """
    occ  = nelec // 2
    virt = VQ.shape[-1] - occ
    diffEps_ov = mo2ovStat(diffEpsMat(valsMO[0,0,:],nelec)).reshape(occ*virt,occ*virt)
    niw  = Pi.shape[0] 
    niw_half = niw // 2 + 1
    effVex_inv = LA.inv(effVex)
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    U1   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    U2   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,:occ,occ:], optimize='optimal')
    # Diagonal approximation to H2p_Dyn at each frequency point.
    # Parallelize over frequency points
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_process_hdyn_frequency)(
            iw, Pi, VQ, effVex_inv, effVex, occ, virt, diffEps_ov, nelec, ex_type, U1, U2
        ) for iw in range(niw_half)
    )
    
    # Collect results (always symmetric)
    H2p_Dyn = np.zeros((niw,2*occ*virt),dtype=np.complex128)
    for iw, result in enumerate(results):
        H2p_Dyn[iw] = result
    for iw, result in enumerate(results):
        H2p_Dyn[niw - 1 - iw] = result
    
    H2p_Dyn = symmetrizeH2p(H2p_Dyn)
    # return effVals, H2p_Dyn, valsMO
    return H2p_Dyn


def symmetrizeH2p(H2p):
    """
    Symmetrize the H2p matrix across frequency points.
    """
    niw = H2p.shape[0]
    niw_half = niw // 2 + 1
    for iw in range(niw_half):
        H2p[iw] = 0.5 * (H2p[iw] + H2p[niw - 1 - iw].conj())
        H2p[niw - 1 - iw] = H2p[iw].conj()
    return H2p


def _process_hdyn_tda_frequency(iw, Pi, VQ, effVex_inv, effVex, occ, virt, diffEps_ov, nelec, ex_type, U1):
    """
    Helper function for HDyn process threading at each single frequency point.
    Only for TDA approximation, where B block is zero.
    """
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,occ:], optimize='optimal') 
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    W   = VPV + U1
    A   = matEleXiStat(VQ,W,nelec,ex_type)
    A   = diffEps_ov + A.reshape(occ*virt,occ*virt)
    B   = np.zeros((occ*virt,occ*virt),dtype=np.complex128) 
    H   = concatAB(A,B)
    return np.einsum('ij,jk,ki->i', effVex_inv, H, effVex)


def HDynDiagApprox_TDA(Pi,VQ,valsMO,nelec,ex_type="singlet",n_jobs=-1):
    """
    Diagonal approximation to the dynamic part of the 
    effective Hamiltonian H2p_Dyn at each frequency point.
    Must be in MO basis already.
    Only for TDA approximation, where B block is zero.
    """
    diffEps_ov = getDiffEps(valsMO,nelec)
    niw  = Pi.shape[0] 
    niw_half = niw // 2 + 1
    effVals, effVex = solveHstatic(Pi[niw//2,0,:,:,0],VQ,diffEps_ov,nelec,ex_type,tda=1)
    effVex_inv = LA.inv(effVex)
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    U1   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    # Diagonal approximation to H2p_Dyn at each frequency point.
    # Parallelize over frequency points
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_process_hdyn_tda_frequency)(
            iw, Pi, VQ, effVex_inv, effVex, occ, virt, diffEps_ov, nelec, ex_type, U1
        ) for iw in range(niw_half)
    )

    # Collect results (always symmetric)
    H2p_Dyn = np.zeros((niw,2*occ*virt),dtype=np.complex128)
    for iw, result in enumerate(results):
        H2p_Dyn[iw] = result
    for iw, result in enumerate(results):
        H2p_Dyn[niw - 1 - iw] = result

    return effVals, H2p_Dyn


def solveHstatic(Pi_stat,VQ,diffEps_ov,nelec,ex_type="singlet",tda=0):
    """
    Solve the static effective Hamiltonian H_stat for the eigenvalues and eigenvectors.
    The concated H_stat is also returned for book-keeping purposes.
    """
    # Must be in MO basis already.
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    # print("*****    Solving static effective Hamiltonian    *****")
    # Qkl: NQ * Nvirt * Nvirt
    PV  = np.einsum("qp,pkl->qkl", Pi_stat, VQ[0,:,occ:,occ:], optimize='optimal') 
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    U   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    W_stat  = VPV + U
    # iajb: Nocc * Nvirt * Nocc * Nvirt
    A_stat = matEleXiStat(VQ,W_stat,nelec,ex_type)
    
    if not tda:
        print("TDA approximation is disabled. Solving full effective Hamiltonian.")
        # Qkl: NQ * Nocc * Nvirt
        PV  = np.einsum("qp,pkl->qkl", Pi_stat, VQ[0,:,:occ,occ:], optimize='optimal') 
        # ijkl: Nocc * Nvirt * Nocc * Nvirt
        VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,occ:], PV, optimize='optimal')
        # ijkl: Nocc * Nvirt * Nocc * Nvirt
        U   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,:occ,occ:], optimize='optimal')
        W_stat  = VPV + U
        B_stat = matEleBStat(VQ,W_stat,nelec,ex_type)
    else:
        print("TDA approximation is enabled. Using TDA effective Hamiltonian.")
        B_stat = np.zeros((occ,virt,occ,virt),dtype=np.complex128)

    A_stat  = diffEps_ov + A_stat.reshape(occ*virt,occ*virt)
    B_stat  = B_stat.reshape(occ*virt,occ*virt)
    
    H_stat  = concatAB(A_stat,B_stat)
    # check condition number.
    cond = np.linalg.cond(H_stat)
    print(f" Solving non-Hermitian eigenvalue equation. Condition number = {cond:10.4f}")
    
    # balance matrix
    H_stat_balanced, scale = matrix_balance(H_stat)
    effVals,effVex = LA.eig(H_stat_balanced)
    effVex = LA.solve(scale,effVex)
    
    # fix sign ambiguity
    idx = np.argmax(abs(effVex.real), axis=0)
    effVex[:,effVex[idx,np.arange(len(effVals))].real<0] *= -1

    return effVals, effVex, H_stat
