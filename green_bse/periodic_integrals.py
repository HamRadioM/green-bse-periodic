#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Reconstruct per-k DF integrals  V^Q_{ij}(k1,k2)  from the green-mbpt      #
#    irreducible-k-pair storage.                                              #
#                                                                             #
#    green-mbpt stores the density-fitting integrals only for the irreducible #
#    k-pairs (reduced by time-reversal even when space symmetry is disabled,  #
#    --space_symm False).  This module rebuilds V^Q(k1,k2) for an arbitrary   #
#    (k1,k2) using the same logic as the C++ reference df_integral_t::         #
#    {v_type,wrap,symmetrize}, and assembles the same-k and off-diagonal       #
#    per-k tensors the periodic two-step BSE solver consumes.                 #
#                                                                             #
#    Stored layout (df_hf_int/VQ_<c>.h5, key "/<c>")                          #
#    -----------------------------------------------                          #
#      raw  : (num_kpair_stored, NQ, nao, 2*nao) float64                      #
#      complex view: raw.view(complex128) -> (num_kpair_stored, NQ, nao, nao) #
#    indexed by position in symmetry/pairs/kpair_irre_list.  Chunking         #
#    (meta.h5/chunk_indices) splits the pair axis; here we concatenate the    #
#    chunks back into a single (num_kpair_stored, NQ, nao, nao) array.        #
#                                                                             #
#    Reconstruction for (k1,k2)  (mirrors df_integral_t)                      #
#    --------------------------------------------------                       #
#      idx0 = k1>=k2 ? k1(k1+1)/2+k2 : k2(k2+1)/2+k1   (lower-triangle index) #
#      sign = +1 if k1>=k2 else -1                                            #
#      if   conj_pairs[idx0]  != idx0 : ctype=conj , src = conj_pairs[idx0]  #
#      elif trans_pairs[idx0] != idx0 : ctype=trans, src = trans_pairs[idx0] #
#      else                           : ctype=direct, src = idx0             #
#      row  = position of src in kpair_irre_list                             #
#      V    = stored[row]                          (NQ, nao, nao)            #
#      if sign<0 : V[Q] <- V[Q].conj().T           (per Q)                   #
#      if conj   : V[Q] <- V[Q].conj()                                       #
#      elif trans: V[Q] <- V[Q].T                                            #
#                                                                             #

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# IBZ -> full-BZ unfolding  (matches green_mbtools.pesto.mb.to_full_bz)
# ---------------------------------------------------------------------------
def to_full_bz(X, conj_list, ibz2bz, bz2ibz, k_ind, k_sym_trans):
    """
    Unfold a k-resolved quantity stored on the irreducible BZ to the full BZ.

        X_full(k) = U_k X_irr(rep(k)) U_k^dagger        (k_ind = 1 or 2)
        X_full(k) = conj(...)  if  tr_conj[k] != 0

    k_ind is the position of the k-axis in X:
        0 : X[k, ...]                 (no AO rotation, e.g. eigenvalues)
        1 : X[s, k, ...]              (e.g. Sigma1, Fock)
        2 : X[t, s, k, ...]           (e.g. G_tau, Selfenergy)

    Port of green_mbtools.pesto.mb.to_full_bz (kept local so green_bse stays
    self-contained, as it already does for the IR routines).
    """
    index_loc_in_ibz = np.zeros(bz2ibz.shape, dtype=int) - 1
    for i, irn in enumerate(ibz2bz):
        index_loc_in_ibz[irn] = i
    new_shape = list(X.shape)
    new_shape[k_ind] = conj_list.shape[0]
    Y = np.zeros(new_shape, dtype=X.dtype)
    for ik, kk in enumerate(bz2ibz):
        k = index_loc_in_ibz[kk]
        Uk = k_sym_trans[ik]
        Uk_dag = Uk.conj().T
        if k_ind == 0:
            Y[ik, ...] = X[k, ...].conj() if conj_list[ik] else X[k, ...]
        elif k_ind == 1:
            Xk = X[:, k, ...].conj() if conj_list[ik] else X[:, k, ...]
            Y[:, ik, ...] = np.einsum('ab,sbc,cd->sad', Uk, Xk, Uk_dag)
        elif k_ind == 2:
            Xk = X[:, :, k, ...].conj() if conj_list[ik] else X[:, :, k, ...]
            Y[:, :, ik, ...] = np.einsum('ab,tsbc,cd->tsad', Uk, Xk, Uk_dag)
        else:
            raise ValueError("k_ind must be 0, 1 or 2")
    return Y


def build_kpoint_maps(input_h5="mean_field_input.h5"):
    """
    Build exact k-vector arithmetic index maps on the BvK mesh.

    Using integer grid coordinates  g = round(mesh_scaled * nk_list) mod nk_list
    (exact, no floating-point modulo), returns

        ksum[a,b]  = index of  k_a + k_b   (mod reciprocal lattice)
        kdiff[a,b] = index of  k_a - k_b

    NOTE: the naive cyclic map (a+b) % nk used elsewhere is only correct for a
    1-D mesh; for a 2-D/3-D mesh it is wrong (here 540/1296 entries differ).
    Use ksum/kdiff for the exciton momentum (k+Q_exc) and the direct-kernel
    momentum transfer (k-k').

    Returns
    -------
    dict(ksum=(nk,nk) int, kdiff=(nk,nk) int, grid=(nk,3) int, dims=(3,) int)
    """
    with h5py.File(input_h5, "r") as f:
        ks = f["symmetry/k/mesh_scaled"][()]
        dims = f["symmetry/k/nk_list"][()].astype(int)
    nk = ks.shape[0]
    g = (np.rint(ks * dims).astype(int)) % dims
    key = {tuple(g[i]): i for i in range(nk)}
    ksum = np.empty((nk, nk), dtype=int)
    kdiff = np.empty((nk, nk), dtype=int)
    for a in range(nk):
        for b in range(nk):
            ksum[a, b] = key[tuple((g[a] + g[b]) % dims)]
            kdiff[a, b] = key[tuple((g[a] - g[b]) % dims)]
    return dict(ksum=ksum, kdiff=kdiff, grid=g, dims=dims)


def load_k_symmetry(input_h5="mean_field_input.h5"):
    """Read the IBZ<->BZ unfolding maps from the mean-field input."""
    with h5py.File(input_h5, "r") as f:
        return dict(
            ibz2bz=f["symmetry/k/ibz2bz"][()].astype(int),
            bz2ibz=f["symmetry/k/bz2ibz"][()].astype(int),
            tr_conj=f["symmetry/k/tr_conj"][()].astype(int),
            U=f["symmetry/k/k_sym_transform_ao"][()],
            ink=int(f["symmetry/k/ink"][()]),
            nk=int(f["symmetry/k/nk"][()]),
        )


def unfold_to_bz(X, ksym, k_ind):
    """Convenience wrapper around to_full_bz using the dict from load_k_symmetry."""
    return to_full_bz(X, ksym["tr_conj"], ksym["ibz2bz"], ksym["bz2ibz"],
                      k_ind, ksym["U"])


# ---------------------------------------------------------------------------
# Stored integrals + pair symmetry
# ---------------------------------------------------------------------------
def load_stored_integrals(int_path="df_hf_int/"):
    """
    Concatenate all VQ chunks into a single (num_kpair_stored, NQ, nao, nao)
    complex array, ordered by position in kpair_irre_list.
    """
    with h5py.File(int_path + "meta.h5", "r") as f:
        chunk_indices = np.asarray(f["/chunk_indices"][()]).ravel()
    blocks = []
    for c in chunk_indices:
        c = int(c)
        with h5py.File(int_path + "VQ_%d.h5" % c, "r") as f:
            raw = f["/%d" % c][()]                       # (chunk, NQ, nao, 2*nao)
        blocks.append(raw.view(np.complex128))           # (chunk, NQ, nao, nao)
    return np.concatenate(blocks, axis=0)


def load_kpair_symmetry(input_h5="mean_field_input.h5"):
    """
    Read the k-pair symmetry maps and build the inverse (pair -> stored-row) map.

    Returns
    -------
    sym : dict with keys
        conj      : (npairs,) int   conj_pairs_list
        trans     : (npairs,) int   trans_pairs_list
        pos_in_irre : (npairs,) int  stored-row index of each pair (-1 if not stored)
        nk        : int
    """
    with h5py.File(input_h5, "r") as f:
        conj = f["symmetry/pairs/conj_pairs_list"][()].astype(int)
        trans = f["symmetry/pairs/trans_pairs_list"][()].astype(int)
        irre = f["symmetry/pairs/kpair_irre_list"][()].astype(int)
        nk = int(f["symmetry/k/nk"][()])
    pos_in_irre = -np.ones(conj.shape[0], dtype=int)
    pos_in_irre[irre] = np.arange(irre.shape[0])
    return dict(conj=conj, trans=trans, pos_in_irre=pos_in_irre, nk=nk)


# ---------------------------------------------------------------------------
# Single-pair reconstruction
# ---------------------------------------------------------------------------
def reconstruct_VQ_pair(stored, sym, k1, k2):
    """
    Reconstruct V^Q_{ij}(k1,k2)  (NQ, nao, nao) from the stored irreducible pairs.
    """
    conj, trans, pos = sym["conj"], sym["trans"], sym["pos_in_irre"]
    idx0 = k1 * (k1 + 1) // 2 + k2 if k1 >= k2 else k2 * (k2 + 1) // 2 + k1
    sign = 1 if k1 >= k2 else -1

    if conj[idx0] != idx0:
        ctype, src = "conj", conj[idx0]
    elif trans[idx0] != idx0:
        ctype, src = "trans", trans[idx0]
    else:
        ctype, src = "direct", idx0

    row = pos[src]
    if row < 0:
        raise ValueError("pair (%d,%d) -> src %d not in stored set" % (k1, k2, src))
    V = stored[row].copy()                               # (NQ, nao, nao)

    if sign < 0:
        V = V.conj().transpose(0, 2, 1)
    if ctype == "conj":
        V = V.conj()
    elif ctype == "trans":
        V = V.transpose(0, 2, 1)
    return V


# ---------------------------------------------------------------------------
# Per-k assembled tensors
# ---------------------------------------------------------------------------
def build_VQ_same_k(stored, sym):
    """V^Q(k,k) for all k -> (nk, NQ, nao, nao)."""
    nk = sym["nk"]
    NQ, nao = stored.shape[1], stored.shape[2]
    out = np.empty((nk, NQ, nao, nao), dtype=np.complex128)
    for k in range(nk):
        out[k] = reconstruct_VQ_pair(stored, sym, k, k)
    return out


def build_VQ_offdiag(stored, sym, kq_map):
    """V^Q(k, kq_map[k]) for all k -> (nk, NQ, nao, nao)."""
    nk = sym["nk"]
    NQ, nao = stored.shape[1], stored.shape[2]
    out = np.empty((nk, NQ, nao, nao), dtype=np.complex128)
    for k in range(nk):
        out[k] = reconstruct_VQ_pair(stored, sym, k, int(kq_map[k]))
    return out


def build_VQ_mo_blocks(stored, sym, SC, occ):
    """
    MO-basis occ-occ and virt-virt DF blocks for ALL k-pairs (K, K').

    For the q-resolved BSE direct kernel W^{k-k'} we need V in the MO basis at
    arbitrary k-pairs, not just same-k.  Here

        Voo[K,K',Q,i,j] = (S_K C_K)^dagger[:,i]  V_Q(K,K')  (S_{K'} C_{K'})[:,j]   (i,j occ)
        Vvv[K,K',Q,b,a] = (S_K C_K)^dagger[:,b]  V_Q(K,K')  (S_{K'} C_{K'})[:,a]   (b,a virt)

    Parameters
    ----------
    stored : (num_kpair_stored, NQ, nao, nao)  from load_stored_integrals.
    sym    : dict from load_kpair_symmetry.
    SC     : (nk, nao, nmo)  the metric-dressed MO coefficients S_k C_k.
    occ    : int

    Returns
    -------
    Voo : (nk, nk, NQ, occ, occ)        complex128
    Vvv : (nk, nk, NQ, virt, virt)      complex128,  virt = nao - occ

    Memory scales as nk^2 * NQ * (occ^2 + virt^2); fine for small cells, but
    consider on-the-fly reconstruction for large k-meshes.
    """
    nk = sym["nk"]
    NQ, nao = stored.shape[1], stored.shape[2]
    virt = nao - occ
    Voo = np.empty((nk, nk, NQ, occ, occ), dtype=np.complex128)
    Vvv = np.empty((nk, nk, NQ, virt, virt), dtype=np.complex128)
    for k in range(nk):
        SCo_k, SCv_k = SC[k][:, :occ].conj(), SC[k][:, occ:].conj()
        for kp in range(nk):
            V = reconstruct_VQ_pair(stored, sym, k, kp)         # (NQ, nao, nao)
            SCo_kp, SCv_kp = SC[kp][:, :occ], SC[kp][:, occ:]
            Voo[k, kp] = np.einsum('mi,Qmn,nj->Qij', SCo_k, V, SCo_kp, optimize=True)
            Vvv[k, kp] = np.einsum('mb,Qmn,na->Qba', SCv_k, V, SCv_kp, optimize=True)
    return Voo, Vvv


def load_per_k_integrals(int_path="df_hf_int/", input_h5="mean_field_input.h5",
                         kq_map=None):
    """
    Convenience: load stored integrals + symmetry and return the same-k tensor
    (and, if kq_map is given, the off-diagonal tensor).

    Returns
    -------
    VQ_kk : (nk, NQ, nao, nao)
    VQ_kq : (nk, NQ, nao, nao)  only if kq_map is not None
    """
    stored = load_stored_integrals(int_path)
    sym = load_kpair_symmetry(input_h5)
    VQ_kk = build_VQ_same_k(stored, sym)
    if kq_map is None:
        return VQ_kk
    return VQ_kk, build_VQ_offdiag(stored, sym, kq_map)
