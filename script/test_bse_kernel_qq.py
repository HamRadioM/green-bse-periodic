#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.   #
#                                                                             #
#    Test suite for green_bse/bse_kernel_qq.py                               #
#                                                                             #
#    Two test suites:                                                         #
#      1. Unit tests  — synthetic (NQ=6, niw=11) data, no file I/O           #
#      2. Integration — H2_fq_4k example via PolarizationSolver + BSEKernelQQ#
#                                                                             #
#    Run from the project root:                                               #
#      python script/test_bse_kernel_qq.py                                   #
#      python script/test_bse_kernel_qq.py --skip-integration                #
#                                                                             #

import sys
import os
import argparse
import tempfile

import numpy as np
from scipy.linalg import inv

# Allow importing from green_bse/ when running from the project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BSE_DIR    = os.path.join(_SCRIPT_DIR, '..', 'green_bse')
_ROOT_DIR   = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, _BSE_DIR)

from bse_kernel_qq import (
    _extract_static,
    _squeeze_iw,
    exchange_kernel,
    bse_kernel_iw,
    bse_dyson_iw,
    read_polarization_h5,
    BSEKernelConfig,
    BSEKernelQQ,
)

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_PASS = 0
_FAIL = 0


def _check(name: str, cond: bool, detail: str = ""):
    global _PASS, _FAIL
    if cond:
        print(f"  PASS  {name}")
        _PASS += 1
    else:
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))
        _FAIL += 1


def _allclose(a, b, atol=1e-10, name="", detail=""):
    err = np.max(np.abs(a - b))
    _check(name, err < atol, detail or f"max_err={err:.2e}, atol={atol:.2e}")


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _make_P0_iw(NQ=6, niw=11, ns=1, seed=42):
    """
    Build a synthetic P0_iw array (niw, ns, 1, NQ, NQ) satisfying:
      - P0 is negative semi-definite at each frequency (ε = I−P0 positive definite)
      - Bosonic symmetry: P0(iΩ) = P0(−iΩ)*
    """
    rng = np.random.default_rng(seed)
    P0_iw = np.zeros((niw, ns, 1, NQ, NQ), dtype=np.complex128)

    for iw in range(niw):
        for s in range(ns):
            A = (rng.standard_normal((NQ, NQ)) +
                 1j * rng.standard_normal((NQ, NQ))) * 0.1
            M = -(A @ A.conj().T)            # negative semi-definite
            M = 0.5 * (M + M.conj().T)       # ensure Hermitian
            P0_iw[iw, s, 0] = M

    P0_iw = 0.5 * (P0_iw + P0_iw.conj().transpose(0, 1, 2, 4, 3))
    return P0_iw


def _make_Ptilde_iw(P0_iw):
    """Compute P̃ = (I − P0)^{-1} P0 at each frequency."""
    niw, ns, _, NQ, _ = P0_iw.shape
    Pt = np.zeros_like(P0_iw)
    I  = np.eye(NQ, dtype=np.complex128)
    for iw in range(niw):
        for s in range(ns):
            p0 = P0_iw[iw, s, 0]
            Pt[iw, s, 0] = inv(I - p0) @ p0
    return Pt


def _make_W_iw(P0_iw):
    """
    W = I + P̃ = (I − P0)^{-1}.
    Returns (W_iw, Pt_iw) both (niw, ns, 1, NQ, NQ).
    """
    Pt   = _make_Ptilde_iw(P0_iw)
    NQ   = P0_iw.shape[-1]
    W_iw = Pt.copy()
    I    = np.eye(NQ, dtype=np.complex128)
    W_iw[:, :, 0, :, :] += I[np.newaxis, np.newaxis]
    return W_iw, Pt


# ---------------------------------------------------------------------------
# SUITE 1 – Unit tests for shape helpers
# ---------------------------------------------------------------------------

def test_extract_static():
    print("\n--- test_extract_static ---")
    NQ, niw = 6, 11
    arr = np.arange(niw * NQ * NQ, dtype=np.float64).reshape(niw, 1, 1, NQ, NQ)

    s = _extract_static(arr)
    _check("shape is (NQ, NQ)",      s.shape == (NQ, NQ))
    _check("selects iw = niw//2",    np.allclose(s, arr[niw // 2, 0, 0]))

    # Alternative layout (niw, ns, NQ, NQ, 1)
    arr2 = arr.reshape(niw, 1, NQ, NQ, 1)
    s2   = _extract_static(arr2)
    _check("alt layout (niw,ns,NQ,NQ,1)", np.allclose(s2, arr[niw // 2, 0, 0]))


def test_squeeze_iw():
    print("\n--- test_squeeze_iw ---")
    NQ, niw = 6, 11
    arr = np.zeros((niw, 1, 1, NQ, NQ), dtype=np.complex128)
    arr[:, 0, 0] = np.eye(NQ)[np.newaxis]    # identity at each frequency

    out = _squeeze_iw(arr)
    _check("shape (niw, NQ, NQ)",  out.shape == (niw, NQ, NQ))
    _check("values preserved",     np.allclose(out, np.eye(NQ)[np.newaxis]))


# ---------------------------------------------------------------------------
# SUITE 1b – exchange_kernel
# ---------------------------------------------------------------------------

def test_exchange_kernel_singlet():
    print("\n--- test_exchange_kernel  (κ=1 singlet) ---")
    NQ = 8
    U  = exchange_kernel(NQ, kappa=1.0)
    _check("shape (NQ, NQ)",       U.shape == (NQ, NQ))
    _check("dtype complex128",     U.dtype == np.complex128)
    _allclose(U, np.eye(NQ, dtype=np.complex128), atol=1e-15, name="κ=1 → U = I")


def test_exchange_kernel_triplet():
    print("\n--- test_exchange_kernel  (κ=0 triplet) ---")
    NQ = 8
    U  = exchange_kernel(NQ, kappa=0.0)
    _check("shape (NQ, NQ)",       U.shape == (NQ, NQ))
    _check("all zero",             np.allclose(U, 0.0))


def test_exchange_kernel_arbitrary():
    print("\n--- test_exchange_kernel  (κ=2.5) ---")
    NQ = 5
    kappa = 2.5
    U  = exchange_kernel(NQ, kappa=kappa)
    _allclose(U, kappa * np.eye(NQ, dtype=np.complex128), atol=1e-15,
              name="κ=2.5 → U = 2.5 I")


# ---------------------------------------------------------------------------
# SUITE 1c – bse_kernel_iw: Ξ(iΩ) = 2U − W(iΩ)
# ---------------------------------------------------------------------------

def test_bse_kernel_iw_shape():
    print("\n--- test_bse_kernel_iw  (shape) ---")
    NQ, niw = 6, 11
    P0_iw = _make_P0_iw(NQ=NQ, niw=niw)
    W_iw, _ = _make_W_iw(P0_iw)

    Xi_iw = bse_kernel_iw(W_iw, kappa=1.0)
    _check("shape (niw, NQ, NQ)",  Xi_iw.shape == (niw, NQ, NQ))
    _check("dtype complex128",     Xi_iw.dtype == np.complex128)


def test_bse_kernel_iw_formula():
    """Ξ(iΩ) = 2κI − W(iΩ)  for both kappa=1 and kappa=0."""
    print("\n--- test_bse_kernel_iw  (formula 2U−W) ---")
    NQ, niw = 6, 11
    P0_iw = _make_P0_iw(NQ=NQ, niw=niw)
    W_iw, _ = _make_W_iw(P0_iw)
    W_all = _squeeze_iw(W_iw)              # (niw, NQ, NQ)
    I_NQ  = np.eye(NQ, dtype=np.complex128)

    # κ=1 (singlet)
    Xi1 = bse_kernel_iw(W_iw, kappa=1.0)
    _allclose(Xi1, 2.0 * I_NQ[np.newaxis] - W_all, atol=1e-12,
              name="κ=1: Ξ = 2I − W")

    # κ=0 (triplet, exchange dropped)
    Xi0 = bse_kernel_iw(W_iw, kappa=0.0)
    _allclose(Xi0, -W_all, atol=1e-12,
              name="κ=0: Ξ = −W")

    # 5D input vs. pre-squeezed 3D input should give same result
    Xi1_3d = bse_kernel_iw(W_all, kappa=1.0)
    _allclose(Xi1, Xi1_3d, atol=1e-15, name="5D and 3D inputs agree")


def test_bse_kernel_iw_varies_with_frequency():
    """Ξ(iΩ) should vary with frequency because W(iΩ) varies."""
    print("\n--- test_bse_kernel_iw  (frequency variation) ---")
    NQ, niw = 6, 11
    P0_iw = _make_P0_iw(NQ=NQ, niw=niw)
    W_iw, _ = _make_W_iw(P0_iw)
    Xi_iw = bse_kernel_iw(W_iw, kappa=1.0)

    norms = np.array([np.linalg.norm(Xi_iw[iw]) for iw in range(niw)])
    _check("Ξ(iΩ) varies across frequencies (std > 0)",
           np.std(norms) > 0,
           f"std(||Ξ||_F) = {np.std(norms):.2e}")


# ---------------------------------------------------------------------------
# SUITE 1d – bse_dyson_iw: Dyson equation
# ---------------------------------------------------------------------------

def test_bse_dyson_iw_identity():
    """Dyson: (I − P0(iΩ) Ξ(iΩ)) P_BSE(iΩ) = P0(iΩ) at every frequency."""
    print("\n--- test_bse_dyson_iw  (Dyson identity) ---")
    NQ, niw = 6, 11
    P0_iw = _make_P0_iw(NQ=NQ, niw=niw)
    W_iw, _ = _make_W_iw(P0_iw)
    Xi_iw = bse_kernel_iw(W_iw, kappa=1.0)   # (niw, NQ, NQ)
    P_bse = bse_dyson_iw(P0_iw, Xi_iw)        # (niw, NQ, NQ)

    _check("P_bse shape (niw, NQ, NQ)",  P_bse.shape == (niw, NQ, NQ))

    I_NQ = np.eye(NQ, dtype=np.complex128)
    max_err = 0.0
    for iw in range(niw):
        P0  = _squeeze_iw(P0_iw)[iw]   # (NQ, NQ)
        Xi  = Xi_iw[iw]
        Pb  = P_bse[iw]
        lhs = (I_NQ - P0 @ Xi) @ Pb
        max_err = max(max_err, np.max(np.abs(lhs - P0)))
    _check(f"(I − P0·Ξ)·P_BSE = P0  (max_err={max_err:.2e})",
           max_err < 1e-10)


def test_bse_dyson_iw_rpa_limit():
    """With Ξ(iΩ) = I (RPA kernel), P_BSE must recover P̃ = (I−P0)^{-1} P0."""
    print("\n--- test_bse_dyson_iw  (RPA limit: Ξ=I → P_BSE = P̃) ---")
    NQ, niw = 6, 11
    P0_iw  = _make_P0_iw(NQ=NQ, niw=niw)
    Pt_iw  = _make_Ptilde_iw(P0_iw)              # (niw, ns, 1, NQ, NQ)

    Xi_identity = np.tile(np.eye(NQ, dtype=np.complex128), (niw, 1, 1))  # (niw, NQ, NQ)
    P_bse = bse_dyson_iw(P0_iw, Xi_identity)     # (niw, NQ, NQ)
    Pt_sq = _squeeze_iw(Pt_iw)                   # (niw, NQ, NQ)

    err = np.max(np.abs(P_bse - Pt_sq))
    _check(f"P_BSE(Ξ=I) = P̃  (max_err={err:.2e})", err < 1e-10)


def test_bse_dyson_iw_triplet():
    """With κ=0 (triplet), Ξ = −W; Dyson identity must still hold."""
    print("\n--- test_bse_dyson_iw  (κ=0 triplet, Ξ = −W) ---")
    NQ, niw = 6, 11
    P0_iw = _make_P0_iw(NQ=NQ, niw=niw)
    W_iw, _ = _make_W_iw(P0_iw)
    Xi_iw = bse_kernel_iw(W_iw, kappa=0.0)
    P_bse = bse_dyson_iw(P0_iw, Xi_iw)

    I_NQ = np.eye(NQ, dtype=np.complex128)
    max_err = 0.0
    for iw in range(niw):
        P0  = _squeeze_iw(P0_iw)[iw]
        Pb  = P_bse[iw]
        lhs = (I_NQ - P0 @ Xi_iw[iw]) @ Pb
        max_err = max(max_err, np.max(np.abs(lhs - P0)))
    _check(f"Dyson identity κ=0  (max_err={max_err:.2e})", max_err < 1e-10)


def test_bse_dyson_iw_5d_input():
    """bse_dyson_iw accepts both 5D (ns, q-slot) and 3D (squeezed) P0."""
    print("\n--- test_bse_dyson_iw  (5D and 3D P0_iw inputs match) ---")
    NQ, niw = 6, 11
    P0_iw = _make_P0_iw(NQ=NQ, niw=niw)
    W_iw, _ = _make_W_iw(P0_iw)
    Xi_iw = bse_kernel_iw(W_iw, kappa=1.0)

    P_bse_5d = bse_dyson_iw(P0_iw, Xi_iw)                    # input is 5D
    P_bse_3d = bse_dyson_iw(_squeeze_iw(P0_iw), Xi_iw)       # input is 3D
    _allclose(P_bse_5d, P_bse_3d, atol=1e-14, name="5D and 3D inputs give same result")


# ---------------------------------------------------------------------------
# SUITE 2 – Integration test using real H2_fq_4k data
# ---------------------------------------------------------------------------

def run_integration_test(example_dir):
    """
    Full pipeline on the H2_fq_4k example at q_idx=1.

    Steps
    -----
    1. Run PolarizationSolver → pol_q1.h5 with P0_iw, P̃_iw, W_iw.
    2. Run BSEKernelQQ(pol_files=[pol_out]) → xi_qq_q1.h5.
    3. Check shapes, the Dyson identity, and HDF5 output keys.
    """
    from polarization import PolarizationConfig, PolarizationSolver

    print("\n" + "=" * 60)
    print("INTEGRATION TEST  (H2_fq_4k, q_idx=1)")
    print("=" * 60)

    input_h5  = os.path.join(example_dir, 'mean_field_input.h5')
    sim_h5    = os.path.join(example_dir, 'sim.h5')
    int_path  = os.path.join(example_dir, 'df_hf_int_fq')
    ir_file   = os.path.join(example_dir, 'irgrid', '1e5.h5')

    for path in [input_h5, sim_h5, int_path, ir_file]:
        if not os.path.exists(path):
            print(f"  SKIP  missing required file/dir: {path}")
            return

    with tempfile.TemporaryDirectory() as tmpdir:
        pol_out = os.path.join(tmpdir, 'pol_q1.h5')
        xi_out  = os.path.join(tmpdir, 'xi_qq_q1.h5')

        # --- Step 1: run PolarizationSolver ---
        print("\n[Step 1] Running PolarizationSolver (q_idx=1) …")
        pol_cfg = PolarizationConfig(
            input_file  = input_h5,
            sim_file    = sim_h5,
            int_path    = int_path,
            ir_file     = ir_file,
            beta        = 1000.0,
            q_idx       = 1,
            iteration   = -1,
            output_file = pol_out,
        )
        PolarizationSolver(pol_cfg).run()

        pol   = read_polarization_h5(pol_out)
        P0_iw = pol['P0_iw']    # (niw, ns, 1, NQ, NQ)
        W_iw  = pol['W_iw']
        NQ    = pol['NQ']
        niw   = P0_iw.shape[0]

        _check("P0_iw has correct ndim=5",  P0_iw.ndim == 5)
        _check("W_iw  has correct ndim=5",  W_iw.ndim  == 5)
        _check("P0_iw/W_iw same shape",     P0_iw.shape == W_iw.shape)

        # --- Step 2: run BSEKernelQQ ---
        print("\n[Step 2] Running BSEKernelQQ (Nq=1) …")
        xi_cfg = BSEKernelConfig(
            pol_files   = [pol_out],
            output_file = xi_out,
            kappa       = 1.0,    # singlet
        )
        results = BSEKernelQQ(xi_cfg).run()

        # New output shapes: (Nq, niw, NQ, NQ) with Nq=1
        Xi_all  = results['Xi_iw']    # (1, niw, NQ, NQ)
        Pbse_all = results['P_bse']   # (1, niw, NQ, NQ)
        Xi_iw   = Xi_all[0]           # (niw, NQ, NQ)
        P_bse   = Pbse_all[0]         # (niw, NQ, NQ)

        _check("Xi_iw shape (Nq=1, niw, NQ, NQ)",   Xi_all.shape  == (1, niw, NQ, NQ))
        _check("P_bse shape (Nq=1, niw, NQ, NQ)",   Pbse_all.shape == (1, niw, NQ, NQ))

        # --- Step 3: Dyson identity (I − P0(iΩ) Ξ(iΩ)) P_BSE(iΩ) = P0(iΩ) ---
        I_NQ  = np.eye(NQ, dtype=np.complex128)
        P0_sq = _squeeze_iw(P0_iw)   # (niw, NQ, NQ)
        max_err = 0.0
        for iw in range(niw):
            P0  = P0_sq[iw]
            Xi  = Xi_iw[iw]
            Pb  = P_bse[iw]
            max_err = max(max_err, np.max(np.abs((I_NQ - P0 @ Xi) @ Pb - P0)))
        _check(f"Dyson identity at all iΩ  (max_err={max_err:.2e})",
               max_err < 1e-8)

        # --- Step 4: Ξ(iΩ) = 2I − W(iΩ) at κ=1 ---
        W_sq = _squeeze_iw(W_iw)    # (niw, NQ, NQ)
        _allclose(Xi_iw, 2.0 * I_NQ[np.newaxis] - W_sq, atol=1e-10,
                  name="Ξ(iΩ) = 2I − W(iΩ)  (κ=1, singlet)")

        # --- Step 5: HDF5 output keys ---
        import h5py
        with h5py.File(xi_out, 'r') as f:
            _check("/Xi_iw in HDF5",     'Xi_iw' in f)
            _check("/P_bse in HDF5",     'P_bse' in f)
            _check("/P0_iw in HDF5",     'P0_iw' in f)
            _check("/Nq in HDF5",        'Nq' in f)
            _check("/NQ in HDF5",        'NQ' in f)
            stored_xi = f['Xi_iw'][()].astype(np.complex128)   # (1, niw, NQ, NQ)
        _allclose(stored_xi[0], Xi_iw, atol=1e-15,
                  name="HDF5 Xi_iw[0] matches in-memory Xi_iw")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_unit_tests():
    print("=" * 60)
    print("UNIT TESTS  (synthetic data)")
    print("=" * 60)
    # Shape helpers
    test_extract_static()
    test_squeeze_iw()
    # exchange_kernel
    test_exchange_kernel_singlet()
    test_exchange_kernel_triplet()
    test_exchange_kernel_arbitrary()
    # bse_kernel_iw
    test_bse_kernel_iw_shape()
    test_bse_kernel_iw_formula()
    test_bse_kernel_iw_varies_with_frequency()
    # bse_dyson_iw
    test_bse_dyson_iw_identity()
    test_bse_dyson_iw_rpa_limit()
    test_bse_dyson_iw_triplet()
    test_bse_dyson_iw_5d_input()


def main():
    parser = argparse.ArgumentParser(
        description='Test suite for bse_kernel_qq.py'
    )
    parser.add_argument(
        '--skip-integration', action='store_true',
        help='Skip the real-data integration test (faster).'
    )
    parser.add_argument(
        '--example-dir', default=None,
        help='Path to the H2_fq_4k example directory. '
             'Defaults to example/H2_fq_4k relative to project root.'
    )
    args = parser.parse_args()

    run_unit_tests()

    if not args.skip_integration:
        example_dir = args.example_dir or os.path.join(_ROOT_DIR, 'example', 'H2_fq_4k')
        run_integration_test(example_dir)

    print("\n" + "=" * 60)
    print(f"Results:  {_PASS} passed,  {_FAIL} failed")
    print("=" * 60)
    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == '__main__':
    main()
