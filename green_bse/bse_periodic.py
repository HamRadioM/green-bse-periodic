#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Periodic BSE solver for crystalline solids.                              #
#    Extends BSESolver (bse.py) to handle Nk k-points.                       #
#                                                                             #
#    Usage: identical to bse.py but calls PeriodicBSESolver instead.         #
#    The input.h5 / sim.h5 files must contain multi-k-point data produced    #
#    by green-mbpt with a periodic mean-field calculation.                   #
#                                                                             #

import time
import numpy as np
import h5py
import scipy
import pyscf

import casidaEq as casida
import casidaEq_periodic as casida_k
import contract as ct
import plasPole
import qp

from bse import BSESolver, BSEConfig

AU2EV = 27.211386245981


class PeriodicBSESolver(BSESolver):
    """
    BSE Casida solver for periodic systems with multiple k-points.

    Inherits from BSESolver and overrides the three methods that depend on
    the single-k-point assumption:
        * load_input_data        -- reads nk from input.h5
        * solve_molecular_orbitals -- applies QP correction at each k
        * prepare_interaction_matrices -- transforms VQ at every k
        * solve_bse_equations    -- uses the k-space Hamiltonian
    Everything else (system monitoring, output, argument parsing) is reused.
    """

    def __init__(self, config: BSEConfig):
        super().__init__(config)
        self.nk = 1          # set in load_input_data

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load_input_data(self):
        """Load k-resolved mean-field and scGW data."""
        start = time.time()

        print("Reading IR file")
        with h5py.File(self.config.ir_file, "r") as f:
            wgrid = f["/bose/ngrid"][()]
        wgrid = 2 * wgrid * np.pi / self.config.beta

        print("Reading input file")
        with h5py.File(self.config.input_file, 'r') as f:
            rSk = f["/HF/S-k"][()].view(complex)
            rSk = rSk.reshape(rSk.shape[:-1])
            rFk_input = f["/HF/Fock-k"][()].view(complex)
            rFk_input = rFk_input.reshape(rFk_input.shape[:-1])
            rHk = f["/HF/H-k"][()].view(complex)
            rHk = rHk.reshape(rHk.shape[:-1])
            self.nao = f["/params/nao"][()]
            self.nelec = f["/params/nel_cell"][()]
            # Determine number of k-points from the Fock matrix shape
            # Shape is (ns, nk, nao, nao)
            self.nk = rFk_input.shape[1]

        self.occ = self.nelec // 2
        self.virt = self.nao - self.occ

        print(f"  Detected {self.nk} k-point(s) in input file.")

        print("Reading sim file")
        with h5py.File(self.config.sim_file, 'r') as f:
            it = f["iter"][()] if self.config.iteration == -1 else self.config.iteration

            if it == 1:
                print("Reading the HF level Fock matrix for G0W0.")
                rFk = rFk_input
            else:
                rFk = f["iter" + str(it) + "/Sigma1"][()].view(complex) + rHk

            rSigmak = f[f"iter{it}/Selfenergy/data"][()].view(complex)
            mu = f[f"iter{it}/mu"][()]

        end = time.time()
        print("*" * 90)
        print(f"    Reading files: {end - start:.4f} secs    ")
        print("*" * 90)

        self.results['rSk'] = rSk
        self.results['rFk'] = rFk
        self.results['rSigmak'] = rSigmak
        self.results['mu'] = mu
        self.results['wgrid'] = wgrid

        return rFk, rSk

    # ------------------------------------------------------------------
    # Molecular orbitals / QP correction
    # ------------------------------------------------------------------

    def solve_molecular_orbitals(self, rFk, rSk):
        """Solve k-resolved eigenvalue problem and apply QP correction at each k."""
        start = time.time()
        print("*" * 90)
        print(f"    Solving for MO eigenvalues at {self.nk} k-point(s)    ")
        print("*" * 90)

        self.valsMO, self.vexMO = casida.solveMO(rFk, rSk)
        # Shape: valsMO (ns, nk, nao),  vexMO (ns, nk, nao, nao)

        mo_coeff = self.vexMO.copy()                        # (ns, nk, nao, nao)
        mo_coeff_adj = mo_coeff.conj().transpose(0, 1, 3, 2)  # C†

        if self.config.qpac_enabled:
            print("Getting QP ac energy levels from self-energy (all k-points).")
            # Σ in MO basis at each k: (niw, ns, nk, nao, nao)
            Sigma_tk_int = np.einsum(
                'skab, tskbc, skcd -> tskad',
                mo_coeff_adj, self.results['rSigmak'], mo_coeff,
                optimize=True
            )
            valsMO_qp = qp.padeSigma(
                Sigma_tk_int, self.valsMO, self.config.beta,
                self.results['mu'], self.config.ir_file
            )
            self._print_qp_summary(valsMO_qp)
            self.valsMO = valsMO_qp
        else:
            print("QUASI-PARTICLE CORRECTION NOT ENABLED.")
            self._print_gap_summary()

        end = time.time()
        print("*" * 90)
        print(f"    Solve MO eigenvalues: {end - start:.4f} secs    ")
        print("*" * 90)

        return mo_coeff_adj, mo_coeff

    def _print_qp_summary(self, valsMO_qp):
        print("\n" + "=" * 90)
        print("QUASI-PARTICLE ENERGY LEVELS (first k-point)")
        print("-" * 90)
        print(f"{'Index':>8} {'Fock/scGW (eV)':>20} {'QP Corrected (eV)':>20} {'Diff (eV)':>15}")
        print("-" * 90)
        n_print = min(20, self.nao)
        for i in range(n_print):
            orig = self.valsMO[0, 0, i] * AU2EV
            qp_e = valsMO_qp[0, 0, i] * AU2EV
            diff = (valsMO_qp[0, 0, i] - self.valsMO[0, 0, i]) * AU2EV
            print(f"{i:8d} {orig:20.6f} {qp_e:20.6f} {diff:15.6f}")
        print("=" * 90 + "\n")

        # Print band gap from QP energies averaged over k-points
        vbm = max(valsMO_qp[0, ik, self.occ - 1] for ik in range(self.nk))
        cbm = min(valsMO_qp[0, ik, self.occ] for ik in range(self.nk))
        gap = (cbm - vbm) * AU2EV
        print("=" * 90)
        print("QUASI-PARTICLE BAND GAP")
        print("-" * 90)
        print(f"    VBM (QP, max over k)  :   {vbm * AU2EV:20.6f} eV")
        print(f"    CBM (QP, min over k)  :   {cbm * AU2EV:20.6f} eV")
        print(f"    Band gap (QP)         :   {gap:20.6f} eV")
        print("=" * 90 + "\n")

    def _print_gap_summary(self):
        vbm = max(self.valsMO[0, ik, self.occ - 1] for ik in range(self.nk)) * AU2EV
        cbm = min(self.valsMO[0, ik, self.occ] for ik in range(self.nk)) * AU2EV
        print("\n" + "=" * 90)
        print("BAND GAP (Kohn-Sham / scGW, no QP correction)")
        print("-" * 90)
        print(f"    VBM (max over k)  :   {vbm:20.6f} eV")
        print(f"    CBM (min over k)  :   {cbm:20.6f} eV")
        print(f"    Band gap          :   {cbm - vbm:20.6f} eV")
        print("=" * 90 + "\n")

    # ------------------------------------------------------------------
    # Interaction matrices
    # ------------------------------------------------------------------

    def prepare_interaction_matrices(self):
        """
        Load and transform the Coulomb integrals and screened polarization.

        For periodic systems, VQ has shape (nk, nQ, nao, nao).  The q=0
        component is loaded from VQ_0.h5 (same filename as molecular).
        The polarization P̃(iΩ) is taken from the q=0 block of the full k-mesh.
        """
        start = time.time()

        print("Reading VQ matrix from integral file.")
        VQ_ao = ct.readVQ(self.config.int_path + "VQ_0.h5")
        # VQ_ao shape: (nk, nQ, nao, nao)
        if VQ_ao.shape[0] != self.nk:
            print(f"  WARNING: VQ has {VQ_ao.shape[0]} k-points but input.h5 has {self.nk}."
                  f"  Proceeding with nk={VQ_ao.shape[0]}.")
            self.nk = VQ_ao.shape[0]

        nQ = VQ_ao.shape[1]
        print(f"  VQ shape: {VQ_ao.shape}  (nk={self.nk}, nQ={nQ})")

        # Transform to MO basis at each k-point
        VQ = casida_k.VQ_ao2mo_k(VQ_ao, self.vexMO)

        # Handle P̃(iΩ) — frequency-dependent polarization
        if self.config.calc_pi_on_fly:
            print(f"Calculating Pi on the fly from iteration {self.config.iter_W} of sim file.")
            tildeP_tau = ct.getPtilde(
                self.config.iter_W, self.nao, nQ,
                tau_h5=self.config.ir_file,
                int_path=self.config.int_path,
                sim_h5=self.config.sim_file
            )
            tildeP_iw = ct.tau2omegaFT(tildeP_tau, beta=self.config.beta,
                                        tau_h5=self.config.ir_file)
            del tildeP_tau
        else:
            tildeP_iw = ct.readPtilde(self.config.pi_file)
            print("! Reading Pi from file; iter_W argument not used. !")

        # tildeP_iw shape may be (niw, ns, nk_q, nQ, nQ) or (niw, ns, nQ, nQ, nk_q)
        # Normalise to a consistent (niw, ns, nQ, nQ, 1) for q=0 extraction
        tildeP_iw = self._extract_q0_polarization(tildeP_iw, nQ)

        niw = tildeP_iw.shape[0]
        print(f"  Processing {niw} imaginary frequency points.")
        print(f"  Problem size: nk={self.nk}, nao={self.nao}, nocc={self.occ}, nvirt={self.virt}")
        print(f"  BSE Hamiltonian dimension: 2 x {self.nk * self.occ * self.virt}")

        end = time.time()
        print(f"    Preparing interaction matrices: {end - start:.4f} secs")

        return VQ, tildeP_iw

    def _extract_q0_polarization(self, tildeP_iw, nQ):
        """
        Extract and return the q=0 block of the polarization as (niw, ns, nQ, nQ, 1).

        Handles the two common storage layouts produced by gwtool.py:
          - (niw, ns, nQ, nQ, nk_q)  -- molecular / single-q layout
          - (niw, ns, nk, nQ, nQ)    -- k-resolved layout; q=0 is k=0
        """
        shape = tildeP_iw.shape
        niw = shape[0]

        if tildeP_iw.ndim == 5 and shape[-1] == 1:
            # Already in (niw, ns, nQ, nQ, 1) — molecular or single-q
            return tildeP_iw

        if tildeP_iw.ndim == 5 and shape[2] == nQ and shape[3] == nQ:
            # (niw, ns, nQ, nQ, nk) -- take q=0 (last index)
            return tildeP_iw[:, :, :, :, :1]

        if tildeP_iw.ndim == 5 and shape[3] == nQ and shape[4] == nQ:
            # (niw, ns, nk, nQ, nQ) -- take k=0 (q=0) slice
            Pi_q0 = tildeP_iw[:, :, 0:1, :, :]           # (niw, ns, 1, nQ, nQ)
            return Pi_q0.transpose(0, 1, 3, 4, 2)         # (niw, ns, nQ, nQ, 1)

        # Fallback: assume (niw, nQ, nQ) or similar and wrap
        print(f"  WARNING: unexpected Pi shape {shape}; attempting reshape.")
        return tildeP_iw.reshape(niw, 1, nQ, nQ, 1)

    # ------------------------------------------------------------------
    # BSE solver
    # ------------------------------------------------------------------

    def solve_bse_equations(self, VQ, tildeP_iw):
        """
        Solve the periodic BSE Casida equation.

        Follows the same flow as BSESolver.solve_bse_equations but uses
        the k-space Hamiltonian functions from casidaEq_periodic.

        Steps
        -----
        1.  Static BSE: diagonalize H_stat = H_eff(iΩ=0) → (effVals, effVex).
        2.  Dynamic correction: evaluate diag(V^{-1} H(iΩ) V) at all Matsubara
            frequencies → H2p_Dyn.
        3.  Build two-particle response function G2p from H2p_Dyn.
        4.  Plasmon-pole fit to extract real excitation energies.
        """
        niw = tildeP_iw.shape[0]

        # Static polarization at Ω=0 (middle of the bosonic Matsubara grid)
        Pi_stat = tildeP_iw[niw // 2, 0, :, :, 0]           # (nQ, nQ)

        # 1. Static BSE — eigenvectors kept in their original (unsorted) order
        #    so that HDynDiagApprox_k and H2p_inf use a consistent V matrix.
        effVals_static, effVex_static, H_stat = casida_k.solveHstatic_k(
            Pi_stat, VQ, self.valsMO, self.nelec,
            ex_type=self.config.excitation_type, tda=False
        )

        # H2p at infinite frequency (bare Coulomb, Pi→0); use same unsorted V
        Pi_inf   = np.zeros_like(Pi_stat)
        _, _, H_inf = casida_k.solveHstatic_k(
            Pi_inf, VQ, self.valsMO, self.nelec,
            ex_type=self.config.excitation_type, tda=False
        )
        effVex_inv = np.linalg.inv(effVex_static)
        H2p_inf    = np.einsum('ij,jk,ki->i', effVex_inv, H_inf, effVex_static).real

        # 2. Dynamic correction — pass unsorted V so columns align with H2p_inf
        H2p_dyn = casida_k.HDynDiagApprox_k(
            tildeP_iw, effVex_static, VQ, self.valsMO, self.nelec,
            ex_type=self.config.excitation_type,
            n_jobs=self.config.n_jobs
        )

        # Sort everything once by the real part of the static eigenvalues.
        # This is the same single-sort pattern used in bse.py.
        idx            = np.argsort(effVals_static.real)
        effVals_static = effVals_static[idx]
        effVex_static  = effVex_static[:, idx]
        H2p_inf        = H2p_inf[idx]
        H2p_dyn        = H2p_dyn[:, idx]

        niw = H2p_dyn.shape[0]

        # 3. Two-particle response function G2p
        G2p_iw  = casida.G2p(H2p_dyn, self.config.ir_file, self.config.beta)
        G2p_tau = ct.omega2tauFT(G2p_iw, self.config.beta, self.config.ir_file)

        # 4. Plasmon-pole fit
        wpole_data, S_data, _, res_norm_data = plasPole.fit_G_update(
            G2p_iw, self.config.ir_file, beta=self.config.beta
        )

        # AO orbital indices (molecular analysis — skip for nk>1 periodic systems)
        if self.nk == 1:
            effVex_occ_ao, effVex_virt_ao = casida.effVex2AO(
                effVex_static, self.vexMO[0, 0, :], self.nelec
            )
        else:
            # For nk>1 the eigenvectors live in the k-space flat-pair basis;
            # AO decomposition requires a k-resolved analysis (not implemented).
            effVex_occ_ao  = np.zeros((0,), dtype=np.complex128)
            effVex_virt_ao = np.zeros((0,), dtype=np.complex128)

        self.results.update({
            'G2p_iw_updated':   G2p_iw,
            'G2p_tau_updated':  G2p_tau,
            'H2p_inf':          H2p_inf,
            'H_stat':           H_stat,
            'effVals_static':   effVals_static,
            'effVex_static':    effVex_static,
            'pole_fit':         wpole_data,
            'S_fit':            S_data,
            'residual_norm_fit': res_norm_data,
            'occ_AO_indices':   effVex_occ_ao,
            'virt_AO_indices':  effVex_virt_ao,
        })

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def print_header(self):
        """Extended header showing periodic-BSE settings."""
        super().print_header()
        print("=" * 90)
        print("PERIODIC BSE SETTINGS")
        print("-" * 90)
        print(f"Number of k-points (from input.h5) : {self.nk}")
        print("BSE at q=0 (optical limit)          : YES")
        print("=" * 90)
