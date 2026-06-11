#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Finite-q periodic BSE solver for crystalline solids.                     #
#    Fork of bse_periodic.py that uses proper off-diagonal GDF integrals      #
#    ⟨φ_μ(k)|V_Q|φ_ν(k+q)⟩ for a fixed momentum transfer q.                 #
#                                                                             #
#    Usage:                                                                   #
#      solver = FiniteQBSESolver(config, q_idx=1)                             #
#      solver.run()                                                            #
#                                                                             #
#    q_idx : int  — index of q in the k-mesh.                                 #
#      q_idx = 0 → q = 0 (optical limit, equivalent to PeriodicBSESolver)    #
#      q_idx > 0 → finite momentum transfer q = kpts[q_idx]                  #
#                                                                             #
#    kq_map[k] = (k + q_idx) % nk  (cyclic shift on the BvK k-mesh)         #
#                                                                             #
#    Two VQ files are required:                                                #
#      int_path/VQ_0.h5          — same-k integrals (standard GDF)            #
#      int_path/VQ_q{q_idx}.h5  — off-diagonal (k, k+q) integrals            #
#    For q_idx=0 the same file is used for both.                              #
#                                                                             #

import time
import numpy as np
import h5py

import casidaEq as casida
import casidaEq_finite_q as casida_fq
import contract as ct
import plasPole

from bse import BSESolver, BSEConfig

AU2EV = 27.211386245981


class FiniteQBSESolver(BSESolver):
    """
    BSE Casida solver for finite momentum transfer q in periodic systems.

    Inherits all I/O, MO-solving, and output machinery from BSESolver and
    overrides the three k-dependent stages:
        * prepare_interaction_matrices — loads VQ_kk and VQ_kq, computes Pi
        * solve_bse_equations          — uses the finite-q Hamiltonian
        * run                          — wires everything together with kq_map

    Physics
    -------
    At finite q (q_idx > 0) the long-range Coulomb exchange is absent (κ=0).
    At q=0 (q_idx == 0) the standard optical-limit result is recovered.

    The proper off-diagonal integrals ⟨φ_μ(k)|V_Q|φ_ν(k+q)⟩ are read from
    VQ_q{q_idx}.h5.  These differ from the approximate same-k integrals used
    in PeriodicBSESolver and must be pre-generated with extract_VQ_kq in
    generate_finite_q_test.py (or equivalent GDF extraction code).
    """

    def __init__(self, config: BSEConfig, q_idx: int = 0):
        """
        Parameters
        ----------
        config : BSEConfig
        q_idx  : int
            Index of q-vector in the BvK k-mesh (0 = optical/Gamma limit).
        """
        # Initialise nk to 1 (updated in load_input_data)
        self.nk     = 1
        self.q_idx  = q_idx
        # kq_map will be properly set in run() after nk is known
        self.kq_map = np.arange(1, dtype=int)
        super().__init__(config)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def print_header(self):
        """Extended header showing finite-q BSE settings."""
        super().print_header()
        q_label = "q=0 (optical limit)" if self.q_idx == 0 else f"q_idx={self.q_idx}"
        print("=" * 90)
        print("FINITE-q PERIODIC BSE SETTINGS")
        print("-" * 90)
        print(f"  q index (q_idx)    : {self.q_idx}")
        print(f"  Momentum transfer  : {q_label}")
        print(f"  VQ_kk file         : {self.config.int_path}VQ_0.h5")
        print(f"  VQ_kq file         : {self.config.int_path}VQ_q{self.q_idx}.h5")
        if self.q_idx == 0:
            print("  Exchange kappa     : 2 (singlet) / 0 (triplet)  [q=0 limit]")
        else:
            print("  Exchange kappa     : 0  [finite q, long-range exchange vanishes]")
        print("=" * 90)

    # ------------------------------------------------------------------
    # I/O — delegate entirely to the base class (same file format)
    # ------------------------------------------------------------------

    def load_input_data(self):
        """Load k-resolved mean-field and scGW data (identical to PeriodicBSESolver)."""
        start = time.time()

        print("Reading IR file")
        with h5py.File(self.config.ir_file, "r") as f:
            wgrid = f["/bose/ngrid"][()]
        wgrid = 2 * wgrid * np.pi / self.config.beta

        print("Reading input file")
        with h5py.File(self.config.input_file, 'r') as f:
            rSk       = f["/HF/S-k"][()].view(complex)
            rSk       = rSk.reshape(rSk.shape[:-1])
            rFk_input = f["/HF/Fock-k"][()].view(complex)
            rFk_input = rFk_input.reshape(rFk_input.shape[:-1])
            rHk       = f["/HF/H-k"][()].view(complex)
            rHk       = rHk.reshape(rHk.shape[:-1])
            self.nao  = f["/params/nao"][()]
            self.nelec = f["/params/nel_cell"][()]
            # Shape is (ns, nk, nao, nao)
            self.nk   = rFk_input.shape[1]

        self.occ  = self.nelec // 2
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
            mu      = f[f"iter{it}/mu"][()]

        end = time.time()
        print("*" * 90)
        print(f"    Reading files: {end - start:.4f} secs    ")
        print("*" * 90)

        self.results['rSk']      = rSk
        self.results['rFk']      = rFk
        self.results['rSigmak']  = rSigmak
        self.results['mu']       = mu
        self.results['wgrid']    = wgrid

        return rFk, rSk

    # ------------------------------------------------------------------
    # MO solver — identical to PeriodicBSESolver
    # ------------------------------------------------------------------

    def solve_molecular_orbitals(self, rFk, rSk):
        """Solve k-resolved eigenvalue problem and apply QP correction at each k."""
        import qp
        start = time.time()
        print("*" * 90)
        print(f"    Solving for MO eigenvalues at {self.nk} k-point(s)    ")
        print("*" * 90)

        self.valsMO, self.vexMO = casida.solveMO(rFk, rSk)

        mo_coeff     = self.vexMO.copy()
        mo_coeff_adj = mo_coeff.conj().transpose(0, 1, 3, 2)

        if self.config.qpac_enabled:
            print("Getting QP ac energy levels from self-energy (all k-points).")
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

        vbm = max(valsMO_qp[0, ik, self.occ - 1] for ik in range(self.nk))
        cbm = min(valsMO_qp[0, ik, self.occ]     for ik in range(self.nk))
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
        cbm = min(self.valsMO[0, ik, self.occ]     for ik in range(self.nk)) * AU2EV
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

    def _extract_q0_polarization(self, tildeP_iw, nQ):
        """
        Extract and normalise polarization to (niw, ns, NQ, NQ, 1).

        For finite-q, the pi_file already stores the polarization for the
        specific q-point (written by generate_finite_q_test.py).  This method
        just ensures a consistent shape regardless of how the file was written.

        Handles the same two layouts as PeriodicBSESolver._extract_q0_polarization.
        """
        shape = tildeP_iw.shape
        niw   = shape[0]

        if tildeP_iw.ndim == 5 and shape[-1] == 1:
            return tildeP_iw

        if tildeP_iw.ndim == 5 and shape[2] == nQ and shape[3] == nQ:
            # (niw, ns, nQ, nQ, nk) — take first q-slice
            return tildeP_iw[:, :, :, :, :1]

        if tildeP_iw.ndim == 5 and shape[3] == nQ and shape[4] == nQ:
            # (niw, ns, nk, nQ, nQ) — take k=0 slice
            Pi_q0 = tildeP_iw[:, :, 0:1, :, :]           # (niw, ns, 1, nQ, nQ)
            return Pi_q0.transpose(0, 1, 3, 4, 2)         # (niw, ns, nQ, nQ, 1)

        print(f"  WARNING: unexpected Pi shape {shape}; attempting reshape.")
        return tildeP_iw.reshape(niw, 1, nQ, nQ, 1)

    def prepare_interaction_matrices(self):
        """
        Load VQ_kk (same-k) and VQ_kq (off-diagonal), MO-transform them,
        and load / compute the screened polarization P̃(q, iΩ).

        Files read
        ----------
        VQ_kk : int_path/VQ_0.h5          — same-k GDF integrals
        VQ_kq : int_path/VQ_q{q_idx}.h5   — off-diagonal (k, k+q) integrals
                (for q_idx=0 VQ_kq == VQ_kk)
        Pi    : config.pi_file             — P̃(q, iΩ) for this q

        Returns
        -------
        VQ_mo_kk : ndarray, shape (nk, NQ, nao, nao)
        VQ_ia    : ndarray, shape (nk, NQ, nao, nao)
        tildeP_iw: ndarray, shape (niw, ns, NQ, NQ, 1)
        """
        start = time.time()

        # --- Same-k VQ (for W_A: oo at k, vv at k+q) ---
        print(f"Reading same-k VQ from {self.config.int_path}VQ_0.h5")
        VQ_kk_ao = ct.readVQ(self.config.int_path + "VQ_0.h5")
        if VQ_kk_ao.shape[0] != self.nk:
            print(f"  WARNING: VQ_kk has {VQ_kk_ao.shape[0]} k-points "
                  f"but input.h5 has {self.nk}.  Using {VQ_kk_ao.shape[0]}.")
            self.nk = VQ_kk_ao.shape[0]

        nQ = VQ_kk_ao.shape[1]
        print(f"  VQ_kk shape: {VQ_kk_ao.shape}  (nk={self.nk}, nQ={nQ})")

        # --- Off-diagonal VQ_kq (for transition density, exchange, W_B) ---
        kq_file = self.config.int_path + f"VQ_q{self.q_idx}.h5"
        if self.q_idx == 0:
            print("  q_idx=0: using same-k integrals for VQ_kq as well.")
            VQ_kq_ao = VQ_kk_ao
        else:
            print(f"Reading off-diagonal VQ from {kq_file}")
            VQ_kq_ao = ct.readVQ(kq_file)
            print(f"  VQ_kq shape: {VQ_kq_ao.shape}")

        # Update kq_map now that nk is confirmed
        self.kq_map = (np.arange(self.nk) + self.q_idx) % self.nk

        # --- MO transforms ---
        print("Transforming VQ to MO basis.")
        rSk = self.results['rSk']
        VQ_mo_kk = casida_fq.VQ_ao2mo_kk(VQ_kk_ao, self.vexMO, rSk=rSk)
        VQ_ia    = casida_fq.VQ_ao2mo_kq_proper(VQ_kq_ao, self.vexMO, self.kq_map, rSk=rSk)

        # --- Screened polarization P̃(q, iΩ) ---
        if self.config.calc_pi_on_fly:
            print(f"Calculating Pi on the fly from iteration {self.config.iter_W}.")
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
            print(f"! Reading Pi from {self.config.pi_file}; "
                  "iter_W argument not used. !")

        # Normalise shape to (niw, ns, NQ, NQ, 1)
        tildeP_iw = self._extract_q0_polarization(tildeP_iw, nQ)

        niw = tildeP_iw.shape[0]
        N   = self.nk * self.occ * self.virt
        print(f"  Processing {niw} imaginary frequency points.")
        print(f"  Problem size: nk={self.nk}, nao={self.nao}, "
              f"nocc={self.occ}, nvirt={self.virt}")
        print(f"  BSE Hamiltonian dimension: 2 x {N}")
        print(f"  kq_map: {self.kq_map.tolist()}")

        end = time.time()
        print(f"    Preparing interaction matrices: {end - start:.4f} secs")

        return VQ_mo_kk, VQ_ia, tildeP_iw

    # ------------------------------------------------------------------
    # BSE equations
    # ------------------------------------------------------------------

    def solve_bse_equations(self, VQ_mo_kk, VQ_ia, tildeP_iw):
        """
        Solve the finite-q periodic BSE Casida equations.

        Steps
        -----
        1.  Static BSE: diagonalize H_stat = H_eff(iΩ=0) → (effVals, effVex).
        2.  Infinite-frequency limit H2p_inf (Pi→0, bare Coulomb only).
        3.  Dynamic correction: compute diag(V^{-1} H(iΩ) V) at all Matsubara
            frequencies → H2p_dyn.
        4.  Build two-particle response function G2p(iΩ) from H2p_dyn.
        5.  Plasmon-pole fit to extract real excitation energies.
        """
        niw = tildeP_iw.shape[0]

        # Static polarization at Ω=0
        Pi_stat   = casida_fq.build_Pi_stat(tildeP_iw)               # (NQ, NQ)
        q_is_zero = (self.q_idx == 0)

        # 1. Static BSE
        print("*" * 90)
        print("    Building and diagonalizing static BSE Hamiltonian    ")
        print("*" * 90)
        t0 = time.time()
        effVals_static, effVex_static, H_stat = casida_fq.solveHstatic_fq(
            Pi_stat, VQ_mo_kk, VQ_ia, self.valsMO, self.nelec, self.kq_map,
            ex_type=self.config.excitation_type, tda=False,
            q_is_zero=q_is_zero
        )
        print(f"    Static BSE: {time.time() - t0:.4f} secs")

        # 2. Infinite-frequency limit (Pi=0)
        Pi_inf = np.zeros_like(Pi_stat)
        _, _, H_inf = casida_fq.solveHstatic_fq(
            Pi_inf, VQ_mo_kk, VQ_ia, self.valsMO, self.nelec, self.kq_map,
            ex_type=self.config.excitation_type, tda=False,
            q_is_zero=q_is_zero
        )
        effVex_inv = np.linalg.inv(effVex_static)
        H2p_inf    = np.einsum('ij,jk,ki->i', effVex_inv, H_inf, effVex_static).real

        # 3. Dynamic correction
        print("*" * 90)
        print("    Computing dynamic BSE correction    ")
        print("*" * 90)
        t1 = time.time()
        H2p_dyn = casida_fq.HDynDiagApprox_fq(
            tildeP_iw, effVex_static, VQ_mo_kk, VQ_ia,
            self.valsMO, self.nelec, self.kq_map,
            Pi_stat=Pi_stat, q_is_zero=q_is_zero,
            n_jobs=self.config.n_jobs
        )
        print(f"    Dynamic correction: {time.time() - t1:.4f} secs")

        # Sort everything by real part of static eigenvalues
        idx            = np.argsort(effVals_static.real)
        effVals_static = effVals_static[idx]
        effVex_static  = effVex_static[:, idx]
        H2p_inf        = H2p_inf[idx]
        H2p_dyn        = H2p_dyn[:, idx]

        niw = H2p_dyn.shape[0]

        # 4. Two-particle response function G2p
        G2p_iw  = casida.G2p(H2p_dyn, self.config.ir_file, self.config.beta)
        G2p_tau = ct.omega2tauFT(G2p_iw, self.config.beta, self.config.ir_file)

        # 5. Plasmon-pole fit
        wpole_data, S_data, _, res_norm_data = plasPole.fit_G_update(
            G2p_iw, self.config.ir_file, beta=self.config.beta
        )

        # AO decomposition: only meaningful for nk=1
        if self.nk == 1:
            effVex_occ_ao, effVex_virt_ao = casida.effVex2AO(
                effVex_static, self.vexMO[0, 0, :], self.nelec
            )
        else:
            effVex_occ_ao  = np.zeros((0,), dtype=np.complex128)
            effVex_virt_ao = np.zeros((0,), dtype=np.complex128)

        self.results.update({
            'G2p_iw_updated':    G2p_iw,
            'G2p_tau_updated':   G2p_tau,
            'H2p_inf':           H2p_inf,
            'H_stat':            H_stat,
            'effVals_static':    effVals_static,
            'effVex_static':     effVex_static,
            'pole_fit':          wpole_data,
            'S_fit':             S_data,
            'residual_norm_fit': res_norm_data,
            'occ_AO_indices':    effVex_occ_ao,
            'virt_AO_indices':   effVex_virt_ao,
        })

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_results(self):
        """Save results to HDF5 and print excitation energies."""
        self.save_results()
        self.print_results()

    def run(self):
        """Main execution method for finite-q BSE."""
        self.print_header()

        # Load raw Hamiltonian
        rFk, rSk = self.load_input_data()

        # Solve MOs and apply QP correction
        self.solve_mo()

        # Update kq_map now that nk is known (also updated in prepare_interaction_matrices)
        self.kq_map = (np.arange(self.nk) + self.q_idx) % self.nk
        print(f"\n  q_idx={self.q_idx}, nk={self.nk}, kq_map={self.kq_map.tolist()}")

        # Load and transform interaction matrices
        VQ_mo_kk, VQ_ia, tildeP_iw = self.prepare_interaction_matrices()

        # Solve BSE
        start_bse = time.time()
        self.solve_bse_equations(VQ_mo_kk, VQ_ia, tildeP_iw)
        print("*" * 90)
        print(f"    BSE Casida equation: {time.time() - start_bse:.4f} secs    ")
        print("*" * 90)

        # Write output
        self.write_results()

    def solve_mo(self):
        """Solve for MOs (bridges load_input_data → prepare_interaction_matrices)."""
        rFk = self.results['rFk']
        rSk = self.results['rSk']
        self.solve_molecular_orbitals(rFk, rSk)
