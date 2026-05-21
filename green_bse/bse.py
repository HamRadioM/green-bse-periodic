#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Main script used for solving the dynamic BSE Casida equations            #
#    on the Matsubara frequency grid.                                         #
#                                                                             #

import platform
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
import pyscf
import scipy
import casidaEq as casida
import h5py
import contract as ct
import numpy as np
import qp
import plasPole

# optional imports for system monitoring
try:
    import psutil
    import multiprocessing as mp
    from joblib import effective_n_jobs
    MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: psutil or joblib not available. System monitoring will be limited.")
    MONITORING_AVAILABLE = False

AU2EV = 27.211386245981

@dataclass
class BSEConfig:
    """
    Configuration class for BSE calculations.
    Default values are provided for all parameters, 
    which can be overridden by command line arguments.
    """
    __version__ = "0.1.0"
    # File paths
    input_file: str = "input.h5"
    sim_file: str = "sim.h5"
    output_file: str = "bse.h5"
    int_path: str = "df_hf_int/"
    ir_file: Optional[str] = None
    pi_file: str = "p_iw_tilde_q0.h5"
    
    # Physical parameters
    beta: float = 1000.0
    iteration: int = -1
    iter_W: int = -1
    
    # Calculation settings
    excitation_type: str = "normal"
    n_spins: int = 1
    qpac_enabled: bool = False
    tda_enabled: bool = False
    calc_pi_on_fly: bool = False
    debug_enabled: bool = False
    
    # Performance
    n_jobs: int = -1
    monitoring_enabled: bool = False
    
    @classmethod
    def from_args(cls, args):
        """
        Create configuration from parsed arguments.
        """
        return cls(
            input_file=args.input,
            sim_file=args.sim,
            output_file=args.output,
            int_path=args.int_path,
            ir_file=args.ir_file,
            pi_file=args.pi_file,
            beta=args.beta,
            iteration=args.iter,
            iter_W=args.iter_W,
            excitation_type=args.type,
            n_spins=args.ns,
            qpac_enabled=bool(args.qpac),
            tda_enabled=bool(args.tda),
            calc_pi_on_fly=bool(args.calc_pi),
            debug_enabled=bool(args.debug),
            n_jobs=args.n_jobs,
            monitoring_enabled=bool(args.monitor)
        )


class SystemMonitor:
    """
    System monitoring utilities.
    Only enabled if psutil and joblib are available 
    and monitoring is enabled in the config.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and MONITORING_AVAILABLE
        self.start_time = time.time()
    
    def print_system_info(self):
        """Print system information."""
        if not self.enabled:
            print("System monitoring not available")
            return
            
        try:
            cpu_physical = psutil.cpu_count(logical=False)
            cpu_logical = psutil.cpu_count(logical=True)
            memory = psutil.virtual_memory()
            
            print("=" * 90)
            print("SYSTEM INFORMATION")
            print("-" * 90)
            print(f"Physical CPU cores:    {cpu_physical}")
            print(f"Logical CPU cores:     {cpu_logical}")
            print(f"Total memory:          {memory.total / (1024**3):.2f} GB")
            print(f"Available memory:      {memory.available / (1024**3):.2f} GB")
            print(f"Used memory:           {memory.used / (1024**3):.2f} GB ({memory.percent:.1f}%)")
            print("=" * 90)
        except Exception as e:
            print(f"System monitoring not available: {e}")
    
    def print_process_info(self):
        """
        Print current process information.
        """
        if not self.enabled:
            return
            
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            memory_percent = process.memory_percent()
            
            print("=" * 90)
            print("PROCESS INFORMATION")
            print("-" * 90)
            print(f"Process ID:            {process.pid}")
            print(f"Memory usage:          {memory_mb:.2f} MB ({memory_percent:.1f}%)")
            print(f"Number of threads:     {process.num_threads()}")
            print("=" * 90)
        except Exception as e:
            print(f"Process monitoring not available: {e}")
    
    def print_joblib_info(self, n_jobs):
        """
        Print joblib parallelization information.
        """
        if not self.enabled:
            print(f"Parallelization info: n_jobs={n_jobs}")
            return
            
        try:
            effective_jobs = effective_n_jobs(n_jobs)
            max_jobs = mp.cpu_count()
            
            print("=" * 90)
            print("PARALLELIZATION SETTINGS")
            print("-" * 90)
            print(f"Requested n_jobs:      {n_jobs}")
            print(f"Effective n_jobs:      {effective_jobs}")
            print(f"Maximum available:     {max_jobs}")
            print("n_jobs=-1 : use all available cores")
            print("n_jobs= 1 : serial execution")
            print("=" * 90)
        except Exception as e:
            print(f"Parallelization info: n_jobs={n_jobs} (monitoring unavailable): {e}")
    
    def estimate_memory_requirements(self, nao, nelec, niw):
        """
        Crude estimation of memory requirements for BSE calculations.
        """
        occ = nelec // 2
        virt = nao - occ
        
        # Estimate major array sizes (complex128 = 16 bytes)
        VQ_mb = (nao**3 * 16) / (1024**2)  # VQ matrix
        Pi_mb = (niw * nao**2 * 16) / (1024**2)  # Pi matrix
        G2p_mb = (niw * 2 * occ * virt * 16) / (1024**2)  # G2p matrices
        working_mb = 4 * (occ**2 * virt**2 * 16) / (1024**2)  # Working arrays

        total_mb = VQ_mb + Pi_mb + G2p_mb + working_mb

        print("=" * 90)
        print("MEMORY ESTIMATES")
        print("-" * 90)
        print(f"Problem size: nao={nao}, nelec={nelec}, niw={niw}")
        print(f"VQ matrix:            {VQ_mb:.2f} MB")
        print(f"Pi matrix:            {Pi_mb:.2f} MB")
        print(f"G2p matrices:         {G2p_mb:.2f} MB")
        print(f"Working arrays:       {working_mb:.2f} MB")
        print(f"Total estimated:      {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")
        print("=" * 90)
    
    def monitor_memory(self, stage: str = ""):
        """Simple memory monitoring."""
        if not self.enabled:
            return
            
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            elapsed = time.time() - self.start_time
            print(f"{stage}: {memory_mb:.2f} MB | Time: {elapsed:.2f}s")
        except Exception:
            pass


class BSESolver:
    """BSE Casida equation solver class."""
    
    def __init__(self, config: BSEConfig):
        """
        Initialize the calculation setup with placeholder values.
        """
        self.config = config
        self.monitor = SystemMonitor(config.monitoring_enabled)
        
        # Initializing basic properties (filled during calculation)
        self.nao = 0
        self.nelec = 0
        self.occ = 0
        self.virt = 0
        
        # Data storage
        self.valsMO = None
        self.vexMO = None
        self.results = {}
    
    def print_python_info(self):
        """Print Python environment information."""
        print("PYTHON ENVIRONMENT")
        print(f"BSE    version:  {self.config.__version__}")
        print(f"python version:  {platform.python_version()}")
        print(f"numpy  version:  {np.__version__}")
        print(f"scipy  version:  {scipy.__version__}")
        print(f"h5py   version:  {h5py.__version__}")
        print(f"pyscf  version:  {pyscf.__version__}")
    
    def print_header(self):
        """Print output header."""
        print("=" * 90)
        print("BSE CASIDA EQUATION SOLVER")
        print("FOR MATUSBARA GREEN'S FUNCTION")
        print("Copyright (c) 2025-2026 ")
        print("    Ming Wen <wenm@umich.edu>, University of Michigan.")
        print("    Gaurav Harsha <gharsha@umich.edu>, University of Michigan.")
        
        print("-" * 90)
        self.print_python_info()
        print("-" * 90)
        print(f"Input file:        {self.config.input_file}")
        print(f"Integral file:     {self.config.int_path}")
        print(f"Simulation file:   {self.config.sim_file}")
        print(f"Output file:       {self.config.output_file}")
        print(f"Beta (inverse T):  {self.config.beta}")
        print(f"Excitation type:   {self.config.excitation_type}")
        print(f"Monitoring:        {'Enabled' if self.config.monitoring_enabled else 'Disabled'}")
        print("=" * 90)
        
        if self.config.monitoring_enabled:
            self.monitor.print_system_info()
            self.monitor.print_process_info()
            self.monitor.print_joblib_info(self.config.n_jobs)
            
        print("*" * 90)
        print(f"    Starting Casida solver for scGW iteration: {self.config.iteration}    ")
        print(f"    iter = -1 means using the last iteration in the sim file.    ")
        print("*" * 90)
        
    def load_input_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load input data from scGW HDF5 files."""
        start = time.time()
        
        print("Reading IR file")
        with h5py.File(self.config.ir_file, "r") as f:
            # legacy support
            # wgrid = f["/bose/wsample"][()]
            # green-irgrid support
            wgrid = f["/bose/ngrid"][()]
            
        wgrid = 2 * wgrid * np.pi / self.config.beta
        
        print("Reading input file")
        with h5py.File(self.config.input_file, 'r') as f:
            rSk = f["/HF/S-k"][()].view(complex)
            rSk = rSk.reshape(rSk.shape[:-1])
            rFk_input = f["/HF/Fock-k"][()].view(complex)
            rFk_input = rFk_input.reshape(rFk_input.shape[:-1])
            # green-mbpt support
            rHk = f["/HF/H-k"][()].view(complex)  # Core Hamiltonian.
            rHk = rHk.reshape(rHk.shape[:-1])
            self.nao = f["/params/nao"][()]
            self.nelec = f["/params/nel_cell"][()]
        
        self.occ = self.nelec // 2
        self.virt = self.nao - self.occ
        
        print("Reading sim file")
        with h5py.File(self.config.sim_file, 'r') as f:
            it = f["iter"][()] if self.config.iteration == -1 else self.config.iteration
            
            if it == 1:
                print("Reading the HF level Fock matrix for G0W0.")
                # green-mbpt support
                rFk = rFk_input
                # legacy support
                # rFk = rFk.reshape(rFk.shape[:-1])  
            else:
                # legacy support
                # rFk = f[f"iter{it}/Fock-k"][()].view(complex)
                # green-mbpt support
                rFk = f["iter" + str(it) + "/Sigma1"][()].view(complex) + rHk
                
            rSigmak = f[f"iter{it}/Selfenergy/data"][()].view(complex)
            # rSigmak = rSigmak.reshape(rSigmak.shape[:-1])
            mu = f[f"iter{it}/mu"][()]
        
        end = time.time()
        print("*" * 90)
        print(f"    Reading files: {end - start:.4f} secs    ")
        print("*" * 90)
        
        # Store for later use
        self.results['rSk'] = rSk
        self.results['rFk'] = rFk
        self.results['rSigmak'] = rSigmak
        self.results['mu'] = mu
        self.results['wgrid'] = wgrid
        
        # Print memory estimates and current usage
        if self.config.monitoring_enabled:
            niw = len(wgrid)
            self.monitor.estimate_memory_requirements(self.nao, self.nelec, niw)
            self.monitor.monitor_memory("After file reading")
        
        return rFk, rSk
    
    def solve_molecular_orbitals(self, rFk: np.ndarray, rSk: np.ndarray):
        """Solve for molecular orbital eigenvalues and eigenvectors."""
        start = time.time()
        print("*" * 90)
        print("    Solving for MO eigenvalues    ")
        print("*" * 90)
        
        self.valsMO, self.vexMO = casida.solveMO(rFk, rSk)
        mo_coeff_gamma = np.zeros((self.config.n_spins, 1, self.nao, self.nao), dtype=np.complex128)
        mo_coeff_gamma[0, 0, :, :] = self.vexMO[...]
        mo_coeff = 1.0 * mo_coeff_gamma
        mo_coeff_adj = np.einsum('skpq -> skqp', mo_coeff.conj())
        
        # Pade interpolation using Sigma if enabled
        if self.config.qpac_enabled:
            print("Getting QP ac energy levels from self-energy.")
            Sigma_tk_int = np.einsum(
                'skab, tskbc, skcd -> tskad',
                mo_coeff_adj, self.results['rSigmak'], mo_coeff,
                optimize=True
            )
            valsMO_qp = qp.padeSigma(
                Sigma_tk_int, self.valsMO, self.config.beta, 
                self.results['mu'], self.config.ir_file
            )
            
            print("\n" + "=" * 90)
            print("QUASI-PARTICLE ENERGY LEVELS")
            print("-" * 90)
            print(f"{'Index':>8} {'Original (eV)':>20} {'QP Corrected (eV)':>20} {'Difference (eV)':>20}")
            print("-" * 90)
            
            n_print = min(30, len(self.valsMO[0, 0, :]))
            for i in range(n_print):
                orig_ev = self.valsMO[0, 0, i] * AU2EV
                qp_ev = valsMO_qp[0, 0, i] * AU2EV
                diff_ev = (valsMO_qp[0, 0, i] - self.valsMO[0, 0, i]) * AU2EV
                print(f"{i:8d} {orig_ev:20.6f} {qp_ev:20.6f} {diff_ev:20.6f}")
            if n_print < len(self.valsMO[0, 0, :]):
                print("...")

            print("=" * 90 + "\n")
            
            self.valsMO = valsMO_qp
            
            # Print energy gap
            homo_qp = valsMO_qp[0, 0, self.occ - 1] * AU2EV
            lumo_qp = valsMO_qp[0, 0, self.occ] * AU2EV
            gap_qp = lumo_qp - homo_qp
            
            print("=" * 90)
            print("QUASI-PARTICLE ENERGY GAP")
            print("-" * 90)
            print(f"    HOMO (QP)       :       {homo_qp:20.6f} eV")
            print(f"    LUMO (QP)       :       {lumo_qp:20.6f} eV")
            print(f"    Energy Gap (QP) :       {gap_qp:20.6f} eV")
            print("=" * 90 + "\n")
        
        else:
            print("QUASI-PARTICLE CORRECTION NOT ENABLED.")
            # Print energy gap
            homo = self.valsMO[0, 0, self.occ - 1] * AU2EV
            lumo = self.valsMO[0, 0, self.occ] * AU2EV
            gap = lumo - homo
            print("\n" + "=" * 90)
            print("ENERGY GAP")
            print("-" * 90)
            print(f"    HOMO       :             {homo:20.6f} eV")
            print(f"    LUMO       :             {lumo:20.6f} eV")
            print(f"    Energy Gap :             {gap:20.6f} eV")
            print("=" * 90 + "\n")
        
        end = time.time()
        print("*" * 90)
        print(f"    Solve MO eigenvalues: {end - start:.4f} secs    ")
        print("*" * 90)
        
        return mo_coeff_adj, mo_coeff
    
    def prepare_interaction_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare VQ and Pi matrices."""
        start = time.time()
        
        # Load VQ matrix
        # Only implemented for q=0 case for now (single unit cell or molecules)
        print("Reading VQ matrix from integral file.")
        VQ_ao = ct.readVQ(self.config.int_path + "VQ_0.h5")
        nQ = VQ_ao.shape[1]
        VQ = casida.VQ_ao2mo(VQ_ao, self.vexMO[0, 0, :])
        
        # Handle Pi matrix (either calculate or read from file)
        if self.config.calc_pi_on_fly:
            print(f"Calculating Pi on the fly from iteration {self.config.iter_W} of sim file.")
            tildeP_tau = ct.getPtilde(
                self.config.iter_W, self.nao, nQ,
                tau_h5=self.config.ir_file,
                int_path=self.config.int_path,
                sim_h5=self.config.sim_file
            )
            tildeP_iw = ct.tau2omegaFT(tildeP_tau, beta=self.config.beta, tau_h5=self.config.ir_file)
            niw = tildeP_iw.shape[0]
            tildeP_iw = tildeP_iw.reshape(niw, 1, nQ, nQ, 1)
            del tildeP_tau
            
            if self.config.monitoring_enabled:
                self.monitor.monitor_memory("After Pi calculation")
        else:
            tildeP_iw = ct.readPtilde(self.config.pi_file)
            print("! Because you chose to read from Pi file, the iteration of W you specified might not be used. !")
            
            if self.config.monitoring_enabled:
                self.monitor.monitor_memory("After Pi file reading")
        
        niw = tildeP_iw.shape[0]
        
        if self.config.monitoring_enabled:
            print(f" Processing {niw} imaginary frequency points (half)")
            print(f" Problem dimensions: nao={self.nao}, nelec={self.nelec}, occ={self.occ}, virt={self.virt}")
        
        return VQ, tildeP_iw
    
    def solve_bse_equations(self, VQ: np.ndarray, tildeP_iw: np.ndarray):
        """
        Solve BSE equations after all matrices are prepared.
        The infinity solution is also saved for book-keeping and comparison purposes.
        G2p refer to the two-particle response function, 
        which is referred to as an auxiliary function F in the manuscript.
        """
        niw = tildeP_iw.shape[0]
        
        if not self.config.tda_enabled:
            # Full BSE calculation
            diffEps_ov = casida.mo2ovStat(
                casida.diffEpsMat(self.valsMO[0, 0, :], self.nelec)
            ).reshape(self.occ * self.virt, self.occ * self.virt)
            
            # Use static limit for initial guess
            effVals_static, effVex_static, H_stat = casida.solveHstatic(
                tildeP_iw[niw//2, 0, :, :, 0], VQ, diffEps_ov, self.nelec, 
                ex_type=self.config.excitation_type, tda=0
            )
            
            H2p_dyn = casida.HDynDiagApprox(
                tildeP_iw,effVex_static,VQ,self.valsMO,
                self.nelec,ex_type=self.config.excitation_type,
                n_jobs=self.config.n_jobs
            )
            
            H2p_inf = H2p_dyn[0]
            
            effVex_occ_ao, effVex_virt_ao = casida.effVex2AO(
                effVex_static, self.vexMO[0, 0, :], self.nelec
            )
            
            # Sort eigenvalues and eigenvectors by eigenvalues
            idx = np.argsort(effVals_static)
            effVals_static = effVals_static[idx]
            H2p_inf = H2p_inf[idx]
            H2p_dyn = H2p_dyn[:, idx]
            effVex_static  = effVex_static[:, idx]
            effVex_occ_ao  = effVex_occ_ao[:, idx] 
            effVex_virt_ao = effVex_virt_ao[:, idx]
            niw = H2p_dyn.shape[0]
            
            # Initiaize the two-particle response function from H2p_inf for referential purposes.
            # Not saved in putput file but can be useful for debugging and comparison.
            # G2p_iw_init_inv = casida.initG2p_inv(H2p_inf, self.config.ir_file, self.config.beta)
            # G2p_iw_init = np.zeros(G2p_iw_init_inv.shape, dtype=np.complex128)
            # for iw in range(niw):
            #     G2p_iw_init[iw, :] = 1 / G2p_iw_init_inv[iw, :]
            
            G2p_iw_updated = casida.G2p(H2p_dyn, self.config.ir_file, self.config.beta)

            G2p_tau_updated = ct.omega2tauFT(G2p_iw_updated, self.config.beta, self.config.ir_file)
            
            wpole_data, S_data, _, res_norm_data = plasPole.fit_G_update(
                G2p_iw_updated, self.config.ir_file, beta=self.config.beta
            )
            
            self.results.update({
                # 'G2p_iw_init': G2p_iw_init,
                'G2p_iw_updated': G2p_iw_updated,
                'G2p_tau_updated': G2p_tau_updated,
                'H2p_inf': H2p_inf,
                'H_stat': H_stat,
                'effVals_static': effVals_static,
                'effVex_static': effVex_static,
                'pole_fit': wpole_data,
                'S_fit': S_data,
                'residual_norm_fit': res_norm_data,
                'occ_AO_indices': effVex_occ_ao,
                'virt_AO_indices': effVex_virt_ao
            })
            
        else:
            print("TDA approximation is not fully implemented yet. Exiting.")
            raise NotImplementedError("TDA approximation not implemented")
    
    def save_results(self):
        """Save calculation results to HDF5 file."""
        start = time.time()
        with h5py.File(self.config.output_file, 'w') as f:
            f["/G2pUpdated/data"] = self.results['G2p_iw_updated']
            f["/G2pUpdated_tau/data"] = self.results['G2p_tau_updated']
            f["/InfExcVals/data"] = self.results['H2p_inf']
            f["/StatExcVals/data"] = self.results['effVals_static']
            f["/StatExcVex/data"] = self.results['effVex_static']
            f["/PoleFit/data"] = self.results['pole_fit']
            f["/SFit/data"] = self.results['S_fit']
            f["/ResNorm/data"] = self.results['residual_norm_fit']
            f["/MOvectors/data"] = self.vexMO
            f["/MOvals/data"] = self.valsMO
            f["/AOindices/occ"] = self.results['occ_AO_indices']
            f["/AOindices/virt"] = self.results['virt_AO_indices']
        
        end = time.time()
        print("*" * 90)
        print(f"    Write output: {end - start:.4f} secs    ")
        print("*" * 90)
    
    def print_results(self):
        """Print calculation results."""
        print("=" * 100)
        print("NEUTRAL EXCITATIONS")
        print("Only printing the first 20 positive eigenvalues.")
        print("-" * 100) 
        
        # Get eigenvalues
        idx_li = self.results['effVals_static'] > 0
        static_pos = self.results['effVals_static'][idx_li].real
        static_ev = static_pos * AU2EV
        inf_pos = self.results['H2p_inf'][idx_li].real
        inf_ev = inf_pos * AU2EV
        pole_data = self.results['pole_fit']
        s_data = self.results['S_fit']
        
        # Check dimensions and extract first peak if needed
        if pole_data.ndim > 1:
            dyn_pos = pole_data[:, 0][idx_li].real
            res = self.results['residual_norm_fit'][:][idx_li].real
        else:
            dyn_pos = pole_data[idx_li].real
            res = self.results['residual_norm_fit'][idx_li].real

        dyn_ev = dyn_pos * AU2EV
        # res = res
        
        # Determine how many to display (first 20 by default)
        n_static = min(20, len(static_ev))
        n_inf = min(20, len(inf_ev))
        n_dyn = min(20, len(dyn_ev))
        n_display = max(n_static, n_inf, n_dyn)
        
        if n_display > 0:
            print(f"{'Index':>5} {'Static (eV)':>15} {'Infinite (eV)':>15} {'Dynamic (eV)':>15} {'Residual':>15}")
            print("-" * 90)
            
            for i in range(n_display):
                static_val = static_ev[i] if i < len(static_ev) else None
                inf_val = inf_ev[i] if i < len(inf_ev) else None
                dyn_val = dyn_ev[i] if i < len(dyn_ev) else None
                
                static_str = f"{static_val:10.4f}" if static_val is not None else "N/A"
                inf_str = f"{inf_val:10.4f}" if inf_val is not None else "N/A"
                dyn_str = f"{dyn_val:10.4f}" if dyn_val is not None else "N/A"
                res_str = f"{res[i]:10.4f}" if i < len(res) else "N/A"

                print(f"{i:5d} {static_str:>15} {inf_str:>15} {dyn_str:>15} {res_str:>15}")
                
            print("......")
            print("All results are saved in the output HDF5 file.")
        else:
            print("No positive eigenvalues found in either calculation")

        print("=" * 90)
    
    def run(self):
        """Main execution method."""
        self.print_header()
        
        # Load input data
        rFk, rSk = self.load_input_data()
        
        # Solve molecular orbitals
        mo_coeff_adj, mo_coeff = self.solve_molecular_orbitals(rFk, rSk)
        
        # Prepare interaction matrices
        VQ, tildeP_iw = self.prepare_interaction_matrices()
        
        # Solve BSE equations
        start_bse = time.time()
        self.solve_bse_equations(VQ, tildeP_iw)
        # self.solve_bse_equations_full(VQ, tildeP_iw)
        end_bse = time.time()
        print("*" * 90)
        print(f"    BSE Casida equation: {end_bse - start_bse:.4f} secs    ")
        print("*" * 90)
        
        # Print results
        self.print_results()
        
        # Save results
        self.save_results()
        
        # Final monitoring summary
        if self.config.monitoring_enabled:
            print("\n" + "=" * 90)
            print("FINAL SYSTEM STATUS")
            print("-" * 90)
            self.monitor.monitor_memory("Final memory usage")
            try:
                if MONITORING_AVAILABLE:
                    process = psutil.Process()
                    print(f"Final thread count:    {process.num_threads()}")
                print("=" * 90)
            except Exception:
                print("Final monitoring unavailable")
                print("=" * 90)

def create_argument_parser():
    """
    Create command line argument parser.
    This function defines all the command line arguments 
    that can be used to configure the BSE calculation.
    Default values are set to do singlet calculation without TDA.
    """
    parser = argparse.ArgumentParser(
        description="BSE Casida equation solver with object-oriented design."
    )
    
    # File paths
    parser.add_argument("--input", type=str, default="input.h5",
                        help="HDF5 file path for input of scGW.")
    parser.add_argument("--sim", type=str, default="sim.h5",
                        help="HDF5 file path for output of scGW.")
    parser.add_argument("--output", type=str, default="bse.h5",
                        help="HDF5 file path for output of BSE.")
    parser.add_argument("--int_path", type=str, default="df_hf_int/",
                        help="Path for the VQ.h5 file that contains information about \
                            the electronic repulsion integrals. \
                            Note that it is the path to the folder \
                            that contains VQ_0.h5, not the file itself.")
    parser.add_argument("--ir_file", type=str, default=None,
                        help="HDF5 file path that contains information about the IR grid.")
    parser.add_argument("--pi_file", type=str, default="p_iw_tilde_q0.h5",
                        help="HDF5 file path that contains information about \
                            non-interacting susceptibility.")
    
    # Parameters
    parser.add_argument("--beta", type=float, default=1000, 
                        help="Inverse temperature. Default is 1000, which corresponds to \
                            a temperature of about 300K.")
    parser.add_argument("--iter", type=int, default=-1,
                        help="Iteration number of the scGW cycle to use for BSE. \
                            Default -1 means using the latest iteration.")
    parser.add_argument("--iter_W", type=int, default=-1,
                        help="Iteration number of screened Coulomb W to use. \
                            Default -1 means using the latest W. \
                            If iter_W = 1, then W will be calculated \
                            from the mean-field (HF/DFT) reference.")
    parser.add_argument("--ns", type=int, default=1,
                        help="Number of spins. It must be 1 for Casida formulation \
                            i.e. restricted calculation.")
    parser.add_argument("--type", type=str, default="singlet",
                        help="Type of excitations (singlet or triplet) to be calculated. \
                            Default is singlet.")

    # Calculation switches
    parser.add_argument("--qpac", type=int, default=1,
                        help="Enable Quasi-Particle (QP) Approximation? \
                            Default is 1, which means using the QP-corrected \
                            energy levels as input for BSE calculation.")
    parser.add_argument("--tda", type=int, default=0,
                        help="Enable Tamm-Dancoff approximation? \
                            Default is 0, which means solving the full BSE equations.")
    parser.add_argument("--calc_pi", type=int, default=1,
                        help="Calculate Pi on the fly. If False, must provide a pi_file to read from. \
                            Default is 1, which means calculating Pi from the scGW output file.")

    # Performance switches
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of parallel jobs for frequency loops (-1 = all cores, 1 = serial) \
                            Default is -1, which means using all available cores in the node.")
    parser.add_argument("--monitor", type=int, default=1,
                        help="Enable system monitoring and memory tracking? \
                            Default is 1, which means enabling system monitoring and memory tracking.")
    parser.add_argument("--debug", type=int, default=0,
                        help="Enable debugging mode? \
                            Default is 0, which means disabling debugging mode.")
    
    return parser
