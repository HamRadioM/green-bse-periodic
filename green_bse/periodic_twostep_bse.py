#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Periodic, finite-Q two-step active-space BSE solver (RPA-scaling),       #
#    driven from a periodic scGW output.                                      #
#                                                                             #
#    This is the crystalline analogue of twostep_bse.py.  It reuses the same  #
#    three-step Q-space construction (periodic_active_bse.two_step_bse_periodic#
#    ) and the finite-q MO/integral conventions of the full Casida solver     #
#    (casidaEq_finite_q), so in the full-active limit it reproduces the full  #
#    periodic BSE.  Excitations are the poles of the Q-space polarization,     #
#    extracted by AAA / plasmon-pole continuation (active_bse.extract_*).      #
#                                                                             #
#    Inputs (raw green-mbpt layout, --space_symm False)                       #
#    --------------------------------------------------                       #
#      input_file : mean_field_input.h5  -- HF/S-k, Fock-k, H-k (full BZ),    #
#                   params, and symmetry/{k,pairs} maps.                      #
#      sim_file   : periodic scGW sim.h5 -- iterN/{Sigma1,Selfenergy,G_tau,mu}#
#                   stored on the IBZ; unfolded to the full BZ here via        #
#                   periodic_integrals.to_full_bz (U_k . X . U_k^dagger).      #
#      int_path   : folder with the raw irreducible-k-pair DF integrals        #
#                   (meta.h5 + VQ_*.h5).  V_Q(k1,k2) for the needed same-k and #
#                   off-diagonal pairs is reconstructed by periodic_integrals. #
#      ir_file    : IR-grid HDF5.                                             #
#    The screened polarization Pi_stat(Q_exc) is built on the fly from the     #
#    scGW Green's function (no external PolarizationSolver file needed).       #
#                                                                             #
#    Quasiparticle energies are generated on the fly per k-point: the static  #
#    GW Fock eigenproblem F(k) C(k) = S(k) C(k) E(k) followed by a Pade QP    #
#    correction from Sigma(iw) (qp.padeSigma over all k), mirroring           #
#    FiniteQBSESolver.solve_molecular_orbitals.                               #
#                                                                             #

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import h5py

import casidaEq as casida
import casidaEq_finite_q as casida_fq
import qp
import periodic_active_bse as pabse
import periodic_integrals as pint
import polarization as pol
from irFT import IR_factory
from twostep_bse import default_active_spaces, auto_active_space

AU2EV = 27.211386245981


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class PeriodicTwoStepBSEConfig:
    """Parameters for the periodic finite-Q two-step active-space BSE run."""
    input_file: str = "mean_field_input.h5"      # mean-field input (full BZ)
    sim_file: str = "scGW_sim.h5"                # periodic scGW output (IBZ)
    int_path: str = "df_hf_int/"                 # raw green-mbpt k-pair integrals
    #   Integrals for the SCREENING (polarization Pi and the screened-direct W).
    #   green-mbpt writes a second set, df_int, identical to df_hf_int except the
    #   q=0 (diagonal k-pair) Coulomb head is Ewald/Madelung-regularized -- this
    #   is the set scGW itself uses for Sigma_c, so it is the consistent choice
    #   for the BSE screening.  None -> auto-detect a sibling 'df_int' next to
    #   int_path, falling back to int_path (no finite-size correction) if absent.
    screening_int_path: Optional[str] = None
    ir_file: Optional[str] = None                # IR grid HDF5
    output_file: str = "periodic_two_step_bse.pdf"

    q_idx: int = 0                               # exciton momentum index (BvK)
    beta: float = 1000.0
    iteration: int = -1                          # scGW iteration (-1 = last)
    qpac_enabled: bool = True                    # QP correction from Sigma(iw)
    channel: str = "singlet"                     # "singlet" or "triplet"
    screening: str = "qresolved"                 # "qresolved" W^{k-k'} or "static" Pi(Q_exc)

    # spectrum / extraction
    eta: float = 0.05                            # Lorentzian broadening (eV)
    w_max: float = 1.0                           # pole window (a.u.) for AAA
    grid_max: float = 25.0                       # plot range (eV)
    method: str = "aaa"                          # aaa | plaspole | hybrid
    active_mode: str = "series"                  # "series" (nested) or "auto"

    @classmethod
    def from_args(cls, args):
        return cls(
            input_file=args.input, sim_file=args.sim, int_path=args.int_path,
            screening_int_path=getattr(args, "screening_int_path", None),
            ir_file=args.ir_file, output_file=args.output,
            q_idx=args.q_idx, beta=args.beta, iteration=args.iter,
            qpac_enabled=bool(args.qpac), channel=args.channel,
            screening=args.screening,
            eta=args.eta, w_max=args.w_max, grid_max=args.grid_max,
            method=args.method, active_mode=args.active,
        )


# ===========================================================================
# Solver
# ===========================================================================
class PeriodicTwoStepBSESolver:
    def __init__(self, config: PeriodicTwoStepBSEConfig):
        self.config = config
        if config.ir_file is None:
            raise ValueError("ir_file is required (path to the IR-grid HDF5).")
        self.results = {}

    # --- 1. read mean-field (full BZ) + scGW output (IBZ -> unfold to BZ) --
    def load_input_data(self):
        c = self.config
        print("Reading input/sim files ...")
        with h5py.File(c.input_file, "r") as f:
            rSk = f["/HF/S-k"][()].view(complex);     rSk = rSk.reshape(rSk.shape[:-1])
            rFk_in = f["/HF/Fock-k"][()].view(complex); rFk_in = rFk_in.reshape(rFk_in.shape[:-1])
            rHk = f["/HF/H-k"][()].view(complex);     rHk = rHk.reshape(rHk.shape[:-1])
            self.nao = int(f["/params/nao"][()])
            self.nelec = int(f["/params/nel_cell"][()])
        # mean-field is on the FULL BZ; rFk_in shape (ns, nk, nao, nao)
        self.nk = rFk_in.shape[1]
        self.occ = self.nelec // 2
        self.virt = self.nao - self.occ

        # IBZ<->BZ unfolding maps (scGW G/Sigma are stored on the IBZ even with
        # --space_symm False, because time-reversal still reduces the k-mesh).
        self.ksym = pint.load_k_symmetry(c.input_file)

        with h5py.File(c.sim_file, "r") as f:
            it = int(f["iter"][()]) if c.iteration == -1 else c.iteration
            Sig1_ibz = f["iter%d/Sigma1" % it][()].view(complex)            # (ns, ink, nao, nao)
            Sig_ibz = f["iter%d/Selfenergy/data" % it][()].view(complex)    # (nt, ns, ink, nao, nao)
            G_ibz = f["iter%d/G_tau/data" % it][()].view(complex)          # (nt, ns, ink, nao, nao)
            mu = f["iter%d/mu" % it][()]

        # Unfold to the full BZ so everything is consistent with the mean field.
        Sig1 = pint.unfold_to_bz(Sig1_ibz, self.ksym, 1)                    # (ns, nk, nao, nao)
        rSigmak = pint.unfold_to_bz(Sig_ibz, self.ksym, 2)                 # (nt, ns, nk, nao, nao)
        self.G_tau = pint.unfold_to_bz(G_ibz, self.ksym, 2)               # (nt, ns, nk, nao, nao)

        if it == 1:                                    # G0W0: HF reference Fock
            rFk = rFk_in
        else:
            rFk = Sig1 + rHk

        # Exact k-vector arithmetic on the BvK mesh (the cyclic (k+q)%nk map is
        # wrong for 2-D/3-D meshes); kq_map[k] = index of k + Q_exc.
        self.kmaps = pint.build_kpoint_maps(c.input_file)
        self.kq_map = self.kmaps["ksum"][:, c.q_idx]
        self.results.update(rSk=rSk, rFk=rFk, rSigmak=rSigmak, mu=mu, iter=it)
        print("    iter=%d, nk=%d (ink=%d), nao=%d, occ=%d, q_idx=%d"
              % (it, self.nk, self.ksym["ink"], self.nao, self.occ, c.q_idx))

    # --- 2. QP energies + eigenvectors per k ON THE FLY -------------------
    def solve_quasiparticles(self):
        c = self.config
        rFk, rSk = self.results["rFk"], self.results["rSk"]

        # static GW Fock eigenproblem  F(k) C(k) = S(k) C(k) E(k), all k at once
        valsMO, vexMO = casida.solveMO(rFk, rSk)          # (ns,nk,nao), (ns,nk,nao,nao)
        self.vexMO = vexMO

        if c.qpac_enabled:
            print("Generating QP energies from Sigma(iw) (Pade + QP equation, per k) ...")
            mo_coeff_adj = vexMO.conj().transpose(0, 1, 3, 2)
            Sigma_mo = np.einsum("skab, tskbc, skcd -> tskad",
                                 mo_coeff_adj, self.results["rSigmak"], vexMO,
                                 optimize=True)
            # qp.padeSigma is written for a single k-point (it collapses the
            # k-axis); call it per k-point and reassemble the QP energies.
            qp_vals = valsMO.copy()
            for k in range(self.nk):
                ek = qp.padeSigma(Sigma_mo[:, :, k:k+1], valsMO[:, k:k+1],
                                  c.beta, self.results["mu"], c.ir_file)
                qp_vals[:, k] = ek[:, 0]
            valsMO = qp_vals
        else:
            print("QP correction disabled: using static-GW Fock eigenvalues.")

        self.mo_energy = valsMO[0].real                    # (nk, nao)
        vbm = max(self.mo_energy[k, self.occ - 1] for k in range(self.nk))
        cbm = min(self.mo_energy[k, self.occ]     for k in range(self.nk))
        print("    VBM %.4f eV, CBM %.4f eV, fundamental gap %.4f eV"
              % (vbm * AU2EV, cbm * AU2EV, (cbm - vbm) * AU2EV))

    # --- 3. DF integrals + MO transform + screening + grid ----------------
    def _resolve_screening_path(self):
        """Path to the integral set used for the screening (Pi, W).

        Defaults to the Ewald-corrected 'df_int' (a sibling of int_path) when
        present -- this matches the integrals scGW used for Sigma_c, so the BSE
        screening is finite-size consistent with its own backbone.  Falls back
        to int_path (df_hf_int, no q=0 correction) if df_int is unavailable.
        """
        c = self.config
        cand = (c.screening_int_path if c.screening_int_path
                else os.path.join(os.path.dirname(os.path.normpath(c.int_path)),
                                  "df_int"))
        # load_stored_integrals concatenates "meta.h5", so keep a trailing sep.
        sep = lambda p: p if p.endswith(os.sep) else p + os.sep
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "meta.h5")):
            differs = os.path.normpath(cand) != os.path.normpath(c.int_path)
            return sep(cand), differs
        if c.screening_int_path:                       # explicit but missing -> error
            raise FileNotFoundError(
                "screening_int_path '%s' has no meta.h5" % cand)
        return sep(c.int_path), False

    def build_intermediates(self):
        c = self.config
        # Reconstruct the per-k DF integrals from the raw green-mbpt irreducible
        # k-pair storage (still pair-reduced by time-reversal at --space_symm False).
        stored = pint.load_stored_integrals(c.int_path)
        sym = pint.load_kpair_symmetry(c.input_file)
        if sym["nk"] != self.nk:
            raise ValueError("integral nk=%d != input nk=%d" % (sym["nk"], self.nk))
        self.NQ = stored.shape[1]

        # Screening integrals (df_int): Ewald-regularized q=0 head, used for Pi/W.
        screen_path, differs = self._resolve_screening_path()
        if differs:
            print("    [screening] using '%s' for Pi/W (q=0 Ewald-corrected, "
                  "finite-size consistent with scGW Sigma_c)." % screen_path)
            stored_s = pint.load_stored_integrals(screen_path)
            if stored_s.shape != stored.shape:
                raise ValueError("screening integrals %s != exchange integrals %s"
                                 % (stored_s.shape, stored.shape))
        else:
            print("    [screening] Pi/W use the same integrals as the exchange "
                  "('%s') -- NO q=0 finite-size correction." % screen_path)
            stored_s = stored

        VQ_kk_ao = pint.build_VQ_same_k(stored, sym)             # V_Q(k,k)
        if c.q_idx == 0:
            VQ_kq_ao = VQ_kk_ao
            print("    q_idx=0: same-k integrals used for the off-diagonal block.")
        else:
            VQ_kq_ao = pint.build_VQ_offdiag(stored, sym, self.kq_map)  # V_Q(k,k+Q)

        rSk = self.results["rSk"]
        # Exchange / bubble transition densities -> bare-Coulomb int_path (df_hf_int)
        self.VQ_mo_kk = casida_fq.VQ_ao2mo_kk(VQ_kk_ao, self.vexMO, rSk=rSk)
        self.VQ_ia = casida_fq.VQ_ao2mo_kq_proper(VQ_kq_ao, self.vexMO,
                                                  self.kq_map, rSk=rSk)

        # bosonic Matsubara grid + IR transformer
        self.ir = IR_factory(c.beta, c.ir_file)
        with h5py.File(c.ir_file, "r") as f:
            nb = f["/bose/ngrid"][()]
        self.omega = 2.0 * nb * np.pi / c.beta

        # Static screened polarization in the aux basis -> screening set (df_int).
        if c.screening == "qresolved":
            print("    Building q-resolved screening Pi(q) for all %d q ..." % self.nk)
            self.M_q = self._build_screening_all_q(stored_s, sym)   # I + Pi(q), (nk,NQ,NQ)
            self.Pi_stat = self.M_q[c.q_idx] - np.eye(self.NQ)      # at Q_exc (for reference)
            # MO occ-occ / virt-virt blocks at every k-pair (for W^{k-k'})
            SC = np.einsum('kmn,knp->kmp', rSk[0], self.vexMO[0])   # S_k C_k, (nk,nao,nmo)
            print("    Building MO DF blocks for all %d k-pairs ..." % (self.nk ** 2))
            self.Voo_all, self.Vvv_all = pint.build_VQ_mo_blocks(stored_s, sym, SC, self.occ)
        else:
            if c.q_idx == 0:
                VQ_kq_ao_s = pint.build_VQ_same_k(stored_s, sym)
            else:
                VQ_kq_ao_s = pint.build_VQ_offdiag(stored_s, sym, self.kq_map)
            self.Pi_stat = self._build_screening_one_q(VQ_kq_ao_s, self.kq_map)
            self.M_q = None
        print("    NQ=%d, niw(bose)=%d" % (self.NQ, self.omega.shape[0]))

    def _build_screening_one_q(self, VQ_kq_ao, kq_map_q):
        """
        P0(q, tau) -> Dyson -> P~(q, iOmega); return the static (iw=0) NQ x NQ
        matrix.  Spin-restricted (ns=1) carries the factor-2 spin sum, matching
        gwtool.eval_P0_tilde_Q (the molecular two-step convention).
        """
        G = self.G_tau                                    # (nt, ns, nk, nao, nao)
        spin = 2.0 if G.shape[1] == 1 else 1.0
        P0_tau = spin * pol.compute_P0_tau(G, VQ_kq_ao, kq_map_q)    # (nt,ns,1,NQ,NQ)
        P0_tau = pol.symmetrize_P0(P0_tau)
        P_iw = pol.dyson_equation(self.ir.tauf_to_wb(P0_tau))        # (niw,ns,1,NQ,NQ)
        return P_iw[P_iw.shape[0] // 2, 0, 0]

    def _build_screening_all_q(self, stored, sym):
        """I + Pi_stat(q) for every momentum transfer q on the BvK mesh -> (nk,NQ,NQ)."""
        I = np.eye(self.NQ, dtype=np.complex128)
        M_q = np.empty((self.nk, self.NQ, self.NQ), dtype=np.complex128)
        for q in range(self.nk):
            kq_map_q = self.kmaps["ksum"][:, q]                     # k + q
            VQ_kq_ao = pint.build_VQ_offdiag(stored, sym, kq_map_q)
            M_q[q] = I + self._build_screening_one_q(VQ_kq_ao, kq_map_q)
        return M_q

    # --- 4. excitations per active space ----------------------------------
    def solve_excitations(self):
        c = self.config
        q_is_zero = (c.q_idx == 0)
        if c.active_mode == "auto":
            label, ao, av = auto_active_space(self.occ, self.nao, self.NQ)
            self.active_spaces = [(label, ao, av)]
            print("Auto active space: occ=%s, virt=%s" % (ao, av))
        else:
            self.active_spaces = default_active_spaces(self.occ, self.nao)

        self.results_exc = {}
        print("\n" + "=" * 78)
        print("Exciton momentum q_idx=%d (%s); channel=%s; screening=%s"
              % (c.q_idx, "optical Q=0" if q_is_zero else "finite Q",
                 c.channel, c.screening))
        print("%-22s %-52s" % ("active space",
                               "lowest excitations (eV)"))
        print("-" * 78)
        for label, ao, av in self.active_spaces:
            K_W = None
            if c.screening == "qresolved":
                K_W = pabse.screened_direct_kernel_qresolved(
                    self.Voo_all, self.Vvv_all, self.M_q, self.kq_map,
                    self.kmaps["kdiff"], ao, av, self.occ, self.nk)
            P_irr, P_bse = pabse.two_step_bse_periodic(
                self.VQ_mo_kk, self.VQ_ia, self.mo_energy, self.Pi_stat,
                self.kq_map, self.occ, self.nao, ao, av, self.omega,
                channel=c.channel, q_is_zero=q_is_zero, K_W=K_W)
            # singlet (+2v exchange, all Q) uses P_bse; triplet uses P_irr
            P_use = P_bse if c.channel == "singlet" else P_irr
            exc = pabse.extract_excitations(P_use, self.omega,
                                            w_max=c.w_max, method=c.method)
            self.results_exc[label] = exc
            e = [f"{ev:.3f}" for ev, _ in exc[:6]]
            print("%-22s %-52s" % (label, " ".join(e)))
        print("=" * 78 + "\n")

    # --- 5. spectral function plot ----------------------------------------
    def plot_spectrum(self):
        import matplotlib.pyplot as plt
        c = self.config
        grid = np.linspace(0.0, c.grid_max, 3000)
        nrows = len(self.active_spaces)
        fig, axes = plt.subplots(nrows, 1, figsize=(7, 1.6 * nrows + 1),
                                 sharex=True, squeeze=False)
        for row, (label, _, _) in enumerate(self.active_spaces):
            pabse.plot_spectrum(self.results_exc[label], ax=axes[row, 0],
                                omega=grid, eta=c.eta,
                                label="Active space = " + label)
            axes[row, 0].set_xlim(0, c.grid_max)
            axes[row, 0].legend(fontsize=8)
        tag = "optical Q=0" if c.q_idx == 0 else "q_idx=%d" % c.q_idx
        fig.suptitle("Periodic two-step active-space BSE (%s, %s) — QP on the fly"
                     % (c.channel, tag), fontsize=12)
        fig.tight_layout()
        fig.savefig(c.output_file, bbox_inches="tight")
        print("Saved spectrum -> %s" % c.output_file)

    # --- driver -----------------------------------------------------------
    def run(self):
        t0 = time.time()
        self.load_input_data()
        self.solve_quasiparticles()
        self.build_intermediates()
        self.solve_excitations()
        self.plot_spectrum()
        print("Done in %.2f s." % (time.time() - t0))


# ===========================================================================
# CLI
# ===========================================================================
def create_argument_parser():
    p = argparse.ArgumentParser(
        description="Periodic finite-Q two-step active-space BSE spectrum from "
                    "a periodic scGW output (QP energies generated on the fly).")
    p.add_argument("--input", type=str, default="mean_field_input.h5",
                   help="Mean-field input HDF5 (HF/S-k, Fock-k, H-k, params).")
    p.add_argument("--sim", type=str, default="scGW_sim.h5",
                   help="Periodic scGW output HDF5 (Sigma1, Selfenergy, G_tau, mu).")
    p.add_argument("--int_path", type=str, default="df_hf_int/",
                   help="Folder with the raw green-mbpt k-pair integrals (meta.h5, VQ_*.h5). "
                        "Used for the bare-Coulomb exchange / bubble vertices.")
    p.add_argument("--screening_int_path", type=str, default=None,
                   help="Folder with the Ewald-corrected DF integrals (df_int) for the "
                        "screening Pi and screened-direct W. Default: auto-detect a sibling "
                        "'df_int' next to --int_path, else reuse --int_path.")
    p.add_argument("--ir_file", type=str, default=None,
                   help="IR-grid HDF5 file (required).")
    p.add_argument("--output", type=str, default="periodic_two_step_bse.pdf",
                   help="Output spectrum PDF path.")
    p.add_argument("--q_idx", type=int, default=0,
                   help="Exciton momentum index on the BvK k-mesh (0 = optical Q=0).")
    p.add_argument("--beta", type=float, default=1000.0, help="Inverse temperature (a.u.).")
    p.add_argument("--iter", type=int, default=-1, help="scGW iteration (-1 = last).")
    p.add_argument("--qpac", type=int, default=1,
                   help="QP-correct energies from Sigma(iw)? 1=yes (default), 0=static GW.")
    p.add_argument("--channel", type=str, default="singlet",
                   choices=["singlet", "triplet"], help="Spin channel.")
    p.add_argument("--screening", type=str, default="qresolved",
                   choices=["qresolved", "static"],
                   help="Direct-kernel screening: fully q-resolved W^{k-k'} "
                        "(default) or a single static Pi(Q_exc).")
    p.add_argument("--eta", type=float, default=0.05, help="Lorentzian broadening (eV).")
    p.add_argument("--w_max", type=float, default=1.0, help="Pole window (a.u.) for extraction.")
    p.add_argument("--grid_max", type=float, default=25.0, help="Plot upper bound (eV).")
    p.add_argument("--method", type=str, default="aaa", help="Pole extractor: aaa|plaspole|hybrid.")
    p.add_argument("--active", type=str, default="series", choices=["series", "auto"],
                   help="Active-space selection: nested 'series' (default) or single 'auto'.")
    return p
