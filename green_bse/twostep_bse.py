#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Two-step active-space BSE solver (RPA-scaling), driven from scGW output. #
#                                                                             #
#    Excitations are the poles of the Q-space polarization built by           #
#    active_bse.two_step_bse:                                                  #
#       P^act  : screened -W ladder in a small active space                   #
#       P^irr  : P^act - P^act(0) + P^(0)                                      #
#       P^BSE  : [I - 2 P^irr]^{-1} P^irr   (singlet)                         #
#    A spectral function A(w) is assembled from the extracted poles.          #
#                                                                             #
#    Quasiparticle energies/eigenvectors are generated ON THE FLY from the    #
#    scGW output (NO bse_singlet.h5 input):                                   #
#       * solve  F C = S C E  with the static GW Fock  F = Sigma1 + H_core    #
#         (casidaEq.solveMO)  ->  MO eigenvectors + static-GW energies;       #
#       * Pade-continue the dynamic self-energy and solve the QP equation     #
#         (qp.padeSigma)      ->  QP energies.                                 #
#    This mirrors bse.BSESolver.solve_molecular_orbitals.                     #
#                                                                             #

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import h5py

import gwtool
import casidaEq as casida
import contract as ct
import qp
from irFT import IR_factory
from polarization import dyson_equation
import active_bse as abse

AU2EV = 27.211386245981


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class TwoStepBSEConfig:
    """Parameters for the two-step active-space BSE spectrum run."""
    input_file: str = "mean_field_input.h5"      # scGW mean-field input
    sim_file: str = "scGW_sim.h5"                # scGW output (G, Sigma)
    int_path: str = "df_hf_int/"                 # folder with VQ_0.h5
    ir_file: Optional[str] = None                # IR grid HDF5
    output_file: str = "two_step_bse_spectrum.pdf"

    beta: float = 1000.0
    iteration: int = -1                          # scGW iteration (-1 = last)
    qpac_enabled: bool = True                    # QP correction from Sigma(iw)

    # spectrum / extraction
    eta: float = 0.05                            # Lorentzian broadening (eV)
    w_max: float = 1.0                           # pole window (a.u.) for AAA
    grid_max: float = 25.0                       # plot range (eV)
    method: str = "aaa"
    active_mode: str = "series"                  # "series" (nested) or "auto"

    @classmethod
    def from_args(cls, args):
        return cls(
            input_file=args.input, sim_file=args.sim, int_path=args.int_path,
            ir_file=args.ir_file, output_file=args.output, beta=args.beta,
            iteration=args.iter, qpac_enabled=bool(args.qpac), eta=args.eta,
            w_max=args.w_max, grid_max=args.grid_max, method=args.method,
            active_mode=args.active,
        )


# ===========================================================================
# Active-space schedule (nested convergence series toward the Fermi level)
# ===========================================================================
def default_active_spaces(occ, nao):
    """A few nested (occ|virt) windows shrinking toward the gap."""
    virt = nao - occ
    spaces = [("[%d..%d | %d..%d]" % (0, occ - 1, occ, nao - 1),
               list(range(occ)), list(range(occ, nao)))]
    seen = {(tuple(range(occ)), tuple(range(occ, nao)))}
    for no, nv in [(5, 3), (3, 3), (2, 2), (1, 1)]:
        no, nv = min(no, occ), min(nv, virt)
        ao, av = list(range(occ - no, occ)), list(range(occ, occ + nv))
        key = (tuple(ao), tuple(av))
        if key in seen:
            continue
        seen.add(key)
        label = "[%d..%d | %d..%d]" % (ao[0], ao[-1], av[0], av[-1])
        spaces.append((label, ao, av))
    return spaces


def auto_active_space(occ, nao, NQ):
    """
    Recommend a single active space, sized so the particle-hole ladder fits the
    Q-space (the -W ladder is an (n_occ*n_virt) object, kept <= NQ):

      (1) maximise  n_occ * n_virt  subject to  n_occ * n_virt <= NQ
          (and n_occ <= occ, n_virt <= nao - occ);
      (2) HOMO index = occ - 1,  LUMO index = occ;
      (3) take the n_occ highest occupied (down from HOMO) and the n_virt
          lowest virtual (up from LUMO).

    Product ties are broken toward a balanced window (smallest |n_occ - n_virt|).

    Returns (label, act_occ, act_virt).
    """
    n_occ_avail, n_virt_avail = occ, nao - occ
    best = None                       # (product, -|no-nv|, no, nv), maximised
    for no in range(1, n_occ_avail + 1):
        for nv in range(1, n_virt_avail + 1):
            if no * nv > NQ:
                continue
            cand = (no * nv, -abs(no - nv), no, nv)
            if best is None or cand > best:
                best = cand
    if best is None:                  # NQ < 1 (never in practice): minimal window
        no = nv = 1
    else:
        _, _, no, nv = best
    act_occ = list(range(occ - no, occ))       # (2)-(3): n_occ highest occupied
    act_virt = list(range(occ, occ + nv))      #          n_virt lowest virtual
    label = "[%d..%d | %d..%d]" % (act_occ[0], act_occ[-1], act_virt[0], act_virt[-1])
    return label, act_occ, act_virt


# ===========================================================================
# Solver
# ===========================================================================
class TwoStepBSESolver:
    def __init__(self, config: TwoStepBSEConfig):
        self.config = config
        if config.ir_file is None:
            raise ValueError("ir_file is required (path to the IR-grid HDF5).")
        self.results = {}

    # --- 1. read scGW input/output ----------------------------------------
    def load_input_data(self):
        c = self.config
        print("Reading input/sim files ...")
        with h5py.File(c.input_file, "r") as f:
            rSk = f["/HF/S-k"][()].view(complex);   rSk = rSk.reshape(rSk.shape[:-1])
            rFk_in = f["/HF/Fock-k"][()].view(complex); rFk_in = rFk_in.reshape(rFk_in.shape[:-1])
            rHk = f["/HF/H-k"][()].view(complex);   rHk = rHk.reshape(rHk.shape[:-1])
            self.nao = int(f["/params/nao"][()])
            self.nelec = int(f["/params/nel_cell"][()])
        self.occ = self.nelec // 2

        with h5py.File(c.sim_file, "r") as f:
            it = int(f["iter"][()]) if c.iteration == -1 else c.iteration
            if it == 1:                               # G0W0: HF reference Fock
                rFk = rFk_in
            else:
                rFk = f["iter%d/Sigma1" % it][()].view(complex) + rHk
            rSigmak = f["iter%d/Selfenergy/data" % it][()].view(complex)
            mu = f["iter%d/mu" % it][()]

        self.results.update(rSk=rSk, rFk=rFk, rSigmak=rSigmak, mu=mu, iter=it)
        print("    iter=%d, nao=%d, occ=%d" % (it, self.nao, self.occ))

    # --- 2. QP energies + eigenvectors ON THE FLY -------------------------
    def solve_quasiparticles(self):
        c = self.config
        rFk, rSk = self.results["rFk"], self.results["rSk"]

        # static GW Fock eigenproblem  F C = S C E
        valsMO, vexMO = casida.solveMO(rFk, rSk)
        mo_coeff = np.zeros((1, 1, self.nao, self.nao), dtype=np.complex128)
        mo_coeff[0, 0] = vexMO[0, 0]
        mo_coeff_adj = np.einsum("skpq -> skqp", mo_coeff.conj())

        if c.qpac_enabled:
            print("Generating QP energies from Sigma(iw) (Pade + QP equation) ...")
            Sigma_mo = np.einsum("skab, tskbc, skcd -> tskad",
                                 mo_coeff_adj, self.results["rSigmak"], mo_coeff,
                                 optimize=True)
            valsMO = qp.padeSigma(Sigma_mo, valsMO, c.beta, self.results["mu"], c.ir_file)
        else:
            print("QP correction disabled: using static-GW Fock eigenvalues.")

        self.mo_energy = valsMO[0, 0].real
        self.mo_coeff = vexMO[0, 0]
        homo, lumo = self.mo_energy[self.occ - 1], self.mo_energy[self.occ]
        print("    HOMO %.4f eV, LUMO %.4f eV, gap %.4f eV"
              % (homo * AU2EV, lumo * AU2EV, (lumo - homo) * AU2EV))

    # --- 3. DF integrals + screening + Matsubara grid ---------------------
    def build_intermediates(self):
        c = self.config
        VQ_ao = ct.readVQ(c.int_path + "VQ_0.h5")
        self.NQ = VQ_ao.shape[1]
        self.VQ_mo = casida.VQ_ao2mo(VQ_ao, self.mo_coeff)[0]      # (NQ, nmo, nmo), C-only

        self.ir = IR_factory(c.beta, c.ir_file)
        P0t = gwtool.symmetrize_P0(
            gwtool.eval_P0_tilde_Q(self.results["iter"], self.nao, self.NQ,
                                   int_path=c.int_path, sim_h5=c.sim_file))
        P0w = self.ir.tauf_to_wb(P0t)
        niw = P0w.shape[0]
        self.tildeP_static = dyson_equation(P0w)[niw // 2, 0, 0]
        _, wb, *_ = self.ir.read_IR_matrices(c.ir_file, c.beta, ptype="bose")
        self.omega = wb.real
        print("    NQ=%d, niw=%d" % (self.NQ, niw))

    # --- 4. excitations per active space ----------------------------------
    def solve_excitations(self):
        c = self.config
        if c.active_mode == "auto":
            label, ao, av = auto_active_space(self.occ, self.nao, self.NQ)
            print("Auto active space: occ=%s, virt=%s  (n_occ*n_virt=%d <= NQ=%d; "
                  "HOMO=%d, LUMO=%d)"
                  % (ao, av, len(ao) * len(av), self.NQ, self.occ - 1, self.occ))
            self.active_spaces = [(label, ao, av)]
        else:
            self.active_spaces = default_active_spaces(self.occ, self.nao)
        self.results_exc = {}
        print("\n" + "=" * 78)
        print("%-20s %-28s %-28s" % ("active space", "singlet (eV)", "triplet (eV)"))
        print("-" * 78)
        for label, ao, av in self.active_spaces:
            P_irr, P_bse = abse.two_step_bse(self.VQ_mo, self.mo_energy,
                                             self.tildeP_static, self.occ, ao, av,
                                             self.omega, channel="singlet")
            singlet = abse.extract_excitations(P_bse, self.omega, w_max=c.w_max, method=c.method)
            triplet = abse.extract_excitations(P_irr, self.omega, w_max=c.w_max, method=c.method)
            self.results_exc[label] = (singlet, triplet)
            s = [f"{e:.3f}" for e, _ in singlet[:4]]
            t = [f"{e:.3f}" for e, _ in triplet[:4]]
            print("%-20s %-28s %-28s" % (label, " ".join(s), " ".join(t)))
        print("=" * 78)
        print("reference (full-active limit): TDA-BSE singlet 9.393 eV, full-BSE 9.294 eV\n")

    # --- 5. spectral function plot ----------------------------------------
    def plot_spectrum(self):
        import matplotlib.pyplot as plt
        c = self.config
        grid = np.linspace(0.0, c.grid_max, 3000)
        nrows = len(self.active_spaces)
        fig, axes = plt.subplots(nrows, 2, figsize=(10, 1.6 * nrows + 1), sharey=True)
        axes = np.atleast_2d(axes)
        for row, (label, _, _) in enumerate(self.active_spaces):
            singlet, triplet = self.results_exc[label]
            abse.plot_spectrum(singlet, ax=axes[row, 0], omega=grid, eta=c.eta,
                               label="Active space = " + label)
            abse.plot_spectrum(triplet, ax=axes[row, 1], omega=grid, eta=c.eta,
                               label="Active space = " + label)
        for ax in axes[:, 0]:
            ax.axvline(9.393, color="0.6", ls="--", lw=1.0, label="BSE (TDA) 1st singlet 9.393 eV")
        for col in range(2):
            for ax in axes[:, col]:
                ax.set_xlim(0, c.grid_max)
                ax.legend(fontsize=8)
        axes[0, 0].set_title("singlet  $P$")
        axes[0, 1].set_title("triplet  $P$")
        fig.suptitle("Two-step active-space BSE (BSE@scGW) — QP input generated on the fly\n"
                     "(AAA analytic continuation; no bse_singlet.h5)", fontsize=12)
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
        description="Two-step active-space BSE spectrum from scGW output "
                    "(QP energies generated on the fly; no bse_singlet.h5).")
    p.add_argument("--input", type=str, default="mean_field_input.h5",
                   help="scGW mean-field input HDF5 (HF/S-k, Fock-k, H-k, params).")
    p.add_argument("--sim", type=str, default="scGW_sim.h5",
                   help="scGW output HDF5 (Sigma1, Selfenergy, G_tau, mu).")
    p.add_argument("--int_path", type=str, default="df_hf_int/",
                   help="Folder containing VQ_0.h5 (density-fitting integrals).")
    p.add_argument("--ir_file", type=str, default=None,
                   help="IR-grid HDF5 file (required).")
    p.add_argument("--output", type=str, default="two_step_bse_spectrum.pdf",
                   help="Output spectrum PDF path.")
    p.add_argument("--beta", type=float, default=1000.0, help="Inverse temperature (a.u.).")
    p.add_argument("--iter", type=int, default=-1, help="scGW iteration (-1 = last).")
    p.add_argument("--qpac", type=int, default=1,
                   help="Generate QP-corrected energies from Sigma(iw)? 1=yes (default), "
                        "0=use static-GW Fock eigenvalues.")
    p.add_argument("--eta", type=float, default=0.05, help="Lorentzian broadening (eV).")
    p.add_argument("--w_max", type=float, default=1.0, help="Pole window (a.u.) for extraction.")
    p.add_argument("--grid_max", type=float, default=25.0, help="Plot upper bound (eV).")
    p.add_argument("--method", type=str, default="aaa", help="Pole extractor: aaa|plaspole|hybrid.")
    p.add_argument("--active", type=str, default="series", choices=["series", "auto"],
                   help="Active-space selection. 'series' (default): nested convergence "
                        "series toward the gap. 'auto': a single window around HOMO/LUMO "
                        "sized so n_occ*n_virt is maximal but <= NQ.")
    return p
