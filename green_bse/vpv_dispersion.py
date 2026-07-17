#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    vpv_dispersion.py                                                        #
#    =================                                                        #
#    General solver for the VPV-diagonal exciton-dispersion pipeline          #
#    (generalization of example/*/test_vpv_diag_interp.py):                   #
#                                                                             #
#      1. collect the k-traced occ/virt VPV diagonal (and the FULL static     #
#         pair-space matrix, for the off-diagonal report) at every exciton    #
#         momentum on the BvK mesh (cached),                                  #
#      2. validate (node-exactness, static-frame reality, off-diagonal        #
#         weight of VPV(Omega ~ 0)),                                          #
#      3. Wannier-interpolate the diagonal to a dense high-symmetry path,     #
#      4. continue to the real axis per (i,a) pair ("aaa" | "plaspole" |      #
#         "nevanlinna"), overlaying exact-grid poles at on-mesh points,       #
#      5. save crucial results to HDF5, print an error/diagnostics report,    #
#         and (optionally) render the 4-panel figure.                         #
#                                                                             #
#    CLI:  python vpv_dispersion.py --input ... --sim ... (see --help);       #
#    or import { VPVDispersionConfig, VPVDispersionSolver } and drive it      #
#    programmatically.                                                        #
#                                                                             #

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import h5py

sys.path.append(str(Path(__file__).resolve().parent))

import periodic_twostep_bse as ptb
import periodic_integrals as pint
import wannier_exciton as wx

AU2EV = 27.211386245981

HEX_PATH = [(r"$\Gamma$", (0.0, 0.0, 0.0)), ("M", (0.5, 0.0, 0.0)),
            ("K", (1.0 / 3.0, 1.0 / 3.0, 0.0)), (r"$\Gamma$", (0.0, 0.0, 0.0))]


@dataclass
class VPVDispersionConfig:
    input_file: str
    sim_file: str
    int_path: str
    ir_file: str
    beta: float = 1000.0
    channel: str = "singlet"
    cont_method: str = "nevanlinna"  # "aaa" | "plaspole" | "nevanlinna"
    eta_eV: float = 0.5
    e_min: float = 0.0
    e_max: float = None           # None -> auto from the QP pair-energy range
    e_npts: int = 700
    n_act_occ: int = None         # None -> full occupied space
    n_act_vir: int = None         # None -> full virtual space
    corners: list = field(default_factory=lambda: list(HEX_PATH))
    npts: int = 60                # path points per segment
    dim: int = 2
    cache_file: str = "vpv_mesh_cache.npz"
    out_file: str = "vpv_dispersion.h5"
    fig_base: str = "vpv_dispersion"   # None -> skip the figure
    offdiag: bool = True          # evaluate off-diagonal weight of VPV(0)
    observable: str = "trP"       # "trP" : basis-free exciton DOS (Tr P) --
                                  #   the trustworthy spectral reference (the
                                  #   VPV maps can smear head-channel weight
                                  #   into the gap; see the error report's
                                  #   below-onset metric)
                                  # "vpv" : channel-resolved VPV diagonal
                                  # "diagP": aux-basis diagonal of P --
                                  #   fixed atom-centered channel labels
                                  #   (crossing-free), per-channel Herglotz,
                                  #   sums to Tr P; NQ channels, so the
                                  #   continuation is ~NQ/(no*nv) x slower
    channel_basis: str = "eigen"  # "ov"    : energy-ordered (i,a) band pairs
                                  # "eigen" : eigenvectors of the static full
                                  #   pair matrix VPV(Omega~0) per mesh Q --
                                  #   the diagonal approximation is then
                                  #   EXACT at the static frame; channels are
                                  #   ordered strongest (most negative) first


class VPVDispersionSolver:
    """Orchestrates the collect / validate / interpolate / continue pipeline.

    All diagnostics are accumulated in self.errors (printed by run() and
    stored under /errors in the output HDF5).
    """

    def __init__(self, config):
        self.c = config
        self.errors = {}

    # ------------------------------------------------------------------ #
    #  setup + mesh sweep                                                 #
    # ------------------------------------------------------------------ #
    def setup(self):
        c = self.c
        cfg = ptb.PeriodicTwoStepBSEConfig(
            input_file=c.input_file, sim_file=c.sim_file, int_path=c.int_path,
            ir_file=c.ir_file, method="plaspole", beta=c.beta, q_idx=0,
            screening="qresolved", active_mode="series")
        s = ptb.PeriodicTwoStepBSESolver(cfg)
        s.load_input_data()
        s.solve_quasiparticles()
        s.build_intermediates()
        self.s = s

        self.stored = pint.load_stored_integrals(c.int_path)
        self.sym = pint.load_kpair_symmetry(c.input_file)

        n_occ = s.occ if c.n_act_occ is None else min(c.n_act_occ, s.occ)
        n_vir = (s.nao - s.occ if c.n_act_vir is None
                 else min(c.n_act_vir, s.nao - s.occ))
        self.ao = list(range(s.occ - n_occ, s.occ))
        self.av = list(range(s.occ, s.occ + n_vir))
        self.no, self.nv = len(self.ao), len(self.av)
        print("Active ladder space: %d occ x %d virt bands -> %d (i,a) pairs"
              % (self.no, self.nv, self.no * self.nv))
        self.basis = c.channel_basis if c.observable == "vpv" else "ov"
        if c.observable != "vpv" and c.channel_basis == "eigen":
            print("observable='%s' is basis-free; channel_basis is ignored"
                  % c.observable)

    def collect_mesh(self):
        """VPV diagonal (all frequencies) + full static pair matrix per mesh Q.

        Cached in c.cache_file; delete the cache after changing the solver
        setup.  An older cache without the static matrices is accepted (the
        off-diagonal report is then skipped with a warning).
        """
        c, s = self.c, self.s
        self.iw0 = int(np.argmin(np.abs(s.omega)))
        self.eigvals = self.eigvecs = None
        cache = Path(c.cache_file)
        if cache.exists():
            print("Loading VPV mesh from %s ..." % cache)
            dat = np.load(cache)
            basis = str(dat["channel_basis"]) if "channel_basis" in dat.files \
                else "ov"
            if basis != self.basis:
                raise ValueError(
                    "cache %s holds channel_basis='%s' but config asks '%s'; "
                    "use a different --cache file" %
                    (cache, basis, self.basis))
            self.VPV_mesh, self.q_frac = dat["VPV_mesh"], dat["q_frac"]
            if self.VPV_mesh.shape[2:] != (self.no, self.nv):
                raise ValueError(
                    "cache %s holds a %dx%d active space but config asks "
                    "%dx%d; use a different --cache file"
                    % ((cache,) + self.VPV_mesh.shape[2:]
                       + (self.no, self.nv)))
            self.M_stat = dat["M_stat"] if "M_stat" in dat.files else None
            if "eigvals" in dat.files:
                self.eigvals, self.eigvecs = dat["eigvals"], dat["eigvecs"]
            self.trP_mesh = dat["trP_mesh"] if "trP_mesh" in dat.files \
                else None
            self.diagP_mesh = dat["diagP_mesh"] if "diagP_mesh" in dat.files \
                else None
            if c.observable == "trP" and self.trP_mesh is None:
                raise ValueError(
                    "cache %s predates the trP observable; delete it to "
                    "regenerate" % cache)
            if c.observable == "diagP" and self.diagP_mesh is None:
                raise ValueError(
                    "cache %s predates the diagP observable; delete it to "
                    "regenerate" % cache)
            if self.M_stat is None and c.offdiag:
                print("  WARNING: cache has no static matrices; off-diagonal "
                      "report skipped (delete the cache to regenerate).")
            self._select_field()
            return

        with h5py.File(c.input_file, "r") as f:
            kfrac = f["symmetry/k/mesh_scaled"][:]
        npair = self.no * self.nv
        niw = s.omega.shape[0]
        eigen = self.basis == "eigen"
        self.VPV_mesh = np.empty((s.nk, niw, self.no, self.nv),
                                 dtype=np.complex128)
        self.trP_mesh = np.empty((s.nk, niw), dtype=np.complex128)
        self.diagP_mesh = np.empty((s.nk, niw, s.NQ), dtype=np.complex128)
        self.M_stat = (np.empty((s.nk, npair, npair), dtype=np.complex128)
                       if (c.offdiag or eigen) else None)
        if eigen:
            self.eigvals = np.empty((s.nk, npair))
            self.eigvecs = np.empty((s.nk, npair, npair),
                                    dtype=np.complex128)
        qf = []
        print("Building Tr P + VPV %s-channel diag over the full "
              "%d-point Q-mesh ..." % (self.basis, s.nk))
        for q in range(s.nk):
            print("  building P at Q-mesh point %3d/%d" % (q + 1, s.nk))
            P, VQ_ia = wx._P_and_VQia_at_q(s, self.stored, self.sym, q,
                                           self.ao, self.av, c.channel)
            self.trP_mesh[q] = np.einsum("wqq->w", P)
            self.diagP_mesh[q] = np.einsum("wqq->wq", P)
            if eigen:
                # full-frequency pair matrix; channels = eigenvectors of the
                # Hermitized static frame, strongest (most negative) first
                Miw = wx.vpv_matrix_from_P(P, VQ_ia, self.ao, self.av)
                Mst = 0.5 * (Miw[self.iw0] + Miw[self.iw0].conj().T)
                lam, U = np.linalg.eigh(Mst)          # ascending eigenvalues
                self.eigvals[q], self.eigvecs[q] = lam, U
                diag = np.einsum("pr,wpq,qr->wr", U.conj(), Miw, U,
                                 optimize=True)       # (niw, npair)
                self.VPV_mesh[q] = diag.reshape(niw, self.no, self.nv)
                if self.M_stat is not None:
                    self.M_stat[q] = Mst
            else:
                self.VPV_mesh[q] = wx.vpv_diag_from_P(P, VQ_ia,
                                                      self.ao, self.av)
                if self.M_stat is not None:
                    self.M_stat[q] = wx.vpv_static_matrix_from_P(
                        P[self.iw0], VQ_ia, self.ao, self.av)
            qf.append(kfrac[q])
        self.q_frac = np.asarray(qf)
        kw = dict(VPV_mesh=self.VPV_mesh, q_frac=self.q_frac,
                  channel_basis=self.basis, trP_mesh=self.trP_mesh,
                  diagP_mesh=self.diagP_mesh)
        if self.M_stat is not None:
            kw["M_stat"] = self.M_stat
        if eigen:
            kw.update(eigvals=self.eigvals, eigvecs=self.eigvecs)
        np.savez(cache, **kw)
        self._select_field()

    def _select_field(self):
        """Point self.VPV_mesh at the observable to interpolate/continue;
        the raw VPV channel data stays in self.vpv_raw for the h5."""
        self.vpv_raw = self.VPV_mesh
        if self.c.observable == "trP":
            self.VPV_mesh = self.trP_mesh[:, :, None, None]
        elif self.c.observable == "diagP":
            self.VPV_mesh = self.diagP_mesh[:, :, :, None]

    # ------------------------------------------------------------------ #
    #  validation / diagnostics                                           #
    # ------------------------------------------------------------------ #
    def validate(self):
        c, s = self.c, self.s
        err = wx.validate_VPV_node_exactness(self.VPV_mesh, self.q_frac,
                                             dim=c.dim)
        self.errors["node_exactness"] = err
        self.errors["static_imag_max"] = float(
            np.max(np.abs(self.VPV_mesh[:, self.iw0].imag)))

        if self.M_stat is not None:
            # (i) how large are the off-diagonal elements of VPV(Omega ~ 0)?
            rmax, rfro = zip(*(wx.offdiag_ratios(M) for M in self.M_stat))
            self.offdiag_max = np.asarray(rmax)     # per mesh Q
            self.offdiag_fro = np.asarray(rfro)
            self.errors["offdiag_max_worst"] = float(self.offdiag_max.max())
            self.errors["offdiag_max_median"] = float(
                np.median(self.offdiag_max))
            self.errors["offdiag_fro_worst"] = float(self.offdiag_fro.max())
        else:
            self.offdiag_max = self.offdiag_fro = None

    # ------------------------------------------------------------------ #
    #  path, interpolation, continuation                                  #
    # ------------------------------------------------------------------ #
    def build_path(self):
        c = self.c
        segs, bounds = [], [0]
        pts = [np.asarray(p, dtype=float) for _, p in c.corners]
        for a, b in zip(pts[:-1], pts[1:]):
            segs.append(np.linspace(a, b, c.npts, endpoint=False))
            bounds.append(bounds[-1] + c.npts)
        segs.append(pts[-1][None, :])
        bounds[-1] += 1
        self.path_frac, self.bounds = np.vstack(segs), bounds

        with h5py.File(c.input_file, "r") as f:
            kfrac = f["symmetry/k/mesh_scaled"][:]
            kcart = f["symmetry/k/mesh"][:]
        B, *_ = np.linalg.lstsq(kfrac, kcart, rcond=None)
        pc = self.path_frac @ B
        self.xp = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=1))))

        # data-driven match of path points to mesh Q (periodic distance)
        self.matches = []
        for n, fr in enumerate(self.path_frac):
            df = self.q_frac - fr
            df -= np.round(df)
            d = np.linalg.norm(df, axis=1)
            i = int(np.argmin(d))
            if d[i] < 1e-6:
                self.matches.append((n, i))

    def interpolate(self):
        print("Interpolating VPV diag to %d path points ..."
              % len(self.path_frac))
        self.VPV_path = wx.interpolate_VPV_diag_over_Q(
            self.VPV_mesh, self.q_frac, self.path_frac, dim=self.c.dim)

    def continue_to_real_axis(self):
        c, s = self.c, self.s
        if c.e_max is None:
            d_max = (s.mo_energy[:, s.occ:].max()
                     - s.mo_energy[:, :s.occ].min())
            e_max = d_max * AU2EV + 5.0
        else:
            e_max = c.e_max
        self.egrid = np.linspace(c.e_min, e_max, c.e_npts)

        self.vpv_cont = None      # complex real-axis field (aaa / plaspole)
        self.A_signed = None      # signed spectral map (nevanlinna)
        self.pp_fit = None
        if c.cont_method == "plaspole":
            print("One-plasmon-pole continuing each (i,a) pair ...")
            self.vpv_cont, self.pp_fit = wx.pp_continue_vpv_pairs(
                self.VPV_path, s.omega, self.egrid, eta_eV=c.eta_eV)
            self.vpv_cont = self.vpv_cont * AU2EV
            r = self.pp_fit["resid"]
            self.errors["pp_resid_median"] = float(np.nanmedian(r))
            self.errors["pp_resid_worst"] = float(np.nanmax(r))
        elif c.cont_method == "nevanlinna":
            print("Nevanlinna (bosonic->fermionic aux) per (i,a) pair ...")
            self.A_signed = wx.nevan_continue_vpv_pairs(
                self.VPV_path, s.ir, self.egrid, eta_eV=c.eta_eV)
            self.errors["nevan_A_min"] = float(self.A_signed.min())
            self.errors["nevan_neg_fraction"] = float(np.mean(
                self.A_signed < -1e-3 * self.A_signed.max()))
        else:
            print("AAA-continuing each (i,a) pair ...")
            self.vpv_cont, n_fail = wx.aaa_continue_vpv_pairs(
                self.VPV_path, s.omega, self.egrid, eta_eV=c.eta_eV)
            self.vpv_cont = self.vpv_cont * AU2EV
            self.errors["aaa_failed_pair_fits"] = int(n_fail.sum())

        if self.A_signed is not None:
            self.Aw = np.maximum(self.A_signed, 0.0)
        else:
            self.Aw = np.nan_to_num(
                np.maximum(-self.vpv_cont.imag / np.pi, 0.0))

    def grid_poles(self):
        """Exact (non-interpolated) grid poles at on-mesh path points."""
        method = "plaspole" if self.c.cont_method == "plaspole" else "aaa"
        self.pole_x, self.pole_e = [], []
        for n, i in self.matches:
            es = wx.vpv_pair_poles(self.VPV_mesh[i], self.s.omega,
                                   method=method, w_min_eV=self.egrid[0],
                                   w_max_eV=self.egrid[-1])
            self.pole_x += [self.xp[n]] * len(es)
            self.pole_e += list(es)
        if self.pole_e:
            onset = float(min(self.pole_e))
            self.errors["onset_eV"] = onset
            mask = self.egrid < onset - 0.5
            tot = self.Aw.sum()
            self.errors["below_onset_weight"] = float(
                self.Aw[mask].sum() / tot) if tot > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  output                                                             #
    # ------------------------------------------------------------------ #
    def save_h5(self):
        """(ii) crucial results -> HDF5."""
        c = self.c
        with h5py.File(c.out_file, "w") as f:
            g = f.create_group("config")
            for k, v in vars(c).items():
                if k == "corners":
                    g["corner_labels"] = np.array(
                        [lb for lb, _ in v], dtype="S32")
                    g["corner_frac"] = np.array([p for _, p in v], dtype=float)
                elif v is not None:
                    g[k] = v
            f["mesh/q_frac"] = self.q_frac
            f["mesh/omega_bose"] = self.s.omega
            f["mesh/observable"] = self.VPV_mesh
            f["mesh/VPV_diag"] = self.vpv_raw
            if self.trP_mesh is not None:
                f["mesh/trP"] = self.trP_mesh
            if self.diagP_mesh is not None:
                f["mesh/diagP"] = self.diagP_mesh
            if self.M_stat is not None:
                f["mesh/VPV_static_full"] = self.M_stat
                f["mesh/offdiag_ratio_max"] = self.offdiag_max
                f["mesh/offdiag_ratio_fro"] = self.offdiag_fro
            f["path/frac"] = self.path_frac
            f["path/x"] = self.xp
            f["path/segment_bounds"] = np.asarray(self.bounds)
            f["path/VPV_diag_interp"] = self.VPV_path
            f["continuation/egrid_eV"] = self.egrid
            f["continuation/A"] = self.Aw
            if self.vpv_cont is not None:
                f["continuation/F_real_axis"] = self.vpv_cont
            if self.A_signed is not None:
                f["continuation/A_signed"] = self.A_signed
            if self.pp_fit is not None:
                for k, v in self.pp_fit.items():
                    f["continuation/pp_" + k] = v
            if self.eigvals is not None:
                f["mesh/eigen_channel_values"] = self.eigvals
                f["mesh/eigen_channel_vectors"] = self.eigvecs
            f["grid_poles/x"] = np.asarray(self.pole_x)
            f["grid_poles/energy_eV"] = np.asarray(self.pole_e)
            g = f.create_group("errors")
            for k, v in self.errors.items():
                g[k] = v
        print("Saved results -> %s" % c.out_file)

    def print_error_report(self):
        """(iii) error / diagnostics report (also stored under /errors)."""
        print("\n==================  error / diagnostics report  "
              "==================")
        print("  node-exactness of interpolation : %.3e   (expect ~1e-15; "
              "large -> Q-gauge problem)" % self.errors["node_exactness"])
        print("  max |Im VPV| on static frame    : %.3e   (expect ~0)"
              % self.errors["static_imag_max"])
        if "offdiag_max_worst" in self.errors:
            print("  off-diag of VPV(0), max-ratio   : worst %.3f, "
                  "median %.3f over mesh Q"
                  % (self.errors["offdiag_max_worst"],
                     self.errors["offdiag_max_median"]))
            print("  off-diag of VPV(0), fro-ratio   : worst %.3f   "
                  "(large -> diagonal approx. is lossy)"
                  % self.errors["offdiag_fro_worst"])
        if "onset_eV" in self.errors:
            print("  exact grid-pole onset           : %.2f eV"
                  % self.errors["onset_eV"])
            print("  map weight below onset          : %.3f   (~0.05-0.08 = "
                  "eta tails; >>0.1 -> in-gap contamination)"
                  % self.errors["below_onset_weight"])
        if "pp_resid_median" in self.errors:
            print("  plaspole per-pair fit residual  : median %.2e, "
                  "worst %.2e (large -> multi-mode channel)"
                  % (self.errors["pp_resid_median"],
                     self.errors["pp_resid_worst"]))
        if "aaa_failed_pair_fits" in self.errors:
            print("  AAA failed pair fits            : %d"
                  % self.errors["aaa_failed_pair_fits"])
        if "nevan_A_min" in self.errors:
            print("  Nevanlinna signed A_B minimum   : %.3f  "
                  "(negative = Pick violation of interpolated data)"
                  % self.errors["nevan_A_min"])
            print("  Nevanlinna negative fraction    : %.3f"
                  % self.errors["nevan_neg_fraction"])
        print("=================================================="
              "================\n")

    def plot(self):
        if self.c.fig_base is None:
            return
        import matplotlib.pyplot as plt
        c = self.c
        no, nv = self.VPV_mesh.shape[2], self.VPV_mesh.shape[3]
        trp = c.observable == "trP"
        diagp = c.observable == "diagP"
        eigen = c.observable == "vpv" and self.basis == "eigen"
        io_f, iv_f = (0, 0) if (eigen or trp or diagp) else (no - 1, 0)
        obs_tex = (r"Tr $P$" if trp else r"$\sum_Q$ $P_{QQ}$" if diagp
                   else r"$\sum_{ia}$ VPV$_{ia,ia}$")
        xp, egrid = self.xp, self.egrid
        iw0 = self.iw0
        vpv_stat = self.VPV_path[:, iw0].real.reshape(len(xp), -1) * AU2EV
        if diagp:                      # highlight the strongest aux channel
            io_f, iv_f = int(np.argmin(vpv_stat[0])), 0
        mesh_x = [xp[n] for n, _ in self.matches]
        mesh_v = [self.VPV_mesh[i, iw0, io_f, iv_f].real * AU2EV
                  for _, i in self.matches]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9.2, 8.0))

        cmap = plt.cm.viridis(np.linspace(0, 1, no * nv))
        for p in range(no * nv):
            ax1.plot(xp, vpv_stat[:, p], lw=0.7, color=cmap[p], alpha=0.6)
        ax1.plot(xp, vpv_stat[:, io_f * nv + iv_f], lw=1.8, color="crimson",
                 label=("Tr P (basis-free)" if trp else
                        "strongest aux channel" if diagp else
                        "strongest eigen-channel" if eigen
                        else "(i,a) = (HOMO, LUMO)"))
        ax1.scatter(mesh_x, mesh_v, s=26, facecolors="none",
                    edgecolors="black", lw=1.0, zorder=3,
                    label="exact mesh-Q (frontier)")
        ax1.set_ylabel(r"Re %s$(Q,\,i\Omega\!\approx\!0)$ (eV)"
                       % (r"Tr $P$" if trp else r"VPV$_{ia,ia}$"))
        ax1.set_title("Static frame: %s\n(BvK k-trace built in)"
                      % ("Tr P (basis-free)" if trp
                         else "one curve per aux channel (fixed labels)"
                         if diagp
                         else "one curve per eigen-channel" if eigen
                         else "one curve per (i, a) pair"))
        ax1.legend(fontsize=7, loc="best")

        order = np.argsort(self.s.omega)
        vpv_sum = self.VPV_path.reshape(len(xp), -1, no * nv).sum(axis=2)
        vpv_w = vpv_sum.real[:, order].T * AU2EV
        pcm = ax2.pcolormesh(xp, self.s.omega[order], vpv_w,
                             shading="gouraud", cmap="inferno")
        ax2.set_yscale("symlog", linthresh=1.0)
        ax2.set_ylabel(r"$\Omega_n$ (a.u.)")
        ax2.set_title(r"Re %s$(Q,\,i\Omega_n)$" % obs_tex)
        fig.colorbar(pcm, ax=ax2).set_label("eV")

        Aw = self.Aw
        pcm3 = ax3.pcolormesh(xp, egrid, Aw, shading="gouraud", cmap="inferno",
                              vmax=np.percentile(Aw[Aw > 0], 99)
                              if np.any(Aw > 0) else 1.0)
        if self.pp_fit is not None:
            wp_eV = self.pp_fit["wp"].reshape(len(xp), -1) * AU2EV
            resid = self.pp_fit["resid"].reshape(len(xp), -1)
            wp_eV = np.where(resid < 0.05, wp_eV, np.nan)
            ax3.plot(xp, wp_eV[:, 0], color="cyan", lw=0.5, ls="--", alpha=0.6,
                     label=r"per-pair $\omega_p(Q)$ (resid < 0.05)")
            ax3.plot(xp, wp_eV[:, 1:], color="cyan", lw=0.5, ls="--",
                     alpha=0.6)
        ax3.scatter(self.pole_x, self.pole_e, s=16, facecolors="none",
                    edgecolors="white", lw=0.7, zorder=4, clip_on=False,
                    label="exact grid poles")
        ax3.legend(fontsize=7, loc="best")
        ax3.set_ylabel(r"$\omega$ (eV)")
        ax3.set_title(r"$-\frac{1}{\pi}$ Im %s$(Q,\,\omega+i\eta)$"
                      "\n(%s, %s)"
                      % (obs_tex,
                         "exciton DOS" if trp
                         else "aux-site-resolved exciton DOS" if diagp
                         else "pair-weighted exciton DOS",
                         c.cont_method))
        fig.colorbar(pcm3, ax=ax3).set_label("eV")

        if self.A_signed is not None:
            rlim = np.percentile(np.abs(self.A_signed), 99.5) or 1.0
            pcm4 = ax4.pcolormesh(xp, egrid, self.A_signed, shading="gouraud",
                                  cmap="RdBu_r", vmin=-rlim, vmax=rlim)
            ax4.set_title(r"signed $A_B(Q,\,\omega)$, nevanlinna"
                          "\n(blue = Pick violation of interpolated data)")
        else:
            Rw = np.nan_to_num(self.vpv_cont.real)
            rlim = np.percentile(np.abs(Rw), 99) or 1.0
            pcm4 = ax4.pcolormesh(xp, egrid, Rw, shading="gouraud",
                                  cmap="RdBu_r", vmin=-rlim, vmax=rlim)
            ax4.set_title(r"Re %s$(Q,\,\omega+i\eta)$"
                          "\n(dynamic screening level shift, %s)"
                          % (obs_tex, c.cont_method))
        ax4.set_ylabel(r"$\omega$ (eV)")
        fig.colorbar(pcm4, ax=ax4).set_label("eV")

        labels = [c.corners[0][0]] + [lb for lb, _ in c.corners[1:]]
        for ax in (ax1, ax2, ax3, ax4):
            ax.set_xlabel("exciton momentum  Q")
            ax.set_xlim(xp[0], xp[-1])
            ticks = [xp[b if b < len(xp) else -1] for b in self.bounds]
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            for b in self.bounds:
                ax.axvline(xp[b if b < len(xp) else -1], color="gray",
                           lw=0.5, alpha=0.5)
        for ax in (ax3, ax4):
            ax.set_ylim(egrid[0], egrid[-1])

        fig.suptitle("%s, %s channel\n"
                     "(interpolated over Q before any continuation)"
                     % ("Tr P (basis-free exciton DOS)" if trp
                        else "aux-basis diagonal of P" if diagp
                        else "VPV diagonal in the occ/virt MO space",
                        c.channel), fontsize=9)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(c.fig_base + "." + ext, dpi=150, bbox_inches="tight")
        print("Saved -> %s.pdf / .png" % c.fig_base)

    def run(self):
        self.setup()
        self.collect_mesh()
        self.validate()
        self.build_path()
        self.interpolate()
        self.continue_to_real_axis()
        self.grid_poles()
        self.save_h5()
        self.plot()
        self.print_error_report()


def parse_corners(spec):
    """'G:0,0,0 M:0.5,0,0 ...' -> [(label, (fx,fy,fz)), ...]."""
    corners = []
    for tok in spec.split():
        label, xyz = tok.split(":")
        label = r"$\Gamma$" if label in ("G", "Gamma") else label
        corners.append((label, tuple(float(x) for x in xyz.split(","))))
    return corners


def main():
    p = argparse.ArgumentParser(
        description="General VPV-diagonal exciton-dispersion solver")
    p.add_argument("--input", required=True, help="mean-field input h5")
    p.add_argument("--sim", required=True, help="scGW sim h5")
    p.add_argument("--int_path", required=True, help="DF integral directory")
    p.add_argument("--ir_file", required=True, help="IR grid h5")
    p.add_argument("--beta", type=float, default=1000.0)
    p.add_argument("--channel", default="singlet",
                   choices=["singlet", "triplet"])
    p.add_argument("--cont_method", default="nevanlinna",
                   choices=["aaa", "plaspole", "nevanlinna"])
    p.add_argument("--eta", type=float, default=0.5, help="broadening (eV)")
    p.add_argument("--e_min", type=float, default=0.0)
    p.add_argument("--e_max", type=float, default=None,
                   help="default: auto from the QP pair-energy range")
    p.add_argument("--e_npts", type=int, default=700)
    p.add_argument("--n_act_occ", type=int, default=None)
    p.add_argument("--n_act_vir", type=int, default=None)
    p.add_argument("--path", default="G:0,0,0 M:0.5,0,0 "
                   "K:0.333333333,0.333333333,0 G:0,0,0",
                   help="high-symmetry corners as LABEL:fx,fy,fz tokens")
    p.add_argument("--npts", type=int, default=60,
                   help="path points per segment")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--cache", default="vpv_mesh_cache.npz")
    p.add_argument("--out", default="vpv_dispersion.h5")
    p.add_argument("--fig", default="vpv_dispersion",
                   help="figure basename ('' to skip)")
    p.add_argument("--observable", default="trP",
                   choices=["trP", "vpv", "diagP"],
                   help="Tr P (basis-free DOS, default), the VPV diagonal, "
                   "or the aux-basis diagonal of P (site-resolved, "
                   "crossing-free channels)")
    p.add_argument("--channel_basis", default="eigen", choices=["ov", "eigen"],
                   help="pair channels: energy-ordered (i,a) bands or "
                   "eigenvectors of the static VPV(0) pair matrix per mesh Q")
    p.add_argument("--no_offdiag", action="store_true",
                   help="skip the off-diagonal VPV(0) evaluation")
    args = p.parse_args()

    cfg = VPVDispersionConfig(
        input_file=args.input, sim_file=args.sim, int_path=args.int_path,
        ir_file=args.ir_file, beta=args.beta, channel=args.channel,
        cont_method=args.cont_method, eta_eV=args.eta, e_min=args.e_min,
        e_max=args.e_max, e_npts=args.e_npts, n_act_occ=args.n_act_occ,
        n_act_vir=args.n_act_vir, corners=parse_corners(args.path),
        npts=args.npts, dim=args.dim, cache_file=args.cache,
        out_file=args.out, fig_base=args.fig or None,
        offdiag=not args.no_offdiag, channel_basis=args.channel_basis,
        observable=args.observable)
    VPVDispersionSolver(cfg).run()


if __name__ == "__main__":
    main()
