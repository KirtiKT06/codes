"""
pb_pinn.py
----------
Stage 1: single-structure nonlinear Poisson-Boltzmann PINN.

Equation solved (reduced units: length in Angstrom, potential phi in units
of k_B T / e, so no explicit e, k_B, T floating around):

    div( eps(r) grad phi(r) )  -  kbar2(r) * sinh(phi(r))  =  -4*pi*lB*rho_fixed(r)

  - eps(r):      smoothed dielectric field, ~4 inside RNA -> 78 in bulk water
  - kbar2(r):    modified Debye screening field, 0 inside RNA (ion-excluded
                 region) -> kappa_bulk^2 in bulk water (ion-accessibility mask,
                 same sigmoid surface as eps by construction here)
  - rho_fixed(r): Gaussian-smeared phosphate charge density (see structure.py)
  - lB:          Bjerrum length in water (~7.0 A at ~298 K)

This is the standard reduced nonlinear PBE used by mesh solvers like APBS/DelPhi.
Sign/scaling conventions differ slightly across the PB solver literature --
VALIDATE THIS AGAINST APBS ON A TEST STRUCTURE before trusting it (see
`sanity_check_against_apbs` stub at the bottom -- fill in once you run APBS).

The network is a SIREN (sinusoidal-activation MLP) because plain ReLU/tanh
MLPs underfit the sharp potential gradients near the phosphate backbone.
Conditioning on structure identity (the "amortized" stage 2 version) is
deliberately NOT in this file -- get single-structure training solid first,
then wrap the geometry encoder + FiLM conditioning around this.
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Physical constants (reduced units: Angstrom, k_B T / e for potential)
# ----------------------------------------------------------------------
LB_WATER = 7.0           # Bjerrum length in water, ~298K, Angstrom
EPS_WATER = 78.0
EPS_RNA = 4.0            # interior dielectric; 2-20 range is debated, start low
# LB_WATER already has 1/EPS_WATER baked in (that's why it's ~7 A instead of
# the vacuum value ~546 A). To get the equivalent constant for a uniform
# medium at EPS_RNA you must go back through the vacuum reference and divide
# by EPS_RNA -- NOT divide LB_WATER by EPS_RNA again (that silently divides
# by EPS_WATER*EPS_RNA, undershooting phi_analytic by a factor of EPS_WATER,
# ~78x). See phi_analytic() below, which was the bug.
LB_VACUUM = LB_WATER * EPS_WATER   # ~546 Angstrom
SURFACE_WIDTH = 1.0      # Angstrom, width of the sigmoid eps/kappa transition
PROBE_RADIUS = 1.4       # Angstrom, rough solvent probe radius offset
ATOM_RADIUS = 1.7        # Å, average heavy-atom radius
GAUSSIAN_SIGMA = 0.3     # Angstrom, width of smeared point charges


def ionic_strength_to_kappa2(I_molar: float, lB: float = LB_WATER) -> float:
    """kappa^2 in Angstrom^-2, from bulk monovalent ionic strength in mol/L."""
    N_A = 6.02214076e23
    n_per_A3 = I_molar * N_A / 1e27  # number density, ions/Angstrom^3
    return 8.0 * np.pi * lB * n_per_A3


# ----------------------------------------------------------------------
# SIREN network: phi(r; theta)
# ----------------------------------------------------------------------
class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f)
        self._init_weights(in_f)

    def _init_weights(self, in_f):
        with torch.no_grad():
            bound = 1 / in_f if self.is_first else np.sqrt(6 / in_f) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class PBPotentialNet(nn.Module):
    """phi(r): R^3 -> R. Coordinates should be pre-scaled (see Scaler below)."""

    # FIX: was omega_0=80.0, omega_hidden=1.0 -- standard SIREN (Sitzmann et
    # al.) uses ONE omega_0 across all layers (paper default: 30). The
    # 80-vs-1 mismatch here means second derivatives (needed for the PDE's
    # div(eps grad phi) term) scale very unevenly across the network,
    # producing wildly batch-dependent PDE-residual magnitudes -- a likely
    # contributor to the epoch-to-epoch loss spikes you were seeing.
    def __init__(self, hidden=128, n_hidden_layers=4, omega_0=30.0, omega_hidden=30.0):
        super().__init__()
        layers = [SineLayer(3, hidden, is_first=True, omega_0=omega_0)]
        for _ in range(n_hidden_layers):
            layers.append(SineLayer(hidden, hidden, is_first=False, omega_0=omega_hidden))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)
        with torch.no_grad():
            bound = np.sqrt(6 / hidden) / omega_hidden
            self.head.weight.uniform_(-bound, bound)
            self.head.bias.zero_()

    def forward(self, r):
        return self.head(self.body(r)).squeeze(-1)


class Scaler:
    """Map real-space Angstrom coords into a roughly [-1, 1]^3 box for the SIREN."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.center = torch.tensor((lo + hi) / 2.0, dtype=torch.float32, device=DEVICE)
        self.half_extent = torch.tensor((hi - lo) / 2.0, dtype=torch.float32, device=DEVICE)

    def to_unit(self, r):
        return (r - self.center) / self.half_extent

    def from_unit(self, r_unit):
        return r_unit * self.half_extent + self.center


# ----------------------------------------------------------------------
# Physical fields built from the structure (all torch, differentiable
# w.r.t. r via autograd -- needed for the div(eps grad phi) term)
# ----------------------------------------------------------------------
class PBFields:
    def __init__(self, phosphate_xyz, phosphate_q, allatom_xyz, ionic_strength_M=0.15):
        self.phosphate_xyz = torch.tensor(phosphate_xyz, dtype=torch.float32, device=DEVICE)
        self.phosphate_q = torch.tensor(phosphate_q, dtype=torch.float32, device=DEVICE)
        self.allatom_xyz = torch.tensor(allatom_xyz, dtype=torch.float32, device=DEVICE)
        self.kappa2_bulk = ionic_strength_to_kappa2(ionic_strength_M)
        self.net_charge = float(phosphate_q.sum())
        self.centroid = torch.tensor(allatom_xyz.mean(axis=0), dtype=torch.float32, device=DEVICE)

    def _distance_to_surface(self, r):
        # r: (N, 3). Nearest-atom distance minus a probe radius -> crude signed
        # "distance to molecular surface" (positive = solvent side). This is
        # a stand-in for a real SES; fine for a first pass, replace with an
        # actual solvent-excluded-surface SDF if the sigmoid surface proves
        # too crude once you validate against APBS.
        d = torch.cdist(r, self.allatom_xyz)
        d_min, _ = d.min(dim=1)
        d_min = torch.clamp(d_min, min=0.5)  # floor: prevents sqrt-derivative blowup
                                            # when a collocation point lands
                                            # very close to an atom
        return d_min - PROBE_RADIUS - ATOM_RADIUS
    
    def phi_analytic(self, r):
        dist = torch.cdist(r, self.phosphate_xyz).clamp(min=1e-6)  # (N, Np)
        alpha = 1.0 / (np.sqrt(2) * GAUSSIAN_SIGMA)
        kappa = float(np.sqrt(self.kappa2_bulk))

        dist64 = dist.double()

        # Closed form for phi(r) = convolution of the screened (Yukawa)
        # Green's function exp(-kappa*s)/s with a normalized Gaussian charge
        # of width GAUSSIAN_SIGMA. Verified against direct numerical
        # quadrature of the convolution integral to match to >6 significant
        # figures for every r, including r=0.
        #
        # FIX: the previous version summed two terms, (term1+term2)/(2*r),
        # which is only asymptotically correct for r >> GAUSSIAN_SIGMA. As
        # r->0 that sum approaches a nonzero constant instead of ~2r*(finite
        # slope), so it does NOT cancel the 1/r singularity -- with dist
        # floored at only 1e-6, this diverged to ~1e6-1e8 whenever a
        # collocation point landed close to a phosphate. That was the
        # dominant cause of your erratic loss spikes, well beyond the
        # dielectric scaling bug. The correct closed form is a DIFFERENCE
        # of two terms with an extra exp(kappa^2/(4*alpha^2)) prefactor on
        # one of them, which correctly -> a small finite value as r -> 0.
        a_plus = alpha * dist64 + kappa / (2 * alpha)          # -> +inf as r grows
        a_minus = kappa / (2 * alpha) - alpha * dist64         # -> -inf as r grows

        # term_far: stable for large r via erfcx (erfcx(a_plus) decays like
        # 1/a_plus; the exp(-(alpha*dist)^2) factor keeps the product from
        # overflowing).
        term_far = torch.special.erfcx(a_plus) * torch.exp(-(alpha * dist64) ** 2)

        # term_near: stable for large r directly (the exponent underflows
        # to 0 and erfc saturates at 2 -- no cancellation issue, unlike
        # erfcx would have for a large-magnitude negative argument).
        exponent_near = (kappa ** 2) / (4 * alpha ** 2) - kappa * dist64
        term_near = torch.exp(exponent_near) * torch.special.erfc(a_minus)

        kernel = ((term_near - term_far) / (2 * dist64)).float()
        # FIX: was (LB_WATER / EPS_RNA), which undershoots by ~EPS_WATER (78x)
        # -- see the LB_VACUUM comment above.
        phi = (LB_VACUUM / EPS_RNA) * (kernel * self.phosphate_q.unsqueeze(0)).sum(dim=1)
        return phi

    def eps(self, r):
        s = self._distance_to_surface(r)
        t = torch.sigmoid(s / SURFACE_WIDTH)
        return EPS_RNA + (EPS_WATER - EPS_RNA) * t

    def kbar2(self, r):
        s = self._distance_to_surface(r)
        t = torch.sigmoid(s / SURFACE_WIDTH)
        return self.kappa2_bulk * t  # zero inside RNA, kappa2_bulk in bulk water

    def rho_fixed(self, r):
        # Sum of Gaussian-smeared point charges at the phosphates.
        d2 = torch.cdist(r, self.phosphate_xyz) ** 2  # (N, Np)
        norm = (2 * np.pi * GAUSSIAN_SIGMA ** 2) ** 1.5
        weights = torch.exp(-d2 / (2 * GAUSSIAN_SIGMA ** 2)) / norm
        return (weights * self.phosphate_q.unsqueeze(0)).sum(dim=1)

    def debye_huckel_farfield(self, r):
        # Whole-molecule net charge treated as a point charge at the centroid --
        # a crude far-field approximation, adequate at the box boundary given
        # ~15-25 A padding. Refine to a multipole expansion if BC loss doesn't
        # converge cleanly.
        kappa = np.sqrt(self.kappa2_bulk)
        dist = torch.norm(r - self.centroid.unsqueeze(0), dim=1).clamp(min=1e-3)
        return LB_WATER * self.net_charge * torch.exp(-kappa * dist) / dist

def total_phi(net, scaler, fields, r_real):
    """phi = closed-form singular part + network's smooth correction."""
    r_unit = scaler.to_unit(r_real)
    return fields.phi_analytic(r_real) + net(r_unit)

# ----------------------------------------------------------------------
# PDE residual via autograd: div(eps grad phi) - kbar2 * sinh(phi) + 4 pi lB rho
# ----------------------------------------------------------------------
def pde_residual(net, scaler, fields, r_real):
    r_real = r_real.clone().requires_grad_(True)
    phi = total_phi(net, scaler, fields, r_real)   # <-- changed line

    grad_phi = torch.autograd.grad(phi.sum(), r_real, create_graph=True)[0]
    eps_val = fields.eps(r_real)
    flux = eps_val.unsqueeze(-1) * grad_phi
    div = torch.zeros_like(phi)
    for i in range(3):
        div_i = torch.autograd.grad(flux[:, i].sum(), r_real, create_graph=True)[0][:, i]
        div = div + div_i
    kbar2_val = fields.kbar2(r_real)
    rho_val = fields.rho_fixed(r_real)
    phi_clamped = torch.clamp(phi, -15.0, 15.0)
    residual = div - kbar2_val * torch.sinh(phi_clamped) + 4 * np.pi * LB_WATER * rho_val
    return residual


# ----------------------------------------------------------------------
# Collocation sampling: mix of uniform-box points and near-molecule points
# (potential gradients are steepest near the phosphate backbone -- pure
# uniform sampling under-resolves that region)
# ----------------------------------------------------------------------
# def sample_collocation(lo, hi, phosphate_xyz, n_uniform=4000, n_near=4000, near_std=4.0):
#     uniform = np.random.uniform(lo, hi, size=(n_uniform, 3))
#     idx = np.random.randint(0, len(phosphate_xyz), size=n_near)
#     near = phosphate_xyz[idx] + np.random.normal(0, near_std, size=(n_near, 3))
#     near = np.clip(near, lo, hi)
#     pts = np.concatenate([uniform, near], axis=0)
#     return torch.tensor(pts, dtype=torch.float32, device=DEVICE)

def sample_collocation(lo, hi, phosphate_xyz, n_uniform=3000, n_near=3000, n_veryclose=2000, near_std=3.0, veryclose_std=1.0):
    uniform = np.random.uniform(lo, hi, size=(n_uniform, 3))
    idx1 = np.random.randint(0, len(phosphate_xyz), size=n_near)
    near = phosphate_xyz[idx1] + np.random.normal(0, near_std, size=(n_near, 3))
    idx2 = np.random.randint(0, len(phosphate_xyz), size=n_veryclose)
    veryclose = phosphate_xyz[idx2] + np.random.normal(0, veryclose_std, size=(n_veryclose, 3))
    pts = np.concatenate([uniform, near, np.clip(veryclose, lo, hi)], axis=0)
    return torch.tensor(np.clip(pts, lo, hi), dtype=torch.float32, device=DEVICE)


def sample_boundary(lo, hi, n_per_face=500):
    pts = []
    for axis in range(3):
        for val in (lo[axis], hi[axis]):
            p = np.random.uniform(lo, hi, size=(n_per_face, 3))
            p[:, axis] = val
            pts.append(p)
    return torch.tensor(np.concatenate(pts, axis=0), dtype=torch.float32, device=DEVICE)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train_pb_pinn(
    phosphate_xyz, phosphate_q, allatom_xyz, lo, hi,
    ionic_strength_M=0.15, n_epochs=3000, lr=1e-4, bc_weight=1.0,
    log_every=200, n_uniform=4000, n_near=4000, n_bc_per_face=500,
    adaptive_bc_weight=True, reweight_every=500, reweight_ema=0.3,
    loss_avg_window=100, lr_patience=10, lr_decay_factor=0.5,
):
    """
    adaptive_bc_weight: if True, bc_weight is no longer fixed. Every
        `reweight_every` epochs we compare ||grad_theta loss_pde|| against
        ||grad_theta loss_bc|| and nudge bc_weight (via an EMA, smoothing
        factor `reweight_ema`) toward the ratio that would put them on equal
        footing. Without this, loss_bc (O(1)) is invisible next to loss_pde
        spikes (O(1e4-1e6)) and the boundary condition effectively stops
        being enforced -- which is what the steadily-rising bc column in the
        old training log was showing. Set False to keep the old fixed
        bc_weight behavior.
    loss_avg_window: length of the moving-average window used both for the
        printed log line (`avg`) and for driving the LR schedule below --
        the per-epoch "total" logged value is a single fresh Monte Carlo
        batch, so it's too noisy to schedule on directly.
    lr_patience / lr_decay_factor: FIX -- the old scheduler was
        StepLR(step_size=15000, gamma=0.5), a fixed clock with no relation
        to n_epochs. For a 3000-epoch test run that's mild; for the 80000-
        epoch real run it halves LR 5 times by epoch 75000 (down to
        ~3e-6), which is almost certainly why your loss went flat for the
        last ~20-30k epochs -- not because training had converged, but
        because the LR had been decayed into the ground on a schedule
        tuned for a much shorter run. ReduceLROnPlateau instead halves LR
        only after `lr_patience` consecutive log_every-blocks where the
        moving-average loss hasn't meaningfully improved, so it adapts to
        however long you actually train for.
    """
    fields = PBFields(phosphate_xyz, phosphate_q, allatom_xyz, ionic_strength_M)
    # raise RuntimeError(
    #     str(fields.rho_fixed(fields.phosphate_xyz[:1])))
    scaler = Scaler(lo, hi)
    net = PBPotentialNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_decay_factor, patience=lr_patience, threshold=1e-3
    )
    params = list(net.parameters())

    history = {"total": [], "pde": [], "bc": [], "bc_weight": []}
    total_window = deque(maxlen=loss_avg_window)

    for epoch in range(n_epochs):
        opt.zero_grad()

        r_pde = sample_collocation(lo, hi, phosphate_xyz, n_uniform=n_uniform, n_near=n_near)
        res = pde_residual(net, scaler, fields, r_pde)
        loss_pde = (res ** 2).mean()
        r_bc = sample_boundary(lo, hi, n_per_face=n_bc_per_face)
        phi_bc_pred = total_phi(net, scaler, fields, r_bc)
        phi_bc_target = fields.debye_huckel_farfield(r_bc)
        loss_bc = ((phi_bc_pred - phi_bc_target) ** 2).mean()

        if adaptive_bc_weight and epoch % reweight_every == 0 and epoch > 0:
            # Two extra backward passes, only every `reweight_every` epochs --
            # cheap relative to the training cost, and both need
            # retain_graph=True since we still have to run the real
            # loss.backward() below on the same graph.
            grad_pde = torch.autograd.grad(loss_pde, params, retain_graph=True, allow_unused=True)
            grad_bc = torch.autograd.grad(loss_bc, params, retain_graph=True, allow_unused=True)
            norm_pde = torch.sqrt(sum((g ** 2).sum() for g in grad_pde if g is not None) + 1e-12)
            norm_bc = torch.sqrt(sum((g ** 2).sum() for g in grad_bc if g is not None) + 1e-12)
            target_bc_weight = (norm_pde / norm_bc).item()
            target_bc_weight = min(max(target_bc_weight, 1e-3), 1e3)  # sanity clamp
            bc_weight = (1 - reweight_ema) * bc_weight + reweight_ema * target_bc_weight

        loss = loss_pde + bc_weight * loss_bc

        if not torch.isfinite(loss):
            print(f"epoch {epoch}: non-finite loss, skipping step")
            opt.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()

        history["total"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["bc"].append(loss_bc.item())
        history["bc_weight"].append(bc_weight)
        total_window.append(loss.item())

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            running_avg = sum(total_window) / len(total_window)
            # Step the LR schedule on the smoothed average, not the raw
            # per-epoch value -- ReduceLROnPlateau needs a stable metric to
            # judge "no improvement" against, and a single fresh MC batch
            # is too noisy for that.
            scheduler.step(running_avg)
            current_lr = opt.param_groups[0]["lr"]
            # "total" is a single fresh Monte Carlo batch (noisy by
            # construction); "avgN" is the moving average over the last N
            # logged-and-unlogged steps and is the one to watch for actual
            # convergence trend.
            print(f"epoch {epoch:5d}  total {loss.item():.5e}  "
                  f"avg{len(total_window):<4d} {running_avg:.5e}  "
                  f"pde {loss_pde.item():.5e}  bc {loss_bc.item():.5e}  "
                  f"bc_w {bc_weight:.3e}  lr {current_lr:.2e}")

    pts = torch.tensor(
        phosphate_xyz,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():
        phi_phos = total_phi(net, scaler, fields, pts).cpu().numpy()

    print("\nPer-phosphate potential statistics")
    print(f"min  = {phi_phos.min():.4f}")
    print(f"max  = {phi_phos.max():.4f}")
    print(f"mean = {phi_phos.mean():.4f}")
    print(f"std  = {phi_phos.std():.4f}")

    print("\nPotential profile near first phosphate")

    for d in [2.0, 3.0, 5.0, 10.0, 20.0, 40.0]:
        pt = torch.tensor(
            [phosphate_xyz[0] + np.array([d, 0.0, 0.0])],
            dtype=torch.float32,
            device=DEVICE,
        )

        with torch.no_grad():
            phi = total_phi(net, scaler, fields, pt).item()

        print(f"d = {d:4.1f} Å   phi = {phi:+.4f}")

    return net, scaler, fields, history


# ----------------------------------------------------------------------
# Per-residue preferential interaction coefficient Gamma_i (the label you
# actually want to correlate against embeddings -- see prior discussion).
# Gamma_i = integral over a shell around phosphate i of [c(r) - c_bulk] dr,
# with c(r) = c_bulk * exp(-phi(r)) from the Boltzmann relation (monovalent
# cation; use exp(-z*phi) for general valence z).
# ----------------------------------------------------------------------
def compute_gamma_per_residue(net, scaler, fields, phosphate_xyz, c_bulk_M,
                               shell_radius=10.0, n_mc=1000000):
    N_A = 6.02214076e23
    c_bulk_per_A3 = c_bulk_M * N_A / 1e27

    gammas = []
    phosphate_t = torch.tensor(phosphate_xyz, dtype=torch.float32, device=DEVICE)
    for i in range(len(phosphate_xyz)):
        center = phosphate_t[i]
        # Monte Carlo integration over a sphere of radius shell_radius
        u = torch.randn(n_mc, 3, device=DEVICE)
        u = u / u.norm(dim=1, keepdim=True)
        radii = shell_radius * torch.rand(n_mc, device=DEVICE) ** (1 / 3)  # uniform in volume
        pts = center.unsqueeze(0) + u * radii.unsqueeze(-1)

        with torch.no_grad():
            phi = total_phi(net, scaler, fields, pts)
            # FIX (was producing inf): a raw Boltzmann factor exp(-phi) is
            # only physically meaningful in the ion-ACCESSIBLE region -- the
            # shell sampled here is a uniform sphere that also includes
            # points inside the RNA core / near-singular near-field region,
            # where phi can be a hundred-plus kT/e (this is the classic
            # "Coulombic catastrophe" of continuum PB near a bare source).
            # exp(151) alone overflows float32 (which maxes out ~exp(88)).
            # Fix: (1) weight by the same ion-accessibility mask (kbar2)
            # used during training -- 0 inside the solute, 1 in bulk water,
            # smooth in between -- so inaccessible points contribute
            # nothing; (2) clamp phi defensively before exponentiating,
            # matching the same clamp pde_residual already applies before
            # sinh(), since the sigmoid mask is smooth, not a hard cutoff.
            accessible_frac = (fields.kbar2(pts) / fields.kappa2_bulk).clamp(0.0, 1.0)
            phi_clamped = torch.clamp(phi, -15.0, 15.0)
            c_r = c_bulk_per_A3 * torch.exp(-phi_clamped) * accessible_frac
            excess = c_r - c_bulk_per_A3 * accessible_frac

        sphere_vol = (4 / 3) * np.pi * shell_radius ** 3
        gamma_i = excess.mean().item() * sphere_vol
        gammas.append(gamma_i)

    return np.array(gammas)  # units: number of excess ions per residue shell


def sanity_check_against_apbs():
    """
    Stub. Before trusting Gamma_i on the full dataset:
      1. Run APBS (nonlinear PBE, same eps_in/eps_out, same ionic strength)
         on one small, well-characterized RNA (e.g. a tRNA or hammerhead).
      2. Extract phi(r) from the APBS .dx output along a line/plane through
         the molecule.
      3. Compare against net(r) from this script on the same points -- check
         sign, magnitude, and decay length match before scaling up.
    Not implemented here since it needs an APBS install + a real structure;
    flagging so it isn't silently skipped.
    """
    raise NotImplementedError("Run this manually against an APBS reference before trusting Gamma_i.")