"""
Phase 1: Nonlinear Poisson-Boltzmann PINN — validation on a manufactured-solution
2D interface problem before touching real RNA geometry or real point charges.

Design rationale (combining three papers, see chat for full reasoning):
  - Two-branch domain-decomposition network (Achondo et al. 2025 / Chen et al. 2025)
  - Direct nonlinear sinh(u) residual, no linearization (Park & Jo 2025 / Chen et al. 2025)
  - Input/output scaling + random Fourier features + trainable tanh activations +
    adaptive loss balancing (Achondo et al. 2025 — their ablation showed these are
    what separates an accurate PINN from a barely-converging one)
  - Periodic collocation resampling from a larger pool (Chen et al. 2025)

Test problem: an ellipse-shaped solute-solvent interface with a MANUFACTURED
piecewise solution (Park & Jo, Example 1 style). We pick the exact answer, use
autograd to derive exactly what forcing term f(x,y) makes the nonlinear PBE
produce that answer, and then check whether the PINN can recover it. This
isolates "does our nonlinear architecture work" from "is the RNA geometry/
mesh/real electrostatics right" — those come in later phases.
"""

import torch
import torch.nn as nn
import numpy as np

torch.set_default_dtype(torch.float64)  # PDE residuals need double precision;
                                         # single precision noise swamps the signal
                                         # once losses get small.

DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# 1. GEOMETRY: ellipse interface, following Achondo Ex.1 / Park & Jo Ex.1
# ---------------------------------------------------------------------------
class EllipseInterface:
    """
    Interface Gamma: x^2/a^2 + y^2/b^2 = 1
    Omega_m (solute, "inside")  : x^2/a^2 + y^2/b^2 < 1
    Omega_w (solvent, "outside"): x^2/a^2 + y^2/b^2 > 1, truncated at box [-L,L]^2
    """

    def __init__(self, a=0.7, b=0.5, L=1.0):
        self.a, self.b, self.L = a, b, L

    def level(self, xy):
        # >0 outside (solvent), <0 inside (solute), ==0 on Gamma
        x, y = xy[:, 0], xy[:, 1]
        return (x / self.a) ** 2 + (y / self.b) ** 2 - 1.0

    def is_inside(self, xy):
        return self.level(xy) < 0

    def sample_box(self, n):
        return (torch.rand(n, 2) * 2 - 1) * self.L

    def sample_omega_m(self, n):
        """Rejection-sample n points strictly inside the ellipse."""
        pts = []
        while sum(p.shape[0] for p in pts) < n:
            cand = self.sample_box(n * 3)
            mask = self.is_inside(cand)
            pts.append(cand[mask])
        return torch.cat(pts, dim=0)[:n].clone().requires_grad_(True)

    def sample_omega_w(self, n):
        """Rejection-sample n points strictly outside the ellipse but inside the box."""
        pts = []
        while sum(p.shape[0] for p in pts) < n:
            cand = self.sample_box(n * 3)
            mask = ~self.is_inside(cand)
            pts.append(cand[mask])
        return torch.cat(pts, dim=0)[:n].clone().requires_grad_(True)

    def sample_gamma(self, n):
        """Parametrize the ellipse boundary directly (exact, no rejection needed)."""
        theta = torch.rand(n) * 2 * np.pi
        x = self.a * torch.cos(theta)
        y = self.b * torch.sin(theta)
        pts = torch.stack([x, y], dim=1).requires_grad_(True)
        # outward unit normal to the ellipse at (x,y): gradient of the level
        # function, normalized. grad(level) = (2x/a^2, 2y/b^2).
        nx = 2 * x / self.a ** 2
        ny = 2 * y / self.b ** 2
        norm = torch.sqrt(nx ** 2 + ny ** 2)
        normals = torch.stack([nx / norm, ny / norm], dim=1)
        return pts, normals

    def sample_boundary_box(self, n):
        """Sample points on the outer square boundary d(Omega)."""
        n_side = n // 4
        sides = []
        L = self.L
        t = torch.rand(n_side) * 2 * L - L
        sides.append(torch.stack([torch.full_like(t, L), t], dim=1))
        sides.append(torch.stack([torch.full_like(t, -L), t], dim=1))
        sides.append(torch.stack([t, torch.full_like(t, L)], dim=1))
        sides.append(torch.stack([t, torch.full_like(t, -L)], dim=1))
        return torch.cat(sides, dim=0).requires_grad_(True)


# ---------------------------------------------------------------------------
# 2. ARCHITECTURE: one "enhanced branch" = scaling + Fourier features +
#    trainable-activation MLP + output scaling. Two of these (Nm, Nw) form
#    the full two-branch network.
# ---------------------------------------------------------------------------
class TrainableTanh(nn.Module):
    """tanh(a * x) with a trainable per-neuron scale (Jagtap et al., used in Achondo)."""

    def __init__(self, n_features):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_features))

    def forward(self, x):
        return torch.tanh(self.a * x)


class FourierFeatures(nn.Module):
    """Random (non-trainable) Fourier feature layer, mitigates spectral bias."""

    def __init__(self, in_dim, n_features=64, sigma=1.0):
        super().__init__()
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer("B", B)  # non-trainable, per Achondo eq. 18

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)


class EnhancedBranch(nn.Module):
    """
    One domain's network: input scaling -> Fourier features -> hidden layers with
    trainable tanh -> linear output -> output scaling.
    """

    def __init__(self, xmin, xmax, ymin_out, ymax_out,
                 n_fourier=64, sigma=1.0, hidden=(64, 64, 64), in_dim=2):
        super().__init__()
        self.register_buffer("xmin", torch.tensor(xmin))
        self.register_buffer("xmax", torch.tensor(xmax))
        self.ymin_out = ymin_out
        self.ymax_out = ymax_out

        self.fourier = FourierFeatures(in_dim, n_fourier, sigma)
        dims = [2 * n_fourier] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(TrainableTanh(dims[i + 1]))
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], 1)

    def input_scale(self, x):
        # maps [xmin, xmax] -> [-1, 1] componentwise (Achondo eq. 15)
        return 2 * (x - self.xmin) / (self.xmax - self.xmin) - 1

    def output_scale(self, y_scaled):
        # maps network's raw scalar output -> physical range (Achondo eq. 16)
        return 0.5 * (y_scaled + 1) * (self.ymax_out - self.ymin_out) + self.ymin_out

    def forward(self, x):
        xs = self.input_scale(x)
        f = self.fourier(xs)
        h = self.hidden(f)
        y_scaled = self.out(h)
        return self.output_scale(y_scaled).squeeze(-1)


class TwoBranchPINN(nn.Module):
    """Full model: Nm for Omega_m (solute), Nw for Omega_w (solvent)."""

    def __init__(self, geom: EllipseInterface):
        super().__init__()
        self.geom = geom
        L = geom.L
        # Omega_m occupies roughly [-a,a] x [-b,b]; Omega_w occupies the box.
        # Output range hyperparameters: for this manufactured test we know the
        # exact solution's range analytically, so we set generous bounds here.
        # (In the real RNA problem this becomes the Born-ion estimate, Achondo eq. 17.)
        self.Nm = EnhancedBranch(xmin=[-geom.a, -geom.b], xmax=[geom.a, geom.b],
                                  ymin_out=-1.2, ymax_out=1.2)
        self.Nw = EnhancedBranch(xmin=[-L, -L], xmax=[L, L],
                                  ymin_out=0.8, ymax_out=3.2)

    def u_m(self, xy):
        return self.Nm(xy)

    def u_w(self, xy):
        return self.Nw(xy)
