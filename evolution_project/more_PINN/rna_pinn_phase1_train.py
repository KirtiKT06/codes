"""
Phase 1 training script. See rna_pinn_phase1.py for architecture/geometry.

Physics setup (Park & Jo 2025, Example 1 -- ellipse interface, manufactured
regular solution; equation form follows the nonlinear PBE, Chen et al. 2025 /
Achondo et al. 2025 notation):

    In Omega_m (solute, "-"):  -eps_m * Laplacian(u) = f_m        (kappa_m = 0)
    In Omega_w (solvent,"+"):  -eps_w * Laplacian(u) + kappa_w^2*sinh(u) = f_w

    Interface Gamma:  [[u]]           = u_w - u_m  = jump_u  (manufactured, != 0)
                       [[eps du/dn]]  = eps_w du_w/dn - eps_m du_m/dn = jump_flux

    Boundary dOmega:  u = g (manufactured Dirichlet)

We pick the EXACT answer first:
    u_m_exact(x,y) = cos(x^2 + y^2)      on Omega_m
    u_w_exact(x,y) = 2 + sin(x*y)        on Omega_w

...and use autograd on these closed-form functions to derive f_m, f_w,
jump_u, jump_flux automatically (no hand-differentiation, which is exactly
where manufactured-solution papers tend to introduce silent algebra bugs).
This is the same trick Park & Jo and Chen et al. use, just automated.
"""

import torch
import numpy as np
from tqdm import tqdm
from rna_pinn_phase1 import EllipseInterface, TwoBranchPINN

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# Physical parameters (Park & Jo / Achondo-style: solute low-eps/no-salt,
# solvent high-eps/salt)
# ---------------------------------------------------------------------------
EPS_M = 2.0
EPS_W = 80.0
KAPPA_W = 2.974   # kappa_m = 0 (no salt inside the solute)
DEVICE = 'cuda'

# ---------------------------------------------------------------------------
# Exact manufactured solution (used ONLY to generate training targets/forcing
# and for computing validation error -- the network never sees these formulas)
# ---------------------------------------------------------------------------
def u_m_exact(xy):
    x, y = xy[:, 0], xy[:, 1]
    return torch.cos(x ** 2 + y ** 2)


def u_w_exact(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 2 + torch.sin(x * y)


def laplacian(u_fn, xy):
    """Autograd Laplacian of a scalar function u_fn at points xy (n,2)."""
    xy = xy.clone().requires_grad_(True)
    u = u_fn(xy)
    grad_u = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u),
                                  create_graph=True)[0]
    lap = 0.0
    for i in range(2):
        grad2 = torch.autograd.grad(grad_u[:, i], xy, grad_outputs=torch.ones_like(grad_u[:, i]),
                                     create_graph=True)[0]
        lap = lap + grad2[:, i]
    return lap


def gradient(u_fn, xy):
    xy = xy.clone().requires_grad_(True)
    u = u_fn(xy)
    return torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]


def forcing_m(xy):
    """f_m such that -eps_m*Laplacian(u_m_exact) = f_m  (linear domain, kappa=0)."""
    lap = laplacian(u_m_exact, xy)
    return -EPS_M * lap


def forcing_w(xy):
    """f_w such that -eps_w*Laplacian(u_w_exact) + kappa_w^2*sinh(u_w_exact) = f_w."""
    lap = laplacian(u_w_exact, xy)
    u = u_w_exact(xy)
    return -EPS_W * lap + KAPPA_W ** 2 * torch.sinh(u)


def interface_jump_u(xy_gamma):
    return u_w_exact(xy_gamma) - u_m_exact(xy_gamma)


def interface_jump_flux(xy_gamma, normals):
    grad_m = gradient(u_m_exact, xy_gamma)
    grad_w = gradient(u_w_exact, xy_gamma)
    dun_m = (grad_m * normals).sum(dim=1)
    dun_w = (grad_w * normals).sum(dim=1)
    return EPS_W * dun_w - EPS_M * dun_m


# ---------------------------------------------------------------------------
# PINN residuals (network's own autograd, NOT the exact solution's)
# ---------------------------------------------------------------------------
def pde_residual_m(model, xy):
    xy = xy.clone().requires_grad_(True)
    u = model.u_m(xy)
    grad_u = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    lap = 0.0
    for i in range(2):
        g2 = torch.autograd.grad(grad_u[:, i], xy, grad_outputs=torch.ones_like(grad_u[:, i]),
                                  create_graph=True)[0]
        lap = lap + g2[:, i]
    f = forcing_m(xy)
    residual = -EPS_M * lap - f       # kappa_m = 0, purely linear here
    return residual


def pde_residual_w(model, xy):
    xy = xy.clone().requires_grad_(True)
    u = model.u_w(xy)
    grad_u = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    lap = 0.0
    for i in range(2):
        g2 = torch.autograd.grad(grad_u[:, i], xy, grad_outputs=torch.ones_like(grad_u[:, i]),
                                  create_graph=True)[0]
        lap = lap + g2[:, i]
    f = forcing_w(xy)
    residual = -EPS_W * lap + KAPPA_W ** 2 * torch.sinh(u) - f   # the nonlinear term, live
    return residual


def interface_residuals(model, xy_gamma, normals):
    xy_gamma = xy_gamma.clone().requires_grad_(True)
    u_m = model.u_m(xy_gamma)
    u_w = model.u_w(xy_gamma)
    grad_m = torch.autograd.grad(u_m, xy_gamma, grad_outputs=torch.ones_like(u_m),
                                  create_graph=True)[0]
    grad_w = torch.autograd.grad(u_w, xy_gamma, grad_outputs=torch.ones_like(u_w),
                                  create_graph=True)[0]
    dun_m = (grad_m * normals).sum(dim=1)
    dun_w = (grad_w * normals).sum(dim=1)

    jump_u_pred = u_w - u_m
    jump_u_target = interface_jump_u(xy_gamma)
    res_u = jump_u_pred - jump_u_target

    flux_pred = EPS_W * dun_w - EPS_M * dun_m
    flux_target = interface_jump_flux(xy_gamma, normals)
    res_flux = flux_pred - flux_target
    return res_u, res_flux


def boundary_residual(model, xy_b):
    # outer boundary box is entirely in Omega_w for this geometry
    u_pred = model.u_w(xy_b)
    u_target = u_w_exact(xy_b)
    return u_pred - u_target


# ---------------------------------------------------------------------------
# Adaptive loss balancing (Achondo et al. eq. 24-25)
# ---------------------------------------------------------------------------
class LossBalancer:
    def __init__(self, names, alpha=0.7):
        self.alpha = alpha
        self.weights = {n: 1.0 for n in names}

    def update(self, per_term_losses, model):
        """per_term_losses: dict name -> scalar loss tensor (already computed
        with create_graph=True upstream isn't needed here; we just need fresh
        grads w.r.t. model params for each term)."""
        grad_norms = {}
        params = [p for p in model.parameters() if p.requires_grad]
        for name, loss in per_term_losses.items():
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            total = 0.0
            for g in grads:
                if g is not None:
                    total = total + (g ** 2).sum()
            grad_norms[name] = torch.sqrt(total + 1e-12).item()
        mean_norm = np.mean(list(grad_norms.values())) + 1e-12
        for name in self.weights:
            w_hat = mean_norm / (grad_norms[name] + 1e-12)
            self.weights[name] = self.alpha * self.weights[name] + (1 - self.alpha) * w_hat


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(n_epochs=3000, n_m=400, n_w=800, n_gamma=200, n_b=200,
          resample_every=100, reweight_every=200, lr=1e-3, verbose_every=200):

    geom = EllipseInterface()
    model = TwoBranchPINN(geom).to(DEVICE)
    print("Model device:", next(model.parameters()).device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    balancer = LossBalancer(["m", "w", "gamma_u", "gamma_flux", "bc"])

    pts_m = geom.sample_omega_m(n_m).to(DEVICE)
    pts_w = geom.sample_omega_w(n_w).to(DEVICE)
    pts_gamma, normals_gamma = geom.sample_gamma(n_gamma)
    pts_gamma = pts_gamma.to(DEVICE)
    normals_gamma = normals_gamma.to(DEVICE)
    pts_b = geom.sample_boundary_box(n_b).to(DEVICE)

    history = []

    for epoch in tqdm(range(1, n_epochs + 1)):
        if epoch % resample_every == 0:
            pts_m = geom.sample_omega_m(n_m).to(DEVICE)
            pts_w = geom.sample_omega_w(n_w).to(DEVICE)
            pts_gamma, normals_gamma = geom.sample_gamma(n_gamma)
            pts_gamma = pts_gamma.to(DEVICE)
            normals_gamma = normals_gamma.to(DEVICE)
            pts_b = geom.sample_boundary_box(n_b).to(DEVICE)

        optimizer.zero_grad()

        res_m = pde_residual_m(model, pts_m).to(DEVICE)
        res_w = pde_residual_w(model, pts_w).to(DEVICE)
        res_gu, res_gf = interface_residuals(model, pts_gamma, normals_gamma)
        res_gu = res_gu.to(DEVICE)
        res_gf = res_gf.to(DEVICE)
        res_b = boundary_residual(model, pts_b).to(DEVICE)

        loss_m = (res_m ** 2).mean()
        loss_w = (res_w ** 2).mean()
        loss_gu = (res_gu ** 2).mean()
        loss_gf = (res_gf ** 2).mean()
        loss_bc = (res_b ** 2).mean()

        if epoch % reweight_every == 0:
            balancer.update({"m": loss_m, "w": loss_w, "gamma_u": loss_gu,
                              "gamma_flux": loss_gf, "bc": loss_bc}, model)

        w = balancer.weights
        total_loss = (w["m"] * loss_m + w["w"] * loss_w +
                      w["gamma_u"] * loss_gu + w["gamma_flux"] * loss_gf +
                      w["bc"] * loss_bc)

        total_loss.backward()
        optimizer.step()

        if epoch % verbose_every == 0 or epoch == 1:
            l2_err = validation_error(model, geom)
            history.append((epoch, total_loss.item(), l2_err))
            print(f"epoch {epoch:5d} | total_loss {total_loss.item():.4e} | "
                  f"loss_m {loss_m.item():.2e} loss_w {loss_w.item():.2e} "
                  f"gamma_u {loss_gu.item():.2e} gamma_flux {loss_gf.item():.2e} "
                  f"bc {loss_bc.item():.2e} | rel_L2_err {l2_err:.4e} | "
                  f"weights {[f'{k}:{v:.2f}' for k, v in w.items()]}")

    return model, geom, history


@torch.no_grad()
def validation_error(model, geom, n_test=4000, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    box = (torch.rand(n_test, 2, generator=g, device=DEVICE) * 2 - 1) * geom.L
    inside = geom.is_inside(box)

    u_pred = torch.empty(n_test, device=DEVICE)
    u_true = torch.empty(n_test, device=DEVICE)
    u_pred[inside] = model.u_m(box[inside])
    u_pred[~inside] = model.u_w(box[~inside])
    u_true[inside] = u_m_exact(box[inside])
    u_true[~inside] = u_w_exact(box[~inside])

    rel_l2 = torch.sqrt(((u_pred - u_true) ** 2).mean() / (u_true ** 2).mean())
    return rel_l2.item()


if __name__ == "__main__":
    model, geom, history = train()
