"""
Phase 2: real point charge + Green's-function singularity regularization,
nonlinear PBE, 3D Born ion sphere. No manufactured solution this time -- the
network solves the actual homogeneous regularized nonlinear PBE, and is
validated against the independent radial-ODE numerical solver in
rna_pinn_phase2_reference.py (scipy solve_bvp).

Reuses TrainableTanh / FourierFeatures / EnhancedBranch from Phase 1 unchanged.
"""

import torch
import numpy as np
from rna_pinn_phase1 import TrainableTanh, FourierFeatures, EnhancedBranch
import rna_pinn_phase2_reference as ref

torch.set_default_dtype(torch.float64)

R, L = ref.R, ref.L
EPS_M, EPS_W, KAPPA_W, Q = ref.EPS_M, ref.EPS_W, ref.KAPPA_W, ref.Q
DEVICE = 'cuda'
# Interface jump targets, corrected signs (verified symbolically above)
JUMP_U = Q / (4 * np.pi * EPS_M * R)          # u_r(R+) - u_r(R-)
JUMP_FLUX = -Q / (4 * np.pi * R ** 2)         # eps_w*du_r/dn|_{R+} - eps_m*du_r/dn|_{R-}


# ---------------------------------------------------------------------------
# 3D sphere geometry
# ---------------------------------------------------------------------------
class SphereInterface:
    def __init__(self, R=R, L=L):
        self.R, self.L = R, L

    def _rand_unit_vectors(self, n):
        v = torch.randn(n, 3)
        return v / v.norm(dim=1, keepdim=True)

    def sample_omega_m(self, n):
        """Uniform in the ball of radius R (interior, solute)."""
        u = self._rand_unit_vectors(n)
        r = self.R * torch.rand(n, 1) ** (1.0 / 3.0)   # correct radial density in 3D
        pts = u * r
        return pts.clone().requires_grad_(True)

    def sample_omega_w(self, n):
        """Uniform in the shell R < r < L (exterior, solvent, truncated)."""
        u = self._rand_unit_vectors(n)
        v = torch.rand(n, 1)
        r3 = self.R ** 3 + v.squeeze(-1) * (self.L ** 3 - self.R ** 3)
        r = r3 ** (1.0 / 3.0)
        pts = u * r.unsqueeze(-1)
        return pts.clone().requires_grad_(True)

    def sample_gamma(self, n):
        u = self._rand_unit_vectors(n)
        pts = (u * self.R).clone().requires_grad_(True)
        normals = u  # outward radial unit normal, exact
        return pts, normals

    def sample_boundary(self, n):
        u = self._rand_unit_vectors(n)
        return (u * self.L).clone().requires_grad_(True)


# ---------------------------------------------------------------------------
# Model: two branches, 3D input
# ---------------------------------------------------------------------------
class BornIonPINN(torch.nn.Module):
    def __init__(self, geom: SphereInterface):
        super().__init__()
        self.geom = geom
        # Output range hyperparameters (Achondo's Born-ion approximation, eq.17,
        # would normally set these; here a generous manual bound suffices since
        # we already know the reference magnitude is ~0.04).
        self.Nm = EnhancedBranch(xmin=[-geom.R] * 3, xmax=[geom.R] * 3,
                                  ymin_out=-0.15, ymax_out=0.15, in_dim=3)
        self.Nw = EnhancedBranch(xmin=[-geom.L] * 3, xmax=[geom.L] * 3,
                                  ymin_out=-0.15, ymax_out=0.15, in_dim=3)

    def u_m(self, xyz):
        return self.Nm(xyz)

    def u_w(self, xyz):
        return self.Nw(xyz)


def laplacian_3d(u, xyz):
    grad_u = torch.autograd.grad(u, xyz, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    lap = 0.0
    for i in range(3):
        g2 = torch.autograd.grad(grad_u[:, i], xyz, grad_outputs=torch.ones_like(grad_u[:, i]),
                                  create_graph=True)[0]
        lap = lap + g2[:, i]
    return lap, grad_u


def pde_residual_m(model, xyz):
    """Homogeneous Laplace equation inside (no charge left after regularization,
    kappa_m = 0). NOTE: no manufactured forcing term -- this is the real physics."""
    xyz = xyz.clone().requires_grad_(True)
    u = model.u_m(xyz)
    lap, _ = laplacian_3d(u, xyz)
    return -EPS_M * lap


def pde_residual_w(model, xyz):
    """Real nonlinear PBE, homogeneous (no manufactured forcing)."""
    xyz = xyz.clone().requires_grad_(True)
    u = model.u_w(xyz)
    lap, _ = laplacian_3d(u, xyz)
    return -EPS_W * lap + KAPPA_W ** 2 * torch.sinh(u)


def interface_residuals(model, xyz_gamma, normals):
    xyz_gamma = xyz_gamma.clone().requires_grad_(True)
    u_m = model.u_m(xyz_gamma)
    u_w = model.u_w(xyz_gamma)
    grad_m = torch.autograd.grad(u_m, xyz_gamma, grad_outputs=torch.ones_like(u_m),
                                  create_graph=True)[0]
    grad_w = torch.autograd.grad(u_w, xyz_gamma, grad_outputs=torch.ones_like(u_w),
                                  create_graph=True)[0]
    dun_m = (grad_m * normals).sum(dim=1)
    dun_w = (grad_w * normals).sum(dim=1)

    res_u = (u_w - u_m) - JUMP_U
    res_flux = (EPS_W * dun_w - EPS_M * dun_m) - JUMP_FLUX
    return res_u, res_flux


def yukawa_target(xyz):
    """Debye-screened far-field target (Achondo eq. 12 analogue), rather than
    plain zero -- more correct even though at r=L it is numerically tiny here."""
    r = xyz.norm(dim=1)
    return Q * torch.exp(-KAPPA_W * (r - R)) / (4 * np.pi * EPS_W * r)


def boundary_residual(model, xyz_b):
    u_pred = model.u_w(xyz_b)
    return u_pred - yukawa_target(xyz_b)


class LossBalancer:
    def __init__(self, names, alpha=0.7):
        self.alpha = alpha
        self.weights = {n: 1.0 for n in names}

    def update(self, per_term_losses, model):
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


@torch.no_grad()
def symmetry_check(model, r=0.5, n_dirs=6):
    """Cross-check the PINN's solution is actually spherically symmetric --
    it is never told to be; that has to emerge purely from training on
    uniformly-sampled directions. A big spread here would mean the network
    has learned a direction-dependent artifact, not the true radial physics."""
    dirs = torch.randn(n_dirs, 3, device=DEVICE)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    pts = dirs * r
    branch = model.u_m if r < R else model.u_w
    vals = branch(pts)
    return vals.std().item(), vals.mean().item()


def train(n_epochs=8000, n_m=600, n_w=1200, n_gamma=400, n_b=400,
          resample_every=100, reweight_every=200, lr_init=1e-3, lr_final=1e-5,
          verbose_every=300, checkpoint_path=None, resume_from=None, stop_at=None):
    """
    n_epochs   : total length of the intended run -- fixes the LR decay schedule.
    stop_at    : actually stop after this many epochs (<= n_epochs), saving a
                 checkpoint so a later call can resume up to n_epochs. Lets a
                 long run be split into several shorter, timeout-safe calls.
    """
    stop_at = stop_at or n_epochs

    geom = SphereInterface()
    model = BornIonPINN(geom).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    balancer = LossBalancer(["m", "w", "gamma_u", "gamma_flux", "bc"])
    start_epoch = 1

    if resume_from is not None:
        ckpt = torch.load(resume_from, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # IMPORTANT: optimizer.load_state_dict restores the *already-decayed*
        # LR from the previous chunk. ExponentialLR treats whatever LR the
        # optimizer has *right now* as its epoch-0 base -- so if we don't
        # reset it, fast-forwarding below compounds the decay a second time
        # on top of the first chunk's decay. Reset to lr_init first.
        for pg in optimizer.param_groups:
            pg["lr"] = lr_init
        balancer.weights = ckpt["balancer_weights"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    # Exponential decay across the full intended run (Achondo et al.: "Adam
    # with an exponentially decaying learning rate starting from 0.001").
    gamma = (lr_final / lr_init) ** (1.0 / n_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # fast-forward the scheduler if resuming
    for _ in range(start_epoch - 1):
        scheduler.step()

    sol, u_in_ref = ref.solve_reference()

    pts_m = geom.sample_omega_m(n_m).to(DEVICE)
    pts_w = geom.sample_omega_w(n_w).to(DEVICE)
    pts_gamma, normals_gamma = geom.sample_gamma(n_gamma)
    pts_gamma = pts_gamma.to(DEVICE)
    normals_gamma = normals_gamma.to(DEVICE)
    pts_b = geom.sample_boundary(n_b).to(DEVICE)

    for epoch in range(start_epoch, stop_at + 1):
        if epoch % resample_every == 0:
            pts_m = geom.sample_omega_m(n_m).to(DEVICE)
            pts_w = geom.sample_omega_w(n_w).to(DEVICE)
            pts_gamma, normals_gamma = geom.sample_gamma(n_gamma)
            pts_gamma = pts_gamma.to(DEVICE)
            normals_gamma = normals_gamma.to(DEVICE)
            pts_b = geom.sample_boundary(n_b).to(DEVICE)

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
        scheduler.step()

        if epoch % verbose_every == 0 or epoch == 1:
            with torch.no_grad():
                r_test = torch.linspace(0.05, L, 200, device=DEVICE)
                pts_test = torch.zeros(200, 3, device=DEVICE)
                pts_test[:, 0] = r_test  # sample along the x-axis
                inside = r_test < R
                u_pred = torch.empty(200, device=DEVICE)
                u_pred[inside] = model.u_m(pts_test[inside])
                u_pred[~inside] = model.u_w(pts_test[~inside])
                u_true = torch.tensor(ref.u_r_reference(r_test.cpu().numpy(), sol, u_in_ref), device=DEVICE)
                rel_l2 = torch.sqrt(((u_pred - u_true) ** 2).mean() / (u_true ** 2).mean()).item()
                dG_pred = 0.5 * Q * model.u_m(torch.zeros(1, 3, device=DEVICE)).item()
                dG_ref = ref.solvation_energy_reference(u_in_ref)
            sym_std, sym_mean = symmetry_check(model, r=2.0)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"epoch {epoch:5d} | lr {cur_lr:.2e} | total_loss {total_loss.item():.3e} | "
                  f"rel_L2_err(radial) {rel_l2:.3e} | "
                  f"dG_solv pred={dG_pred:.5f} ref={dG_ref:.5f} "
                  f"(rel err {abs(dG_pred-dG_ref)/abs(dG_ref):.2%}) | "
                  f"symmetry_std/mean at r=2: {sym_std:.2e}/{sym_mean:.2e}")
            if checkpoint_path is not None:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "balancer_weights": balancer.weights, "epoch": epoch}, checkpoint_path)

    if checkpoint_path is not None:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "balancer_weights": balancer.weights, "epoch": stop_at}, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path} at epoch {stop_at}")

    return model, geom, sol, u_in_ref


if __name__ == "__main__":
    train()