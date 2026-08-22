"""
Phase 4, architectural revision: single shared network, value-jump condition
hard-constrained (satisfied exactly, not trained), instead of the two-branch
architecture used until now.

WHY (see chat for full derivation): the diagnostic showed u_m had collapsed
to a near-constant function (span ~0.1 across the whole molecule). Root
cause: Omega_m's own PDE is just Laplace's equation (kappa_m=0, singularity
already removed) -- and ANY constant trivially satisfies Laplace's equation
with zero residual. Omega_w's homogeneous nonlinear PBE similarly has a
trivial solution (u_w=0). The ONLY thing that could force non-trivial
structure was the interface conditions -- and those were consistently the
worst-satisfied terms in every diagnostic. Two independent networks each had
an easy escape route, and interface coupling was too weak to prevent it.

FIX: subtract the Green's function g_C EVERYWHERE (not just in Omega_m, the
convention used until now). g_C is smooth away from the charges (all inside
Omega_m), so this introduces no new singularity. The resulting reaction
potential u_r = phi - g_C is then CONTINUOUS across Gamma as a mathematical
fact (g_C is the same smooth function evaluated from both sides) -- so a
SINGLE network representing u_r automatically satisfies the value-jump
condition exactly, removing gamma_u from the loss and removing the
"two independent trivial solutions" escape route, since Omega_m and Omega_w
physics now constrain the SAME weights.

The flux condition still needs a soft loss term (the true physical solution
has a genuine kink in the gradient at Gamma, which a smooth network can't
represent exactly) -- but it's now a single condition on one network's
gradient, not a difference between two separately-scaled branches.

Derivation of the new residuals (u_r = U(x) everywhere, phi = U(x) + g_C(x)):
  Omega_m: -eps_m * Laplacian(U) = 0                              (unchanged)
  Omega_w: -eps_w * Laplacian(U) + kappa^2 * sinh(U + g_C) = 0     (sinh acts
           on the TOTAL potential U+g_C now, not U alone -- g_C is harmonic
           in Omega_w since all charges are in Omega_m, so Laplacian(g_C)=0
           there and only contributes inside the sinh nonlinearity)
  Gamma flux (physical D-field continuity, target is exactly ZERO now --
           no separate "jump target" bookkeeping needed):
           eps_w*(dU/dn + dgC/dn) - eps_m*dU/dn = 0
  Boundary: U(x_b) + g_C(x_b) ~= Yukawa-screened far-field target

NOTE: this file was rebuilt after a sandbox environment reset wiped the
working directory mid-testing. The logic is unchanged from what was smoke-
tested (ran cleanly for the first training steps, no crash/NaN) -- re-verify
below before treating it as final.
"""

import argparse
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from rna_pinn_phase1 import TrainableTanh, FourierFeatures
from rna_pinn_phase4_mesh import parse_pqr, MoleculeGeometry
from rna_pinn_phase4_gpu import (RealChargeSystem, get_phosphate_coords, build_gamma_sampler,
                                  sample_omega_w_np, sample_boundary_np, LossBalancer,
                                  EPS_M, EPS_W, KAPPA_W, DEVICE)

torch.set_default_dtype(torch.float64)


class SharedBranch(torch.nn.Module):
    """One network, scaled over the FULL domain (Omega_m is a strict subset
    of Omega_w's bounding sphere), representing u_r everywhere."""

    def __init__(self, xmin, xmax, ymin_out, ymax_out, n_fourier=512, sigma=12.0,
                 hidden=(128, 128, 128)):
        super().__init__()
        self.register_buffer("xmin", torch.tensor(xmin))
        self.register_buffer("xmax", torch.tensor(xmax))
        self.ymin_out, self.ymax_out = ymin_out, ymax_out
        self.fourier = FourierFeatures(3, n_fourier, sigma)
        dims = [2 * n_fourier] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            layers.append(TrainableTanh(dims[i + 1]))
        self.hidden = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(dims[-1], 1)

    def input_scale(self, x):
        return 2 * (x - self.xmin) / (self.xmax - self.xmin) - 1

    def output_scale(self, y):
        return 0.5 * (y + 1) * (self.ymax_out - self.ymin_out) + self.ymin_out

    def forward(self, x):
        h = self.hidden(self.fourier(self.input_scale(x)))
        return self.output_scale(self.out(h)).squeeze(-1)


class SharedPINN(torch.nn.Module):
    """Single network U(x) = u_r(x), valid everywhere. u_m and u_w are no
    longer separate objects -- this IS both, by construction."""

    def __init__(self, geom: MoleculeGeometry, charges: RealChargeSystem, L_outer_margin=15.0):
        super().__init__()
        centroid = geom.coords.mean(axis=0)
        self.centroid = centroid
        self.L_outer = np.linalg.norm(geom.coords - centroid, axis=1).max() + L_outer_margin
        ymin, ymax = charges.born_ion_output_range(geom.radii.clip(min=0.5))
        print(f"Born-ion-superposition output range estimate: [{ymin:.4f}, {ymax:.4f}]")
        outer_lo = centroid - self.L_outer
        outer_hi = centroid + self.L_outer
        self.U = SharedBranch(xmin=outer_lo, xmax=outer_hi, ymin_out=ymin, ymax_out=ymax)

    def u(self, xyz):
        return self.U(xyz)


def laplacian_3d(u, xyz):
    grad_u = torch.autograd.grad(u, xyz, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    lap = 0.0
    for i in range(3):
        g2 = torch.autograd.grad(grad_u[:, i], xyz, grad_outputs=torch.ones_like(grad_u[:, i]),
                                  create_graph=True)[0]
        lap = lap + g2[:, i]
    return lap


def yukawa_boundary_target(xyz, charges: RealChargeSystem):
    out = torch.zeros(xyz.shape[0], dtype=torch.float64, device=DEVICE)
    for i in range(0, len(charges.charges), charges.chunk):
        c = charges.centers[i:i + charges.chunk]
        q = charges.charges[i:i + charges.chunk]
        diffs = xyz.unsqueeze(1) - c.unsqueeze(0)
        dists = diffs.norm(dim=2).clamp_min(1e-3)
        out = out + (q.unsqueeze(0) * torch.exp(-KAPPA_W * dists) / (4 * np.pi * EPS_W * dists)).sum(dim=1)
    return out


def train(pqr_path="4tna_out.pqr", n_epochs=30000, n_m=2000, n_w=4000, n_gamma=1200, n_b=600,
          resample_every=100, reweight_every=200, lr_init=1e-3, lr_final=1e-6,
          verbose_every=250, checkpoint_path="rna_phase4_hardconstraint_checkpoint.pt",
          resume_from=None, grid_spacing=1.0, min_radius=1.3):

    print("Loading real structure and mesh...")
    coords, charges_arr, radii = parse_pqr(pqr_path)
    geom = MoleculeGeometry(coords, radii, grid_spacing=grid_spacing, min_radius=min_radius)
    geom.build_surface()
    geom.build_volume()
    print(f"Mesh: {len(geom.tet_nodes)} nodes, {len(geom.tet_elems)} tets, "
          f"{len(geom.surf_mesh.vertices)} surface verts")

    charges = RealChargeSystem(coords, charges_arr)
    model = SharedPINN(geom, charges).to(DEVICE)
    print(f"Outer truncation radius: {model.L_outer:.1f} A around centroid")

    phosphate_coords = get_phosphate_coords(pqr_path)
    sample_gamma_np = build_gamma_sampler(geom, phosphate_coords)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    balancer = LossBalancer(["m", "w", "gamma_flux", "bc"])  # NOTE: no "gamma_u" -- hard-constrained
    start_epoch = 1

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr_init
        balancer.weights = ckpt["balancer_weights"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    gamma_decay = (lr_final / lr_init) ** (1.0 / n_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_decay)
    for _ in range(start_epoch - 1):
        scheduler.step()

    v = geom.tet_nodes[geom.tet_elems]
    tetvols = np.abs(np.einsum('ij,ij->i', v[:, 0] - v[:, 3],
                                np.cross(v[:, 1] - v[:, 3], v[:, 2] - v[:, 3]))) / 6.0
    tetvols = tetvols / tetvols.sum()

    def sample_omega_m_np(n):
        idx = np.random.choice(len(tetvols), size=n, p=tetvols)
        vv = v[idx]
        a, b, c, d = vv[:, 0], vv[:, 1], vv[:, 2], vv[:, 3]
        s, t, u = np.random.rand(n), np.random.rand(n), np.random.rand(n)
        m1 = s + t > 1
        s[m1], t[m1] = 1 - s[m1], 1 - t[m1]
        m2 = t + u > 1
        tmp = u[m2].copy(); u[m2] = 1 - s[m2] - t[m2]; t[m2] = 1 - tmp
        m3 = (s + t + u) > 1
        s2 = s[m3].copy(); s[m3] = 1 - t[m3] - u[m3]; t[m3] = 1 - s2
        return a + s[:, None] * (b - a) + t[:, None] * (c - a) + u[:, None] * (d - a)

    def resample():
        pts_m = torch.tensor(sample_omega_m_np(n_m), device=DEVICE, requires_grad=True)
        pts_w = torch.tensor(sample_omega_w_np(geom, model.centroid, model.L_outer, n_w),
                              device=DEVICE, requires_grad=True)
        g_pts_np, g_norm_np = sample_gamma_np(n_gamma)
        g_pts = torch.tensor(g_pts_np, device=DEVICE, requires_grad=True)
        g_norm = torch.tensor(g_norm_np, device=DEVICE)
        pts_b = torch.tensor(sample_boundary_np(model.centroid, model.L_outer, n_b),
                              device=DEVICE, requires_grad=True)
        return pts_m, pts_w, g_pts, g_norm, pts_b

    pts_m, pts_w, g_pts, g_norm, pts_b = resample()
    print(f"Starting training (hard-constrained value jump): {n_epochs} epochs, device={DEVICE}")

    for epoch in range(start_epoch, n_epochs + 1):
        if epoch % resample_every == 0:
            pts_m, pts_w, g_pts, g_norm, pts_b = resample()

        optimizer.zero_grad()

        # Omega_m: plain Laplace residual on the shared network
        u_m = model.u(pts_m)
        res_m = -EPS_M * laplacian_3d(u_m, pts_m)

        # Omega_w: nonlinear residual, sinh acts on U + g_C (the TOTAL potential)
        u_w = model.u(pts_w)
        g_c_w = charges.u_s(pts_w)
        res_w = -EPS_W * laplacian_3d(u_w, pts_w) + KAPPA_W ** 2 * torch.sinh(u_w + g_c_w)

        # Gamma: flux condition only -- value jump is automatically zero,
        # nothing to compute or penalize for it
        u_g = model.u(g_pts)
        grad_u = torch.autograd.grad(u_g, g_pts, grad_outputs=torch.ones_like(u_g), create_graph=True)[0]
        dun_u = (grad_u * g_norm).sum(dim=1)
        grad_gc = charges.grad_u_s(g_pts)
        dun_gc = (grad_gc * g_norm).sum(dim=1)
        # target is exactly 0: physical D-field continuity, no free surface charge
        res_flux = EPS_W * (dun_u + dun_gc) - EPS_M * dun_u

        # Boundary: total potential (U + g_C) ~= Yukawa far-field target
        u_b = model.u(pts_b)
        g_c_b = charges.u_s(pts_b)
        res_b = (u_b + g_c_b) - yukawa_boundary_target(pts_b, charges)

        loss_m = (res_m ** 2).mean()
        loss_w = (res_w ** 2).mean()
        loss_gf = (res_flux ** 2).mean()
        loss_bc = (res_b ** 2).mean()

        if epoch % reweight_every == 0:
            balancer.update({"m": loss_m, "w": loss_w, "gamma_flux": loss_gf, "bc": loss_bc}, model)

        w = balancer.weights
        total_loss = w["m"] * loss_m + w["w"] * loss_w + w["gamma_flux"] * loss_gf + w["bc"] * loss_bc

        if torch.isnan(total_loss):
            print(f"epoch {epoch}: NaN -- stopping.")
            break

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % verbose_every == 0 or epoch == 1:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"epoch {epoch:6d} | lr {cur_lr:.2e} | total {total_loss.item():.3e} | "
                  f"m {loss_m.item():.2e} w {loss_w.item():.2e} "
                  f"gamma_flux {loss_gf.item():.2e} bc {loss_bc.item():.2e}")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "balancer_weights": balancer.weights, "epoch": epoch,
                        "geometry_config": {"pqr_path": pqr_path, "grid_spacing": grid_spacing,
                                             "min_radius": min_radius}}, checkpoint_path)

    print("Training complete.")
    return model, geom, charges


def solvation_energy(model, charges: RealChargeSystem):
    """Delta G_solv = 0.5*sum(q_i * u_r(x_i)) -- u_r is now the SAME function
    for all charges (they're all in Omega_m), no branch selection needed."""
    with torch.no_grad():
        psi = model.u(charges.centers).cpu().numpy()
    return 0.5 * float((charges.charges.cpu().numpy() * psi).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pqr", default="4tna_out.pqr")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    model, geom, charges = train(pqr_path=args.pqr, n_epochs=args.epochs, resume_from=args.resume_from)
    dG = solvation_energy(model, charges)
    print(f"\nFinal estimated Delta G_solv: {dG:.4f} kT = {dG*0.593:.4f} kcal/mol")
    print("Compare against APBS's -15912 kcal/mol -- should be MUCH closer than the previous -2.23.")
