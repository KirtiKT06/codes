"""
Diagnostic script for the stuck gamma_flux term seen in the 30k-epoch run.

Three diagnostics, each answering a specific question:

  1. WEIGHT HISTORY: was gamma_flux actually up-weighted by the adaptive
     balancer the way it should have been if the optimizer "knew" it needed
     more attention? (This was silently computed but never logged before --
     that's a real blind spot, fixed here.)

  2. SPATIAL RESIDUAL MAP: is the flux residual uniformly bad everywhere on
     the surface, or concentrated in specific regions (e.g. near densely
     packed backbone/phosphate atoms, which would support the domain-scaling
     /resolution explanation)?

  3. CONTROLLED COMPARISON: same short run length, baseline architecture vs.
     a resolution-boosted one (more Fourier features, higher frequency scale,
     more interface collocation points). If gamma_flux's DECAY RATE improves
     with more capacity, that confirms it's a resolution/capacity problem,
     not, say, a sign error or a fundamentally unlearnable target.

Run with:
    python3 rna_pinn_phase4_diagnostic.py --epochs 3000
"""

import argparse
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from rna_pinn_phase1 import TrainableTanh, FourierFeatures
from rna_pinn_phase4_mesh import parse_pqr, MoleculeGeometry
from rna_pinn_phase4_gpu import (RealChargeSystem, RealMoleculeBranch, RealMoleculePINN,
                                  laplacian_3d, sample_omega_w_np, sample_boundary_np,
                                  LossBalancer, EPS_M, EPS_W, KAPPA_W, DEVICE)


def build_geometry(pqr_path="4tna_out.pqr", grid_spacing=1.0):
    coords, charges_arr, radii = parse_pqr(pqr_path)
    geom = MoleculeGeometry(coords, radii, grid_spacing=grid_spacing)
    geom.build_surface()
    geom.build_volume()
    charges = RealChargeSystem(coords, charges_arr)
    return geom, charges, coords, charges_arr, radii


def run_diagnostic(geom, charges, n_epochs, n_fourier, sigma, n_gamma, label,
                    n_m=2000, n_w=4000, n_b=600, lr_init=1e-3, lr_final=1e-4,
                    resample_every=100, reweight_every=100, verbose_every=250):
    """One controlled run, returns the full weight/loss history for comparison."""

    class BoostedBranch(RealMoleculeBranch):
        def __init__(self, xmin, xmax, ymin_out, ymax_out):
            super().__init__(xmin, xmax, ymin_out, ymax_out, n_fourier=n_fourier, sigma=sigma)

    class BoostedPINN(RealMoleculePINN):
        def __init__(self, geom, charges):
            torch.nn.Module.__init__(self)
            bbox_lo = geom.tet_nodes.min(axis=0)
            bbox_hi = geom.tet_nodes.max(axis=0)
            centroid = geom.coords.mean(axis=0)
            self.centroid = centroid
            self.L_outer = np.linalg.norm(geom.coords - centroid, axis=1).max() + 15.0
            ymin, ymax = charges.born_ion_output_range(geom.radii.clip(min=0.5))
            self.Nm = BoostedBranch(bbox_lo, bbox_hi, ymin, ymax)
            outer_lo, outer_hi = centroid - self.L_outer, centroid + self.L_outer
            self.Nw = BoostedBranch(outer_lo, outer_hi, ymin, ymax)

    model = BoostedPINN(geom, charges).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    balancer = LossBalancer(["m", "w", "gamma_u", "gamma_flux", "bc"])
    gamma_decay = (lr_final / lr_init) ** (1.0 / n_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_decay)

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

    def sample_gamma_np(n):
        pts, face_idx = trimesh.sample.sample_surface(geom.surf_mesh, n)
        normals = geom.surf_mesh.face_normals[face_idx]
        return np.asarray(pts), np.asarray(normals)

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
    history = {"epoch": [], "loss_gamma_flux": [], "weight_gamma_flux": [],
               "loss_gamma_u": [], "weight_gamma_u": []}

    print(f"\n=== Run '{label}': n_fourier={n_fourier}, sigma={sigma}, n_gamma={n_gamma} ===")
    for epoch in range(1, n_epochs + 1):
        if epoch % resample_every == 0:
            pts_m, pts_w, g_pts, g_norm, pts_b = resample()

        optimizer.zero_grad()
        u_m = model.u_m(pts_m)
        res_m = -EPS_M * laplacian_3d(u_m, pts_m)
        u_w = model.u_w(pts_w)
        res_w = -EPS_W * laplacian_3d(u_w, pts_w) + KAPPA_W ** 2 * torch.sinh(u_w)

        u_m_g = model.u_m(g_pts)
        u_w_g = model.u_w(g_pts)
        grad_m = torch.autograd.grad(u_m_g, g_pts, grad_outputs=torch.ones_like(u_m_g), create_graph=True)[0]
        grad_w = torch.autograd.grad(u_w_g, g_pts, grad_outputs=torch.ones_like(u_w_g), create_graph=True)[0]
        dun_m = (grad_m * g_norm).sum(dim=1)
        dun_w = (grad_w * g_norm).sum(dim=1)
        u_s_val = charges.u_s(g_pts)
        grad_us_val = charges.grad_u_s(g_pts)
        dun_us = (grad_us_val * g_norm).sum(dim=1)
        res_gu = (u_w_g - u_m_g) - u_s_val
        res_gf = (EPS_W * dun_w - EPS_M * dun_m) - EPS_M * dun_us

        res_b = model.u_w(pts_b)

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

        if torch.isnan(total_loss):
            print(f"epoch {epoch}: NaN -- stopping this run")
            break

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % verbose_every == 0 or epoch == 1:
            history["epoch"].append(epoch)
            history["loss_gamma_flux"].append(loss_gf.item())
            history["weight_gamma_flux"].append(w["gamma_flux"])
            history["loss_gamma_u"].append(loss_gu.item())
            history["weight_gamma_u"].append(w["gamma_u"])
            print(f"epoch {epoch:5d} | total {total_loss.item():.3e} | "
                  f"loss_gamma_flux {loss_gf.item():.3e} (weight {w['gamma_flux']:.2f}) | "
                  f"loss_gamma_u {loss_gu.item():.3e} (weight {w['gamma_u']:.2f})")

    return model, history


def spatial_residual_map(model, geom, charges, coords, n_gamma=5000, top_pct=1.0):
    """Where on the real molecule is the flux residual worst? Reports the
    nearest real atoms to the highest-residual surface points."""
    pts, face_idx = trimesh.sample.sample_surface(geom.surf_mesh, n_gamma)
    normals_np = geom.surf_mesh.face_normals[face_idx]
    g_pts = torch.tensor(np.asarray(pts), device=DEVICE, requires_grad=True)
    g_norm = torch.tensor(np.asarray(normals_np), device=DEVICE)

    u_m_g = model.u_m(g_pts)
    u_w_g = model.u_w(g_pts)
    grad_m = torch.autograd.grad(u_m_g, g_pts, grad_outputs=torch.ones_like(u_m_g), create_graph=True)[0]
    grad_w = torch.autograd.grad(u_w_g, g_pts, grad_outputs=torch.ones_like(u_w_g), create_graph=True)[0]
    dun_m = (grad_m * g_norm).sum(dim=1)
    dun_w = (grad_w * g_norm).sum(dim=1)
    grad_us_val = charges.grad_u_s(g_pts)
    dun_us = (grad_us_val * g_norm).sum(dim=1)
    res_gf = (EPS_W * dun_w - EPS_M * dun_m) - EPS_M * dun_us
    res_abs = res_gf.detach().cpu().numpy() ** 2

    print(f"\n=== Spatial flux-residual breakdown ({n_gamma} surface points) ===")
    pctiles = [50, 90, 99, 99.9]
    for p in pctiles:
        print(f"  {p}th percentile of squared flux residual: {np.percentile(res_abs, p):.4e}")

    n_top = max(1, int(n_gamma * top_pct / 100))
    top_idx = np.argsort(res_abs)[-n_top:]
    top_pts = np.asarray(pts)[top_idx]

    tree = cKDTree(coords)
    with open("4tna_out.pqr") as f:
        lines = [l for l in f if l.startswith("ATOM") or l.startswith("HETATM")]
    print(f"\nTop {top_pct}% worst-residual points -- nearest real atom for a sample of them:")
    for pt in top_pts[:: max(1, len(top_pts) // 15)]:
        d, idx = tree.query(pt, k=1)
        parts = lines[idx].split()
        print(f"  point {pt} -> nearest atom: resname={parts[3]} resnum={parts[4]} "
              f"atomname={parts[2]} dist={d:.2f}A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--pqr", default="4tna_out.pqr")
    args = parser.parse_args()

    geom, charges, coords, charges_arr, radii = build_geometry(args.pqr)
    print(f"Mesh: {len(geom.tet_nodes)} nodes, {len(geom.surf_mesh.vertices)} surface verts")

    # Baseline: same settings as the 30k-epoch run that got stuck
    model_base, hist_base = run_diagnostic(
        geom, charges, n_epochs=args.epochs, n_fourier=256, sigma=4.0, n_gamma=1200,
        label="baseline (matches the stuck 30k run's settings)")

    # Boosted: more capacity specifically aimed at the interface
    model_boost, hist_boost = run_diagnostic(
        geom, charges, n_epochs=args.epochs, n_fourier=512, sigma=12.0, n_gamma=3500,
        label="boosted (more Fourier features/frequency, more interface points)")

    print("\n=== Comparison: gamma_flux loss trajectory ===")
    print(f"{'epoch':>8} | {'baseline':>12} | {'boosted':>12}")
    for i in range(len(hist_base["epoch"])):
        print(f"{hist_base['epoch'][i]:>8} | {hist_base['loss_gamma_flux'][i]:>12.3e} | "
              f"{hist_boost['loss_gamma_flux'][i]:>12.3e}")

    base_slope = (np.log(hist_base["loss_gamma_flux"][-1] + 1e-30) -
                  np.log(hist_base["loss_gamma_flux"][0] + 1e-30))
    boost_slope = (np.log(hist_boost["loss_gamma_flux"][-1] + 1e-30) -
                   np.log(hist_boost["loss_gamma_flux"][0] + 1e-30))
    print(f"\nLog-decay of gamma_flux over the run: baseline={base_slope:.3f}, boosted={boost_slope:.3f}")
    print("(more negative = decayed more; if boosted is meaningfully more negative, "
          "that confirms this is a resolution/capacity issue, not something else)")

    print("\n=== Spatial residual map (boosted model) ===")
    spatial_residual_map(model_boost, geom, charges, coords)
