"""
Phase 4, GPU-ready training script.

Run this on a CUDA machine, e.g.:
    python3 rna_pinn_phase4_gpu.py --epochs 30000

This is the same physics/architecture validated in rna_pinn_phase4_train.py's
sanity check (Phases 1-3 validated the underlying method; Phase 4's sanity
check confirmed the real-data pipeline runs without crashing), with three
things added for a real run:
  1. CUDA device support (falls back to CPU automatically if no GPU found)
  2. Achondo et al.'s exponential LR decay (Phase 2/3 showed flat LR alone
     plateaus early -- this was NOT in the Phase 4 sanity-check version)
  3. Periodic checkpointing + resume, in case a long run gets interrupted

STILL OPEN (see chat discussion, not solved here):
  - No independent numerical reference at this scale (APBS comparison is
    the recommended next step, run separately)
  - Domain-scaling issue only partially mitigated (wider/higher-frequency
    Fourier bank), not fully solved -- a genuine multiscale architecture
    would be the real fix if results degrade near the outer boundary or
    are sensitive to the truncation radius
  - van der Waals surface used as an approximation of the true SES
    (NanoShaper unavailable in the dev sandbox this was built in -- worth
    installing on your own machine and comparing if accuracy matters)

Requires: torch, numpy, scipy, tetgen, scikit-image, trimesh, and
4tna_out.pqr (or your own pdb2pqr output) in the working directory.
"""

import argparse
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

from rna_pinn_phase1 import TrainableTanh, FourierFeatures
from rna_pinn_phase4_mesh import parse_pqr, MoleculeGeometry

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

EPS_M = 2.0
EPS_W = 80.0
KAPPA_W = 0.125  # 1/Angstrom, ~150 mM monovalent salt


class RealChargeSystem:
    def __init__(self, centers, charges, chunk_size=500):
        self.centers = torch.tensor(centers, dtype=torch.float64, device=DEVICE)
        self.charges = torch.tensor(charges, dtype=torch.float64, device=DEVICE)
        self.chunk = chunk_size
        self.scale = 1.0  # continuation/curriculum multiplier -- ramped during training
                          # (see chat: full-charge nonlinearity was likely trapping the
                          # optimizer in a small-magnitude local solution). u_s/grad_u_s
                          # use this; born_ion_* estimates deliberately do NOT, since those
                          # need to reflect the FULL final-target magnitude regardless of
                          # what stage of the curriculum training is currently active.

    def u_s(self, xyz):
        out = torch.zeros(xyz.shape[0], dtype=torch.float64, device=DEVICE)
        for i in range(0, len(self.charges), self.chunk):
            c = self.centers[i:i + self.chunk]
            q = self.charges[i:i + self.chunk] * self.scale
            diffs = xyz.unsqueeze(1) - c.unsqueeze(0)
            dists = diffs.norm(dim=2).clamp_min(1e-3)
            out = out + (q.unsqueeze(0) / (4 * np.pi * EPS_M * dists)).sum(dim=1)
        return out

    def grad_u_s(self, xyz):
        out = torch.zeros_like(xyz)
        for i in range(0, len(self.charges), self.chunk):
            c = self.centers[i:i + self.chunk]
            q = self.charges[i:i + self.chunk] * self.scale
            diffs = xyz.unsqueeze(1) - c.unsqueeze(0)
            dists = diffs.norm(dim=2).clamp_min(1e-3)
            g = -(q.unsqueeze(0) / (4 * np.pi * EPS_M * dists ** 3)).unsqueeze(-1) * diffs
            out = out + g.sum(dim=1)
        return out

    def born_ion_self_energy_per_atom(self, atom_radii):
        """Per-atom generalized-Born-style self-energy estimate -- a cheap,
        physically-motivated approximation of the reaction potential at each
        charge, used as an anti-collapse seeding target (see chat)."""
        centers_np = self.centers.cpu().numpy()
        charges_np = self.charges.cpu().numpy()
        bi_self = charges_np / (4 * np.pi) * (1.0 / (EPS_W * (1 + KAPPA_W * atom_radii) * atom_radii)
                                                - 1.0 / (EPS_M * atom_radii))
        return bi_self

    def born_ion_output_range(self, atom_radii):
        centers_np = self.centers.cpu().numpy()
        charges_np = self.charges.cpu().numpy()
        n = len(charges_np)
        bi_self = self.born_ion_self_energy_per_atom(atom_radii)
        idx = np.random.choice(n, size=min(n, 300), replace=False)
        cross_est = 0.0
        for i in idx:
            d = np.linalg.norm(centers_np - centers_np[i], axis=1)
            d[i] = np.inf
            cross_est = max(cross_est, np.abs(charges_np / (4 * np.pi * EPS_M * d)).sum())
        span = np.abs(bi_self).max() + cross_est
        return -1.2 * span, 1.2 * span


def laplacian_3d(u, xyz):
    grad_u = torch.autograd.grad(u, xyz, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    lap = 0.0
    for i in range(3):
        g2 = torch.autograd.grad(grad_u[:, i], xyz, grad_outputs=torch.ones_like(grad_u[:, i]),
                                  create_graph=True)[0]
        lap = lap + g2[:, i]
    return lap


class RealMoleculeBranch(torch.nn.Module):
    """Boosted capacity (512 features, higher frequency) vs. the original
    256/sigma=4 -- the diagnostic's boosted config was still improving at
    epoch 3000 where baseline had flatlined, so this is now the default."""

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


class RealMoleculePINN(torch.nn.Module):
    def __init__(self, geom: MoleculeGeometry, charges: RealChargeSystem, L_outer_margin=15.0,
                 output_scale_multiplier=15.0):
        """
        output_scale_multiplier: widens the Born-ion-superposition output
        range estimate by this factor. Testing whether the previous ~+/-2.3
        bound was itself the bottleneck (see chat: three very different
        fixes all landed near the same ~-2.2 kcal/mol answer, which points
        at a shared ceiling rather than a training-dynamics issue). Default
        15x is a cheap direct test, not a first-principles derivation --
        if this moves the result substantially, that confirms the
        hypothesis; the actual correct bound would need more careful work.
        """
        super().__init__()
        bbox_lo = geom.tet_nodes.min(axis=0)
        bbox_hi = geom.tet_nodes.max(axis=0)
        centroid = geom.coords.mean(axis=0)
        self.centroid = centroid
        self.L_outer = np.linalg.norm(geom.coords - centroid, axis=1).max() + L_outer_margin

        ymin, ymax = charges.born_ion_output_range(geom.radii.clip(min=0.5))
        ymin, ymax = ymin * output_scale_multiplier, ymax * output_scale_multiplier
        print(f"Born-ion-superposition output range estimate (x{output_scale_multiplier} widened): "
              f"[{ymin:.4f}, {ymax:.4f}]")

        self.Nm = RealMoleculeBranch(xmin=bbox_lo, xmax=bbox_hi, ymin_out=ymin, ymax_out=ymax)
        outer_lo = centroid - self.L_outer
        outer_hi = centroid + self.L_outer
        self.Nw = RealMoleculeBranch(xmin=outer_lo, xmax=outer_hi, ymin_out=ymin, ymax_out=ymax)

    def u_m(self, xyz):
        return self.Nm(xyz)

    def u_w(self, xyz):
        return self.Nw(xyz)


def get_phosphate_coords(pqr_path):
    """Coordinates of backbone phosphate atoms (P and non-bridging oxygens) --
    the diagnostic found these are where the flux residual concentrates,
    independent of the zero-radius-hydrogen issue. Used to bias interface
    collocation sampling toward this region."""
    names = {"P", "O1P", "O2P", "OP1", "OP2", "OP3"}
    coords = []
    with open(pqr_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                if parts[2] in names:
                    coords.append(tuple(map(float, parts[-5:-2])))
    return np.array(coords)


def build_gamma_sampler(geom, phosphate_coords, bias_radius=2.0, bias_fraction=0.5):
    """Returns a sampling function that draws bias_fraction of points from
    surface faces near phosphate atoms, and the rest uniformly -- rather than
    uniformly oversampling the whole molecule (wasteful) or the whole network
    (the diagnostic showed this is a localized problem)."""
    face_centroids = geom.surf_mesh.triangles.mean(axis=1)
    ptree = cKDTree(phosphate_coords)
    dists, _ = ptree.query(face_centroids, k=1)
    near_mask = dists < bias_radius
    near_face_idx = np.where(near_mask)[0]
    print(f"Phosphate-biased sampling: {len(near_face_idx)}/{len(face_centroids)} "
          f"surface faces within {bias_radius}A of a phosphate atom")

    if len(near_face_idx) == 0:
        def sampler(n):
            pts, face_idx = trimesh.sample.sample_surface(geom.surf_mesh, n)
            return np.asarray(pts), geom.surf_mesh.face_normals[face_idx]
        return sampler

    near_submesh = geom.surf_mesh.submesh([near_face_idx], append=True)

    def sampler(n):
        n_bias = int(n * bias_fraction)
        n_uniform = n - n_bias
        pts_u, face_idx_u = trimesh.sample.sample_surface(geom.surf_mesh, n_uniform)
        norms_u = geom.surf_mesh.face_normals[face_idx_u]
        pts_b, face_idx_b = trimesh.sample.sample_surface(near_submesh, n_bias)
        norms_b = near_submesh.face_normals[face_idx_b]
        pts = np.concatenate([np.asarray(pts_u), np.asarray(pts_b)], axis=0)
        norms = np.concatenate([np.asarray(norms_u), np.asarray(norms_b)], axis=0)
        return pts, norms

    return sampler


def sample_omega_w_np(geom, centroid, L_outer, n):
    pts = []
    got = 0
    while got < n:
        cand = centroid + (np.random.rand(n * 2, 3) * 2 - 1) * L_outer
        r = np.linalg.norm(cand - centroid, axis=1)
        in_ball = r < L_outer
        outside_solute = ~geom.is_inside(cand)
        mask = in_ball & outside_solute
        pts.append(cand[mask])
        got += mask.sum()
    return np.concatenate(pts, axis=0)[:n]


def sample_boundary_np(centroid, L_outer, n):
    v = np.random.randn(n, 3)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return centroid + v * L_outer


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


def train(pqr_path="4tna_out.pqr", n_epochs=30000, n_m=2000, n_w=4000, n_gamma=1200, n_b=600,
          resample_every=100, reweight_every=200, lr_init=1e-3, lr_final=1e-6,
          verbose_every=250, checkpoint_path="rna_phase4_checkpoint.pt", resume_from=None,
          grid_spacing=1.0, min_radius=1.3, output_scale_multiplier=15.0,
          charge_ramp_fraction=0.4, charge_start_scale=0.05):
    """
    charge_ramp_fraction: fraction of n_epochs spent ramping the charge
        magnitude from charge_start_scale up to 1.0 (full real charges).
    charge_start_scale: starting charge fraction. Rationale (see chat): the
        nonlinear sinh(u_w) term grows explosively, and at full charge
        magnitude this likely traps the optimizer in a small-magnitude,
        easy-but-wrong local solution (sinh(u)~=u there). Training first at
        reduced charge -- where the problem is closer to the already-
        validated Phase 2 regime -- lets the network find real structure
        before the nonlinearity gets hard, then continuation ramps up to
        the true problem using that as a warm start.
    """

    print("Loading real structure and mesh...")
    coords, charges_arr, radii = parse_pqr(pqr_path)
    geom = MoleculeGeometry(coords, radii, grid_spacing=grid_spacing, min_radius=min_radius)
    geom.build_surface()
    geom.build_volume()
    print(f"Mesh: {len(geom.tet_nodes)} nodes, {len(geom.tet_elems)} tets, "
          f"{len(geom.surf_mesh.vertices)} surface verts")

    charges = RealChargeSystem(coords, charges_arr)
    model = RealMoleculePINN(geom, charges, output_scale_multiplier=output_scale_multiplier).to(DEVICE)
    print(f"Outer truncation radius: {model.L_outer:.1f} A around centroid")

    charge_ramp_epochs = max(1, int(charge_ramp_fraction * n_epochs))
    print(f"Charge curriculum: ramping from {charge_start_scale} to 1.0 over "
          f"the first {charge_ramp_epochs} epochs")

    # Anti-collapse seeding target: cheap per-atom Born-ion self-energy
    # estimate, used to pull u_m away from the trivial constant solution
    # early in training (see chat -- diagnosed root cause of the previous
    # run's catastrophically wrong Delta G_solv). Weight decays over
    # training so the real physics dominates once it has taken hold.
    bi_self_np = charges.born_ion_self_energy_per_atom(radii.clip(min=0.5))
    seed_targets = torch.tensor(bi_self_np, dtype=torch.float64, device=DEVICE)
    seed_points = torch.tensor(coords, dtype=torch.float64, device=DEVICE)
    print(f"Seed target range: [{bi_self_np.min():.4f}, {bi_self_np.max():.4f}] "
          f"(compare to the network's collapsed range from the previous run: [0.056, 0.152])")

    phosphate_coords = get_phosphate_coords(pqr_path)
    print(f"Found {len(phosphate_coords)} phosphate-group atoms for biased sampling")
    sample_gamma_np = build_gamma_sampler(geom, phosphate_coords)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    balancer = LossBalancer(["m", "w", "gamma_u", "gamma_flux", "bc"])
    start_epoch = 1

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr_init  # reset before fast-forwarding -- see Phase 2 bug writeup
        balancer.weights = ckpt["balancer_weights"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    gamma = (lr_final / lr_init) ** (1.0 / n_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
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
    print(f"Starting real training run: {n_epochs} epochs, device={DEVICE}")

    for epoch in range(start_epoch, n_epochs + 1):
        if epoch % resample_every == 0:
            pts_m, pts_w, g_pts, g_norm, pts_b = resample()

        # Charge continuation: linearly ramp from charge_start_scale to 1.0
        # over charge_ramp_epochs, then hold at 1.0 for the rest of training.
        charges.scale = min(1.0, charge_start_scale +
                             (1.0 - charge_start_scale) * epoch / charge_ramp_epochs)

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

        res_b = model.u_w(pts_b)  # far-field target ~0 given kappa*L_outer >> 1

        loss_m = (res_m ** 2).mean()
        loss_w = (res_w ** 2).mean()
        loss_gu = (res_gu ** 2).mean()
        loss_gf = (res_gf ** 2).mean()
        loss_bc = (res_b ** 2).mean()

        # Anti-collapse seeding loss: NOT part of the adaptive balancer --
        # this is a temporary scaffold with its own fast-decaying schedule,
        # not a permanent physics constraint. seed_decay_epochs controls how
        # fast it fades; by ~20% of the run it should be negligible.
        seed_decay_epochs = max(1, int(0.2 * n_epochs))
        seed_weight = 0.1 * max(0.0, 1.0 - epoch / seed_decay_epochs)
        if seed_weight > 0:
            u_m_seed = model.u_m(seed_points)
            loss_seed = ((u_m_seed - seed_targets * charges.scale) ** 2).mean()
        else:
            loss_seed = torch.tensor(0.0)

        if epoch % reweight_every == 0:
            balancer.update({"m": loss_m, "w": loss_w, "gamma_u": loss_gu,
                              "gamma_flux": loss_gf, "bc": loss_bc}, model)

        w = balancer.weights
        total_loss = (w["m"] * loss_m + w["w"] * loss_w +
                      w["gamma_u"] * loss_gu + w["gamma_flux"] * loss_gf +
                      w["bc"] * loss_bc + seed_weight * loss_seed)

        if torch.isnan(total_loss):
            print(f"epoch {epoch}: NaN detected -- stopping. Check collocation "
                  f"sampling and output-scaling bounds before resuming.")
            break

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % verbose_every == 0 or epoch == 1:
            cur_lr = scheduler.get_last_lr()[0]
            u_m_std = model.u_m(seed_points[:200]).std().item()  # cheap collapse check
            print(f"epoch {epoch:6d} | lr {cur_lr:.2e} | q_scale {charges.scale:.3f} | "
                  f"total {total_loss.item():.3e} | "
                  f"m {loss_m.item():.2e} w {loss_w.item():.2e} "
                  f"gamma_u {loss_gu.item():.2e} gamma_flux {loss_gf.item():.2e} "
                  f"bc {loss_bc.item():.2e} seed {loss_seed.item():.2e} (w={seed_weight:.3f}) | "
                  f"u_m_std {u_m_std:.4f}")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "balancer_weights": balancer.weights, "epoch": epoch,
                        "geometry_config": {"pqr_path": pqr_path, "grid_spacing": grid_spacing,
                                             "min_radius": min_radius}}, checkpoint_path)

    print("Training complete.")
    return model, geom, charges


def solvation_energy(model, coords, charges_arr):
    """Delta G_solv = 0.5 * sum(q_i * psi(x_i)), psi = u_m (reaction potential,
    since the singular part is excluded by construction -- Achondo et al. eq. 28)."""
    with torch.no_grad():
        pts = torch.tensor(coords, dtype=torch.float64, device=DEVICE)
        psi = model.u_m(pts).cpu().numpy()
    return 0.5 * float((charges_arr * psi).sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pqr", default="4tna_out.pqr")
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--output_scale_multiplier", type=float, default=15.0,
                         help="Widens the output-scaling bound by this factor, testing "
                              "whether the previous ~+/-2.3 bound was itself the bottleneck.")
    parser.add_argument("--charge_ramp_fraction", type=float, default=0.4,
                         help="Fraction of training spent ramping charge magnitude up to full.")
    parser.add_argument("--charge_start_scale", type=float, default=0.05,
                         help="Starting charge fraction for the continuation/curriculum schedule.")
    args = parser.parse_args()

    model, geom, charges = train(pqr_path=args.pqr, n_epochs=args.epochs, resume_from=args.resume_from,
                                  output_scale_multiplier=args.output_scale_multiplier,
                                  charge_ramp_fraction=args.charge_ramp_fraction,
                                  charge_start_scale=args.charge_start_scale)

    coords, charges_arr, radii = parse_pqr(args.pqr)
    dG = solvation_energy(model, coords, charges_arr)
    print(f"\nFinal estimated Delta G_solv (nondimensionalized, kT/e units): {dG:.4f} kT "
          f"= {dG*0.593:.4f} kcal/mol")
    print("Reminder: no independent reference at this scale yet -- compare against "
          "an APBS run on the same structure/charges before trusting this number.")