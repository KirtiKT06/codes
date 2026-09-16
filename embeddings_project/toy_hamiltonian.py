"""
Level 1 toy system: charged polymer ("DNA stand-in") + charged sphere ("histone stand-in").

Purpose of this script: NOT to be physically calibrated yet. It's a mechanism test.
We want to know, before touching real embeddings or real data, whether the core
pipeline actually works end to end:

    effective parameters --> minimize a hand-written Hamiltonian --> predicted P(q)
                                        ^
                                        |
                          can we get a USABLE gradient back through this step?

This is exactly the step that will later sit between "embedding -> effective
parameters" and "loss against experimental P(q)". If gradients don't flow
sensibly here, they won't flow sensibly with real embeddings either -- and this
is also where the RNA-PBE-PINN local-minima problem is most likely to show up
again, so we check for it deliberately.

Units are arbitrary/toy (k_B T = 1, charges and lengths are dimensionless).
"""

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Fixed "known" system constants (things we are NOT learning -- analogous to
# things a real force field or known geometry would give you for free)
# ---------------------------------------------------------------------------
N_BEADS = 20        # polymer beads standing in for a short DNA stretch
BOND_LEN = 1.0       # equilibrium bond length between consecutive beads
K_BOND = 30.0        # bond stiffness (kept fixed -- this is NOT an effective param)
Q_SPHERE = 5.0       # sphere ("histone") charge magnitude, fixed
R_SPHERE = 2.0       # sphere radius, fixed
N_GD_STEPS = 250      # inner-loop minimization steps
LR_POS = 0.01         # inner-loop step size (positions)

Q_GRID = np.logspace(-2, 0, 25)  # q values for P(q), toy units


def polymer_forces(pos, q_p, kappa_inv, exvol_eps):
    """
    Analytic forces on each bead, given current positions and the three
    EFFECTIVE parameters we ultimately want an embedding to predict:
        q_p        : effective charge per bead
        kappa_inv  : effective Debye screening length
        exvol_eps  : effective excluded-volume repulsion strength
    pos: (N_BEADS, 2) array
    """
    forces = np.zeros_like(pos)

    # --- bond springs (backbone connectivity) ---
    for i in range(N_BEADS - 1):
        d = pos[i + 1] - pos[i]
        dist = np.linalg.norm(d) + 1e-9
        f = 2 * K_BOND * (dist - BOND_LEN) * (d / dist)
        forces[i] += f
        forces[i + 1] -= f

    # --- screened electrostatics: each bead vs. the sphere at the origin ---
    dist = np.linalg.norm(pos, axis=1) + 1e-9  # (N_BEADS,)
    # energy_i = q_p * Q_SPHERE * exp(-dist/kappa_inv) / dist
    # force_i  = -dE/dr, pointing along pos[i]/dist
    coeff = q_p * Q_SPHERE * np.exp(-dist / kappa_inv) * (1.0 / dist + 1.0 / kappa_inv) / dist
    forces += -(coeff[:, None]) * (pos / dist[:, None])

    # --- soft excluded volume: penalize beads inside the sphere radius ---
    inside = dist < R_SPHERE
    if np.any(inside):
        overlap = (R_SPHERE - dist[inside])
        exv_force_mag = 2 * exvol_eps * overlap  # points OUTWARD (repulsive)
        direction = pos[inside] / dist[inside, None]
        forces[inside] += exv_force_mag[:, None] * direction

    return forces


def minimize_structure(q_p, kappa_inv, exvol_eps, init_pos=None):
    """Run the inner-loop gradient descent to find the bound configuration."""
    if init_pos is None:
        # start as a loose extended chain, off to one side of the sphere
        init_pos = np.stack([
            np.linspace(R_SPHERE + 1.0, R_SPHERE + 1.0 + N_BEADS * BOND_LEN, N_BEADS),
            np.zeros(N_BEADS),
        ], axis=1)
    pos = init_pos.copy()
    for _ in range(N_GD_STEPS):
        f = polymer_forces(pos, q_p, kappa_inv, exvol_eps)
        pos = pos + LR_POS * f
    return pos


def compute_pq(pos, q_grid=Q_GRID):
    """Debye scattering formula from a set of point scatterers (beads + sphere)."""
    # treat sphere center as one extra, heavier scatterer (proxy for histone electron density)
    scatterers = np.vstack([pos, np.zeros((1, 2))])
    weights = np.concatenate([np.ones(N_BEADS), [5.0]])
    n = len(scatterers)
    diffs = scatterers[:, None, :] - scatterers[None, :, :]
    rij = np.linalg.norm(diffs, axis=-1)  # (n, n)
    w = weights[:, None] * weights[None, :]
    pq = []
    for q in q_grid:
        qr = q * rij
        sinc = np.ones_like(qr)
        nz = qr > 1e-8
        sinc[nz] = np.sin(qr[nz]) / qr[nz]
        pq.append(np.sum(w * sinc) / (weights.sum() ** 2))
    return np.array(pq)


def forward(params, init_pos=None):
    q_p, kappa_inv, exvol_eps = params
    pos = minimize_structure(q_p, kappa_inv, exvol_eps, init_pos=init_pos)
    return compute_pq(pos), pos


def loss_fn(params, target_pq):
    pq, _ = forward(params)
    return np.mean((pq - target_pq) ** 2)


def finite_diff_grad(params, target_pq, eps=1e-3):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy(); p_plus[i] += eps
        p_minus = params.copy(); p_minus[i] -= eps
        grad[i] = (loss_fn(p_plus, target_pq) - loss_fn(p_minus, target_pq)) / (2 * eps)
    return grad


def loss_fn_fixed_kappa(free_params, kappa_inv_fixed, target_pq):
    q_p, exvol_eps = free_params
    return loss_fn(np.array([q_p, kappa_inv_fixed, exvol_eps]), target_pq)


def finite_diff_grad_fixed_kappa(free_params, kappa_inv_fixed, target_pq, eps=1e-3):
    grad = np.zeros_like(free_params)
    for i in range(len(free_params)):
        p_plus = free_params.copy(); p_plus[i] += eps
        p_minus = free_params.copy(); p_minus[i] -= eps
        grad[i] = (loss_fn_fixed_kappa(p_plus, kappa_inv_fixed, target_pq)
                   - loss_fn_fixed_kappa(p_minus, kappa_inv_fixed, target_pq)) / (2 * eps)
    return grad


if __name__ == "__main__":
    # "Ground truth" effective parameters -- stand-in for what a real embedding
    # SHOULD eventually predict for a real DNA/histone pair.
    true_params = np.array([1.2, 1.5, 4.0])   # q_p, kappa_inv, exvol_eps
    target_pq, _ = forward(true_params)
    target_pq = target_pq + rng.normal(0, 0.002, size=target_pq.shape)  # experimental-style noise

    print("=== Test A: learn all 3 params freely (kappa_inv included) ===")
    n_restarts = 6
    recovered = []
    for trial in range(n_restarts):
        params = np.array([
            rng.uniform(0.2, 3.0),   # q_p guess
            rng.uniform(0.3, 3.0),   # kappa_inv guess
            rng.uniform(0.5, 8.0),   # exvol_eps guess
        ])
        lr = np.array([0.05, 0.05, 0.2])
        for step in range(60):
            g = finite_diff_grad(params, target_pq)
            params = params - lr * g
            params = np.clip(params, 0.05, 10.0)
        final_loss = loss_fn(params, target_pq)
        recovered.append((params.copy(), final_loss))
        print(f"trial {trial}: q_p={params[0]:.2f}, kappa_inv={params[1]:.2f}, "
              f"exvol_eps={params[2]:.2f} | final loss={final_loss:.6f}")
    print("true params: q_p={:.2f}, kappa_inv={:.2f}, exvol_eps={:.2f}".format(*true_params))
    losses = [r[1] for r in recovered]
    print(f"loss range: min={min(losses):.6f}, max={max(losses):.6f}")

    print("\n=== Test B: fix kappa_inv at its (known, buffer-derived) true value, "
          "learn only q_p and exvol_eps ===")
    recovered_b = []
    for trial in range(n_restarts):
        free_params = np.array([
            rng.uniform(0.2, 3.0),   # q_p guess
            rng.uniform(0.5, 8.0),   # exvol_eps guess
        ])
        lr = np.array([0.05, 0.2])
        for step in range(60):
            g = finite_diff_grad_fixed_kappa(free_params, true_params[1], target_pq)
            free_params = free_params - lr * g
            free_params = np.clip(free_params, 0.05, 10.0)
        final_loss = loss_fn_fixed_kappa(free_params, true_params[1], target_pq)
        recovered_b.append((free_params.copy(), final_loss))
        print(f"trial {trial}: q_p={free_params[0]:.2f}, exvol_eps={free_params[1]:.2f} "
              f"| final loss={final_loss:.6f}")
    print("true params: q_p={:.2f}, exvol_eps={:.2f}".format(true_params[0], true_params[2]))
    losses_b = [r[1] for r in recovered_b]
    print(f"loss range: min={min(losses_b):.6f}, max={max(losses_b):.6f}")

    print("\n=== Test C: joint fit across 3 known salt conditions "
          "(kappa_inv = 0.9, 1.5, 2.4), same q_p/exvol_eps must explain all 3 ===")
    kappa_conditions = [0.9, 1.5, 2.4]
    targets_multi = []
    for k in kappa_conditions:
        pq, _ = forward(np.array([true_params[0], k, true_params[2]]))
        targets_multi.append(pq + rng.normal(0, 0.002, size=pq.shape))

    def loss_multi(free_params):
        q_p, exvol_eps = free_params
        total = 0.0
        for k, tgt in zip(kappa_conditions, targets_multi):
            total += loss_fn(np.array([q_p, k, exvol_eps]), tgt)
        return total / len(kappa_conditions)

    def grad_multi(free_params, eps=1e-3):
        grad = np.zeros_like(free_params)
        for i in range(len(free_params)):
            p_plus = free_params.copy(); p_plus[i] += eps
            p_minus = free_params.copy(); p_minus[i] -= eps
            grad[i] = (loss_multi(p_plus) - loss_multi(p_minus)) / (2 * eps)
        return grad

    recovered_c = []
    for trial in range(n_restarts):
        free_params = np.array([
            rng.uniform(0.2, 3.0),
            rng.uniform(0.5, 8.0),
        ])
        lr = np.array([0.05, 0.2])
        for step in range(60):
            g = grad_multi(free_params)
            free_params = free_params - lr * g
            free_params = np.clip(free_params, 0.05, 10.0)
        final_loss = loss_multi(free_params)
        recovered_c.append((free_params.copy(), final_loss))
        print(f"trial {trial}: q_p={free_params[0]:.2f}, exvol_eps={free_params[1]:.2f} "
              f"| final loss={final_loss:.6f}")
    print("true params: q_p={:.2f}, exvol_eps={:.2f}".format(true_params[0], true_params[2]))
    losses_c = [r[1] for r in recovered_c]
    print(f"loss range: min={min(losses_c):.6f}, max={max(losses_c):.6f}")
