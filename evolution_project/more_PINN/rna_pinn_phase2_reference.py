"""
Independent numerical reference for Phase 2: a single point charge q at the
center of a sphere of radius R (the "Born ion" problem), solved for the
NONLINEAR regularized PBE using spherical symmetry.

This is a real numerical solver (BVP shooting via scipy), not a manufactured
answer -- it is the validation target for the PINN, in the spirit of the
original ask ("validate against numerical solvers").

Physics: after subtracting the Coulomb singularity u_s = q/(4*pi*eps_m*r)
(only inside Omega_m, zero outside -- Achondo et al.'s regularization),
the regular potential u_r is:

  Omega_m (r < R): Laplace's equation, kappa_m = 0, no singular charge left
                    -> spherically symmetric regular solution is CONSTANT.
  Omega_w (r > R): -eps_w * (1/r^2) d/dr(r^2 du_r/dr) + kappa_w^2*sinh(u_r) = 0

Interface conditions at r=R (derived analytically from the known jump in u_s
and its flux -- see chat derivation):
  jump_u    := u_r(R+) - u_r(R-) = -[u_s]_Gamma = +q / (4*pi*eps_m*R)
  jump_flux := eps_w*du_r/dr|_{R+} - eps_m*du_r/dr|_{R-}
             = -q / (4*pi*R^2)     (and du_r/dr|_{R-} = 0 since u_r is const inside)

  => du_r/dr|_{R+} = -q / (4*pi*eps_w*R^2)      [Neumann BC feeding the exterior ODE]

Far field: u_r(L) ~ 0 for L >> 1/kappa_w (checked numerically below).
"""

import numpy as np
from scipy.integrate import solve_bvp

# Physical parameters (chosen for this Phase 2 test -- not RNA yet, that's later)
R = 1.0          # solute sphere radius (Angstrom)
L = 10.0         # truncated outer domain radius
EPS_M = 2.0
EPS_W = 80.0
KAPPA_W = 1.0    # 1/Angstrom -> Debye length = 1 A, so L=10 is >>1/kappa: safe truncation
Q = 1.0          # elementary charge (nondimensionalized, as in Achondo et al.)


def solve_reference(n_points=2000):
    """Solve the exterior nonlinear radial ODE via scipy's BVP solver."""

    def rhs(r, y):
        # y[0] = u_r, y[1] = u_r'
        # From -eps_w*(1/r^2)*(r^2*y1')' + kappa^2*sinh(y0) = 0:
        #   y1' = (kappa^2/eps_w)*sinh(y0) - (2/r)*y1
        dy0 = y[1]
        dy1 = (KAPPA_W ** 2 / EPS_W) * np.sinh(y[0]) - (2.0 / r) * y[1]
        return np.vstack([dy0, dy1])

    def bc(ya, yb):
        neumann_target = -Q / (4 * np.pi * EPS_W * R ** 2)
        return np.array([ya[1] - neumann_target, yb[0] - 0.0])

    r_mesh = np.linspace(R, L, n_points)
    y_guess = np.zeros((2, r_mesh.size))
    # reasonable initial guess: Debye-Huckel-like decay for y0, and its derivative
    y_guess[0] = 0.05 * np.exp(-KAPPA_W * (r_mesh - R))
    y_guess[1] = -0.05 * KAPPA_W * np.exp(-KAPPA_W * (r_mesh - R))

    sol = solve_bvp(rhs, bc, r_mesh, y_guess, tol=1e-10, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(f"BVP solver failed: {sol.message}")

    u_r_at_R_plus = sol.sol(R)[0]
    jump_u = Q / (4 * np.pi * EPS_M * R)        # u_r(R+) - u_r(R-) = -[u_s]_Gamma = +q/(4*pi*eps_m*R)
    u_r_inside_const = u_r_at_R_plus - jump_u   # u_r(R-) = u_r(R+) - jump_u

    return sol, u_r_inside_const


def u_r_reference(r, sol, u_r_inside_const):
    """Evaluate the reference regular potential at radius/radii r (array or scalar)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty_like(r)
    inside = r < R
    out[inside] = u_r_inside_const
    out[~inside] = sol.sol(r[~inside])[0]
    return out


def solvation_energy_reference(u_r_inside_const):
    """Delta G_solv = 0.5 * sum(q_i * psi(x_qi)); here psi at the charge (origin)
    is just the regular potential inside (constant), since the singular part is
    excluded by construction (Achondo et al. eq. 28)."""
    return 0.5 * Q * u_r_inside_const


if __name__ == "__main__":
    sol, u_in = solve_reference()
    print("u_r inside (constant):", u_in)
    print("u_r at R+ :", sol.sol(R)[0])
    print("u_r at 2R :", sol.sol(2 * R)[0])
    print("u_r at 5R :", sol.sol(5 * R)[0])
    print("u_r at L  :", sol.sol(L)[0])
    print("Delta G_solv (reference, nondim units):", solvation_energy_reference(u_in))