"""
run_demo.py
-----------
Two things in here:

1. `demo_synthetic()` -- builds a toy double-helical phosphate backbone (no
   PDB needed) and runs the full train -> Gamma_i pipeline, just to confirm
   the code is mechanically correct (shapes, gradients, convergence trend).
   This is NOT a physics validation -- see sanity_check_against_apbs().

2. `run_on_pdb(path)` -- the actual entry point once you have a real
   structure: parses it with structure.py and runs the same pipeline.

Run `python run_demo.py` to execute the synthetic check.
"""

import numpy as np

from structure import load_rna_structure, get_bounding_box
from pb_pinn import train_pb_pinn, compute_gamma_per_residue, DEVICE


def synthetic_rna_helix(n_bp=12, rise=2.8, radius=9.0, twist_deg=32.7):
    """
    Crude idealized A-form-like phosphate helix (two strands), just to give
    the PINN something structured to fit without needing a real PDB. Returns
    the same tuple shape as structure.load_rna_structure.
    """
    twist = np.radians(twist_deg)
    phosphates = []
    for strand, phase in enumerate([0, np.pi]):  # two antiparallel-ish strands
        for i in range(n_bp):
            theta = i * twist + phase
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = i * rise
            phosphates.append([x, y, z])
    phosphate_xyz = np.array(phosphates, dtype=np.float64)
    phosphate_q = -1.0 * np.ones(len(phosphate_xyz))

    # Fake a sparse all-atom cloud (base + sugar ring approx) around each P
    # for the eps(r) surface -- just jittered points, not real chemistry.
    rng = np.random.default_rng(0)
    allatom = [phosphate_xyz]
    for p in phosphate_xyz:
        allatom.append(p + rng.normal(0, 2.5, size=(6, 3)))
    allatom_xyz = np.concatenate(allatom, axis=0)

    return phosphate_xyz, phosphate_q, allatom_xyz


def demo_synthetic():
    print(f"Using device: {DEVICE}\n")
    phosphate_xyz, phosphate_q, allatom_xyz = synthetic_rna_helix()
    lo, hi = get_bounding_box(allatom_xyz, padding=15.0)
    print(f"Box: lo={lo.round(1)}  hi={hi.round(1)}  "
          f"n_phosphates={len(phosphate_xyz)}  n_allatom={len(allatom_xyz)}\n")

    net, scaler, fields, history = train_pb_pinn(
        phosphate_xyz, phosphate_q, allatom_xyz, lo, hi,
        ionic_strength_M=0.15,
        n_epochs=60,       # short run for a smoke test; use 3000-10000 for real fits
        lr=1e-4,
        log_every=10,
        n_uniform=500, n_near=500, n_bc_per_face=80,
    )

    print("\nTraining loss trend (should be decreasing):")
    print(f"  first 100 avg: {np.mean(history['total'][:100]):.4e}")
    print(f"  last  100 avg: {np.mean(history['total'][-100:]):.4e}")

    print("\nComputing Gamma_i (per-residue excess monovalent cation count)...")
    gammas = compute_gamma_per_residue(
        net, scaler, fields, phosphate_xyz, c_bulk_M=0.15, shell_radius=10.0, n_mc=1000
    )
    print(f"  Gamma_i range: [{gammas.min():.3f}, {gammas.max():.3f}]")
    print(f"  Gamma_i mean:  {gammas.mean():.3f}")
    print("  (sign check: should be POSITIVE -- cations should be enriched, "
          "not depleted, near a negatively charged backbone)")

    return net, scaler, fields, gammas


def run_on_pdb(pdb_path, chain_id=None, ionic_strength_M=0.15, n_epochs=5000):
    phosphate_xyz, phosphate_q, allatom_xyz = load_rna_structure(pdb_path, chain_id=chain_id)
    lo, hi = get_bounding_box(allatom_xyz, padding=20.0)
    print(f"[sanity check] n_phosphates detected: {len(phosphate_xyz)}  (1EHZ should be ~76)")

    net, scaler, fields, history = train_pb_pinn(
        phosphate_xyz, phosphate_q, allatom_xyz, lo, hi,
        ionic_strength_M=ionic_strength_M, n_epochs=n_epochs,
    )
    gammas = compute_gamma_per_residue(net, scaler, fields, phosphate_xyz, c_bulk_M=ionic_strength_M)
    return net, scaler, fields, gammas


if __name__ == "__main__":
    demo_synthetic()