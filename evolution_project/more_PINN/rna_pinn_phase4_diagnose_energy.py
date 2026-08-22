"""
Diagnostic for the ~7000x solvation-energy discrepancy found against APBS.

Checks the ACTUAL trained model's output at real charge positions (not a
proxy/estimate) and compares against:
  1. The output-scaling range the training run assumed
  2. The magnitude psi would need to reach, at the most-charged atoms
     (phosphates), for the sum to plausibly reach APBS's ballpark

This does not assume the output-scaling hypothesis is correct -- it's a
genuine check, and could come back showing something else entirely (e.g.
psi values that ARE large enough individually but cancel in the sum due to
a sign error, which would point somewhere completely different).

Run with:
    python3 rna_pinn_phase4_diagnose_energy.py --checkpoint rna_phase4_checkpoint.pt
"""

import argparse
import numpy as np
import torch

from rna_pinn_phase4_mesh import parse_pqr, MoleculeGeometry
from rna_pinn_phase4_gpu import RealChargeSystem, RealMoleculePINN, DEVICE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rna_phase4_checkpoint.pt")
    parser.add_argument("--pqr", default=None)
    parser.add_argument("--min_radius", type=float, default=None)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

    if "geometry_config" in ckpt:
        gcfg = ckpt["geometry_config"]
        pqr_path = args.pqr or gcfg["pqr_path"]
        grid_spacing, min_radius = gcfg["grid_spacing"], gcfg["min_radius"]
    else:
        if args.min_radius is None:
            raise RuntimeError("Pre-fix checkpoint: pass --min_radius matching your training run.")
        pqr_path = args.pqr or "4tna_out.pqr"
        grid_spacing, min_radius = 1.0, args.min_radius

    coords, charges_arr, radii = parse_pqr(pqr_path)
    geom = MoleculeGeometry(coords, radii, grid_spacing=grid_spacing, min_radius=min_radius)
    geom.build_surface()
    geom.build_volume()
    charges = RealChargeSystem(coords, charges_arr)

    model = RealMoleculePINN(geom, charges).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}\n")

    with torch.no_grad():
        pts = torch.tensor(coords, dtype=torch.float64, device=DEVICE)
        psi_all = model.u_m(pts).cpu().numpy()

    print("=== What the trained network actually outputs at real charge sites ===")
    print(f"psi(x_i) over all {len(coords)} atoms: min={psi_all.min():.4f}, max={psi_all.max():.4f}, "
          f"mean={psi_all.mean():.4f}, std={psi_all.std():.4f}  (all in kT/e units)")
    print(f"(training run's assumed output range was roughly +/-2.3 -- if actual values "
          f"are clustered much closer to 0 than that, the network converged to a "
          f"near-flat function, not because it's mathematically capped, but because "
          f"nothing in training pushed it to use the range it was given)")

    # focus specifically on phosphate atoms -- the largest individual charges
    phosphate_mask = np.array([True if l.split()[2] in
                                {"P", "O1P", "O2P", "OP1", "OP2", "OP3"} else False
                                for l in open(pqr_path) if l.startswith("ATOM") or l.startswith("HETATM")])
    print(f"\n=== Phosphate atoms specifically ({phosphate_mask.sum()} atoms) ===")
    print(f"psi at phosphate atoms: min={psi_all[phosphate_mask].min():.4f}, "
          f"max={psi_all[phosphate_mask].max():.4f}, mean={psi_all[phosphate_mask].mean():.4f}")
    print(f"charge at phosphate atoms: min={charges_arr[phosphate_mask].min():.4f}, "
          f"max={charges_arr[phosphate_mask].max():.4f}")

    dG_actual = 0.5 * (charges_arr * psi_all).sum()
    dG_phosphate_only = 0.5 * (charges_arr[phosphate_mask] * psi_all[phosphate_mask]).sum()
    print(f"\nFull sum 0.5*sum(q_i * psi_i): {dG_actual:.4f} kT = {dG_actual*0.593:.4f} kcal/mol")
    print(f"Phosphate-only contribution:   {dG_phosphate_only:.4f} kT = "
          f"{dG_phosphate_only*0.593:.4f} kcal/mol "
          f"({100*dG_phosphate_only/dG_actual:.1f}% of total)")

    # what magnitude WOULD psi need to average, at phosphates, to reach APBS's ballpark?
    target_kT = -15912 / 0.593  # APBS result converted to kT
    q_phos_sum_abs = np.abs(charges_arr[phosphate_mask]).sum()
    implied_avg_psi = target_kT / (0.5 * q_phos_sum_abs) if q_phos_sum_abs > 0 else float('nan')
    print(f"\nFor comparison: to reach APBS's ballpark ({target_kT:.0f} kT) via phosphate "
          f"charges alone,\naverage |psi| at phosphates would need to be roughly "
          f"{abs(implied_avg_psi):.2f} kT --\ncompare that to what the network actually outputs above.")