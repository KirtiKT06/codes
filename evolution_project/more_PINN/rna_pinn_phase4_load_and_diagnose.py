"""
Loads the checkpoint from the completed (post-radius-fix) 30k-epoch run and
reruns the spatial flux-residual map on it -- answers: did fixing the
zero-radius-hydrogen issue move the problem elsewhere, or is it the same
atoms as before (which would mean the radius floor didn't fully solve even
that specific issue), or is it now spread out more generally (which would
point toward the capacity/domain-scaling explanation being dominant)?

Run with:
    python3 rna_pinn_phase4_load_and_diagnose.py --checkpoint rna_phase4_checkpoint.pt
"""

import argparse
import torch

from rna_pinn_phase4_mesh import parse_pqr, MoleculeGeometry
from rna_pinn_phase4_gpu import RealChargeSystem, RealMoleculePINN, DEVICE
from rna_pinn_phase4_diagnostic import spatial_residual_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rna_phase4_checkpoint.pt")
    parser.add_argument("--pqr", default=None, help="Overrides the checkpoint's recorded PQR path if given")
    parser.add_argument("--n_gamma", type=int, default=8000)
    parser.add_argument("--top_pct", type=float, default=1.0)
    parser.add_argument("--min_radius", type=float, default=None,
                         help="REQUIRED override for checkpoints saved before geometry_config "
                              "was tracked. Must match exactly what your training run printed "
                              "(e.g. 'Applied min_radius=1.3A floor...').")
    parser.add_argument("--grid_spacing", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)

    if "geometry_config" in ckpt:
        gcfg = ckpt["geometry_config"]
        pqr_path = args.pqr or gcfg["pqr_path"]
        grid_spacing, min_radius = gcfg["grid_spacing"], gcfg["min_radius"]
        print("Geometry config found in checkpoint -- using it exactly, no guessing.")
    else:
        if args.min_radius is None:
            raise RuntimeError(
                "This checkpoint predates geometry_config tracking, so min_radius can't be "
                "read back automatically -- and I won't silently default it again (that's "
                "exactly the bug that produced the mismatched diagnostic earlier). Rerun with "
                "--min_radius set to EXACTLY what your training run's log printed, e.g.:\n"
                "  python3 rna_pinn_phase4_load_and_diagnose.py --min_radius 1.3")
        pqr_path = args.pqr or "4tna_out.pqr"
        grid_spacing, min_radius = args.grid_spacing, args.min_radius
        print(f"WARNING: using manually-supplied min_radius={min_radius} -- double check this "
              f"matches your training run's printed value exactly.")

    print(f"Rebuilding geometry: pqr={pqr_path}, grid_spacing={grid_spacing}, min_radius={min_radius}")
    coords, charges_arr, radii = parse_pqr(pqr_path)
    geom = MoleculeGeometry(coords, radii, grid_spacing=grid_spacing, min_radius=min_radius)
    geom.build_surface()
    geom.build_volume()
    charges = RealChargeSystem(coords, charges_arr)

    model = RealMoleculePINN(geom, charges).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model weights from epoch {ckpt['epoch']}")

    spatial_residual_map(model, geom, charges, coords, n_gamma=args.n_gamma, top_pct=args.top_pct)