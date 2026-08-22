"""
compare_apbs.py
---------------
Load an APBS .dx potential grid and compare phi(r) against the trained
PB-PINN at the same points, near the same phosphate used in the PINN's
own diagnostic print (phosphate_xyz[0], offsets along +x).

Units note: APBS's default OpenDX output is in kT/e (same reduced units
your PINN uses) when run with "outputs" set to potential in its input
file -- pdb2pqr's auto-generated APBS input typically requests this by
default. If your .dx values look ~25x too large/small compared to the
PINN, the file may be in units of kJ/mol/e or similar -- check the APBS
run log for the "write pot dx" line and the units comment at the top of
the .dx file itself.

Usage:
    python compare_PINN_APBS.py 1EHZ_simple_potential.dx 1EHZ.pdb --chain A --checkpoint pinn_1ehz.pt
"""

import argparse
import numpy as np
import torch

from structure import load_rna_structure, get_bounding_box
from pb_pinn import PBPotentialNet, Scaler, DEVICE
from pb_pinn import PBFields, total_phi

def read_dx(path):
    """
    Minimal OpenDX (.dx) grid reader for APBS output.
    Returns:
        origin: (3,) grid origin in Angstrom
        deltas: (3,3) grid spacing vectors (usually diagonal)
        dims:   (3,) number of grid points along each axis
        data:   (nx, ny, nz) numpy array of phi values
    """
    with open(path, "r") as f:
        lines = f.readlines()

    dims = None
    origin = None
    deltas = []
    data_start = None
    n_total = None

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#") or s == "":
            continue
        if s.startswith("object 1") and "gridpositions" in s:
            parts = s.split()
            dims = np.array([int(parts[-3]), int(parts[-2]), int(parts[-1])])
        elif s.startswith("origin"):
            origin = np.array([float(x) for x in s.split()[1:4]])
        elif s.startswith("delta"):
            deltas.append(np.array([float(x) for x in s.split()[1:4]]))
        elif s.startswith("object 3") and "array type double" in s:
            n_total = int(dims[0]) * int(dims[1]) * int(dims[2])
            data_start = i + 1
            break

    if dims is None or origin is None or data_start is None:
        raise ValueError(
            f"Could not parse header of {path} -- is this a real APBS OpenDX "
            "potential file (not a charge/kappa/eps map)?"
        )

    deltas = np.array(deltas)  # (3,3)

    values = []
    for line in lines[data_start:]:
        s = line.strip()
        if s.startswith("attribute") or s == "":
            continue
        if s.startswith("object"):
            break
        values.extend(float(x) for x in s.split())
        if len(values) >= n_total:
            break

    data = np.array(values[:n_total]).reshape(dims)
    return origin, deltas, dims, data


def sample_dx(origin, deltas, dims, data, points_xyz):
    """
    Trilinear interpolation of the .dx grid at arbitrary (x,y,z) points.
    points_xyz: (N,3) array in Angstrom, same frame as the PDB used to
    generate the .pqr/.dx (APBS keeps original PDB coordinates by default).
    """
    spacing = np.diag(deltas)  # assumes axis-aligned grid, true for APBS default
    frac = (points_xyz - origin) / spacing  # (N,3), fractional grid index

    out = np.full(len(points_xyz), np.nan)
    for k, (fx, fy, fz) in enumerate(frac):
        if not (0 <= fx <= dims[0] - 1 and 0 <= fy <= dims[1] - 1 and 0 <= fz <= dims[2] - 1):
            continue  # point outside the APBS grid -- can't compare here
        i0, j0, k0 = int(np.floor(fx)), int(np.floor(fy)), int(np.floor(fz))
        i1, j1, k1 = min(i0 + 1, dims[0] - 1), min(j0 + 1, dims[1] - 1), min(k0 + 1, dims[2] - 1)
        tx, ty, tz = fx - i0, fy - j0, fz - k0

        c000 = data[i0, j0, k0]; c100 = data[i1, j0, k0]
        c010 = data[i0, j1, k0]; c110 = data[i1, j1, k0]
        c001 = data[i0, j0, k1]; c101 = data[i1, j0, k1]
        c011 = data[i0, j1, k1]; c111 = data[i1, j1, k1]

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        out[k] = c0 * (1 - tz) + c1 * tz

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dx_path", help="APBS potential .dx file")
    parser.add_argument("pdb_path", help="Same PDB used for training the PINN")
    parser.add_argument("--chain", default=None)
    parser.add_argument("--checkpoint", default=None,
                         help="Optional path to a saved PINN state_dict (see note below)")
    args = parser.parse_args()

    print(f"Reading {args.dx_path} ...")
    origin, deltas, dims, data = read_dx(args.dx_path)
    print(f"  grid dims: {dims}, origin: {origin.round(2)}, "
          f"spacing: {np.diag(deltas).round(3)}")
    print(f"  phi range in file: [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")

    print(f"\nLoading structure {args.pdb_path} ...")
    phosphate_xyz, phosphate_q, allatom_xyz = load_rna_structure(args.pdb_path, chain_id=args.chain)
    lo, hi = get_bounding_box(allatom_xyz, padding=20.0)
    scaler = Scaler(lo, hi)

    # NOTE: this assumes you still have `net` in memory from training (e.g. running
    # this in the same interactive session as run_on_pdb). If running as a fresh
    # script, you'll need to have saved the model first:
    #   torch.save(net.state_dict(), "pinn_1ehz.pt")
    # and load it here:
    net = PBPotentialNet().to(DEVICE)
    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        print(f"Loaded PINN weights from {args.checkpoint}")
    else:
        print("WARNING: no --checkpoint given -- using an UNTRAINED network. "
              "Pass --checkpoint pinn_1ehz.pt (see note in this script's source "
              "about saving it after training).")
    net.eval()

    center = phosphate_xyz[0]
    offsets = np.array([2.0, 3.0, 5.0, 10.0, 20.0, 40.0])  # Angstrom offsets along +x from the first phosphate
    query_points = center[None, :] + np.stack([offsets, np.zeros_like(offsets), np.zeros_like(offsets)], axis=1)

    phi_apbs = sample_dx(origin, deltas, dims, data, query_points)

    fields = PBFields(phosphate_xyz, phosphate_q, allatom_xyz, ionic_strength_M=0.15)
    with torch.no_grad():
        pts_t = torch.tensor(query_points, dtype=torch.float32, device=DEVICE)
        phi_pinn = total_phi(net, scaler, fields, pts_t).cpu().numpy()

    print("\n%-8s %-14s %-14s %-10s" % ("d (Å)", "APBS phi", "PINN phi", "ratio"))
    for d, pa, pp in zip(offsets, phi_apbs, phi_pinn):
        ratio = pp / pa if (pa == pa and pa != 0) else float("nan")  # pa==pa filters NaN
        print(f"{d:<8.1f} {pa:<14.4f} {pp:<14.4f} {ratio:<10.3f}")

    print(
        "\nInterpretation: ratio near 1.0 across all distances is a good sign. "
        "A roughly CONSTANT ratio far from 1 (e.g. always ~0.3 or always ~3) "
        "suggests a systematic scale factor somewhere (units, charge convention, "
        "or eps_in/eps_out mismatch) -- fixable by rescaling. A ratio that "
        "DRIFTS with distance suggests a shape/decay-length mismatch (e.g. "
        "wrong ionic strength/kappa, or the sigmoid dielectric surface being "
        "too crude) -- a deeper physics issue, not just a scale bug."
    )


if __name__ == "__main__":
    main()