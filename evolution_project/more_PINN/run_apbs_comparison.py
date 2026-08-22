"""
Run APBS on the same structure/charges as the PINN, using the standard
two-calculation method for solvation energy (APBS doesn't give Delta G_solv
directly -- you compute total electrostatic energy in the real solvent
environment, then again in a reference state with no solvent contrast and no
salt, and subtract).

CAVEAT, stated plainly: I don't have APBS installed in the environment this
was built in, and couldn't test this end-to-end the way everything else in
this project was tested. The APBS input-file syntax below follows standard,
well-documented conventions (mg-auto, npbe, ion statements) but you should
treat the first run as a debugging pass. If it errors, send me the exact
APBS error message and we'll fix it the same way we fixed everything else.

Physical parameters matched to the PINN (rna_pinn_phase4_gpu.py):
  pdie (solute)  = 2.0
  sdie (solvent) = 80.0
  salt           = 0.145 M (1:1), from kappa=0.125 1/A via the standard
                   kappa[1/A] ~= 0.328 * sqrt(I[mol/L]) relation at 25C

Usage:
    python3 run_apbs_comparison.py --pqr 4tna_out.pqr --apbs_bin apbs
"""

import argparse
import re
import subprocess
import numpy as np


def parse_pqr_coords(path):
    coords = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                x, y, z = map(float, parts[-5:-2])
                coords.append((x, y, z))
    return np.array(coords)


def next_valid_dime(n):
    """APBS multigrid requires dime = c*2^(l+1)+1. Round up to a common
    valid value rather than requiring the user to hand-pick one."""
    valid = [65, 97, 129, 161, 193, 225, 257]
    for v in valid:
        if v >= n:
            return v
    return valid[-1]


def build_apbs_input(pqr_path, out_prefix, center, cglen, fglen, dime,
                      sdie, salt_conc, calc_type="npbe"):
    """One elec block; salt_conc=0 and sdie==pdie(2.0) gives the reference
    (vacuum-like) state; salt_conc=0.145, sdie=80 gives the solvated state."""
    ion_lines = ""
    if salt_conc > 0:
        ion_lines = (f"    ion charge  1 conc {salt_conc} radius 2.0\n"
                     f"    ion charge -1 conc {salt_conc} radius 2.0\n")

    return f"""read
    mol pqr {pqr_path}
end
elec name {out_prefix}
    mg-auto
    dime {dime[0]} {dime[1]} {dime[2]}
    cglen {cglen[0]:.2f} {cglen[1]:.2f} {cglen[2]:.2f}
    fglen {fglen[0]:.2f} {fglen[1]:.2f} {fglen[2]:.2f}
    cgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
    fgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
    mol 1
    {calc_type}
    bcfl mdh
    pdie 2.0
    sdie {sdie}
{ion_lines}    srfm smol
    chgm spl2
    srad 1.4
    swin 0.3
    sdens 10.0
    temp 298.15
    calcenergy total
    calcforce no
    write pot dx {out_prefix}_pot
end
quit
"""


def run_apbs(apbs_bin, in_path, log_path):
    print(f"Running: {apbs_bin} {in_path}  (log -> {log_path})")
    with open(log_path, "w") as logf:
        result = subprocess.run([apbs_bin, in_path], stdout=logf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"WARNING: APBS exited with code {result.returncode} -- check {log_path}")
    return log_path


def parse_energy(log_path):
    """Looks for APBS's total electrostatic energy line. APBS's exact wording
    has varied across versions historically -- if this fails to match, open
    the log file and look for a line with 'energy' and a scientific-notation
    number; tell me the exact wording and I'll fix the regex."""
    text = open(log_path).read()
    patterns = [
        r"Total electrostatic energy\s*=\s*([-\d.eE+]+)\s*(kJ/mol|kcal/mol)",
        r"Global net ELEC energy\s*=\s*([-\d.eE+]+)\s*(kJ/mol|kcal/mol)",
        r"local net energy\s*=\s*([-\d.eE+]+)\s*(kJ/mol|kcal/mol)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            value, unit = float(m.group(1)), m.group(2)
            return value, unit
    print(f"COULD NOT auto-parse energy from {log_path}. Open it and search for "
          f"'energy' manually -- send me the exact line and I'll fix the parser.")
    return None, None


def to_kcal(value, unit):
    if unit is None:
        return None
    if unit.lower() == "kj/mol":
        return value / 4.184
    return value  # already kcal/mol


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pqr", default="4tna_out.pqr")
    parser.add_argument("--apbs_bin", default="apbs")
    parser.add_argument("--pinn_dG_kT", type=float, default=-3.767,
                         help="Your PINN's Delta G_solv in kT units (from its printed output) "
                              "for the final comparison")
    args = parser.parse_args()

    coords = parse_pqr_coords(args.pqr)
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    print(f"Structure extent: {extent}, centroid: {center}")

    # coarse grid: generous margin for far-field boundary condition, roughly
    # matching the PINN's L_outer truncation radius (~65A -> ~130A box)
    cglen = extent + 90.0
    # fine grid: tight around the molecule for resolution where it matters
    fglen = extent + 20.0
    dime = tuple(next_valid_dime(int(min(fg / 0.5, 225))) for fg in fglen)
    print(f"cglen={cglen}, fglen={fglen}, dime={dime}")

    kappa = 0.125  # 1/A, matches KAPPA_W in the PINN
    salt_conc = (kappa / 0.328) ** 2  # mol/L, standard 25C 1:1 electrolyte relation
    print(f"Using salt concentration {salt_conc:.4f} M (from kappa={kappa} 1/A)")

    in_solv = build_apbs_input(args.pqr, "solv", center, cglen, fglen, dime,
                                sdie=80.0, salt_conc=salt_conc)
    in_ref = build_apbs_input(args.pqr, "ref", center, cglen, fglen, dime,
                               sdie=2.0, salt_conc=0.0)

    with open("apbs_solv.in", "w") as f:
        f.write(in_solv)
    with open("apbs_ref.in", "w") as f:
        f.write(in_ref)

    run_apbs(args.apbs_bin, "apbs_solv.in", "apbs_solv.log")
    run_apbs(args.apbs_bin, "apbs_ref.in", "apbs_ref.log")

    e_solv, unit_solv = parse_energy("apbs_solv.log")
    e_ref, unit_ref = parse_energy("apbs_ref.log")

    if e_solv is None or e_ref is None:
        print("\nCould not complete the comparison automatically -- see warnings above. "
              "The raw logs (apbs_solv.log, apbs_ref.log) are saved; send me what the "
              "energy lines actually look like and I'll fix the parser.")
    else:
        dG_kcal = to_kcal(e_solv, unit_solv) - to_kcal(e_ref, unit_ref)
        pinn_dG_kcal = args.pinn_dG_kT * 0.593  # kT -> kcal/mol at 298K

        print(f"\n=== Comparison ===")
        print(f"APBS: E_solvated = {e_solv:.4f} {unit_solv}, E_reference = {e_ref:.4f} {unit_ref}")
        print(f"APBS Delta G_solv = {dG_kcal:.4f} kcal/mol")
        print(f"PINN Delta G_solv = {pinn_dG_kcal:.4f} kcal/mol  ({args.pinn_dG_kT} kT)")
        if dG_kcal != 0:
            print(f"Relative difference: {abs(dG_kcal - pinn_dG_kcal) / abs(dG_kcal):.1%}")