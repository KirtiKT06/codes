"""
run_temp_scan.py
================
Launches a series of SOP-SC simulations at different temperatures
for heat capacity and Rg analysis.

Usage:
    python run_temp_scan.py

This script modifies input.json for each temperature, runs runSOP.py,
and collects the energy and data files.

Requirements:
    - runSOP.py and all supporting files in the same directory
    - input.json present and working
    - A base output directory (set BASE_DIR below)
"""

import json
import os
import shutil
import subprocess
import numpy as np

# ============================================================
# USER SETTINGS — edit these
# ============================================================

# Temperatures to simulate (Kelvin)
# Covers the range around Tm ≈ 353 K with fine spacing near the transition
TEMPERATURES = [
    300, 310, 320, 330, 335, 340, 345,
    350, 353, 356, 360, 365, 370, 380
]

# Steps per temperature (10 ns at dt=0.001 ps = 10,000,000 steps)
# For a first pass use 2,000,000 (2 ns) to go faster
NUMSTEPS = 2000000

# Base output directory — all results go here
BASE_DIR = "/data/cg_dna_histone/sop_sc/temp_scan"

# Path to the working directory with runSOP.py
WORK_DIR = os.path.abspath(".")

# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================

def run_temperature(T, base_dir, numsteps):
    """Run a single temperature simulation."""

    T_str = f"{T:.2f}K"
    out_dir = os.path.join(base_dir, T_str)
    os.makedirs(out_dir, exist_ok=True)

    # Load base config
    with open("input.json") as f:
        config = json.load(f)

    # Modify for this temperature
    config["Temp"] = float(T)
    config["numsteps"] = numsteps
    config["pdb_prefix"]      = os.path.join(out_dir, "cg_ion")
    config["dcd_prefix"]      = os.path.join(out_dir, "anim")
    config["data_name"]       = os.path.join(out_dir, "data.out")
    config["energy_name"]     = os.path.join(out_dir, "energy.dat")
    config["checkpoint_name"] = os.path.join(out_dir, "chkout.chk")
    config["minimization"]    = False
    config["restart"]         = False

    # Write modified config
    tmp_config = os.path.join(out_dir, "input.json")
    with open(tmp_config, "w") as f:
        json.dump(config, f, indent=2)

    # Copy the CG ion PDB to the output directory
    # runSOP.py reads pdb_prefix + ".pdb"
    src_pdb = config["pdb_prefix"] + ".pdb"

    # The source PDB is the one from the main run
    with open("input.json") as f:
        base_config = json.load(f)
    original_pdb = base_config["pdb_prefix"] + ".pdb"

    dest_pdb = os.path.join(out_dir, "cg_ion.pdb")
    if not os.path.exists(dest_pdb):
        shutil.copy(original_pdb, dest_pdb)

    print(f"\n{'='*60}")
    print(f"Running T = {T} K  →  {out_dir}")
    print(f"{'='*60}")

    # Run simulation with the modified config
    env = os.environ.copy()
    result = subprocess.run(
        ["python", "runSOP.py"],
        cwd=WORK_DIR,
        env={**env, "INPUT_JSON": tmp_config},
        capture_output=False
    )

    if result.returncode != 0:
        print(f"WARNING: simulation at T={T} K exited with code {result.returncode}")
    else:
        print(f"Done: T = {T} K")

    return out_dir


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    print("SOP-SC Temperature Scan")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Steps per T:  {NUMSTEPS}")
    print(f"Output dir:   {BASE_DIR}")

    completed = []

    for T in TEMPERATURES:
        out_dir = run_temperature(T, BASE_DIR, NUMSTEPS)
        energy_file = os.path.join(out_dir, "energy.dat")

        if os.path.exists(energy_file):
            completed.append((T, energy_file))
            print(f"  Energy file written: {energy_file}")
        else:
            print(f"  WARNING: No energy file found for T={T} K")

    print(f"\nCompleted {len(completed)}/{len(TEMPERATURES)} temperatures.")
    print("Run analyse_cv.py to compute heat capacity and Rg.")


if __name__ == "__main__":
    main()