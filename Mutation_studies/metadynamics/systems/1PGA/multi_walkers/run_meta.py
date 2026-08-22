"""
run_meta.py  –  one walker of a multi-walker metaD run.

Usage (single process):
    python run_meta.py --walker 0 --total-walkers 4

Usage (via launcher):
    python launch_walkers.py --walkers 4        # starts all 4 at once

Directory layout expected at runtime:

    /data/mutation_study/metaD/WT/          ← BASE_DIR
        wt_solvated.pdb
        plumed_template.dat
        HILLS.0, HILLS.1, …                 ← written here (shared)
        walker_0/
            plumed.dat                      ← generated from template
            traj.dcd
            log.txt
            checkpoint.chk
            checkpoint_backup.chk
            COLVAR
        walker_1/ …
"""

import argparse
import os
import shutil
import sys

from openmm.app import *
from openmm import *
from openmm.unit import nanometer, picosecond, picoseconds, kelvin
from openmmplumed import PlumedForce

# ==========================================
# CLI
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--walker",        type=int, required=True,
                    help="ID of this walker (0-based)")
parser.add_argument("--total-walkers", type=int, default=4,
                    help="Total number of walkers (must match make_plumed.py)")
args = parser.parse_args()

WALKER_ID   = args.walker
N_WALKERS   = args.total_walkers

print(f"\n=== Walker {WALKER_ID} / {N_WALKERS} ===\n")

# ==========================================
# PATHS
# ==========================================

BASE_DIR   = "/home/feynman/projects/codes/Mutation_studies/metadynamics/WT/multi_walkers"
OUT_DIR    = "/data/mutation_study/metaD/WT/multi_walker"
WALKER_DIR = os.path.join(OUT_DIR, f"walker_{WALKER_ID}")
os.makedirs(WALKER_DIR, exist_ok=True)

CHECKPOINT     = os.path.join(WALKER_DIR, "checkpoint.chk")
CHECKPOINT_BAK = os.path.join(WALKER_DIR, "checkpoint_backup.chk")
TRAJ           = os.path.join(WALKER_DIR, "traj.dcd")
LOG            = os.path.join(WALKER_DIR, "log.txt")

PDB_FILE      = os.path.join(BASE_DIR, "wt_solvated.pdb")
TEMPLATE_FILE = os.path.join(BASE_DIR, "plumed_template.dat")
LOCAL_PLUMED  = os.path.join(WALKER_DIR, "plumed.dat")  # FIXED: was BASE_DIR → walkers overwrote each other

# ==========================================
# GENERATE LOCAL plumed.dat FROM TEMPLATE
# Fill in this walker's ID so PLUMED knows
# which HILLS file to write and which to read.
# ==========================================

# Absolute paths for all PLUMED output files so nothing depends on cwd.
HILLS_FILE  = os.path.join(OUT_DIR,    "HILLS")          # → OUT_DIR/HILLS.<walker_id>
COLVAR_FILE = os.path.join(WALKER_DIR, "COLVAR")         # → OUT_DIR/walker_N/COLVAR

with open(TEMPLATE_FILE) as f:
    plumed_script = (
        f.read()
        .replace("{WALKER_ID}",            str(WALKER_ID))
        .replace("STRUCTURE=wt_solvated.pdb", f"STRUCTURE={PDB_FILE}")  # absolute PDB path
        .replace("FILE=../HILLS",          f"FILE={HILLS_FILE}")         # absolute HILLS path
        .replace("FILE=COLVAR",            f"FILE={COLVAR_FILE}")        # absolute COLVAR path
    )

with open(LOCAL_PLUMED, "w") as f:
    f.write(plumed_script)

print(f"Wrote {LOCAL_PLUMED}  (WALKERS_ID={WALKER_ID})")

# ==========================================
# LOAD PDB
# No os.chdir() — all paths are absolute so
# cwd never matters.
# ==========================================

pdb = PDBFile(PDB_FILE)

forcefield = ForceField(
    "amber19/protein.ff19SB.xml",
    "amber19/tip3pfb.xml"
)

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0 * nanometer,
    constraints=HBonds
)

# ==========================================
# PLATFORM SETUP  (shared by both systems)
# ==========================================

import subprocess as _sp
try:
    _out = _sp.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        stderr=_sp.DEVNULL
    )
    N_GPUS = len(_out.strip().splitlines())
except Exception:
    N_GPUS = 1

print(f"[Walker {WALKER_ID}]  Detected {N_GPUS} GPU(s) → using DeviceIndex={WALKER_ID % N_GPUS}")

platform   = Platform.getPlatformByName("CUDA")
properties = {
    "Precision":   "mixed",
    "DeviceIndex": str(WALKER_ID % N_GPUS),
}

# ==========================================
# RESTART OR FRESH START
# ==========================================

if os.path.exists(CHECKPOINT):
    # ── Restart path: PLUMED system only ──────────────────────────────
    print(f"\nWalker {WALKER_ID}: loading checkpoint …\n")

    system.addForce(PlumedForce(plumed_script))
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.loadCheckpoint(CHECKPOINT)

else:
    # ── Fresh start: minimise on a PLAIN system (no PLUMED) ───────────
    # OpenMM's L-BFGS minimiser crashes with PLUMED forces active.
    # We minimise and equilibrate on a bare system, capture positions
    # and velocities, then hand them to the PLUMED production system.
    print(f"\nWalker {WALKER_ID}: fresh start …\n")

    plain_system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds
    )
    plain_integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    plain_sim = Simulation(pdb.topology, plain_system, plain_integrator, platform, properties)
    plain_sim.context.setPositions(pdb.positions)

    print("  Minimizing (plain system, no PLUMED) …")
    plain_sim.minimizeEnergy()

    print("  Equilibrating 200 ps (plain system, no PLUMED) …")
    plain_sim.step(50000)

    # Capture positions + velocities from plain sim
    plain_state = plain_sim.context.getState(getPositions=True, getVelocities=True)
    init_positions  = plain_state.getPositions()
    init_velocities = plain_state.getVelocities()
    del plain_sim, plain_system, plain_integrator   # free GPU memory before PLUMED init

    # Now build the production system WITH PLUMED
    print("  Building PLUMED production system …")
    system.addForce(PlumedForce(plumed_script))
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(init_positions)
    simulation.context.setVelocities(init_velocities)

# ==========================================
# REPORTERS
# ==========================================

simulation.reporters.append(DCDReporter(TRAJ, 5000))

simulation.reporters.append(
    StateDataReporter(
        LOG, 5000,
        step=True, time=True, temperature=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        speed=True, progress=True, remainingTime=True,
        totalSteps=25_000_000, separator="\t"
    )
)

simulation.reporters.append(
    StateDataReporter(
        sys.stdout, 5000,
        step=True, time=True, temperature=True,
        potentialEnergy=True, speed=True,
        progress=True, remainingTime=True,
        totalSteps=25_000_000, separator="\t"
    )
)

# ==========================================
# PRODUCTION  (chunked with checkpointing)
# ==========================================

total_steps = 25_000_000    # 100 ns
chunk       = 50_000        # 200 ps per save
completed   = 0

print(f"\nWalker {WALKER_ID}: starting production\n")

while completed < total_steps:

    simulation.step(chunk)
    completed += chunk

    simulation.saveCheckpoint(CHECKPOINT)
    shutil.copy(CHECKPOINT, CHECKPOINT_BAK)

    state = simulation.context.getState(getEnergy=True)
    pe    = state.getPotentialEnergy()
    print(f"[Walker {WALKER_ID}]  Step {completed:>10d}  PE = {pe}")

print(f"\nWalker {WALKER_ID}: finished.\n")