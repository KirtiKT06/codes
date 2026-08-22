import os
import shutil  # moved to top
import sys
from openmm.app import *
from openmm import *
from openmm.unit import nanometer, picosecond, picoseconds, kelvin

from openmmplumed import PlumedForce

# ==========================================
# LOAD PDB
# ==========================================

seed = int(sys.argv[1])

pdb = PDBFile("wt_solvated.pdb")

forcefield = ForceField(
    "amber19/protein.ff19SB.xml",   # FIXED: was "amber19-all.xml" (doesn't exist)
    "amber19/tip3pfb.xml"
)

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0*nanometer,
    constraints=HBonds
)

# ==========================================
# LOAD PLUMED
# ==========================================

with open("plumed.dat") as f:
    plumed_script = f.read()

system.addForce(
    PlumedForce(plumed_script)
)

# ==========================================
# INTEGRATOR
# ==========================================

integrator = LangevinMiddleIntegrator(
    300*kelvin,
    1/picosecond,
    0.002*picoseconds
)
integrator.setRandomNumberSeed(seed)
platform = Platform.getPlatformByName("CUDA")

properties = {
    "Precision": "mixed"
}

simulation = Simulation(
    pdb.topology,
    system,
    integrator,
    platform,
    properties
)

# ==========================================
# RESTART OR FRESH START
# ==========================================

CHECKPOINT = "/data/mutation_study/metaD/WT/checkpoint.chk"
CHECKPOINT_BAK = "/data/mutation_study/metaD/WT/checkpoint_backup.chk"
TRAJ = "/data/mutation_study/metaD/WT/traj.dcd"
LOG = "/data/mutation_study/metaD/WT/log.txt"

if os.path.exists(CHECKPOINT):
    print("\nLoading checkpoint...\n")
    simulation.loadCheckpoint(CHECKPOINT)
else:
    print("\nStarting new simulation...\n")
    simulation.context.setPositions(pdb.positions)

    print("Minimizing...")
    simulation.minimizeEnergy()

    print("Equilibrating...")
    simulation.step(50000)

# ==========================================
# REPORTERS
# ==========================================

simulation.reporters.append(
    DCDReporter(TRAJ, 5000)
)

simulation.reporters.append(
    StateDataReporter(
        LOG,
        5000,
        step=True,
        time=True,
        temperature=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=25000000,
        separator="\t"
    )
)

simulation.reporters.append(
    StateDataReporter(
        sys.stdout,          # prints to terminal
        5000,
        step=True,
        time=True,
        temperature=True,
        potentialEnergy=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=25000000,
        separator="\t"
    )
)

# ==========================================
# PRODUCTION
# ==========================================

total_steps = 25000000     # 100 ns
chunk = 50000                # 200 ps
completed = 0

print("\nStarting Metadynamics Production\n")

while completed < total_steps:

    simulation.step(chunk)
    completed += chunk

    simulation.saveCheckpoint(CHECKPOINT)
    shutil.copy(CHECKPOINT, CHECKPOINT_BAK)  # FIXED: shutil now imported at top

    state = simulation.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy()

    print(f"Step {completed:>10d}  PE = {pe}")

print("\nFinished.\n")
