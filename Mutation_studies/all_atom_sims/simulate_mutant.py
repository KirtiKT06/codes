import os
import sys

from openmm.app import *
from openmm import *
from openmm.unit import kelvin, picoseconds, nanoseconds, picosecond
from sys import stdout


# =====================================================
# INPUT
# =====================================================

mut = sys.argv[1]
rep = int(sys.argv[2])

print(f"Running mutant: {mut}, replica: {rep}")

PDB_DIR = "/home/feynman/projects/codes/Mutation_studies/mutant_pdbs"

OUT_ROOT = "/data/openmm_out/1PGA_mutations/run_330K"

SIM_NS = 125

TIMESTEP = 0.002 * picoseconds

EQUIL_STEPS = 2500000  # 5 ns equilibration
TOTAL_STEPS = int((SIM_NS * nanoseconds) / TIMESTEP) + EQUIL_STEPS

# =====================================================
# PATHS
# =====================================================

pdb_file = os.path.join(
    PDB_DIR,
    f"{mut}.pdb"
)

outdir = os.path.join(
    OUT_ROOT,
    f"{mut}_rep{rep}"
)

os.makedirs(
    outdir,
    exist_ok=True
)

traj_file = os.path.join(
    outdir,
    f"{mut}_rep{rep}_trajectory.dcd"
)

log_file = os.path.join(
    outdir,
    f"{mut}_rep{rep}_log.csv"
)

chk_file = os.path.join(
    outdir,
    f"{mut}_rep{rep}.chk"
)

xml_file = os.path.join(
    outdir,
    f"{mut}_rep{rep}_state.xml"
)

min_file = os.path.join(
    outdir,
    f"{mut}_rep{rep}_minimized.pdb"
)

final_file = os.path.join(
    outdir,
    f"{mut}_rep{rep}_final.pdb"
)

# =====================================================
# LOAD STRUCTURE
# =====================================================

print(f"\nRunning {mut}")

pdb = PDBFile(pdb_file)

forcefield = ForceField(
    "charmm36.xml",
    "implicit/obc2.xml"
)

modeller = Modeller(
    pdb.topology,
    pdb.positions
)

modeller.deleteWater()

modeller.addHydrogens(forcefield)

# =====================================================
# SYSTEM
# =====================================================

system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=NoCutoff,
    constraints=HBonds
)

temperature = 350 * kelvin

integrator = LangevinIntegrator(
    temperature,
    1/picosecond,
    TIMESTEP
)

integrator.setRandomNumberSeed(
    rep * 1000
)

platform = Platform.getPlatformByName(
    "CUDA"
)

properties = {
    "CudaPrecision": "mixed"
}

simulation = Simulation(
    modeller.topology,
    system,
    integrator,
    platform,
    properties
)

# # =====================================================
# # RESTART OR NEW
# # =====================================================

# if os.path.exists(chk_file):

#     print("Checkpoint found.")

#     simulation.loadCheckpoint(
#         chk_file
#     )

# else:

#     print("New simulation.")

#     simulation.context.setPositions(
#         modeller.positions
#     )

#     simulation.context.setVelocitiesToTemperature(
#         temperature
#     )

#     print("Minimizing...")

#     simulation.minimizeEnergy()

#     positions = simulation.context.getState(
#         getPositions=True
#     ).getPositions()

#     with open(min_file, "w") as f:

#         PDBFile.writeFile(
#             simulation.topology,
#             positions,
#             f
#         )

#     print("Equilibrating...")

#     simulation.step(EQUIL_STEPS)

# =====================================================
# RESTART OR NEW  — replace your current block with this
# =====================================================

if os.path.exists(chk_file):
    print("Checkpoint found.")
    simulation.loadCheckpoint(chk_file)

else:
    print("New simulation.")

    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    # ── Step 1: Aggressive minimization ──
    print("Minimizing (stage 1 — loose tolerance)...")
    simulation.minimizeEnergy(tolerance=10)

    print("Minimizing (stage 2 — tight tolerance)...")
    simulation.minimizeEnergy(tolerance=1)

    positions = simulation.context.getState(
        getPositions=True
    ).getPositions()

    with open(min_file, "w") as f:
        PDBFile.writeFile(simulation.topology, positions, f)

    # ── Step 2: Gradual heating from 50K to target temperature ──
    print("Gradual heating...")
    target_temp = temperature.value_in_unit(kelvin)
    heating_steps_per_stage = 50000  # 0.1 ns per stage

    for stage_temp in range(50, int(target_temp) + 1, 50):
        simulation.integrator.setTemperature(stage_temp * kelvin)
        simulation.context.setVelocitiesToTemperature(stage_temp * kelvin)
        simulation.step(heating_steps_per_stage)
        print(f"  Heated to {stage_temp} K")

    # Restore target temperature
    simulation.integrator.setTemperature(temperature)

    # ── Step 3: Equilibration at target temperature ──
    print("Equilibrating...")
    simulation.step(EQUIL_STEPS)

# =====================================================
# REPORTERS
# =====================================================

simulation.reporters.append(
    DCDReporter(
        traj_file,
        5000,
        append=os.path.exists(chk_file)
    )
)

simulation.reporters.append(
    StateDataReporter(
        log_file,
        5000,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=TOTAL_STEPS,
        separator=","
    )
)

simulation.reporters.append(
    StateDataReporter(
        stdout,
        5000,
        step=True,
        temperature=True,
        potentialEnergy=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=TOTAL_STEPS,
        separator=" | "
    )
)

simulation.reporters.append(
    CheckpointReporter(
        chk_file,
        100000
    )
)

# =====================================================
# RUN
# =====================================================

current_step = simulation.currentStep

remaining_steps = TOTAL_STEPS - current_step

print(f"Current step : {current_step}")
print(f"Remaining    : {remaining_steps}")

if remaining_steps > 0:

    print(f"Running remaining steps")

    simulation.step(
        remaining_steps
    )

else:

    print("Simulation already complete.")

# =====================================================
# SAVE FINAL
# =====================================================

simulation.saveState(
    xml_file
)

final_positions = simulation.context.getState(
    getPositions=True
).getPositions()

with open(final_file, "w") as f:

    PDBFile.writeFile(
        simulation.topology,
        final_positions,
        f
    )

print(f"{mut} complete")