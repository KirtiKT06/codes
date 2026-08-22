"""
Explicit Solvent MD Simulation for 1PGA Mutants
================================================
Uses TIP3P water, PME electrostatics, NPT ensemble.
This is the proper setup for computing ΔΔG proxies.

Key differences from implicit solvent:
  - TIP3P explicit water box (10 Å padding)
  - PME electrostatics (no cutoff artifacts)
  - NPT ensemble (pressure coupling)
  - Proper desolvation energy included

Usage:
  python simulate_explicit.py <mutant_name> <replica>

Example:
  python simulate_explicit.py E27L 1
  python simulate_explicit.py 1PGA 1
"""

import os
import sys

from openmm.app import *
from openmm import *
from openmm.unit import picoseconds, nanoseconds, kilojoules_per_mole, angstroms, molar, bar, kelvin, amu, nanometer, picosecond
from sys import stdout

# =====================================================
# INPUT
# =====================================================

mut = sys.argv[1]
rep = int(sys.argv[2])

print(f"Running mutant: {mut}, replica: {rep}")

PROJECT_ROOT = os.getcwd()
PDB_DIR  = os.path.join(PROJECT_ROOT, "mutant_pdbs")
OUT_ROOT = os.path.join(PROJECT_ROOT, "Outputs")

# Shorter simulation for explicit — 50 ns is sufficient
# explicit solvent is ~10x slower than implicit
SIM_NS    = 5
TIMESTEP  = 0.002 * picoseconds
EQUIL_STEPS = 1000000   # 2 ns equilibration (shorter — NPT handles it)
TOTAL_STEPS = int((SIM_NS * nanoseconds) / TIMESTEP) + EQUIL_STEPS

# =====================================================
# PATHS
# =====================================================

pdb_file = os.path.join(PDB_DIR, f"{mut}.pdb")

outdir = os.path.join(OUT_ROOT, f"{mut}_rep{rep}")
os.makedirs(outdir, exist_ok=True)

traj_file  = os.path.join(outdir, f"{mut}_rep{rep}_trajectory.dcd")
log_file   = os.path.join(outdir, f"{mut}_rep{rep}_log.csv")
chk_file   = os.path.join(outdir, f"{mut}_rep{rep}.chk")
xml_file   = os.path.join(outdir, f"{mut}_rep{rep}_state.xml")
min_file   = os.path.join(outdir, f"{mut}_rep{rep}_minimized.pdb")
final_file = os.path.join(outdir, f"{mut}_rep{rep}_final.pdb")
# Add this line with your other path definitions
solvated_pdb = os.path.join(outdir, f"{mut}_rep{rep}_solvated.pdb")

# =====================================================
# LOAD STRUCTURE
# =====================================================

print(f"\nLoading {mut}...")

pdb = PDBFile(pdb_file)

# CHARMM36 force field + TIP3P water
forcefield = ForceField(
    "charmm36.xml",
    "charmm36/water.xml"   # TIP3P parameters
)

# modeller = Modeller(pdb.topology, pdb.positions)

# # Add hydrogens
# modeller.addHydrogens(forcefield)

# # Add explicit water box — 10 Å padding, TIP3P
# print("Adding water box...")
# modeller.addSolvent(
#     forcefield,
#     model="tip3p",
#     padding=10.0 * angstroms,
#     ionicStrength=0.15 * molar,   # 150 mM NaCl — physiological
#     positiveIon="Na+",
#     negativeIon="Cl-"
# )

modeller = Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield)

# Only build water box if we don't have a saved solvated structure
if os.path.exists(solvated_pdb):
    print("Loading saved solvated structure...")
    solvated = PDBFile(solvated_pdb)
    modeller = Modeller(solvated.topology, solvated.positions)
    n_waters = sum(1 for r in modeller.topology.residues()
                   if r.name == "HOH")
    print(f"  Loaded {n_waters} water molecules")
    print(f"  Total atoms: {modeller.topology.getNumAtoms()}")
else:
    print("Adding water box...")
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        padding=10.0 * angstroms,
        ionicStrength=0.15 * molar,
        positiveIon="Na+",
        negativeIon="Cl-"
    )
    n_waters = sum(1 for r in modeller.topology.residues()
                   if r.name == "HOH")
    print(f"  Added {n_waters} water molecules")
    print(f"  Total atoms: {modeller.topology.getNumAtoms()}")

    # Save solvated structure immediately for future restarts
    with open(solvated_pdb, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    print(f"  Saved solvated structure for restarts.")

n_waters = sum(
    1 for r in modeller.topology.residues()
    if r.name == "HOH"
)
print(f"  Added {n_waters} water molecules")
print(f"  Total atoms: {modeller.topology.getNumAtoms()}")

# =====================================================
# SYSTEM
# =====================================================

system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME,           # Particle Mesh Ewald — correct electrostatics
    nonbondedCutoff=12.0 * angstroms,
    switchDistance=10.0 * angstroms,
    constraints=HBonds,
    hydrogenMass=1.5 * amu          # HMass repartitioning — allows 2fs timestep
)

print("\nForces in the system:")
for i in range(system.getNumForces()):
    print(i, type(system.getForce(i)))

# Add Monte Carlo barostat for NPT (constant pressure)
temperature = 350 * kelvin
pressure    = 1 * bar
system.addForce(
    MonteCarloBarostat(pressure, temperature, 25)
)

integrator = LangevinMiddleIntegrator(
    temperature,
    1 / picosecond,
    TIMESTEP
)
integrator.setRandomNumberSeed(rep * 1000)

platform   = Platform.getPlatformByName("CUDA")
properties = {"CudaPrecision": "mixed"}

simulation = Simulation(
    modeller.topology,
    system,
    integrator,
    platform,
    properties
)

# =====================================================
# RESTART OR NEW
# =====================================================

if os.path.exists(chk_file):
    print("Checkpoint found — resuming.")
    simulation.loadCheckpoint(chk_file)

else:
    print("New simulation.")
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    # ── Stage 1: Energy minimization ──
    # print("Minimizing (stage 1)...")
    # simulation.minimizeEnergy(tolerance=10)
    # print("Minimizing (stage 2)...")
    # simulation.minimizeEnergy(tolerance=1)

    # positions = simulation.context.getState(
    #     getPositions=True
    # ).getPositions()
    # with open(min_file, "w") as f:
    #     PDBFile.writeFile(simulation.topology, positions, f)
    # print(f"  Saved minimized structure.")

    # # ── Stage 1: Aggressive minimization ──
    # print("Minimizing (stage 1 — water only)...")
    # protein_atoms = [
    #     a.index for a in simulation.topology.atoms()
    #     if a.residue.name not in ("HOH", "Na+", "Cl-")
    # ]
    # # Fix protein, minimize water first
    # for idx in protein_atoms:
    #     simulation.system.setParticleMass(idx, 0)   # freeze protein
    # simulation.context.reinitialize(preserveState=True)
    # simulation.minimizeEnergy(tolerance=10, maxIterations=500)

    # # Restore protein masses
    # ff_system = forcefield.createSystem(
    #     modeller.topology,
    #     nonbondedMethod=PME,
    #     nonbondedCutoff=12.0 * angstroms,
    #     switchDistance=10.0 * angstroms,
    #     constraints=HBonds,
    #     hydrogenMass=1.5 * amu
    # )
    # for i in range(ff_system.getNumParticles()):
    #     simulation.system.setParticleMass(
    #         i, ff_system.getParticleMass(i)
    #     )
    # simulation.context.reinitialize(preserveState=True)

    
    print("Minimizing (stage 2 — full system)...")
    simulation.minimizeEnergy(tolerance=1, maxIterations=1000)
    positions = simulation.context.getState(
    getPositions=True
    ).getPositions()

    with open(min_file, "w") as f:
        PDBFile.writeFile(
            simulation.topology,
            positions,
            f
        )

    print("Saved minimized structure.")

    state = simulation.context.getState(
    getEnergy=True
    )

    print(
        "PE after minimization:",
        state.getPotentialEnergy()
    )

    # ── Stage 2: NVT equilibration (protein restrained) ──
    print("Equilibrating NVT (protein restrained)...")
    protein_atoms = [
        a.index for a in simulation.topology.atoms()
        if a.residue.name not in ("HOH", "Na+", "Cl-")
    ]
    restraint = CustomExternalForce(
        "k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
    )
    restraint.addGlobalParameter("k", 100.0 * kilojoules_per_mole / nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    state_min = simulation.context.getState(getPositions=True)
    pos = state_min.getPositions()
    for idx in protein_atoms:
        restraint.addParticle(
            idx,
            [pos[idx].x, pos[idx].y, pos[idx].z]
        )
    system.addForce(restraint)
    simulation.context.reinitialize(preserveState=True)

    # NVT equilibration with restraints (200 ps)
    simulation.context.setVelocitiesToTemperature(
    100 * kelvin
    )
    simulation.step(100000)

    # Remove restraints completely
    system.removeForce(system.getNumForces() - 1)
    simulation.context.reinitialize(preserveState=True)

    print("Protein restraints removed.")

    # ── Stage 2: NVT equilibration — gradual heating ──
    print("Heating gradually from 100K to 475K...")
    for stage_temp in [100, 150, 200, 250, 300, 350]:

        simulation.integrator.setTemperature(
            stage_temp * kelvin
        )

        simulation.context.setParameter(
            MonteCarloBarostat.Temperature(),
            stage_temp * kelvin
        )

        simulation.step(25000)

        print(f"  Heated to {stage_temp} K")

    # ── Stage 3: NPT equilibration at target temperature ──
    print("Equilibrating NPT (unrestrained)...")
    simulation.integrator.setTemperature(temperature)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.step(EQUIL_STEPS)
    print("Equilibration complete.")

# =====================================================
# REPORTERS
# =====================================================

simulation.reporters.append(
    DCDReporter(traj_file, 5000, append=os.path.exists(chk_file))
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
        volume=True,        # important for NPT
        density=True,       # important for NPT
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
        density=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=TOTAL_STEPS,
        separator=" | "
    )
)

simulation.reporters.append(
    CheckpointReporter(chk_file, 100000)
)

# =====================================================
# RUN
# =====================================================

current_step   = simulation.currentStep
remaining_steps = TOTAL_STEPS - current_step

print(f"Current step  : {current_step}")
print(f"Remaining     : {remaining_steps}")

if remaining_steps > 0:
    print("Running production...")
    simulation.step(remaining_steps)
else:
    print("Simulation already complete.")

# =====================================================
# SAVE FINAL STATE
# =====================================================

simulation.saveState(xml_file)

final_positions = simulation.context.getState(
    getPositions=True
).getPositions()

with open(final_file, "w") as f:
    PDBFile.writeFile(simulation.topology, final_positions, f)

print(f"\n{mut} rep {rep} complete.")