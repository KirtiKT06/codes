"""
sop_dna_only_openmm.py
=======================
DNA-ONLY TIS simulation (no protein), explicit ions, for isolated
validation before recombining with the SOP protein model.

Reads the same DNA topology files as the combined script:
    dna_tis.pdb, dna_bonds.dat, dna_angles.dat, dna_stacking.dat,
    dna_hbonds.dat, dna_charges.dat
plus a trimmed set of input.json keys (see bottom of this file for the
minimal keys required).

All divergent-potential floors (stacking/H-bond rational-form clamp,
EV floor, Coulomb-correction floor) are already applied, matching the
fixes validated in the combined script.
"""

import json, sys, os
import numpy as np
from openmm import *
from openmm.app import *
import openmm.unit as unit
from openmm.app import PDBxFile

import dna_tis_params as DP

cfg = json.load(open("input.json"))

dna_pdb          = cfg.get("cg_pdb_dna", "dna_tis.pdb")
dna_bonds_file   = cfg.get("dna_bonds_file", "dna_bonds.dat")
dna_angles_file  = cfg.get("dna_angles_file", "dna_angles.dat")
dna_stack_file   = cfg.get("dna_stacking_file", "dna_stacking.dat")
dna_hbond_file   = cfg.get("dna_hbonds_file", "dna_hbonds.dat")
dna_charge_file  = cfg.get("dna_charges_file", "dna_charges.dat")

dcd_out   = cfg["dcd_prefix"] + "_dnaonly.dcd"
log_out   = cfg["data_name"].replace(".csv", "_dnaonly.csv")
chk_equil = cfg["checkpoint_name"].replace(".chk", "_dnaonly_equil.chk")
chk_prod  = chk_equil.replace("_equil.chk", "_prod.chk")
final_out = cfg["final_state_name"].replace(".pdb", "_dnaonly.cif")

DIELECTRIC_WATER = cfg["dielectric_water"]
KCl_mM, MgCl2_mM = cfg["KCl_mM"], cfg["MgCl2_mM"]
SIGMA_K, SIGMA_CL, SIGMA_Mg = cfg["rK"], cfg["rCl"], cfg["rMg"]

TEMPERATURE   = cfg["Temp"]
FRICTION      = cfg["friction_coeff_dna"]
TIMESTEP_PS   = cfg["time_step"]
N_STEPS_EQUIL = cfg["numsteps_equil"]
N_STEPS_PROD  = cfg["numsteps_prod"]
REPORT_EVERY  = cfg["data_interval"]
BOX_SCALE     = cfg["box_scale_factor"]
DO_MINIMISE   = cfg["minimization"]
MIN_TOL       = cfg["minimization_tol"]
PLATFORM      = cfg["platform_type"]
RESTART       = cfg["restart"]
DO_CMR        = cfg["ComMotionRemover"]
CMR_FREQ      = cfg["CMMR_frequency"]
PME_CUTOFF    = cfg.get("pme_cutoff", 30.0) * 0.1
NONLOCAL_CUTOFF = cfg.get("nonlocal_cutoff", 30.0) * 0.1

def kcal_to_kJ(x): return x * 4.184
def A_to_nm(x):    return x * 0.1

ONE_4PI_EPS0 = 138.935458
dielectric_sqrt = np.sqrt(DIELECTRIC_WATER)
print(f"T = {TEMPERATURE} K, DIELECTRIC_WATER = {DIELECTRIC_WATER}, "
      f"dt = {TIMESTEP_PS*1000:.3f} fs")

# ─────────────────────────────────────────────────────────────────────
# LOAD DNA
# ─────────────────────────────────────────────────────────────────────
def read_pdb_beads(path):
    beads, types = [], []
    with open(path) as f:
        for line in f:
            if line[:6].strip() != "ATOM":
                continue
            beads.append({
                "chain": line[21], "resid": int(line[22:26]),
                "resname": line[17:20].strip(),
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            })
            types.append(line[12:16].strip())
    return beads, types

print("Loading DNA TIS structure ...")
dna_beads, dna_bead_types = read_pdb_beads(dna_pdb)
N_dna = len(dna_beads)
print(f"  {N_dna} DNA beads loaded.")

def load_table(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows

dna_bonds_raw  = load_table(dna_bonds_file)
dna_angles_raw = load_table(dna_angles_file)
dna_stack_raw  = load_table(dna_stack_file)
dna_hbond_raw  = load_table(dna_hbond_file)
dna_charge_raw = load_table(dna_charge_file)
print(f"  {len(dna_bonds_raw)} bonds, {len(dna_angles_raw)} angles, "
      f"{len(dna_stack_raw)} stacks, {len(dna_hbond_raw)} H-bonds, "
      f"{len(dna_charge_raw)} charged phosphates.")

# ─────────────────────────────────────────────────────────────────────
# TOPOLOGY
# ─────────────────────────────────────────────────────────────────────
print("Building OpenMM topology ...")
topology = Topology()
dna_chain_ids = sorted(set(b["chain"] for b in dna_beads))
omm_chains = {ch: topology.addChain(id=ch) for ch in dna_chain_ids}
for b, btype in zip(dna_beads, dna_bead_types):
    res = topology.addResidue(b["resname"], omm_chains[b["chain"]])
    elem = Element.getBySymbol("P") if btype == "P" else Element.getBySymbol("C")
    topology.addAtom(btype, elem, res)

dna_coords_nm = np.array([[b["x"], b["y"], b["z"]] for b in dna_beads]) * 0.1
solute_min, solute_max = dna_coords_nm.min(0), dna_coords_nm.max(0)
solute_span = solute_max - solute_min
min_box = 2.0 * NONLOCAL_CUTOFF
box_size = np.maximum(solute_span * BOX_SCALE, min_box * np.ones(3))
centre = 0.5 * box_size
solute_centre = 0.5 * (solute_min + solute_max)
dna_coords_nm = dna_coords_nm - solute_centre + centre

topology.setPeriodicBoxVectors((Vec3(box_size[0],0,0)*unit.nanometer,
                                 Vec3(0,box_size[1],0)*unit.nanometer,
                                 Vec3(0,0,box_size[2])*unit.nanometer))

# ─────────────────────────────────────────────────────────────────────
# CHARGES + IONS
# ─────────────────────────────────────────────────────────────────────
dna_charges = {int(r[0]): float(r[1]) for r in dna_charge_raw}
net_charge = sum(dna_charges.values())
print(f"  Net DNA charge: {net_charge:+.1f} e")

N_A = 6.022e23
box_vol_L = float(np.prod(box_size)) * 1e-27 * 1e3
n_k = max(0, round(KCl_mM * 1e-3 * N_A * box_vol_L))
n_mg = max(0, round(MgCl2_mM * 1e-3 * N_A * box_vol_L))
n_cl = n_k + 2*n_mg
if net_charge > 0:
    n_cl += int(round(net_charge))
else:
    n_k += int(round(-net_charge))
N_ions = n_k + n_cl + n_mg
print(f"  Ions: {n_k} K+, {n_cl} Cl-, {n_mg} Mg2+ = {N_ions} total")

ion_chain = topology.addChain(id="Z")
for _ in range(n_k):
    res = topology.addResidue("K", ion_chain); topology.addAtom("K", Element.getBySymbol("K"), res)
for _ in range(n_cl):
    res = topology.addResidue("CL", ion_chain); topology.addAtom("CL", Element.getBySymbol("Cl"), res)
for _ in range(n_mg):
    res = topology.addResidue("MG", ion_chain); topology.addAtom("MG", Element.getBySymbol("Mg"), res)

print("Placing ions (checked against solute AND previously placed ions) ...")
np.random.seed(cfg.get("random_seed", 42))
placed = dna_coords_nm.copy()
ion_positions = []
for _ in range(N_ions):
    for _ in range(50000):
        pos = np.random.rand(3) * box_size
        if np.linalg.norm(placed - pos, axis=1).min() > 0.6:
            ion_positions.append(pos)
            placed = np.vstack([placed, pos])
            break
    else:
        pos = np.random.rand(3) * box_size
        ion_positions.append(pos)
        placed = np.vstack([placed, pos])

all_positions = np.vstack([dna_coords_nm, np.array(ion_positions) if N_ions else np.zeros((0,3))])
positions_openmm = [Vec3(*p)*unit.nanometer for p in all_positions]
N_total = N_dna + N_ions
print(f"  Total particles: {N_total}")

ION_K_START = N_dna
ION_CL_START = ION_K_START + n_k
ION_MG_START = ION_CL_START + n_cl

# ─────────────────────────────────────────────────────────────────────
# SYSTEM
# ─────────────────────────────────────────────────────────────────────
print("Building system ...")
system = System()
system.setDefaultPeriodicBoxVectors(Vec3(box_size[0],0,0)*unit.nanometer,
                                     Vec3(0,box_size[1],0)*unit.nanometer,
                                     Vec3(0,0,box_size[2])*unit.nanometer)
for _ in range(N_total):
    system.addParticle(1.0 * unit.amu)

dna_indices = list(range(0, N_dna))

print("  DNA harmonic bonds ...")
dna_bond_force = CustomBondForce("k*(r-r0)^2")
dna_bond_force.addPerBondParameter("r0"); dna_bond_force.addPerBondParameter("k")
dna_bond_force.setUsesPeriodicBoundaryConditions(True)
dna_bond_pairs = set()
for row in dna_bonds_raw:
    i, j = int(row[0]), int(row[1])
    r0, k = A_to_nm(float(row[2])), kcal_to_kJ(float(row[3])) * 100.0
    dna_bond_force.addBond(i, j, [r0, k])
    dna_bond_pairs.add((min(i,j), max(i,j)))
dna_bond_force.setForceGroup(0)
system.addForce(dna_bond_force)
print(f"    {dna_bond_force.getNumBonds()} bonds.")

print("  DNA harmonic angles ...")
dna_angle_force = CustomAngleForce("k*(theta-theta0)^2")
dna_angle_force.addPerAngleParameter("theta0"); dna_angle_force.addPerAngleParameter("k")
for row in dna_angles_raw:
    i, j, k_idx = int(row[0]), int(row[1]), int(row[2])
    theta0, k = float(row[3]), kcal_to_kJ(float(row[4]))
    dna_angle_force.addAngle(i, j, k_idx, [theta0, k])
dna_angle_force.setForceGroup(1)
system.addForce(dna_angle_force)
print(f"    {dna_angle_force.getNumAngles()} angles.")

print("  DNA stacking (denominator-clamped) ...")
TWOPI = "6.283185307"
stack5 = CustomCompoundBondForce(
    5,
    "U0/max(1 + kl*(l-l0)^2 + kphi*dphi1w^2 + kphi*dphi2w^2, 0.05);"
    "l = distance(p2,p3);"
    f"dphi1w = dphi1 - {TWOPI}*floor(dphi1/{TWOPI} + 0.5);"
    "dphi1 = dihedral(p1,p2,p3,p4) - phi1_0;"
    f"dphi2w = dphi2 - {TWOPI}*floor(dphi2/{TWOPI} + 0.5);"
    "dphi2 = dihedral(p5,p1,p2,p3) - phi2_0"
)
for p in ("U0", "l0", "phi1_0", "phi2_0"):
    stack5.addPerBondParameter(p)
stack5.addGlobalParameter("kl", DP.K_L / 0.01)
stack5.addGlobalParameter("kphi", DP.K_PHI)

stack4 = CustomCompoundBondForce(
    4,
    "U0/max(1 + kl*(l-l0)^2 + kphi*dphi1w^2, 0.05);"
    "l = distance(p2,p3);"
    f"dphi1w = dphi1 - {TWOPI}*floor(dphi1/{TWOPI} + 0.5);"
    "dphi1 = dihedral(p1,p2,p3,p4) - phi1_0"
)
for p in ("U0", "l0", "phi1_0"):
    stack4.addPerBondParameter(p)
stack4.addGlobalParameter("kl", DP.K_L / 0.01)
stack4.addGlobalParameter("kphi", DP.K_PHI)

kB = DP.KB_KCAL
n_stack5 = n_stack4 = 0
for row in dna_stack_raw:
    s_i, b_i, b_j, s_j, p_i = [int(x) for x in row[0:5]]
    l0_nm = A_to_nm(float(row[5]))
    phi1_0, phi2_0 = float(row[6]), float(row[7])
    h, s, dG0 = float(row[8]), float(row[9]), float(row[10])
    U0_S = kcal_to_kJ(-h + kB * (TEMPERATURE - DP.STACKING_TREF) * s)
    gi = [s_i, b_i, b_j, s_j]
    if p_i >= 0:
        stack5.addBond(gi + [p_i], [U0_S, l0_nm, phi1_0, phi2_0])
        n_stack5 += 1
    else:
        stack4.addBond(gi, [U0_S, l0_nm, phi1_0])
        n_stack4 += 1
stack5.setForceGroup(2); stack4.setForceGroup(2)
system.addForce(stack5); system.addForce(stack4)
print(f"    {n_stack5} 5-particle + {n_stack4} 4-particle stacking terms.")

print("  DNA Watson-Crick hydrogen bonds (denominator-clamped) ...")
hb6 = CustomCompoundBondForce(
    6,
    "UHB0/max(1 + kd*(d-d0)^2 + kth*dth1w^2 + kth*dth2w^2"
    " + kps*dpsi1w^2 + kps*dpsi2w^2 + kps*dpsi3w^2, 0.05);"
    "d = distance(p2,p3);"
    f"dth1w = dth1 - {TWOPI}*floor(dth1/{TWOPI} + 0.5); dth1 = angle(p1,p2,p3) - th1_0;"
    f"dth2w = dth2 - {TWOPI}*floor(dth2/{TWOPI} + 0.5); dth2 = angle(p4,p3,p2) - th2_0;"
    f"dpsi1w = dpsi1 - {TWOPI}*floor(dpsi1/{TWOPI} + 0.5); dpsi1 = dihedral(p1,p2,p3,p4) - psi1_0;"
    f"dpsi2w = dpsi2 - {TWOPI}*floor(dpsi2/{TWOPI} + 0.5); dpsi2 = dihedral(p5,p4,p3,p2) - psi2_0;"
    f"dpsi3w = dpsi3 - {TWOPI}*floor(dpsi3/{TWOPI} + 0.5); dpsi3 = dihedral(p6,p1,p2,p3) - psi3_0"
)
for p in ("UHB0", "d0", "th1_0", "th2_0", "psi1_0", "psi2_0", "psi3_0"):
    hb6.addPerBondParameter(p)
hb6.addGlobalParameter("kd", DP.K_D / 0.01)
hb6.addGlobalParameter("kth", DP.K_THETA)
hb6.addGlobalParameter("kps", DP.K_PSI)

hb4 = CustomCompoundBondForce(
    4,
    "UHB0/max(1 + kd*(d-d0)^2 + kth*dth1w^2 + kth*dth2w^2 + kps*dpsi1w^2, 0.05);"
    "d = distance(p2,p3);"
    f"dth1w = dth1 - {TWOPI}*floor(dth1/{TWOPI} + 0.5); dth1 = angle(p1,p2,p3) - th1_0;"
    f"dth2w = dth2 - {TWOPI}*floor(dth2/{TWOPI} + 0.5); dth2 = angle(p4,p3,p2) - th2_0;"
    f"dpsi1w = dpsi1 - {TWOPI}*floor(dpsi1/{TWOPI} + 0.5); dpsi1 = dihedral(p1,p2,p3,p4) - psi1_0"
)
for p in ("UHB0", "d0", "th1_0", "th2_0", "psi1_0"):
    hb4.addPerBondParameter(p)
hb4.addGlobalParameter("kd", DP.K_D / 0.01)
hb4.addGlobalParameter("kth", DP.K_THETA)
hb4.addGlobalParameter("kps", DP.K_PSI)

n_hb6 = n_hb4 = 0
dna_hbond_pairs = set()
for row in dna_hbond_raw:
    s1, b1, b5, s5, p6, p2 = [int(x) for x in row[0:6]]
    d0 = A_to_nm(float(row[6]))
    th1_0, th2_0, psi1_0, psi2_0, psi3_0 = [float(x) for x in row[7:12]]
    UHB0_kJ = kcal_to_kJ(float(row[12]))
    gi = [s1, b1, b5, s5]
    dna_hbond_pairs.add((min(b1,b5), max(b1,b5)))
    if p6 >= 0 and p2 >= 0:
        hb6.addBond(gi + [p6, p2], [UHB0_kJ, d0, th1_0, th2_0, psi1_0, psi2_0, psi3_0])
        n_hb6 += 1
    else:
        hb4.addBond(gi, [UHB0_kJ, d0, th1_0, th2_0, psi1_0])
        n_hb4 += 1
hb6.setForceGroup(3); hb4.setForceGroup(3)
system.addForce(hb6); system.addForce(hb4)
print(f"    {n_hb6} 6-particle + {n_hb4} 4-particle H-bond terms.")

print("  DNA excluded volume (WCA, floor-clamped) ...")
ev_dna = CustomNonbondedForce("step(D0-r) * eps0 * ((D0/max(r,0.15))^12 - 2*(D0/max(r,0.15))^6 + 1)")
ev_dna.addGlobalParameter("D0", A_to_nm(DP.EV_D0))
ev_dna.addGlobalParameter("eps0", kcal_to_kJ(DP.EV_EPS0))
ev_dna.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
ev_dna.setCutoffDistance(A_to_nm(DP.EV_D0) * unit.nanometer)
for _ in range(N_total):
    ev_dna.addParticle([])
ev_dna.addInteractionGroup(dna_indices, dna_indices)

from collections import defaultdict
adj = defaultdict(set)
for i, j in dna_bond_pairs:
    adj[i].add(j); adj[j].add(i)

dna_master_exclusions = set(dna_bond_pairs)   # 1-2 only
for pair in dna_hbond_pairs:
    dna_master_exclusions.add((min(pair), max(pair)))

for i, j in dna_master_exclusions:
    ev_dna.addExclusion(i, j)
ev_dna.setForceGroup(4)
system.addForce(ev_dna)
print(f"    {len(dna_master_exclusions)} EV exclusions applied.")

print("  Electrostatics (explicit ions, PME) ...")
nb_force = NonbondedForce()
nb_force.setNonbondedMethod(NonbondedForce.PME)
nb_force.setCutoffDistance(PME_CUTOFF * unit.nanometer)
nb_force.setEwaldErrorTolerance(1e-4)
nb_force.setReactionFieldDielectric(DIELECTRIC_WATER)

for i in range(N_dna):
    q = dna_charges.get(i, 0.0) / dielectric_sqrt
    nb_force.addParticle(q, A_to_nm(DP.EV_D0)*unit.nanometer, 0.0)
for _ in range(n_k):
    nb_force.addParticle(+1.0/dielectric_sqrt, A_to_nm(SIGMA_K)*unit.nanometer, 0.0)
for _ in range(n_cl):
    nb_force.addParticle(-1.0/dielectric_sqrt, A_to_nm(SIGMA_CL)*unit.nanometer, 0.0)
for _ in range(n_mg):
    nb_force.addParticle(+2.0/dielectric_sqrt, A_to_nm(SIGMA_Mg)*unit.nanometer, 0.0)

for i, j in dna_master_exclusions:
    nb_force.addException(i, j, 0.0, 1.0, 0.0)
nb_force.setForceGroup(5)
system.addForce(nb_force)

if DO_CMR:
    system.addForce(CMMotionRemover(CMR_FREQ))

# ─────────────────────────────────────────────────────────────────────
# INTEGRATOR / RUN
# ─────────────────────────────────────────────────────────────────────
integrator = LangevinMiddleIntegrator(TEMPERATURE*unit.kelvin,
                                       FRICTION/unit.picosecond,
                                       TIMESTEP_PS*unit.picoseconds)
platform = Platform.getPlatformByName(PLATFORM)
simulation = Simulation(topology, system, integrator, platform)

if RESTART:
    simulation.loadCheckpoint(chk_equil)
else:
    simulation.context.setPositions(positions_openmm)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE*unit.kelvin)

print("Scanning DNA angles for large deviations ...")
state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
worst = []
for idx in range(dna_angle_force.getNumAngles()):
    i, j, k, params = dna_angle_force.getAngleParameters(idx)
    theta0, kk = params
    v1 = pos[i] - pos[j]
    v2 = pos[k] - pos[j]
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    theta = np.arccos(cos_t)
    dev = abs(theta - theta0)
    worst.append((dev, idx, i, j, k, np.degrees(theta), np.degrees(theta0)))
worst.sort(reverse=True)
for dev, idx, i, j, k, th, th0 in worst[:10]:
    print(f"  angle_idx={idx}: beads({i},{j},{k})  live={th:.1f}°  eq={th0:.1f}°  dev={np.degrees(dev):.1f}°")

pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
# bead 854 was the middle/S atom in one collapsed angle (853,854,856)
mg_start, mg_end = ION_CL_START, ION_CL_START + n_mg  # adjust to your actual Mg index range
for p_idx in (853, 856):
    dists = np.linalg.norm(pos[mg_start:mg_end] - pos[p_idx], axis=1) * 10
    print(f"P bead {p_idx}: closest Mg2+ = {dists.min():.2f} A")

FORCE_GROUPS = [(0,"DNA bonds"), (1,"DNA angles"), (2,"DNA stacking"),
                (3,"DNA H-bonds"), (4,"DNA EV (WCA)"), (5,"PME electrostatics")]

def print_energy_breakdown(label):
    print(f"\nEnergy breakdown ({label}):")
    for group, name in FORCE_GROUPS:
        e = simulation.context.getState(getEnergy=True, groups={group}
                ).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        print(f"  {name:24s}: {e:12.2f} kcal/mol")

if DO_MINIMISE and not RESTART:
    print("Energy minimisation (staged) ...")
    for i in range(50):
        simulation.minimizeEnergy(maxIterations=200, tolerance=MIN_TOL)
        e = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        if e != e:
            raise RuntimeError(f"Energy went NaN at minimization stage {i}")

print_energy_breakdown("initial")

simulation.reporters.append(DCDReporter(dcd_out, REPORT_EVERY))
simulation.reporters.append(StateDataReporter(
    log_out, REPORT_EVERY, step=True, time=True, potentialEnergy=True,
    kineticEnergy=True, temperature=True, progress=True, remainingTime=True,
    totalSteps=N_STEPS_EQUIL+N_STEPS_PROD, separator=","))
simulation.reporters.append(StateDataReporter(
    sys.stdout, 5000, step=True, temperature=True, potentialEnergy=True,
    speed=True, progress=True, remainingTime=True,
    totalSteps=N_STEPS_EQUIL+N_STEPS_PROD, separator=" | "))

print(f"\nRunning equilibration ({N_STEPS_EQUIL} steps) ...")
simulation.step(N_STEPS_EQUIL)
print_energy_breakdown("after equilibration")
simulation.saveCheckpoint(chk_equil)

print(f"\nRunning production ({N_STEPS_PROD} steps) ...")
simulation.step(N_STEPS_PROD)
simulation.saveCheckpoint(chk_prod)

state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
with open(final_out, "w") as f:
    PDBxFile.writeFile(simulation.topology, state.getPositions(), f)

print("\nDNA-only simulation complete.")
print(f"  Trajectory      : {dcd_out}")
print(f"  Log             : {log_out}")
print(f"  Final structure : {final_out}")