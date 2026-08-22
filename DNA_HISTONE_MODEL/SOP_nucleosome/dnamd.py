"""
dnamd.py
========
TIS-DNA (Chakraborty, Hori & Thirumalai, JCTC 2018) OpenMM driver.

Box sizing and explicit-ion counting/placement here are ported DIRECTLY
from sop_hpc_openmm.py's logic (same formulas, same auto_ions philosophy:
box = DNA span x box_scale_factor, floored by 2x cutoff; ion counts from
salt concentration x box volume + neutralization of the DNA's net bare
charge) so behavior is consistent between your protein and DNA drivers.
"""

import json, sys, os
import numpy as np
from openmm import *
from openmm.app import *
import openmm.unit as unit
from openmm import XmlSerializer

# ─────────────────────────────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────────────────────────────
cfg = json.load(open("input_dna.json"))

pdb_file        = cfg["cg_pdb_dna"]
dna_chains_cfg  = cfg["dna_chains"]

dcd_out    = cfg["dcd_prefix"] + ".dcd"
log_out    = cfg["data_name"]
chk_equil  = cfg["checkpoint_name"]
chk_prod   = chk_equil.replace(".chk", "_prod_final.chk")
energy_breakdown = cfg["energy_breakdown"]
initial_pdb = cfg["initial_pdb_name"]
final_pdb  = cfg["final_state_name"]

CHK_DIR   = cfg["CHK_DIR"]
STATE_DIR = cfg["STATE_DIR"]

D0        = cfg["D0"]          # eq 8, Angstrom
EPS0      = cfg["eps0"]        # eq 8, kcal/mol
U0_HB     = cfg["U0_HB"]
K_L       = cfg["kl_stack"]
K_PHI     = cfg["kphi_stack"]
K_D       = cfg["kd_HB"]
K_THETA   = cfg["ktheta_HB"]
K_PSI     = cfg["kpsi_HB"]

electrostatics_mode = cfg["electrostatics_mode"]
monovalent_mM       = cfg["monovalent_salt_mM"]
PME_CUTOFF          = cfg["pme_cutoff"] * 0.1   # Angstrom -> nm

auto_ions  = cfg.get("auto_ions", True)
KCl_mM     = cfg["KCl_mM"]
MgCl2_mM   = cfg["MgCl2_mM"]
neutralize = cfg.get("neutralize", True)
SIGMA_K, SIGMA_CL, SIGMA_MG = cfg["rK"], cfg["rCl"], cfg["rMg"]
EPS_K, EPS_CL, EPS_MG       = cfg["epsK"], cfg["epsCl"], cfg["epsMg"]
MASS_K, MASS_CL, MASS_MG    = cfg["mK"], cfg["mCl"], cfg["mMg"]
Q_K, Q_CL, Q_MG             = cfg["qK"], cfg["qCl"], cfg["qMg"]
ION_MIN_DIST = cfg["ion_min_dist"] * 0.1   # Angstrom -> nm

BOX_PADDING = cfg["box_padding"]
BOX_SCALE   = cfg["box_scale_factor"]
NONLOCAL_CUTOFF = cfg["nonlocal_cutoff"]

TEMPERATURE   = cfg["Temp"]
Boltz_Const   = cfg["Boltz_Const"]
FRICTION      = cfg["friction_coeff"]
TIMESTEP_PS   = cfg["time_step"]
N_STEPS_EQUIL = cfg["numsteps_equil"]
N_STEPS_PROD  = cfg["numsteps_prod"]
REPORT_EVERY  = cfg["data_interval"]

DO_MINIMISE  = cfg["minimization"]
MIN_MAXITER  = cfg["minimization_maxiter"]
MIN_TOL      = cfg["minimization_tol"]
PLATFORM     = cfg["platform_type"]
RESTART      = cfg["restart"]
DO_CMR       = cfg["ComMotionRemover"]
CMR_FREQ     = cfg["CMMR_frequency"]

pi = np.pi


def A_to_nm(x):
    return x * 0.1

print("timestep = %.4f ps, friction = %.1f /ps, temperature = %.1f K" % (TIMESTEP_PS, FRICTION, TEMPERATURE))
# ─────────────────────────────────────────────────────────────────────
# LOAD DNA CG STRUCTURE (fixed-column PDB parse -- same fix applied to
# set_radius_eps_mass_charge_dna.py; app.PDBFile / np.loadtxt both choke
# on AA2TIS_dna.py's output, so we parse ATOM/HETATM lines by column.
# ─────────────────────────────────────────────────────────────────────
print("Loading DNA CG structure ...")
beads = []
with open(pdb_file) as f:
    for line in f:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        beads.append({
            "name"   : line[12:16].strip(),   # P / S / A,T,G,C
            "resname": line[17:20].strip(),
            "chain"  : line[21],
            "resid"  : int(line[22:26]),
            "x"      : float(line[30:38]),
            "y"      : float(line[38:46]),
            "z"      : float(line[46:54]),
        })

N_dna = len(beads)
print(f"  {N_dna} DNA CG beads loaded, chains: {sorted(set(b['chain'] for b in beads))}")

# ─────────────────────────────────────────────────────────────────────
# LOAD TOPOLOGY FILES (from build_dna_topology.py)
# ─────────────────────────────────────────────────────────────────────
def _load_columns(path, skip_last_as_str=False):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            rows.append(parts[:-1] if skip_last_as_str else parts)
    return rows

bond_rows  = _load_columns(cfg["dna_bonds_file"])
angle_rows = _load_columns(cfg["dna_angles_file"])
stack_rows = _load_columns(cfg["dna_stacking_file"], skip_last_as_str=True)
hbond_rows = _load_columns(cfg["dna_hbonds_file"], skip_last_as_str=True)
charge_rows = _load_columns(cfg["dna_charges_file"])

mass_dna = np.loadtxt(cfg["dna_mass_file"], usecols=(3,), unpack=True, skiprows=1, delimiter=',')
print("N_dna =", N_dna)
print("mass file =", cfg["dna_mass_file"])
print("len(mass_dna) =", len(mass_dna))

print(f"  Bonds={len(bond_rows)} Angles={len(angle_rows)} Stacks={len(stack_rows)} HBonds={len(hbond_rows)}")

bond_pairs = [(int(r[0]), int(r[1])) for r in bond_rows]
angle_triples = [(int(r[0]), int(r[2])) for r in angle_rows]
phosphate_indices = set(int(r[0]) for r in charge_rows)
net_dna_charge = -1 * len(phosphate_indices)   # bare charge, explicit_ions mode
print(f"  Net DNA charge (bare): {net_dna_charge:+d} e  ({len(phosphate_indices)} phosphates)")

# ─────────────────────────────────────────────────────────────────────
# BUILD OpenMM TOPOLOGY: DNA chains first, ions appended after
# ─────────────────────────────────────────────────────────────────────
print("Building topology ...")
topology = Topology()
omm_chains = {}
for ch_id in dna_chains_cfg:
    omm_chains[ch_id] = topology.addChain(id=ch_id)

omm_atoms_dna = []
for b in beads:
    res = topology.addResidue(b["resname"], omm_chains[b["chain"]])
    elem = {"P": "P", "S": "C"}.get(b["name"], "C")
    atom = topology.addAtom(b["name"], Element.getBySymbol(elem), res)
    omm_atoms_dna.append(atom)

# ─────────────────────────────────────────────────────────────────────
# COMPUTE BOX SIZE (same philosophy/formula as sop_hpc_openmm.py):
# box = max(DNA span x box_scale_factor, 2 x nonlocal_cutoff)
# ─────────────────────────────────────────────────────────────────────
coords_nm = np.array([[b["x"], b["y"], b["z"]] for b in beads]) * 0.1  # A -> nm

dna_min, dna_max = coords_nm.min(axis=0), coords_nm.max(axis=0)
dna_span = dna_max - dna_min

min_box = 2.0 * NONLOCAL_CUTOFF * 0.1   # A -> nm
padding_nm = BOX_PADDING * 0.1

# box_size = np.maximum(dna_span * BOX_SCALE, min_box * np.ones(3))
box_size = np.array([60.0, 60.0, 60.0])  # nm

centre = 0.5 * box_size
dna_centre = 0.5 * (dna_min + dna_max)
coords_nm = coords_nm - dna_centre + centre   # centre DNA in box

topology.setPeriodicBoxVectors((
    Vec3(box_size[0], 0, 0) * unit.nanometer,
    Vec3(0, box_size[1], 0) * unit.nanometer,
    Vec3(0, 0, box_size[2]) * unit.nanometer,
))

# ─────────────────────────────────────────────────────────────────────
# COMPUTE ION COUNTS (auto, same formula as sop_hpc_openmm.py) or read
# fixed nK/nCl/nMg from config if auto_ions == False.
# ─────────────────────────────────────────────────────────────────────
if electrostatics_mode != 'explicit_ions':
    n_k = n_cl = n_mg = 0
    print("electrostatics_mode = implicit_debye_huckel -> no explicit ions added.")
elif auto_ions:
    N_A = 6.022e23
    box_vol_nm3 = float(np.prod(box_size))
    box_vol_L = box_vol_nm3 * 1e-27 * 1e3
    n_kcl_pairs = max(0, round(KCl_mM * 1e-3 * N_A * box_vol_L))
    n_mgcl2_pairs = max(0, round(MgCl2_mM * 1e-3 * N_A * box_vol_L))

    n_k_salt = n_kcl_pairs
    n_mg_salt = n_mgcl2_pairs
    n_cl_salt = n_kcl_pairs + 2 * n_mgcl2_pairs

    # DNA is (essentially always) net negative -> neutralize with K+
    if neutralize:
        if net_dna_charge < 0:
            n_k = n_k_salt + abs(net_dna_charge)
            n_cl = n_cl_salt
        else:
            n_k = n_k_salt
            n_cl = n_cl_salt + net_dna_charge
    else:
        n_k, n_cl = n_k_salt, n_cl_salt
    n_mg = n_mg_salt
else:
    n_k, n_cl, n_mg = cfg.get("nK", 0), cfg.get("nCl", 0), cfg.get("nMg", 0)

N_ions = n_k + n_cl + n_mg
print(f"  DNA span:  {dna_span[0]:.2f} x {dna_span[1]:.2f} x {dna_span[2]:.2f} nm")
print(f"  Box size:  {box_size[0]:.2f} x {box_size[1]:.2f} x {box_size[2]:.2f} nm")
print(f"  Ions:      {n_k} K+ + {n_cl} Cl- + {n_mg} Mg2+ = {N_ions} total")

ion_chain = topology.addChain(id="Z")
ion_atoms = []
for _ in range(n_k):
    res = topology.addResidue("K", ion_chain)
    ion_atoms.append(("K", topology.addAtom("K", Element.getBySymbol("K"), res)))
for _ in range(n_cl):
    res = topology.addResidue("CL", ion_chain)
    ion_atoms.append(("Cl", topology.addAtom("CL", Element.getBySymbol("Cl"), res)))
for _ in range(n_mg):
    res = topology.addResidue("MG", ion_chain)
    ion_atoms.append(("Mg", topology.addAtom("MG", Element.getBySymbol("Mg"), res)))

nbeads_dna = N_dna
nbeads = N_dna + N_ions
dna_indices = list(range(nbeads_dna))
ion_species_ranges = {}
if N_ions:
    idx = nbeads_dna
    for species, count in [('K', n_k), ('Cl', n_cl), ('Mg', n_mg)]:
        ion_species_ranges[species] = (idx, idx + count)
        idx += count
ion_indices = list(range(nbeads_dna, nbeads))

# ─────────────────────────────────────────────────────────────────────
# PLACE IONS (random, avoiding clashes with DNA beads) -- same approach
# as sop_hpc_openmm.py.
# ─────────────────────────────────────────────────────────────────────
print("Placing ions ...")
np.random.seed(42)
ion_positions = []
# for _ in range(N_ions):
#     for attempt in range(50000):
#         pos = np.random.rand(3) * box_size
#         dists = np.linalg.norm(coords_nm - pos, axis=1)
#         if dists.min() > ION_MIN_DIST:
#             ion_positions.append(pos)
#             break
#     else:
#         ion_positions.append(np.random.rand(3) * box_size)

# all_positions = np.vstack([coords_nm, np.array(ion_positions)]) if N_ions else coords_nm
# positions_openmm = [Vec3(*p) * unit.nanometer for p in all_positions]

# print(f"  Total particles: {nbeads}")

# A more careful placement of ions, avoiding clashes with DNA and other ions
for _ in range(N_ions):
    placed = False
    for attempt in range(50000):
        pos = np.random.rand(3) * box_size
        ok = True
        dna_dists = np.linalg.norm(coords_nm - pos, axis=1)

        if dna_dists.min() <= ION_MIN_DIST:
            ok = False
        if ok and len(ion_positions):
            ion_dists = np.linalg.norm(
                np.array(ion_positions) - pos,
                axis=1
            )
            if ion_dists.min() <= ION_MIN_DIST:
                ok = False
        if ok:
            ion_positions.append(pos)
            placed = True
            break
    if not placed:
        ion_positions.append(np.random.rand(3) * box_size)

# all_positions = np.vstack([coords_nm, np.array(ion_positions)]) if N_ions else coords_nm
# positions_openmm = [Vec3(*p) * unit.nanometer for p in all_positions]

all_positions = np.vstack([coords_nm, np.array(ion_positions)]) if N_ions else coords_nm

positions_openmm = unit.Quantity(
    [Vec3(*p) for p in all_positions],
    unit.nanometer
)

print("Placed ions =", len(ion_positions))
print("Expected ions =", N_ions)
print(f"  Total particles: {nbeads}")

# A quick check of the minimum ion-ion distance, to see if any are unreasonably close
ions = np.array(ion_positions)
mind = 1e9
for i in range(len(ions)):
    for j in range(i+1,len(ions)):
        d = np.linalg.norm(ions[i]-ions[j])
        mind = min(mind,d)
print("Minimum ion-ion distance =", mind)

# print("Writing initial DNA+ions structure...")

# with open(initial_pdb, "w") as f:
#     PDBFile.writeFile(
#         topology,
#         positions_openmm,
#         f
#     )

##*************************************************##
############ Building OpenMM simulation #############
print('\n       Constructing OpenMM System')
print("========================================")
system = System()
system.setDefaultPeriodicBoxVectors(
    Vec3(box_size[0], 0, 0) * unit.nanometer,
    Vec3(0, box_size[1], 0) * unit.nanometer,
    Vec3(0, 0, box_size[2]) * unit.nanometer,
)

for i in range(nbeads_dna):
    system.addParticle(float(mass_dna[i]) * unit.amu)
ion_mass = {'K': MASS_K, 'Cl': MASS_CL, 'Mg': MASS_MG}
for species, atom in ion_atoms:
    system.addParticle(ion_mass[species] * unit.amu)

#*****************************************************************#
###### BONDS (eq 2) ######
#*****************************************************************#
HarmonicBondForce = HarmonicBondForce()
for row in bond_rows:
    i, j, r0, kr = int(row[0]), int(row[1]), float(row[2]), float(row[3])
    HarmonicBondForce.addBond(i, j, r0 * unit.angstrom, 2.0 * kr * unit.kilocalorie_per_mole / (unit.angstrom) ** 2)

#*****************************************************************#
###### ANGLES (eq 3) ######
#*****************************************************************#
HarmonicAngleForce = HarmonicAngleForce()
for row in angle_rows:
    i, j, k, a0, ka = int(row[0]), int(row[1]), int(row[2]), float(row[3]), float(row[4])
    HarmonicAngleForce.addAngle(i, j, k, a0 * unit.radians, 2.0 * ka * (unit.kilocalories_per_mole / (unit.radians) ** 2))

#*****************************************************************#
###### EXCLUDED VOLUME: uniform WCA (paper eq 8), DNA-DNA only ######
#*****************************************************************#
EVEnergyExpression = "select(step(D0-r), eps0*(((D0/r)^12) - 2.0*((D0/r)^6) + 1.0), 0);"
EVForce = CustomNonbondedForce(EVEnergyExpression)
EVForce.addGlobalParameter('D0', D0 * unit.angstrom)
EVForce.addGlobalParameter('eps0', EPS0 * unit.kilocalorie_per_mole)
EVForce.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
EVForce.setCutoffDistance(float(D0) * unit.angstrom)
for i in range(nbeads):
    EVForce.addParticle([])
if N_ions:
    EVForce.addInteractionGroup(set(dna_indices), set(dna_indices))
for (i, j) in bond_pairs:
    EVForce.addExclusion(i, j)
for (i, k) in angle_triples:
    try:
        EVForce.addExclusion(i, k)
    except OpenMMException:
        pass

#*****************************************************************#
###### ION-ION / DNA-ION STERICS (explicit_ions mode) ######
# Combining-rule repulsive WCA: sigma_ij = r_i+r_j, eps_ij = sqrt(eps_i*eps_j)
#*****************************************************************#
if N_ions:
    ion_radius = {'K': SIGMA_K, 'Cl': SIGMA_CL, 'Mg': SIGMA_MG}
    ion_eps    = {'K': EPS_K, 'Cl': EPS_CL, 'Mg': EPS_MG}

    MixExpr = ("select(step(sigma_ij-r), 4.0*eps_ij*(((sigma_ij/r)^12)-((sigma_ij/r)^6)) + eps_ij, 0);"
               "sigma_ij = radius1+radius2; eps_ij = sqrt(epsval1*epsval2);")

    LJForce_ion_ion = CustomNonbondedForce(MixExpr)
    LJForce_ion_ion.addPerParticleParameter('radius')
    LJForce_ion_ion.addPerParticleParameter('epsval')
    LJForce_ion_ion.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    LJForce_ion_ion.setCutoffDistance(2.0 * max(SIGMA_K, SIGMA_CL, SIGMA_MG) * unit.angstrom)
    for i in range(nbeads):
        if i in dna_indices:
            LJForce_ion_ion.addParticle([0.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole])
        else:
            for species, (lo, hi) in ion_species_ranges.items():
                if lo <= i < hi:
                    LJForce_ion_ion.addParticle([ion_radius[species] * unit.angstrom, ion_eps[species] * unit.kilocalorie_per_mole])
                    break
    LJForce_ion_ion.addInteractionGroup(set(ion_indices), set(ion_indices))
    
    for (i, j) in bond_pairs:
        LJForce_ion_ion.addExclusion(i, j)
    for (i, k) in angle_triples:
        try:
            LJForce_ion_ion.addExclusion(i, k)
        except OpenMMException:
            pass

    LJForce_dna_ion = CustomNonbondedForce(MixExpr)
    LJForce_dna_ion.addPerParticleParameter('radius')
    LJForce_dna_ion.addPerParticleParameter('epsval')
    LJForce_dna_ion.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    LJForce_dna_ion.setCutoffDistance((D0 / 2.0 + max(SIGMA_K, SIGMA_CL, SIGMA_MG)) * unit.angstrom)
    for i in range(nbeads):
        if i in dna_indices:
            LJForce_dna_ion.addParticle([D0 / 2.0 * unit.angstrom, EPS0 * unit.kilocalorie_per_mole])
        else:
            for species, (lo, hi) in ion_species_ranges.items():
                if lo <= i < hi:
                    LJForce_dna_ion.addParticle([ion_radius[species] * unit.angstrom, ion_eps[species] * unit.kilocalorie_per_mole])
                    break
    LJForce_dna_ion.addInteractionGroup(set(dna_indices), set(ion_indices))
    for (i, j) in bond_pairs:
        LJForce_dna_ion.addExclusion(i, j)

    for (i, k) in angle_triples:
        try:
            LJForce_dna_ion.addExclusion(i, k)
        except OpenMMException:
            pass

#*****************************************************************#
#### STACKING INTERACTION (paper eq 9) ####
#*****************************************************************#
krST    = K_L / (unit.angstrom) ** 2
kphi1ST = K_PHI / (unit.radian) ** 2
kphi2ST = K_PHI / (unit.radian) ** 2

StackEnergyExpression  = "u0ST /(1.0 + kr*delRsq + kphi1*delphi1sq + kphi2*delphi2sq);"
StackEnergyExpression += "delRsq = delR*delR; delR = (RST-R0ST); RST = distance(p3,p6);"
StackEnergyExpression += "delphi1sq = delphi1*delphi1;"
StackEnergyExpression += "delphi1 = dphi1 - 2.0*pi*step(dphi1-pi) + 2.0*pi*step(-dphi1-pi);"
StackEnergyExpression += "dphi1 =(phi1ST-phi10ST); phi1ST= dihedral(p1,p2,p4,p5);"
StackEnergyExpression += "delphi2sq = delphi2*delphi2;"
StackEnergyExpression += "delphi2 = dphi2 - 2.0*pi*step(dphi2-pi) + 2.0*pi*step(-dphi2-pi);"
StackEnergyExpression += "dphi2 =(phi2ST-phi20ST); phi2ST= dihedral(p2,p4,p5,p7);"

StackForce = CustomCompoundBondForce(7, StackEnergyExpression)
StackForce.addPerBondParameter("u0ST")
StackForce.addPerBondParameter("R0ST")
StackForce.addPerBondParameter("phi10ST")
StackForce.addPerBondParameter("phi20ST")
StackForce.addGlobalParameter("kr", krST)
StackForce.addGlobalParameter("kphi1", kphi1ST)
StackForce.addGlobalParameter("kphi2", kphi2ST)
StackForce.addGlobalParameter("pi", pi)

for row in stack_rows:
    p_i, s_i, b_i, p_j, s_j, b_j, p_k = [int(x) for x in row[0:7]]
    l0, phi1_0, phi2_0 = float(row[7]), float(row[8]), float(row[9])
    h, s, Tm, dG0 = float(row[10]), float(row[11]), float(row[12]), float(row[13])
    u0ST_val = -h + float(Boltz_Const) * (TEMPERATURE - Tm) * s
    StackForce.addBond([p_i, s_i, b_i, p_j, s_j, b_j, p_k],
                        [u0ST_val * unit.kilocalorie_per_mole, l0 * unit.angstroms,
                         phi1_0 * unit.radian, phi2_0 * unit.radian])

#*****************************************************************#
#### WATSON-CRICK HYDROGEN-BOND POTENTIAL (paper eq 13) ####
#*****************************************************************#
kdHB     = K_D / (unit.angstrom) ** 2
kthetaHB = K_THETA / (unit.radian) ** 2
kpsiHB   = K_PSI / (unit.radian) ** 2

HBEnergyExpression  = "u0HB_bond / (1.0 + kd_hb*delDsq + ktheta_hb*deltheta1sq + ktheta_hb*deltheta2sq"
HBEnergyExpression += " + kpsi_hb*delpsi1sq + kpsi_hb*delpsi2sq + kpsi_hb*delpsi3sq);"
HBEnergyExpression += "delDsq = delD*delD; delD = (DHB - D0HB); DHB = distance(p3,p4);"
HBEnergyExpression += "deltheta1sq = deltheta1*deltheta1; deltheta1 = (theta1HB - theta10HB); theta1HB = angle(p2,p3,p4);"
HBEnergyExpression += "deltheta2sq = deltheta2*deltheta2; deltheta2 = (theta2HB - theta20HB); theta2HB = angle(p3,p4,p5);"
HBEnergyExpression += "delpsi1sq = delpsi1*delpsi1;"
HBEnergyExpression += "delpsi1 = dpsi1 - 2.0*pi*step(dpsi1-pi) + 2.0*pi*step(-dpsi1-pi);"
HBEnergyExpression += "dpsi1 = (psi1HB - psi10HB); psi1HB = dihedral(p2,p3,p4,p5);"
HBEnergyExpression += "delpsi2sq = delpsi2*delpsi2;"
HBEnergyExpression += "delpsi2 = dpsi2 - 2.0*pi*step(dpsi2-pi) + 2.0*pi*step(-dpsi2-pi);"
HBEnergyExpression += "dpsi2 = (psi2HB - psi20HB); psi2HB = dihedral(p6,p5,p4,p3);"
HBEnergyExpression += "delpsi3sq = delpsi3*delpsi3;"
HBEnergyExpression += "delpsi3 = dpsi3 - 2.0*pi*step(dpsi3-pi) + 2.0*pi*step(-dpsi3-pi);"
HBEnergyExpression += "dpsi3 = (psi3HB - psi30HB); psi3HB = dihedral(p1,p2,p3,p4);"

HBForce_WC = CustomCompoundBondForce(6, HBEnergyExpression)
for pname in ["D0HB", "theta10HB", "theta20HB", "psi10HB", "psi20HB", "psi30HB", "u0HB_bond"]:
    HBForce_WC.addPerBondParameter(pname)
HBForce_WC.addGlobalParameter("kd_hb", kdHB)
HBForce_WC.addGlobalParameter("ktheta_hb", kthetaHB)
HBForce_WC.addGlobalParameter("kpsi_hb", kpsiHB)
HBForce_WC.addGlobalParameter("pi", pi)

n_hbonds_skipped = 0
for row in hbond_rows:
    s1, b1, b5, s5, p6, p2 = [int(x) for x in row[0:6]]
    d0, theta1_0, theta2_0, psi1_0, psi2_0, psi3_0 = [float(x) for x in row[6:12]]
    UHB0_bond = float(row[12])
    if p6 == -1 or p2 == -1:
        n_hbonds_skipped += 1
        continue
    HBForce_WC.addBond([p2, s1, b1, b5, s5, p6],
                        [d0 * unit.angstrom, theta1_0 * unit.radian, theta2_0 * unit.radian,
                         psi1_0 * unit.radian, psi2_0 * unit.radian, psi3_0 * unit.radian,
                         UHB0_bond * unit.kilocalorie_per_mole])
if n_hbonds_skipped:
    print("WARNING: skipped %d WC pair(s) at chain ends lacking a flanking phosphate" % n_hbonds_skipped)

#*****************************************************************#
###### ELECTROSTATICS ######
#*****************************************************************#
if electrostatics_mode == 'explicit_ions':
    print("Electrostatics: PME with explicit ions (bare DNA phosphate charge -1e)")
    ESForce = NonbondedForce()
    ESForce.setNonbondedMethod(NonbondedForce.PME)
    ESForce.setCutoffDistance(PME_CUTOFF * unit.nanometer)
    ESForce.setEwaldErrorTolerance(0.0005)

    tcent = TEMPERATURE - 273.15
    dielectric = (
        87.740
        - 0.4008*tcent
        + 9.398e-4*tcent*tcent
        - 1.410e-6*tcent*tcent*tcent
    )

    print("dielectric =", dielectric)
    print("sqrt(dielectric) =", np.sqrt(dielectric))

    for i in range(nbeads):
        if i in dna_indices:
            q = -1.0 if i in phosphate_indices else 0.0
        else:
            for species, (lo, hi) in ion_species_ranges.items():
                if lo <= i < hi:
                    q = {'K': Q_K, 'Cl': Q_CL, 'Mg': Q_MG}[species]
                    break
        ESForce.addParticle(q/np.sqrt(dielectric), 1.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole)

    for (i, j) in bond_pairs:
        ESForce.addException(i, j, 0.0, 1.0, 0.0 * unit.kilocalorie_per_mole)
    for (i, k) in angle_triples:
        try:
            ESForce.addException(i, k, 0.0, 1.0, 0.0 * unit.kilocalorie_per_mole)
        except OpenMMException:
            pass
else:
    print("Electrostatics: implicit Debye-Huckel (paper eq 14-17, monovalent only)")
    tcent = TEMPERATURE - 273.15
    dielectric = 87.740 - (0.4008 * tcent) + (9.398e-4 * tcent ** 2) - (1.410e-6 * tcent ** 3)
    N_Avogadro = 6.02214076e23
    ion_conc_per_A3 = (monovalent_mM / 1000.0) * N_Avogadro / 1.0e27
    e_charge_sq_kcalA = 332.0637090
    lambdaD_sq_inv = (4.0 * pi * e_charge_sq_kcalA / dielectric) * (2.0 * ion_conc_per_A3) / (Boltz_Const * TEMPERATURE)
    lambdaD = (1.0 / lambdaD_sq_inv) ** 0.5
    print("Debye length lambda_D = %.3f Angstrom at %.1f mM monovalent salt" % (lambdaD, monovalent_mM))
    charge_map = {int(r[0]): float(r[1]) for r in charge_rows}

    ESForce = CustomNonbondedForce("(Q1*Q2)*(coulomb/eps_diel)*exp(-r/lambdaD)/r;")
    ESForce.addPerParticleParameter('Q')
    ESForce.addGlobalParameter('coulomb', e_charge_sq_kcalA * unit.kilocalorie_per_mole * unit.angstrom)
    ESForce.addGlobalParameter('eps_diel', dielectric)
    ESForce.addGlobalParameter('lambdaD', lambdaD * unit.angstrom)
    ESForce.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    ESForce.setCutoffDistance(lambdaD * 3.0 * unit.angstrom)
    for i in range(nbeads):
        ESForce.addParticle([charge_map.get(i, 0.0)])
    for (i, j) in bond_pairs:
        ESForce.addExclusion(i, j)
    for (i, k) in angle_triples:
        try:
            ESForce.addExclusion(i, k)
        except OpenMMException:
            pass

#*****************************************************************#
system.addForce(HarmonicBondForce)
system.addForce(HarmonicAngleForce)
system.addForce(EVForce)
system.addForce(StackForce)
system.addForce(HBForce_WC)
system.addForce(ESForce)
if N_ions:
    system.addForce(LJForce_ion_ion)
    system.addForce(LJForce_dna_ion)
if DO_CMR:
    system.addForce(CMMotionRemover(CMR_FREQ))

for i in range(system.getNumForces()):
    system.getForce(i).setForceGroup(i)

fe = ["HarmonicBondForce", "HarmonicAngleForce", "EVForce_dna_dna", "StackForce", "HBForce_WC",
      "ESForce_" + ("PME_explicit_ions" if electrostatics_mode == 'explicit_ions' else "DebyeHuckel_implicit")]
if N_ions:
    fe += ["LJForce_ion_ion", "LJForce_dna_ion"]
# fe = ["ESForce"]
if DO_CMR:
    fe.append("CMMotionRemover")

print("EV exclusions =", EVForce.getNumExclusions())
print("ES exceptions =", ESForce.getNumExceptions())
print("Has LJForce_ion_ion =", "LJForce_ion_ion" in globals())
print("Has LJForce_dna_ion =", "LJForce_dna_ion" in globals())

print("EV exclusions =", EVForce.getNumExclusions())
print("ES exceptions =", ESForce.getNumExceptions())
print("LJ ion-ion exclusions =", LJForce_ion_ion.getNumExclusions())
print("LJ dna-ion exclusions =", LJForce_dna_ion.getNumExclusions())

#─────────────────────────────────────────────────────────────────────
# REPORTER: Energy breakdown by force group
#─────────────────────────────────────────────────────────────────────
class EnergyBreakdownReporter:
    def __init__(self, file, reportInterval, forceNames):
        self._out = open(file, "w")
        self._reportInterval = reportInterval
        self._forceNames = forceNames

        header = "Step"
        for name in forceNames:
            header += f",{name}"
        self._out.write(header + "\n")

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, True, None)

    def report(self, simulation, state):

        step = simulation.currentStep

        values = [str(step)]

        print("\n=== Energy Breakdown @ step %d ===" % step)

        for i, name in enumerate(self._forceNames):

            e = simulation.context.getState(
                getEnergy=True,
                groups={i}
            ).getPotentialEnergy()

            ekj = e.value_in_unit(unit.kilojoule_per_mole)

            values.append(f"{ekj:.6f}")

            print(f"{name:30s} : {ekj:15.6f} kJ/mol")

        self._out.write(",".join(values) + "\n")
        self._out.flush()

    def __del__(self):
        try:
            self._out.close()
        except:
            pass

# --------------------------------------------------
# Save unique checkpoints
# --------------------------------------------------

class MultiCheckpointReporter:

    def __init__(self, prefix, interval):
        self.prefix = prefix
        self.interval = interval

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        fname = f"{self.prefix}_{simulation.currentStep}.chk"
        simulation.saveCheckpoint(fname)
        print(f"[CHECKPOINT] Saved {fname}")


# --------------------------------------------------
# Save positions + velocities
# --------------------------------------------------

class StateSaverReporter:

    def __init__(self, prefix, interval):
        self.prefix = prefix
        self.interval = interval

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):

        state = simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True
        )
        fname = f"{self.prefix}_{simulation.currentStep}.xml"

        with open(fname, "w") as f:
            f.write(XmlSerializer.serialize(state))

        print(f"[STATE] Saved {fname}")


# --------------------------------------------------
# Record largest force in system
# --------------------------------------------------

class MaxForceReporter:
    def __init__(self, filename, interval):
        self.interval = interval
        self.out = open(filename, "w")

        self.out.write(
            "Step,Particle,MaxForce\n"
        )

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):

        state = simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        fmag = np.sqrt(np.sum(forces._value**2, axis=1))
        imax = np.argmax(fmag)
        self.out.write(f"{simulation.currentStep},{imax},{fmag[imax]}\n")
        self.out.flush()


# ─────────────────────────────────────────────────────────────────────
# INTEGRATOR + SIMULATION
# ─────────────────────────────────────────────────────────────────────
integrator = LangevinMiddleIntegrator(TEMPERATURE * unit.kelvin, FRICTION / unit.picosecond, TIMESTEP_PS * unit.picoseconds)
integrator.setRandomNumberSeed(69)
platform = Platform.getPlatformByName(PLATFORM)
simulation = Simulation(topology, system, integrator, platform)

if RESTART:
    print(f"Restarting from checkpoint: {chk_equil}")
    simulation.loadCheckpoint(chk_equil)
else:
    simulation.context.setPositions(positions_openmm)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE * unit.kelvin, 69)

if DO_MINIMISE and not RESTART:
    print("Energy minimisation ...")
    simulation.minimizeEnergy(maxIterations=MIN_MAXITER, tolerance=MIN_TOL)
    print("Energy after minimisation:",
          simulation.context.getState(getEnergy=True).getPotentialEnergy())
    
# A quick check of the forces on each particle, to see if any are unreasonably large
state = simulation.context.getState(getForces=True)
forces = state.getForces(asNumpy=True)
fmag = np.sqrt(np.sum(forces._value**2, axis=1))
imax = np.argmax(fmag)
print("Largest force particle =", imax)
print("Largest force magnitude =", fmag[imax])
print("Mean force =", np.mean(fmag))
print("95 percentile =", np.percentile(fmag,95))

# A quick check of the forces on each particle, to see if any are unreasonably large, broken down by force group
for i,name in enumerate(fe):

    state = simulation.context.getState(
        getForces=True,
        groups={i}
    )

    forces = state.getForces(asNumpy=True)

    fmag = np.sqrt(np.sum(forces._value**2,axis=1))

    imax = np.argmax(fmag)

    print(name)
    print("   max particle =",imax)
    print("   max force =",fmag[imax])

os.makedirs(os.path.dirname(dcd_out) if os.path.dirname(dcd_out) else ".", exist_ok=True)
os.makedirs(CHK_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
simulation.reporters.append(DCDReporter(dcd_out, REPORT_EVERY))
simulation.reporters.append(StateDataReporter(
    log_out, REPORT_EVERY, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
    temperature=True, progress=True, remainingTime=True, speed=True,
    totalSteps=N_STEPS_EQUIL + N_STEPS_PROD, separator=","))
simulation.reporters.append(StateDataReporter(
    sys.stdout, 5000, step=True, temperature=True, potentialEnergy=True,
    speed=True, progress=True, remainingTime=True,
    totalSteps=N_STEPS_EQUIL + N_STEPS_PROD, separator=" | "))
simulation.reporters.append(
    EnergyBreakdownReporter(
        energy_breakdown,
        100000,      # every 100k steps
        fe
    )
)

# checkpoint every 100k Steps
simulation.reporters.append(MultiCheckpointReporter(os.path.join(CHK_DIR, "chk"), 100000))

# full state every 10k steps
simulation.reporters.append(StateSaverReporter(os.path.join(STATE_DIR, "state"), 100000))

# largest force every 10k steps
simulation.reporters.append(MaxForceReporter(os.path.join(STATE_DIR, "maxforce.csv"), 100000))

print("\nEnergy components before dynamics")
for i, name in enumerate(fe):
    e = simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
    print(f"  {name}: {e}")

print(f"\nRunning equilibration ({N_STEPS_EQUIL} steps) ...")
simulation.step(N_STEPS_EQUIL)
simulation.saveCheckpoint(chk_equil)

print(f"\nRunning production ({N_STEPS_PROD} steps) ...")
simulation.step(N_STEPS_PROD)
simulation.saveCheckpoint(chk_prod)



state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
simulation.topology.setPeriodicBoxVectors(None)

with open(final_pdb, "w") as f:
    PDBFile.writeFile(
        simulation.topology,
        state.getPositions(),
        f
    )

print("\nSimulation complete.")
print(f"  Trajectory: {dcd_out}")
print(f"  Log: {log_out}")
print(f"  Final structure: {final_pdb}")