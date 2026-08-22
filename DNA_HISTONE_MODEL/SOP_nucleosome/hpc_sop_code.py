"""
sop_hpc_openmm.py
==================
SOP (Self-Organised Polymer) OpenMM driver for the Histone Protein Core
(HPC) with histone tails, in explicit monovalent salt (K+ / Cl-).

E_total = E_FENE + E_EV + E_native + E_Coul

  (1) E_FENE   -- chain connectivity (bonds between consecutive beads)
  (2) E_EV     -- excluded volume, r^-6 repulsion (non-native, non-bonded pairs)
  (3) E_native -- native contacts, 12-6 LJ (attractive, crystal contacts)
  (4) E_Coul   -- explicit-ion Coulomb (protein-protein at eps_protein via a
                  dielectric correction on top of a uniform eps_water PME force)

Reddy & Thirumalai use a screened (Debye-Huckel) Coulomb term with the
ions integrated out; this driver instead places K+/Cl-/Mg2+ EXPLICITLY
and lets the screening emerge dynamically from ion positions.
"""

import json, sys, os
import numpy as np
from openmm import *
from openmm.app import *
import openmm.unit as unit
from openmm.app import PDBxFile, PDBFile

# ─────────────────────────────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────────────────────────────
cfg = json.load(open("input.json"))

pdb_file             = cfg["cg_pdb_with_tails"]
native_contacts_file = cfg["native_contacts_file"]
charged_beads_file   = cfg["charged_beads_file"]
CHAIN_ORDER          = cfg["histone_chains"]

dcd_out    = cfg["dcd_prefix"] + ".dcd"
log_out    = cfg["data_name"]
chk_equil  = cfg["checkpoint_name"]
chk_prod   = chk_equil.replace(".chk", "_prod_final.chk")
energy_breakdown = cfg.get("energy_breakdown", "hpc_energy_breakdown.csv")
final_pdb  = cfg["final_state_name"]

K_FENE          = cfg["fene_k"]           # kcal/(mol.A^2)
R0_FENE         = cfg["fene_R0"]          # A, max FENE extension
SIGMA_PROTEIN   = cfg["sigma_protein"]    # A
EPS_H_NATIVE    = cfg["eps_h_native"]     # kcal/mol, native-contact well depth
EPS_L_NONNATIVE = cfg["eps_local"]        # kcal/mol, EV prefactor

DIELECTRIC_PROTEIN = cfg["dielectric_protein"]   # protein interior
DIELECTRIC_WATER    = cfg["dielectric_water"]     # bulk water

auto_ions = cfg.get("auto_ions", True)
KCl_mM    = cfg["KCl_mM"]
MgCl2_mM  = cfg["MgCl2_mM"]
SIGMA_K, SIGMA_CL, SIGMA_MG = cfg["rK"], cfg["rCl"], cfg["rMg"]
EPS_K, EPS_CL, EPS_MG       = cfg["epsK"], cfg["epsCl"], cfg["epsMg"]

BOX_PADDING = cfg["box_padding"]
BOX_SCALE   = cfg["box_scale_factor"]
NONLOCAL_CUTOFF = cfg.get("nonlocal_cutoff", 30.0)
PME_CUTOFF  = cfg["pme_cutoff"] * 0.1   # Angstrom -> nm

TEMPERATURE   = cfg["Temp"]
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

BOX_MODE = cfg.get("box_mode", False)

BOX_X = cfg.get("box_x_nm", 20.0)
BOX_Y = cfg.get("box_y_nm", 20.0)
BOX_Z = cfg.get("box_z_nm", 20.0)

pi = np.pi


def A_to_nm(x):
    return x * 0.1


def kcal_to_kJ(x):
    return x * 4.184


print("timestep = %.4f ps, friction = %.1f /ps, temperature = %.1f K" % (TIMESTEP_PS, FRICTION, TEMPERATURE))

tcent = TEMPERATURE - 273.15
eps_water = 87.740 - 0.40008 * tcent + 9.398e-4 * tcent ** 2 - 1.410e-6 * tcent ** 3
dielectric_sqrt = np.sqrt(eps_water)
print("water dielectric = %.2f, sqrt(eps) = %.2f" % (eps_water, dielectric_sqrt))

# ─────────────────────────────────────────────────────────────────────
# LOAD PROTEIN CG STRUCTURE (fixed-column PDB parse)
# ─────────────────────────────────────────────────────────────────────
print("Loading protein CG structure ...")
beads = []
with open(pdb_file) as f:
    for line in f:
        if line[:6].strip() != "ATOM":
            continue
        beads.append({
            "resname": line[17:20].strip(),
            "chain"  : line[21],
            "resid"  : int(line[22:26]),
            "x"      : float(line[30:38]),
            "y"      : float(line[38:46]),
            "z"      : float(line[46:54]),
        })

N_protein = len(beads)
print(f"  {N_protein} protein CG beads loaded, chains: {sorted(set(b['chain'] for b in beads))}")

# ─────────────────────────────────────────────────────────────────────
# LOAD TOPOLOGY FILES (native contacts + charged beads) -- native
# contacts were built from the 761-bead core PDB (0-indexed); after
# prepending tail beads per chain the core indices shift, so each
# contact index is remapped chain-by-chain.
# ─────────────────────────────────────────────────────────────────────
def _load_columns(path, comment="#"):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(comment):
                continue
            rows.append(line.split())
    return rows

chain_order     = CHAIN_ORDER
old_chain_sizes = {'A': 97, 'B': 78, 'C': 108, 'D': 96, 'E': 99, 'F': 85, 'G': 104, 'H': 94}
tail_sizes      = {'A': 37, 'B': 24, 'C': 10, 'D': 26, 'E': 36, 'F': 17, 'G': 14, 'H': 27}

old_offset = 0
old_idx_to_chain_pos = {}
for ch in chain_order:
    for pos in range(old_chain_sizes[ch]):
        old_idx_to_chain_pos[old_offset + pos] = (ch, pos)
    old_offset += old_chain_sizes[ch]

new_chain_start = {}
offset = 0
for ch in chain_order:
    new_chain_start[ch] = offset
    offset += tail_sizes[ch] + old_chain_sizes[ch]

def remap_contact_index(old_idx):
    """Convert a 0-based index from native_contacts.dat to new bead index."""
    ch, pos_in_core = old_idx_to_chain_pos[old_idx]
    return new_chain_start[ch] + tail_sizes[ch] + pos_in_core

native_rows = _load_columns(native_contacts_file)
native_contacts = []
for row in native_rows:
    i_old, j_old, r0 = int(row[0]), int(row[1]), float(row[2])
    native_contacts.append((remap_contact_index(i_old), remap_contact_index(j_old), r0))

charge_rows = _load_columns(charged_beads_file)
charged_protein = {int(r[0]): float(r[4]) for r in charge_rows}

print(f"  NativeContacts={len(native_contacts)} ChargedBeads={len(charged_protein)}")

# ─────────────────────────────────────────────────────────────────────
# BUILD OpenMM TOPOLOGY: protein chains first, ions appended after
# ─────────────────────────────────────────────────────────────────────
print("Building topology ...")
topology = Topology()
omm_chains = {}
for ch_id in chain_order:
    omm_chains[ch_id] = topology.addChain(id=ch_id)

omm_atoms_protein = []
for b in beads:
    res = topology.addResidue(b["resname"], omm_chains[b["chain"]])
    atom = topology.addAtom("BB", Element.getBySymbol("C"), res)
    omm_atoms_protein.append(atom)

# ─────────────────────────────────────────────────────────────────────
# COMPUTE BOX SIZE: box = max(protein span x box_scale_factor,
# 2 x nonlocal_cutoff) -- same philosophy/formula as dnamd.py. There is
# no explicit solvent in the CG model, so the box is a minimum-image
# periodic cell around the protein, not a concentration-setting volume.
# ─────────────────────────────────────────────────────────────────────
# coords_nm = np.array([[b["x"], b["y"], b["z"]] for b in beads]) * 0.1   # A -> nm

# prot_min, prot_max = coords_nm.min(axis=0), coords_nm.max(axis=0)
# prot_span = prot_max - prot_min

# min_box = 2.0 * NONLOCAL_CUTOFF * 0.1   # A -> nm
# padding_nm = BOX_PADDING * 0.1

# box_size = np.maximum(prot_span * BOX_SCALE, min_box * np.ones(3))

# centre = 0.5 * box_size
# prot_centre = 0.5 * (prot_min + prot_max)
# coords_nm = coords_nm - prot_centre + centre   # centre protein in box

# topology.setPeriodicBoxVectors((
#     Vec3(box_size[0], 0, 0) * unit.nanometer,
#     Vec3(0, box_size[1], 0) * unit.nanometer,
#     Vec3(0, 0, box_size[2]) * unit.nanometer,
# ))

coords_nm = np.array([[b["x"], b["y"], b["z"]] for b in beads]) * 0.1
prot_min = coords_nm.min(axis=0)
prot_max = coords_nm.max(axis=0)
prot_span = prot_max - prot_min

# --------------------------------------------------
# MANUAL BOX
# --------------------------------------------------

if BOX_MODE == "manual":
    box_size = np.array([
        BOX_X,
        BOX_Y,
        BOX_Z
    ])
    print("\nUsing MANUAL box")
    print(
        f"Box size: "
        f"{BOX_X:.2f} x "
        f"{BOX_Y:.2f} x "
        f"{BOX_Z:.2f} nm"
    )

# --------------------------------------------------
# AUTOMATIC BOX (current behaviour)
# --------------------------------------------------

else:
    min_box = 2.0 * cfg.get("nonlocal_cutoff", 30.0) * 0.1

    box_size = np.maximum(
        prot_span * BOX_SCALE,
        np.ones(3) * min_box
    )
    print("\nUsing AUTO box")
    print(
        f"Protein span: "
        f"{prot_span[0]:.2f} x "
        f"{prot_span[1]:.2f} x "
        f"{prot_span[2]:.2f} nm"
    )

# --------------------------------------------------
# CENTER PROTEIN IN BOX
# --------------------------------------------------

centre = 0.5 * box_size
prot_centre = 0.5 * (prot_min + prot_max)

coords_nm = coords_nm - prot_centre + centre

# --------------------------------------------------
# PERIODIC BOX
# --------------------------------------------------

topology.setPeriodicBoxVectors((
    Vec3(box_size[0], 0, 0) * unit.nanometer,
    Vec3(0, box_size[1], 0) * unit.nanometer,
    Vec3(0, 0, box_size[2]) * unit.nanometer
))

# ─────────────────────────────────────────────────────────────────────
# COMPUTE ION COUNTS (auto, volume-based) or read fixed nK/nCl/nMg from
# config if auto_ions == False. Neutralisation ions are added on top of
# the salt-derived counts to make the system charge-neutral.
# ─────────────────────────────────────────────────────────────────────
net_protein_charge = int(round(sum(charged_protein.values())))

if auto_ions:
    N_A = 6.022e23
    box_vol_nm3 = float(np.prod(box_size))
    box_vol_L = box_vol_nm3 * 1e-27 * 1e3
    n_kcl_pairs = max(0, round(KCl_mM * 1e-3 * N_A * box_vol_L))
    n_mgcl2_pairs = max(0, round(MgCl2_mM * 1e-3 * N_A * box_vol_L))

    if net_protein_charge > 0:
        n_k = n_kcl_pairs + cfg.get("nK", 0)
        n_cl = (n_kcl_pairs + 2 * n_mgcl2_pairs) + cfg.get("nCl", 0) + net_protein_charge
    else:
        n_k = n_kcl_pairs + cfg.get("nK", 0) + abs(net_protein_charge)
        n_cl = (n_kcl_pairs + 2 * n_mgcl2_pairs) + cfg.get("nCl", 0)
    n_mg = n_mgcl2_pairs
else:
    n_k, n_cl, n_mg = cfg.get("nK", 0), cfg.get("nCl", 0), cfg.get("nMg", 0)

N_ions = n_k + n_cl + n_mg
print(f"  Protein span: {prot_span[0]:.2f} x {prot_span[1]:.2f} x {prot_span[2]:.2f} nm")
print(f"  Box size:     {box_size[0]:.2f} x {box_size[1]:.2f} x {box_size[2]:.2f} nm")
print(f"  Net protein charge (bare): {net_protein_charge:+d} e")
print(f"  Ions: {n_k} K+ + {n_cl} Cl- + {n_mg} Mg2+ = {N_ions} total")

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

nbeads_protein = N_protein
nbeads = N_protein + N_ions
protein_indices = list(range(nbeads_protein))
ion_species_ranges = {}
if N_ions:
    idx = nbeads_protein
    for species, count in [('K', n_k), ('Cl', n_cl), ('Mg', n_mg)]:
        ion_species_ranges[species] = (idx, idx + count)
        idx += count
ion_indices = list(range(nbeads_protein, nbeads))

# ─────────────────────────────────────────────────────────────────────
# PLACE IONS (random, avoiding clashes with protein beads)
# ─────────────────────────────────────────────────────────────────────
print("Placing ions ...")
np.random.seed(42)
ion_positions = []
ION_MIN_DIST = 0.8   # nm, 8 A minimum from a protein bead

for _ in range(N_ions):
    placed = False
    for attempt in range(50000):
        pos = np.random.rand(3) * box_size
        dists = np.linalg.norm(coords_nm - pos, axis=1)
        if dists.min() > ION_MIN_DIST:
            ion_positions.append(pos)
            placed = True
            break
    if not placed:
        ion_positions.append(np.random.rand(3) * box_size)

all_positions = np.vstack([coords_nm, np.array(ion_positions)]) if N_ions else coords_nm
positions_openmm = unit.Quantity(
    [Vec3(*p) for p in all_positions],
    unit.nanometer
)

print("Placed ions =", len(ion_positions))
print("Expected ions =", N_ions)
print(f"  Total particles: {nbeads}")

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

# Unit mass for every bead (rescaled implicitly via the friction coefficient)
for i in range(nbeads):
    system.addParticle(1.0 * unit.amu)

#*****************************************************************#
###### FENE BONDS (chain connectivity) ######
# U_FENE(r) = -(k/2) R0^2 ln[1 - (r - r_cry)^2 / R0^2]
# r_cry = equilibrium (crystal) bond length taken from the CG structure;
# R0 = max extension beyond equilibrium; diverges as r-r_cry -> R0,
# preventing unphysical chain crossing (unlike a harmonic spring).
#*****************************************************************#
FENEForce = CustomBondForce("-0.5 * K * R0*R0 * log(1 - ((r - r0)/R0)^2)")
FENEForce.addPerBondParameter("r0")
FENEForce.addPerBondParameter("R0")
FENEForce.addPerBondParameter("K")
FENEForce.setUsesPeriodicBoundaryConditions(True)

chain_boundaries = []
offset = 0
for ch in chain_order:
    n_ch = tail_sizes[ch] + old_chain_sizes[ch]
    chain_boundaries.append((offset, offset + n_ch - 1))
    offset += n_ch

bond_pairs = set()
for ch_start, ch_end in chain_boundaries:
    for i in range(ch_start, ch_end):
        j = i + 1
        r0_bond = float(np.linalg.norm(all_positions[j] - all_positions[i]))   # nm
        FENEForce.addBond(i, j, [r0_bond, A_to_nm(R0_FENE), kcal_to_kJ(K_FENE) * 100.0])
        bond_pairs.add((min(i, j), max(i, j)))

print(f"  FENE bonds = {FENEForce.getNumBonds()}")

#*****************************************************************#
###### EXCLUDED VOLUME (non-native, r^-6 repulsion) ######
# U_EV(r) = eps_l * (sigma/r)^6, sigma = (sigma_i + sigma_j)/2
# Covers protein-protein, protein-ion and ion-ion pairs; native
# contacts and near-sequence pairs (seq_sep < 3) are excluded here
# since they carry their own bonded/attractive terms.
#*****************************************************************#
EVForce = CustomNonbondedForce("eps_l * (sigma / r)^6 ; sigma = 0.5*(sigma1 + sigma2)")
EVForce.addGlobalParameter("eps_l", kcal_to_kJ(EPS_L_NONNATIVE))
EVForce.addPerParticleParameter("sigma")
EVForce.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
EVForce.setCutoffDistance(PME_CUTOFF * unit.nanometer)

for i in range(nbeads_protein):
    EVForce.addParticle([A_to_nm(SIGMA_PROTEIN)])
ion_sigma = {'K': SIGMA_K, 'Cl': SIGMA_CL, 'Mg': SIGMA_MG}
for species, atom in ion_atoms:
    EVForce.addParticle([A_to_nm(ion_sigma[species])])

native_contact_set = set((min(i, j), max(i, j)) for i, j, _ in native_contacts)
master_exclusions = set()
for i, j in bond_pairs:
    master_exclusions.add((min(i, j), max(i, j)))
for ch_start, ch_end in chain_boundaries:
    for i in range(ch_start, ch_end + 1):
        for j in range(i + 1, min(i + 3, ch_end + 1)):
            master_exclusions.add((i, j))
for i, j, _ in native_contacts:
    master_exclusions.add((min(i, j), max(i, j)))

for (i, j) in master_exclusions:
    EVForce.addExclusion(i, j)

#*****************************************************************#
###### NATIVE CONTACTS (12-6 Lennard-Jones) ######
# U_native(r) = eps_h * [(r_cry/r)^12 - 2(r_cry/r)^6]
# minimum at r = r_cry, well depth exactly -eps_h.
#*****************************************************************#
NativeForce = CustomBondForce("eps_h * ((r0/r)^12 - 2*(r0/r)^6)")
NativeForce.addPerBondParameter("r0")
NativeForce.addPerBondParameter("eps_h")
NativeForce.setUsesPeriodicBoundaryConditions(True)

eps_h_kJ = kcal_to_kJ(EPS_H_NATIVE)
for i, j, r0_A in native_contacts:
    NativeForce.addBond(i, j, [A_to_nm(r0_A), eps_h_kJ])

print(f"  Native contact pairs = {NativeForce.getNumBonds()}")

#*****************************************************************#
###### ELECTROSTATICS ######
# U_Coul(r) = q_i q_j e^2 / (4 pi eps0 eps r)   (bare Coulomb; screening
# emerges from the explicit ions, not from a Debye-Huckel pre-factor)
#
# Implementation (two forces, exactly as in the original driver):
#   ESForce      -- uniform PME NonbondedForce at eps = eps_water = 80,
#                    covering protein-protein, protein-ion, ion-ion.
#   ESCorrForce  -- CustomBondForce correcting protein-protein charged
#                    pairs down to eps = eps_protein = 10 (partially
#                    desolvated protein interior):
#                    dU_corr = q_i q_j * ONE_4PI_EPS0 * (1/eps_prot - 1/eps_water) / r
#*****************************************************************#
print("Electrostatics: PME at eps_water=%.1f + protein-protein correction to eps_protein=%.1f" %
      (DIELECTRIC_WATER, DIELECTRIC_PROTEIN))

ONE_4PI_EPS0 = 138.935458   # kJ.nm / (mol.e^2)

ESForce = NonbondedForce()
ESForce.setNonbondedMethod(NonbondedForce.PME)
ESForce.setCutoffDistance(PME_CUTOFF * unit.nanometer)
ESForce.setEwaldErrorTolerance(1e-4)
ESForce.setReactionFieldDielectric(DIELECTRIC_WATER)

for i in range(nbeads_protein):
    q = charged_protein.get(i, 0.0) / dielectric_sqrt
    ESForce.addParticle(q, A_to_nm(SIGMA_PROTEIN) * unit.nanometer, 0.0)
ion_charge = {'K': +1.0, 'Cl': -1.0, 'Mg': +2.0}
for species, atom in ion_atoms:
    ESForce.addParticle(ion_charge[species] / dielectric_sqrt, A_to_nm(ion_sigma[species]) * unit.nanometer, 0.0)

for (i, j) in master_exclusions:
    ESForce.addException(i, j, 0.0, 1.0, 0.0)

corr_factor = ONE_4PI_EPS0 * (1.0 / DIELECTRIC_PROTEIN - 1.0 / DIELECTRIC_WATER)
ESCorrForce = CustomBondForce(f"{corr_factor} * qi * qj / max(r, 0.1)")
ESCorrForce.addPerBondParameter("qi")
ESCorrForce.addPerBondParameter("qj")
ESCorrForce.setUsesPeriodicBoundaryConditions(True)

charged_indices = sorted(charged_protein.keys())
for ii in range(len(charged_indices)):
    for jj in range(ii + 1, len(charged_indices)):
        i_idx, j_idx = charged_indices[ii], charged_indices[jj]
        pair = (min(i_idx, j_idx), max(i_idx, j_idx))
        if pair in bond_pairs or pair in master_exclusions:
            continue
        qi = charged_protein[i_idx] / dielectric_sqrt
        qj = charged_protein[j_idx] / dielectric_sqrt
        ESCorrForce.addBond(i_idx, j_idx, [qi, qj])

print(f"  Protein-protein Coulomb correction pairs = {ESCorrForce.getNumBonds()}")

#*****************************************************************#
system.addForce(FENEForce)
system.addForce(EVForce)
system.addForce(NativeForce)
system.addForce(ESForce)
system.addForce(ESCorrForce)
if DO_CMR:
    system.addForce(CMMotionRemover(CMR_FREQ))

for i in range(system.getNumForces()):
    system.getForce(i).setForceGroup(i)

fe = ["FENEForce", "EVForce_protein_ion_all", "NativeForce",
      "ESForce_PME_eps_water", "ESForce_protein_protein_dielectric_correction"]

# fe = ["ESCorrForce"]
if DO_CMR:
    fe.append("CMMotionRemover")

print("EV exclusions =", EVForce.getNumExclusions())
print("ES exceptions =", ESForce.getNumExceptions())

# ─────────────────────────────────────────────────────────────────────
# REPORTER: Energy breakdown by force group
# ─────────────────────────────────────────────────────────────────────
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
            e = simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
            ekj = e.value_in_unit(unit.kilojoule_per_mole)
            values.append(f"{ekj:.6f}")
            print(f"{name:45s} : {ekj:15.6f} kJ/mol")

        self._out.write(",".join(values) + "\n")
        self._out.flush()

    def __del__(self):
        try:
            self._out.close()
        except:
            pass

# ─────────────────────────────────────────────────────────────────────
# INTEGRATOR + SIMULATION
# ─────────────────────────────────────────────────────────────────────
integrator = LangevinMiddleIntegrator(TEMPERATURE * unit.kelvin, FRICTION / unit.picosecond, TIMESTEP_PS * unit.picoseconds)
platform = Platform.getPlatformByName(PLATFORM)
simulation = Simulation(topology, system, integrator, platform)

if RESTART:
    print(f"Restarting from checkpoint: {chk_equil}")
    simulation.loadCheckpoint(chk_equil)
else:
    simulation.context.setPositions(positions_openmm)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE * unit.kelvin)

if DO_MINIMISE and not RESTART:
    print("Energy minimisation ...")
    simulation.minimizeEnergy(maxIterations=MIN_MAXITER, tolerance=MIN_TOL)
    print("Energy after minimisation:",
          simulation.context.getState(getEnergy=True).getPotentialEnergy())

# A quick check of the forces on each particle, to see if any are unreasonably large
state = simulation.context.getState(getForces=True)
forces = state.getForces(asNumpy=True)
fmag = np.sqrt(np.sum(forces._value ** 2, axis=1))
imax = np.argmax(fmag)
print("Largest force particle =", imax)
print("Largest force magnitude =", fmag[imax])
print("Mean force =", np.mean(fmag))
print("95 percentile =", np.percentile(fmag, 95))

# Same check broken down by force group
for i, name in enumerate(fe):
    state = simulation.context.getState(getForces=True, groups={i})
    forces = state.getForces(asNumpy=True)
    fmag = np.sqrt(np.sum(forces._value ** 2, axis=1))
    imax = np.argmax(fmag)
    print(name)
    print("   max particle =", imax)
    print("   max force =", fmag[imax])

os.makedirs(os.path.dirname(dcd_out) if os.path.dirname(dcd_out) else ".", exist_ok=True)
simulation.reporters.append(DCDReporter(dcd_out, REPORT_EVERY))
simulation.reporters.append(StateDataReporter(
    log_out, REPORT_EVERY, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
    temperature=True, progress=True, remainingTime=True, speed=True,
    totalSteps=N_STEPS_EQUIL + N_STEPS_PROD, separator=","))
simulation.reporters.append(StateDataReporter(
    sys.stdout, 5000, step=True, temperature=True, potentialEnergy=True,
    density=True, speed=True, progress=True, remainingTime=True,
    totalSteps=N_STEPS_EQUIL + N_STEPS_PROD, separator=" | "))
simulation.reporters.append(
    EnergyBreakdownReporter(
        energy_breakdown,
        100000,      # every 10k steps
        fe
    )
)

print("\nEnergy components before dynamics")
for i, name in enumerate(fe):
    e = simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
    print(f"  {name}: {e}")

print(f"\nRunning equilibration ({N_STEPS_EQUIL} steps = {N_STEPS_EQUIL * TIMESTEP_PS * 1e-3:.2f} ns) ...")
simulation.step(N_STEPS_EQUIL)
simulation.saveCheckpoint(chk_equil)

print(f"\nRunning production ({N_STEPS_PROD} steps = {N_STEPS_PROD * TIMESTEP_PS * 1e-3:.2f} ns) ...")
simulation.step(N_STEPS_PROD)
simulation.saveCheckpoint(chk_prod)

# Final structure is written as .cif (not .pdb): the periodic box for this
# system can exceed the fixed-width CRYST1 field in the legacy PDB format,
# so the box vectors are cleared and mmCIF is used instead -- unchanged
# from the original driver.
simulation.topology.setPeriodicBoxVectors(None)
state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)

final_cif = final_pdb.replace(".pdb", ".cif")
with open(final_cif, "w") as f:
    PDBxFile.writeFile(simulation.topology, state.getPositions(), f)

print("\nSimulation complete.")
print(f"  Trajectory: {dcd_out}")
print(f"  Log: {log_out}")
print(f"  Final structure: {final_cif}")