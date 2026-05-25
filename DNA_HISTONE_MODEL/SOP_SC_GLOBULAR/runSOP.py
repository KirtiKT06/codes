# This script sets up and runs a molecular dynamics simulation of a globular protein using the SOP-SC model with explicit ions for globular proteins.

# imports
from __future__ import print_function

import json
from datetime import datetime

import numpy as np
from sys import stdout

import openmm as mm
import openmm.app as app
import openmm.unit as u

from bt_matrix import BT_ORDER, BT_MATRIX

# Load native contacts from the CD PDB file.
start_time = datetime.now()

config = json.load(open("input.json"))

pdb_prefix = config["pdb_prefix"]
pdb_file = pdb_prefix + ".pdb"

dcd_prefix = config["dcd_prefix"]
data_name = config["data_name"]
energy_name = config.get("energy_name", "energy.dat")
checkpoint_name = config.get("checkpoint_name", "chkout.chk")

sequence = config["sequence"].replace(" ", "").replace("\n", "").upper()

# Simulation parameters
box_length_A = float(config["box_length"])
box_length = box_length_A * u.angstrom

Temp = float(config["Temp"])
T = Temp * u.kelvin

Boltz_Const = float(config["Boltz_Const"])
kBT_kcal = Boltz_Const * Temp

friction_coeff = float(config["friction_coeff"])
time_step = float(config["time_step"])

numsteps = int(config["numsteps"])
data_interval = int(config["data_interval"])
snap_interval = int(config["snap_interval"])

platform_type = config["platform_type"]

final_state = bool(config["final_state"])
minimization = bool(config["minimization"])
restart = bool(config["restart"])

ComMotionRemover = bool(config["ComMotionRemover"])
CMMR_frequency = int(config["CMMR_frequency"])

dielectric = float(config.get("dielectric", 78.0))

pme_cutoff = float(config.get("pme_cutoff", config["LJ_cutoff"])) * u.angstrom
ev_cutoff = float(config["LJ_cutoff"]) * u.angstrom

fene_k = float(config.get("fene_k", 20.0)) * u.kilocalorie_per_mole / (u.angstrom ** 2)
fene_R0 = float(config.get("fene_R0", 2.0)) * u.angstrom
eps_local = float(config.get("eps_local", 1.0)) * u.kilocalorie_per_mole
eps_protein_ion = float(config.get("eps_protein_ion", 0.2)) * u.kilocalorie_per_mole

omega = float(config.get("omega", 0.12))
bt_offset = float(config.get("bt_offset", 0.7))
nonlocal_cutoff = float(config.get("nonlocal_cutoff", 30.0)) * u.angstrom

histidine_charge = float(config.get("histidine_charge", 0.0))
mProtein = float(config.get("mProtein", 108.4))

# Native contact parameters
AA_PARAMS = {
    "BB": {"radius_A": 1.90, "charge": 0.0},
    "G": {"radius_A": 2.25, "charge": 0.0},
    "A": {"radius_A": 2.52, "charge": 0.0},
    "R": {"radius_A": 3.28, "charge": 1.0},
    "K": {"radius_A": 3.18, "charge": 1.0},
    "H": {"radius_A": 3.04, "charge": histidine_charge},
    "D": {"radius_A": 2.79, "charge": -1.0},
    "E": {"radius_A": 2.96, "charge": -1.0},
    "S": {"radius_A": 2.59, "charge": 0.0},
    "T": {"radius_A": 2.81, "charge": 0.0},
    "N": {"radius_A": 2.84, "charge": 0.0},
    "Q": {"radius_A": 3.01, "charge": 0.0},
    "C": {"radius_A": 2.74, "charge": 0.0},
    "P": {"radius_A": 2.78, "charge": 0.0},
    "I": {"radius_A": 3.09, "charge": 0.0},
    "L": {"radius_A": 3.09, "charge": 0.0},
    "M": {"radius_A": 3.09, "charge": 0.0},
    "F": {"radius_A": 3.18, "charge": 0.0},
    "W": {"radius_A": 3.39, "charge": 0.0},
    "Y": {"radius_A": 3.23, "charge": 0.0},
    "V": {"radius_A": 2.93, "charge": 0.0}
}

def ion_params_from_config():           #Extracts mass, charge, radius, and epsilon (van der Waals well depth) for K, Mg, and Cl ions.
    return {
        "K": {
            "radius_A": float(config["rK"]),
            "charge": float(config["qK"]),
            "mass_amu": float(config["mK"]),
            "epsilon_kcal": float(config["epsK"])
        },
        "Mg": {
            "radius_A": float(config["rMg"]),
            "charge": float(config["qMg"]),
            "mass_amu": float(config["mMg"]),
            "epsilon_kcal": float(config["epsMg"])
        },
        "Cl": {
            "radius_A": float(config["rCl"]),
            "charge": float(config["qCl"]),
            "mass_amu": float(config["mCl"]),
            "epsilon_kcal": float(config["epsCl"])
        }
    }

def build_beads_from_sequence(sequence):
    """
      Gly = one bead only, located at CA, using Gly radius.
      All non-Gly residues = BB + SC.
    """
    beads = []
    residue_to_beads = []

    for resid, aa in enumerate(sequence):
        if aa not in AA_PARAMS:
            raise ValueError("Unknown amino acid: %s" % aa)

        # Glycine: single bead only.
        # It is written as BB in the PDB, but in the force field we mark it as SINGLE
        # so it can use Gly radius/type rather than generic BB radius/type.
        if aa == "G":
            idx = len(beads)

            beads.append({
                "name": "BB",
                "resid": resid,
                "kind": "SINGLE",
                "aa": aa,
                "radius_A": AA_PARAMS["G"]["radius_A"],
                "charge": AA_PARAMS["G"]["charge"]
            })

            residue_to_beads.append([idx, None])
            continue

        # Non-Gly residues: BB + SC
        bb_idx = len(beads)

        beads.append({
            "name": "BB",
            "resid": resid,
            "kind": "BB",
            "aa": aa,
            "radius_A": AA_PARAMS["BB"]["radius_A"],
            "charge": 0.0
        })

        sc_idx = len(beads)

        beads.append({
            "name": "SC",
            "resid": resid,
            "kind": "SC",
            "aa": aa,
            "radius_A": AA_PARAMS[aa]["radius_A"],
            "charge": AA_PARAMS[aa]["charge"]
        })

        residue_to_beads.append([bb_idx, sc_idx])

    return beads, residue_to_beads

def ion_counts(ions):
    return {
        "K": sum(1 for x in ions if x["name"] == "K"),
        "Mg": sum(1 for x in ions if x["name"] == "Mg"),
        "Cl": sum(1 for x in ions if x["name"] == "Cl")
    }

def total_ion_charge(ions):
    return sum(float(x["charge"]) for x in ions)

def total_protein_charge(beads):
    return sum(float(x["charge"]) for x in beads)

def build_ions_from_pdb_order(pdb_file):
    """
    Assumes:
      protein beads are ATOM records
      ions are HETATM records
    """
    ion_params = ion_params_from_config()
    ions = []

    allowed = {"K": "K", "MG": "Mg", "CL": "Cl"}

    with open(pdb_file) as f:
        for line in f:
            rec = line[:6].strip()

            if rec != "HETATM":
                continue

            atom_name = line[12:16].strip().upper()

            if atom_name not in allowed:
                raise RuntimeError(
                    "Unknown ion atom name in PDB: %s. Expected K, Mg, or Cl."
                    % atom_name
                )

            ion_name = allowed[atom_name]

            ion = dict(ion_params[ion_name])
            ion["name"] = ion_name
            ions.append(ion)

    return ions
beads, residue_to_beads = build_beads_from_sequence(sequence)
ions = build_ions_from_pdb_order(pdb_file)


n_protein_beads = len(beads)
n_ions = len(ions)
n_total = n_protein_beads + n_ions

pdb = app.PDBFile(pdb_file)
positions = pdb.positions
pos_A = np.array([p.value_in_unit(u.angstrom) for p in positions])

if len(pos_A) != n_total:
    raise RuntimeError(
        "Particle count mismatch: PDB has %d positions, reconstructed system has %d "
        "(%d protein beads + %d ions)."
        % (len(pos_A), n_total, n_protein_beads, n_ions)
    )

BT_INDEX = {aa: i for i, aa in enumerate(BT_ORDER)}

def bt_score(aa1, aa2):
    i = BT_INDEX[aa1.upper()]
    j = BT_INDEX[aa2.upper()]
    return 0.5 * (float(BT_MATRIX[i, j]) + float(BT_MATRIX[j, i]))

def bt_prefactor(aa1, aa2):
    return omega * kBT_kcal * abs(bt_score(aa1, aa2) - bt_offset) * u.kilocalorie_per_mole

def sigma_A(i, j):
    return float(beads[i]["radius_A"]) + float(beads[j]["radius_A"])

def residue_separation(i, j):
    return abs(int(beads[i]["resid"]) - int(beads[j]["resid"]))

def pair_type(i, j):
    ki = beads[i]["kind"]
    kj = beads[j]["kind"]

    if ki == "BB" and kj == "BB":
        return "BB"
    if ki == "SC" and kj == "SC":
        return "SS"
    return "BS"

def bs_sidechain_aa(i, j):
    if beads[i]["kind"] == "SC":
        return beads[i]["aa"]
    return beads[j]["aa"]

print("Constructing explicit-ion SOP-SC simulation for globular protein with %d protein beads and %d ions." % (n_protein_beads, n_ions))

system = mm.System()    #Initializes the empty OpenMM system.

system.setDefaultPeriodicBoxVectors(
    mm.Vec3(box_length_A, 0.0, 0.0) * u.angstrom,
    mm.Vec3(0.0, box_length_A, 0.0) * u.angstrom,
    mm.Vec3(0.0, 0.0, box_length_A) * u.angstrom
)

for bead in beads:
    system.addParticle(mProtein * u.amu)

for ion in ions:
    system.addParticle(float(ion["mass_amu"]) * u.amu)

#========================================================================================================
# 1. Bonding: apply a FENE spring between adjacent backbone beads and between backbones and their sidechains. 
#========================================================================================================
fene_expr = "-0.5*k*R0^2*log(1 - ((r-rref)/R0)^2)"

FENEForce = mm.CustomBondForce(fene_expr)
FENEForce.addPerBondParameter("k")
FENEForce.addPerBondParameter("R0")
FENEForce.addPerBondParameter("rref")

bonded_pairs = set()
rref_dict = {}


def add_fene(i, j):
    rref = np.linalg.norm(pos_A[i] - pos_A[j])
    FENEForce.addBond(int(i), int(j), [fene_k, fene_R0, rref * u.angstrom])
    pair = tuple(sorted((int(i), int(j))))
    bonded_pairs.add(pair)
    rref_dict[pair] = rref


for r in range(len(residue_to_beads) - 1):
    add_fene(residue_to_beads[r][0], residue_to_beads[r + 1][0])

for bb, sc in residue_to_beads:
    if sc is not None:
        add_fene(bb, sc)

system.addForce(FENEForce)

#==========================================================================================================
#2. Steric Clash: repulsive 1/r^6 interaction for beads separated by <= 2 residues to prevent overlapping.
#==========================================================================================================
# LocalRepForce = mm.CustomBondForce("eps*(sig/r)^6") #idp
# LocalRepForce = mm.CustomBondForce("eps*(sig/r)^12") #globular
LocalRepForce = mm.CustomBondForce("step(sig-r)*(4*eps*((sig/r)^12 - (sig/r)^6) + eps)") #globular
LocalRepForce.addPerBondParameter("eps")
LocalRepForce.addPerBondParameter("sig")

local_pairs = set()

for i in range(n_protein_beads):
    for j in range(i + 1, n_protein_beads):
        pair = tuple(sorted((i, j)))

        if pair in bonded_pairs:
            continue

        if residue_separation(i, j) <= 2:
            LocalRepForce.addBond(i, j, [eps_local, sigma_A(i, j) * u.angstrom])
            local_pairs.add(pair)

system.addForce(LocalRepForce)

# ============================================================
# 3. Native-contact interactions for globular SOP-SC model
# ============================================================

NativeForce = mm.CustomBondForce(
    "eps*((r0/r)^12 - 2*(r0/r)^6)"
)

NativeForce.addGlobalParameter(
    "eps",
    1.0 * u.kilocalorie_per_mole
)

NativeForce.addPerBondParameter("r0")

native_pairs = set()

with open("native_contacts.dat") as f:
    for line in f:

        if line.startswith("#"):
            continue

        fields = line.split()

        i = int(fields[0])
        j = int(fields[1])
        r0 = float(fields[2])

        NativeForce.addBond(
            i,
            j,
            [r0 * u.angstrom]
        )

        native_pairs.add(tuple(sorted((i, j))))

print("Native contacts loaded:", len(native_pairs))

# ------------------------------------------------------------
# Global exclusion list
# Used consistently across all nonbonded forces
# ------------------------------------------------------------

all_exclusions = set()

for i, j in bonded_pairs:
    all_exclusions.add(tuple(sorted((int(i), int(j)))))

for i, j in local_pairs:
    all_exclusions.add(tuple(sorted((int(i), int(j)))))

for i, j in native_pairs:
    all_exclusions.add(tuple(sorted((int(i), int(j)))))

system.addForce(NativeForce)

# ============================================================
# 4. Non-native excluded volume interactions
# ============================================================

NonNativeRepForce = mm.CustomNonbondedForce(
    "step(rc-r)*(4*eps*((sig/r)^12 - (sig/r)^6) + eps);"
    "sig = 0.5*(sig1 + sig2)"
)

NonNativeRepForce.addGlobalParameter(
    "eps",
    1.0 * u.kilocalorie_per_mole
)

NonNativeRepForce.addGlobalParameter(
    "rc",
    0.5612310241546865
)

NonNativeRepForce.addPerParticleParameter("sig")

NonNativeRepForce.setNonbondedMethod(
    mm.CustomNonbondedForce.CutoffPeriodic
)

NonNativeRepForce.setCutoffDistance(
    20.0 * u.angstrom
)

# ------------------------------------------------------------
# Add particles
# ------------------------------------------------------------

for bead in beads:

    sigma_nm = 2.0 * float(bead["radius_A"]) * 0.1

    NonNativeRepForce.addParticle([sigma_nm])

for ion in ions:

    NonNativeRepForce.addParticle([0.3])

# ------------------------------------------------------------
# Exclusions
# ------------------------------------------------------------

for i, j in bonded_pairs:
    NonNativeRepForce.addExclusion(i, j)

for i, j in local_pairs:
    NonNativeRepForce.addExclusion(i, j)

for i, j in native_pairs:
    NonNativeRepForce.addExclusion(i, j)

system.addForce(NonNativeRepForce)

#==================================================================================================================================
# 5. Protein-Protein Interactions: LJ potential to handle interactions between protein beads that are further apart in the sequence. 
# The strength (eps) is determined by the bt_prefactor.
#==================================================================================================================================

AA_ORDER = list(BT_ORDER)
AA_TO_TYPE = {aa: i + 1 for i, aa in enumerate(AA_ORDER)}

BB_TYPE = 0
NTYPES = 1 + len(AA_ORDER)

eps_table = np.zeros((NTYPES, NTYPES), dtype=float)   # kcal/mol
sig_table = np.zeros((NTYPES, NTYPES), dtype=float)   # Angstrom


def radius_for_type(t):
    if t == BB_TYPE:
        return AA_PARAMS["BB"]["radius_A"]

    aa = AA_ORDER[t - 1]
    return AA_PARAMS[aa]["radius_A"]


def aa_for_type(t):
    if t == BB_TYPE:
        return "G"

    return AA_ORDER[t - 1]

# ------------------------------------------------------------
# Build epsilon/sigma lookup tables
# ------------------------------------------------------------

for t1 in range(NTYPES):
    for t2 in range(NTYPES):

        if t1 == BB_TYPE and t2 == BB_TYPE:
            eps = bt_prefactor("G", "G").value_in_unit(u.kilocalorie_per_mole)
            sig = 3.8

        elif t1 == BB_TYPE and t2 != BB_TYPE:
            aa = aa_for_type(t2)
            eps = bt_prefactor("G", aa).value_in_unit(u.kilocalorie_per_mole)
            sig = radius_for_type(t1) + radius_for_type(t2)

        elif t2 == BB_TYPE and t1 != BB_TYPE:
            aa = aa_for_type(t1)
            eps = bt_prefactor("G", aa).value_in_unit(u.kilocalorie_per_mole)
            sig = radius_for_type(t1) + radius_for_type(t2)

        else:
            aa1 = aa_for_type(t1)
            aa2 = aa_for_type(t2)
            eps = bt_prefactor(aa1, aa2).value_in_unit(u.kilocalorie_per_mole)
            sig = radius_for_type(t1) + radius_for_type(t2)

        eps_table[t1, t2] = eps
        sig_table[t1, t2] = sig

# ------------------------------------------------------------
# Convert table values to OpenMM internal units
# CustomNonbondedForce expression uses:
#   r in nm
#   energy in kJ/mol
# OpenMM table values are dimensionless numbers.
# Convert model units:
#   kcal/mol -> kJ/mol
#   Angstrom -> nm
# ------------------------------------------------------------

eps_table_kj = eps_table * 4.184
sig_table_nm = sig_table * 0.1
rc_nm = float(config.get("nonlocal_cutoff", 30.0)) * 0.1

# NonlocalPPForce = mm.CustomNonbondedForce(
#     "epsTable(type1,type2)*("
#     "(sigTable(type1,type2)/r)^12"
#     "-2*(sigTable(type1,type2)/r)^6"
#     "-("
#     "(sigTable(type1,type2)/rc)^12"
#     "-2*(sigTable(type1,type2)/rc)^6"
#     ")"
#     ")"
# )

NonlocalPPForce = mm.CustomNonbondedForce(
    "isSC1*isSC2*"
    "epsTable(type1,type2)*("
    "(sigTable(type1,type2)/r)^12"
    "-2*(sigTable(type1,type2)/r)^6"
    "-("
    "(sigTable(type1,type2)/rc_bt)^12"
    "-2*(sigTable(type1,type2)/rc_bt)^6"
    ")"
    ")"
)
NonlocalPPForce.addPerParticleParameter("type")
NonlocalPPForce.addPerParticleParameter("isSC")
NonlocalPPForce.addGlobalParameter("rc_bt", rc_nm)

NonlocalPPForce.addTabulatedFunction(
    "epsTable",
    mm.Discrete2DFunction(
        NTYPES,
        NTYPES,
        eps_table_kj.flatten().tolist()
    )
)

NonlocalPPForce.addTabulatedFunction(
    "sigTable",
    mm.Discrete2DFunction(
        NTYPES,
        NTYPES,
        sig_table_nm.flatten().tolist()
    )
)

NonlocalPPForce.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
NonlocalPPForce.setCutoffDistance(rc_nm * u.nanometer)


# ------------------------------------------------------------
# Add particles
# All particles in the System must be added to CustomNonbondedForce.
# Protein particles get meaningful types.
# Ions get dummy type but are excluded by interaction group.
# ------------------------------------------------------------

for bead in beads:
    if bead["kind"] == "BB":
        # Generic backbone bead
        t = BB_TYPE
    else:
        # SC beads and single-bead Gly use residue-specific type
        t = AA_TO_TYPE[bead["aa"]]

#     NonlocalPPForce.addParticle([float(t)])

    is_sc = 1.0 if bead["kind"] == "SC" else 0.0

    NonlocalPPForce.addParticle([
        float(t),
        is_sc
    ])

for ion in ions:
    # NonlocalPPForce.addParticle([float(BB_TYPE)])
    NonlocalPPForce.addParticle([
    float(BB_TYPE),
    0.0
])


# ------------------------------------------------------------
# Exclude bonded and local PP pairs
# ------------------------------------------------------------

# for i, j in bonded_pairs:
#     NonlocalPPForce.addExclusion(int(i), int(j))

# for i, j in local_pairs:
#     NonlocalPPForce.addExclusion(int(i), int(j))

for i, j in all_exclusions:
    NonlocalPPForce.addExclusion(i, j)


# ------------------------------------------------------------
# Restrict this force to protein-protein interactions only
# ------------------------------------------------------------

protein_indices = set(range(n_protein_beads))
NonlocalPPForce.addInteractionGroup(protein_indices, protein_indices)

system.addForce(NonlocalPPForce)

#=====================================================================================================================
# 4. Electrostatics: PME to calculate long-range electrostatic interactions between charged beads and ions. 
# explicitly scales charges by 1 / sqrt(dielectric) to account for implicit water screening. Bonded pairs are excluded.
#======================================================================================================================
ESForce = mm.NonbondedForce()
ESForce.setNonbondedMethod(mm.NonbondedForce.PME)
ESForce.setEwaldErrorTolerance(5e-3)
ESForce.setCutoffDistance(pme_cutoff)

charge_scale = 1.0 / np.sqrt(dielectric)

for bead in beads:
    q = float(bead["charge"]) * charge_scale
    ESForce.addParticle(
        q * u.elementary_charge,
        0.0 * u.angstrom,
        0.0 * u.kilocalorie_per_mole
    )

for ion in ions:
    q = float(ion["charge"]) * charge_scale
    ESForce.addParticle(
        q * u.elementary_charge,
        0.0 * u.angstrom,
        0.0 * u.kilocalorie_per_mole
    )

# for i, j in bonded_pairs:
#     ESForce.addException(
#         int(i), int(j),
#         0.0 * u.elementary_charge * u.elementary_charge,
#         0.0 * u.angstrom,
#         0.0 * u.kilocalorie_per_mole
#     )

# for i, j in local_pairs:
#     ESForce.addException(
#         int(i), int(j),
#         0.0 * u.elementary_charge * u.elementary_charge,
#         0.0 * u.angstrom,
#         0.0 * u.kilocalorie_per_mole
#     )

# ------------------------------------------------------------
# Electrostatic exclusions
# ------------------------------------------------------------

for i, j in all_exclusions:

    ESForce.addException(
        i,
        j,
        0.0 * u.elementary_charge * u.elementary_charge,
        1.0 * u.nanometer,
        0.0 * u.kilojoule_per_mole,
        replace=True
    )

system.addForce(ESForce)

#===============================================================================================================================================
# 5. Excluded Volume / Ion Interactions: LJ interactions between Ions, and between Ions and the Protein. 
# The variable factor = 1 - isProtein1*isProtein2 ensures this force ignores protein-protein pairs (since NonlocalPPForce already handles them).
#===============================================================================================================================================
EVForce = mm.CustomNonbondedForce(
    "factor*select(step(sig-r), eps*((sig/r)^12 - 2*(sig/r)^6 + 1), 0);"
    "sig = radius1 + radius2;"
    "eps = sqrt(epsilon1*epsilon2);"
    "factor = 1 - isProtein1*isProtein2"
)

EVForce.addPerParticleParameter("radius")
EVForce.addPerParticleParameter("epsilon")
EVForce.addPerParticleParameter("isProtein")

EVForce.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
EVForce.setCutoffDistance(ev_cutoff)

for bead in beads:
    EVForce.addParticle([
        float(bead["radius_A"]) * u.angstrom,
        eps_protein_ion,
        1.0
    ])

for ion in ions:
    EVForce.addParticle([
        float(ion["radius_A"]) * u.angstrom,
        float(ion["epsilon_kcal"]) * u.kilocalorie_per_mole,
        0.0
    ])

# for i, j in bonded_pairs:
#     EVForce.addExclusion(int(i), int(j))

# for i, j in local_pairs:
#     EVForce.addExclusion(int(i), int(j))

for i, j in all_exclusions:
    EVForce.addExclusion(i, j)

system.addForce(EVForce)

#==========COM Motion Remover =================
if ComMotionRemover:
    cmm = mm.CMMotionRemover()
    cmm.setFrequency(CMMR_frequency)
    system.addForce(cmm)

for i in range(system.getNumForces()):
    system.getForce(i).setForceGroup(i)


class EnergyReporter(object):
    def __init__(self, filename, interval, nforces):
        self._out = open(filename, "w")
        self._interval = interval
        self._nforces = nforces

    def __del__(self):
        try:
            self._out.close()
        except Exception:
            pass

    def describeNextReport(self, simulation):
        step = self._interval - simulation.currentStep % self._interval
        return (step, False, False, False, True)

    def report(self, simulation, state):
        self._out.write(str(simulation.currentStep))
        for i in range(self._nforces):
            s = simulation.context.getState(getEnergy=True, groups={i})
            e = s.getPotentialEnergy() / u.kilocalorie_per_mole
            self._out.write("," + str(e))
        self._out.write("\n")
        self._out.flush()


def compute_protein_rg_A(state):
    p = state.getPositions(asNumpy=True).value_in_unit(u.angstrom)
    p = p[:n_protein_beads]
    cm = p.mean(axis=0)
    return np.sqrt(np.mean(np.sum((p - cm) ** 2, axis=1)))


def check_fene_bonds(context):
    state = context.getState(getPositions=True)
    p = state.getPositions(asNumpy=True).value_in_unit(u.angstrom)

    max_dev = -1.0
    worst = None

    for (i, j), rref in rref_dict.items():
        r = np.linalg.norm(p[i] - p[j])
        dev = abs(r - rref)

        if dev > max_dev:
            max_dev = dev
            worst = (i, j, r, rref, dev)

    print("Max FENE deviation:", max_dev, "A")
    print("Worst FENE bond:", worst)


integrator = mm.LangevinMiddleIntegrator(
    T,
    friction_coeff / u.picosecond,
    time_step * u.picoseconds
)

platform = mm.Platform.getPlatformByName(platform_type)

if platform_type == "CUDA":
    properties = {"CudaPrecision": "mixed"}
    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
else:
    simulation = app.Simulation(pdb.topology, system, integrator, platform)

if restart:
    print("restart: True")
    simulation.loadCheckpoint("chkin.chk")
else:
    print("restart: False")
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(T)

counts = ion_counts(ions)

print("Sequence length:", len(sequence))
print("Protein beads:", n_protein_beads)
print("nK:", counts["K"])
print("nMg:", counts["Mg"])
print("nCl:", counts["Cl"])
print("Total ions:", n_ions)
print("Total particles:", system.getNumParticles())
print("Protein charge:", total_protein_charge(beads))
print("Ion charge:", total_ion_charge(ions))
print("Total charge:", total_protein_charge(beads) + total_ion_charge(ions))
print("Box length:", box_length_A, "A")
print("Protein mass per bead:", mProtein, "amu")
print("omega:", omega)
print("bt_offset:", bt_offset)

print("\nInitial energy:")
initial_state = simulation.context.getState(getEnergy=True)
initial_energy = initial_state.getPotentialEnergy()
print(initial_energy)

labels = [
    "FENEForce",
    "LocalRepForce",
    "NativeForce",
    "NonNativeRepForce",
    "NonlocalPPForce",
    "EVForce_PI_II",
    "ESForce_PME",
    "CMMotionRemover"
]

print("\nEnergy components of initial position")
for i in range(system.getNumForces()):
    label = labels[i] if i < len(labels) else system.getForce(i).__class__.__name__
    e = simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
    print("%d.%s :" % (i + 1, label), e)

initial_kcal = initial_energy.value_in_unit(u.kilocalorie_per_mole)

if np.isnan(initial_kcal) or np.isinf(initial_kcal):
    raise RuntimeError("Initial energy is NaN/Inf")

if minimization:
    print("\nPerforming minimization")
    # simulation.minimizeEnergy(tolerance=1e-5 * u.kilojoule_per_mole / u.nanometer)
    simulation.minimizeEnergy(tolerance=10 * u.kilojoule_per_mole / u.nanometer)

    state = simulation.context.getState(getPositions=True, getEnergy=True)
    e = state.getPotentialEnergy()
    print("Minimized energy:", e)

    e_kcal = e.value_in_unit(u.kilocalorie_per_mole)

    if np.isnan(e_kcal) or np.isinf(e_kcal):
        raise RuntimeError("Energy became NaN/Inf after minimization.")

    check_fene_bonds(simulation.context)

    with open("%s_minimized.pdb" % pdb_prefix, "w") as f:
        app.PDBFile.writeFile(pdb.topology, state.getPositions(), f)

print("\nSimulating with T =", T)
print("Initiating MD simulation for %d steps" % numsteps)

simulation.reporters.append(app.DCDReporter("%s.dcd" % dcd_prefix, snap_interval))

simulation.reporters.append(
    app.StateDataReporter(
        file=data_name,
        reportInterval=data_interval,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        progress=True,
        remainingTime=True,
        speed=True,
        totalSteps=numsteps,
        separator=","
    )
)

simulation.reporters.append(
    app.StateDataReporter(
        stdout,
        reportInterval=data_interval,
        step=True,
        time=True,
        potentialEnergy=True,
        kineticEnergy=True,
        temperature=True,
        progress=True,
        remainingTime=True,
        speed=True,
        totalSteps=numsteps,
        separator="   |   "
    )
)

simulation.reporters.append(app.CheckpointReporter(checkpoint_name, data_interval))
simulation.reporters.append(EnergyReporter(energy_name, data_interval, system.getNumForces()))

simulation.step(numsteps)

final_state_obj = simulation.context.getState(getPositions=True, getEnergy=True)

print("Final energy:", final_state_obj.getPotentialEnergy())
print("Final protein Rg:", compute_protein_rg_A(final_state_obj), "A")

check_fene_bonds(simulation.context)

if final_state:
    with open("%s_final.pdb" % pdb_prefix, "w") as f:
        app.PDBFile.writeFile(pdb.topology, final_state_obj.getPositions(), f)

    print("\nEnergy components of final position")
    for i in range(system.getNumForces()):
        label = labels[i] if i < len(labels) else system.getForce(i).__class__.__name__
        e = simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
        print("%d.%s :" % (i + 1, label), e)

end_time = datetime.now()
print("Duration:", end_time - start_time)