"""
Stage 3b — add extra salt species (Mg2+, Zn2+, Cd2+, Ca2+, or any other ion
CHARMM36(m) has a template for) beyond the monovalent Na+/Cl- that
03_build_system.py's Modeller.addSolvent() already placed.

WHY THIS EXISTS: OpenMM's Modeller.addSolvent() only supports monovalent
ions — the positiveIon argument is hardcoded to accept just
{Cs+, K+, Li+, Na+, Rb+} and negativeIon to {Cl-, Br-, F-, I-}. There is no
way to ask it for Mg2+, Ca2+, Zn2+, etc. even though CHARMM36 ships force
field parameters for them. So this script does it manually: for each salt
you list in input.json, it works out how many formula units belong in the
box at the target concentration, then converts that many randomly-chosen
water molecules into ions (deleting the water's H1/H2, replacing O with the
ion), charge-balancing each salt internally (e.g. MgCl2 -> 1 Mg2+ + 2 Cl-
per formula unit) so the system stays neutral independent of whatever
02b/03 already did for Na+/Cl-.

CHARMM36(m)'s standard ion set (checked directly against the bundled
parameter file) covers: Li+, Na+, Mg2+, K+, Ca2+, Rb+, Cs+, Ba2+, Zn2+,
Cd2+, Cl-. It does NOT include Co2+ — if you need cobalt you'll have to
source and validate your own LJ parameters and add a custom residue
template; this script won't fabricate one.

No distance-based clash avoidance is done beyond "don't reuse the same
water twice" — minimizeEnergy() in 04_run_md.py resolves any close contacts
from an ion landing near solute/another ion. If you're placing many ions
at once, do sanity-check final placements aren't literally on top of the
protein (see the printed summary).

Reads:  paths.solvated_pdb, solvation.extra_salts (input.json)
Writes: overwrites paths.solvated_pdb, and re-serializes paths.system_xml
        against the new topology (so 04_run_md.py doesn't need changes).
"""
import random

import numpy as np
from openmm import app, unit, XmlSerializer

from config import load_config, outpath

# CHARMM36(m) atom names / elements for ions this script knows how to place.
# Extend this dict if you validate parameters for something else.
ION_TEMPLATES = {
    "SOD": {"element": app.element.sodium, "charge": 1},
    "POT": {"element": app.element.potassium, "charge": 1},
    "LIT": {"element": app.element.lithium, "charge": 1},
    "RUB": {"element": app.element.rubidium, "charge": 1},
    "CES": {"element": app.element.cesium, "charge": 1},
    "CLA": {"element": app.element.chlorine, "charge": -1},
    "MG": {"element": app.element.magnesium, "charge": 2},
    "CAL": {"element": app.element.calcium, "charge": 2},
    "ZN2": {"element": app.element.zinc, "charge": 2},
    "CD2": {"element": app.element.cadmium, "charge": 2},
    "BAR": {"element": app.element.barium, "charge": 2},
}

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "SOL"}


def box_volume_liters(topology):
    a, b, c = topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    vol_nm3 = float(np.abs(np.dot(a, np.cross(b, c))))
    return vol_nm3 * 1e-24  # nm^3 -> L


def main():
    cfg = load_config()
    solv_cfg = cfg["solvation"]
    extra_salts = solv_cfg.get("extra_salts", [])

    if not extra_salts:
        print("No solvation.extra_salts configured in input.json — nothing to do.")
        return

    solvated_pdb = outpath(cfg, "solvated_pdb")
    pdb = app.PDBFile(solvated_pdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)

    volume_L = box_volume_liters(modeller.topology)
    NA = 6.02214076e23

    water_residues = [r for r in modeller.topology.residues() if r.name in WATER_RESNAMES]
    random.shuffle(water_residues)
    pool = iter(water_residues)

    to_delete = []
    ions_to_add = []  # list of (resname, element, charge, position)

    for salt in extra_salts:
        cation_resname = salt["cation_resname"]
        anion_resname = salt["anion_resname"]
        conc_M = salt["concentration_M"]

        if cation_resname not in ION_TEMPLATES:
            raise ValueError(
                f"'{cation_resname}' has no known CHARMM36 template in "
                f"ION_TEMPLATES. Known: {list(ION_TEMPLATES)}. If this is "
                f"Co2+, see the module docstring — you need to source and "
                f"add your own parameters first."
            )
        if anion_resname not in ION_TEMPLATES:
            raise ValueError(f"'{anion_resname}' has no known CHARMM36 template.")

        cation_charge = ION_TEMPLATES[cation_resname]["charge"]
        anion_charge = ION_TEMPLATES[anion_resname]["charge"]

        n_formula_units = round(conc_M * NA * volume_L)
        n_cations = n_formula_units
        n_anions = round(n_formula_units * cation_charge / abs(anion_charge))

        print(
            f"{cation_resname}{anion_resname if anion_charge != -1 else anion_resname}: "
            f"{conc_M} M -> {n_cations}x {cation_resname} ({cation_charge:+d}), "
            f"{n_anions}x {anion_resname} ({anion_charge:+d})"
        )

        for _ in range(n_cations):
            wat = next(pool)
            to_delete.append(wat)
            o_atom = next(a for a in wat.atoms() if a.name == "O")
            pos = pdb.positions[o_atom.index] if wat in pdb.topology.residues() else None
            ions_to_add.append((cation_resname, ION_TEMPLATES[cation_resname]["element"], o_atom))

        for _ in range(n_anions):
            wat = next(pool)
            to_delete.append(wat)
            o_atom = next(a for a in wat.atoms() if a.name == "O")
            ions_to_add.append((anion_resname, ION_TEMPLATES[anion_resname]["element"], o_atom))

    # Capture positions of the oxygen atoms BEFORE deleting anything
    # (atom indices shift once modeller.delete() runs).
    positions_before = modeller.positions
    ion_positions = [positions_before[o_atom.index] for _, _, o_atom in ions_to_add]

    modeller.delete(to_delete)

    ion_topology = app.Topology()
    chain = ion_topology.addChain()
    for resname, element, _ in ions_to_add:
        res = ion_topology.addResidue(resname, chain)
        ion_topology.addAtom(resname, element, res)

    modeller.add(ion_topology, ion_positions)

    from collections import Counter

    counts = Counter(r.name for r in modeller.topology.residues())

    print("\n=== Final Solvent/Ion Composition ===")

    for resname in sorted(counts):
        if resname in {
            "HOH", "TIP3", "WAT", "SOL",
            "NA", "SOD",
            "CL", "CLA",
            "MG", "CAL",
            "ZN2", "CD2", "BAR"
        }:
            print(f"{resname:6s}: {counts[resname]}")

    with open(solvated_pdb, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    # Rebuild the System against the new topology so 04_run_md.py sees a
    # consistent solvated_pdb + system_xml pair.
    ff_cfg = cfg["forcefield"]
    sim_cfg = cfg["simulation"]
    forcefield = app.ForceField(*ff_cfg["files"])

    nonbonded_method = {"PME": app.PME, "CutoffPeriodic": app.CutoffPeriodic, "NoCutoff": app.NoCutoff}[
        sim_cfg["nonbonded_method"]
    ]
    constraints = {"HBonds": app.HBonds, "AllBonds": app.AllBonds, "HAngles": app.HAngles, "None": None}[
        sim_cfg["constraints"]
    ]

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=sim_cfg["nonbonded_cutoff_nm"] * unit.nanometer,
        switchDistance=sim_cfg["switch_distance_nm"] * unit.nanometer,
        constraints=constraints,
        rigidWater=sim_cfg.get("rigid_water", True),
        hydrogenMass=sim_cfg["hydrogen_mass_amu"] * unit.amu,
    )

    with open(outpath(cfg, "system_xml"), "w") as f:
        f.write(XmlSerializer.serialize(system))

    print(f"Replaced {len(to_delete)} waters with {len(ions_to_add)} ions.")
    print(f"Updated {solvated_pdb} and {outpath(cfg, 'system_xml')} in place.")


if __name__ == "__main__":
    main()