"""
Stage 3 — solvate the fixed protein in a water box, neutralize + add bulk
ion concentration, and build the OpenMM System (PME electrostatics, HMR,
HBonds constraints per input.json).

Reads:  paths.fixed_pdb
Writes: paths.solvated_pdb, paths.system_xml
"""
from openmm import app, unit, XmlSerializer

from config import load_config, outpath


def main():
    cfg = load_config()
    ff_cfg = cfg["forcefield"]
    solv_cfg = cfg["solvation"]
    sim_cfg = cfg["simulation"]

    if cfg["structure"].get("use_compacted_tails", True):
        source_pdb = outpath(cfg, "compacted_pdb")
    else:
        source_pdb = outpath(cfg, "fixed_pdb")
    out_pdb = outpath(cfg, "solvated_pdb")
    out_system = outpath(cfg, "system_xml")

    print(f"Building system from: {source_pdb}")
    pdb = app.PDBFile(source_pdb)

    forcefield = app.ForceField(*ff_cfg["files"])

    modeller = app.Modeller(pdb.topology, pdb.positions)

    box_shape = solv_cfg.get("box_shape", "cube")
    explicit_box_nm = solv_cfg.get("box_size_nm")  # e.g. [12.0, 12.0, 12.0]

    solvent_kwargs = dict(
        forcefield=forcefield,
        model=ff_cfg["water_model"],
        positiveIon=solv_cfg["positive_ion"],
        negativeIon=solv_cfg["negative_ion"],
        ionicStrength=solv_cfg["ion_concentration_M"] * unit.molar,
        neutralize=solv_cfg.get("neutralize", True),
    )

    if explicit_box_nm:
        # Modeller.addSolvent takes EITHER padding+boxShape OR an explicit
        # boxSize — not both. Explicit size wins if set; make sure it's
        # actually big enough to contain the solute plus your cutoff, or
        # addSolvent will raise (or silently clip the solute at the edge).
        from openmm import Vec3
        solvent_kwargs["boxSize"] = Vec3(*explicit_box_nm) * unit.nanometer
        print(f"Using explicit box size: {explicit_box_nm} nm")
    else:
        solvent_kwargs["padding"] = solv_cfg["padding_nm"] * unit.nanometer
        if box_shape:
            solvent_kwargs["boxShape"] = box_shape

    modeller.addSolvent(**solvent_kwargs)
    
    from collections import Counter

    counts = Counter(res.name for res in modeller.topology.residues())

    print("\n=== After OpenMM addSolvent() ===")
    for name in sorted(counts):
        if name in {"HOH", "WAT", "TIP3", "SOL", "NA", "CL", "CLA", "SOD"}:
            print(f"{name:>5s}: {counts[name]}")

    with open(out_pdb, "w") as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    nonbonded_method = {
        "PME": app.PME,
        "CutoffPeriodic": app.CutoffPeriodic,
        "NoCutoff": app.NoCutoff,
    }[sim_cfg["nonbonded_method"]]

    constraints = {
        "HBonds": app.HBonds,
        "AllBonds": app.AllBonds,
        "HAngles": app.HAngles,
        "None": None,
    }[sim_cfg["constraints"]]

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=sim_cfg["nonbonded_cutoff_nm"] * unit.nanometer,
        switchDistance=sim_cfg["switch_distance_nm"] * unit.nanometer,
        constraints=constraints,
        rigidWater=sim_cfg.get("rigid_water", True),
        hydrogenMass=sim_cfg["hydrogen_mass_amu"] * unit.amu,
    )

    with open(out_system, "w") as f:
        f.write(XmlSerializer.serialize(system))

    n_atoms = modeller.topology.getNumAtoms()
    box = modeller.topology.getPeriodicBoxVectors()
    print(f"Solvated system atom count: {n_atoms}")
    print(f"Box vectors (nm): {box}")
    print(f"Solvated PDB written to: {out_pdb}")
    print(f"Serialized System written to: {out_system}")


if __name__ == "__main__":
    main()