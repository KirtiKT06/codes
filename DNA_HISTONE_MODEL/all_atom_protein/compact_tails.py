"""
Stage 2b — compact the newly built tails before explicit solvation.

PDBFixer's addMissingAtoms() builds missing stretches with a simple local
geometric extension. For short gaps (a few residues) this is fine. For long
disordered termini (here: up to ~39 residues) it tends to build them nearly
fully extended (~0.38-0.5 nm per CA-CA step, i.e. close to a straight line)
because there's no energetic collapse involved — it's pure geometry, not a
simulation. For chain A alone this produced a bounding box of ~19 x 6 x 15 nm,
which would force an explicit water box costing orders of magnitude more
compute than the folded octamer actually needs.

This stage relaxes that artificial geometry using implicit solvent (fast,
no water molecules) before you ever build the real explicit-solvent box in
build_system.py:
  - the ordered core (every residue PDBFixer did NOT add, from
    added_residues.json) is harmonically restrained to its crystal
    position
  - the built tail residues are free and are given a short Langevin run
    under GBSA-GBn2 implicit solvent, letting them collapse away from the
    fully-extended starting geometry into something with realistic radius
    of gyration
  - this is still just a better *starting guess* for a disordered region,
    not a claim about the true ensemble — production sampling in explicit
    solvent is what actually samples tail conformational space

FIX (see chat): a previous version of this file had
`from sympy import false` followed by `if false:` instead of
`if key not in flexible_res_keys:`. sympy.false is a SymPy symbolic
Boolean, not Python's builtin False, but it still evaluates as falsy in
an `if` — so that branch never ran and ZERO atoms were ever restrained
("Restrained (core) atoms: 0" in that run's output), including the
ordered core. That let the whole chain (core + tail together) run free
under implicit-solvent Langevin dynamics with no anchor at all, which is
the most likely cause of the NaN crash mid-run: a locally strained
contact that minimization plateaus near without fully resolving, then
blows up once nothing is holding the rest of the structure in place.
This version restores the original restrain-core / free-tail logic.

Reads:  paths.fixed_pdb, paths.added_residues_json
Writes: paths.compacted_pdb (paths.fixed_pdb is left untouched so you can
        compare before/after). build_system.py reads
        structure.use_compacted_tails to decide which of the two to solvate.
"""
import json

from openmm import (
    app,
    unit,
    LangevinMiddleIntegrator,
    CustomExternalForce,
    Platform,
)

from config import load_config, outpath


def main():
    cfg = load_config()

    fixed_pdb = outpath(cfg, "fixed_pdb")
    added_json = outpath(cfg, "added_residues_json")
    out_pdb = outpath(cfg, "compacted_pdb")

    with open(added_json) as f:
        added = json.load(f)
    added_chains = {a["chain"] for a in added}

    if not added:
        print("No added residues recorded — nothing to compact. Skipping.")
        return

    pdb = app.PDBFile(fixed_pdb)

    forcefield = app.ForceField("charmm36m.xml", "implicit/gbn2.xml")
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.5 * unit.nanometer,
        constraints=app.HBonds,
    )

    # Harmonic positional restraint on every atom NOT in a newly-built
    # residue. We identify "built" residues by (chain id, residue index
    # within chain) using the same offset logic as fix_structure.py:
    # PDBFixer inserts `offset` residues immediately after
    # insert_after_residue_index, so we mark the resulting residue indices
    # in the FIXED topology as flexible.
    chains = list(pdb.topology.chains())
    flexible_res_keys = set()
    by_chain = {}
    for a in added:
        by_chain.setdefault(a["chain_index"], []).append(a)
    for chain_index, entries in by_chain.items():
        entries.sort(key=lambda e: (e["insert_after_residue_index"], e["offset"]))
        for e in entries:
            new_res_index = e["insert_after_residue_index"] + 1 + e["offset"]
            flexible_res_keys.add((chain_index, new_res_index))

    restraint = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    positions = pdb.positions
    n_restrained = 0
    for atom in pdb.topology.atoms():
        chain_index = list(pdb.topology.chains()).index(atom.residue.chain)
        res_index_in_chain = list(atom.residue.chain.residues()).index(atom.residue)
        key = (chain_index, res_index_in_chain)
        if key not in flexible_res_keys:
            pos = positions[atom.index]
            restraint.addParticle(
                atom.index,
                [pos.x, pos.y, pos.z] * unit.nanometer,
            )
            n_restrained += 1
    system.addForce(restraint)

    print(f"Restrained (core) atoms: {n_restrained}")
    print(f"Free (built-tail) atoms: {pdb.topology.getNumAtoms() - n_restrained}")
    if n_restrained == 0:
        print(
            "WARNING: 0 atoms restrained — that means either added_residues.json "
            "is empty/misread, or the restraint condition is broken again. "
            "Do not proceed with an unrestrained core; investigate before running."
        )

    # Diagnostic: which FORCE TYPE is actually responsible for the
    # catastrophic energy, not just which atom. Individual bad-looking
    # XML parameters can be red herrings if their magnitude is too small
    # to matter (learned that the hard way) — this settles it directly by
    # decomposing the total potential energy by force group. MUST happen
    # before the Simulation/Context below is created.
    class_to_group = {}
    group_names = {}
    next_group = 0
    for force in system.getForces():
        cls = force.__class__.__name__
        if cls not in class_to_group:
            class_to_group[cls] = next_group
            group_names[next_group] = cls
            next_group += 1
        force.setForceGroup(class_to_group[cls])
    print("Force groups:", group_names)
    import sys
    sys.stdout.flush()

    integrator = LangevinMiddleIntegrator(
        298 * unit.kelvin,
        5 / unit.picosecond,   # restored to a standard, well-damped friction —
                                # 0.1/ps was unusually low and under-damped for
                                # a just-minimized, still-strained structure
        1 * unit.femtosecond,
    )
    platform = Platform.getPlatformByName("CUDA")
    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    print("Minimizing...")
    import sys
    sys.stdout.flush()
    for i in range(0, 10001, 1000):
        simulation.minimizeEnergy(maxIterations=1000)
        state = simulation.context.getState(getEnergy=True)
        print(f"  {i:5d} steps: potential energy = {state.getPotentialEnergy()}")
        sys.stdout.flush()

    # Per-force-group energy breakdown — tells us WHICH term (Bond, Angle,
    # CustomTorsion/improper, NonbondedForce, restraint, ...) is actually
    # carrying the catastrophic energy, rather than guessing from
    # individual XML parameter values.
    print("Per-force-group potential energy after minimization:")
    sys.stdout.flush()
    for group_idx, name in group_names.items():
        e = simulation.context.getState(getEnergy=True, groups={group_idx}).getPotentialEnergy()
        print(f"  group {group_idx} ({name}): {e}")
        sys.stdout.flush()

    # Diagnostic: a minimizer can report a flat, converged-looking energy
    # across the WHOLE system while one atom is still sitting in a
    # genuinely pathological, near-singular local contact — that's
    # invisible in the aggregate energy/gradient metric but explodes the
    # instant real dynamics evaluates the force there. Report the worst
    # offenders now, before stepping, rather than after another crash.
    #
    # IMPORTANT: rank by force WITHIN EACH GROUP SEPARATELY, not just
    # total force summed across all groups — HarmonicAngleForce carries
    # by far the largest total energy (see per-group breakdown above),
    # but the atom with the highest TOTAL force (all groups combined)
    # is not necessarily the atom driving that specific group's energy;
    # it could be dominated by nonbonded/GB instead. Find the actual
    # angle-force outliers directly.
    import numpy as np

    angle_group_idx = None
    for idx, name in group_names.items():
        if name == "HarmonicAngleForce":
            angle_group_idx = idx
            break

    if angle_group_idx is not None:
        angle_force_state = simulation.context.getState(getForces=True, groups={angle_group_idx})
        angle_forces = angle_force_state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        angle_force_mags = np.linalg.norm(angle_forces, axis=1)
        worst_angle_idx = np.argsort(angle_force_mags)[::-1][:15]
        atoms_list = list(pdb.topology.atoms())
        print("Top 15 atoms by HarmonicAngleForce-ONLY residual force:")
        for idx in worst_angle_idx:
            atom = atoms_list[idx]
            print(
                f"  atom {idx}: {atom.residue.chain.id} {atom.residue.name}"
                f"{atom.residue.id} {atom.name}  |F_angle| = {angle_force_mags[idx]:.1f} kJ/mol/nm"
            )
        sys.stdout.flush()

    force_state = simulation.context.getState(getForces=True)
    forces = force_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    force_mags = np.linalg.norm(forces, axis=1)
    worst_idx = np.argsort(force_mags)[::-1][:10]
    atoms_list = list(pdb.topology.atoms())
    print("Top 10 atoms by residual force magnitude after minimization:")
    sys.stdout.flush()
    for idx in worst_idx:
        atom = atoms_list[idx]
        print(
            f"  atom {idx}: {atom.residue.chain.id} {atom.residue.name}"
            f"{atom.residue.id} {atom.name}  |F| = {force_mags[idx]:.1f} kJ/mol/nm"
        )
        sys.stdout.flush()
    max_force = force_mags.max()
    if max_force > 100000.0:
        print(
            f"WARNING: max residual force ({max_force:.1f}) is almost certainly "
            f"going to blow up into NaN once dynamics starts. Consider "
            f"identifying/fixing the underlying clash (likely two different "
            f"chains' independently-built extended tails overlapping each "
            f"other — PDBFixer does not check across chains) before proceeding."
        )

    # Explicitly thermalize before dynamics rather than relying on the
    # zero-velocity default + slow Langevin drift toward 298 K.
    simulation.context.setVelocitiesToTemperature(298 * unit.kelvin)

    print("Running short implicit-solvent collapse (50 ps at 1 fs)...")
    simulation.step(50000)  # 50 ps at 1 fs

    state = simulation.context.getState(getPositions=True)
    with open(out_pdb, "w") as f:
        app.PDBFile.writeFile(pdb.topology, state.getPositions(), f, keepIds=True)

    print(f"Compacted PDB written to: {out_pdb}")
    print(
        "NOTE: point paths.fixed_pdb (or build_system.py's input) at "
        "this file instead of fixed.pdb if you want to solvate the "
        "compacted geometry."
    )


if __name__ == "__main__":
    main()