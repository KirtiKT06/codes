"""
Stage 4 — minimize, equilibrate, and run production MD on the solvated
system built by build_system.py.

Equilibration restrains the ordered histone-fold core (everything PDBFixer
did NOT add — see 02_added_residues.json) to its crystallographic position
while letting the newly-built tails move freely, so the tails can relax
into the solvent environment without the core drifting away from the
experimental structure under the new force field. Restraints are optionally
tapered off over NPT before production starts unrestrained.

Also writes a per-force-type energy breakdown (bond, angle, dihedral,
nonbonded, CMAP, restraint, ...) to a .dat file every
simulation.energy_report_interval_steps, and mirrors normal progress +
energy-breakdown output continuously to the terminal for the entire run
(NVT, NPT, and production — not just NVT, which the previous version
silently dropped after reporters.clear()).

Reads:  paths.system_xml, paths.solvated_pdb, paths.added_residues_json
Writes: paths.state_minimized, paths.state_equilibrated,
        paths.production_dcd, paths.production_log, paths.checkpoint,
        paths.energy_breakdown_dat
"""
import json
import sys

from openmm import (
    app,
    unit,
    XmlSerializer,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    CustomExternalForce,
    Platform,
)

from config import load_config, outpath


def build_core_restraint_force(topology, positions, added_residues, k):
    """Harmonic restraint on every atom whose residue was NOT built by
    PDBFixer (i.e. the ordered, crystallographically-resolved core)."""
    by_chain = {}
    for a in added_residues:
        by_chain.setdefault(a["chain_index"], []).append(a)
    flexible_res_keys = set()
    for chain_index, entries in by_chain.items():
        entries.sort(key=lambda e: (e["insert_after_residue_index"], e["offset"]))
        for e in entries:
            new_res_index = e["insert_after_residue_index"] + 1 + e["offset"]
            flexible_res_keys.add((chain_index, new_res_index))

    force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    chains = list(topology.chains())
    n_restrained = 0
    for atom in topology.atoms():
        if atom.residue.chain.id not in {c.id for c in chains}:
            continue
        if atom.residue.name in ("HOH", "TIP3", "SOD", "CLA", "NA", "CL", "WAT"):
            continue
        chain_index = chains.index(atom.residue.chain)
        res_index_in_chain = list(atom.residue.chain.residues()).index(atom.residue)
        key = (chain_index, res_index_in_chain)
        if key not in flexible_res_keys:
            pos = positions[atom.index]
            force.addParticle(atom.index, [pos.x, pos.y, pos.z] * unit.nanometer)
            n_restrained += 1

    return force, n_restrained


def assign_force_groups(system):
    """Assign every Force in the System an OpenMM force group, grouped by
    class name (so e.g. all HarmonicBondForce instances share a group and
    their energies sum together; CustomExternalForce restraints get their
    own group, separate from real bonded/nonbonded terms).

    MUST be called before the Context/Simulation is created — OpenMM
    caches force group assignment at Context creation, so setting groups
    afterward requires context.reinitialize(preserveState=True) to take
    effect (this matters for the barostat, which gets added later — see
    main()).

    OpenMM allows at most 32 force groups (0-31); this raises if your
    system has more than 32 distinct Force subclasses, which shouldn't
    happen for a standard CHARMM36m protein+water+ions system (typically
    ~6-9: HarmonicBond, HarmonicAngle, PeriodicTorsion, CMAP, CustomTorsion
    /improper, NonbondedForce, CMMotionRemover, plus your restraint).

    Returns {group_index: force_class_name}, in order of first appearance.
    """
    class_to_group = {}
    group_names = {}
    next_group = 0
    for force in system.getForces():
        cls = force.__class__.__name__
        if cls not in class_to_group:
            if next_group > 31:
                raise RuntimeError(
                    "System has more than 32 distinct Force types — can't "
                    "assign unique OpenMM force groups (hard limit of 32)."
                )
            class_to_group[cls] = next_group
            group_names[next_group] = cls
            next_group += 1
        force.setForceGroup(class_to_group[cls])
    return group_names


class ForceGroupReporter:
    """Custom OpenMM reporter: writes step, time, per-force-type potential
    energy (one column per group from assign_force_groups), and total
    potential energy to a tab-separated .dat file every `interval` steps.
    Also mirrors the same line to stdout if also_stdout=True, so you get
    live energy-breakdown output in the terminal alongside the .dat file.
    """

    def __init__(self, file_path, interval, group_names, also_stdout=True):
        self._interval = interval
        self._group_names = group_names
        self._also_stdout = also_stdout
        self._out = open(file_path, "w")
        header = (
            "#Step\tTime_ps\t"
            + "\t".join(f"{name}_kJ/mol" for name in group_names.values())
            + "\tTotal_kJ/mol"
        )
        self._out.write(header + "\n")
        self._out.flush()
        if self._also_stdout:
            print(header)
            sys.stdout.flush()

    def describeNextReport(self, simulation):
        steps_left = self._interval - simulation.currentStep % self._interval
        # (steps_until_next_report, needs_positions, needs_velocities,
        #  needs_forces, needs_energy, wrap_periodic_positions)
        return (steps_left, False, False, False, True, None)

    def report(self, simulation, state):
        context = simulation.context
        step = simulation.currentStep
        time_ps = state.getTime().value_in_unit(unit.picosecond)

        energies = []
        total = 0.0
        for group_idx in self._group_names:
            e_val = (
                context.getState(getEnergy=True, groups={group_idx})
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
            energies.append(e_val)
            total += e_val

        line = (
            f"{step}\t{time_ps:.3f}\t"
            + "\t".join(f"{e:.4f}" for e in energies)
            + f"\t{total:.4f}"
        )
        self._out.write(line + "\n")
        self._out.flush()
        if self._also_stdout:
            print(line)
            sys.stdout.flush()

    def close(self):
        self._out.close()


def main():
    cfg = load_config()
    sim_cfg = cfg["simulation"]
    restr_cfg = cfg["restraints"]
    plat_cfg = cfg["platform"]

    solvated_pdb = outpath(cfg, "solvated_pdb")
    system_xml = outpath(cfg, "system_xml")
    added_json = outpath(cfg, "added_residues_json")

    pdb = app.PDBFile(solvated_pdb)
    with open(system_xml) as f:
        system = XmlSerializer.deserialize(f.read())

    with open(added_json) as f:
        added_residues = json.load(f)

    platform = Platform.getPlatformByName(plat_cfg["name"])
    platform_properties = {"Precision": plat_cfg["precision"]} if plat_cfg["name"] in ("CUDA", "OpenCL") else {}

    restraint_force_index = None
    if restr_cfg.get("restrain_core_during_equilibration", True) and added_residues:
        k = restr_cfg["core_force_constant_kj_per_mol_per_nm2"] * unit.kilojoule_per_mole / unit.nanometer**2
        force, n_restrained = build_core_restraint_force(pdb.topology, pdb.positions, added_residues, k)
        restraint_force_index = system.addForce(force)
        print(f"Core restraint applied to {n_restrained} atoms.")
    else:
        print("No core restraint applied (either disabled or no added residues on record).")

    # Force groups MUST be assigned before the Simulation/Context is
    # created — see assign_force_groups() docstring.
    group_names = assign_force_groups(system)
    print("Force groups for energy breakdown:")
    for idx, name in group_names.items():
        print(f"  group {idx}: {name}")

    integrator = LangevinMiddleIntegrator(
        sim_cfg["temperature_K"] * unit.kelvin,
        sim_cfg["friction_per_ps"] / unit.picosecond,
        sim_cfg["timestep_fs"] * unit.femtosecond,
    )

    simulation = app.Simulation(pdb.topology, system, integrator, platform, platform_properties)
    simulation.context.setPositions(pdb.positions)
    if pdb.topology.getPeriodicBoxVectors() is not None:
        simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    energy_interval = sim_cfg.get("energy_report_interval_steps", sim_cfg["equilibration_report_interval_steps"])
    energy_reporter = ForceGroupReporter(
        outpath(cfg, "energy_breakdown_dat"),
        energy_interval,
        group_names,
        also_stdout=True,
    )
    # energy_reporter stays in simulation.reporters for the entire run
    # (NVT + NPT + production) — never removed, unlike the plain
    # StateDataReporter instances below which are phase-specific.
    simulation.reporters.append(energy_reporter)

    # ---- Minimize ----
    print("Minimizing...")
    simulation.minimizeEnergy(maxIterations=sim_cfg["minimize_max_iterations"])
    save_state(simulation, outpath(cfg, "state_minimized"))
    state = simulation.context.getState(getEnergy=True)
    print("Potential energy after minimization:", state.getPotentialEnergy())

    # Diagnostic: find the atom under the largest residual force after
    # minimization. A "successful" minimization (finite energy, ran to
    # completion) can still leave one or two atoms in a bad steric clash
    # if the energy landscape near that clash is too steep for the
    # minimizer to fully resolve in a fixed iteration budget — those
    # atoms are exactly what blows up into NaN a few steps into dynamics.
    # Report the worst offenders BEFORE starting NVT so we know in advance
    # rather than after a crash.
    force_state = simulation.context.getState(getForces=True)
    forces = force_state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    import numpy as np
    force_mags = np.linalg.norm(forces, axis=1)
    worst_idx = np.argsort(force_mags)[::-1][:10]
    atoms = list(pdb.topology.atoms())
    print("Top 10 atoms by residual force magnitude after minimization:")
    for idx in worst_idx:
        atom = atoms[idx]
        print(
            f"  atom {idx}: {atom.residue.chain.id} {atom.residue.name}"
            f"{atom.residue.id} {atom.name}  |F| = {force_mags[idx]:.1f} kJ/mol/nm"
        )
    max_force = force_mags.max()
    force_warn_threshold = sim_cfg.get("max_force_warn_kj_per_mol_per_nm", 100000.0)
    if max_force > force_warn_threshold:
        print(
            f"WARNING: max residual force ({max_force:.1f}) exceeds "
            f"{force_warn_threshold} kJ/mol/nm — this atom is likely to "
            f"blow up into NaN once dynamics starts. Consider fixing the "
            f"underlying clash (see atom listed above) before proceeding, "
            f"rather than just increasing minimize_max_iterations."
        )

    # ---- Equilibrate: NVT ----
    simulation.context.setVelocitiesToTemperature(sim_cfg["temperature_K"] * unit.kelvin)
    nvt_steps = int(sim_cfg["equilibration_nvt_ps"] * 1000 / sim_cfg["timestep_fs"])
    print(f"Equilibrating NVT for {sim_cfg['equilibration_nvt_ps']} ps ({nvt_steps} steps)...")

    nvt_stdout_reporter = app.StateDataReporter(
        sys.stdout,
        sim_cfg["equilibration_report_interval_steps"],
        step=True, temperature=True, potentialEnergy=True, volume=True,
    )
    simulation.reporters.append(nvt_stdout_reporter)
    simulation.step(nvt_steps)
    simulation.reporters.remove(nvt_stdout_reporter)

    # ---- Equilibrate: NPT ----
    barostat = MonteCarloBarostat(
        sim_cfg["pressure_bar"] * unit.bar,
        sim_cfg["temperature_K"] * unit.kelvin,
        sim_cfg["barostat_interval_steps"],
    )
    barostat_index = system.addForce(barostat)
    # Barostat contributes no potential energy of its own (it's a Monte
    # Carlo volume move, not a potential term), so leaving it in the
    # default group 0 doesn't corrupt that group's reported energy — no
    # need to extend group_names for it.
    simulation.context.reinitialize(preserveState=True)

    npt_steps = int(sim_cfg["equilibration_npt_ps"] * 1000 / sim_cfg["timestep_fs"])
    print(f"Equilibrating NPT for {sim_cfg['equilibration_npt_ps']} ps ({npt_steps} steps)...")

    npt_stdout_reporter = app.StateDataReporter(
        sys.stdout,
        sim_cfg["equilibration_report_interval_steps"],
        step=True, temperature=True, potentialEnergy=True, volume=True,
    )
    simulation.reporters.append(npt_stdout_reporter)

    if restraint_force_index is not None and restr_cfg.get("taper_restraints_over_npt", True):
        n_stages = 10
        steps_per_stage = max(npt_steps // n_stages, 1)
        k0 = restr_cfg["core_force_constant_kj_per_mol_per_nm2"]
        for stage in range(n_stages):
            k_now = k0 * (1 - stage / n_stages)
            simulation.context.setParameter("k", k_now * unit.kilojoule_per_mole / unit.nanometer**2)
            simulation.step(steps_per_stage)
        simulation.context.setParameter("k", 0.0)
    else:
        simulation.step(npt_steps)

    simulation.reporters.remove(npt_stdout_reporter)
    save_state(simulation, outpath(cfg, "state_equilibrated"))

    state = simulation.context.getState()
    print("Box volume:", state.getPeriodicBoxVolume())

    # ---- Production ----
    prod_steps = int(sim_cfg["production_ns"] * 1e6 / sim_cfg["timestep_fs"])
    print(f"Running production for {sim_cfg['production_ns']} ns ({prod_steps} steps)...")

    # NOTE: energy_reporter is intentionally NOT removed here — it keeps
    # writing the .dat file (and stdout) through production too. Only the
    # phase-specific plain progress reporter is (re)added.
    prod_stdout_reporter = app.StateDataReporter(
        sys.stdout,
        sim_cfg["production_report_interval_steps"],
        step=True, time=True, temperature=True, potentialEnergy=True,
        volume=True, speed=True,
    )
    simulation.reporters.append(prod_stdout_reporter)
    simulation.reporters.append(
        app.DCDReporter(outpath(cfg, "production_dcd"), sim_cfg["production_dcd_interval_steps"])
    )
    simulation.reporters.append(
        app.StateDataReporter(
            outpath(cfg, "production_log"),
            sim_cfg["production_report_interval_steps"],
            step=True, time=True, temperature=True, potentialEnergy=True,
            volume=True, speed=True,
        )
    )
    simulation.reporters.append(
        app.CheckpointReporter(outpath(cfg, "checkpoint"), sim_cfg["production_checkpoint_interval_steps"])
    )
    simulation.step(prod_steps)

    energy_reporter.close()
    print("Production complete.")


def save_state(simulation, path):
    state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    with open(path, "w") as f:
        f.write(XmlSerializer.serialize(state))
    print(f"State saved to: {path}")


if __name__ == "__main__":
    main()