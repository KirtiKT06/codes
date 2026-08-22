# Nucleosome octamer (2CV5) protein-only MD pipeline

Run in order, all reading/writing through `input.json`:

```
python3 01_strip_dna.py         # remove DNA chains I/J, keep histone chains A-H
python3 02_fix_structure.py     # PDBFixer: build missing tail residues + atoms
python3 02b_compact_tails.py    # implicit-solvent relaxation of the built tails
python3 03_build_system.py      # explicit TIP3P solvation + ionization + System
python3 04_run_md.py            # minimize -> NVT -> NPT (restrained) -> production
```

## Things found while testing this against your actual 2CV5.pdb

**1. `structure.protein_chain_ids` / `dna_chain_ids` in `input.json` are already
set correctly for 2CV5** — protein: A-H, DNA: I/J — confirmed against the file.

**2. Stage 1 uses a text-level PDB filter, not MDAnalysis.** An MDAnalysis
rewrite silently drops `SEQRES` records. PDBFixer's `findMissingResidues()`
depends entirely on `SEQRES` to know the true full sequence — without it, it
found **zero** missing residues (it can only "see" what's already there).
The current script preserves `SEQRES` for the kept chains, which correctly
surfaces the 12 missing stretches (the tails) across the 8 chains.

**3. PDBFixer builds long missing termini almost fully extended.** For chain
A alone (39 missing N-terminal residues) this produced a ~19 x 6 x 15 nm
bounding box — CA-CA spacing of ~0.4-0.5 nm, i.e. close to a straight line,
because `addMissingAtoms()` is pure geometry with no energetic collapse.
Solvating that directly would waste enormous box volume (and real compute)
on an artificial, physically implausible starting conformation. `02b_compact_tails.py`
fixes this: it restrains the ordered core and lets the built tails relax
under implicit solvent (GBSA-OBC2) before you ever build the explicit water
box. `03_build_system.py` reads `structure.use_compacted_tails` (default
`true`) to solvate the compacted geometry instead of the raw PDBFixer output.

**4. `forcefield.files` in `input.json` are placeholders — read the comment
in there.** OpenMM's bundled `charmm36.xml` is plain CHARMM36
(`toppar_c36_aug15`), **not** CHARMM36m. You specifically wanted 36m (correct
choice, since it fixes backbone/CMAP sampling for disordered regions like
these tails), so you need to get the real thing: run the protein-only PDB
through CHARMM-GUI (PDB Reader & Manipulator or Solution Builder), select
CHARMM36m + OpenMM as the output, and point `forcefield.files` at the
resulting XMLs. Everything downstream (Stage 3/4) is force-field-agnostic —
it just reads whatever's in that list.

## What's verified vs. not

Verified against your real file in a sandboxed test run:
- Stage 1 (chain filtering + SEQRES preservation) — full run, correct output.
- Stage 2 (PDBFixer missing-residue detection + build) — full run on all 8
  chains for detection; full atom-building tested on chain A alone (works,
  just slow — budget 10-20+ min for all 8 chains on real hardware, this
  sandbox's CPU couldn't finish it in the time available here).
- Stage 2b logic — standard OpenMM restrained-implicit-solvent pattern,
  syntax-checked and dry-run-started successfully; not run to completion
  here (sandbox CPU-only, no GPU — this step will be fast on your 4070 Ti).

Not run end-to-end here (sandbox compute-bound, not a script issue):
Stage 3 (explicit solvation — box will be large: nucleosome octamer even
protein-only is ~750 residues) and Stage 4. Both use standard, well-worn
OpenMM APIs (`Modeller.addSolvent`, `ForceField.createSystem`,
`LangevinMiddleIntegrator`, `MonteCarloBarostat`) — run these on your own
machine/cluster where solvating ~750 residues + water will actually be fast.

## Restraint bookkeeping

`02_fix_structure.py` writes `02_added_residues.json`, listing every
residue PDBFixer built. `02b_compact_tails.py` and `04_run_md.py` both read
this to know which atoms are "core" (restrain to crystal position) vs.
"built tail" (leave free). `04_run_md.py` tapers the core restraint to zero
over the NPT equilibration stage so production runs fully unrestrained.

## Before running for real

- Swap in genuine CHARMM36m force field files (see point 4 above).
- Double check `simulation.hydrogen_mass_amu: 1.5` + `timestep_fs: 4.0` —
  this is HMR (hydrogen mass repartitioning) enabling a 4 fs step; drop both
  to `1.0`/`2.0` if you'd rather not use HMR.
- `production_ns: 200` in `input.json` is a placeholder — set it to
  whatever your actual target length is before launching.

# Nucleosome octamer (2CV5) protein-only MD pipeline

Run in order, all reading/writing through `input.json`:

```
python3 01_strip_dna.py         # remove DNA chains I/J, keep histone chains A-H
python3 02_fix_structure.py     # PDBFixer: build missing tail residues + atoms
python3 02b_compact_tails.py    # implicit-solvent relaxation of the built tails
python3 03_build_system.py      # explicit TIP3P solvation + ionization + System
python3 03b_add_extra_ions.py   # optional: add Mg2+ or other extra salts (see below)
python3 04_run_md.py            # minimize -> NVT -> NPT (restrained) -> production
```

## Adding extra salts (Mg2+, etc.) beyond NaCl

`03_build_system.py` / `Modeller.addSolvent()` only neutralizes and sets
ionic strength with **monovalent** ions — that's a hardcoded OpenMM
limitation, not a config option. To add Mg2+ or other divalent (or
otherwise non-monovalent) salts, run `03b_add_extra_ions.py` after Stage 3.
It reads `solvation.extra_salts` in `input.json`:

```json
"extra_salts": [
  {"cation_resname": "MG", "anion_resname": "CLA", "concentration_M": 0.005}
]
```

It replaces random water molecules with the requested ions at the target
concentration, charge-balancing each salt internally (e.g. 1 Mg2+ + 2 Cl-
per MgCl2 formula unit), then rewrites `solvated_pdb` and re-serializes
`system_xml` so `04_run_md.py` needs no changes. Verified end-to-end in a
fast standalone test (pure water box, MgCl2 at both low and high
concentration — correct counts, correct charge balance, System builds
cleanly).

**CHARMM36's standard ion set** (checked directly against the bundled
parameter file) covers Li+, Na+, Mg2+, K+, Ca2+, Rb+, Cs+, Ba2+, Zn2+,
Cd2+, Cl-. It does **not** include Co2+ — if you need cobalt specifically,
you'll need to source and validate your own nonbonded parameters and add a
residue template; `03b_add_extra_ions.py` will raise a clear error rather
than silently using a wrong/fabricated parameter set.

## Things found while testing this against your actual 2CV5.pdb

**1. `structure.protein_chain_ids` / `dna_chain_ids` in `input.json` are already
set correctly for 2CV5** — protein: A-H, DNA: I/J — confirmed against the file.

**2. Stage 1 uses a text-level PDB filter, not MDAnalysis.** An MDAnalysis
rewrite silently drops `SEQRES` records. PDBFixer's `findMissingResidues()`
depends entirely on `SEQRES` to know the true full sequence — without it, it
found **zero** missing residues (it can only "see" what's already there).
The current script preserves `SEQRES` for the kept chains, which correctly
surfaces the 12 missing stretches (the tails) across the 8 chains.

**3. PDBFixer builds long missing termini almost fully extended.** For chain
A alone (39 missing N-terminal residues) this produced a ~19 x 6 x 15 nm
bounding box — CA-CA spacing of ~0.4-0.5 nm, i.e. close to a straight line,
because `addMissingAtoms()` is pure geometry with no energetic collapse.
Solvating that directly would waste enormous box volume (and real compute)
on an artificial, physically implausible starting conformation. `02b_compact_tails.py`
fixes this: it restrains the ordered core and lets the built tails relax
under implicit solvent (GBSA-OBC2) before you ever build the explicit water
box. `03_build_system.py` reads `structure.use_compacted_tails` (default
`true`) to solvate the compacted geometry instead of the raw PDBFixer output.

**4. `forcefield.files` in `input.json` are placeholders — read the comment
in there.** OpenMM's bundled `charmm36.xml` is plain CHARMM36
(`toppar_c36_aug15`), **not** CHARMM36m. You specifically wanted 36m (correct
choice, since it fixes backbone/CMAP sampling for disordered regions like
these tails), so you need to get the real thing: run the protein-only PDB
through CHARMM-GUI (PDB Reader & Manipulator or Solution Builder), select
CHARMM36m + OpenMM as the output, and point `forcefield.files` at the
resulting XMLs. Everything downstream (Stage 3/4) is force-field-agnostic —
it just reads whatever's in that list.

## What's verified vs. not

Verified against your real file in a sandboxed test run:
- Stage 1 (chain filtering + SEQRES preservation) — full run, correct output.
- Stage 2 (PDBFixer missing-residue detection + build) — full run on all 8
  chains for detection; full atom-building tested on chain A alone (works,
  just slow — budget 10-20+ min for all 8 chains on real hardware, this
  sandbox's CPU couldn't finish it in the time available here).
- Stage 2b logic — standard OpenMM restrained-implicit-solvent pattern,
  syntax-checked and dry-run-started successfully; not run to completion
  here (sandbox CPU-only, no GPU — this step will be fast on your 4070 Ti).

Not run end-to-end here (sandbox compute-bound, not a script issue):
Stage 3 (explicit solvation — box will be large: nucleosome octamer even
protein-only is ~750 residues) and Stage 4. Both use standard, well-worn
OpenMM APIs (`Modeller.addSolvent`, `ForceField.createSystem`,
`LangevinMiddleIntegrator`, `MonteCarloBarostat`) — run these on your own
machine/cluster where solvating ~750 residues + water will actually be fast.

## Restraint bookkeeping

`02_fix_structure.py` writes `02_added_residues.json`, listing every
residue PDBFixer built. `02b_compact_tails.py` and `04_run_md.py` both read
this to know which atoms are "core" (restrain to crystal position) vs.
"built tail" (leave free). `04_run_md.py` tapers the core restraint to zero
over the NPT equilibration stage so production runs fully unrestrained.

## Before running for real

- Swap in genuine CHARMM36m force field files (see point 4 above).
- Double check `simulation.hydrogen_mass_amu: 1.5` + `timestep_fs: 4.0` —
  this is HMR (hydrogen mass repartitioning) enabling a 4 fs step; drop both
  to `1.0`/`2.0` if you'd rather not use HMR.
- `production_ns: 200` in `input.json` is a placeholder — set it to
  whatever your actual target length is before launching.