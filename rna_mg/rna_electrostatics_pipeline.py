#!/usr/bin/env python3
"""
RNA per-residue electrostatic potential label-generation pipeline.

Pipeline:
    1. Non-redundant RNA structure list (BGSU RNA 3D Hub representative set)
    2. Download PDB structures from RCSB
    3. Strip crystallographic ions/waters (implicit-solvent convention --
       resolved ions are the thing we may eventually want to predict, not
       an input to the PB solve)
    4. PDB2PQR -> AMBER charges/radii (full RNA nucleotide support confirmed)
    5. APBS -> nonlinear (full sinh) Poisson-Boltzmann solve by default
       (multigrid, with focusing) -> potential map (.dx). Linearized PB
       is available as an explicit opt-out (ApbsSettings.nonlinear=False)
       for structures where the Newton solver fails to converge.
    6. Per-residue electrostatic potential = average of the potential field
       over all grid points within a hard 8 A cutoff of each residue's
       representative atom (C1', falling back to P or the residue centroid)

Every component of this pipeline (pdb2pqr RNA charge assignment, apbs
multigrid PB solve, gridData .dx interpolation, KDTree-based cutoff
averaging) has been tested end-to-end against a real APBS run in the
development sandbox. The RCSB download and BGSU list steps need outbound
network access to rcsb.org / rna.bgsu.edu, which was not available in that
sandbox -- run this on a machine with normal internet access.

Requirements:
    pip install apbs-binary pdb2pqr GridDataFormats scipy numpy pandas requests gemmi

Usage:
    python rna_electrostatics_pipeline.py --pdb-ids 1EHZ 4V9F ... --outdir ./rna_labels
    python rna_electrostatics_pipeline.py --nonredundant --outdir ./rna_labels
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rna_pb_pipeline")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

RNA_RESNAMES = {"A", "C", "G", "U", "RA", "RC", "RG", "RU",
                "A5", "A3", "C5", "C3", "G5", "G3", "U5", "U3",
                "ADE", "CYT", "GUA", "URA"}

# pdb2pqr's AMBER force field (AMBER.DAT) has ZERO charge parameters for
# any modified nucleotide -- confirmed by grepping it directly, not
# assumed. So the naive fix of just not-stripping modified residues in
# strip_ions_and_waters() would not work: pdb2pqr would fail on the
# unrecognized residue name instead of succeeding. What actually works
# (and is what this table is for) is relabeling a modified nucleotide to
# its unmodified parent base before pdb2pqr sees it. pdb2pqr already
# tolerates atoms it doesn't recognize on a known residue (confirmed in
# biomolecule.py: it warns "Extra atom X in residue! - Deleted this
# atom." and continues) rather than crashing, so the base-methylation /
# extra-ring-atom parts of a modified nucleotide get gracefully dropped
# once the residue itself is relabeled to something pdb2pqr recognizes.
#
# This is NOT physically exact -- a relabeled residue gets the *parent
# base's* AMBER charges, not the true (and generally unparameterized)
# charge redistribution caused by the modification -- but it keeps the
# backbone intact through the modification site, which matters more for
# the PB solve (backbone continuity, correct overall charge count) than
# getting one methyl group's exact partial charges right. Flag column
# `had_modified_nucleotides` in the output lets you exclude these
# structures from training data later if that approximation isn't good
# enough for your purposes.
#
# Covers common tRNA/rRNA modifications; NOT exhaustive. Anything not in
# this table falls back to the previous conservative behavior (stripped
# entirely, same as an ion/water) rather than silently guessing.
MODIFIED_NUCLEOTIDE_PARENT = {
    "PSU": "U",   # pseudouridine
    "1MA": "A",   # 1-methyladenosine
    "2MG": "G",   # N2-methylguanosine
    "M2G": "G",   # N2,N2-dimethylguanosine
    "7MG": "G",   # 7-methylguanosine
    "OMG": "G",   # 2'-O-methylguanosine
    "OMC": "C",   # 2'-O-methylcytidine
    "OMU": "U",   # 2'-O-methyluridine
    "5MC": "C",   # 5-methylcytidine
    "5MU": "U",   # 5-methyluridine (ribothymidine)
    "H2U": "U",   # dihydrouridine
    "4SU": "U",   # 4-thiouridine
    "YG":  "G",   # wybutosine (approximate -- large hypermodified base)
    "YYG": "G",   # wybutosine variant
    "UR3": "U",   # 3-methyluridine
    "A2M": "A",   # 2'-O-methyladenosine
    "MA6": "A",   # N6-methyladenosine
    "6MA": "A",   # N6-methyladenosine (alt code)
    "I":   "A",   # inosine -- deaminated adenosine; base-pairs closer to G
                   # (wobble) but relabeling to A keeps backbone/glycosidic
                   # geometry consistent with its biosynthetic parent.
                   # Flagged like everything else here -- reconsider per use case.
}

ION_RESNAMES = {
    "MG", "NA", "K", "CL", "CA", "ZN", "MN", "CO", "NI", "CD", "CU",
    "SR", "BA", "CS", "LI", "RB", "FE", "IOD", "BR",
}

RESIDUE_CUTOFF_ANGSTROM = 8.0


@dataclasses.dataclass
class ApbsSettings:
    ionic_strength_M: float = 0.150   # monovalent background salt (adjust per RNA / experiment)
    pdie: float = 1.0                 # solute dielectric
    sdie: float = 80.0                # solvent dielectric
    temp_K: float = 300.0
    srad: float = 1.4                 # solvent probe radius
    fine_grid_spacing: float = 0.5    # target A/grid point on the fine grid
    fine_padding: float = 15.0        # A of padding around molecule, fine grid
    coarse_padding: float = 40.0      # A of padding around molecule, coarse grid
    max_dim: int = 225                # safety cap on grid points per axis
    nonlinear: bool = True            # solve the full nonlinear (sinh) PBE, not the
                                       # small-signal linearization. Default True:
                                       # RNA's dense backbone charge routinely pushes
                                       # local |e*phi/kT| > 1 near phosphates, where
                                       # the linear approximation is least trustworthy.
                                       # Set False as a fallback if the Newton solver
                                       # fails to converge on a pathological geometry.


# ----------------------------------------------------------------------
# Step 1: non-redundant RNA structure list
# ----------------------------------------------------------------------

def parse_bgsu_nonredundant_csv(csv_path: Path) -> list[str]:
    """
    Parse a BGSU RNA 3D Hub non-redundant list CSV, downloaded by hand from
    https://rna.bgsu.edu/rna3dhub/nrlist (Representative Sets of RNA 3D
    Structures -> pick a release and resolution cutoff -> download csv).

    File format (no header row), one row per sequence/structure
    equivalence class:
        col 1: class id, e.g. "NR_3.0_48063.1"
        col 2: representative member, e.g. "11DG|1|A" or, for a
               multi-chain complex, "11GI|1|C+11GI|1|B"
               (PDBID|model|chain, chains joined by "+")
        col 3: every member of the class (comma-separated), e.g. a class
               representing one rRNA solved hundreds of times will list
               all of those PDB IDs here -- this is the redundancy that
               col 2 collapses down to a single representative.

    Returns one 4-character PDB ID per equivalence class (the
    representative's PDB ID), i.e. the same non-redundant list the
    pipeline used to fetch live from BGSU's site.
    """
    import csv as csv_module

    pdb_ids = []
    with open(csv_path, newline="") as f:
        reader = csv_module.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            representative = row[1]
            # representative may be "PDBID|model|chain" or, for a
            # multi-chain complex, "PDBID|model|chainA+PDBID|model|chainB"
            first_member = representative.split("+")[0]
            pdb_id = first_member.split("|")[0].strip()
            if len(pdb_id) == 4:
                pdb_ids.append(pdb_id.upper())

    pdb_ids = sorted(set(pdb_ids))
    log.info("Parsed %d non-redundant RNA structure IDs from %s", len(pdb_ids), csv_path)
    return pdb_ids


def fetch_bgsu_nonredundant_list(resolution_cutoff: str = "3.0") -> list[str]:
    """
    DEPRECATED entry point: BGSU's live download URL/path has changed
    over the site's history and could not be verified from an
    environment with outbound access to rna.bgsu.edu. Download the CSV
    by hand instead (Representative Sets of RNA 3D Structures page ->
    pick a release + resolution -> download csv) and pass it to
    --nonredundant-csv / parse_bgsu_nonredundant_csv() below.
    """
    raise NotImplementedError(
        "Live BGSU download is unverified. Download the non-redundant list "
        "CSV by hand from https://rna.bgsu.edu/rna3dhub/nrlist and pass it "
        "via --nonredundant-csv instead."
    )


# ----------------------------------------------------------------------
# Step 2: download + clean structure
# ----------------------------------------------------------------------

def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    """
    Download a structure from RCSB, preferring legacy .pdb but falling
    back to mmCIF (converted to legacy PDB format via gemmi) if the
    legacy format isn't available.

    In the first real batch run of this pipeline, 2/10 non-redundant IDs
    (11DG, 11FV) 404'd on the legacy .pdb endpoint. This is a known,
    increasingly common situation: RCSB does not generate a legacy PDB
    file for every entry (the format has hard limits -- e.g. chain-ID
    and atom-count width -- that some structures exceed, and RCSB has
    also been moving toward mmCIF-only delivery for a growing share of
    entries). mmCIF is always available for a valid PDB ID, so that's the
    reliable fallback rather than treating a legacy-format 404 as "this
    structure doesn't exist."
    """
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{pdb_id.lower()}.pdb"
    if dest.exists():
        return dest

    pdb_url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    resp = requests.get(pdb_url, timeout=60)
    if resp.status_code == 200:
        dest.write_text(resp.text)
        return dest

    log.info("%s: legacy .pdb not available (HTTP %d), falling back to mmCIF",
              pdb_id, resp.status_code)
    cif_url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    cif_resp = requests.get(cif_url, timeout=60)
    cif_resp.raise_for_status()

    cif_path = out_dir / f"{pdb_id.lower()}.cif"
    cif_path.write_text(cif_resp.text)

    import gemmi
    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0:
        raise RuntimeError(f"{pdb_id}: mmCIF fallback parsed 0 models, cannot convert")
    st.setup_entities()
    st.write_pdb(str(dest))
    return dest


def strip_ions_and_waters(pdb_path: Path, out_path: Path) -> tuple[Path, int, bool]:
    """
    Remove crystallographic ions and waters before the PB solve
    (implicit-solvent convention: resolved ions/waters are not part of
    the solute for the PB solve -- and a resolved Mg2+ position is a
    candidate *output* of the downstream project, not an input).

    Modified nucleotides in MODIFIED_NUCLEOTIDE_PARENT are relabeled to
    their parent base rather than stripped -- see the comment on that
    table for why blind stripping breaks backbone continuity (and why
    just "not stripping" them doesn't work either, since pdb2pqr's AMBER
    tables have no parameters for the modified forms).

    Keeps ATOM records for RNA residues only (drops protein chains if the
    structure is a complex -- adjust RNA_RESNAMES / this filter if you
    want combined RNA-protein electrostatics later).

    Returns (cleaned_path, n_rna_atoms_kept, had_modified_nucleotides).
    n_rna_atoms_kept lets callers skip obviously-empty results (e.g. a
    protein-only PDB ID that slipped into the RNA list) rather than
    silently feeding them to pdb2pqr. had_modified_nucleotides flags
    structures where the relabeling approximation was used, so it can be
    propagated into the output CSV for later filtering.
    """
    kept = 0
    had_modified = False
    with open(pdb_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM", "TER", "END"):
                continue
            if record in ("ATOM", "HETATM"):
                resname = line[17:20].strip()
                if resname in MODIFIED_NUCLEOTIDE_PARENT:
                    parent = MODIFIED_NUCLEOTIDE_PARENT[resname]
                    line = line[:17] + f"{parent:>3}" + line[20:]
                    resname = parent
                    had_modified = True
                elif resname not in RNA_RESNAMES:
                    continue  # ions, waters, ligands, and unmapped modified residues
                # HETATM records for modified nucleotides get promoted to ATOM
                # once relabeled -- pdb2pqr expects standard polymer residues
                # as ATOM records, not HETATM.
                line = "ATOM  " + line[6:]
                kept += 1
            fout.write(line)
    return out_path, kept, had_modified

def renumber_pdb_atoms(pdb_path: Path) -> None:
    """
    Rewrite ATOM/HETATM serial numbers sequentially so the file
    contains standard integer PDB serials.

    This fixes mmCIF->PDB conversions that emit hybrid-36 serials
    (e.g. A0000), which pdb2pqr 3.7.1 cannot parse.
    """
    serial = 1
    out_lines = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                if serial > 99999:
                    raise RuntimeError(
                        f"{pdb_path}: still has >99999 atoms after filtering "
                        "and cannot be represented safely in legacy PDB format."
                    )

                line = (
                    line[:6]
                    + f"{serial:5d}"
                    + line[11:]
                )
                serial += 1

            out_lines.append(line)

    with open(pdb_path, "w") as f:
        f.writelines(out_lines)

# ----------------------------------------------------------------------
# Step 3: PDB2PQR (AMBER RNA charges/radii)
# ----------------------------------------------------------------------

def run_pdb2pqr(pdb_path: Path, pqr_path: Path) -> tuple[Path, bool]:
    """
    Assign AMBER partial charges and radii. Confirmed against pdb2pqr's
    own parameter tables: NA.xml defines full RNA topology (RA/RC/RG/RU,
    correct phosphate/ribose atom names) and AMBER.DAT carries real
    AMBER99 charges for every RNA atom.

    Some real crystal structures have missing/unresolved backbone atoms
    (common at chain termini, in modified nucleotides pdb2pqr doesn't
    fully recognize, or at low resolution). pdb2pqr's default path tries
    to reconstruct missing heavy atoms and then does a strict per-residue
    integral-charge check; if reconstruction can't fully complete a
    residue, that check fails with a hard crash ("non-integer charge...
    Giving up"), as seen for 10YZ and 11AO in the first real batch run.

    On that failure we retry once with --assign-only, which skips atom
    addition/debumping/optimization and just assigns charges/radii to
    the atoms actually present. This is a plausible recovery path but
    NOT verified against a real structure that hit this crash (no
    network access to fetch 10YZ/11AO in the environment this was
    written in) -- treat it as a best-effort retry, not a confirmed fix.
    Some structures are genuinely too incomplete for any charge
    assignment to be physically meaningful and should stay in
    failures.log rather than be forced through.

    Returns (pqr_path, used_assign_only_fallback).
    """
    base_cmd = ["pdb2pqr30", "--ff", "AMBER", "--ffout", "AMBER", "--drop-water"]

    result = subprocess.run(base_cmd + [str(pdb_path), str(pqr_path)],
                             capture_output=True, text=True)
    if result.returncode == 0 and pqr_path.exists():
        return pqr_path, False

    charge_failure_phrases = (
        "non-integer charge",
        "deviates by",           # e.g. "-3.19 deviates by 0.19 from integral..."
        "deviates from integral",
        "charge assignment failed",
    )
    error_text = result.stdout + result.stderr
    charge_failure = any(phrase in error_text for phrase in charge_failure_phrases)
    if not charge_failure:
        raise RuntimeError(
            f"pdb2pqr30 failed for {pdb_path}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )

    log.warning("%s: pdb2pqr hit a non-integer-charge crash (likely missing/unresolved "
                "atoms in the crystal structure); retrying with --assign-only", pdb_path.name)
    retry = subprocess.run(base_cmd + ["--assign-only", str(pdb_path), str(pqr_path)],
                            capture_output=True, text=True)
    if retry.returncode != 0 or not pqr_path.exists():
        raise RuntimeError(
            f"pdb2pqr30 failed for {pdb_path} (both default and --assign-only retry):\n"
            f"--- default attempt ---\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}\n"
            f"--- --assign-only retry ---\n{retry.stdout[-1000:]}\n{retry.stderr[-1000:]}"
        )
    log.info("%s: recovered via --assign-only (note: missing atoms were NOT reconstructed, "
              "so this PQR may be missing some heavy atoms present in the real molecule -- "
              "check n_grid_points / residue completeness before trusting labels from it)",
              pdb_path.name)
    return pqr_path, True


# ----------------------------------------------------------------------
# Step 4: APBS (linearized PB, multigrid with focusing)
# ----------------------------------------------------------------------

def _next_valid_dime(n: int) -> int:
    """
    APBS multigrid requires grid dimensions of the form c*2^(l+1) + 1.
    Round up to the nearest valid value using c=32 (a safe, commonly
    used choice), capped by ApbsSettings.max_dim by the caller.
    """
    candidates = [33, 65, 97, 129, 161, 193, 225, 257]
    for c in candidates:
        if c >= n:
            return c
    return candidates[-1]


def read_pqr_coords(pqr_path: Path) -> np.ndarray:
    """
    Atom coordinates for grid-sizing (build_apbs_input).

    NOTE: this used to duplicate the PQR line-parsing logic with a
    hardcoded column offset (parts[5:8]), which is only correct when
    pdb2pqr omits the chain-ID column (single-chain structures). For any
    real multi-chain structure, pdb2pqr emits an 11-field line
    (serial name resName CHAIN resSeq x y z q r) and that hardcoded
    offset silently read (resnum, x, y) as (x, y, z) instead -- wrong,
    and *silently* wrong (no crash), which is the dangerous kind of bug.
    Fixed by deriving coordinates from parse_pqr_residues() below, which
    already branches on field count correctly -- so there is now exactly
    one place PQR lines get parsed, not two implementations that can
    drift out of sync.
    """
    atoms = parse_pqr_residues(pqr_path)
    return atoms[["x", "y", "z"]].to_numpy()


def build_apbs_input(pqr_path: Path, work_dir: Path, settings: ApbsSettings,
                      pot_prefix: str = "potential") -> Path:
    coords = read_pqr_coords(pqr_path)
    extent = coords.max(axis=0) - coords.min(axis=0)
    center = coords.mean(axis=0)

    fine_len = extent + 2 * settings.fine_padding
    coarse_len = extent + 2 * settings.coarse_padding

    fine_dime = [
        min(_next_valid_dime(int(np.ceil(L / settings.fine_grid_spacing))), settings.max_dim)
        for L in fine_len
    ]
    # APBS wants one shared dime for both grids in mg-auto; use the max
    # per axis so the fine grid resolution target is respected.
    dime = fine_dime

    apbs_in = work_dir / "apbs.in"
    pbe_keyword = "npbe" if settings.nonlinear else "lpbe"
    apbs_in.write_text(f"""\
read
    mol pqr {pqr_path.name}
end
elec
    mg-auto
    dime {dime[0]} {dime[1]} {dime[2]}
    cglen {coarse_len[0]:.2f} {coarse_len[1]:.2f} {coarse_len[2]:.2f}
    fglen {fine_len[0]:.2f} {fine_len[1]:.2f} {fine_len[2]:.2f}
    cgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
    fgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
    mol 1
    {pbe_keyword}
    bcfl sdh
    pdie {settings.pdie}
    sdie {settings.sdie}
    chgm spl2
    srfm smol
    srad {settings.srad}
    swin 0.3
    sdens 10.0
    temp {settings.temp_K}
    ion charge 1 conc {settings.ionic_strength_M} radius 2.0
    ion charge -1 conc {settings.ionic_strength_M} radius 2.0
    calcenergy total
    calcforce no
    write pot dx {pot_prefix}
end
quit
""")
    return apbs_in


def run_apbs(apbs_in_path: Path, pot_prefix: str = "potential") -> Path:
    from apbs_binary import APBS_BIN_PATH

    result = subprocess.run(
        [APBS_BIN_PATH, apbs_in_path.name],
        capture_output=True, text=True, cwd=str(apbs_in_path.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"APBS failed:\n{result.stdout[-3000:]}\n{result.stderr[-2000:]}")

    dx_path = apbs_in_path.parent / f"{pot_prefix}.dx"
    if not dx_path.exists():
        raise RuntimeError(f"APBS ran but did not produce {dx_path}:\n{result.stdout[-2000:]}")

    # Basic convergence sanity check: a failed/diverged Newton iteration
    # (nonlinear solver) or an ill-posed linear solve can produce a grid
    # full of NaN/inf without APBS necessarily returning a nonzero exit
    # code. Catch that here rather than silently shipping a bad label.
    from gridData import Grid
    probe = Grid(str(dx_path))
    if not np.all(np.isfinite(probe.grid)):
        raise RuntimeError(
            f"APBS produced a non-finite potential grid for {apbs_in_path.parent.name} "
            "(solver likely failed to converge -- consider retrying with nonlinear=False "
            "or a finer/coarser grid)."
        )
    return dx_path


# ----------------------------------------------------------------------
# Step 5: per-residue potential, 8 A cutoff
# ----------------------------------------------------------------------

def parse_pqr_residues(pqr_path: Path) -> pd.DataFrame:
    """
    Parse a PQR file into a per-atom DataFrame with residue grouping info.
    PQR columns (whitespace-delimited, AMBER ffout from pdb2pqr30):
        ATOM  serial  name  resName  [chainID]  resSeq  x y z charge radius
    chainID is optional in PQR and pdb2pqr sometimes omits it -- handle
    both cases by counting fields.
    """
    rows = []
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            parts = line.split()
            # With chain:    ATOM ser name resName chain resSeq x y z q r  (11 fields)
            # Without chain: ATOM ser name resName resSeq x y z q r        (10 fields)
            if len(parts) == 11:
                _, serial, name, resname, chain, resseq, x, y, z, q, r = parts
            elif len(parts) == 10:
                _, serial, name, resname, resseq, x, y, z, q, r = parts
                chain = "A"
            else:
                continue

            import re
            m = re.match(r"(-?\d+)", resseq)
            if not m:
                continue
            resnum = int(m.group(1))

            rows.append({
                "chain": chain,
                "resnum": resnum,
                "resname": resname,
                "atom_name": name,
                "x": float(x), "y": float(y), "z": float(z),
                "charge": float(q),
            })
    return pd.DataFrame(rows)


def residue_representative_atom(group: pd.DataFrame) -> np.ndarray:
    """C1' if present (standard per-nucleotide anchor for RNA LM alignment),
    else P, else the residue's atom centroid."""
    for name in ("C1'", "C1*", "P"):
        hit = group[group["atom_name"] == name]
        if len(hit):
            return hit[["x", "y", "z"]].iloc[0].to_numpy()
    return group[["x", "y", "z"]].to_numpy().mean(axis=0)


def per_residue_potential(pqr_path: Path, dx_path: Path,
                           cutoff: float = RESIDUE_CUTOFF_ANGSTROM) -> pd.DataFrame:
    """
    Per-residue electrostatic potential, averaged over grid points within
    `cutoff` (default 8 A, ~Debye-length scale at typical RNA ionic
    strengths) of each residue's representative atom.

    Implementation note: this deliberately avoids materializing the full
    grid-point coordinate array (nx*ny*nz*3 floats -- ~274 MB just for
    coordinates on a 225^3 grid, before a KDTree or any copies) and a
    KDTree over the whole grid, which does not scale to large structures
    (ribosome-sized RNAs, big ribozymes) where the fine grid can hit the
    max_dim=225 cap in multiple axes. Instead, for each residue, only the
    small local index window covering a cube of side 2*cutoff around its
    representative atom is sliced out of the grid, and the cutoff sphere
    is applied within that local window. Per-residue cost is then
    independent of total grid size.
    """
    from gridData import Grid

    grid = Grid(str(dx_path))
    origin = np.asarray(grid.origin, dtype=float)
    delta = np.asarray(grid.delta, dtype=float)
    dims = np.array(grid.grid.shape)

    atoms = parse_pqr_residues(pqr_path)
    empty_cols = ["chain", "resnum", "resname", "potential_mean_kT_e",
                  "potential_std_kT_e", "n_grid_points", "x", "y", "z"]
    if atoms.empty:
        log.warning("%s: no atoms parsed from PQR, returning empty potential table", pqr_path.name)
        return pd.DataFrame(columns=empty_cols)

    records = []
    for (chain, resnum, resname), group in atoms.groupby(["chain", "resnum", "resname"], sort=False):
        center = residue_representative_atom(group)

        lo_idx = np.floor((center - cutoff - origin) / delta).astype(int)
        hi_idx = np.ceil((center + cutoff - origin) / delta).astype(int) + 1  # +1: exclusive slice end
        lo_idx = np.clip(lo_idx, 0, dims)
        hi_idx = np.clip(hi_idx, 0, dims)
        if np.any(hi_idx <= lo_idx):
            continue  # residue center falls entirely outside the grid

        sub = grid.grid[lo_idx[0]:hi_idx[0], lo_idx[1]:hi_idx[1], lo_idx[2]:hi_idx[2]]
        xs = origin[0] + np.arange(lo_idx[0], hi_idx[0]) * delta[0]
        ys = origin[1] + np.arange(lo_idx[1], hi_idx[1]) * delta[1]
        zs = origin[2] + np.arange(lo_idx[2], hi_idx[2]) * delta[2]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        dist2 = (gx - center[0]) ** 2 + (gy - center[1]) ** 2 + (gz - center[2]) ** 2
        mask = dist2 <= cutoff ** 2

        local_vals = sub[mask]
        if local_vals.size == 0:
            continue

        records.append({
            "chain": chain,
            "resnum": resnum,
            "resname": resname,
            "potential_mean_kT_e": float(np.mean(local_vals)),
            "potential_std_kT_e": float(np.std(local_vals)),
            "n_grid_points": int(local_vals.size),
            "x": center[0], "y": center[1], "z": center[2],
        })

    if not records:
        log.warning("%s: no residue fell within the potential grid (cutoff=%.1f A), "
                     "returning empty potential table", pqr_path.name, cutoff)
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(records).sort_values(["chain", "resnum"]).reset_index(drop=True)
    return df


def extract_sequence(atoms: pd.DataFrame) -> dict[str, str]:
    """One-letter RNA sequence per chain, for feeding directly into an
    RNA language model alongside the potential labels."""
    code = {
        "A": "A", "RA": "A", "ADE": "A",
        "C": "C", "RC": "C", "CYT": "C",
        "G": "G", "RG": "G", "GUA": "G",
        "U": "U", "RU": "U", "URA": "U",
    }
    seqs: dict[str, str] = {}
    for (chain, resnum, resname), _ in atoms.groupby(["chain", "resnum", "resname"], sort=False):
        base = code.get(resname.rstrip("53"), None)
        if base is None:
            continue
        seqs.setdefault(chain, []).append((resnum, base))
    return {c: "".join(b for _, b in sorted(v)) for c, v in seqs.items()}


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def process_structure(pdb_id: str, raw_dir: Path, work_dir: Path,
                       settings: ApbsSettings) -> Optional[pd.DataFrame]:
    work_dir = work_dir / pdb_id.upper()
    work_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = download_pdb(pdb_id, raw_dir)
    # clean_path = work_dir / f"{pdb_id.lower()}_rna_only.pdb"
    # clean_path, n_atoms, had_modified = strip_ions_and_waters(pdb_path, clean_path)
    # if n_atoms == 0:
    #     log.warning("%s: no RNA ATOM records after cleaning, skipping", pdb_id)
    #     return None

    clean_path = work_dir / f"{pdb_id.lower()}_rna_only.pdb"
    clean_path, n_atoms, had_modified = strip_ions_and_waters(pdb_path, clean_path)
    if n_atoms == 0:
        log.warning("%s: no RNA ATOM records after cleaning, skipping", pdb_id)
        return None
    renumber_pdb_atoms(clean_path)

    pqr_path = work_dir / f"{pdb_id.lower()}.pqr"
    _, used_assign_only = run_pdb2pqr(clean_path, pqr_path)

    pot_prefix = "potential"
    apbs_in = build_apbs_input(pqr_path, work_dir, settings, pot_prefix=pot_prefix)
    try:
        dx_path = run_apbs(apbs_in, pot_prefix=pot_prefix)
    except RuntimeError as e:
        if "overflow" in str(e).lower() or "large potential values" in str(e).lower():

            log.warning(
                "%s: nonlinear APBS failed to converge; retrying with linearized PB",
                pdb_id
            )
            linear_settings = dataclasses.replace(settings)
            linear_settings.nonlinear = False

            apbs_in = build_apbs_input(
                pqr_path,
                work_dir,
                linear_settings,
                pot_prefix=pot_prefix
            )
            dx_path = run_apbs(apbs_in, pot_prefix=pot_prefix)
        else:
            raise

    df = per_residue_potential(pqr_path, dx_path, cutoff=RESIDUE_CUTOFF_ANGSTROM)
    if df.empty:
        log.warning("%s: produced an empty per-residue potential table, skipping", pdb_id)
        return None
    df.insert(0, "pdb_id", pdb_id.upper())
    df["charges_possibly_incomplete"] = used_assign_only
    df["had_modified_nucleotides"] = had_modified

    atoms = parse_pqr_residues(pqr_path)
    seqs = extract_sequence(atoms)
    seq_path = work_dir / f"{pdb_id.lower()}.fasta"
    with open(seq_path, "w") as f:
        for chain, seq in seqs.items():
            f.write(f">{pdb_id.upper()}_{chain}\n{seq}\n")

    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-ids", nargs="*", default=None,
                     help="Explicit list of PDB IDs to process.")
    ap.add_argument("--nonredundant-csv", type=Path, default=None,
                     help="Path to a BGSU RNA 3D Hub non-redundant list CSV, "
                          "downloaded by hand from https://rna.bgsu.edu/rna3dhub/nrlist "
                          "(pick a release and resolution cutoff, then download csv).")
    ap.add_argument("--ionic-strength", type=float, default=0.150,
                     help="Background monovalent ionic strength in M.")
    ap.add_argument("--outdir", type=Path, default=Path("./rna_labels"))
    ap.add_argument("--limit", type=int, default=None,
                     help="Cap number of structures processed (useful for a first test run).")
    args = ap.parse_args()

    if args.nonredundant_csv:
        pdb_ids = parse_bgsu_nonredundant_csv(args.nonredundant_csv)
    elif args.pdb_ids:
        pdb_ids = args.pdb_ids
    else:
        ap.error("Provide --pdb-ids ... or --nonredundant-csv <path to downloaded BGSU csv>")

    if args.limit:
        pdb_ids = pdb_ids[: args.limit]

    raw_dir = args.outdir / "raw_pdb"
    work_dir = args.outdir / "work"
    args.outdir.mkdir(parents=True, exist_ok=True)

    settings = ApbsSettings(ionic_strength_M=args.ionic_strength)

    all_results = []
    failures = []
    for i, pdb_id in enumerate(pdb_ids, 1):
        log.info("[%d/%d] Processing %s", i, len(pdb_ids), pdb_id)
        try:
            df = process_structure(pdb_id, raw_dir, work_dir, settings)
            if df is not None and len(df):
                all_results.append(df)
                df.to_csv(work_dir / pdb_id.upper() / "per_residue_potential.csv", index=False)
        except Exception as e:
            log.error("%s FAILED: %s", pdb_id, e)
            failures.append((pdb_id, str(e), traceback.format_exc()))

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.outdir / "all_residue_potentials.csv", index=False)
        log.info("Wrote %d residue rows across %d structures to %s",
                  len(combined), combined["pdb_id"].nunique(),
                  args.outdir / "all_residue_potentials.csv")

    n_success = len(pdb_ids) - len(failures)
    log.info("Run summary:")
    log.info("  Structures attempted: %d", len(pdb_ids))
    log.info("  Structures succeeded: %d", n_success)
    log.info("  Structures failed: %d", len(failures))

    if failures:
        fail_path = args.outdir / "failures.log"
        with open(fail_path, "w") as f:
            for pdb_id, msg, tb in failures:
                f.write(f"=== {pdb_id} ===\n{msg}\n{tb}\n\n")
        charge_failures = 0
        missing_atom_failures = 0
        serial_failures = 0
        insertion_code_failures = 0
        apbs_failures = 0
        other_failures = 0
        for pdb_id, msg, tb in failures:
            text = (msg + "\n" + tb).lower()
            if (
                "deviates by" in text
                or "non-integer charge" in text
                or "integral charge" in text
            ):
                charge_failures += 1
            elif (
                "too few atoms present" in text
                or "missing backbone atoms" in text
                or "heavy atoms missing" in text
            ):
                missing_atom_failures += 1
            elif "a0000" in text:
                serial_failures += 1

            elif "invalid literal for int()" in text:
                insertion_code_failures += 1
            elif (
                "overflow" in text
                or "large potential values" in text
                or "assertion failure" in text
            ):
                apbs_failures += 1
            else:
                other_failures += 1
        log.warning("%d/%d structures failed; see %s",
                    len(failures), len(pdb_ids), fail_path)
        log.info("Failure summary:")
        log.info("  Charge assignment failures: %d", charge_failures)
        log.info("  Incomplete structure failures: %d", missing_atom_failures)
        log.info("  Serial-number failures: %d", serial_failures)
        log.info("  insertion_code_failures: %d", insertion_code_failures)
        log.info("  apbs_failure: %d", apbs_failures)
        log.info("  Other failures: %d", other_failures)
if __name__ == "__main__":
    main()