"""
RNA per-residue electrostatic potential label-generation pipeline (v2).

Changes vs. v1, driven by the downstream plan (probe RNA LM embeddings for
electrostatic information, then fine-tune toward Mg2+ binding-site
prediction using MD trajectories):

  * Ion sidecar extraction. Resolved metal ions (Mg2+ in particular) are
    captured from the RAW structure -- position, resname, occupancy,
    B-factor, nearest RNA residue, and distance to it -- BEFORE they are
    stripped for the PB solve. This is your downstream Mg2+ ground truth.
    Generating it now avoids re-downloading and re-parsing 1.8k structures
    a second time later.
  * Explicit chain/resnum -> sequence-index map per structure. The
    per-residue potential table is keyed by (chain, resnum) from the PDB,
    but RNA LM tokenizers key by position in the FASTA string, and PDB
    numbering routinely has gaps, insertion codes, or non-1 starts. This
    map is the single source of truth an embedding-probing script should
    use to align the two -- do not try to re-derive it from resnum
    arithmetic downstream.
  * Structure metadata (resolution, experimental method) extracted so
    ions resolved at low resolution (approx. >3.0 A, where ion placement
    gets unreliable) can be flagged or filtered before they're used as
    Mg2+ training labels.
  * Deterministic structure-level train/val/test split written into the
    output CSV (hash of pdb_id), so any residue-level modeling downstream
    can't leak neighboring residues from the same RNA across splits.
  * Resume support: structures with an existing per_residue_potential.csv
    are skipped unless --overwrite is passed.
  * Optional multiprocessing across structures (--n-workers).
  * Removed a dead duplicated code block in process_structure().
  * Post-review fix: residues are renumbered to a synthetic, collision-free
    id per chain (renumber_pdb_residues) before pdb2pqr runs. The original
    (chain, resnum, resname)-only keying used everywhere downstream never
    reads the PDB insertion-code column, and insertion codes are routine in
    tRNA depositions (D-loop/variable-arm positions like 17a, 20a, e11) --
    two distinct residues sharing a resSeq could silently collapse into one
    groupby bucket, corrupting both the averaged potential and, more
    importantly, every seq_index after that point in the chain. True
    numbering is preserved via residue_renumber_map.csv and merged back
    into the output tables as original_resnum/insertion_code.

Pipeline stages (unchanged from v1 unless noted above):
    1. Non-redundant RNA structure list (BGSU RNA 3D Hub representative set,
       parsed from a hand-downloaded CSV -- see parse_bgsu_nonredundant_csv)
    2. Download PDB structures from RCSB (legacy .pdb, mmCIF fallback)
    3. Extract ion sidecar + structure metadata from the RAW structure
    4. Strip crystallographic ions/waters, relabel modified nucleotides
    5. PDB2PQR -> AMBER charges/radii
    6. APBS -> nonlinear (full sinh) Poisson-Boltzmann solve by default,
       linearized PB as an explicit fallback on convergence failure
    7. Per-residue electrostatic potential (8 A cutoff around C1'/P/centroid)
    8. Sequence-index map + FASTA export

IMPORTANT -- same caveat as v1: pdb2pqr RNA charge assignment, the APBS
multigrid PB solve, .dx interpolation, and the local-window cutoff
averaging have been exercised end-to-end against a real APBS run. The ion
sidecar extraction, resolution parsing, and --assign-only retry path are
new in this version and have NOT been run against a real batch -- they are
written defensively (best-effort parsing, never raise on a missing/odd
REMARK line) but you should spot-check the first batch's ions.csv and
structure_metadata.csv against a few known structures (e.g. 1EHZ) before
trusting them at scale.

Requirements:
    pip install apbs-binary pdb2pqr GridDataFormats scipy numpy pandas requests gemmi

Usage:
    python rna_electrostatics_pipeline_v2.py --pdb-ids 1EHZ 4V9F ... --outdir /data/rna_mg/cutoff_8
    python rna_electrostatics_pipeline_v2.py --nonredundant-csv nrlist.csv --outdir /data/rna_mg/cutoff_8 --n-workers 8
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import re
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# See v1 comment for why relabeling (not stripping, not leaving alone) is
# the only approach that actually works with pdb2pqr's AMBER tables.
MODIFIED_NUCLEOTIDE_PARENT = {
    "PSU": "U", "1MA": "A", "2MG": "G", "M2G": "G", "7MG": "G",
    "OMG": "G", "OMC": "C", "OMU": "U", "5MC": "C", "5MU": "U",
    "H2U": "U", "4SU": "U", "YG": "G", "YYG": "G", "UR3": "U",
    "A2M": "A", "MA6": "A", "6MA": "A",
    "I": "A",  # inosine -- relabeled to its biosynthetic parent, not its
               # wobble partner G; reconsider per use case (see v1 note).
}

ION_RESNAMES = {
    "MG", "NA", "K", "CL", "CA", "ZN", "MN", "CO", "NI", "CD", "CU",
    "SR", "BA", "CS", "LI", "RB", "FE", "IOD", "BR",
}

# Resolution (A) beyond which resolved ion positions get flagged as
# unreliable for use as binding-site ground truth. This is a common rule
# of thumb, not a hard physical cutoff -- adjust to your tolerance.
ION_RESOLUTION_FLAG_THRESHOLD = 3.0

RESIDUE_CUTOFF_ANGSTROM = 8.0
MIN_RNA_RESIDUES = 2

# Sweep of averaging radii computed per residue in one pass (see
# per_residue_potential). 8.0 A is kept as RESIDUE_CUTOFF_ANGSTROM above for
# reference/back-compat, but nothing downstream should hardcode "the"
# cutoff -- which radius (or the point-value alternative) correlates best
# with LM representations is an open experimental question, not a fixed
# choice baked into label generation.
POTENTIAL_CUTOFFS_ANGSTROM = (4.0, 8.0, 12.0)

SEQ_CODE = {
    "A": "A", "RA": "A", "ADE": "A", "A5": "A", "A3": "A",
    "C": "C", "RC": "C", "CYT": "C", "C5": "C", "C3": "C",
    "G": "G", "RG": "G", "GUA": "G", "G5": "G", "G3": "G",
    "U": "U", "RU": "U", "URA": "U", "U5": "U", "U3": "U",
}

def resname_to_base(resname: str) -> str | None:
    """Map residue-name variants emitted by PDB2PQR to canonical bases."""

    resname = resname.strip().upper()

    if resname in {"A", "RA", "ADE", "A3", "A5", "RA3", "RA5"}:
        return "A"

    if resname in {"C", "RC", "CYT", "C3", "C5", "RC3", "RC5"}:
        return "C"

    if resname in {"G", "RG", "GUA", "G3", "G5", "RG3", "RG5"}:
        return "G"

    if resname in {"U", "RU", "URA", "U3", "U5", "RU3", "RU5"}:
        return "U"

    return None

@dataclasses.dataclass
class ApbsSettings:
    ionic_strength_M: float = 0.150
    pdie: float = 1.0
    sdie: float = 80.0
    temp_K: float = 300.0
    srad: float = 1.4
    fine_grid_spacing: float = 0.5
    fine_padding: float = 15.0
    coarse_padding: float = 40.0
    max_dim: int = 225
    nonlinear: bool = True


# ----------------------------------------------------------------------
# Step 1: non-redundant RNA structure list
# ----------------------------------------------------------------------

def parse_bgsu_nonredundant_csv(csv_path: Path) -> list[str]:
    """Parse a BGSU RNA 3D Hub non-redundant list CSV (hand-downloaded
    from https://rna.bgsu.edu/rna3dhub/nrlist). See v1 for the exact
    column format. Returns one representative PDB ID per equivalence
    class."""
    import csv as csv_module

    pdb_ids = []
    with open(csv_path, newline="") as f:
        reader = csv_module.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            representative = row[1]
            first_member = representative.split("+")[0]
            pdb_id = first_member.split("|")[0].strip()
            if len(pdb_id) == 4:
                pdb_ids.append(pdb_id.upper())

    pdb_ids = sorted(set(pdb_ids))
    log.info("Parsed %d non-redundant RNA structure IDs from %s", len(pdb_ids), csv_path)
    return pdb_ids


# ----------------------------------------------------------------------
# Step 2: download
# ----------------------------------------------------------------------

def download_pdb(pdb_id: str, out_dir: Path) -> Path:
    """Download from RCSB, preferring legacy .pdb, falling back to mmCIF
    (converted via gemmi) when legacy isn't available -- see v1 docstring
    for why this fallback is necessary (~20% 404 rate on legacy format in
    the first real batch)."""
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
    import string

    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0:
        raise RuntimeError(
            f"{pdb_id}: mmCIF fallback parsed 0 models, cannot convert"
        )

    st.setup_entities()
    try:
        st.write_pdb(str(dest))
    except RuntimeError as e:
        if "chain name too long for the PDB format" in str(e):
            raise RuntimeError(
                f"{pdb_id}: mmcif_long_chain_ids"
            )
        raise
    
    st.write_pdb(str(dest))
    return dest

# ----------------------------------------------------------------------
# Step 3: structure metadata + ion sidecar (from RAW structure, pre-strip)
# ----------------------------------------------------------------------

def extract_structure_metadata(pdb_path: Path) -> dict:
    """Best-effort extraction of resolution and experimental method from
    a legacy PDB file's header records. Never raises -- a structure with
    unparseable/absent header lines just gets NaN/None fields, which
    callers should treat as 'unknown', not 'high confidence'.

    NOTE: this only works on legacy-PDB-format headers. If download_pdb
    fell back to the mmCIF->PDB conversion path, gemmi's write_pdb does
    carry over REMARK 2 RESOLUTION for most entries, but this has not
    been verified across a real batch -- spot check.
    """
    resolution = None
    method = None
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("REMARK   2 RESOLUTION."):
                    m = re.search(r"(\d+\.\d+)\s*ANGSTROM", line)
                    if m:
                        resolution = float(m.group(1))
                elif line.startswith("EXPDTA"):
                    method = line[10:].strip()
                elif line.startswith("ATOM") or line.startswith("HETATM"):
                    break  # header is done, stop scanning
    except Exception:
        log.debug("Could not parse header metadata for %s", pdb_path.name, exc_info=True)
    return {"resolution_A": resolution, "experimental_method": method}


def extract_ion_positions(pdb_path: Path, ion_resnames: set[str] = ION_RESNAMES) -> pd.DataFrame:
    """Pull resolved ion HETATM records from the RAW (pre-strip)
    structure. This is the ground truth the Mg2+ binding-site fine-tuning
    stage will eventually need -- captured here so it doesn't require a
    second download/parse pass later.

    Nearest-residue assignment is deliberately NOT done here (it needs
    the cleaned residue representative-atom centers, which don't exist
    yet at this point in the pipeline) -- see label_ion_proximity(),
    called after per_residue_potential() in process_structure().
    """
    rows = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname not in ion_resnames:
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                occupancy = float(line[54:60]) if line[54:60].strip() else np.nan
                bfactor = float(line[60:66]) if line[60:66].strip() else np.nan
            except ValueError:
                continue
            chain = line[21].strip() or "A"
            resseq_field = line[22:26].strip()
            m = re.match(r"(-?\d+)", resseq_field)
            resseq = int(m.group(1)) if m else None
            rows.append({
                "chain": chain, "resnum": resseq, "resname": resname,
                "x": x, "y": y, "z": z,
                "occupancy": occupancy, "bfactor": bfactor,
            })
    return pd.DataFrame(rows)


def label_ion_proximity(ions_df: pd.DataFrame, residue_centers: pd.DataFrame) -> pd.DataFrame:
    """For each resolved ion, find the nearest RNA residue (by
    representative-atom center, any chain) and the distance to it. This
    distance is your Mg2+ binding-site label signal downstream -- e.g.
    threshold at ~3.5 A (typical inner-sphere Mg-phosphate/base contact)
    to get a binary 'this residue coordinates a resolved Mg2+' label per
    residue, or keep the continuous distance for a softer target.
    """
    if ions_df.empty or residue_centers.empty:
        ions_df = ions_df.copy()
        ions_df["nearest_chain"] = None
        ions_df["nearest_resnum"] = None
        ions_df["nearest_resname"] = None
        ions_df["distance_to_nearest_residue_A"] = np.nan
        return ions_df

    from scipy.spatial import cKDTree
    centers_xyz = residue_centers[["x", "y", "z"]].to_numpy()
    tree = cKDTree(centers_xyz)
    ion_xyz = ions_df[["x", "y", "z"]].to_numpy()
    dist, idx = tree.query(ion_xyz, k=1)

    out = ions_df.copy()
    out["nearest_chain"] = residue_centers["chain"].to_numpy()[idx]
    out["nearest_resnum"] = residue_centers["resnum"].to_numpy()[idx]
    out["nearest_resname"] = residue_centers["resname"].to_numpy()[idx]
    out["distance_to_nearest_residue_A"] = dist
    return out

def remove_alternate_locations(in_pdb: Path, out_pdb: Path):
    """
    Keep blank altLoc and altLoc A only.
    Remove B/C/D... alternate conformations.
    """

    with open(in_pdb) as fin, open(out_pdb, "w") as fout:
        for line in fin:

            if line.startswith(("ATOM", "HETATM")):
                altloc = line[16]

                if altloc not in (" ", "A"):
                    continue

                if altloc == "A":
                    line = line[:16] + " " + line[17:]

            fout.write(line)

def remove_incomplete_residues(in_pdb: Path, out_pdb: Path):
    """
    Remove residues lacking C1'.
    These are typically dangling phosphate stubs:
        P, OP1, OP2, O5'
    which can trigger PDB2PQR charge failures.
    """

    residue_atoms = {}
    lines = []

    with open(in_pdb) as f:
        for line in f:
            lines.append(line)

            if not line.startswith(("ATOM", "HETATM")):
                continue

            chain = line[21]
            resnum = line[22:26].strip()
            atom = line[12:16].strip()

            key = (chain, resnum)

            if key not in residue_atoms:
                residue_atoms[key] = set()

            residue_atoms[key].add(atom)

    keep_residues = {
        key
        for key, atoms in residue_atoms.items()
            if (
                "C1'" in atoms
                and
                ("N9" in atoms or "N1" in atoms)
            )}
    removed = len(residue_atoms) - len(keep_residues)

    if removed:
        log.info(
            "%s: removed %d incomplete residues lacking C1'",
            in_pdb.name,
            removed,
        )

    with open(out_pdb, "w") as f:
        for line in lines:

            if not line.startswith(("ATOM", "HETATM")):
                f.write(line)
                continue

            chain = line[21]
            resnum = line[22:26].strip()

            if (chain, resnum) in keep_residues:
                f.write(line)
# ----------------------------------------------------------------------
# Step 4: strip ions/waters, relabel modified nucleotides
# ----------------------------------------------------------------------

def strip_ions_and_waters(pdb_path: Path, out_path: Path) -> tuple[Path, int, bool]:
    """Same behavior as v1: keep RNA ATOM records only, relabel known
    modified nucleotides to their parent base (flagging that this
    happened), drop everything else (ions, waters, ligands, unmapped
    modified residues)."""
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
                    continue
                line = "ATOM  " + line[6:]
                kept += 1
            fout.write(line)
    return out_path, kept, had_modified


def renumber_pdb_residues(pdb_path: Path) -> pd.DataFrame:
    """Reassign a synthetic, collision-free residue number (resSeq) per
    chain, blanking any insertion code, and return a mapping table back
    to the original (chain, resnum, insertion_code, resname).

    WHY THIS EXISTS (post-review fix): every downstream step keys
    residues on (chain, resnum, resname) parsed with a digit-only regex
    that never reads the PDB insertion-code column. Many real RNA
    structures -- tRNAs especially, whose canonical numbering uses named
    insertions in the D-loop/variable arm (17a, 20a, 20b, e11, e12, ...)
    -- have two DISTINCT residues sharing the same numeric resSeq,
    differing only by insertion code. Two such residues with the same
    resname would silently collapse into a single groupby bucket: their
    atoms averaged into one bogus 'residue', and every seq_index after
    that point in the chain shifted by one relative to the true
    sequence -- with no exception raised. That silent shift corrupts
    exactly the embedding-to-label alignment the probing script depends
    on. It is also not safe to assume pdb2pqr preserves insertion codes
    into its PQR output resSeq field (unverified either way), so fixing
    the parser alone would not be a complete fix.

    The fix applies the same principle already used for atom serials in
    renumber_pdb_atoms(): eliminate the ambiguity before pdb2pqr ever
    sees the file, rather than relying on a third-party tool's format
    handling for something ambiguity-prone. Call this BEFORE pdb2pqr
    (order relative to renumber_pdb_atoms doesn't matter). The returned
    mapping (also useful to merge back into per_residue_potential.csv /
    seq_index_map.csv for provenance) is keyed by (chain,
    synthetic_resnum) -> (original_resnum, insertion_code, resname).
    """
    lines = pdb_path.read_text().splitlines(keepends=True)
    chain_counters: dict[str, int] = {}
    seen: dict[tuple, int] = {}
    mapping_rows = []
    out_lines = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 27:
            chain = (line[21].strip() or "A")
            orig_resnum_field = line[22:26]
            icode = line[26]
            resname = line[17:20].strip()
            key = (chain, orig_resnum_field, icode)

            if key not in seen:
                chain_counters[chain] = chain_counters.get(chain, 0) + 1
                synthetic_id = chain_counters[chain]
                if synthetic_id > 9999:
                    raise RuntimeError(
                        f"{pdb_path}: chain {chain} exceeds 9999 residues, cannot "
                        "represent in legacy PDB resSeq field."
                    )
                seen[key] = synthetic_id
                try:
                    orig_resnum = int(orig_resnum_field.strip())
                except ValueError:
                    orig_resnum = None
                mapping_rows.append({
                    "chain": chain, "synthetic_resnum": synthetic_id,
                    "original_resnum": orig_resnum,
                    "insertion_code": icode.strip(), "resname": resname,
                })

            synthetic_id = seen[key]
            line = line[:22] + f"{synthetic_id:4d}" + " " + line[27:]
        out_lines.append(line)

    pdb_path.write_text("".join(out_lines))
    return pd.DataFrame(mapping_rows)


def renumber_pdb_atoms(pdb_path: Path) -> None:
    """Rewrite ATOM/HETATM serials sequentially -- fixes hybrid-36
    serials (e.g. A0000) from mmCIF->PDB conversion that pdb2pqr 3.7.1
    cannot parse."""
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
                line = line[:6] + f"{serial:5d}" + line[11:]
                serial += 1
            out_lines.append(line)
    with open(pdb_path, "w") as f:
        f.writelines(out_lines)

# ----------------------------------------------------------------------
# Step 5: PDB2PQR
# ----------------------------------------------------------------------

def run_pdb2pqr(pdb_path: Path, pqr_path: Path) -> tuple[Path, bool]:
    """Unchanged from v1: default run, retry once with --assign-only on
    a non-integer-charge crash (missing/unresolved atoms). See v1
    docstring -- the --assign-only recovery path is a plausible but
    UNVERIFIED fallback; treat charges_possibly_incomplete=True rows as
    lower-confidence labels, not equal-confidence ones."""
    base_cmd = ["pdb2pqr30", "--ff", "AMBER", "--ffout", "AMBER", "--drop-water"]

    result = subprocess.run(base_cmd + [str(pdb_path), str(pqr_path)],
                             capture_output=True, text=True)
    if result.returncode == 0 and pqr_path.exists():
        return pqr_path, False

    charge_failure_phrases = (
        "non-integer charge", "deviates by", "deviates from integral",
        "charge assignment failed",
    )
    error_text = result.stdout + result.stderr
    charge_failure = any(phrase in error_text for phrase in charge_failure_phrases)
    if not charge_failure:
        raise RuntimeError(
            f"pdb2pqr30 failed for {pdb_path}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )

    log.warning("%s: pdb2pqr hit a non-integer-charge crash; retrying with --assign-only",
                pdb_path.name)
    retry = subprocess.run(base_cmd + ["--assign-only", str(pdb_path), str(pqr_path)],
                            capture_output=True, text=True)
    if retry.returncode != 0 or not pqr_path.exists():
        raise RuntimeError(
            f"pdb2pqr30 failed for {pdb_path} (both default and --assign-only retry):\n"
            f"--- default ---\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}\n"
            f"--- --assign-only ---\n{retry.stdout[-1000:]}\n{retry.stderr[-1000:]}"
        )
    log.info("%s: recovered via --assign-only (missing atoms NOT reconstructed -- "
              "check completeness before trusting labels)", pdb_path.name)
    return pqr_path, True

def reformat_pqr_for_apbs(pqr_path: Path):
    """
    Fix concatenated coordinate fields that APBS cannot parse.
    Example:
        12.291-100.204
    becomes:
        12.291 -100.204
    """

    fixed_lines = []
    with open(pqr_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                # Insert space before a negative coordinate
                line = re.sub(
                    r'(\d\.\d+)(-\d+\.\d+)',
                    r'\1 \2',
                    line,
                )
            fixed_lines.append(line)

    with open(pqr_path, "w") as f:
        f.writelines(fixed_lines)

# ----------------------------------------------------------------------
# Step 6: APBS
# ----------------------------------------------------------------------

def _next_valid_dime(n: int) -> int:
    candidates = [33, 65, 97, 129, 161, 193, 225, 257]
    for c in candidates:
        if c >= n:
            return c
    return candidates[-1]


def parse_pqr_residues(pqr_path: Path) -> pd.DataFrame:
    """The single place PQR lines get parsed (v1 had a second,
    independent, silently-wrong parser inside read_pqr_coords -- removed
    in v2; everything now goes through this function)."""
    rows = []
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            parts = line.split()
            if len(parts) == 11:
                _, serial, name, resname, chain, resseq, x, y, z, q, r = parts
            elif len(parts) == 10:
                _, serial, name, resname, resseq, x, y, z, q, r = parts
                chain = "A"
            else:
                continue
            m = re.match(r"(-?\d+)", resseq)
            if not m:
                continue
            resnum = int(m.group(1))
            rows.append({
                "chain": chain, "resnum": resnum, "resname": resname,
                "atom_name": name, "x": float(x), "y": float(y), "z": float(z),
                "charge": float(q),
            })
    return pd.DataFrame(
    rows,
    columns=[
        "chain",
        "resnum",
        "resname",
        "atom_name",
        "x",
        "y",
        "z",
        "charge",
    ],
)


def read_pqr_coords(pqr_path: Path) -> np.ndarray:
    atoms = parse_pqr_residues(pqr_path)
    if atoms.empty:
        raise RuntimeError(
            f"{pqr_path.name}: PQR contains no parsable RNA atom coordinates"
        )
    return atoms[["x", "y", "z"]].to_numpy()



def build_apbs_input(pqr_path: Path, work_dir: Path, settings: ApbsSettings,
                      pot_prefix: str = "potential") -> Path:
    coords = read_pqr_coords(pqr_path)
    extent = coords.max(axis=0) - coords.min(axis=0)
    center = coords.mean(axis=0)

    fine_len = extent + 2 * settings.fine_padding
    coarse_len = extent + 2 * settings.coarse_padding

    dime = [
        min(_next_valid_dime(int(np.ceil(L / settings.fine_grid_spacing))), settings.max_dim)
        for L in fine_len
    ]

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

    from gridData import Grid
    probe = Grid(str(dx_path))
    if not np.all(np.isfinite(probe.grid)):
        raise RuntimeError(
            f"APBS produced a non-finite potential grid for {apbs_in_path.parent.name} "
            "(solver likely failed to converge -- consider nonlinear=False or a different grid)."
        )
    return dx_path


# ----------------------------------------------------------------------
# Step 7: per-residue potential
# ----------------------------------------------------------------------

def residue_representative_atom(group: pd.DataFrame) -> np.ndarray:
    for name in ("C1'", "C1*", "P"):
        hit = group[group["atom_name"] == name]
        if len(hit):
            return hit[["x", "y", "z"]].iloc[0].to_numpy()
    return group[["x", "y", "z"]].to_numpy().mean(axis=0)


def trilinear_interpolate(values: np.ndarray, origin: np.ndarray, delta: np.ndarray,
                           dims: np.ndarray, point: np.ndarray) -> float:
    """Standard 8-corner trilinear interpolation of a scalar field at an
    arbitrary point -- used to get the potential AT an atom's exact
    coordinates (point value), as distinct from averaging over a cutoff
    sphere around it. Returns NaN if the point (or its interpolation
    cell) falls outside the grid."""
    frac_idx = (point - origin) / delta
    i0 = np.floor(frac_idx).astype(int)
    i1 = i0 + 1
    if np.any(i0 < 0) or np.any(i1 >= dims):
        return float("nan")
    frac = frac_idx - i0

    c000 = values[i0[0], i0[1], i0[2]]; c100 = values[i1[0], i0[1], i0[2]]
    c010 = values[i0[0], i1[1], i0[2]]; c110 = values[i1[0], i1[1], i0[2]]
    c001 = values[i0[0], i0[1], i1[2]]; c101 = values[i1[0], i0[1], i1[2]]
    c011 = values[i0[0], i1[1], i1[2]]; c111 = values[i1[0], i1[1], i1[2]]

    c00 = c000 * (1 - frac[0]) + c100 * frac[0]
    c10 = c010 * (1 - frac[0]) + c110 * frac[0]
    c01 = c001 * (1 - frac[0]) + c101 * frac[0]
    c11 = c011 * (1 - frac[0]) + c111 * frac[0]
    c0 = c00 * (1 - frac[1]) + c10 * frac[1]
    c1 = c01 * (1 - frac[1]) + c11 * frac[1]
    return float(c0 * (1 - frac[2]) + c1 * frac[2])


def per_residue_potential(pqr_path: Path, dx_path: Path,
                           cutoffs: tuple[float, ...] = POTENTIAL_CUTOFFS_ANGSTROM) -> pd.DataFrame:
    """Per-residue electrostatic potential, computed several ways at
    once off the SAME PB solve (post-review addition):

      * potential_mean_kT_e_{c}A / potential_std_kT_e_{c}A / n_grid_points_{c}A
        for every cutoff radius in `cutoffs` (default 4/8/12 A) -- the
        residue-center-average, same method as before, just swept.
      * potential_at_c1prime_kT_e / potential_at_phosphorus_kT_e -- the
        potential AT the atom's exact coordinates via trilinear
        interpolation (radius 0), for comparison against the sphere
        averages.

    Rationale: the choice of averaging radius (and center-vs-point) is
    an open experimental-design question for probing LM embeddings, not
    something to bake in silently -- and it costs nothing extra to sweep
    here, since every cutoff for a given residue reuses the same sliced
    local window (built once, at the largest requested cutoff) rather
    than re-slicing per radius. Re-running APBS per cutoff would be far
    more expensive and is never necessary.
    """
    from gridData import Grid

    grid = Grid(str(dx_path))
    origin = np.asarray(grid.origin, dtype=float)
    delta = np.asarray(grid.delta, dtype=float)
    dims = np.array(grid.grid.shape)
    values = grid.grid

    atoms = parse_pqr_residues(pqr_path)
    cutoff_cols = []
    for c in cutoffs:
        tag = f"{c:g}A"
        cutoff_cols += [f"potential_mean_kT_e_{tag}", f"potential_std_kT_e_{tag}", f"n_grid_points_{tag}"]
    empty_cols = (["chain", "resnum", "resname"] + cutoff_cols +
                  ["potential_at_c1prime_kT_e", "potential_at_phosphorus_kT_e", "x", "y", "z"])

    if atoms.empty:
        log.warning("%s: no atoms parsed from PQR, returning empty potential table", pqr_path.name)
        return pd.DataFrame(columns=empty_cols)

    max_cutoff = max(cutoffs)
    records = []
    for (chain, resnum, resname), group in atoms.groupby(["chain", "resnum", "resname"], sort=False):
        center = residue_representative_atom(group)

        lo_idx = np.floor((center - max_cutoff - origin) / delta).astype(int)
        hi_idx = np.ceil((center + max_cutoff - origin) / delta).astype(int) + 1
        lo_idx = np.clip(lo_idx, 0, dims)
        hi_idx = np.clip(hi_idx, 0, dims)
        if np.any(hi_idx <= lo_idx):
            continue

        sub = values[lo_idx[0]:hi_idx[0], lo_idx[1]:hi_idx[1], lo_idx[2]:hi_idx[2]]
        xs = origin[0] + np.arange(lo_idx[0], hi_idx[0]) * delta[0]
        ys = origin[1] + np.arange(lo_idx[1], hi_idx[1]) * delta[1]
        zs = origin[2] + np.arange(lo_idx[2], hi_idx[2]) * delta[2]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        dist2 = (gx - center[0]) ** 2 + (gy - center[1]) ** 2 + (gz - center[2]) ** 2

        row = {"chain": chain, "resnum": resnum, "resname": resname,
               "x": center[0], "y": center[1], "z": center[2]}
        any_valid = False
        for c in cutoffs:
            tag = f"{c:g}A"
            vals = sub[dist2 <= c ** 2]
            if vals.size == 0:
                row[f"potential_mean_kT_e_{tag}"] = np.nan
                row[f"potential_std_kT_e_{tag}"] = np.nan
                row[f"n_grid_points_{tag}"] = 0
            else:
                row[f"potential_mean_kT_e_{tag}"] = float(np.mean(vals))
                row[f"potential_std_kT_e_{tag}"] = float(np.std(vals))
                row[f"n_grid_points_{tag}"] = int(vals.size)
                any_valid = True
        if not any_valid:
            continue

        c1p = group[group["atom_name"].isin(["C1'", "C1*"])]
        p_atom = group[group["atom_name"] == "P"]
        row["potential_at_c1prime_kT_e"] = (
            trilinear_interpolate(values, origin, delta, dims, c1p[["x", "y", "z"]].iloc[0].to_numpy())
            if len(c1p) else np.nan
        )
        row["potential_at_phosphorus_kT_e"] = (
            trilinear_interpolate(values, origin, delta, dims, p_atom[["x", "y", "z"]].iloc[0].to_numpy())
            if len(p_atom) else np.nan
        )
        records.append(row)

    if not records:
        log.warning("%s: no residue fell within the potential grid (cutoffs=%s)",
                     pqr_path.name, cutoffs)
        return pd.DataFrame(columns=empty_cols)

    df = pd.DataFrame(records).sort_values(["chain", "resnum"]).reset_index(drop=True)
    return df


def extract_sequence_and_index_map(atoms: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    """Returns (per-chain one-letter sequence, index-map DataFrame).

    The index-map DataFrame has one row per (chain, resnum, resname) with
    a `seq_index` column giving that residue's 0-based position in the
    chain's FASTA string. THIS is what an embedding-probing script should
    join against -- not resnum arithmetic -- since resnum can have gaps.
    """
    seqs: dict[str, list[tuple[int, str]]] = {}
    for (chain, resnum, resname), _ in atoms.groupby(["chain", "resnum", "resname"], sort=False):
        base = resname_to_base(resname)
        if base is None:
            continue
        seqs.setdefault(chain, []).append((resnum, resname, base))

    seq_strs: dict[str, str] = {}
    index_rows = []
    for chain, entries in seqs.items():
        entries_sorted = sorted(entries, key=lambda e: e[0])
        seq_strs[chain] = "".join(b for _, _, b in entries_sorted)
        for i, (resnum, resname, base) in enumerate(entries_sorted):
            index_rows.append({"chain": chain, "resnum": resnum, "resname": resname,
                                "base": base, "seq_index": i})

    index_df = pd.DataFrame(
    index_rows, columns=[
                    "chain",
                    "resnum",
                    "resname",
                    "base",
                    "seq_index",
                    ],
                )
    if index_df.empty:
        print(
            "DEBUG: residue names seen in PQR:",
            sorted(atoms["resname"].dropna().unique())
        )
    
    return seq_strs, index_df


def assign_split(pdb_id: str) -> str:
    """Deterministic structure-level split assignment (80/10/10) via
    hashing, so residue-level modeling downstream can't leak neighboring
    residues from the same RNA across train/val/test."""
    h = int(hashlib.md5(pdb_id.upper().encode()).hexdigest(), 16)
    bucket = h % 100
    if bucket < 80:
        return "train"
    elif bucket < 90:
        return "val"
    return "test"


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def process_structure(pdb_id: str, raw_dir: Path, work_dir: Path,
                       settings: ApbsSettings, overwrite: bool = False) -> Optional[dict]:
    struct_work_dir = work_dir / pdb_id.upper()
    struct_work_dir.mkdir(parents=True, exist_ok=True)

    result_csv = struct_work_dir / "per_residue_potential.csv"
    if result_csv.exists() and not overwrite:
        log.info("%s: already processed, skipping (--overwrite to redo)", pdb_id)
        df = pd.read_csv(result_csv)
        return {"pdb_id": pdb_id.upper(), "df": df, "skipped": True}

    pdb_path = download_pdb(pdb_id, raw_dir)
    metadata = extract_structure_metadata(pdb_path)
    raw_ions = extract_ion_positions(pdb_path)

    clean_path = struct_work_dir / f"{pdb_id.lower()}_rna_only.pdb"
    altloc_filtered = struct_work_dir / f"{pdb_id.lower()}_altloc_filtered.pdb"
    remove_alternate_locations(
        pdb_path,
        altloc_filtered,
    )
    clean_path, n_atoms, had_modified = strip_ions_and_waters(
        altloc_filtered,
        clean_path,
    )

    filtered_path = struct_work_dir / f"{pdb_id.lower()}_complete_residues.pdb"
    remove_incomplete_residues(clean_path,filtered_path,)
    clean_path = filtered_path

    if n_atoms == 0:
        raise RuntimeError(
            f"{pdb_id}: no RNA ATOM records after cleaning"
        )
    residue_map = renumber_pdb_residues(clean_path)
    residue_map.insert(0, "pdb_id", pdb_id.upper())
    residue_map.to_csv(struct_work_dir / "residue_renumber_map.csv", index=False)

    # ------------------------------------------------------------
    # Reject structures that collapse to too few RNA residues
    # after RNA extraction + residue filtering.
    # Examples: 216D leaves only a single cytidine residue.
    # ------------------------------------------------------------
    n_residues = len(residue_map)

    if n_residues < MIN_RNA_RESIDUES:
        raise RuntimeError(
            f"{clean_path}: only {n_residues} RNA residue(s) remain "
            f"after extraction/filtering (minimum={MIN_RNA_RESIDUES})"
        )

    renumber_pdb_atoms(clean_path)

    pqr_path = struct_work_dir / f"{pdb_id.lower()}.pqr"
    _, used_assign_only = run_pdb2pqr(
        clean_path,
        pqr_path,
    )

    reformat_pqr_for_apbs(pqr_path)

    pot_prefix = "potential"
    apbs_in = build_apbs_input(pqr_path, struct_work_dir, settings, pot_prefix=pot_prefix)
    try:
        dx_path = run_apbs(apbs_in, pot_prefix=pot_prefix)
    except RuntimeError as e:
        if "overflow" in str(e).lower() or "large potential values" in str(e).lower() \
                or "converge" in str(e).lower():
            log.warning("%s: nonlinear APBS failed to converge; retrying with linearized PB", pdb_id)
            linear_settings = dataclasses.replace(settings, nonlinear=False)
            apbs_in = build_apbs_input(pqr_path, struct_work_dir, linear_settings, pot_prefix=pot_prefix)
            dx_path = run_apbs(apbs_in, pot_prefix=pot_prefix)
        else:
            raise

    df = per_residue_potential(pqr_path, dx_path, cutoffs=POTENTIAL_CUTOFFS_ANGSTROM)
    if df.empty:
        log.warning("%s: produced an empty per-residue potential table, skipping", pdb_id)
        return None

    # `resnum` at this point is the SYNTHETIC, collision-free id assigned by
    # renumber_pdb_residues (assumed passed through unchanged by pdb2pqr,
    # which is a far safer assumption than expecting insertion-code
    # passthrough). Merge back the true canonical numbering for provenance
    # -- downstream code (and the probing script's seq_index join) should
    # keep using `resnum` as-is; `original_resnum`/`insertion_code` are for
    # inspection and for relating results back to the literature's numbering.
    rmap = residue_map[["chain", "synthetic_resnum", "original_resnum", "insertion_code"]]
    df = df.merge(rmap, left_on=["chain", "resnum"], right_on=["chain", "synthetic_resnum"],
                  how="left").drop(columns=["synthetic_resnum"])

    df.insert(0, "pdb_id", pdb_id.upper())
    df["charges_possibly_incomplete"] = used_assign_only
    df["had_modified_nucleotides"] = had_modified
    df["split"] = assign_split(pdb_id)
    df["resolution_A"] = metadata["resolution_A"]
    df["experimental_method"] = metadata["experimental_method"]

    atoms = parse_pqr_residues(pqr_path)
    seqs, index_map = extract_sequence_and_index_map(atoms)
    if index_map.empty:
        raise RuntimeError(
        f"{pdb_id}: no recognizable RNA residues remained after PDB2PQR parsing"
        )
    index_map = index_map.merge(rmap, left_on=["chain", "resnum"],
                                 right_on=["chain", "synthetic_resnum"],
                                 how="left").drop(columns=["synthetic_resnum"])

    seq_path = struct_work_dir / f"{pdb_id.lower()}.fasta"
    with open(seq_path, "w") as f:
        for chain, seq in seqs.items():
            f.write(f">{pdb_id.upper()}_{chain}\n{seq}\n")

    index_map.insert(0, "pdb_id", pdb_id.upper())
    index_map.to_csv(struct_work_dir / "seq_index_map.csv", index=False)

    ions_labeled = label_ion_proximity(raw_ions, df[["chain", "resnum", "resname", "x", "y", "z"]])
    if not ions_labeled.empty:
        # `nearest_resnum` from label_ion_proximity is the SYNTHETIC numbering
        # (df's `resnum` is synthetic; raw_ions' own chain/resnum are still the
        # true PDB numbering and are left untouched, geometry doesn't care).
        # Attach canonical numbering for the matched residue too, so ions.csv
        # exposes the same original_resnum/insertion_code every other output
        # table does, rather than surprising future-you with a lone synthetic
        # column.
        ions_labeled = ions_labeled.merge(
            rmap.rename(columns={
                "chain": "nearest_chain", "synthetic_resnum": "nearest_resnum",
                "original_resnum": "nearest_original_resnum",
                "insertion_code": "nearest_insertion_code",
            }),
            on=["nearest_chain", "nearest_resnum"], how="left",
        )
        low_res_flag = (
            metadata["resolution_A"] is not None
            and metadata["resolution_A"] > ION_RESOLUTION_FLAG_THRESHOLD
        )
        ions_labeled.insert(0, "pdb_id", pdb_id.upper())
        ions_labeled["resolution_A"] = metadata["resolution_A"]
        ions_labeled["low_resolution_flag"] = low_res_flag
    ions_labeled.to_csv(struct_work_dir / "ions.csv", index=False)


    df.to_csv(result_csv, index=False)
    return {"pdb_id": pdb_id.upper(), "df": df, "skipped": False}


def validate_alignment(labels_dir: Path, pdb_id: str) -> dict:
    """Sanity-check the numbering invariant that matters most for the
    downstream probing script: per chain,
        len(fasta_sequence) == seq_index_map['seq_index'].nunique()
                              == number of per_residue_potential rows
    Run this on a handful of ALREADY-PROCESSED structures -- especially
    ones known to carry insertion codes, e.g. classic tRNA depositions
    like 1EHZ, 6TNA, 4TNA -- before trusting a full batch run. A mismatch
    here means the seq_index/embedding alignment cannot be trusted for
    that structure even if every earlier stage completed without error.
    """
    struct_dir = labels_dir / "work" / pdb_id.upper()
    fasta_path = struct_dir / f"{pdb_id.lower()}.fasta"
    idx_path = struct_dir / "seq_index_map.csv"
    pot_path = struct_dir / "per_residue_potential.csv"

    if not (fasta_path.exists() and idx_path.exists() and pot_path.exists()):
        return {"pdb_id": pdb_id.upper(), "error": "missing one or more expected output files"}

    seqs: dict[str, str] = {}
    chain = None
    for line in fasta_path.read_text().splitlines():
        if line.startswith(">"):
            chain = line.split("_")[-1].strip()
        elif chain is not None:
            seqs[chain] = line.strip()

    idx = pd.read_csv(idx_path)
    pot = pd.read_csv(pot_path)

    per_chain = {}
    all_consistent = True
    for chain, seq in seqs.items():
        n_fasta = len(seq)
        n_idx = int(idx.loc[idx["chain"] == chain, "seq_index"].nunique())
        n_pot = int((pot["chain"] == chain).sum())
        consistent = n_fasta == n_idx == n_pot
        all_consistent &= consistent
        per_chain[chain] = {"fasta_len": n_fasta, "seq_index_nunique": n_idx,
                             "n_potential_rows": n_pot, "consistent": consistent}

    return {"pdb_id": pdb_id.upper(), "all_chains_consistent": all_consistent,
            "per_chain": per_chain}


def _process_one(args):
    pdb_id, raw_dir, work_dir, settings, overwrite = args
    try:
        result = process_structure(pdb_id, raw_dir, work_dir, settings, overwrite)
        return pdb_id, result, None
    except Exception as e:
        return pdb_id, None, (str(e), traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-ids", nargs="*", default=None)
    ap.add_argument("--nonredundant-csv", type=Path, default=None)
    ap.add_argument("--ionic-strength", type=float, default=0.150)
    ap.add_argument("--outdir", type=Path, default=Path("./rna_labels"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=1,
                     help="Parallel structures processed at once. Start small (2-4) and "
                          "watch memory -- APBS grids can be large.")
    ap.add_argument("--overwrite", action="store_true",
                     help="Reprocess structures even if per_residue_potential.csv exists.")
    ap.add_argument("--validate", action="store_true",
                     help="Skip processing. Instead, for each of --pdb-ids (must already be "
                          "processed), check the FASTA/seq_index_map/per_residue_potential "
                          "row-count invariant (see validate_alignment docstring) and print a "
                          "report. Use this to spot-check a few known-tricky structures (e.g. "
                          "tRNAs with insertion codes) before launching a full batch run.")
    args = ap.parse_args()

    if args.validate:
        if not args.pdb_ids:
            ap.error("--validate requires --pdb-ids <already-processed PDB IDs to check>")
        reports = [validate_alignment(args.outdir, pdb_id) for pdb_id in args.pdb_ids]
        for r in reports:
            log.info("%s", json.dumps(r, indent=2))
        n_bad = sum(1 for r in reports if not r.get("all_chains_consistent", False))
        log.info("Validation summary: %d/%d structures fully consistent",
                  len(reports) - n_bad, len(reports))
        return

    if args.nonredundant_csv:
        pdb_ids = parse_bgsu_nonredundant_csv(args.nonredundant_csv)
    elif args.pdb_ids:
        pdb_ids = args.pdb_ids
    else:
        ap.error("Provide --pdb-ids ... or --nonredundant-csv <path>")

    if args.limit:
        pdb_ids = pdb_ids[: args.limit]

    raw_dir = args.outdir / "raw_pdb"
    work_dir = args.outdir / "work"
    args.outdir.mkdir(parents=True, exist_ok=True)

    settings = ApbsSettings(ionic_strength_M=args.ionic_strength)

    all_results = []
    all_ions = []
    failures = []

    tasks = [(pdb_id, raw_dir, work_dir, settings, args.overwrite) for pdb_id in pdb_ids]

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = {ex.submit(_process_one, t): t[0] for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                pdb_id, result, err = fut.result()
                log.info("[%d/%d] %s done", i, len(pdb_ids), pdb_id)
                if err:
                    log.error("%s FAILED: %s", pdb_id, err[0])
                    failures.append((pdb_id, err[0], err[1]))
                elif result is not None:
                    all_results.append(result["df"])
                    ions_path = work_dir / pdb_id / "ions.csv"
                    if ions_path.exists():
                        all_ions.append(pd.read_csv(ions_path))
    else:
        for i, pdb_id in enumerate(pdb_ids, 1):
            log.info("[%d/%d] Processing %s", i, len(pdb_ids), pdb_id)
            try:
                result = process_structure(pdb_id, raw_dir, work_dir, settings, args.overwrite)
                if result is not None:
                    all_results.append(result["df"])
                    ions_path = work_dir / pdb_id.upper() / "ions.csv"
                    if ions_path.exists():
                        all_ions.append(pd.read_csv(ions_path))
            except Exception as e:
                log.error("%s FAILED: %s", pdb_id, e)
                failures.append((pdb_id, str(e), traceback.format_exc()))

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.outdir / "all_residue_potentials.csv", index=False)
        log.info("Wrote %d residue rows across %d structures to %s",
                  len(combined), combined["pdb_id"].nunique(),
                  args.outdir / "all_residue_potentials.csv")

    if all_ions:
        combined_ions = pd.concat(all_ions, ignore_index=True)
        combined_ions.to_csv(args.outdir / "all_ions.csv", index=False)
        log.info("Wrote %d resolved ion rows to %s", len(combined_ions),
                  args.outdir / "all_ions.csv")

    n_success = len(pdb_ids) - len(failures)
    log.info("Run summary: attempted=%d succeeded=%d failed=%d",
              len(pdb_ids), n_success, len(failures))

    if failures:
        fail_path = args.outdir / "failures.log"
        with open(fail_path, "w") as f:
            for pdb_id, msg, tb in failures:
                f.write(f"=== {pdb_id} ===\n{msg}\n{tb}\n\n")
        log.warning("%d/%d structures failed; see %s", len(failures), len(pdb_ids), fail_path)


if __name__ == "__main__":
    main()