"""
Find structures hit by the residue-numbering collision bug and build a
"clean subset" labels directory that every existing script can be pointed
at with no code changes (they all take --labels-dir and read the structure
list from all_residue_potentials.csv).

THE BUG: renumber_pdb_residues() used to key residues on
(chain, resSeq, icode) only, so two different molecules sharing a chain ID
AND numbering (e.g. both strands of a duplex written as chain "A", 1-7;
PDB 2VAL) received the SAME synthetic ids. Sequences were then built by
sorting on resnum, interleaving the strands into a scrambled "sequence"
that every LM / k-mer baseline / ViennaRNA fold was fed; and when the
colliding residues also shared a resname they merged into one residue.

(v2.1) A second cause was found: pdb2pqr was run without --keep-chain, so
the chain letters assigned by the renumbering step were dropped from the
PQR and every atom fell back to chain "A". The pipeline now passes
--keep-chain; this script's PQR reader raises on a missing chain column
instead of guessing, and reports such files as unparsed.

TWO detectors (a structure is flagged if either fires):
  csv   -- repeated (chain, resnum) rows in all_residue_potentials.csv.
           Catches collisions where the resnames differ (the residues
           survive as separate rows with the same key).
  pqr   -- the same (chain, resSeq) appearing in more than one CONTIGUOUS
           run of atoms in work/<ID>/<id>.pqr. Also catches the case the
           CSV cannot see: colliding residues with the SAME resname, which
           collapse into a single row and leave no duplicate behind.
           (Atom order in the PQR is file order, so a residue is
           contiguous and a repeated key means a second molecule.)

Output (--out-dir):
  all_residue_potentials.csv   only the unflagged structures
  all_ions.csv                 likewise, if present
  work/, raw_pdb/              symlinks to the originals (nothing copied)
  structure_integrity_report.csv   per-structure detector results

Usage:
    python make_clean_subset.py --labels-dir /data/rna_mg/res_phi \
        --out-dir /data/rna_mg/res_phi_clean
then run any existing script with --labels-dir /data/rna_mg/res_phi_clean
(and a new --outdir). Note the subset is NOT a random sample -- it excludes
multi-strand entries -- so report it as a sensitivity analysis, not as a
replacement dataset.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("make_clean_subset")


def split_pqr_line(line: str):
    """Parse ONE PQR ATOM/HETATM line. Returns
    (name, resname, chain, resseq, x, y, z, q, r), or None if the line
    has an unrecognized layout. Raises ValueError if the chain column is
    missing (pdb2pqr run without --keep-chain) -- we never guess "A".

    Strategy (v2.2):
      1. FIXED-WIDTH first. pdb2pqr writes PDB-style columns (name 12-16,
         resname 17-20, chain 21, resSeq 22-26, x/y/z 30-54, charge 54-62,
         radius 62-69). Reading by column is immune to the two things that
         broke whitespace splitting: a chain letter/digit fused to a
         4-digit residue number ("01000" = chain '0', residue 1000) and
         fused negative coordinates ("-194.332-152.767"). Accepted only if
         every sanity check passes (blank spacer columns, integer resSeq,
         floats parse); otherwise fall through to step 2.
      2. WHITESPACE fallback, for lines whose columns are shifted (e.g. a
         6-digit atom serial). 11 fields = chain present. 10 fields =
         the chain is fused to resSeq, which is only possible when the
         token is exactly 5 characters (chain + 4-digit resSeq); any
         shorter token is a bare residue number, i.e. NO chain column.
    """
    # ---- 1. fixed-width ------------------------------------------------
    if len(line.rstrip("\n")) >= 69 and line[11] == " " and line[16] == " " and line[20] == " ":
        try:
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            resseq = line[22:26].strip()
            int(resseq)
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            q, r = float(line[54:62]), float(line[62:69])
        except ValueError:
            pass
        else:
            if chain == " ":
                raise ValueError(
                    "PQR line has no chain column - was pdb2pqr run with --keep-chain? "
                    f"offending line: {line.rstrip()}"
                )
            return name, resname, chain, resseq, str(x), str(y), str(z), str(q), str(r)

    # ---- 2. whitespace fallback ---------------------------------------
    line = re.sub(r"(\d\.\d+)(-\d+\.\d+)", r"\1 \2", line)  # un-fuse "1.234-5.678"
    p = line.split()
    if len(p) == 11:
        _, _, name, resname, chain, resseq, x, y, z, q, r = p
    elif len(p) == 10:
        _, _, name, resname, fused, x, y, z, q, r = p
        if len(fused) == 5 and re.fullmatch(r"\d{4}", fused[1:]):   # chain + 4-digit resSeq
            chain, resseq = fused[0], fused[1:]
        else:
            raise ValueError(
                "PQR line has no chain column - was pdb2pqr run with --keep-chain? "
                f"offending line: {line.rstrip()}"
            )
    else:
        return None
    return name, resname, chain, resseq, x, y, z, q, r


def pqr_repeated_keys(pqr_path: Path) -> tuple[int, int]:
    """(# of (chain, resSeq) keys occurring in more than one contiguous
    run, # of unparseable ATOM lines). Uses split_pqr_line, the same
    parser as the pipeline. A line with no chain column counts as
    unparseable rather than being assigned to chain "A"."""
    runs_seen: dict[tuple, int] = {}
    prev = None
    unparsed = 0
    with open(pqr_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                if line.startswith("TER"):
                    prev = None
                continue
            try:
                parsed = split_pqr_line(line)
            except ValueError:
                parsed = None
            if parsed is None:
                unparsed += 1
                continue
            key = (parsed[2], parsed[3])   # (chain, resseq)
            if key != prev:
                runs_seen[key] = runs_seen.get(key, 0) + 1
                prev = key
    return sum(1 for n in runs_seen.values() if n > 1), unparsed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    csv_path = args.labels_dir / "all_residue_potentials.csv"
    df = pd.read_csv(csv_path, dtype={"pdb_id": str}, low_memory=False)

    csv_repeats = df.groupby("pdb_id")[["chain", "resnum"]].apply(
        lambda g: int(g.duplicated(["chain", "resnum"]).sum()))
    rows = []
    for pdb_id in sorted(df["pdb_id"].unique()):
        pqr = args.labels_dir / "work" / pdb_id.upper() / f"{pdb_id.lower()}.pqr"
        if pqr.exists():
            n_runs, unparsed = pqr_repeated_keys(pqr)
        else:
            n_runs, unparsed = -1, -1
        rows.append({"pdb_id": pdb_id, "csv_repeated_keys": int(csv_repeats.get(pdb_id, 0)),
                     "pqr_repeated_runs": n_runs, "pqr_unparsed_lines": unparsed})
    report = pd.DataFrame(rows)
    report["flagged"] = (report["csv_repeated_keys"] > 0) | (report["pqr_repeated_runs"] > 0)
    report["pqr_missing"] = report["pqr_repeated_runs"] < 0

    n = len(report)
    log.info("structures: %d | flagged by csv: %d | flagged by pqr: %d | flagged by either: %d | "
             "pqr missing (csv-only check): %d",
             n, int((report.csv_repeated_keys > 0).sum()), int((report.pqr_repeated_runs > 0).sum()),
             int(report.flagged.sum()), int(report.pqr_missing.sum()))
    only_pqr = int(((report.pqr_repeated_runs > 0) & (report.csv_repeated_keys == 0)).sum())
    log.info("of those, %d were invisible to the CSV check (same-resname collisions)", only_pqr)
    if (report.pqr_unparsed_lines > 0).any():
        log.warning("%d structures had unparseable PQR lines (pqr detector may under-count there)",
                    int((report.pqr_unparsed_lines > 0).sum()))

    clean_ids = set(report.loc[~report.flagged, "pdb_id"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df[df["pdb_id"].isin(clean_ids)].to_csv(args.out_dir / "all_residue_potentials.csv", index=False)
    ions = args.labels_dir / "all_ions.csv"
    if ions.exists():
        idf = pd.read_csv(ions, dtype={"pdb_id": str}, low_memory=False)
        idf[idf["pdb_id"].isin(clean_ids)].to_csv(args.out_dir / "all_ions.csv", index=False)
    for sub in ("work", "raw_pdb"):
        src, dst = (args.labels_dir / sub).resolve(), args.out_dir / sub
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    report.to_csv(args.out_dir / "structure_integrity_report.csv", index=False)
    log.info("clean subset: %d of %d structures written to %s", len(clean_ids), n, args.out_dir)


if __name__ == "__main__":
    main()