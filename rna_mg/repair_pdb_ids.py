"""
Repair PDB IDs that pandas mangled into floats in all_residue_potentials.csv
(and all_ions.csv, if present).

Cause: the label pipeline re-reads cached per-structure CSVs with a
single-valued pdb_id column; an ID of the form <digit>E<digit><digit>
(e.g. 9E71) parses as the float 9e+71 and is written back out that way.
Downstream loaders then look for work/9E+71/, find nothing, and silently
drop the structure from every analysis.

Recovery is unambiguous: a 4-character ID d-E-n-n parses to d * 10**nn,
so the original is the leading digit, 'E', and the exponent zero-padded
to two digits ("9e+71" -> 9E71, "100000.0" -> 1E05). Every repaired ID is
validated against the work/<ID>/ directories that actually exist; anything
that cannot be validated is reported and left untouched, never guessed.

Usage:
    python repair_pdb_ids.py --labels-dir /data/rna_mg/res_phi
Writes <file>.bak before modifying anything.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

VALID_ID = re.compile(r"^[0-9][A-Z0-9]{3}$")


def recover_id(raw: str) -> str | None:
    """Best-effort inverse of float-mangling for d-E-nn IDs, else None."""
    raw = str(raw).strip()
    if VALID_ID.match(raw.upper()):
        return raw.upper()
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    digits = f"{value:.0f}" if value < 1e15 else f"{value:.0e}"
    m = re.match(r"^(\d)e\+?(\d+)$", digits.lower())
    if m:
        lead, exp = m.group(1), int(m.group(2))
    else:  # plain decimal like 100000 -> lead digit followed by zeros
        digits = digits.split(".")[0]
        if not re.match(r"^[1-9]0*$", digits):
            return None
        lead, exp = digits[0], len(digits) - 1
    if exp > 99:
        return None
    return f"{lead}E{exp:02d}"


def repair_file(csv_path: Path, work_dirs: set[str]) -> None:
    df = pd.read_csv(csv_path, dtype={"pdb_id": str}, low_memory=False)
    ids = df["pdb_id"].astype(str)
    bad = sorted(i for i in ids.unique() if not VALID_ID.match(i.upper()))
    if not bad:
        print(f"{csv_path.name}: all {ids.nunique()} pdb_ids already well-formed")
        return
    mapping, unresolved = {}, []
    for raw in bad:
        cand = recover_id(raw)
        if cand is not None and cand in work_dirs:
            mapping[raw] = cand
        else:
            unresolved.append((raw, cand))
    print(f"{csv_path.name}: {len(bad)} malformed ids -> repaired {len(mapping)}, unresolved {len(unresolved)}")
    for raw, cand in mapping.items():
        print(f"    {raw!r} -> {cand}")
    for raw, cand in unresolved:
        print(f"    UNRESOLVED {raw!r} (candidate {cand}, no matching work/ dir) -- left unchanged")
    shutil.copy(csv_path, str(csv_path) + ".bak")
    df["pdb_id"] = ids.map(lambda i: mapping.get(i, i))
    df.to_csv(csv_path, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    args = ap.parse_args()
    work = args.labels_dir / "work"
    work_dirs = {p.name for p in work.iterdir() if p.is_dir()} if work.exists() else set()
    for name in ("all_residue_potentials.csv", "all_ions.csv"):
        path = args.labels_dir / name
        if path.exists():
            repair_file(path, work_dirs)


if __name__ == "__main__":
    main()
