#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict, Counter
import json
import re

FAIL_LOG = Path("/data/rna_mg/cutoff_8/failures.log")

if not FAIL_LOG.exists():
    raise FileNotFoundError(f"{FAIL_LOG} not found")

text = FAIL_LOG.read_text(errors="ignore")

# ------------------------------------------------------------------
# Split into failure blocks
# ------------------------------------------------------------------

parts = re.split(
    r"^===\s+([A-Za-z0-9]+)\s+===\s*$",
    text,
    flags=re.MULTILINE,
)

entries = []

for i in range(1, len(parts), 2):
    pdb_id = parts[i]
    block = parts[i + 1]
    entries.append((pdb_id, block))

print(f"\nParsed {len(entries)} failed structures\n")

# ------------------------------------------------------------------
# Categorization
# ------------------------------------------------------------------

categories = defaultdict(list)

for pdb_id, block in entries:

    block_lower = block.lower()

    # ------------------------------------------------------------------
    # Missing residues / backbone
    # ------------------------------------------------------------------
    if (
        "too few atoms present to reconstruct or cap residue" in block_lower
        or "missing backbone atoms" in block_lower
    ):
        cat = "missing_atoms"

    # ------------------------------------------------------------------
    # PDB2PQR charge assignment failures
    # ------------------------------------------------------------------
    elif (
        "deviates by" in block_lower
        and "integral" in block_lower
    ):
        cat = "non_integral_charge"

    # ------------------------------------------------------------------
    # Huge structures
    # ------------------------------------------------------------------
    elif ">99999 atoms" in block:
        cat = "too_many_atoms"

    # ------------------------------------------------------------------
    # APBS failures
    # ------------------------------------------------------------------
    elif "apbs failed" in block_lower:
        cat = "apbs_failure"

    # ------------------------------------------------------------------
    # KeyError: chain
    # ------------------------------------------------------------------
    elif "'chain'" in block:
        cat = "chain_keyerror"

    # ------------------------------------------------------------------
    # Missing xyz dataframe columns
    # ------------------------------------------------------------------
    elif "['x', 'y', 'z']" in block:
        cat = "xyz_dataframe_bug"

    # ------------------------------------------------------------------
    # Download/parsing issues
    # ------------------------------------------------------------------
    elif (
        "404" in block_lower
        or "download" in block_lower
        or "raise_for_status" in block_lower
    ):
        cat = "download_error"

    elif "only" in block_lower and "RNA residue(s) remain" in block_lower:
        cat = "too_small_after_rna_extraction"

    # ------------------------------------------------------------------
    # Everything else
    # ------------------------------------------------------------------
    else:
        cat = "other"

    categories[cat].append(pdb_id)

# ------------------------------------------------------------------
# Print summary
# ------------------------------------------------------------------

total = len(entries)

print("=" * 70)
print("FAILURE SUMMARY")
print("=" * 70)

for cat, ids in sorted(
    categories.items(),
    key=lambda x: len(x[1]),
    reverse=True,
):
    n = len(ids)
    pct = 100 * n / total

    print(
        f"{cat:25s} "
        f"{n:4d} "
        f"({pct:5.1f}%) "
        f"example={ids[0]}"
    )

print("=" * 70)
print(f"TOTAL FAILURES: {total}")
print("=" * 70)

# ------------------------------------------------------------------
# Save detailed report
# ------------------------------------------------------------------

summary = {
    cat: {
        "count": len(ids),
        "examples": ids[:10],
        "all_ids": ids,
    }
    for cat, ids in categories.items()
}

with open("failure_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nDetailed report written to:")
print("failure_summary.json\n")