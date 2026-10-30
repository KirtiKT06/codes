#!/usr/bin/env python
"""
PDB release date for every labelled chain, so the pretrained advantage can be split by when a structure became public
(structures released after a model's pretraining cutoff cannot have been in its corpus).

    python fetch_release_dates.py --classes /data/rna_electrostats/chain_classes.csv \
                                  --out /data/rna_electrostats/chain_release.csv

Needs internet (RCSB GraphQL, ~20 requests of 100 structures; cached next to --out). Output columns: key (PDB::chain),
pdb_id, chain, release_date, release_year, release_bin. The bins are deliberately coarse; check each model's paper for
its actual corpus date and adjust --edges if needed (default edges: 2020, 2022, 2024 -> '<=2020', '2021-2022',
'2023-2024', '>=2025').
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

GRAPHQL = "https://data.rcsb.org/graphql"
QUERY = "query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id rcsb_accession_info{initial_release_date}}}"


def fetch(ids, retries=3):
    body = json.dumps({"query": QUERY, "variables": {"ids": ids}}).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(GRAPHQL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"RCSB request failed: {last}")


def parse(resp: dict) -> dict[str, str]:
    out = {}
    for e in (resp.get("data") or {}).get("entries") or []:
        if e:
            out[e["rcsb_id"].upper()] = ((e.get("rcsb_accession_info") or {}).get("initial_release_date") or "")[:10]
    return out


def bin_year(y: float, edges) -> str:
    if y != y:
        return "unknown"
    e = sorted(edges)
    if y <= e[0]:
        return f"<={e[0]}"
    for lo, hi in zip(e[:-1], e[1:]):
        if lo < y <= hi:
            return f"{lo + 1}-{hi}"
    return f">={e[-1] + 1}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", type=Path, default=Path("/data/rna_electrostats/chain_classes.csv"))
    ap.add_argument("--out", type=Path, default=Path("/data/rna_electrostats/chain_release.csv"))
    ap.add_argument("--edges", type=int, nargs="+", default=[2020, 2022, 2024])
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args()
    c = pd.read_csv(a.classes, usecols=["pdb_id", "chain", "key"])
    ids = sorted(c.pdb_id.str.upper().unique())
    cache = a.out.with_suffix(".cache.json")
    dates = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [i for i in ids if i not in dates]
    print(f"{len(ids)} structures; {len(todo)} to fetch")
    for i in range(0, len(todo), a.batch):
        b = todo[i:i + a.batch]
        try:
            got = parse(fetch(b))
        except Exception as e:  # noqa: BLE001
            print(f"  batch failed ({e}); continuing")
            continue
        for p in b:
            dates[p] = got.get(p, "")
        cache.write_text(json.dumps(dates))
        print(f"  fetched {min(i + a.batch, len(todo))}/{len(todo)}")
    c["release_date"] = c.pdb_id.str.upper().map(dates).fillna("")
    c["release_year"] = pd.to_datetime(c.release_date, errors="coerce").dt.year
    c["release_bin"] = [bin_year(y, a.edges) for y in c.release_year]
    c.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    print(c.release_bin.value_counts().sort_index().to_string())
    print(f"structures without a date: {(c.release_date == '').mean():.1%} of chains")


if __name__ == "__main__":
    sys.exit(main())
