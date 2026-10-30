#!/usr/bin/env python
"""
Give every labelled chain an RNA CLASS (rRNA, tRNA, riboswitch, ribozyme, intron, mRNA, ...), taken from the RCSB
entity descriptions (e.g. "16S ribosomal RNA", "tRNA-Phe", "group II intron").

    python fetch_chain_classes.py --labels-dir /data/rna_electrostats/phi_res \
                                  --out /data/rna_electrostats/chain_classes.csv

Output columns: pdb_id, chain, key (PDB::chain, the key used by audit_chain_stats), n_res, description, rna_class,
source ('rcsb' or 'length'). Needs internet for the RCSB call (one GraphQL request per 50 structures, ~40 requests);
responses are cached in <out>.rcsb_cache.json so a rerun is instant. With --offline only a coarse length class is
assigned (no internet needed), which is a much weaker breakdown.

Chain letters in the FASTA headers (>ID_chain) are matched to RCSB auth_asym_ids. Chains that cannot be matched fall
back to the length class and are marked source = 'length' so you can see how many were annotated.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

GRAPHQL = "https://data.rcsb.org/graphql"
QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id polymer_entities{
  rcsb_polymer_entity{pdbx_description}
  entity_poly{rcsb_entity_polymer_type}
  rcsb_polymer_entity_container_identifiers{auth_asym_ids asym_ids}}}}"""

# ordered rules: first match wins
RULES = [
    ("rRNA", r"\b(5s|5\.8s|12s|16s|18s|21s|23s|25s|26s|28s|rrna)\b|ribosomal rna|r-rna"),
    ("tRNA", r"trna|transfer rna|t-rna"),
    ("riboswitch", r"riboswitch"),
    ("group II intron", r"group ii intron|group 2 intron|\bgroup ii\b"),
    ("group I intron", r"group i intron|group 1 intron|\bgroup i\b"),
    ("ribozyme / RNase P", r"ribozyme|hammerhead|hairpin|hdv|delta virus|twister|pistol|hatchet|rnase p|rnase mrp|glms"),
    ("guide RNA (CRISPR)", r"guide rna|crrna|sgrna|tracrrna|crispr|\bsgrna\b"),
    ("snRNA / spliceosome", r"snrna|\bu[1-6]\b|\bu[1-6] |spliceosom|\bu12\b|\bu4atac\b|\bu6atac\b"),
    ("SRP RNA", r"\bsrp\b|signal recognition"),
    ("telomerase / TERC", r"telomerase|\bterc\b|\btr\b rna"),
    ("aptamer", r"aptamer"),
    ("mRNA / viral / IRES", r"mrna|messenger|\bires\b|viral|virus|hiv|hcv|sars|frameshift|pseudoknot|5'-?utr|3'-?utr"),
    ("RNA duplex / model", r"duplex|hybrid|oligonucleotide|synthetic|model rna|designed"),
]


def classify(desc: str, n_res: int) -> tuple[str, str]:
    d = (desc or "").lower()
    if d.strip():
        for name, pat in RULES:
            if re.search(pat, d):
                return name, "rcsb"
        if n_res <= 30:
            return "short oligo / duplex (<=30 nt)", "rcsb"
        return "other (annotated)", "rcsb"
    return length_class(n_res), "length"


def length_class(n: int) -> str:
    if n <= 30:
        return "short (<=30 nt)"
    if n <= 100:
        return "31-100 nt"
    if n <= 300:
        return "101-300 nt"
    return "long (>300 nt)"


def read_fastas(labels_dir: Path):
    rows = []
    for fa in sorted((labels_dir / "work").glob("*/*.fasta")):
        name, seq = None, []
        for ln in fa.read_text().splitlines() + [">END"]:
            if ln.startswith(">"):
                if name is not None:
                    pdb, _, ch = name.partition("_")
                    rows.append((pdb.upper(), ch, "".join(seq)))
                name, seq = ln[1:].strip(), []
            else:
                seq.append(ln.strip())
    return rows


def fetch_batch(ids: list[str], retries: int = 3) -> dict:
    body = json.dumps({"query": QUERY, "variables": {"ids": ids}}).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(GRAPHQL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"RCSB request failed: {last}")


def parse_entries(resp: dict) -> dict[tuple[str, str], str]:
    """{(PDB, auth_chain): description} from a GraphQL response."""
    out = {}
    for e in (resp.get("data") or {}).get("entries") or []:
        if not e:
            continue
        for pe in e.get("polymer_entities") or []:
            desc = ((pe.get("rcsb_polymer_entity") or {}).get("pdbx_description")) or ""
            ids = (pe.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids") or []
            for ch in ids:
                out[(e["rcsb_id"].upper(), ch)] = desc
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, default=Path("/data/rna_electrostats/phi_res"))
    ap.add_argument("--out", type=Path, default=Path("/data/rna_electrostats/chain_classes.csv"))
    ap.add_argument("--offline", action="store_true", help="length classes only, no internet")
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()

    chains = read_fastas(a.labels_dir)
    if not chains:
        raise SystemExit(f"no work/<ID>/<id>.fasta files under {a.labels_dir}")
    print(f"{len(chains)} chains in {len({c[0] for c in chains})} structures")
    desc: dict[tuple[str, str], str] = {}
    cache = a.out.with_suffix(".rcsb_cache.json")
    if not a.offline:
        cached = {}
        if cache.exists():
            cached = {tuple(k.split("|")): v for k, v in json.loads(cache.read_text()).items()}
        desc.update(cached)
        ids = sorted({c[0] for c in chains})
        have = {k[0] for k in cached}
        todo = [i for i in ids if i not in have]
        print(f"{len(have)} structures cached, {len(todo)} to fetch")
        for i in range(0, len(todo), a.batch):
            batch = todo[i:i + a.batch]
            try:
                got = parse_entries(fetch_batch(batch))
            except Exception as e:  # noqa: BLE001
                print(f"  batch {i // a.batch} failed ({e}); continuing, those chains get the length class")
                continue
            desc.update(got)
            for pid in batch:                       # remember structures with no polymer entry too
                desc.setdefault((pid, ""), "")
            print(f"  fetched {min(i + a.batch, len(todo))}/{len(todo)}")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"|".join(k): v for k, v in desc.items()}))
    rows = []
    for pdb, ch, seq in chains:
        d = desc.get((pdb, ch), "")
        cls, src = classify(d, len(seq))
        rows.append({"pdb_id": pdb, "chain": ch, "key": f"{pdb}::{ch}", "n_res": len(seq), "description": d,
                     "rna_class": cls, "source": src})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    print(df.groupby(["rna_class", "source"]).size().sort_values(ascending=False).to_string())
    print(f"annotated from RCSB: {(df.source == 'rcsb').mean():.1%} of chains")


if __name__ == "__main__":
    sys.exit(main())
