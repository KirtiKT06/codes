#!/usr/bin/env python
"""
RNA class for every labelled chain, version 2.

Why v2: v1 matched a chain to an RCSB entity by chain LETTER only. For large structures (ribosomes) our chain
letters do not line up with RCSB author ids, so RNA chains received protein descriptions and 38 structures got nothing.
v2 (i) offers each chain only RNA-type entities, (ii) accepts a chain-id match only if the chain's sequence is contained
in the entity sequence (8-mer containment >= 0.6), (iii) otherwise searches all RNA entities of the entry by sequence,
and (iv) splits uninformative descriptions ("RNA (36-MER)") by length instead of lumping them into 'other'.
'hairpin' no longer counts as a ribozyme.

    python fetch_chain_classes_v2.py --labels-dir /data/rna_electrostats/phi_res --out /data/rna_electrostats/chain_classes.csv

Output: pdb_id, chain, key, n_res, description, rna_class, source. source is
  rcsb-id   matched by chain id and confirmed by sequence
  rcsb-seq  matched by sequence only (chain ids did not match)
  length    no RCSB entity matched; length class only
Cache: <out>.rcsb_v2.json (raw entities per structure), so reruns are instant.
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
  entity_poly{rcsb_entity_polymer_type pdbx_seq_one_letter_code_can}
  rcsb_polymer_entity_container_identifiers{auth_asym_ids asym_ids}}}}"""

RULES = [
    ("rRNA", r"\b(5s|5\.8s|12s|16s|18s|21s|23s|25s|26s|28s|rrna)\b|ribosomal rna|r-rna"),
    ("tRNA", r"trna|transfer rna|t-rna"),
    ("riboswitch", r"riboswitch"),
    ("group II intron", r"group ii intron|group 2 intron|\bgroup ii\b"),
    ("group I intron", r"group i intron|group 1 intron|\bgroup i\b"),
    ("ribozyme / RNase P", r"ribozyme|hammerhead|\bhdv\b|delta virus|twister|pistol|hatchet|rnase p|rnase mrp|glms"),
    ("guide RNA (CRISPR)", r"guide rna|crrna|sgrna|tracrrna|crispr"),
    ("snRNA / spliceosome", r"snrna|\bu[1-6]\b|spliceosom|\bu12\b|\bu4atac\b|\bu6atac\b"),
    ("SRP RNA", r"\bsrp\b|signal recognition"),
    ("telomerase / TERC", r"telomerase|\bterc\b"),
    ("aptamer", r"aptamer"),
    ("toxin-antitoxin RNA", r"antitoxin|\btoxi\b|toxin"),
    ("mRNA / viral / IRES", r"mrna|messenger|\bires\b|viral|virus|hiv|hcv|sars|frameshift|pseudoknot|5'-?utr|3'-?utr|\bene\b"),
    ("RNA duplex / model", r"duplex|hybrid|oligonucleotide|synthetic|model rna|designed"),
]
GENERIC = re.compile(r"(rna|non-?coding rna|ncrna|rna \(\d+-mer\)|rna \(.*mer\)|rna sequence|rna chain|"
                     r"rna strand( \w+)?|unknown|unidentified.*|rna fragment)")


def length_bucket(n: int) -> str:
    return "<=30 nt" if n <= 30 else "31-100 nt" if n <= 100 else "101-300 nt" if n <= 300 else ">300 nt"


def classify(desc: str, n_res: int) -> str:
    d = (desc or "").lower().strip()
    if d and not GENERIC.fullmatch(d):
        for name, pat in RULES:
            if re.search(pat, d):
                return name
        return "short oligo / duplex (<=30 nt)" if n_res <= 30 else "other named ncRNA"
    return f"unspecified RNA ({length_bucket(n_res)})"


def norm(s: str) -> str:
    return (s or "").upper().replace("U", "T")


def kmers(s: str, k: int = 8) -> set:
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def containment(q: set, ref: set) -> float:
    return len(q & ref) / len(q) if q else 1.0


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


def fetch_batch(ids, retries: int = 3) -> dict:
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


def parse_entries(resp: dict) -> dict[str, list[dict]]:
    out = {}
    for e in (resp.get("data") or {}).get("entries") or []:
        if not e:
            continue
        ents = []
        for pe in e.get("polymer_entities") or []:
            ep = pe.get("entity_poly") or {}
            ids = pe.get("rcsb_polymer_entity_container_identifiers") or {}
            ents.append({"desc": (pe.get("rcsb_polymer_entity") or {}).get("pdbx_description") or "",
                         "type": ep.get("rcsb_entity_polymer_type") or "",
                         "seq": norm(ep.get("pdbx_seq_one_letter_code_can") or ""),
                         "auth": ids.get("auth_asym_ids") or [], "label": ids.get("asym_ids") or []})
        out[e["rcsb_id"].upper()] = ents
    return out


def assign(chain: str, seq: str, entities: list[dict], thr: float = 0.6):
    """(entity, how) or (None, 'length'). Protein entities are never offered."""
    q = kmers(norm(seq))
    rna = [e for e in entities if e["type"] != "Protein"]
    for e in rna:
        if chain in e["auth"] or chain in e["label"]:
            if len(norm(seq)) < 8 or containment(q, e.setdefault("_k", kmers(e["seq"]))) >= thr:
                return e, "rcsb-id"
    best, bc = None, thr
    for e in rna:
        c = containment(q, e.setdefault("_k", kmers(e["seq"])))
        if c >= bc:
            best, bc = e, c
    return (best, "rcsb-seq") if best else (None, "length")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, default=Path("/data/rna_electrostats/phi_res"))
    ap.add_argument("--out", type=Path, default=Path("/data/rna_electrostats/chain_classes.csv"))
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()
    chains = read_fastas(a.labels_dir)
    if not chains:
        raise SystemExit(f"no work/<ID>/<id>.fasta under {a.labels_dir}")
    ids = sorted({c[0] for c in chains})
    print(f"{len(chains)} chains in {len(ids)} structures")
    cache = a.out.with_suffix(".rcsb_v2.json")
    ents = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [i for i in ids if i not in ents]
    print(f"{len(ents)} cached, {len(todo)} to fetch")
    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        try:
            got = parse_entries(fetch_batch(batch))
        except Exception as e:  # noqa: BLE001
            print(f"  batch {i // a.batch} failed ({e}); those chains get the length class")
            continue
        for pid in batch:
            ents[pid] = got.get(pid, [])
        print(f"  fetched {min(i + a.batch, len(todo))}/{len(todo)}")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(ents))
    rows = []
    for pdb, ch, seq in chains:
        e, how = assign(ch, seq, [dict(x) for x in ents.get(pdb, [])]) if ents.get(pdb) else (None, "length")
        desc = e["desc"] if e else ""
        cls = classify(desc, len(seq)) if e else f"unspecified RNA ({length_bucket(len(seq))})"
        rows.append({"pdb_id": pdb, "chain": ch, "key": f"{pdb}::{ch}", "n_res": len(seq), "description": desc,
                     "rna_class": cls, "source": how})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    print(df.groupby(["rna_class", "source"]).size().sort_values(ascending=False).to_string())
    print(f"matched to an RCSB RNA entity: {(df.source != 'length').mean():.1%}   "
          f"(by chain id: {(df.source == 'rcsb-id').mean():.1%}, by sequence only: {(df.source == 'rcsb-seq').mean():.1%})")


if __name__ == "__main__":
    sys.exit(main())
