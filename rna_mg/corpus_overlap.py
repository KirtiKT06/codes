#!/usr/bin/env python
"""
How close is each labelled chain to the sequences a language model may have been pretrained on?

    python corpus_overlap.py --labels-dir /data/rna_electrostats/phi_res --reference <FASTA or mmseqs DB> \
                             --out /data/rna_electrostats/corpus_overlap_rnacentral.csv --threads 16 \
                             [--only-test-chains /data/rna_electrostats/controls_matched/pretrained/ernierna]

Nucleotide search with MMseqs2 (U is converted to T on both sides). Output per chain: best identity and query
coverage of the best hit, the best identity among hits covering >= 80% of the chain, whether a (near-)identical copy
exists, and the best identity after excluding near-identical copies.

REFERENCE CORPUS (checked against the model papers by the project owner):
  RNA-FM, ERNIE-RNA, RNABERT : RNAcentral (ncRNA sequences), e.g. rnacentral_active.fasta.gz (several GB compressed),
                               so ONE search serves all three models
  SpliceBERT                 : human pre-mRNA, so not comparable with PDB chains; skip
RNAcentral contains sequences taken from PDB entries, so a chain can match itself at 100%. Interpret
'best identity' together with 'best identity excluding near-identical copies' and with the release date of the
corpus (a 2025 structure cannot be in a 2022 corpus, but its homologues can).

BUILD THE REFERENCE ONCE (so searches are fast; U->T conversion is required for nucleotide search):
  zcat rnacentral_active.fasta.gz | sed '/^>/!s/U/T/g' | mmseqs createdb stdin rnacentralDB --dbtype 2
  mmseqs createindex rnacentralDB tmp_idx --search-type 3 --threads 16
  python corpus_overlap.py ... --reference rnacentralDB          (a prebuilt DB is used as it is)
A FASTA given as --reference is converted (U->T) into a temporary file first; fine for Rfam, slow for RNAcentral.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def read_fastas(labels_dir: Path):
    rows = []
    for fa in sorted((labels_dir / "work").glob("*/*.fasta")):
        name, seq = None, []
        for ln in fa.read_text().splitlines() + [">END"]:
            if ln.startswith(">"):
                if name is not None:
                    pdb, _, ch = name.partition("_")
                    rows.append((pdb.upper(), ch, "".join(seq).upper().replace("U", "T")))
                name, seq = ln[1:].strip(), []
            else:
                seq.append(ln.strip())
    return rows


def keys_from_runs(folders):
    keys = set()
    for f in folders:
        d = pd.read_csv(Path(f) / "audit_chain_stats.csv.gz", usecols=["pdb", "chain"])
        keys |= set(d.pdb.str.upper() + "::" + d.chain.str.split("::").str[-1])
    return keys


def convert_fasta(src: Path, dst: Path):
    op = gzip.open if str(src).endswith(".gz") else open
    with op(src, "rt") as f, open(dst, "w") as o:
        for ln in f:
            o.write(ln if ln.startswith(">") else ln.upper().replace("U", "T"))


def is_mmseqs_db(p: Path) -> bool:
    return p.with_name(p.name + ".dbtype").exists() or Path(str(p) + ".dbtype").exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, default=Path("/data/rna_electrostats/phi_res"))
    ap.add_argument("--reference", type=Path, required=True, help="FASTA(.gz) or a prebuilt mmseqs nucleotide DB")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mmseqs", default="mmseqs")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only-test-chains", nargs="*", default=None,
                    help="run folders with audit_chain_stats.csv.gz: restrict to the chains they contain")
    ap.add_argument("--evalue", default="1e-3")
    ap.add_argument("--kmer", default="12", help="k-mer length for the nucleotide prefilter (default 12; short RNAs need <= 12)")
    ap.add_argument("--split-memory-limit", default=None, help="e.g. 24G, to cap MMseqs2 memory on a big reference")
    ap.add_argument("--sensitivity", default="6.5")
    ap.add_argument("--keep-tmp", action="store_true")
    a = ap.parse_args()
    if shutil.which(a.mmseqs) is None:
        raise SystemExit(f"{a.mmseqs} not found; install MMseqs2 or pass --mmseqs /path/to/mmseqs")

    chains = read_fastas(a.labels_dir)
    if a.only_test_chains:
        keep = keys_from_runs(a.only_test_chains)
        chains = [c for c in chains if f"{c[0]}::{c[1]}" in keep]
    chains = [c for c in chains if len(c[2]) >= 12]
    print(f"{len(chains)} query chains")
    tmp = Path(tempfile.mkdtemp(prefix="corpus_overlap_"))
    q = tmp / "query.fasta"
    q.write_text("".join(f">{p}::{c}\n{s}\n" for p, c, s in chains))
    target = a.reference
    if not target.exists() and not is_mmseqs_db(target):
        raise SystemExit(f"ERROR: reference {target} does not exist. Build the database first "
                         f"(bash build_rnacentral_db.sh) or pass a FASTA file.")
    if not is_mmseqs_db(target):
        target = tmp / "reference_T.fasta"
        print("converting reference FASTA U->T ...")
        convert_fasta(a.reference, target)
    m8 = tmp / "hits.m8"
    cmd = [a.mmseqs, "easy-search", str(q), str(target), str(m8), str(tmp / "work"), "--search-type", "3",
           "-e", a.evalue, "-s", a.sensitivity, "-k", a.kmer, "--max-seqs", "300", "--threads", str(a.threads),
           "--format-output", "query,target,fident,qcov,tcov,alnlen,evalue,bits"]
    if a.split_memory_limit:
        cmd += ["--split-memory-limit", a.split_memory_limit]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    cols = ["query", "target", "fident", "qcov", "tcov", "alnlen", "evalue", "bits"]
    h = pd.read_csv(m8, sep="\t", names=cols) if m8.exists() and m8.stat().st_size else pd.DataFrame(columns=cols)
    rows = []
    for p, c, s in chains:
        key = f"{p}::{c}"
        g = h[h["query"] == key]
        r = {"pdb_id": p, "chain": c, "key": key, "n_res": len(s), "n_hits": len(g), "best_fident": 0.0, "best_qcov": 0.0,
             "best_fident_cov80": 0.0, "near_identical_copy": False, "best_fident_excl_identical": 0.0,
             "n_hits_ge90_cov80": 0, "n_hits_ge70_cov50": 0}
        if len(g):
            top = g.sort_values("bits", ascending=False).iloc[0]
            r["best_fident"], r["best_qcov"] = float(top.fident), float(top.qcov)
            cov = g[g.qcov >= 0.8]
            r["best_fident_cov80"] = float(cov.fident.max()) if len(cov) else 0.0
            ident = (g.fident >= 0.995) & (g.qcov >= 0.95)
            r["near_identical_copy"] = bool(ident.any())
            rest = g[~ident]
            rest = rest[rest.qcov >= 0.5]
            r["best_fident_excl_identical"] = float(rest.fident.max()) if len(rest) else 0.0
            r["n_hits_ge90_cov80"] = int(((g.fident >= 0.9) & (g.qcov >= 0.8)).sum())
            r["n_hits_ge70_cov50"] = int(((g.fident >= 0.7) & (g.qcov >= 0.5)).sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    print(f"chains with a near-identical copy in the reference: {out.near_identical_copy.mean():.1%}")
    print(f"chains with a hit of >= 90% identity over >= 80% of the chain: {(out.n_hits_ge90_cov80 > 0).mean():.1%}")
    print(out.best_fident_excl_identical.describe().round(3).to_string())
    if not a.keep_tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
