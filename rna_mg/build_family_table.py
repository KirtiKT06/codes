#!/usr/bin/env python
"""
Assign every labelled RNA chain to a FAMILY, for the leave-family-out test.

Output CSV: pdb_id, chain, family, source   (chain = the internal chain letter in your labels / FASTA)

Two ways to define families
  --method cmscan   REAL Rfam families. Runs Infernal's `cmscan` against Rfam.cm (RhoFold's database build
                    already downloads and presses Rfam.cm; you also need the `cmscan` program:
                    `conda install -c bioconda infernal`). Chains with no Rfam hit fall back to
                    sequence clusters (below) unless --no-fallback.
  --method kmer     Sequence-similarity groups, no external program. Chains that share many more 6-mers
                    than chance are linked; connected groups become "families". These are similarity
                    clusters, NOT curated families -- say so in the paper, or use cmscan.

Usage
    python build_family_table.py --labels-dir /data/rna_electrostats/phi_res --out families_kmer.csv
    python build_family_table.py --labels-dir /data/rna_electrostats/phi_res --out families_rfam.csv \
        --method cmscan --rfam-cm /data/RhoFold/database/Rfam.cm --cpu 16
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_family_table")

K = 6
LUT = {"A": 0, "C": 1, "G": 2, "U": 3}


def read_chains(labels_dir: Path):
    """[(pdb_id, chain, seq)] from work/<ID>/<id>.fasta (headers '>ID_chain')."""
    out = []
    for fa in sorted((labels_dir / "work").glob("*/*.fasta")):
        hdr, seq = None, []
        for ln in fa.read_text().splitlines() + [">END"]:
            if ln.startswith(">"):
                if hdr is not None:
                    pid, _, ch = hdr[1:].rpartition("_")
                    out.append((pid or fa.parent.name, ch, "".join(seq)))
                hdr, seq = ln, []
            else:
                seq.append(ln.strip())
    return out


def kmer_matrix(seqs, k=K):
    from scipy import sparse
    rows, cols = [], []
    for i, s in enumerate(seqs):
        ids = set()
        code = [LUT.get(c, -1) for c in s.upper().replace("T", "U")]
        for j in range(len(code) - k + 1):
            w = code[j:j + k]
            if -1 in w:
                continue
            v = 0
            for x in w:
                v = v * 4 + x
            ids.add(v)
        rows += [i] * len(ids)
        cols += sorted(ids)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(seqs), 4 ** k))


def kmer_clusters(seqs, threshold=0.3, min_shared=12, k=K):
    """Connected components of the 'shares clearly more k-mers than chance' graph.
    score = (shared - expected_by_chance) / (smaller_set - expected_by_chance)."""
    from scipy.sparse.csgraph import connected_components
    from scipy import sparse
    M = kmer_matrix(seqs, k)
    size = np.asarray(M.sum(axis=1)).ravel()
    inter = (M @ M.T).toarray()
    small = np.minimum(size[:, None], size[None, :])
    big = np.maximum(size[:, None], size[None, :])
    expected = small * big / float(4 ** k)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (inter - expected) / (small - expected)
    adj = (score >= threshold) & (inter >= min_shared) & (small >= min_shared)
    np.fill_diagonal(adj, False)
    n_comp, labels = connected_components(sparse.csr_matrix(adj), directed=False)
    return labels


def run_cmscan(chains, rfam_cm: Path, cmscan: str, cpu: int):
    """{ 'PDB_chain': 'RFxxxxx' } best-scoring Rfam hit per chain (GA-thresholded)."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        q = td / "q.fa"
        q.write_text("".join(f">{p}_{c}\n{s}\n" for p, c, s in chains))
        tbl = td / "out.tbl"
        cmd = [cmscan, "--rfam", "--cut_ga", "--nohmmonly", "--cpu", str(cpu), "--tblout", str(tbl),
               "-o", "/dev/null", str(rfam_cm), str(q)]
        log.info("running: %s", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"cmscan failed:\n{r.stderr[-1500:]}")
        return parse_tblout(tbl)


def parse_tblout(path: Path) -> dict:
    best = {}
    for ln in Path(path).read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split()
        if len(f) < 17:
            continue
        acc, query = f[1], f[2]
        try:
            score = float(f[14])
        except ValueError:
            continue
        if f[16] == "?":                       # below inclusion threshold
            continue
        if query not in best or score > best[query][0]:
            best[query] = (score, acc)
    return {q: a for q, (s, a) in best.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", choices=["kmer", "cmscan"], default="kmer")
    ap.add_argument("--threshold", type=float, default=0.3, help="kmer: link score (higher = smaller groups)")
    ap.add_argument("--min-shared", type=int, default=12, help="kmer: minimum shared 6-mers to link")
    ap.add_argument("--rfam-cm", type=Path, default=None)
    ap.add_argument("--cmscan", default="cmscan")
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--no-fallback", action="store_true", help="cmscan: leave unannotated chains out of the table")
    a = ap.parse_args()

    chains = read_chains(a.labels_dir)
    if not chains:
        raise SystemExit(f"no FASTA files under {a.labels_dir}/work")
    log.info("%d chains", len(chains))
    ids = [f"{p}_{c}" for p, c, _ in chains]

    fam = {}
    src = {}
    if a.method == "cmscan":
        if a.rfam_cm is None:
            raise SystemExit("--rfam-cm is required for --method cmscan")
        hits = run_cmscan(chains, a.rfam_cm, a.cmscan, a.cpu)
        for i in ids:
            if i in hits:
                fam[i], src[i] = hits[i], "rfam"
        log.info("%d of %d chains have an Rfam hit", len(fam), len(ids))

    todo = [i for i in range(len(chains)) if ids[i] not in fam] if not (a.method == "cmscan" and a.no_fallback) else []
    if todo:
        lab = kmer_clusters([chains[i][2] for i in todo], a.threshold, a.min_shared)
        sizes = pd.Series(lab).value_counts()
        rank = {l: r for r, l in enumerate(sizes.index)}
        for i, l in zip(todo, lab):
            fam[ids[i]] = f"kmer_{rank[l]:04d}"
            src[ids[i]] = "kmer"
        log.info("k-mer groups: %d groups for %d chains; largest group %d chains (%.1f%%)",
                 len(sizes), len(todo), int(sizes.iloc[0]), 100 * sizes.iloc[0] / len(todo))
        if sizes.iloc[0] / len(todo) > 0.3:
            log.warning("one group holds >30%% of chains -- raise --threshold, or use --method cmscan")

    rows = [{"pdb_id": p, "chain": c, "family": fam[f"{p}_{c}"], "source": src[f"{p}_{c}"]}
            for p, c, _ in chains if f"{p}_{c}" in fam]
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    top = out.family.value_counts().head(8)
    log.info("wrote %d rows, %d families -> %s\nlargest families:\n%s", len(out), out.family.nunique(), a.out, top.to_string())


if __name__ == "__main__":
    main()
