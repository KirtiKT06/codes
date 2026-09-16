"""
Build sequence-identity clusters for the RNA structures using MMseqs2
(preferred) or CD-HIT-EST (fallback), replacing the quick k-mer-Jaccard
approximation used elsewhere in this project with an actual
alignment-based identity clustering. Produces a CSV mapping
chain_id ('pdb_id::chain') -> cluster_rep, which
rna_probe_priority_experiments.py can load via --cluster-assignments-csv
for a stricter Priority 3 split (whole sequence families held out
together, not just near-duplicates caught by a k-mer proxy).

Requires ONE of, on PATH:
    mmseqs2    (conda install -c bioconda mmseqs2)   -- preferred, faster
    cd-hit-est (conda install -c bioconda cd-hit)     -- fallback

Usage:
    python mmseqs_cluster.py --labels-dir /data/rna_mg/cutoff_4_8_12 \
        --outdir /data/rna_mg/clustering --identity 0.8 --tool mmseqs

Then, in rna_probe_priority_experiments.py:
    --cluster-assignments-csv /data/rna_mg/clustering/chain_cluster_assignments.csv
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("mmseqs_cluster")

from rna_embedding_probe_v2 import load_fasta_by_chain  # noqa: E402


def gather_combined_fasta(labels_dir: Path, pdb_ids: list[str], out_fasta: Path) -> int:
    """One combined multi-FASTA across every chain of every structure,
    headers named 'pdb_id::chain' to match the rest of the pipeline's
    chain-id convention."""
    n = 0
    with open(out_fasta, "w") as f:
        for pdb_id in pdb_ids:
            seqs = load_fasta_by_chain(labels_dir, pdb_id)
            for chain, seq in seqs.items():
                if not seq:
                    continue
                f.write(f">{pdb_id}::{chain}\n{seq}\n")
                n += 1
    return n


def run_mmseqs(fasta_path: Path, workdir: Path, identity: float, coverage: float = 0.8) -> Path:
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "mmseqs not found on PATH -- install via `conda install -c bioconda mmseqs2`, "
            "or rerun with --tool cdhit if you have cd-hit-est instead."
        )
    workdir.mkdir(parents=True, exist_ok=True)
    prefix = workdir / "clusterRes"
    tmp = workdir / "tmp"
    cmd = ["mmseqs", "easy-cluster", str(fasta_path), str(prefix), str(tmp),
           "--min-seq-id", str(identity), "-c", str(coverage), "--cov-mode", "0"]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mmseqs failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    cluster_tsv = Path(str(prefix) + "_cluster.tsv")
    if not cluster_tsv.exists():
        raise RuntimeError(f"Expected output {cluster_tsv} not found -- check mmseqs stdout/stderr above")
    return cluster_tsv


def run_cdhit(fasta_path: Path, workdir: Path, identity: float) -> Path:
    if shutil.which("cd-hit-est") is None:
        raise RuntimeError(
            "cd-hit-est not found on PATH -- install via `conda install -c bioconda cd-hit`, "
            "or rerun with --tool mmseqs if you have mmseqs2 instead."
        )
    workdir.mkdir(parents=True, exist_ok=True)
    out_prefix = workdir / "cdhit_out"
    # word size must roughly match the identity band -- cd-hit-est's own convention
    word_size = 8 if identity >= 0.8 else (5 if identity >= 0.7 else 4)
    cmd = ["cd-hit-est", "-i", str(fasta_path), "-o", str(out_prefix), "-c", str(identity),
           "-n", str(word_size), "-M", "4000", "-T", "0"]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cd-hit-est failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    clstr_path = Path(str(out_prefix) + ".clstr")
    if not clstr_path.exists():
        raise RuntimeError(f"Expected output {clstr_path} not found -- check cd-hit-est stdout/stderr above")
    return clstr_path


def parse_mmseqs_clusters(cluster_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["cluster_rep", "chain_id"])
    return df


def parse_cdhit_clusters(clstr_path: Path) -> pd.DataFrame:
    rows = []
    current_cluster = None
    with open(clstr_path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                current_cluster = line.strip().split()[-1]
            else:
                member = line.split(">")[1].split("...")[0]
                rows.append({"cluster_rep": current_cluster, "chain_id": member})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("./clustering"))
    ap.add_argument("--identity", type=float, default=0.8, help="Sequence identity threshold (0-1)")
    ap.add_argument("--tool", choices=["mmseqs", "cdhit"], default="mmseqs")
    ap.add_argument("--limit-structures", type=int, default=None)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_pot_path = args.labels_dir / "all_residue_potentials.csv"
    pdb_ids = sorted(pd.read_csv(all_pot_path, low_memory=False)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]

    fasta_path = args.outdir / "all_chains.fasta"
    n = gather_combined_fasta(args.labels_dir, pdb_ids, fasta_path)
    log.info("Wrote %d chain sequences to %s", n, fasta_path)

    if args.tool == "mmseqs":
        cluster_file = run_mmseqs(fasta_path, args.outdir / "mmseqs_work", args.identity)
        clusters = parse_mmseqs_clusters(cluster_file)
    else:
        cluster_file = run_cdhit(fasta_path, args.outdir / "cdhit_work", args.identity)
        clusters = parse_cdhit_clusters(cluster_file)

    out_csv = args.outdir / "chain_cluster_assignments.csv"
    clusters.to_csv(out_csv, index=False)
    log.info("%d chains grouped into %d clusters (identity>=%.2f) -- wrote %s",
              len(clusters), clusters["cluster_rep"].nunique(), args.identity, out_csv)


if __name__ == "__main__":
    main()
