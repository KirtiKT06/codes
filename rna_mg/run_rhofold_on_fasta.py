#!/usr/bin/env python
"""
Run RhoFold+ on every chain in a FASTA file and save the structures as preds/<PDBID>_<chain>.pdb,
the names predicted_structure_benchmark.py expects.

Run it with the RhoFold conda environment active (conda activate RhoFold).

    python run_rhofold_on_fasta.py --fasta test_chains.fasta \
        --rhofold-dir ~/RhoFold --ckpt ~/RhoFold/pretrained/RhoFold_pretrained.pt \
        --out-dir preds --device cuda:0 --single-seq --max-chains 3

Notes
  * Without the ~900 GB MSA databases RhoFold+ has to run in --single-seq mode, which its authors
    describe as still in testing and less accurate than the MSA mode. Say so in the paper.
  * Resumable: chains whose output PDB already exists are skipped.
  * One chain failing (e.g. out of GPU memory on a long sequence) does not stop the loop;
    failures are listed in <out-dir>/failed.tsv.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def read_fasta(path):
    name, seq, out = None, [], []
    for ln in Path(path).read_text().splitlines() + [">END"]:
        if ln.startswith(">"):
            if name is not None:
                out.append((name, "".join(seq)))
            name, seq = ln[1:].strip().split()[0], []
        else:
            seq.append(ln.strip())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--rhofold-dir", required=True, help="folder that contains inference.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="preds")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--single-seq", action="store_true", help="no MSA (needed unless the 900 GB databases are installed)")
    ap.add_argument("--relax-steps", type=int, default=0,
                    help="0 = use unrelaxed_model.pdb (fast, default); >0 = Amber relax and use the relaxed model")
    ap.add_argument("--max-chains", type=int, default=0)
    a = ap.parse_args()

    rf = Path(a.rhofold_dir).expanduser().resolve()
    infer = rf / "inference.py"
    if not infer.exists():
        sys.exit(f"ERROR: {infer} not found; check --rhofold-dir")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_fasta(a.fasta)
    if a.max_chains:
        records = records[:a.max_chains]
    failed = []
    for i, (name, seq) in enumerate(records, 1):
        dest = out_dir / f"{name}.pdb"
        if dest.exists():
            print(f"[{i}/{len(records)}] {name}: already done, skipping")
            continue
        print(f"[{i}/{len(records)}] {name} ({len(seq)} nt)", flush=True)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fa = td / f"{name}.fasta"
            fa.write_text(f">{name}\n{seq}\n")
            cmd = [sys.executable, str(infer), "--input_fas", str(fa), "--output_dir", str(td / "out"),
                   "--ckpt", str(Path(a.ckpt).expanduser().resolve()), "--device", a.device,
                   "--relax_steps", str(a.relax_steps)]
            if a.single_seq:
                cmd += ["--single_seq_pred", "True"]
            r = subprocess.run(cmd, cwd=str(rf), capture_output=True, text=True)
            produced = None
            if r.returncode == 0:
                cands = (sorted((td / "out").glob("relaxed_*_model.pdb")) if a.relax_steps > 0 else []) \
                    + [td / "out" / "unrelaxed_model.pdb"]
                produced = next((c for c in cands if c.exists()), None)
            if produced is None:
                reason = (r.stderr or r.stdout)[-300:].replace("\n", " ")
                print(f"    FAILED: {reason}")
                failed.append((name, reason))
                continue
            shutil.copy(produced, dest)
    if failed:
        (out_dir / "failed.tsv").write_text("".join(f"{n}\t{r}\n" for n, r in failed))
        print(f"{len(failed)} chains failed; see {out_dir / 'failed.tsv'}")
    print("done")


if __name__ == "__main__":
    main()
