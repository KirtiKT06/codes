#!/usr/bin/env python
"""
FAIR relaxation test for the predicted-structure benchmark.

Question. Unrelaxed RhoFold+ models give electrostatic maps with a large systematic offset at the C1' and 4 A scales
(about -40 kT/e at C1'). Relaxing the PREDICTIONS removed the C1' offset but made the sphere maps worse. That test was not
fair: the native reference is a crystal structure that was never force-field relaxed, so relaxed-prediction vs
unrelaxed-native mixes "local strain removed" with "relaxed vs unrelaxed". This script relaxes BOTH with the same
protocol (RhoFold's own AmberRelaxation: AMBER14, explicit TIP3P-FB water, minimisation, no restraints) and compares
four things per chain:

    A  pred raw       vs native raw        (the original benchmark)
    B  pred relaxed   vs native raw        (the earlier, unfair relaxation test)
    C  pred relaxed   vs native relaxed    (THE FAIR TEST)
    D  native relaxed vs native raw        (what relaxation alone does to a perfect structure)
    E  pred raw       vs native relaxed

Reading. If D is already poor at C1'/4 A (low Pearson, big offset), relaxation itself perturbs the maps and B was
confounded. If C is high (Pearson ~0.9+, offsets ~0) at all four scales for chains with lDDT >= 0.9, then near-native
predictions are electrostatically equivalent to the native once local strain is regularised the same way, i.e. the
original gap was local-geometry strain, not misplaced residues. If C stays poor, the sub-angstrom placement differs
beyond strain.

THREE STAGES, THREE ENVIRONMENTS (file hand-off only)
    select   (venv)     choose chains with lDDT >= --min-lddt, write native.pdb and pred.pdb exactly as the benchmark
                        prepares them
    relax    (rhofold)  relax native.pdb and pred.pdb with RhoFold's AmberRelaxation (needs OpenMM, no pandas)
    score    (venv)     APBS maps for the four structures per chain, comparisons A to E
    report   (venv)     medians over chains

Use the driver run_fair_relax_test.sh, or run the stages by hand (see --help of each).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("fair_relax")
QTY = ["c1p", "m4", "m8", "m12"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).expanduser()))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------- stage 1
def cmd_select(a):
    import pandas as pd
    bench = load_module(a.bench_module, "psb")
    m = pd.read_csv(a.metrics)
    m = m[(m.status == "ok") & (m.lddt_c1p >= a.min_lddt)].sort_values("lddt_c1p", ascending=False)
    if a.max_chains:
        m = m.head(a.max_chains)
    log.info("%d chains with lDDT >= %.2f in %s", len(m), a.min_lddt, a.metrics)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    idx = bench.index_pred_dir(Path(a.pred_dir))
    ok, rows = 0, []
    for _, r in m.iterrows():
        pdb_id, chain = r.pdb_id, r.chain
        d = work / f"{pdb_id}_{chain}"
        try:
            src = Path(a.labels_dir) / "work" / pdb_id / f"{pdb_id.lower()}_complete_residues.pdb"
            iso = "".join(ln for ln in src.read_text().splitlines(keepends=True)
                          if ln.startswith(("ATOM", "HETATM")) and len(ln) > 21 and ln[21] == chain)
            d.mkdir(exist_ok=True)
            (d / "native_raw.pdb").write_text(iso)
            bases = bench.prepare_chain_pdb(d / "native_raw.pdb", d / "native.pdb")
            bench.prepare_chain_pdb(idx[(pdb_id.upper(), chain)], d / "pred.pdb", expect_bases=bases)
            rows.append({"pdb_id": pdb_id, "chain": chain, "lddt_c1p": float(r.lddt_c1p),
                         "rmsd_c1p": float(r.rmsd_c1p), "n_res": len(bases)})
            ok += 1
        except Exception as e:  # noqa: BLE001
            log.warning("%s_%s skipped: %s", pdb_id, chain, e)
    pd.DataFrame(rows).to_csv(work / "selected.csv", index=False)
    log.info("prepared %d chains in %s", ok, work)


# ---------------------------------------------------------------------------------------------- stage 2
def cmd_relax(a):
    # stdlib + rhofold only: must run in the rhofold environment, no pandas needed
    sys.path.insert(0, str(Path(a.rhofold_dir).expanduser()))
    from rhofold.relax.relax import AmberRelaxation  # noqa: E402
    relaxer = AmberRelaxation(max_iterations=a.steps, use_gpu=a.gpu, logger=log)
    work = Path(a.work)
    failed = []
    dirs = sorted(p for p in work.iterdir() if p.is_dir())
    for i, d in enumerate(dirs, 1):
        for kind in ("native", "pred"):
            src, dst = d / f"{kind}.pdb", d / f"{kind}_relaxed.pdb"
            if dst.exists() or not src.exists():
                continue
            log.info("[%d/%d] relaxing %s/%s", i, len(dirs), d.name, kind)
            try:
                relaxer.process(str(src), str(dst))
                if not dst.exists() or dst.stat().st_size == 0:
                    raise RuntimeError("empty output")
            except Exception as e:  # noqa: BLE001
                failed.append((d.name, kind, str(e)[:200].replace("\n", " ")))
                log.warning("   FAILED %s/%s: %s", d.name, kind, e)
    if failed:
        (work / "relax_failed.tsv").write_text("".join("\t".join(f) + "\n" for f in failed))
    log.info("relax done; %d failures", len(failed))


# ---------------------------------------------------------------------------------------------- stage 3
def score_chain(job: dict) -> dict:
    import numpy as np
    d = Path(job["dir"])
    out_json = d / "fair_metrics.json"
    if out_json.exists() and not job["overwrite"]:
        return json.loads(out_json.read_text())
    row = {"pdb_id": job["pdb_id"], "chain": job["chain"], "status": "ok", "error": ""}
    try:
        bench = job["_bench"] if "_bench" in job else load_module(job["bench_module"], "psb")
        pipe = job["_pipe"] if "_pipe" in job else load_module(job["pipeline_module"], "pipe_mod")
        files = {k: d / f"{k}.pdb" for k in ("native", "native_relaxed", "pred", "pred_relaxed")}
        miss = [k for k, f in files.items() if not f.exists()]
        if miss:
            raise FileNotFoundError("missing " + ",".join(miss))
        tabs = {}
        for k, f in files.items():
            # relaxed files come from RhoFold's relaxer: renumber/normalise exactly like the benchmark does
            if k.endswith("_relaxed"):
                bench.prepare_chain_pdb(f, d / f"{k}_prep.pdb")
                f = d / f"{k}_prep.pdb"
            tabs[k] = bench.potentials_for_pdb(f, d / f"w_{k}", "apbs", pipe, None, "dhS")
        pairs = {"A": ("native", "pred"), "B": ("native", "pred_relaxed"), "C": ("native_relaxed", "pred_relaxed"),
                 "D": ("native", "native_relaxed"), "E": ("native_relaxed", "pred")}
        for tag, (ref, oth) in pairs.items():
            row.update(bench.compare_potentials(tabs[ref], tabs[oth], tag))
        # geometry (C1' only)
        c1 = {}
        for k in files:
            f = d / f"{k}_prep.pdb" if k.endswith("_relaxed") else files[k]
            _, atoms = bench.read_pdb_atoms(f)
            c1[k] = bench.c1p_coords(atoms)
        for tag, (ref, oth) in {"geo_A": ("native", "pred"), "geo_C": ("native_relaxed", "pred_relaxed"),
                                "geo_D": ("native", "native_relaxed"), "geo_pred_move": ("pred", "pred_relaxed")}.items():
            n = min(len(c1[ref]), len(c1[oth]))
            row[f"{tag}_rmsd"] = float(bench.kabsch_rmsd(c1[oth][:n], c1[ref][:n]))
            row[f"{tag}_lddt"] = float(bench.lddt_c1p(c1[ref][:n], c1[oth][:n]))
        row["n_res"] = int(len(c1["native"]))
        row["solver_native"] = tabs["native"].attrs.get("solver", "")
    except Exception as e:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"{type(e).__name__}: {e}"[:300]
        log.warning("%s_%s failed: %s", job["pdb_id"], job["chain"], row["error"])
    out_json.write_text(json.dumps(row, default=float))
    return row


def cmd_score(a):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    work = Path(a.work)
    sel = pd.read_csv(work / "selected.csv")
    jobs = [dict(dir=str(work / f"{r.pdb_id}_{r.chain}"), pdb_id=r.pdb_id, chain=r.chain,
                 bench_module=str(Path(a.bench_module).expanduser()),
                 pipeline_module=str(Path(a.pipeline_module).expanduser()), overwrite=a.overwrite)
            for r in sel.itertuples()]
    log.info("%d chains to score with APBS (4 solves each)", len(jobs))
    rows = []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for fut in as_completed([ex.submit(score_chain, j) for j in jobs]):
                rows.append(fut.result())
    else:
        rows = [score_chain(j) for j in jobs]
    df = pd.DataFrame(rows).merge(sel[["pdb_id", "chain", "lddt_c1p", "rmsd_c1p"]], on=["pdb_id", "chain"], how="left")
    df.to_csv(work / "fair_chain_metrics.csv", index=False)
    log.info("scored: %d ok, %d failed -> %s", int((df.status == "ok").sum()), int((df.status != "ok").sum()),
             work / "fair_chain_metrics.csv")


# ---------------------------------------------------------------------------------------------- stage 4
LABEL = {"A": "A  pred raw vs native raw (original)", "B": "B  pred relaxed vs native raw (earlier test)",
         "C": "C  pred relaxed vs native relaxed (FAIR)", "D": "D  native relaxed vs native raw (relaxation alone)",
         "E": "E  pred raw vs native relaxed"}


def cmd_report(a):
    import numpy as np
    import pandas as pd
    d = pd.read_csv(Path(a.work) / "fair_chain_metrics.csv")
    d = d[d.status == "ok"]
    if d.empty:
        bad = pd.read_csv(Path(a.work) / "fair_chain_metrics.csv")
        raise SystemExit("no chain scored successfully. Errors:\n" + bad.error.value_counts().head(8).to_string())
    print(f"chains scored: {len(d)}   median lDDT of the prediction: {d.lddt_c1p.median():.3f}   "
          f"median RMSD: {d.rmsd_c1p.median():.2f} A\n")
    rows = []
    for tag in "ABCDE":
        for q in QTY:
            p, o = d.get(f"{tag}_pearson_{q}"), d.get(f"{tag}_mean_offset_{q}")
            if p is None:
                continue
            rows.append({"comparison": LABEL[tag], "quantity": q, "median_pearson": p.median(),
                         "q25": p.quantile(.25), "q75": p.quantile(.75), "median_offset_kTe": o.median(),
                         "median_abs_offset": o.abs().median()})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for q in QTY:
        print(t[t.quantity == q].drop(columns="quantity").round(3).to_string(index=False), "\n")
    geo = {c: d[c].median() for c in d.columns if c.startswith("geo_")}
    print("C1' geometry, medians:", {k: round(v, 3) for k, v in geo.items()})
    try:
        from scipy import stats
        for q in QTY:
            x, y = d[f"A_pearson_{q}"].dropna(), d[f"C_pearson_{q}"].dropna()
            n = min(len(x), len(y))
            if n >= 6:
                w = stats.wilcoxon(d[f"C_pearson_{q}"] - d[f"A_pearson_{q}"], nan_policy="omit")
                print(f"paired Wilcoxon, Pearson C minus A, {q}: median change "
                      f"{(d[f'C_pearson_{q}'] - d[f'A_pearson_{q}']).median():+.3f}, p = {w.pvalue:.3g}")
    except Exception as e:  # noqa: BLE001
        print("wilcoxon skipped:", e)
    # structure-level view: chains of one structure are not independent (e.g. six 25-nt chains of 2QUX), so average
    # the chains of each structure first and summarise across structures
    num = [c for c in d.columns if c.startswith(tuple("ABCDE")) and ("_pearson_" in c or "_mean_offset_" in c)]
    sd = d.groupby("pdb_id")[num].mean()
    print(f"\nSTRUCTURE-LEVEL (chains averaged within each of {len(sd)} structures; median across structures)")
    rows2 = []
    for tag in "ABCDE":
        r = {"comparison": LABEL[tag][:46]}
        for q in QTY:
            if f"{tag}_pearson_{q}" in sd:
                r[f"r_{q}"] = sd[f"{tag}_pearson_{q}"].median()
                r[f"off_{q}"] = sd[f"{tag}_mean_offset_{q}"].median()
        rows2.append(r)
    print(pd.DataFrame(rows2).round(3).to_string(index=False))
    try:
        from scipy import stats
        for q in QTY:
            diff = (sd[f"C_pearson_{q}"] - sd[f"A_pearson_{q}"]).dropna()
            if len(diff) >= 6:
                print(f"structure-level Wilcoxon, Pearson C minus A, {q}: median change {diff.median():+.3f}, "
                      f"p = {stats.wilcoxon(diff).pvalue:.3g}  (n = {len(diff)} structures)")
    except Exception as e:  # noqa: BLE001
        print("structure-level wilcoxon skipped:", e)
    t.to_csv(Path(a.work) / "fair_summary.csv", index=False)
    print("\nHow to read: D poor at C1'/4 A -> relaxation itself perturbs the maps (B was confounded). "
          "C high with offsets near 0 -> near-native predictions agree with the native once both are relaxed the same "
          "way (the gap was local strain). C still poor -> sub-angstrom placement differs beyond strain.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("select")
    s.add_argument("--metrics", required=True, help="bench_apbs_v2/chain_metrics.csv")
    s.add_argument("--labels-dir", required=True)
    s.add_argument("--pred-dir", required=True, help="preds/ with the UNRELAXED predictions")
    s.add_argument("--work", required=True)
    s.add_argument("--min-lddt", type=float, default=0.9)
    s.add_argument("--max-chains", type=int, default=0)
    s.add_argument("--bench-module", default="predicted_structure_benchmark.py")
    r = sp.add_parser("relax")
    r.add_argument("--work", required=True)
    r.add_argument("--rhofold-dir", default="/data/RhoFold")
    r.add_argument("--steps", type=int, default=1000)
    r.add_argument("--gpu", action="store_true")
    c = sp.add_parser("score")
    c.add_argument("--work", required=True)
    c.add_argument("--pipeline-module", required=True)
    c.add_argument("--bench-module", default="predicted_structure_benchmark.py")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--overwrite", action="store_true")
    p = sp.add_parser("report")
    p.add_argument("--work", required=True)
    a = ap.parse_args()
    {"select": cmd_select, "relax": cmd_relax, "score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
