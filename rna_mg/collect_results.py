#!/usr/bin/env python
"""
Collect everything the project has produced into ONE compact markdown file (+ merged tidy CSVs).

    python collect_results.py --root /data/rna_electrostats --code-dir ~/projects/codes/rna_mg \
                              --out project_snapshot.md

Upload project_snapshot.md. It is read-only: nothing is modified or recomputed except cheap summaries.
It finds files BY NAME (rglob), so it copes with folder layouts it has not seen; whatever it cannot
interpret it lists in an inventory, so nothing is silently lost. Each section is wrapped in try/except:
a failure is written into the report instead of stopping the run.

Sections
  0  inventory (data tree, code files, logs mentioning layers)
  1  dataset / labels
  2  raw probe results (all_probe_results.csv, final_test rows)
  3  per-variant ridge ceilings (the staircase) and control-explained R^2
  4  audit: predictor ladder + variance partition, pretrained vs random-init vs leave-family-out
  5  physics baseline (Coulomb report)
  6  cross-model agreement
  7  predicted-structure benchmark (every engine found): summaries, nulls, fidelity vs lDDT / length
  8  Mg2+ checks
  9  anything else small and tabular (generic dump) + results_bundle.txt
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["rnabert", "rnafm", "splicebert", "ernierna"]
OUT: list[str] = []
MAXLINES = 120


def md(title: str, body: str, level: int = 2):
    OUT.append(f"\n{'#' * level} {title}\n")
    OUT.append(body.rstrip() + "\n")


def code(text: str) -> str:
    return "```\n" + text.rstrip() + "\n```"


def tbl(df: pd.DataFrame, floatfmt="{:.3f}", max_rows=200) -> str:
    if df is None or len(df) == 0:
        return "(empty)"
    d = df.head(max_rows)
    s = d.to_string(index=False, float_format=lambda x: floatfmt.format(x))
    if len(df) > max_rows:
        s += f"\n... ({len(df) - max_rows} more rows)"
    return code(s)


def safe(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                md(f"{name} -- FAILED", code("".join(traceback.format_exception_only(type(e), e)) +
                                             traceback.format_exc(limit=3)))
        return wrap
    return deco


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def model_of(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part in MODELS:
            return part
    return None


def run_kind(path: Path) -> str:
    s = "/".join(path.parts).lower()
    pre = "matched_" if "controls_matched" in s else ""
    m = re.search(r"famrand_t([0-9.]+)", s)
    if m:
        return pre + f"famrandom_t{m.group(1)}"
    if "random" in s:
        return pre + "random"
    if "family" in s or "fold" in s:
        m = re.search(r"family_t([0-9.]+)", s)
        return pre + "family" + (f"_t{m.group(1)}" if m and pre else "")
    return pre + "pretrained"


def fold_of(path: Path) -> str:
    m = re.search(r"fold(\d+)", "/".join(path.parts))
    return m.group(1) if m else ""


def small_csv(path: Path, nrows=None) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------------------
@safe("0. Inventory")
def sec_inventory(root: Path, code_dir: Path):
    lines = []
    if root.exists():
        for d in sorted([p for p in root.iterdir() if p.is_dir()]):
            files = [f for f in d.rglob("*") if f.is_file()]
            size = sum(f.stat().st_size for f in files)
            subs = sorted({f.relative_to(d).parts[0] for f in files if len(f.relative_to(d).parts) > 1})[:12]
            lines.append(f"{d.name:<34} {len(files):>6} files {human(size):>8}   sub: {', '.join(subs)}")
        for f in sorted([p for p in root.iterdir() if p.is_file()]):
            lines.append(f"{f.name:<34} {human(f.stat().st_size):>8}")
    else:
        lines.append(f"ROOT NOT FOUND: {root}")
    md("0a. Data tree (top level of --root)", code("\n".join(lines)))

    cl = []
    if code_dir.exists():
        for f in sorted(code_dir.glob("*.py")):
            try:
                n = sum(1 for _ in open(f, errors="ignore"))
            except Exception:  # noqa: BLE001
                n = -1
            cl.append(f"{f.name:<46} {n:>5} lines")
        for f in sorted(code_dir.glob("*.md")) + sorted(code_dir.glob("*.txt")):
            cl.append(f"{f.name:<46} {human(f.stat().st_size):>8}")
    md("0b. Code directory", code("\n".join(cl) or "(none found)"))

    # layers / models mentioned in logs (the audit CSVs do not record the layer)
    logs = []
    pats = ["log*.txt", "*.log", "run_*.txt"]
    seen = set()
    for pat in pats:
        for f in list(root.rglob(pat))[:300]:
            if f in seen or not f.is_file() or f.stat().st_size > 80e6:
                continue
            seen.add(f)
            try:
                txt = open(f, errors="ignore").read()
            except Exception:  # noqa: BLE001
                continue
            layers = re.findall(r"Selected layer (\d+)", txt)
            models = sorted(set(re.findall(r"\b(rnabert|rnafm|splicebert|ernierna)\b", txt)))
            if layers:
                logs.append(f"{f}  models={models}  selected_layers={layers}")
    md("0c. Logs with 'Selected layer' (the layer is NOT stored in the audit CSVs)",
       code("\n".join(logs[:60]) or "none found -- layer per run unknown from files alone"))


@safe("1. Dataset / labels")
def sec_dataset(root: Path):
    lines = []
    p = root / "phi_res" / "all_residue_potentials.csv"
    if p.exists():
        head = pd.read_csv(p, nrows=2)
        cols = list(head.columns)
        use = [c for c in ("pdb_id", "chain", "pb_solver", "resname") if c in cols]
        d = pd.read_csv(p, usecols=use, low_memory=False)
        lines.append(f"labels file: {p}  columns: {cols}")
        lines.append(f"residues: {len(d):,}   structures: {d.pdb_id.nunique():,}" +
                     (f"   chains: {d.drop_duplicates(['pdb_id', 'chain']).shape[0]:,}" if "chain" in d else ""))
        if "pb_solver" in d:
            lines.append("solver per structure:\n" + d.drop_duplicates("pdb_id").pb_solver.value_counts().to_string())
        if "resname" in d:
            lines.append("residue types:\n" + d.resname.value_counts().head(8).to_string())
    else:
        lines.append(f"{p} not found")
    for name in ("structure_controls.csv.gz", "dssr_controls.csv.gz", "coulomb_baseline.csv.gz"):
        f = root / name
        lines.append(f"{name}: " + (f"{human(f.stat().st_size)}" if f.exists() else "missing"))
    md("1. Dataset / labels", code("\n".join(lines)))


@safe("2. Raw probes")
def sec_raw(root: Path):
    f = root / "probe_all_models_v2" / "all_probe_results.csv"
    if not f.exists():
        md("2. Raw probes", f"missing: {f}")
        return
    d = pd.read_csv(f)
    fin = d[d.probe == "final_test"] if "probe" in d else d
    pv = fin.pivot_table(index="label_column", columns="model", values="r2")
    if "baseline_r2" in fin:
        pv = pv.join(fin.groupby("label_column").baseline_r2.mean().rename("identity_baseline"))
    md("2. Raw probe, final held-out R^2 (best layer by val; rows probe == final_test)",
       tbl(pv.reset_index()))
    lay = fin.pivot_table(index="label_column", columns="model", values="layer")
    md("2b. Layer chosen per (target, model)", tbl(lay.reset_index(), "{:.0f}"))
    for p in sorted(root.glob("priority_results*/*"))[:8]:
        if p.is_dir():
            fs = sorted(x.name for x in p.iterdir())[:20]
            OUT.append(f"\n(priority results {p}: {', '.join(fs)})\n")


@safe("3. Staircase / controls")
def sec_staircase(root: Path):
    v3 = root / "centered_latent_v3"
    rows = []
    for f in v3.rglob("pls_dimensionality_sweep.csv"):
        m, var = model_of(f), f.parent.name
        d = pd.read_csv(f)
        for t, g in d.groupby("target"):
            k5 = g[g.n_components == 5].test_r2
            rows.append({"model": m, "variant": var, "target": t,
                         "ceiling_r2": g.single_target_ceiling_r2.iloc[0],
                         "pls_k5_r2": float(k5.iloc[0]) if len(k5) else np.nan})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(Path(OUT_DIR) / "tidy_staircase.csv", index=False)
        pv = df.pivot_table(index=["target", "model"], columns="variant", values="ceiling_r2")
        md("3a. Single-target ridge ceiling per variant (pretrained, shared layer)", tbl(pv.reset_index()))
    else:
        md("3a. Staircase", f"no pls_dimensionality_sweep.csv under {v3}")
    ce = []
    for f in root.rglob("control_explained_r2.csv"):
        try:
            d = pd.read_csv(f).assign(model=model_of(f), run=run_kind(f))
        except pd.errors.EmptyDataError:      # runs restricted to --variants centered_base_resid write an empty file
            continue
        ce.append(d)
    if ce:
        c = pd.concat(ce, ignore_index=True)
        c = c[c.run == "pretrained"]
        pv = c.pivot_table(index=["target", "model"], columns="variant", values="controls_only_test_r2")
        md("3b. Variance of centered labels explained by controls alone (control_explained_r2.csv; pretrained runs)",
           tbl(pv.reset_index()))
    nb = []
    for f in (root / "centered_latent_v3").rglob("nuisance_baseline_r2.csv"):
        nb.append(pd.read_csv(f).assign(model=model_of(f)))
    if nb:
        d = pd.concat(nb)
        d = d[d.model == "ernierna"] if d.model.nunique() > 1 else d
        md("3c. Nuisance-only baselines (linear variants; one model shown, they are identical across models)",
           tbl(d.drop(columns="model")))


def fmt_ci(v, lo, hi):
    if pd.isna(lo):
        return f"{v:+.3f}"
    return f"{v:+.3f} [{lo:+.3f},{hi:+.3f}]"


@safe("4. Audit")
def sec_audit(root: Path):
    files = sorted(root.rglob("audit_variance_partition.csv"))
    pf = sorted(root.rglob("audit_probe_r2.csv"))
    if not files:
        md("4. Audit", "no audit_variance_partition.csv found")
        return
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["run"], d["model"], d["fold"], d["path"] = run_kind(f), model_of(f), fold_of(f), str(f)
        frames.append(d)
    A = pd.concat(frames, ignore_index=True)
    A.to_csv(Path(OUT_DIR) / "tidy_partition.csv", index=False)
    inv = A.groupby(["run", "model"]).agg(files=("path", "nunique")).reset_index()
    md("4a. Which audit files were found", tbl(inv, "{:.0f}"))

    def cell(g):
        if len(g) == 1:
            r = g.iloc[0]
            return fmt_ci(r.value, r.lo, r.hi)
        return f"{g.value.mean():+.3f} [{g.value.min():+.3f}..{g.value.max():+.3f}] n={len(g)}"

    for ctrl in [c for c in ("controls_geom", "controls_geom_coulomb") if c in set(A.controls)]:
        for w in ("micro", "macro"):
            sub = A[(A.controls == ctrl) & (A.weighting == w)]
            rows = []
            runs = [r for r in ("pretrained", "random", "family") if r in set(A.run)] + \
                   sorted(r for r in set(A.run) if r.startswith("matched_"))
            for (t, m, comp), g in sub.groupby(["target", "model", "component"]):
                r = {"target": t, "model": m, "component": comp}
                for run in runs:
                    gg = g[g.run == run]
                    r[run] = cell(gg) if len(gg) else "-"
                rows.append(r)
            if rows:
                d = pd.DataFrame(rows)
                d["component"] = pd.Categorical(d.component, ["unique_lm", "shared", "unique_controls", "unexplained"])
                d = d.sort_values(["target", "model", "component"])
                md(f"4b. Variance partition: controls={ctrl}, {w}  "
                   f"(single file: value [95% bootstrap CI]; family folds: mean [min..max])", tbl(d), level=3)

    # predictor ladder, pretrained
    if pf:
        P = pd.concat([pd.read_csv(f).assign(model=model_of(f), run=run_kind(f)) for f in pf], ignore_index=True)
        P = P[P.run == "pretrained"]
        P.to_csv(Path(OUT_DIR) / "tidy_probe_ladder_pretrained.csv", index=False)
        for w in ("macro", "micro"):
            pv = P.pivot_table(index=["target", "model"], columns="predictors", values=f"{w}_r2")
            md(f"4c. Predictor ladder, pretrained, {w} R^2 (within-chain centered)", tbl(pv.reset_index()), level=3)
    # random-init marker
    for f in root.rglob("RANDOM_INIT.txt"):
        OUT.append(f"\n(random-init marker: {f}: {open(f).read().strip()})\n")
    fs = sorted(root.rglob("family_split_summary.csv"))
    if fs:
        d = small_csv(fs[0])
        if d is not None:
            md(f"4d. Family split summary (first of {len(fs)} files: {fs[0].parent})", tbl(d, max_rows=15), level=3)
    infos = []
    for f in sorted(root.rglob("family_split_info.json")):
        j = json.load(open(f))
        infos.append({"folder": str(f.parent.relative_to(root)), **{k: j[k] for k in (
            "fold", "n_families", "n_test_families", "n_structures_dropped", "n_structures_test", "residues_test",
            "residues_dropped", "family_leakage") if k in j}})
    if infos:
        md("4e. Family split info (all folds)", tbl(pd.DataFrame(infos), "{:.0f}", max_rows=60), level=3)
    # layer per run (new runs write LAYER.txt)
    lay = [f"{f.parent.relative_to(root)}: layer {open(f).read().strip()}" for f in sorted(root.rglob("LAYER.txt"))]
    if lay:
        md("4f. Layer used per matched run (LAYER.txt)", code("\n".join(lay[:80])), level=3)
    for name, title in (("paired_differences.csv", "4g. Paired pretrained-minus-random (audit_extras diff)"),
                        ("length_partition.csv", "4h. Chain-length stratification (audit_extras length)"),
                        ("spike_in_summary.csv", "4i. Spike-in power analysis (recovered unique_lm)")):
        for f in sorted(root.rglob(name))[:5]:
            d = small_csv(f)
            if d is None:
                continue
            if name == "length_partition.csv":
                d = d[d.component.isin(["shared", "unique_lm"]) & (d.weighting == "micro")]
            if name == "spike_in_summary.csv":
                d = d[d.controls == "controls_geom_coulomb"].drop(columns=[c for c in ("component", "controls") if c in d])
            md(f"{title}: {f.relative_to(root)}", tbl(d, max_rows=150), level=3)


@safe("5. Coulomb")
def sec_coulomb(root: Path):
    f = next(iter(root.rglob("coulomb_residual_report.csv")), None)
    if f is None:
        md("5. Physics baseline", "coulomb_residual_report.csv not found")
        return
    md("5. Physics baseline (coulomb_residual_report.csv)", tbl(pd.read_csv(f)))


@safe("6. Cross-model")
def sec_cross(root: Path):
    fs = sorted(root.rglob("cross_model_agreement*.csv"))
    if not fs:
        md("6. Cross-model agreement", "none found")
        return
    for f in fs:
        d = pd.read_csv(f)
        d["pair"] = d.model_a + "-" + d.model_b
        pv = d.pivot_table(index="pair", columns="condition", values="mean_canonical_corr")
        extra = ""
        if "controls" in d:
            extra = f"  controls={d.controls.iloc[0]} residualizer={d.get('residualizer', pd.Series(['']))[0]}"
        md(f"6. {f.relative_to(root)}{extra}", tbl(pv.reset_index()), level=3)


def fidelity_tables(d: pd.DataFrame, engine: str):
    out = []
    qs = [q for q in ("c1p", "m4", "m8", "m12") if f"pred_pearson_{q}" in d]
    bins = [0, .4, .5, .6, .7, .8, 1.01]
    d = d.assign(lddt_bin=pd.cut(d.lddt_c1p, bins))
    rows = []
    for q in qs:
        g = d.groupby("lddt_bin", observed=True)[f"pred_pearson_{q}"].agg(["median", "count"])
        for b, r in g.iterrows():
            rows.append({"quantity": q, "lddt_bin": str(b), "median_pearson": r["median"], "n": int(r["count"])})
    out.append("Median within-chain Pearson (pred vs native) by C1'-lDDT bin\n" + pd.DataFrame(rows)
               .pivot_table(index="lddt_bin", columns="quantity", values="median_pearson").to_string(float_format="%.3f"))
    cnt = d.groupby("lddt_bin", observed=True).size()
    out.append("chains per lDDT bin:\n" + cnt.to_string())
    if "n_res" in d:
        d = d.assign(len_bin=pd.cut(d.n_res, [0, 50, 100, 150, 260]))
        pv = d.groupby("len_bin", observed=True)[[f"pred_pearson_{q}" for q in qs] + ["lddt_c1p", "rmsd_c1p"]].median()
        pv["n"] = d.groupby("len_bin", observed=True).size()
        out.append("By chain length (median):\n" + pv.to_string(float_format="%.3f"))
        # OLS: fidelity ~ lddt + log rmsd + log n  (standardised) -- a regression alternative to a joint null
        X = np.column_stack([d.lddt_c1p, np.log(d.rmsd_c1p.clip(lower=0.1)), np.log(d.n_res)])
        ok = np.isfinite(X).all(1)
        Xs = (X[ok] - X[ok].mean(0)) / X[ok].std(0)
        rr = []
        for q in qs:
            y = d[f"pred_pearson_{q}"].to_numpy(float)[ok]
            m = np.isfinite(y)
            A = np.column_stack([np.ones(m.sum()), Xs[m]])
            beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            r2 = 1 - ((y[m] - A @ beta) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
            rr.append({"quantity": q, "b_lddt": beta[1], "b_log_rmsd": beta[2], "b_log_len": beta[3], "R2": r2})
        out.append("Standardised OLS, Pearson ~ lDDT + log RMSD + log length:\n" +
                   pd.DataFrame(rr).to_string(index=False, float_format="%.3f"))
    return "\n\n".join(out)


@safe("7. Benchmark")
def sec_bench(root: Path):
    mets = sorted(root.rglob("chain_metrics_with_excess.csv")) or sorted(root.rglob("chain_metrics.csv"))
    if not mets:
        md("7. Predicted-structure benchmark", "no chain_metrics*.csv found")
        return
    for f in mets:
        d = pd.read_csv(f)
        if "status" in d:
            nbad = int((d.status != "ok").sum())
            d = d[d.status == "ok"]
        else:
            nbad = 0
        title = f"7. Benchmark: {f.parent.name}"
        head = [f"file: {f}", f"chains ok: {len(d)} (not ok: {nbad})"]
        if "solver" in d:
            head.append("solver: " + d.solver.value_counts().to_dict().__repr__())
        if {"rmsd_c1p", "lddt_c1p"} <= set(d.columns):
            head.append(f"median C1' RMSD {d.rmsd_c1p.median():.2f}  median lDDT {d.lddt_c1p.median():.3f}")
        md(title, code("\n".join(head)))
        sm = f.parent / "summary.txt"
        if sm.exists():
            md(f"{f.parent.name}: summary.txt", code("\n".join(open(sm).read().splitlines()[:MAXLINES])), level=3)
        ex = f.parent / "excess_vs_null.csv"
        if ex.exists():
            md(f"{f.parent.name}: excess_vs_null.csv", tbl(pd.read_csv(ex)), level=3)
        if "lddt_c1p" in d:
            md(f"{f.parent.name}: fidelity vs lDDT / length", code(fidelity_tables(d, f.parent.name)), level=3)
        if "n_res" in d and "pred_pearson_m8" in d:
            pass


@safe("8. Mg2+")
def sec_mg(root: Path):
    ds = [p for p in sorted(root.glob("mg*")) if p.is_dir()]
    if not ds:
        md("8. Mg2+", "no mg* directories found")
        return
    for d in ds:
        fs = sorted(x for x in d.rglob("*") if x.is_file())
        md(f"8. {d.name}", code("\n".join(f"{x.relative_to(d)}  {human(x.stat().st_size)}" for x in fs[:40])))
        for x in fs:
            if x.suffix == ".txt" and x.stat().st_size < 40000 and "summary" in x.name.lower():
                md(f"{d.name}/{x.name}", code("\n".join(open(x, errors='ignore').read().splitlines()[:MAXLINES])), level=3)
        for x in fs:
            if x.suffix == ".csv" and x.stat().st_size < 3e6 and any(k in x.name.lower() for k in ("summary", "bin", "dist")):
                dd = small_csv(x)
                if dd is not None:
                    md(f"{d.name}/{x.name}  shape={dd.shape}", tbl(dd, max_rows=40), level=3)


@safe("9. Other")
def sec_other(root: Path, code_dir: Path):
    for c in (code_dir / "results_bundle.txt", root / "results_bundle.txt"):
        if c.exists():
            md(f"9a. {c.name} (first {MAXLINES} lines)", code("\n".join(open(c, errors="ignore").read().splitlines()[:MAXLINES])))
            break
    # generic dump of small CSVs in directories the sections above do not interpret
    skip = ("audit_", "pls_dim", "cca_", "latent_axis", "negative_controls", "nuisance", "control_explained",
            "cross_model", "chain_metrics", "excess_vs_null", "coulomb_residual", "all_probe", "family_split")
    shown = 0
    lines = []
    for f in sorted(root.rglob("*.csv")):
        if any(f.name.startswith(s) for s in skip) or f.stat().st_size > 2e6 or "per_residue" in f.parts:
            continue
        d = small_csv(f, nrows=3)
        if d is None:
            continue
        lines.append(f"{f.relative_to(root)}  cols={list(d.columns)[:10]}")
        shown += 1
        if shown >= 40:
            break
    md("9b. Other small CSVs (names and columns only)", code("\n".join(lines) or "none"))


def main():
    global OUT_DIR, MAXLINES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/data/rna_electrostats"))
    ap.add_argument("--code-dir", type=Path, default=Path.home() / "projects/codes/rna_mg")
    ap.add_argument("--out", type=Path, default=Path("project_snapshot.md"))
    ap.add_argument("--max-lines", type=int, default=120)
    a = ap.parse_args()
    MAXLINES = a.max_lines
    OUT_DIR = str(a.out.parent / "snapshot_tables")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    OUT.append(f"# Project snapshot\n\nroot: `{a.root}`   code: `{a.code_dir}`\n")
    for fn, args in ((sec_inventory, (a.root, a.code_dir)), (sec_dataset, (a.root,)), (sec_raw, (a.root,)),
                     (sec_staircase, (a.root,)), (sec_audit, (a.root,)), (sec_coulomb, (a.root,)),
                     (sec_cross, (a.root,)), (sec_bench, (a.root,)), (sec_mg, (a.root,)),
                     (sec_other, (a.root, a.code_dir))):
        print("running", fn.__name__)
        fn(*args)
    text = "\n".join(OUT)
    a.out.write_text(text)
    print(f"wrote {a.out}  ({human(len(text.encode()))});  tidy tables in {OUT_DIR}/")
    if len(text) > 300_000:
        print("NOTE: report is large; consider --max-lines 40")


if __name__ == "__main__":
    sys.exit(main())
