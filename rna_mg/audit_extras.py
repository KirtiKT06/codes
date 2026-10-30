#!/usr/bin/env python
"""
Offline analyses on audit_chain_stats.csv.gz (written by the PATCHED centered_latent_manifold.py --audit).
Nothing is refitted; everything is recomputed from per-chain sums, with bootstrap over STRUCTURES.

1) length  -- does the pretrained-vs-random "shared variance" gap live in long chains?
       python audit_extras.py length \
           --runs ernierna/pretrained=$R/controls_matched/pretrained/ernierna \
                  ernierna/random_s0=$R/controls_matched/random_s0/ernierna \
                  ernierna/random_s1=... ernierna/random_s2=... \
           --out $R/controls_matched/analysis/length_partition.csv
   Label format is  <model>/<run>=<folder>. For each chain-length bin (test residues per chain) it reports
   unique_lm, shared, R^2(controls), R^2(LM) with 95% CIs, micro-weighted.

2) diff    -- PAIRED comparison on the identical test chains (pretrained minus random, same split):
       python audit_extras.py diff --tag ernierna \
           --a $R/controls_matched/pretrained/ernierna \
           --b $R/controls_matched/random_s0/ernierna $R/controls_matched/random_s1/ernierna \
               $R/controls_matched/random_s2/ernierna \
           --out $R/controls_matched/analysis/paired_differences.csv
   Reports a - mean(b) for unique_lm and shared with a bootstrap CI that resamples the same structures for every
   run (so the CI is for the DIFFERENCE, not for each side), micro and macro, plus the spread across the b runs.
   Appends to --out so several models can share one file (use --tag).

6) bygroup -- pretrained minus random within RNA classes or within bins of similarity to a pretraining corpus:
       python audit_extras.py bygroup --tag ernierna --a $O/pretrained/ernierna --b $O/random_s0/ernierna $O/random_s1/ernierna $O/random_s2/ernierna \\
           --groups $R/chain_classes.csv --group-col rna_class --out $O/analysis/bygroup.csv
       python audit_extras.py bygroup --tag ernierna_famfolds --a $O/family_t0.5_fold{0,1,2,3,4}/ernierna --b-mode pool \\
           --b $O/famrand_t0.5_fold{0,1,2,3,4}_s0/ernierna --groups $R/corpus_overlap_rnacentral.csv \\
           --group-col best_fident_excl_identical --bins 0 0.5 0.7 0.9 1.0001 --out $O/analysis/bygroup.csv

5) perchain -- which LONG chains carry the pretrained advantage (per-chain shared variance, pretrained vs random):
       python audit_extras.py perchain --tag ernierna --a $R/controls_matched/pretrained/ernierna \\
           --b $R/controls_matched/random_s0/ernierna $R/controls_matched/random_s1/ernierna $R/controls_matched/random_s2/ernierna \\
           --min-len 500 --targets potential_mean_kT_e_12A potential_mean_kT_e_4A --out $R/controls_matched/analysis/perchain_long.csv

4) spikepool -- the spike-in power analysis on the pooled family folds (run_spike_family.sh first):
       python audit_extras.py spikepool --tag ernierna --folds $R/controls_matched/famspike_t0.5_fold{0,1,2,3,4}/ernierna \\
           --out $R/controls_matched/analysis/spike_family_pooled.csv

3) famdiff -- leave-family-out, POOLED over the 5 folds (their test chains are disjoint), pretrained minus random-init
   evaluated on the identical held-out chains. Needs run_controls_matched.sh with RUN_FAMILY=1 RUN_FAMILY_RANDOM=1:
       python audit_extras.py famdiff --tag ernierna_t0.5 \
           --a $R/controls_matched/family_t0.5_fold{0,1,2,3,4}/ernierna \
           --b $R/controls_matched/famrand_t0.5_fold{0,1,2,3,4}_s0/ernierna \
           --out $R/controls_matched/analysis/family_paired_differences.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LM = "lm_gbm_pca"
DEFAULT_TARGETS = ["potential_at_c1prime_kT_e", "potential_mean_kT_e_4A", "potential_mean_kT_e_12A"]


def load(folder: str) -> pd.DataFrame:
    f = Path(folder) / "audit_chain_stats.csv.gz"
    if not f.exists():
        raise SystemExit(f"{f} not found -- run the audit with the patched centered_latent_manifold.py "
                         "(python patch_audit.py <file>) first")
    return pd.read_csv(f)


def wide(d: pd.DataFrame, target: str, ctrl: str) -> pd.DataFrame:
    """One row per chain: n, sst, and sse for LM-only (lm), controls (c), controls+LM (b)."""
    g = d[d.target == target]
    base = g[g.predictors == LM][["pdb", "chain", "n", "sst"]].set_index(["pdb", "chain"])
    for k, p in (("lm", LM), ("c", ctrl), ("b", ctrl + "+lm")):
        x = g[g.predictors == p].set_index(["pdb", "chain"]).sse.rename(f"sse_{k}")
        base = base.join(x, how="inner")
    return base.reset_index()


def struct_sums(w: pd.DataFrame, pdbs: list[str]) -> pd.DataFrame:
    w = w.copy()
    w["sst_m"] = w.sst / w.n
    for k in ("lm", "c", "b"):
        w[f"sse_m_{k}"] = w[f"sse_{k}"] / w.n
    cols = ["sst", "sst_m"] + [f"sse_{k}" for k in ("lm", "c", "b")] + [f"sse_m_{k}" for k in ("lm", "c", "b")]
    return w.groupby("pdb")[cols].sum().reindex(pdbs).fillna(0.0)


def parts(g: pd.DataFrame, W: np.ndarray, kind: str) -> dict:
    """W: (B, S) structure weights. Returns each quantity as a (B,) array."""
    sst = g["sst"] if kind == "micro" else g["sst_m"]
    pre = "sse_" if kind == "micro" else "sse_m_"
    den = W @ sst.to_numpy()
    r = {k: 1.0 - (W @ g[pre + k].to_numpy()) / den for k in ("lm", "c", "b")}
    return {"unique_lm": r["b"] - r["c"], "shared": r["c"] + r["lm"] - r["b"],
            "unique_controls": r["b"] - r["lm"], "unexplained": 1.0 - r["b"],
            "r2_controls": r["c"], "r2_lm": r["lm"]}


def boot_weights(S: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=(B, S))
    W = np.zeros((B, S))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=S)
    return W


def ci(x: np.ndarray):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


# ------------------------------------------------------------------------------------------------
def cmd_length(a):
    bins = [0] + list(a.bins) + [10 ** 6]
    names = [f"{lo + 1 if lo else 1}-{hi}" if hi < 10 ** 6 else f">{lo}" for lo, hi in zip(bins[:-1], bins[1:])]
    rows = []
    for item in a.runs:
        label, _, folder = item.partition("=")
        model, _, run = label.partition("/")
        d = load(folder)
        for t in a.targets:
            for ctrl in a.controls:
                try:
                    wd = wide(d, t, ctrl)
                except Exception as e:  # noqa: BLE001
                    print(f"skip {label} {t} {ctrl}: {e}")
                    continue
                wd["bin"] = pd.cut(wd.n, bins, labels=names)
                for bname in ["all"] + names:
                    sub = wd if bname == "all" else wd[wd.bin == bname]
                    pdbs = sorted(sub.pdb.unique())
                    if len(pdbs) < a.min_structures:
                        continue
                    g = struct_sums(sub, pdbs)
                    W = boot_weights(len(pdbs), a.n_boot, a.seed)
                    one = np.ones((1, len(pdbs)))
                    for kind in ("micro", "macro"):
                        p0, pb = parts(g, one, kind), parts(g, W, kind)
                        for comp in ("unique_lm", "shared", "r2_controls", "r2_lm"):
                            lo, hi = ci(pb[comp])
                            rows.append({"model": model, "run": run, "target": t, "controls": ctrl,
                                         "weighting": kind, "length_bin": bname, "n_chains": len(sub),
                                         "n_structures": len(pdbs), "n_residues": int(sub.n.sum()),
                                         "component": comp, "value": float(p0[comp][0]), "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    show = out[(out.weighting == "micro") & out.component.isin(["shared", "unique_lm"])]
    pv = show.pivot_table(index=["model", "target", "controls", "length_bin", "n_chains"], columns=["component", "run"],
                          values="value")
    print(pv.round(3).to_string())
    print(f"\nwrote {a.out}")


def cmd_diff(a):
    A = load(a.a)
    Bs = [load(p) for p in a.b]
    rows = []
    for t in a.targets:
        for ctrl in a.controls:
            wa = wide(A, t, ctrl)
            wbs = [wide(B, t, ctrl) for B in Bs]
            keys = set(zip(wa.pdb, wa.chain))
            for wb in wbs:
                keys &= set(zip(wb.pdb, wb.chain))
            if not keys:
                print(f"no common chains for {t} {ctrl}")
                continue
            keep = lambda w: w[[(p, c) in keys for p, c in zip(w.pdb, w.chain)]]  # noqa: E731
            wa, wbs = keep(wa), [keep(w) for w in wbs]
            pdbs = sorted(wa.pdb.unique())
            ga, gbs = struct_sums(wa, pdbs), [struct_sums(w, pdbs) for w in wbs]
            W = boot_weights(len(pdbs), a.n_boot, a.seed)
            one = np.ones((1, len(pdbs)))
            for kind in ("micro", "macro"):
                pa0, pab = parts(ga, one, kind), parts(ga, W, kind)
                pb0 = [parts(g, one, kind) for g in gbs]
                pbb = [parts(g, W, kind) for g in gbs]
                for comp in ("unique_lm", "shared"):
                    b_pts = np.array([p[comp][0] for p in pb0])
                    diff_boot = pab[comp] - np.mean([p[comp] for p in pbb], axis=0)
                    lo, hi = ci(diff_boot)
                    rows.append({"tag": a.tag, "target": t, "controls": ctrl, "weighting": kind, "component": comp,
                                 "a_value": float(pa0[comp][0]), "b_mean": float(b_pts.mean()),
                                 "b_min": float(b_pts.min()), "b_max": float(b_pts.max()), "n_b_runs": len(Bs),
                                 "diff_a_minus_b": float(pa0[comp][0] - b_pts.mean()), "lo": lo, "hi": hi,
                                 "n_structures": len(pdbs), "n_chains": len(keys)})
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        old = pd.read_csv(a.out)
        old = old[old.tag != a.tag]                 # re-running a model replaces its rows
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(a.out, index=False)
    show = out[out.tag == a.tag][["target", "controls", "weighting", "component", "a_value", "b_mean", "b_min",
                                  "b_max", "diff_a_minus_b", "lo", "hi"]]
    print(show.round(3).to_string(index=False))
    print(f"\nwrote {a.out}")


def load_many(folders) -> pd.DataFrame:
    """Concatenate chain stats of several folders (family folds have disjoint test chains)."""
    d = pd.concat([load(f) for f in folders], ignore_index=True)
    dup = d.duplicated(["target", "predictors", "pdb", "chain"]).sum()
    if dup:
        raise SystemExit(f"{dup} duplicated chains across the folders: folds must have disjoint test chains")
    return d


def cmd_famdiff(a):
    """Leave-family-out, pooled over folds: pretrained folds minus random folds on the identical held-out chains."""
    A, B = load_many(a.a), load_many(a.b)
    rows = []
    for t in a.targets:
        for ctrl in a.controls:
            wa, wb = wide(A, t, ctrl), wide(B, t, ctrl)
            keys = set(zip(wa.pdb, wa.chain)) & set(zip(wb.pdb, wb.chain))
            if not keys:
                print(f"no common chains for {t} {ctrl}")
                continue
            keep = lambda w: w[[(p, c) in keys for p, c in zip(w.pdb, w.chain)]]  # noqa: E731
            wa, wb = keep(wa), keep(wb)
            pdbs = sorted(wa.pdb.unique())
            ga, gb = struct_sums(wa, pdbs), struct_sums(wb, pdbs)
            W = boot_weights(len(pdbs), a.n_boot, a.seed)
            one = np.ones((1, len(pdbs)))
            for kind in ("micro", "macro"):
                pa0, pab, pb0, pbb = parts(ga, one, kind), parts(ga, W, kind), parts(gb, one, kind), parts(gb, W, kind)
                for comp in ("unique_lm", "shared"):
                    lo, hi = ci(pab[comp] - pbb[comp])
                    la, ha = ci(pab[comp]); lb, hb = ci(pbb[comp])
                    rows.append({"tag": a.tag, "target": t, "controls": ctrl, "weighting": kind, "component": comp,
                                 "pretrained_familyout": float(pa0[comp][0]), "pre_lo": la, "pre_hi": ha,
                                 "random_familyout": float(pb0[comp][0]), "rand_lo": lb, "rand_hi": hb,
                                 "diff": float(pa0[comp][0] - pb0[comp][0]), "lo": lo, "hi": hi,
                                 "n_structures": len(pdbs), "n_chains": len(keys), "n_residues": int(wa.n.sum())})
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        old = pd.read_csv(a.out)
        out = pd.concat([old[old.tag != a.tag], out], ignore_index=True)
    out.to_csv(a.out, index=False)
    show = out[out.tag == a.tag][["target", "controls", "weighting", "component", "pretrained_familyout",
                                  "random_familyout", "diff", "lo", "hi", "n_chains"]]
    print(show.round(3).to_string(index=False))
    print(f"\nwrote {a.out}")


def cmd_spikepool(a):
    """Power analysis on the pooled leave-family-out folds: for every spike sub-folder, pool the chain statistics of
    the five folds and report the recovered unique-to-LM variance with a structure-level bootstrap CI."""
    import re
    fold_dirs = [Path(x) for x in a.folds]
    names = sorted(p.name for p in fold_dirs[0].glob("spike_*") if p.is_dir())
    if not names:
        raise SystemExit(f"no spike_* sub-folders in {fold_dirs[0]}: run run_spike_family.sh first")
    rows = []
    for nm in names:
        m = re.fullmatch(r"spike_(linear|interaction)_(.+)_f([0-9.eE+-]+)", nm)
        if not m:
            continue
        stype, target, frac = m.group(1), m.group(2), float(m.group(3))
        folders = [d / nm for d in fold_dirs if (d / nm / "audit_chain_stats.csv.gz").exists()]
        if len(folders) < len(fold_dirs):
            print(f"warning: {nm}: only {len(folders)} of {len(fold_dirs)} folds present")
        if not folders:
            continue
        d = load_many(folders)
        for ctrl in a.controls:
            wd = wide(d, target, ctrl)
            pdbs = sorted(wd.pdb.unique())
            g = struct_sums(wd, pdbs)
            W = boot_weights(len(pdbs), a.n_boot, a.seed)
            one = np.ones((1, len(pdbs)))
            for kind in ("micro", "macro"):
                p0, pb = parts(g, one, kind), parts(g, W, kind)
                lo, hi = ci(pb["unique_lm"])
                rows.append({"tag": a.tag, "spike_type": stype, "target": target, "injected": frac,
                             "controls": ctrl, "weighting": kind, "recovered": float(p0["unique_lm"][0]),
                             "lo": lo, "hi": hi, "n_chains": len(wd), "n_folds": len(folders)})
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        old = pd.read_csv(a.out)
        out = pd.concat([old[old.tag != a.tag], out], ignore_index=True)
    out.to_csv(a.out, index=False)
    show = out[out.tag == a.tag].sort_values(["spike_type", "target", "controls", "weighting", "injected"])
    print(show[["spike_type", "target", "controls", "weighting", "injected", "recovered", "lo", "hi", "n_chains"]]
          .round(4).to_string(index=False))
    print(f"\nwrote {a.out}")


def cmd_perchain(a):
    """Per-chain shared / unique variance of the LONG chains, pretrained vs the mean of random seeds.
    Each chain is evaluated on its own (R2 = 1 - SSE/SST within that chain), so it shows WHICH long chains carry the
    pretrained advantage (e.g. ribosomal chains versus the group II intron 9IS7)."""
    A = load(a.a)
    Bs = [load(p) for p in a.b]
    parts_ = []
    for t in a.targets:
        for ctrl in a.controls:
            wa = wide(A, t, ctrl)
            wa = wa[wa.n >= a.min_len]
            if wa.empty:
                continue

            def r2(w):
                o = w[["pdb", "chain", "n"]].copy()
                for k in ("lm", "c", "b"):
                    o[f"r2_{k}"] = 1.0 - w[f"sse_{k}"] / w["sst"]
                o["shared"] = o.r2_c + o.r2_lm - o.r2_b
                o["unique_lm"] = o.r2_b - o.r2_c
                return o.set_index(["pdb", "chain"])
            m = r2(wa)
            rb = [r2(wide(B, t, ctrl)).reindex(m.index) for B in Bs]
            for k in ("shared", "r2_lm", "unique_lm"):
                m[f"{k}_random"] = np.mean([b[k].to_numpy() for b in rb], axis=0)
            m["advantage"] = m["shared"] - m["shared_random"]
            parts_.append(m.reset_index().assign(target=t, controls=ctrl, tag=a.tag))
    out = pd.concat(parts_, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        old = pd.read_csv(a.out)
        out = pd.concat([old[old.tag != a.tag], out], ignore_index=True)
    out.to_csv(a.out, index=False)
    show = out[out.tag == a.tag].sort_values(["target", "controls", "advantage"], ascending=[True, True, False])
    print(show[["target", "pdb", "chain", "n", "shared", "shared_random", "advantage", "r2_lm", "r2_lm_random",
                "unique_lm"]].round(3).to_string(index=False))
    print(f"\nwrote {a.out}")


def cmd_bygroup(a):
    """Pretrained minus random shared / unique variance WITHIN GROUPS of chains: RNA class (fetch_chain_classes.py) or
    bins of similarity to a pretraining corpus (corpus_overlap.py). Chains are paired across runs; the bootstrap
    resamples structures within each group. Several --b folders are either separate runs on the same chains (random
    seeds: --b-mode avg, default) or disjoint folds to pool (--b-mode pool, e.g. family folds)."""
    G = pd.read_csv(a.groups)
    G["_key"] = G["key"] if "key" in G.columns else G["pdb_id"].astype(str).str.upper() + "::" + G["chain"].astype(str)
    col = a.group_col
    if a.bins:
        edges = [float(x) for x in a.bins]
        G[col] = pd.cut(G[col].astype(float), edges, include_lowest=True).astype(str)
    gmap = dict(zip(G["_key"], G[col].astype(str)))
    A = load_many(a.a)
    Bs = [load_many(a.b)] if a.b_mode == "pool" else [load(p) for p in a.b]
    rows = []
    for t in a.targets:
        for ctrl in a.controls:
            wa = wide(A, t, ctrl)
            wbs = [wide(B, t, ctrl) for B in Bs]
            keys = set(wa.chain)
            for wb in wbs:
                keys &= set(wb.chain)
            wa = wa[wa.chain.isin(keys)].copy()
            wbs = [w[w.chain.isin(keys)].copy() for w in wbs]
            wa["group"] = wa.chain.map(gmap)
            for w in wbs:
                w["group"] = w.chain.map(gmap)
            groups = ["all"] + sorted(g for g in wa.group.dropna().unique())
            for g in groups:
                sa = wa if g == "all" else wa[wa.group == g]
                if sa.pdb.nunique() < a.min_structures or len(sa) < a.min_chains:
                    continue
                ids = set(sa.chain)
                pdbs = sorted(sa.pdb.unique())
                ga = struct_sums(sa, pdbs)
                gbs = [struct_sums(w[w.chain.isin(ids)], pdbs) for w in wbs]
                W = boot_weights(len(pdbs), a.n_boot, a.seed)
                one = np.ones((1, len(pdbs)))
                for kind in ("micro", "macro"):
                    pa0, pab = parts(ga, one, kind), parts(ga, W, kind)
                    pb0 = [parts(x, one, kind) for x in gbs]
                    pbb = [parts(x, W, kind) for x in gbs]
                    for comp in ("shared", "unique_lm"):
                        b0 = float(np.mean([p[comp][0] for p in pb0]))
                        db = pab[comp] - np.mean([p[comp] for p in pbb], axis=0)
                        lo, hi = ci(db)
                        rows.append({"tag": a.tag, "target": t, "controls": ctrl, "group_col": col, "group": g,
                                     "n_chains": len(sa), "n_structures": len(pdbs), "n_residues": int(sa.n.sum()),
                                     "weighting": kind, "component": comp, "pretrained": float(pa0[comp][0]),
                                     "random": b0, "diff": float(pa0[comp][0]) - b0, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        old = pd.read_csv(a.out)
        out = pd.concat([old[~((old.tag == a.tag) & (old.group_col == col))], out], ignore_index=True)
    out.to_csv(a.out, index=False)
    show = out[(out.tag == a.tag) & (out.group_col == col) & (out.component == "shared")]
    for t in a.targets:
        for w in ("micro", "macro"):
            d = show[(show.target == t) & (show.weighting == w) & (show.controls == a.controls[0])]
            if d.empty:
                continue
            print(f"\n{a.tag}  {t}  shared variance, pretrained minus random, {w}, grouped by {col}")
            print(d[["group", "n_chains", "n_structures", "pretrained", "random", "diff", "lo", "hi"]]
                  .round(3).to_string(index=False))
    print(f"\nwrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("length", "diff", "famdiff", "spikepool", "perchain", "bygroup"):
        p = sp.add_parser(name)
        p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
        p.add_argument("--controls", nargs="+", default=["controls_geom_coulomb"])
        p.add_argument("--n-boot", type=int, default=1000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", type=Path, required=True)
    p = sp.choices["length"]
    p.add_argument("--runs", nargs="+", required=True, help="<model>/<run>=<folder> ...")
    p.add_argument("--bins", type=int, nargs="+", default=[100, 250, 500],
                   help="upper edges of length bins (test residues per chain); last bin is open-ended")
    p.add_argument("--min-structures", type=int, default=8)
    p = sp.choices["diff"]
    p.add_argument("--a", required=True, help="folder of run A (e.g. pretrained)")
    p.add_argument("--b", nargs="+", required=True, help="folder(s) of run(s) B (e.g. random seeds); averaged")
    p.add_argument("--tag", default="run")
    p = sp.choices["famdiff"]
    p.add_argument("--a", nargs="+", required=True, help="pretrained family-fold folders (fold0..4)")
    p.add_argument("--b", nargs="+", required=True, help="random-init family-fold folders (same folds, same seed)")
    p.add_argument("--tag", default="familyout")
    p = sp.choices["spikepool"]
    p.add_argument("--folds", nargs="+", required=True, help="the five family-fold run folders of ONE model (famspike_*)")
    p.add_argument("--tag", default="spikepool")
    p = sp.choices["perchain"]
    p.add_argument("--a", required=True, help="pretrained run folder")
    p.add_argument("--b", nargs="+", required=True, help="random-init run folder(s), averaged")
    p.add_argument("--min-len", type=int, default=500, help="only chains with at least this many test residues")
    p.add_argument("--tag", default="perchain")
    p = sp.choices["bygroup"]
    p.add_argument("--a", nargs="+", required=True, help="pretrained run folder(s); several = pooled disjoint folds")
    p.add_argument("--b", nargs="+", required=True, help="random-init run folder(s)")
    p.add_argument("--b-mode", choices=["avg", "pool"], default="avg",
                   help="avg: --b are runs on the same chains (random seeds); pool: --b are disjoint folds")
    p.add_argument("--groups", type=Path, required=True, help="CSV with a 'key' column (PDB::chain) or pdb_id + chain")
    p.add_argument("--group-col", required=True, help="column of --groups to group by (e.g. rna_class, best_fident_excl_identical)")
    p.add_argument("--bins", nargs="*", default=None, help="numeric column: bin edges, e.g. 0 0.5 0.7 0.9 1.0001")
    p.add_argument("--min-structures", type=int, default=3)
    p.add_argument("--min-chains", type=int, default=5)
    p.add_argument("--tag", default="bygroup")
    a = ap.parse_args()
    {"length": cmd_length, "diff": cmd_diff, "famdiff": cmd_famdiff, "spikepool": cmd_spikepool,
     "perchain": cmd_perchain, "bygroup": cmd_bygroup}[a.cmd](a)


if __name__ == "__main__":
    main()
