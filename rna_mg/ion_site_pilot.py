#!/usr/bin/env python
"""
STEP 0 PILOT: do residues with a strongly negative electrostatic potential flag the crystal Mg2+ / K+ sites?

It uses ONLY data you already have (no APBS): per_residue_potential.csv, <id>.pqr and ions.csv in work/<ID>/.

DEFINITIONS
  positive residue : a residue whose P atom is within --cutoff (5 A, the paper's r_c) of a resolved Mg2+ (or K+) ion.
  score            : higher = more likely a site. The PB labels are NEGATED (more negative = higher score).
  scores compared  : PB labels      pb_c1p, pb_m4, pb_m8, pb_m12 (C1' point, 4/8/12 A sphere means at 150 mM), pb_ppoint
                     baselines      phos_n8 / phos_n12 (number of other phosphates within 8 / 12 A),
                                    phos_screened (sum over other phosphates of exp(-r/9.6 A) / r, a Debye-Hueckel
                                    phosphate-only potential), and a random control
  metric           : WITHIN-CHAIN AUROC (is a positive residue ranked above a negative one of the same chain?),
                     R-precision enrichment (fraction of positives among the top-m residues, m = number of positives,
                     divided by the base rate m / n). Chains are averaged within each structure first and structures are
                     bootstrapped, because chains of one structure are not independent.

THE GATE (decide before building anything bigger)
  Does a PB label beat the best phosphate-only baseline, with a bootstrap CI that excludes zero, for Mg and for K?
  If not, rethink the target before computing P-centred shells for the whole dataset.

USAGE
    python ion_site_pilot.py --labels-dir /data/rna_electrostats/phi_res --out-dir /data/rna_electrostats/ion_pilot
    python ion_site_pilot.py ... --max-structures 100         # quick trial
    python ion_site_pilot.py ... --methods "X-RAY DIFFRACTION" # restrict to crystal structures (see the counts first)

OUTPUT (out-dir)
    ion_labels_5A.csv.gz     one row per phosphate-bearing residue with ion distances, positives, baseline features and
                             the PB labels (this is also the label table for the later embedding probes)
    pilot_chain_metrics.csv  per chain x ion x score: AUROC, R-precision, enrichment
    pilot_summary.txt        counts, resolution audit, headline table, paired differences, subgroup tables
    fig_ion_pilot.png/.pdf
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ion_pilot")

PB_COLS = {"pb_c1p": "potential_at_c1prime_kT_e", "pb_m4": "potential_mean_kT_e_4A", "pb_m8": "potential_mean_kT_e_8A",
           "pb_m12": "potential_mean_kT_e_12A", "pb_ppoint": "potential_at_phosphorus_kT_e"}
BASELINES = ["phos_n8", "phos_n12", "phos_screened"]                # centred on the residue's own P atom
C1_BASELINES = ["c1_n4", "c1_n8", "c1_n12", "c1_screened"]           # SAME centre as the PB labels (the C1' atom)
ALL_BASELINES = BASELINES + C1_BASELINES
MATCHED = [("pb_m4", "c1_n4"), ("pb_m8", "c1_n8"), ("pb_m12", "c1_n12")]   # same centre AND same sphere radius
SCORES = list(PB_COLS) + BASELINES + C1_BASELINES + ["random"]
DEBYE_A = 9.6                                       # screening length at ~0.1 M monovalent


def load_module(path: str, name: str = "pipe_mod"):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).expanduser()))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------
# per structure
# ----------------------------------------------------------------------------------------------------------------
def read_atoms(pqr: Path, pipeline_module: str | None) -> pd.DataFrame:
    """P and C1' atoms (atom, chain, resnum, resname, x, y, z). Fast fixed-width path; falls back to the pipeline's parser."""
    want = {"P", "C1'", "C1*"}
    rows, bad = [], 0
    with open(pqr) as f:
        for ln in f:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            if len(ln.rstrip("\n")) >= 69 and ln[11] == " " and ln[16] == " " and ln[20] == " ":
                nm = ln[12:16].strip()
                if nm not in want:
                    continue
                try:
                    rows.append((nm, ln[21], int(ln[22:26]), ln[17:20].strip(), float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
                except ValueError:
                    bad += 1
            else:
                bad += 1
    if bad and pipeline_module:
        pipe = load_module(pipeline_module)
        at = pipe.parse_pqr_residues(pqr)
        at = at[at["atom_name"].isin(want)].rename(columns={"atom_name": "atom"})
        return at[["atom", "chain", "resnum", "resname", "x", "y", "z"]].reset_index(drop=True)
    return pd.DataFrame(rows, columns=["atom", "chain", "resnum", "resname", "x", "y", "z"])


def process_structure(job: dict) -> dict:
    pid = job["pdb_id"]
    work = Path(job["labels_dir"]) / "work" / pid
    out = {"pdb_id": pid, "status": "ok", "error": "", "res": None, "ions": {}}
    try:
        lab = pd.read_csv(work / "per_residue_potential.csv", dtype={"pdb_id": str, "chain": str})
        A = read_atoms(work / f"{pid.lower()}.pqr", job["pipeline_module"])
        A["chain"] = A["chain"].astype(str)
        P = A[A["atom"] == "P"].drop(columns="atom").reset_index(drop=True)
        if P.empty:
            raise RuntimeError("no P atoms in the PQR")
        C1 = (A[A["atom"].isin(["C1'", "C1*"])].drop_duplicates(["chain", "resnum"])
              .rename(columns={"x": "cx", "y": "cy", "z": "cz"})[["chain", "resnum", "cx", "cy", "cz"]])
        meta = {"resolution_A": np.nan, "experimental_method": ""}
        mf = work / "structure_metadata.csv"
        if mf.exists():
            m = pd.read_csv(mf)
            if len(m):
                meta["resolution_A"] = float(m["resolution_A"].iloc[0]) if "resolution_A" in m and pd.notna(m["resolution_A"].iloc[0]) else np.nan
                meta["experimental_method"] = str(m["experimental_method"].iloc[0]) if "experimental_method" in m else ""
        if np.isnan(meta["resolution_A"]) and "resolution_A" in lab:
            meta["resolution_A"] = float(lab["resolution_A"].iloc[0]) if pd.notna(lab["resolution_A"].iloc[0]) else np.nan

        # ions
        ions_all = pd.DataFrame()
        icsv = work / "ions.csv"
        if icsv.exists() and icsv.stat().st_size > 5:
            try:
                ions_all = pd.read_csv(icsv)
            except pd.errors.EmptyDataError:
                ions_all = pd.DataFrame()
        info = {"n_ions_total": int(len(ions_all)), "n_removed_lowres": 0, "n_removed_occ": 0}
        keep = pd.DataFrame()
        if len(ions_all) and "resname" in ions_all:
            ions = ions_all[ions_all["resname"].astype(str).str.upper().isin(job["ions"])].copy()
            ions["resname"] = ions["resname"].astype(str).str.upper()
            if "occupancy" in ions:
                bad_occ = ions["occupancy"] < job["min_occ"]
                info["n_removed_occ"] = int(bad_occ.sum())
                ions = ions[~bad_occ]
            if "low_resolution_flag" in ions and not job["keep_low_res"]:
                low = ions["low_resolution_flag"].astype(str).str.lower().isin(["true", "1"])
                info["n_removed_lowres"] = int(low.sum())
                ions = ions[~low]
            kept, kept_xyz = [], []
            for i, r in ions.iterrows():                          # merge altloc duplicates
                p = np.array([r.x, r.y, r.z], float)
                if any(np.linalg.norm(p - q) < 0.5 and r.resname == kr for q, kr in kept_xyz):
                    continue
                kept_xyz.append((p, r.resname))
                kept.append(i)
            keep = ions.loc[kept]
        for el in job["ions"]:
            info[f"n_{el}"] = int((keep["resname"] == el).sum()) if len(keep) else 0
        out["ions"] = {**info, **meta}

        # features from the phosphate cloud
        xyz = P[["x", "y", "z"]].to_numpy(float)
        tree = cKDTree(xyz)
        n8 = np.array([len(x) - 1 for x in tree.query_ball_point(xyz, 8.0)])
        n12 = np.array([len(x) - 1 for x in tree.query_ball_point(xyz, 12.0)])
        scr = np.zeros(len(xyz))
        for i, nb in enumerate(tree.query_ball_point(xyz, 25.0)):
            nb = [j for j in nb if j != i]
            if nb:
                d = np.linalg.norm(xyz[nb] - xyz[i], axis=1)
                scr[i] = np.sum(np.exp(-d / DEBYE_A) / np.maximum(d, 1.0))      # clamp: duplicate atoms (d = 0) would give inf
        res = P.assign(phos_n8=n8, phos_n12=n12, phos_screened=scr)
        res = res.merge(C1, on=["chain", "resnum"], how="left")
        cxyz = res[["cx", "cy", "cz"]].to_numpy(float)
        have = np.isfinite(cxyz).all(axis=1)
        for r_, nm in ((4.0, "c1_n4"), (8.0, "c1_n8"), (12.0, "c1_n12")):
            col = np.full(len(res), np.nan)
            if have.any():
                col[have] = [len(x) for x in tree.query_ball_point(cxyz[have], r_)]
            res[nm] = col
        c1s = np.full(len(res), np.nan)
        for i in np.where(have)[0]:
            nb = tree.query_ball_point(cxyz[i], 25.0)
            if nb:
                d = np.linalg.norm(xyz[nb] - cxyz[i], axis=1)
                c1s[i] = np.sum(np.exp(-d / DEBYE_A) / np.maximum(d, 1.0))
            else:
                c1s[i] = 0.0
        res["c1_screened"] = c1s
        res = res.drop(columns=["cx", "cy", "cz"])
        for el in job["ions"]:
            sub = keep[keep["resname"] == el] if len(keep) else keep
            if len(sub):
                d, _ = cKDTree(sub[["x", "y", "z"]].to_numpy(float)).query(xyz, k=1)
            else:
                d = np.full(len(xyz), np.inf)
            res[f"d_{el}"] = d
            res[f"pos_{el}"] = d <= job["cutoff"]
        res["pos_any"] = res[[f"pos_{el}" for el in job["ions"]]].any(axis=1)
        res["d_any"] = res[[f"d_{el}" for el in job["ions"]]].min(axis=1)
        keep_cols = ["chain", "resnum"] + [c for c in PB_COLS.values() if c in lab] + (["original_resnum"] if "original_resnum" in lab else [])
        res = res.merge(lab[keep_cols], on=["chain", "resnum"], how="left")
        res = res.rename(columns={v: k for k, v in PB_COLS.items()})
        res.insert(0, "pdb_id", pid)
        res["resolution_A"] = meta["resolution_A"]
        res["method"] = meta["experimental_method"]
        out["res"] = res
    except Exception as exc:  # noqa: BLE001
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


# ----------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------
def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(score, method="average")
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def r_precision(score: np.ndarray, pos: np.ndarray, rng) -> float:
    m = int(pos.sum())
    order = np.argsort(-(score + 1e-9 * rng.standard_normal(len(score))), kind="stable")
    return float(pos[order[:m]].mean())


def chain_metrics(R: pd.DataFrame, ions: list[str], min_p: int, min_neg: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    targets = [f"pos_{e}" for e in ions] + ["pos_any"]
    for (pid, ch), g in R.groupby(["pdb_id", "chain"], sort=False):
        if len(g) < min_p:
            continue
        sc = {k: -g[k].to_numpy(float) for k in PB_COLS if k in g}                # more negative = more likely a site
        sc.update({k: g[k].to_numpy(float) for k in ALL_BASELINES if k in g})
        sc["random"] = rng.standard_normal(len(g))
        for t in targets:
            pos = g[t].to_numpy(bool)
            if pos.sum() < 1 or (~pos).sum() < min_neg:
                continue
            for name, s in sc.items():
                ok = np.isfinite(s)
                if ok.sum() < min_p or pos[ok].sum() < 1 or (~pos[ok]).sum() < min_neg:
                    continue
                rp = r_precision(s[ok], pos[ok], rng)
                base = pos[ok].mean()
                rows.append({"pdb_id": pid, "chain": ch, "ion": t[4:], "score": name, "n_res": int(ok.sum()),
                             "n_pos": int(pos[ok].sum()), "auroc": auroc(s[ok], pos[ok]), "rprec": rp,
                             "enrichment": rp / base if base > 0 else np.nan,
                             "length_bin": "<=100" if len(g) <= 100 else ("101-300" if len(g) <= 300 else ">300"),
                             "method": g["method"].iloc[0], "resolution_A": g["resolution_A"].iloc[0]})
    return pd.DataFrame(rows)


def boot_ci(per_struct: pd.Series, n_boot: int, seed: int):
    v = per_struct.dropna().to_numpy(float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(v)


def summarize(M: pd.DataFrame, ions: list[str], n_boot: int, seed: int, baseline: str = "phos_screened") -> str:
    lines, say = [], None
    say = lines.append
    for ion in ions + ["any"]:
        sub = M[M["ion"] == ion]
        if sub.empty:
            say(f"\n=== ion = {ion}: no chains with both positives and negatives ===")
            continue
        say(f"\n=== POSITIVES = residues whose P is within the cutoff of a {ion if ion != 'any' else 'Mg or K'} ion ===")
        say(f"chains: {sub[['pdb_id','chain']].drop_duplicates().shape[0]}, structures: {sub['pdb_id'].nunique()}")
        say(f"{'score':<14s}{'AUROC (structure mean) [95% CI]':<36s}{'enrichment (R-precision / base rate)':<40s}")
        piv_a = sub.pivot_table(index=["pdb_id", "chain"], columns="score", values="auroc")
        for sname in SCORES:
            if sname not in sub["score"].values:
                continue
            a = sub[sub["score"] == sname].groupby("pdb_id")["auroc"].mean()
            e = sub[sub["score"] == sname].groupby("pdb_id")["enrichment"].mean()
            am, al, ah, n = boot_ci(a, n_boot, seed)
            em, el, eh, _ = boot_ci(e, n_boot, seed)
            say(f"{sname:<14s}{am:.3f} [{al:.3f}, {ah:.3f}] (n={n:<4d})      {em:5.2f} [{el:.2f}, {eh:.2f}]")
        say(f"\n  paired differences in AUROC, structure mean, 95% bootstrap CI (positive = first score better):")
        say("  -- matched centre AND radius (the fair like-for-like comparison):")
        pairs = [m for m in MATCHED if m[0] in piv_a.columns and m[1] in piv_a.columns]
        pairs += [(pb, "c1_screened") for pb in PB_COLS if pb in piv_a.columns and "c1_screened" in piv_a.columns]
        for sname, b in pairs:
            d = (piv_a[sname] - piv_a[b]).dropna().groupby(level=0).mean()
            dm, dl, dh, n = boot_ci(d, n_boot, seed)
            verdict = "PB better" if dl > 0 else ("baseline better" if dh < 0 else "no clear difference")
            say(f"    {sname:<10s} - {b:<14s}{dm:+.3f} [{dl:+.3f}, {dh:+.3f}]  {verdict}")
        say("     (c1_n4 is degenerate, almost no phosphate lies within 4 A of a C1' atom, and counts at 8 A are coarse: judge the PB spheres")
        say("      against c1_screened, the continuous phosphate-only potential at the same centre)")
        say("  -- against the phosphorus-centred baselines:")
        for sname in [s_ for s_ in PB_COLS if s_ in sub["score"].values]:
            for b in BASELINES:
                if b not in piv_a.columns or sname not in piv_a.columns:
                    continue
                d = (piv_a[sname] - piv_a[b]).dropna().groupby(level=0).mean()
                dm, dl, dh, n = boot_ci(d, n_boot, seed)
                verdict = "PB better" if dl > 0 else ("baseline better" if dh < 0 else "no clear difference")
                say(f"    {sname:<10s} - {b:<14s}{dm:+.3f} [{dl:+.3f}, {dh:+.3f}]  {verdict}")
        say(f"\n  subgroups, AUROC of the best PB score vs {baseline} (structure means):")
        pbs = [s for s in PB_COLS if s in sub["score"].values]
        if pbs:
            best = max(pbs, key=lambda s: sub[sub["score"] == s].groupby("pdb_id")["auroc"].mean().mean())
            say(f"    best PB score overall: {best}")
            for col in ("length_bin", "method"):
                for lev, gg in sub.groupby(col):
                    if gg["pdb_id"].nunique() < 3 or not str(lev):
                        continue
                    a = gg[gg["score"] == best].groupby("pdb_id")["auroc"].mean()
                    b = gg[gg["score"] == baseline].groupby("pdb_id")["auroc"].mean()
                    say(f"      {col} = {lev:<22s} n structures {gg['pdb_id'].nunique():4d}   {best} {a.mean():.3f}   {baseline} {b.mean():.3f}")
    say("\n=== GATE SUMMARY (best PB label vs best phosphate-only baseline, paired over structures) ===")
    for ion in ions + ["any"]:
        sub = M[M["ion"] == ion]
        if sub.empty:
            continue
        pv = sub.pivot_table(index=["pdb_id", "chain"], columns="score", values="auroc")
        pbs = [c for c in PB_COLS if c in pv.columns]
        bls = [c for c in ALL_BASELINES if c in pv.columns]
        if not pbs or not bls:
            continue
        best_pb = max(pbs, key=lambda c: pv[c].groupby(level=0).mean().mean())
        best_bl = max(bls, key=lambda c: pv[c].groupby(level=0).mean().mean())
        d = (pv[best_pb] - pv[best_bl]).dropna().groupby(level=0).mean()
        dm, dl, dh, n = boot_ci(d, n_boot, seed)
        verdict = ("PASS: the PB label beats the baseline (CI above 0)" if dl > 0 else
                   ("FAIL: the baseline is better (CI below 0)" if dh < 0 else
                    "NOT SHOWN: no clear advantage over the baseline (CI includes 0)"))
        say(f"  {ion:<4s} best PB {best_pb} vs best baseline {best_bl}: AUROC difference {dm:+.3f} [{dl:+.3f}, {dh:+.3f}] over {n} structures -> {verdict}")
    return "\n".join(lines)


def make_figure(M: pd.DataFrame, ions: list[str], out: Path, n_boot: int, seed: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    show = [s for s in SCORES if s in M["score"].values]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for k, ion in enumerate(ions):
        sub = M[M["ion"] == ion]
        for i, s in enumerate(show):
            a = sub[sub["score"] == s].groupby("pdb_id")["auroc"].mean()
            m, lo, hi, _ = boot_ci(a, n_boot, seed)
            if np.isfinite(m):
                ax[0].errorbar(i + 0.18 * (k - 0.5), m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4, color=f"C{k}", capsize=2,
                               label=ion if i == 0 else None)
    ax[0].axhline(0.5, color="k", lw=0.5, ls="--")
    ax[0].set_xticks(range(len(show)))
    ax[0].set_xticklabels(show, rotation=45, ha="right")
    ax[0].set_ylabel("within-chain AUROC (structure mean, 95% CI)")
    ax[0].set_title("A  which score flags ion sites?")
    ax[0].legend(frameon=False)
    pbs = [s for s in PB_COLS if s in M["score"].values]
    for k, ion in enumerate(ions):
        sub = M[M["ion"] == ion]
        if not pbs:
            continue
        best = max(pbs, key=lambda s: sub[sub["score"] == s].groupby("pdb_id")["auroc"].mean().mean())
        pv = sub.pivot_table(index=["pdb_id", "chain"], columns="score", values="auroc")
        if best in pv and "phos_screened" in pv:
            ax[1].scatter(pv["phos_screened"], pv[best], s=6, alpha=0.5, color=f"C{k}", label=f"{ion}: {best}")
    ax[1].plot([0, 1], [0, 1], color="k", lw=0.5)
    ax[1].set_xlabel("AUROC, phosphate-only screened Coulomb")
    ax[1].set_ylabel("AUROC, best PB label")
    ax[1].set_title("B  chain by chain (above the line = PB better)")
    ax[1].legend(frameon=False, fontsize=6)
    for k, ion in enumerate(ions):
        sub = M[M["ion"] == ion]
        for j, lev in enumerate(["<=100", "101-300", ">300"]):
            g = sub[sub["length_bin"] == lev]
            if g["pdb_id"].nunique() < 3 or not pbs:
                continue
            best = max(pbs, key=lambda s: sub[sub["score"] == s].groupby("pdb_id")["auroc"].mean().mean())
            a = g[g["score"] == best].groupby("pdb_id")["auroc"].mean()
            b = g[g["score"] == "phos_screened"].groupby("pdb_id")["auroc"].mean()
            ax[2].bar(j + 0.35 * k, (a - b).dropna().mean(), width=0.33, color=f"C{k}", label=ion if j == 0 else None)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xticks([0.17, 1.17, 2.17])
    ax[2].set_xticklabels(["<=100 nt", "101-300 nt", ">300 nt"])
    ax[2].set_ylabel("AUROC(best PB) - AUROC(phosphate screened)")
    ax[2].set_title("C  does the advantage depend on chain length?")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_ion_pilot.png", dpi=300)
    fig.savefig(out / "fig_ion_pilot.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, default=Path("/data/rna_electrostats/phi_res"))
    ap.add_argument("--out-dir", type=Path, default=Path("/data/rna_electrostats/ion_pilot"))
    ap.add_argument("--pipeline-module", default="~/projects/codes/rna_mg/rna_electrostatics_pipeline_v2.py")
    ap.add_argument("--ions", nargs="+", default=["MG", "K"])
    ap.add_argument("--cutoff", type=float, default=5.0, help="max P-ion distance (A) for a positive residue")
    ap.add_argument("--min-occ", type=float, default=0.5)
    ap.add_argument("--keep-low-res", action="store_true", help="keep ions flagged low-resolution (default: drop them)")
    ap.add_argument("--methods", nargs="*", default=None, help="keep only these experimental methods (see the counts first)")
    ap.add_argument("--min-p", type=int, default=8, help="min phosphate-bearing residues per chain")
    ap.add_argument("--min-neg", type=int, default=3, help="min negative residues per chain")
    ap.add_argument("--max-structures", type=int, default=None)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.ions = [x.upper() for x in a.ions]
    a.out_dir.mkdir(parents=True, exist_ok=True)

    dirs = sorted(p.parent.name for p in (a.labels_dir / "work").glob("*/per_residue_potential.csv"))
    if a.max_structures:
        dirs = dirs[:a.max_structures]
    log.info("%d structures", len(dirs))
    jobs = [{"pdb_id": d, "labels_dir": str(a.labels_dir), "pipeline_module": a.pipeline_module, "ions": a.ions,
             "cutoff": a.cutoff, "min_occ": a.min_occ, "keep_low_res": a.keep_low_res} for d in dirs]
    results = []
    with ProcessPoolExecutor(max_workers=a.n_workers) as ex:
        futs = [ex.submit(process_structure, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if n % 200 == 0:
                log.info("%d / %d", n, len(jobs))
    bad = [r for r in results if r["status"] != "ok"]
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        for r in bad[:5]:
            log.error("%s: %s", r["pdb_id"], r["error"])
        raise SystemExit("no structure could be processed (first errors above)")
    R = pd.concat([r["res"] for r in ok], ignore_index=True)
    if a.methods:
        R = R[R["method"].isin(a.methods)]
    R.to_csv(a.out_dir / "ion_labels_5A.csv.gz", index=False)

    L = []
    say = L.append
    say("ION-SITE PILOT (Step 0)")
    say(f"structures processed: {len(ok)}   failed: {len(bad)}" + ("   e.g. " + "; ".join(f"{r['pdb_id']}: {r['error']}" for r in bad[:3]) if bad else ""))
    I = pd.DataFrame([r["ions"] | {"pdb_id": r["pdb_id"]} for r in ok])
    say("\nRESOLUTION AND ION AUDIT (what the labels are built on)")
    say(f"  ions in ions.csv: {int(I['n_ions_total'].sum())}; removed for low-resolution flag: {int(I['n_removed_lowres'].sum())}"
        f"{' (kept: --keep-low-res)' if a.keep_low_res else ''}; removed for occupancy < {a.min_occ}: {int(I['n_removed_occ'].sum())}")
    for el in a.ions:
        say(f"  {el}: {int(I[f'n_{el}'].sum())} ions kept in {int((I[f'n_{el}'] > 0).sum())} structures")
    if I["resolution_A"].notna().any():
        q = I["resolution_A"].dropna()
        say(f"  resolution of structures (A): median {q.median():.2f}; share > 3.0 A: {100 * (q > 3.0).mean():.0f}%; share > 3.5 A: {100 * (q > 3.5).mean():.0f}%")
    meth = I["experimental_method"].astype(str).replace("", "unknown").value_counts()
    say("  experimental methods: " + "; ".join(f"{k} {v}" for k, v in meth.items()))
    say(f"\nresidues: {len(R)} phosphate-bearing; positives within {a.cutoff:g} A: " + ", ".join(
        f"{el} {int(R[f'pos_{el}'].sum())} ({100 * R[f'pos_{el}'].mean():.1f}%)" for el in a.ions) + f", any {int(R['pos_any'].sum())}")
    M = chain_metrics(R, a.ions, a.min_p, a.min_neg, a.seed)
    M.to_csv(a.out_dir / "pilot_chain_metrics.csv", index=False)
    if M.empty:
        say("\nno chains qualify (need >= 1 positive and >= min-neg negatives)")
    else:
        say(summarize(M, a.ions, a.n_boot, a.seed))
        make_figure(M, a.ions, a.out_dir, a.n_boot, a.seed)
    text = "\n".join(L)
    (a.out_dir / "pilot_summary.txt").write_text(text + "\n")
    print(text)
    print(f"\nfiles in {a.out_dir}: ion_labels_5A.csv.gz, pilot_chain_metrics.csv, pilot_summary.txt, fig_ion_pilot.png/.pdf")


if __name__ == "__main__":
    sys.exit(main())
