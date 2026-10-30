#!/usr/bin/env python
"""
Mg2+ sensitivity check for the per-residue electrostatic labels.

QUESTION
    Your labels were computed with the RNA only: Mg2+ (and every other ion) was stripped from the
    PDB before the PB solve. How much would the labels change if the resolved structural Mg2+ ions
    were kept as fixed +2 charges inside the solute?

    The answer decides how to word the paper:
      * labels barely change  -> "ion-free labels" is a harmless simplification, say so in one sentence.
      * labels change a lot near Mg2+ -> Mg2+-aware labels are a second result worth reporting,
        and the audit should be rerun on them (this script writes a drop-in labels folder for that).

WHAT IT DOES (per structure)
  1. Reads work/<ID>/ions.csv (written by your pipeline from the RAW structure) and the PQR.
  2. Keeps ions that are real structural ions: right element (default MG), resolution <= 3.0 A
     (your low_resolution_flag), occupancy >= 0.5, within --max-ion-dist (default 5.0 A) of a heavy RNA atom;
     duplicate altloc positions are merged.
  3. Appends them to the PQR as fixed charges (+2, radius --mg-radii) and reruns APBS with EXACTLY your
     pipeline settings (nonlinear PB, 0.15 M KCl, pdie 1, sdie 80, linearised fallback).
  4. Control run: the same ions with charge 0 (same cavity, same grid, no charge). The difference between this
     and the stored labels is the cavity/grid effect; the difference between the Mg run and this control is the
     pure CHARGE effect of Mg2+.
  5. Compares with the stored (ion-free) labels, within chain (the quantity the audit uses).

OUTPUT (in --outdir)
  mg_sensitivity_by_chain.csv     per structure/chain/variant/quantity: centred Pearson, RMSE (kT/e), global offset ...
  mg_effect_by_distance.csv       pooled over residues: effect size vs distance to the nearest Mg2+
  mg_sensitivity_summary.txt      the numbers to quote + a plain reading
  per_residue/<ID>.csv.gz         base and Mg labels per residue, with distance to nearest ion
  labels_r<radius>/               DROP-IN labels folder (same layout as your phi_res: all_residue_potentials.csv +
                                  work/<ID>/...) so centered_latent_manifold.py --labels-dir labels_r0.79 reruns the
                                  audit on Mg2+-aware labels (subset of structures only)

USAGE
    python mg_sensitivity.py --labels-dir /data/rna_electrostats/phi_res --outdir /data/rna_electrostats/mg_check \
        --pipeline-module ~/projects/codes/rna_mg/rna_electrostatics_pipeline_v2.py --n-structures 120 --workers 8
    (run a quick pilot first: add --n-structures 6)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import logging
import shutil
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mg_sensitivity")

LABEL_COL = {"c1p": "potential_at_c1prime_kT_e", "p": "potential_at_phosphorus_kT_e",
             "m4": "potential_mean_kT_e_4A", "m8": "potential_mean_kT_e_8A", "m12": "potential_mean_kT_e_12A"}
PRIMARY = ["c1p", "m4", "m8", "m12"]          # P point value is not used for conclusions
ION_CHARGE = {"MG": 2.0, "MN": 2.0, "CA": 2.0, "ZN": 2.0, "CO": 2.0, "NI": 2.0, "CD": 2.0, "SR": 2.0,
              "BA": 2.0, "FE": 2.0, "CU": 2.0}
DIST_BINS = [0, 4, 6, 8, 12, 20, 1e9]


def assign_split(pdb_id: str) -> str:
    b = int(hashlib.md5(pdb_id.upper().encode()).hexdigest(), 16) % 100
    return "train" if b < 80 else ("val" if b < 90 else "test")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).expanduser()))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# ion selection
# ----------------------------------------------------------------------

def select_ions(ions_csv: Path, atoms: pd.DataFrame, resnames: set, max_dist: float, min_occ: float,
                allow_low_res: bool, dedupe: float = 0.5) -> pd.DataFrame:
    """Structural ions to add: element in `resnames`, good resolution, occupied, close to a heavy RNA atom,
    deduplicated. Returns columns resname, x, y, z (+ distance_to_rna)."""
    cols = ["resname", "x", "y", "z", "distance_to_rna"]
    if not Path(ions_csv).exists() or Path(ions_csv).stat().st_size < 5:
        return pd.DataFrame(columns=cols)
    try:
        ions = pd.read_csv(ions_csv)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)
    if ions.empty or "resname" not in ions:
        return pd.DataFrame(columns=cols)
    ions = ions[ions["resname"].astype(str).str.upper().isin(resnames)].copy()
    if "occupancy" in ions:
        ions = ions[~(ions["occupancy"] < min_occ)]               # NaN occupancy is kept
    if not allow_low_res and "low_resolution_flag" in ions:
        low = ions["low_resolution_flag"].astype(str).str.lower().isin(["true", "1"])
        ions = ions[~low]
    if ions.empty:
        return pd.DataFrame(columns=cols)
    heavy = atoms[~atoms["atom_name"].astype(str).str.upper().str.startswith("H")]
    tree = cKDTree(heavy[["x", "y", "z"]].to_numpy(float))
    d, _ = tree.query(ions[["x", "y", "z"]].to_numpy(float), k=1)
    ions = ions.assign(distance_to_rna=d)
    ions = ions[ions["distance_to_rna"] <= max_dist]
    keep, kept_xyz = [], []
    for i, r in ions.iterrows():                                # merge altloc duplicates
        p = np.array([r.x, r.y, r.z], float)
        if any(np.linalg.norm(p - q) < dedupe for q in kept_xyz):
            continue
        kept_xyz.append(p)
        keep.append(i)
    out = ions.loc[keep, cols].copy()
    out["resname"] = out["resname"].astype(str).str.upper()
    return out.reset_index(drop=True)


def append_ions_to_pqr(pqr_lines: list, ions: pd.DataFrame, radius: float, charge_scale: float) -> tuple[list, str]:
    """Return (new lines, chain letter used). Ions go in as HETATM with a chain letter that is free in this PQR."""
    used = {ln[21] for ln in pqr_lines if ln.startswith(("ATOM", "HETATM")) and len(ln) > 21}
    chain = next((c for c in "ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqponmlkjihgfedcba" if c not in used), None)
    if chain is None:
        raise RuntimeError("no free chain letter for the ions")
    serial = 0
    for ln in pqr_lines:
        if ln.startswith(("ATOM", "HETATM")):
            try:
                serial = max(serial, int(ln[6:11]))
            except ValueError:
                pass
    body = [ln for ln in pqr_lines if not ln.startswith(("END", "TER"))]
    for k, r in enumerate(ions.itertuples(index=False), 1):
        serial += 1
        q = ION_CHARGE.get(r.resname, 2.0) * charge_scale
        body.append(f"HETATM{serial % 100000:5d} {r.resname:<4s} {r.resname:>3s} {chain}{9000 + k:4d}    "
                    f"{r.x:8.3f}{r.y:8.3f}{r.z:8.3f}{q:8.4f}{radius:7.4f}")
    body.append("END")
    return body, chain


# ----------------------------------------------------------------------
# one APBS solve
# ----------------------------------------------------------------------

def solve(pipe, pqr_lines: list, workdir: Path, drop_chain: str | None):
    workdir.mkdir(parents=True, exist_ok=True)
    pqr = workdir / "x.pqr"
    pqr.write_text("\n".join(pqr_lines) + "\n")
    settings = pipe.ApbsSettings()
    solver = "nonlinear"
    try:
        dx = pipe.run_apbs(pipe.build_apbs_input(pqr, workdir, settings), pqr)
    except RuntimeError as e:
        msg = str(e).lower()
        if not any(w in msg for w in ("overflow", "large potential", "converge", "unphysical")):
            raise
        solver = "linearized_fallback"
        settings = dataclasses.replace(settings, nonlinear=False)
        dx = pipe.run_apbs(pipe.build_apbs_input(pqr, workdir, settings), pqr)
    df = pipe.per_residue_potential(pqr, dx)
    if drop_chain is not None:
        df = df[df["chain"] != drop_chain]
    return df.reset_index(drop=True), solver


# ----------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------

def centred(a: np.ndarray) -> np.ndarray:
    return a - np.nanmean(a)


def compare_chain(base: np.ndarray, other: np.ndarray) -> dict:
    ok = np.isfinite(base) & np.isfinite(other)
    out = {"n": int(ok.sum())}
    if ok.sum() < 10 or np.std(base[ok]) < 1e-9:
        return {**out, "pearson_c": np.nan, "rmse_c": np.nan, "delta_mean": np.nan, "var_frac_changed": np.nan,
                "r2_c": np.nan}
    b, o = base[ok], other[ok]
    bc, oc = centred(b), centred(o)
    dc = oc - bc
    out.update(pearson_c=float(stats.pearsonr(bc, oc)[0]), rmse_c=float(np.sqrt(np.mean(dc ** 2))),
               delta_mean=float(np.mean(o - b)), var_frac_changed=float(np.var(dc) / np.var(bc)),
               r2_c=float(1 - np.sum(dc ** 2) / np.sum(bc ** 2)))
    return out


# ----------------------------------------------------------------------
# per-structure job
# ----------------------------------------------------------------------

def run_structure(job: dict):
    pdb_id = job["pdb_id"]
    labels_dir, out_dir = Path(job["labels_dir"]), Path(job["outdir"])
    res = {"pdb_id": pdb_id, "status": "ok", "error": "", "rows": [], "dist_rows": []}
    try:
        pipe = load_module(job["pipeline_module"], "pipe_mod")
        work = labels_dir / "work" / pdb_id
        base = pd.read_csv(work / "per_residue_potential.csv", dtype={"pdb_id": str})
        pqr_path = work / f"{pdb_id.lower()}.pqr"
        atoms = pipe.parse_pqr_residues(pqr_path)
        ions = select_ions(work / "ions.csv", atoms, set(job["ion_names"]), job["max_ion_dist"], job["min_occ"],
                           job["allow_low_res"])
        res["n_ions"] = int(len(ions))
        if len(ions) < job["min_ions"]:
            res["status"] = "skipped"
            res["error"] = f"only {len(ions)} qualifying ions"
            return res
        pqr_lines = pqr_path.read_text().splitlines()
        base_solver = str(base["pb_solver"].iloc[0]) if "pb_solver" in base else "unknown"
        variants = {}
        ion_xyz = ions[["x", "y", "z"]].to_numpy(float)
        with tempfile.TemporaryDirectory(dir=job.get("tmp_dir")) as td:
            td = Path(td)
            for ri, radius in enumerate(job["radii"]):
                lines, ch = append_ions_to_pqr(pqr_lines, ions, radius, 1.0)
                variants[f"mg_r{radius:g}"] = solve(pipe, lines, td / f"mg_{ri}", ch)
                if ri == 0 and job["null_control"]:
                    lines0, ch0 = append_ions_to_pqr(pqr_lines, ions, radius, 0.0)
                    variants[f"mg0_r{radius:g}"] = solve(pipe, lines0, td / "mg0", ch0)

        key = ["chain", "resnum"]
        merged = base[key + ["x", "y", "z"] + list(LABEL_COL.values())].copy()
        tree = cKDTree(ion_xyz)
        merged["d_nearest_ion"] = tree.query(merged[["x", "y", "z"]].to_numpy(float), k=1)[0]
        merged = merged.rename(columns={v: f"base_{k}" for k, v in LABEL_COL.items()})
        for vname, (df, solver) in variants.items():
            sub = df[key + list(LABEL_COL.values())].rename(columns={v: f"{vname}_{k}" for k, v in LABEL_COL.items()})
            merged = merged.merge(sub, on=key, how="left")
            res[f"solver_{vname}"] = solver
        res["solver_base"] = base_solver

        for chain, g in merged.groupby("chain"):
            for vname in variants:
                for q in LABEL_COL:
                    c = compare_chain(g[f"base_{q}"].to_numpy(float), g[f"{vname}_{q}"].to_numpy(float))
                    res["rows"].append({"pdb_id": pdb_id, "chain": chain, "n_res": len(g), "n_ions": len(ions),
                                        "variant": vname, "qty": q, "solver_mismatch":
                                        variants[vname][1] != base_solver, **c})
        # pooled-by-distance table input (within-chain centred deltas)
        for vname in variants:
            for q in LABEL_COL:
                d = np.full(len(merged), np.nan)
                for chain, idx in merged.groupby("chain").indices.items():
                    b, o = merged[f"base_{q}"].to_numpy(float)[idx], merged[f"{vname}_{q}"].to_numpy(float)[idx]
                    ok = np.isfinite(b) & np.isfinite(o)
                    if ok.sum() >= 10:
                        dd = np.full(len(idx), np.nan)
                        dd[ok] = (o[ok] - o[ok].mean()) - (b[ok] - b[ok].mean())
                        d[idx] = dd
                merged[f"dc_{vname}_{q}"] = d
        (out_dir / "per_residue").mkdir(parents=True, exist_ok=True)
        merged.insert(0, "pdb_id", pdb_id)
        merged.to_csv(out_dir / "per_residue" / f"{pdb_id}.csv.gz", index=False)
        res["dist_path"] = str(out_dir / "per_residue" / f"{pdb_id}.csv.gz")

        # drop-in labels folders (Mg run of each radius)
        for radius in job["radii"]:
            vname = f"mg_r{radius:g}"
            ldir = out_dir / f"labels_r{radius:g}" / "work" / pdb_id
            ldir.mkdir(parents=True, exist_ok=True)
            for f in work.iterdir():
                if f.is_file() and f.name != "per_residue_potential.csv" and f.suffix in (".csv", ".fasta"):
                    shutil.copy(f, ldir / f.name)
            new = base.copy()
            df, solver = variants[vname]
            sub = df[key + list(LABEL_COL.values())]
            new = new.drop(columns=list(LABEL_COL.values())).merge(sub, on=key, how="left")
            new = new[list(base.columns)]
            new["pb_solver"] = solver
            new["mg_ions_included"] = len(ions)
            new.to_csv(ldir / "per_residue_potential.csv", index=False)
    except Exception as exc:
        res["status"] = "failed"
        res["error"] = f"{type(exc).__name__}: {exc}"[:300]
        log.debug(traceback.format_exc())
    return res


# ----------------------------------------------------------------------
# structure selection
# ----------------------------------------------------------------------

def candidate_structures(labels_dir: Path, ion_names: set, min_ions: int, max_residues: int,
                         allow_low_res: bool, min_occ: float) -> list:
    """Cheap pre-filter from ions.csv alone (no PQR read): structures with enough good ions of the right type."""
    out = []
    for d in sorted((labels_dir / "work").glob("*")):
        f = d / "ions.csv"
        if not f.exists() or f.stat().st_size < 5 or not (d / "per_residue_potential.csv").exists():
            continue
        try:
            ions = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if ions.empty or "resname" not in ions:
            continue
        ions = ions[ions["resname"].astype(str).str.upper().isin(ion_names)]
        if "occupancy" in ions:
            ions = ions[~(ions["occupancy"] < min_occ)]
        if not allow_low_res and "low_resolution_flag" in ions:
            ions = ions[~ions["low_resolution_flag"].astype(str).str.lower().isin(["true", "1"])]
        if len(ions) >= min_ions:
            n_res = sum(1 for _ in open(d / "per_residue_potential.csv")) - 1
            if n_res <= max_residues:
                out.append(d.name)
    return out


def pick(cands: list, n: int, min_test: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    test = [c for c in cands if assign_split(c) == "test"]
    rest = [c for c in cands if assign_split(c) != "test"]
    rng.shuffle(test)
    rng.shuffle(rest)
    take = test[:min(min_test, n)]
    take += rest[: max(0, n - len(take))]
    if len(take) < n:
        take += [c for c in test if c not in take][: n - len(take)]
    return sorted(take)


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------

def summarize(outdir: Path, results: list, radii: list):
    rows = [r for res in results for r in res["rows"]]
    by = pd.DataFrame(rows)
    by.to_csv(outdir / "mg_sensitivity_by_chain.csv", index=False)
    lines = []
    ok_structs = [r for r in results if r["status"] == "ok"]
    lines.append(f"structures analysed: {len(ok_structs)}; skipped: {sum(r['status']=='skipped' for r in results)}; "
                 f"failed: {sum(r['status']=='failed' for r in results)}")
    if by.empty:
        (outdir / "mg_sensitivity_summary.txt").write_text("\n".join(lines))
        print("\n".join(lines))
        return
    by = by[~by["solver_mismatch"]]
    lines.append(f"chains: {by[['pdb_id','chain']].drop_duplicates().shape[0]}; median ions per structure: "
                 f"{int(np.median([r['n_ions'] for r in ok_structs]))}; chains kept after dropping solver-mismatched rows")
    for vname in dict.fromkeys(by["variant"]):
        lines.append(f"\n=== {vname} vs stored (ion-free) labels, within-chain centred ===")
        for q in PRIMARY:
            s = by[(by.variant == vname) & (by.qty == q)]
            if s.empty:
                continue
            lines.append(f"  [{q}] chain-median Pearson {s.pearson_c.median():.3f} (IQR {s.pearson_c.quantile(.25):.3f}.."
                         f"{s.pearson_c.quantile(.75):.3f}); RMSE of the within-chain change {s.rmse_c.median():.2f} kT/e; "
                         f"variance of the label changed {100*s.var_frac_changed.median():.1f}% (median); "
                         f"global offset (mean change) {s.delta_mean.median():+.2f} kT/e")

    # pooled by distance to nearest ion
    drows = []
    for res in ok_structs:
        if "dist_path" not in res:
            continue
        d = pd.read_csv(res["dist_path"])
        for vname in {r["variant"] for r in res["rows"]}:
            for q in PRIMARY:
                x = d[["d_nearest_ion", f"dc_{vname}_{q}", f"base_{q}"]].dropna()
                x = x.assign(bin=pd.cut(x["d_nearest_ion"], DIST_BINS), variant=vname, qty=q,
                             pdb_id=res["pdb_id"]).rename(columns={f"dc_{vname}_{q}": "dc", f"base_{q}": "base"})
                drows.append(x)
    if drows:
        P = pd.concat(drows)
        # centred base value per structure for scale
        agg = P.groupby(["variant", "qty", "bin"], observed=True).agg(
            n=("dc", "size"), mean_change=("dc", "mean"), sd_change=("dc", "std"),
            rms_change=("dc", lambda v: float(np.sqrt(np.mean(v ** 2)))))
        sd_base = P.groupby(["variant", "qty"])["base"].std()
        agg = agg.reset_index()
        agg["label_sd_pooled"] = [float(sd_base.loc[(r.variant, r.qty)]) for r in agg.itertuples()]
        agg["rms_change_over_label_sd"] = agg["rms_change"] / agg["label_sd_pooled"]
        agg.to_csv(outdir / "mg_effect_by_distance.csv", index=False)
        first = f"mg_r{radii[0]:g}"
        lines.append(f"\n=== {first}: how the change depends on distance to the nearest Mg2+ (within-chain centred) ===")
        for q in PRIMARY:
            a = agg[(agg.variant == first) & (agg.qty == q)]
            lines.append(f"  [{q}] " + "; ".join(f"{str(r.bin)} A: n={r.n}, rms change {r.rms_change:.2f} kT/e "
                                                  f"({100*r.rms_change_over_label_sd:.0f}% of label sd)" for r in a.itertuples()))
        null = f"mg0_r{radii[0]:g}"
        if null in set(agg.variant):
            lines.append(f"\n=== control {null} (same cavity, charge 0): this is the NON-charge part of the change ===")
            for q in ("m8",):
                a = agg[(agg.variant == null) & (agg.qty == q)]
                lines.append(f"  [{q}] " + "; ".join(f"{str(r.bin)} A: rms change {r.rms_change:.2f} kT/e" for r in a.itertuples()))

    first = f"mg_r{radii[0]:g}"
    m8 = by[(by.variant == first) & (by.qty == "m8")]
    if len(m8):
        r_med = m8.pearson_c.median()
        lines.append("\nPLAIN READING (m8, first radius):")
        if r_med > 0.95:
            lines.append(f"  Median within-chain correlation {r_med:.3f}: keeping Mg2+ barely reorders the residues. "
                         "Ion-free labels are a harmless simplification for the within-chain audit; state it in one sentence.")
        elif r_med > 0.85:
            lines.append(f"  Median within-chain correlation {r_med:.3f}: a modest change, concentrated near the ions "
                         "(see the distance table). Report the sensitivity and rerun the audit on labels_r* to show the conclusion holds.")
        else:
            lines.append(f"  Median within-chain correlation {r_med:.3f}: Mg2+ changes the labels a lot. "
                         "Rerun the audit on labels_r* and treat Mg2+-aware labels as a second result.")
    (outdir / "mg_sensitivity_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--pipeline-module", required=True, help="path to rna_electrostatics_pipeline_v2.py")
    ap.add_argument("--ions", nargs="+", default=["MG"], help="element names to include (default MG)")
    ap.add_argument("--mg-radii", type=float, nargs="+", default=[0.79],
                    help="ion radius in A for the PB cavity (default 0.79, AMBER-style Mg2+). Several values = "
                         "a radius sensitivity; the first one is used for the headline numbers")
    ap.add_argument("--max-ion-dist", type=float, default=5.0, help="keep ions within this distance of a heavy RNA atom")
    ap.add_argument("--min-occupancy", type=float, default=0.5)
    ap.add_argument("--allow-low-resolution", action="store_true", help="also keep ions from structures worse than 3.0 A")
    ap.add_argument("--min-ions", type=int, default=3, help="structures need at least this many qualifying ions")
    ap.add_argument("--max-residues", type=int, default=600)
    ap.add_argument("--n-structures", type=int, default=120)
    ap.add_argument("--min-test", type=int, default=25, help="make sure this many structures are from the test split")
    ap.add_argument("--structures", nargs="+", default=None, help="explicit PDB ids instead of sampling")
    ap.add_argument("--no-null-control", action="store_true", help="skip the charge-0 control run (halves the cost)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp-dir", default=None)
    a = ap.parse_args()

    ion_names = {s.upper() for s in a.ions}
    a.outdir.mkdir(parents=True, exist_ok=True)
    if a.structures:
        chosen = [s.upper() for s in a.structures]
    else:
        cands = candidate_structures(a.labels_dir, ion_names, a.min_ions, a.max_residues, a.allow_low_resolution,
                                     a.min_occupancy)
        log.info("%d structures have >= %d good %s ions and <= %d residues", len(cands), a.min_ions,
                 "/".join(sorted(ion_names)), a.max_residues)
        if not cands:
            sys.exit("ERROR: no candidate structures; check --labels-dir (needs work/<ID>/ions.csv) or relax the filters")
        chosen = pick(cands, a.n_structures, a.min_test, a.seed)
    log.info("running %d structures x %d APBS solves each", len(chosen),
             len(a.mg_radii) + (0 if a.no_null_control else 1))

    jobs = [dict(pdb_id=p, labels_dir=str(a.labels_dir), outdir=str(a.outdir),
                 pipeline_module=str(Path(a.pipeline_module).expanduser()), ion_names=sorted(ion_names),
                 radii=a.mg_radii, max_ion_dist=a.max_ion_dist, min_occ=a.min_occupancy,
                 allow_low_res=a.allow_low_resolution, min_ions=a.min_ions, null_control=not a.no_null_control,
                 tmp_dir=a.tmp_dir) for p in chosen]
    results = []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, fut in enumerate(as_completed([ex.submit(run_structure, j) for j in jobs]), 1):
                r = fut.result()
                results.append(r)
                if i % 10 == 0 or i == len(jobs):
                    log.info("%d/%d done", i, len(jobs))
    else:
        results = [run_structure(j) for j in jobs]
    for r in results:
        if r["status"] != "ok":
            log.info("  %s %s: %s", r["status"], r["pdb_id"], r["error"])

    # drop-in labels table for each radius
    for radius in a.mg_radii:
        ldir = a.outdir / f"labels_r{radius:g}"
        parts = [pd.read_csv(f, dtype={"pdb_id": str}) for f in sorted((ldir / "work").glob("*/per_residue_potential.csv"))]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(ldir / "all_residue_potentials.csv", index=False)
            log.info("wrote %s (%d structures)", ldir / "all_residue_potentials.csv", len(parts))
    summarize(a.outdir, results, a.mg_radii)


if __name__ == "__main__":
    main()
