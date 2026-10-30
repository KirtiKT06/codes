#!/usr/bin/env python
"""
Predicted-structure electrostatics benchmark.

QUESTION
    Does the electrostatic potential map computed from a PREDICTED RNA structure
    agree, residue by residue, with the map from the native structure -- and is
    that agreement better or worse than what a geometric error of the same size
    (same C1' RMSD) would give by itself?

    If the electrostatic map is worse than the RMSD-matched null, the field
    exposes errors that RMSD does not. If it is the same, the field adds nothing
    beyond RMSD and the paper should not claim otherwise.

WHAT IS COMPARED (important design choice)
    Predictors fold ONE chain from its sequence. The stored labels in
    all_residue_potentials.csv were computed on the WHOLE deposited complex, so
    neighbouring chains contribute. To compare like with like, this script
    recomputes the NATIVE chain IN ISOLATION through exactly the same route as the
    predicted chain. The reference is "native, isolated, ion-free", never the stored label.

SUBCOMMANDS
    fasta     write FASTA input for the structure predictor (test split, length window)
    run       score every predicted chain found in --pred-dir
    summarize aggregate chain_metrics.csv into the numbers/tests for the paper

ENGINES (--engine)
    apbs     pdb2pqr + APBS with the SAME settings as your pipeline (uses your module)
    coulomb  pdb2pqr + screened-Coulomb features from coulomb_residual_test.py (fast)

PREDICTED FILE NAMING
    <pred-dir>/<PDBID>_<chain>.pdb   chain = the chain letter in your labels table / FASTA header
    Needs all heavy atoms (pdb2pqr stops on 'too few atoms'). Coarse outputs
    (backbone-only) must be rebuilt to all-atom first.

USAGE
    python predicted_structure_benchmark.py fasta --labels-dir /data/rna_electrostats/phi_res \
        --out test_chains.fasta --min-len 20 --max-len 250
    python predicted_structure_benchmark.py run --labels-dir /data/rna_electrostats/phi_res \
        --pred-dir preds/ --outdir bench_out --pipeline-module ~/projects/codes/rna_mg/rna_electrostatics_pipeline_v2.py \
        --engine coulomb --workers 8
    python predicted_structure_benchmark.py summarize --outdir bench_out
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import tempfile
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pred_bench")

QTY = ["c1p", "p", "m4", "m8", "m12"]
PRIMARY_QTY = ["c1p", "m4", "m8", "m12"]          # P point value excluded from conclusions
RNA_BASES = {
    "A": "A", "RA": "A", "ADE": "A", "A5": "A", "A3": "A", "RA5": "A", "RA3": "A",
    "C": "C", "RC": "C", "CYT": "C", "C5": "C", "C3": "C", "RC5": "C", "RC3": "C",
    "G": "G", "RG": "G", "GUA": "G", "G5": "G", "G3": "G", "RG5": "G", "RG3": "G",
    "U": "U", "RU": "U", "URA": "U", "U5": "U", "U3": "U", "RU5": "U", "RU3": "U",
}


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def assign_split(pdb_id: str) -> str:
    """Same rule as your pipeline (md5 of the upper-case PDB id, 80/10/10)."""
    b = int(hashlib.md5(pdb_id.upper().encode()).hexdigest(), 16) % 100
    return "train" if b < 80 else ("val" if b < 90 else "test")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).expanduser()))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_pdb_atoms(path: Path):
    """(lines, DataFrame of ATOM/HETATM records with a residue index in file order)."""
    lines = Path(path).read_text().splitlines(keepends=True)
    rows, prev, rid = [], None, -1
    for i, ln in enumerate(lines):
        if not ln.startswith(("ATOM", "HETATM")) or len(ln) < 54:
            continue
        key = (ln[17:20], ln[21], ln[22:27])
        if key != prev:
            rid += 1
            prev = key
        rows.append({"line_no": i, "name": ln[12:16].strip(), "resname": ln[17:20].strip(),
                     "res_idx": rid, "x": float(ln[30:38]), "y": float(ln[38:46]), "z": float(ln[46:54])})
    return lines, pd.DataFrame(rows)


def write_pdb_with_xyz(lines, atoms: pd.DataFrame, xyz: np.ndarray, out: Path):
    new = list(lines)
    for (_, r), (x, y, z) in zip(atoms.iterrows(), xyz):
        ln = new[int(r["line_no"])]
        new[int(r["line_no"])] = ln[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + ln[54:]
    Path(out).write_text("".join(new))


def c1p_coords(atoms: pd.DataFrame) -> np.ndarray:
    """One C1' per residue, residue order. Raises if any residue lacks it."""
    sel = atoms[atoms["name"].isin(["C1'", "C1*"])].drop_duplicates("res_idx").sort_values("res_idx")
    n_res = atoms["res_idx"].nunique()
    if len(sel) != n_res:
        raise ValueError(f"{n_res - len(sel)} residues lack C1'")
    return sel[["x", "y", "z"]].to_numpy(float)


def prepare_chain_pdb(src: Path, dst: Path, expect_bases: str | None = None) -> str:
    """Normalise a single-chain PDB: first model, altloc A/blank, RNA residues only, chain 'A',
    residues renumbered 1..N in file order, atoms renumbered. Returns the base string.
    Raises if expect_bases is given and does not match."""
    out, bases = [], []
    prev, resno, serial, model_seen = None, 0, 0, False
    for ln in Path(src).read_text().splitlines():
        if ln.startswith("ENDMDL"):
            break
        if not ln.startswith(("ATOM", "HETATM")) or len(ln) < 54:
            continue
        if ln[16] not in (" ", "A"):
            continue
        resname = ln[17:20].strip()
        if resname not in RNA_BASES:
            continue
        key = (resname, ln[21], ln[22:27])
        if key != prev:
            resno += 1
            prev = key
            bases.append(RNA_BASES[resname])
        serial += 1
        ln = ln.ljust(80)
        out.append("ATOM  " + f"{serial:5d}" + ln[11:16] + " " + f"{RNA_BASES[resname]:>3}" + " A" + f"{resno:4d}" + " "
                   + ln[27:].rstrip())
    seq = "".join(bases)
    if expect_bases is not None and seq != expect_bases:
        if len(seq) != len(expect_bases):
            raise ValueError(f"length {len(seq)} != native {len(expect_bases)}")
        raise ValueError("sequence differs from native")
    Path(dst).write_text("\n".join(out) + "\nEND\n")
    return seq


# ----------------------------------------------------------------------
# structure metrics
# ----------------------------------------------------------------------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD of P onto Q after optimal rigid superposition."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    diff = (R @ Pc.T).T - Qc
    return float(np.sqrt((diff ** 2).sum() / len(P)))


def lddt_c1p(ref: np.ndarray, mod: np.ndarray, radius: float = 15.0,
             thresholds=(0.5, 1.0, 2.0, 4.0)) -> float:
    """Superposition-free C1'-only lDDT in [0,1]."""
    n = len(ref)
    Dr = np.linalg.norm(ref[:, None] - ref[None], axis=2)
    Dm = np.linalg.norm(mod[:, None] - mod[None], axis=2)
    mask = (Dr < radius) & ~np.eye(n, dtype=bool)
    per = []
    for i in range(n):
        m = mask[i]
        if not m.any():
            continue
        dd = np.abs(Dr[i, m] - Dm[i, m])
        per.append(np.mean([(dd < t).mean() for t in thresholds]))
    return float(np.mean(per)) if per else float("nan")


# ----------------------------------------------------------------------
# RMSD-matched perturbation nulls
# ----------------------------------------------------------------------

def _smooth_field(xyz: np.ndarray, rng, wavelength: float = 40.0, n_modes: int = 6) -> np.ndarray:
    """Long-wavelength random displacement field: preserves local geometry, bends/shears the fold."""
    d = np.zeros_like(xyz)
    for _ in range(n_modes):
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        k = (2 * np.pi / wavelength) * rng.uniform(0.7, 1.3) * w
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        d += np.cos(xyz @ k + rng.uniform(0, 2 * np.pi))[:, None] * v
    return d


def _solve_scale(c1: np.ndarray, disp_c1: np.ndarray, target: float) -> float:
    """Scale s so that Kabsch-RMSD(c1 + s*disp, c1) == target (bisection; monotone for small/moderate s)."""
    if target <= 1e-6:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        if kabsch_rmsd(c1 + hi * disp_c1, c1) >= target:
            break
        hi *= 2.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if kabsch_rmsd(c1 + mid * disp_c1, c1) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _solve_scale_lddt(c1: np.ndarray, disp_c1: np.ndarray, target_lddt: float) -> float:
    """Scale s so that C1'-lDDT(native, native + s*disp) == target_lddt (bisection; lDDT falls as s grows)."""
    if not np.isfinite(target_lddt) or target_lddt >= 0.999:
        return 0.0
    lo, hi = 0.0, 0.5
    for _ in range(40):
        if lddt_c1p(c1, c1 + hi * disp_c1) <= target_lddt or hi > 1e3:
            break
        hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if lddt_c1p(c1, c1 + mid * disp_c1) > target_lddt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


NULL_KINDS = ("smooth", "jitter", "smoothL", "jitterL")


def make_null_xyz(atoms: pd.DataFrame, target_rmsd: float, kind: str, seed: int,
                  target_lddt: float | None = None) -> np.ndarray:
    """New coordinates for ALL atoms of the native chain, distorted so the damage matches the prediction.
    kind='smooth'  : long-wavelength bending/shear,                 matched on C1' RMSD
    kind='jitter'  : every residue moved rigidly by its own vector, matched on C1' RMSD
    kind='smoothL' : as 'smooth' but matched on C1' lDDT (needs target_lddt)
    kind='jitterL' : as 'jitter' but matched on C1' lDDT (needs target_lddt)
    RMSD-matching punishes a prediction that gets every helix right but mis-orients two of them; lDDT-matching
    does not. Reporting both shows whether the conclusion depends on which damage measure you match."""
    if kind not in NULL_KINDS:
        raise ValueError(kind)
    by_lddt = kind.endswith("L")
    base = kind[:-1] if by_lddt else kind
    rng = np.random.default_rng(seed)
    xyz = atoms[["x", "y", "z"]].to_numpy(float)
    c1 = c1p_coords(atoms)
    if base == "smooth":
        disp_all = _smooth_field(xyz, rng)
        disp_c1 = _smooth_field(c1, np.random.default_rng(seed))   # same seed -> same field
    else:
        per_res = rng.normal(size=(atoms["res_idx"].max() + 1, 3))
        disp_all = per_res[atoms["res_idx"].to_numpy()]
        disp_c1 = per_res
    if by_lddt:
        if target_lddt is None:
            raise ValueError("lDDT-matched null needs target_lddt")
        s = _solve_scale_lddt(c1, disp_c1, target_lddt)
    else:
        s = _solve_scale(c1, disp_c1, target_rmsd)
    return xyz + s * disp_all


# ----------------------------------------------------------------------
# potentials
# ----------------------------------------------------------------------

def potentials_for_pdb(pdb: Path, workdir: Path, engine: str, pipe, coul, coulomb_family: str) -> pd.DataFrame:
    """seq_index (0-based) + c1p, p, m4, m8, m12 for one single-chain PDB (residues numbered 1..N)."""
    workdir.mkdir(parents=True, exist_ok=True)
    pqr = workdir / "x.pqr"
    pipe.run_pdb2pqr(pdb, pqr)
    pipe.reformat_pqr_for_apbs(pqr)
    if engine == "apbs":
        settings = pipe.ApbsSettings()
        try:
            apbs_in = pipe.build_apbs_input(pqr, workdir, settings)
            dx = pipe.run_apbs(apbs_in, pqr)
            solver = "nonlinear"
        except RuntimeError as e:
            msg = str(e).lower()
            if any(w in msg for w in ("overflow", "large potential", "converge", "unphysical")):
                import dataclasses
                settings = dataclasses.replace(settings, nonlinear=False)
                apbs_in = pipe.build_apbs_input(pqr, workdir, settings)
                dx = pipe.run_apbs(apbs_in, pqr)
                solver = "linearized_fallback"
            else:
                raise
        df = pipe.per_residue_potential(pqr, dx)
        out = pd.DataFrame({
            "seq_index": df["resnum"].astype(int) - 1,
            "c1p": df["potential_at_c1prime_kT_e"], "p": df["potential_at_phosphorus_kT_e"],
            "m4": df["potential_mean_kT_e_4A"], "m8": df["potential_mean_kT_e_8A"],
            "m12": df["potential_mean_kT_e_12A"]})
        out.attrs["solver"] = solver
        return out

    # coulomb engine
    atoms = coul.read_pqr(pqr)
    if atoms.empty:
        raise RuntimeError("empty PQR")
    xyz, q = atoms[["x", "y", "z"]].to_numpy(float), atoms["q"].to_numpy(float)
    keys = atoms[["chain", "resnum", "resname"]].drop_duplicates().reset_index(drop=True)
    c1xyz, pxyz, center = coul.residue_reference_points(atoms, keys)
    specs = {n: (pref, kap) for n, pref, kap in coul.make_family_specs([0.15], 1.0, 80.0, 300.0)}
    if coulomb_family not in specs:
        raise ValueError(f"coulomb family {coulomb_family} not in {list(specs)}")
    pref, kap = specs[coulomb_family]
    vals = {"c1p": coul.coulomb_sums(c1xyz, xyz, q, kap, {}, point=True)["point"],
            "p": coul.coulomb_sums(pxyz, xyz, q, kap, {}, point=True)["point"],
            **coul.coulomb_sums(center, xyz, q, kap, coul.RADII, point=False)}
    out = pd.DataFrame({"seq_index": keys["resnum"].astype(int) - 1})
    for k in QTY:
        out[k] = pref * vals[k]
    out.attrs["solver"] = "coulomb_" + coulomb_family
    return out


# ----------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------

def compare_potentials(ref: pd.DataFrame, other: pd.DataFrame, prefix: str) -> dict:
    """Within-chain centred comparison of two per-residue potential tables."""
    m = ref.merge(other, on="seq_index", suffixes=("_r", "_o"))
    res = {f"{prefix}_n_common": int(len(m))}
    for q in QTY:
        a, b = m[f"{q}_r"].to_numpy(float), m[f"{q}_o"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10 or np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
            for k in ("pearson", "spearman", "mae", "w1_over_sd", "mean_offset"):
                res[f"{prefix}_{k}_{q}"] = np.nan
            continue
        a, b = a[ok], b[ok]
        ac, bc = a - a.mean(), b - b.mean()
        res[f"{prefix}_pearson_{q}"] = float(stats.pearsonr(ac, bc)[0])
        res[f"{prefix}_spearman_{q}"] = float(stats.spearmanr(ac, bc)[0])
        res[f"{prefix}_mae_{q}"] = float(np.mean(np.abs(ac - bc)))
        res[f"{prefix}_w1_over_sd_{q}"] = float(stats.wasserstein_distance(ac, bc) / np.std(ac))
        res[f"{prefix}_mean_offset_{q}"] = float(b.mean() - a.mean())
    return res


# ----------------------------------------------------------------------
# per-chain job
# ----------------------------------------------------------------------

def run_chain(job: dict) -> dict:
    pdb_id, chain = job["pdb_id"], job["chain"]
    outdir = Path(job["outdir"])
    res_json = outdir / "chains" / f"{pdb_id}_{chain}.json"
    row = {"pdb_id": pdb_id, "chain": chain, "status": "ok", "error": ""}
    if res_json.exists() and not job["overwrite"]:
        return json.loads(res_json.read_text())
    try:
        pipe = load_module(job["pipeline_module"], "pipe_mod")
        coul = load_module(job["coulomb_module"], "coul_mod") if job["engine"] == "coulomb" else None

        native_src = Path(job["labels_dir"]) / "work" / pdb_id / f"{pdb_id.lower()}_complete_residues.pdb"
        if not native_src.exists():
            raise FileNotFoundError(f"missing {native_src}")
        with tempfile.TemporaryDirectory(dir=job.get("tmp_dir")) as td:
            td = Path(td)
            # isolate the native chain
            iso = td / "native_raw.pdb"
            iso.write_text("".join(ln for ln in native_src.read_text().splitlines(keepends=True)
                                   if ln.startswith(("ATOM", "HETATM")) and len(ln) > 21 and ln[21] == chain))
            nat = td / "native.pdb"
            native_bases = prepare_chain_pdb(iso, nat)
            pred = td / "pred.pdb"
            pred_bases = prepare_chain_pdb(Path(job["pred_path"]), pred, expect_bases=native_bases)

            n_lines, n_atoms = read_pdb_atoms(nat)
            p_lines, p_atoms = read_pdb_atoms(pred)
            c1_nat, c1_pred = c1p_coords(n_atoms), c1p_coords(p_atoms)
            row.update(n_res=len(native_bases), rmsd_c1p=kabsch_rmsd(c1_pred, c1_nat),
                       lddt_c1p=lddt_c1p(c1_nat, c1_pred))

            tabs = {"native": potentials_for_pdb(nat, td / "w_native", job["engine"], pipe,
                                                 coul, job["coulomb_family"])}
            row["solver"] = tabs["native"].attrs.get("solver", "")
            tabs["pred"] = potentials_for_pdb(pred, td / "w_pred", job["engine"], pipe, coul, job["coulomb_family"])
            row.update(compare_potentials(tabs["native"], tabs["pred"], "pred"))

            for kind in job["null_kinds"]:
                for k in range(job["n_null"]):
                    xyz = make_null_xyz(n_atoms, row["rmsd_c1p"], kind,
                                        seed=1000 * k + zlib.crc32((pdb_id + chain).encode()) % 997,
                                        target_lddt=row["lddt_c1p"])
                    npdb = td / f"null_{kind}_{k}.pdb"
                    write_pdb_with_xyz(n_lines, n_atoms, xyz, npdb)
                    tabs[f"null_{kind}_{k}"] = potentials_for_pdb(npdb, td / f"w_null_{kind}_{k}", job["engine"],
                                                                  pipe, coul, job["coulomb_family"])
                # average the null metrics over seeds
                nul = [compare_potentials(tabs["native"], tabs[f"null_{kind}_{k}"], "n")
                       for k in range(job["n_null"])]
                for key in nul[0]:
                    v = np.array([d[key] for d in nul], float)
                    base = key.replace("n_", f"null_{kind}_", 1)
                    row[base + "_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
                    row[base + "_sd"] = float(np.nanstd(v)) if np.isfinite(v).any() else np.nan

            # per-residue table for map plots
            wide = tabs["native"].add_prefix("native_").rename(columns={"native_seq_index": "seq_index"})
            for name, t in tabs.items():
                if name == "native":
                    continue
                wide = wide.merge(t.add_prefix(name + "_").rename(columns={name + "_seq_index": "seq_index"}),
                                  on="seq_index", how="outer")
            wide.insert(0, "chain", chain)
            wide.insert(0, "pdb_id", pdb_id)
            (outdir / "per_residue").mkdir(parents=True, exist_ok=True)
            wide.to_csv(outdir / "per_residue" / f"{pdb_id}_{chain}.csv.gz", index=False)
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        log.warning("%s_%s failed: %s", pdb_id, chain, row["error"])
        log.debug(traceback.format_exc())
    (outdir / "chains").mkdir(parents=True, exist_ok=True)
    res_json.write_text(json.dumps(row, default=float))
    return row


# ----------------------------------------------------------------------
# subcommands
# ----------------------------------------------------------------------

def index_pred_dir(pred_dir: Path) -> dict:
    """{(PDBID upper, chain): path}. Accepts <ID>_<chain>.pdb, any case, any suffix after the chain
    (e.g. 1abc_A_model.pdb), and files inside sub-folders."""
    idx = {}
    for p in sorted(Path(pred_dir).rglob("*.pdb")):
        parts = p.stem.split("_")
        if len(parts) >= 2 and len(parts[1]) == 1:
            idx.setdefault((parts[0].upper(), parts[1]), p)
    return idx


def read_fasta_dir(labels_dir: Path):
    """[(pdb_id, chain, seq)] from work/<ID>/<id>.fasta (headers '>ID_chain')."""
    out = []
    for fa in sorted((labels_dir / "work").glob("*/*.fasta")):
        name = fa.parent.name
        hdr, seq = None, []
        for ln in fa.read_text().splitlines() + [">END"]:
            if ln.startswith(">"):
                if hdr is not None:
                    pid, _, ch = hdr[1:].rpartition("_")
                    out.append((pid or name, ch, "".join(seq)))
                hdr, seq = ln, []
            else:
                seq.append(ln.strip())
    return out


def cmd_fasta(a):
    rows = [(p, c, s) for p, c, s in read_fasta_dir(Path(a.labels_dir))
            if assign_split(p) == a.split and a.min_len <= len(s) <= a.max_len]
    with open(a.out, "w") as f:
        for p, c, s in rows:
            f.write(f">{p}_{c}\n{s}\n")
    pd.DataFrame(rows, columns=["pdb_id", "chain", "seq"]).assign(length=lambda d: d.seq.str.len()) \
        .drop(columns="seq").to_csv(Path(a.out).with_suffix(".manifest.csv"), index=False)
    log.info("wrote %d %s-split chains (%d-%d nt) to %s", len(rows), a.split, a.min_len, a.max_len, a.out)


def cmd_run(a):
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(a.pred_dir)
    if not pred_dir.is_dir():
        sys.exit(f"ERROR: --pred-dir {pred_dir} does not exist (looked relative to {Path.cwd()})")
    chains = read_fasta_dir(Path(a.labels_dir))
    idx = index_pred_dir(pred_dir)
    log.info("%d chains with FASTA under %s/work; %d predicted PDB files found in %s",
             len(chains), a.labels_dir, len(idx), pred_dir)
    jobs = []
    for p, c, _ in chains:
        pp = idx.get((p.upper(), c))
        if pp is None:
            continue
        jobs.append(dict(pdb_id=p, chain=c, pred_path=str(pp), outdir=str(outdir), labels_dir=a.labels_dir,
                         pipeline_module=str(Path(a.pipeline_module).expanduser()),
                         coulomb_module=str(Path(a.coulomb_module).expanduser()), engine=a.engine,
                         coulomb_family=a.coulomb_family, null_kinds=a.null_kinds, n_null=a.n_null,
                         overwrite=a.overwrite, tmp_dir=a.tmp_dir))
    if not jobs:
        ex_fa = [f"{p}_{c}" for p, c, _ in chains[:5]]
        ex_pr = [f"{k[0]}_{k[1]}" for k in list(idx)[:5]]
        sys.exit("ERROR: 0 predicted chains matched a FASTA chain.\n"
                 f"  FASTA chains look like : {ex_fa}\n"
                 f"  predicted files look like: {ex_pr}  (expected <PDBID>_<chain>.pdb, e.g. 1EHZ_A.pdb)\n"
                 "  If the FASTA list is empty, check --labels-dir (needs work/<ID>/<id>.fasta). "
                 "If the predicted list is empty, check --pred-dir and the file names.")
    if a.max_chains:
        jobs = jobs[:a.max_chains]
    log.info("%d predicted chains to score (engine=%s)", len(jobs), a.engine)
    rows = []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for fut in as_completed([ex.submit(run_chain, j) for j in jobs]):
                rows.append(fut.result())
    else:
        rows = [run_chain(j) for j in jobs]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "chain_metrics.csv", index=False)
    log.info("done: %d ok, %d failed -> %s", int((df.status == "ok").sum()), int((df.status != "ok").sum()),
             outdir / "chain_metrics.csv")
    for _, r in df[df.status != "ok"].head(5).iterrows():
        log.info("  failed %s_%s: %s", r.pdb_id, r.chain, r.error)


def boot_median_ci(values, groups, n_boot=2000, seed=0):
    """Structure-level bootstrap of the median (chains of one structure move together)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    if d.empty:
        return np.nan, np.nan, np.nan
    by = {g: x.v.to_numpy() for g, x in d.groupby("g")}
    keys = list(by)
    meds = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys))
        meds.append(np.median(np.concatenate([by[keys[i]] for i in pick])))
    return float(np.median(d.v)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def cmd_summarize(a):
    outdir = Path(a.outdir)
    df = pd.read_csv(outdir / "chain_metrics.csv")
    ok = df[df.status == "ok"].copy()
    lines = [f"chains scored: {len(ok)} of {len(df)} (failed: {int((df.status != 'ok').sum())})"]
    if ok.empty:
        (outdir / "summary.txt").write_text("\n".join(lines))
        print("\n".join(lines))
        return
    lines.append(f"median C1' RMSD {ok.rmsd_c1p.median():.2f} A, median C1'-lDDT {ok.lddt_c1p.median():.3f}")
    kinds = sorted({c.split("_")[1] for c in ok.columns if c.startswith("null_") and c.endswith("_mean")})
    excess_rows = []
    for q in PRIMARY_QTY:
        pc = f"pred_pearson_{q}"
        if pc not in ok:
            continue
        med, lo, hi = boot_median_ci(ok[pc], ok.pdb_id)
        lines.append(f"\n[{q}] pred-vs-native within-chain Pearson: median {med:.3f} (95% CI {lo:.3f}..{hi:.3f}); "
                     f"median W1/sd {ok[f'pred_w1_over_sd_{q}'].median():.3f}")
        for k in kinds:
            nc = f"null_{k}_pearson_{q}_mean"
            if nc not in ok:
                continue
            d = (ok[pc] - ok[nc]).dropna()
            ok[f"excess_{k}_{q}"] = ok[pc] - ok[nc]
            if len(d) < 5:
                continue
            m2, l2, h2 = boot_median_ci(ok[f"excess_{k}_{q}"], ok.pdb_id)
            w = stats.wilcoxon(d) if (d != 0).any() else None
            lines.append(f"    vs RMSD-matched '{k}' null (median {ok[nc].median():.3f}): excess = pred - null "
                         f"median {m2:+.3f} (CI {l2:+.3f}..{h2:+.3f}); pred worse in {(d < 0).mean():.0%} of chains"
                         + (f"; Wilcoxon p={w.pvalue:.3g}" if w is not None else ""))
            excess_rows.append({"qty": q, "null": k, "median_excess": m2, "ci_lo": l2, "ci_hi": h2,
                                "frac_pred_worse": float((d < 0).mean()), "n": int(len(d))})
        r = stats.spearmanr(ok.rmsd_c1p, ok[pc], nan_policy="omit")
        lines.append(f"    Spearman(RMSD, pred Pearson) = {r.statistic:.2f}; "
                     f"Spearman(lDDT, pred Pearson) = {stats.spearmanr(ok.lddt_c1p, ok[pc], nan_policy='omit').statistic:.2f}")
    lines.append("\nNull kinds: smooth/jitter are matched on C1' RMSD; smoothL/jitterL are matched on C1' lDDT.")
    lines.append("\nInterpretation guide:")
    lines.append("  excess < 0 with CI below 0 : the electrostatic map is damaged MORE than geometry error of the same size "
                 "would predict -> the field carries information RMSD does not.")
    lines.append("  excess ~ 0                 : the map is just a function of RMSD; no added value over RMSD/lDDT.")
    lines.append("  excess > 0                 : predictions are electrostatically better than RMSD suggests.")
    pd.DataFrame(excess_rows).to_csv(outdir / "excess_vs_null.csv", index=False)
    ok.to_csv(outdir / "chain_metrics_with_excess.csv", index=False)
    (outdir / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fasta")
    f.add_argument("--labels-dir", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--split", default="test")
    f.add_argument("--min-len", type=int, default=20)
    f.add_argument("--max-len", type=int, default=250)
    f.set_defaults(fn=cmd_fasta)

    r = sub.add_parser("run")
    r.add_argument("--labels-dir", required=True)
    r.add_argument("--pred-dir", required=True)
    r.add_argument("--outdir", required=True)
    r.add_argument("--pipeline-module", required=True, help="path to rna_electrostatics_pipeline_v2.py")
    r.add_argument("--coulomb-module", default=str(Path(__file__).with_name("coulomb_residual_test.py")))
    r.add_argument("--engine", choices=["apbs", "coulomb"], default="coulomb")
    r.add_argument("--coulomb-family", default="dhP_I0.15", help="vac | dhS_I0.15 | dhP_I0.15")
    r.add_argument("--null-kinds", nargs="+", default=list(NULL_KINDS), choices=list(NULL_KINDS),
                   help="smooth/jitter = matched on C1' RMSD; smoothL/jitterL = matched on C1' lDDT")
    r.add_argument("--n-null", type=int, default=3)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--max-chains", type=int, default=0)
    r.add_argument("--tmp-dir", default=None)
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("summarize")
    s.add_argument("--outdir", required=True)
    s.set_defaults(fn=cmd_summarize)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
