#!/usr/bin/env python
"""
SALT SCAN of the electrostatic potential around the phosphates of one structure (default: the fluoride riboswitch 4ENC).

QUESTION (from the PI)
    Solve Poisson-Boltzmann for the RNA alone at many monovalent salt concentrations. Does the potential near the
    "critical" phosphates (the ligand-binding domain: A6, U7, G8, U41, G42, plus G5 whose phosphate binds a Mg2+ in
    the paper's simulations) behave differently from the other phosphates, and do positions that keep a strongly
    negative potential at high salt coincide with the crystal Mg2+ sites?  Kumar & Reddy, J Phys Chem B 127, 9267 (2023).

WHAT IT DOES
  1. Reads work/<ID>/<id>.pqr (RNA only, ions already stripped) and reruns APBS with EXACTLY the pipeline settings
     (nonlinear PB, pdie 1, sdie 80, linearised fallback) except ionic_strength_M, for every salt in --salts-mM.
  2. For every residue and every salt, around the P atom (and, for comparison, around C1'), the mean potential inside
     spheres of radius --radii (default 4, 5, 6, 7, 7.1 ... 7.9, 8, 9, 10, 11, 12 A), the mean inside the SHELL between
     consecutive radii, and the point value (radius 0). The 5 A and 7.3 A spheres match the radii the paper uses.
     Also log < exp(-2 phi) > over the ion-ACCESSIBLE points of each sphere: a crude Boltzmann weight for a divalent
     cation (relative ranking only; absolute Mg2+ binding is outside PB).
  3. Grid-level, two regions within --max-atom-dist (8 A) of the RNA: SOLVENT = outside every atom's radius + --probe
     (2.0 A, the pipeline's APBS ion radius; screened solution, used for the Boltzmann weight and the main candidate
     sites) and TIGHT = at least --min-atom-dist (1.8 A) from every heavy-atom centre (admits crystal inner-sphere
     positions, ~2 A from O; includes cavity points where the potential is unscreened). At every salt: the potential at
     each crystal ion (percentile among TIGHT points; 0 = most negative) with its nearest phosphate, and the deepest
     local minima of each region ("candidate cation sites") with the nearest P of each.
  4. Analysis + figures: ranking of the critical phosphates, salt dependence critical vs others, robustness over
     radius, crystal-ion check, and a check that the C1' sphere means at 0.15 M reproduce your stored labels.

RUN (in the `venv` environment, which has APBS and gridData; ~1 h for 28 salts with 4 workers)
    python salt_scan.py --pdb-id 4ENC --n-workers 4
    python salt_scan.py --pdb-id 4ENC --analyze-only        # redo tables and figures from saved results

Results are cached per salt (outdir/per_salt/*.pkl); an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import logging
import pickle
import shutil
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree
from scipy.special import logsumexp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("salt_scan")

DEFAULT_SALTS_MM = list(range(10, 201, 10)) + list(range(300, 1001, 100))          # 20 + 8 = 28 solves
DEFAULT_RADII = [4, 5, 6, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8, 9, 10, 11, 12]
DEFAULT_CRITICAL = [5, 6, 7, 8, 41, 42]          # G5 + the five LBD nucleotides A6, U7, G8, U41, G42 (PDB numbering)
PAPER_RADII = (5.0, 7.3)                         # r_c of the paper's local-Mg2+ (5 A) and local-F- (7.3 A) windows
REF_SALT_M = 0.150                               # the salt used for the stored labels


def load_module(path: str, name: str = "pipe_mod"):
    spec = importlib.util.spec_from_file_location(name, str(Path(path).expanduser()))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------------------------
def read_grid(dx: Path):
    from gridData import Grid
    g = Grid(str(dx))
    return np.asarray(g.grid, float), np.asarray(g.origin, float), np.asarray(g.delta, float)


def stamp(origin, delta, dims, xyz, radius) -> np.ndarray:
    """Boolean grid, True within `radius` (scalar or per atom) of any point in xyz (spheres stamped onto the grid)."""
    out = np.zeros(tuple(dims), bool)
    radius = np.broadcast_to(np.asarray(radius, float), (len(xyz),))
    for c, r in zip(xyz, radius):
        lo = np.clip(np.floor((c - r - origin) / delta).astype(int), 0, dims)
        hi = np.clip(np.ceil((c + r - origin) / delta).astype(int) + 1, 0, dims)
        if np.any(hi <= lo):
            continue
        xs = origin[0] + np.arange(lo[0], hi[0]) * delta[0]
        ys = origin[1] + np.arange(lo[1], hi[1]) * delta[1]
        zs = origin[2] + np.arange(lo[2], hi[2]) * delta[2]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= ((gx - c[0]) ** 2 + (gy - c[1]) ** 2 + (gz - c[2]) ** 2) < r * r
    return out


def accessible_masks(origin, delta, dims, heavy_xyz, heavy_radius, dmin: float, dmax: float, probe: float):
    """Two regions within dmax of the RNA:
       solvent: outside every atom's radius + probe (what PB treats as ion-accessible solution; potentials are screened)
       tight  : at least dmin from every heavy-atom centre (admits crystal inner-sphere ion positions, which are ~2 A from O)"""
    near = stamp(origin, delta, dims, heavy_xyz, dmax)
    solvent = near & ~stamp(origin, delta, dims, heavy_xyz, np.asarray(heavy_radius, float) + probe)
    tight = near & ~stamp(origin, delta, dims, heavy_xyz, dmin)
    return solvent, tight


def sphere_stats(values, origin, delta, dims, center, radii, acc=None):
    """Per radius: mean, std, n inside the sphere; mean and n of the shell between this radius and the previous one;
    log mean exp(-2 phi) over accessible points inside the sphere. Returns list of dicts (radius 0 = point value is
    added by the caller)."""
    radii = sorted(radii)
    rmax = radii[-1]
    lo = np.clip(np.floor((center - rmax - origin) / delta).astype(int), 0, dims)
    hi = np.clip(np.ceil((center + rmax - origin) / delta).astype(int) + 1, 0, dims)
    if np.any(hi <= lo):
        return []
    sub = values[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    asub = None if acc is None else acc[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    xs = origin[0] + np.arange(lo[0], hi[0]) * delta[0]
    ys = origin[1] + np.arange(lo[1], hi[1]) * delta[1]
    zs = origin[2] + np.arange(lo[2], hi[2]) * delta[2]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    d2 = (gx - center[0]) ** 2 + (gy - center[1]) ** 2 + (gz - center[2]) ** 2
    out, prev = [], np.zeros_like(d2, bool)
    for r in radii:
        m = d2 <= r * r
        v = sub[m]
        sh = m & ~prev
        vs = sub[sh]
        row = {"radius": float(r),
               "mean_sphere": float(v.mean()) if v.size else np.nan, "std_sphere": float(v.std()) if v.size else np.nan,
               "n_sphere": int(v.size),
               "mean_shell": float(vs.mean()) if vs.size else np.nan, "n_shell": int(vs.size), "logboltz2_sphere": np.nan}
        if asub is not None:
            va = sub[m & asub]
            if va.size:
                row["logboltz2_sphere"] = float(logsumexp(-2.0 * va) - np.log(va.size))
        out.append(row)
        prev = m
    return out


def find_sites(values, acc, origin, delta, n_sites: int, sep: float, size: int = 5):
    """Deepest local minima of the potential over the accessible region, greedily separated by >= sep A."""
    arr = np.where(acc, values, np.inf)
    mn = minimum_filter(arr, size=size, mode="nearest")
    cand = np.argwhere((arr == mn) & np.isfinite(arr))
    if len(cand) == 0:
        return []
    vals = arr[tuple(cand.T)]
    order = np.argsort(vals)
    xyz = origin + cand[order] * delta
    vs = vals[order]
    chosen = []
    for p, v in zip(xyz, vs):
        if all(np.linalg.norm(p - q) >= sep for q, _ in chosen):
            chosen.append((p, float(v)))
            if len(chosen) >= n_sites:
                break
    return chosen


# ----------------------------------------------------------------------------------------------------------------
# one salt
# ----------------------------------------------------------------------------------------------------------------
def solve_salt(job: dict) -> dict:
    salt_M = job["salt_mM"] / 1000.0
    out = {"salt_M": salt_M, "status": "ok", "error": "", "solver": "", "res_rows": [], "site_rows": [], "ion_rows": []}
    try:
        pipe = load_module(job["pipeline_module"])
        settings = dataclasses.replace(pipe.ApbsSettings(), ionic_strength_M=salt_M)
        if job.get("fine_padding") is not None:
            settings = dataclasses.replace(settings, fine_padding=float(job["fine_padding"]))
        if job.get("coarse_padding") is not None:
            settings = dataclasses.replace(settings, coarse_padding=float(job["coarse_padding"]))
        with tempfile.TemporaryDirectory(dir=job.get("tmp_dir")) as td:
            td = Path(td)
            pqr = td / "x.pqr"
            shutil.copy(job["pqr"], pqr)
            solver = "nonlinear"
            try:
                dx = pipe.run_apbs(pipe.build_apbs_input(pqr, td, settings), pqr)
            except RuntimeError as e:
                if not any(w in str(e).lower() for w in ("overflow", "large potential", "converge", "unphysical")):
                    raise
                solver = "linearized_fallback"
                settings = dataclasses.replace(settings, nonlinear=False)
                dx = pipe.run_apbs(pipe.build_apbs_input(pqr, td, settings), pqr)
            out["solver"] = solver
            values, origin, delta = read_grid(dx)
            dims = np.array(values.shape)
            if job["keep_dx"]:
                (Path(job["outdir"]) / "dx").mkdir(parents=True, exist_ok=True)
                shutil.copy(dx, Path(job["outdir"]) / "dx" / f"salt_{job['salt_mM']:04d}mM.dx")

            atoms = pipe.parse_pqr_residues(pqr)
            heavy = atoms[~atoms["atom_name"].astype(str).str.upper().str.startswith("H")]
            acc, acc_t = accessible_masks(origin, delta, dims, heavy[["x", "y", "z"]].to_numpy(float),
                                          heavy["radius"].to_numpy(float), job["min_atom_dist"],
                                          job["max_atom_dist"], job["probe"])
            acc_vals, acc_t_vals = values[acc], values[acc_t]
            omap = job["orig_map"]
            p_xyz, p_orig = [], []
            for (chain, resnum, resname), g in atoms.groupby(["chain", "resnum", "resname"], sort=False):
                oc, orn = omap.get((chain, int(resnum)), (chain, int(resnum)))
                centers = {}
                p = g[g["atom_name"] == "P"]
                c1 = g[g["atom_name"].isin(["C1'", "C1*"])]
                if len(p):
                    centers["P"] = p[["x", "y", "z"]].iloc[0].to_numpy(float)
                    p_xyz.append(centers["P"])
                    p_orig.append((oc, orn, resname))
                if len(c1):
                    centers["C1p"] = c1[["x", "y", "z"]].iloc[0].to_numpy(float)
                for cname, c in centers.items():
                    base = {"salt_M": salt_M, "chain": chain, "resnum": int(resnum), "original_chain": oc,
                            "original_resnum": int(orn), "resname": resname, "center": cname}
                    pt = pipe.trilinear_interpolate(values, origin, delta, dims, c)
                    out["res_rows"].append({**base, "radius": 0.0, "mean_sphere": pt, "std_sphere": np.nan,
                                            "n_sphere": 1, "mean_shell": np.nan, "n_shell": 0,
                                            "logboltz2_sphere": np.nan})
                    for row in sphere_stats(values, origin, delta, dims, c, job["radii"], acc if cname == "P" else None):
                        out["res_rows"].append({**base, **row})

            # candidate cation sites (two regions) and the crystal ions
            ptree = cKDTree(np.array(p_xyz)) if p_xyz else None
            stree = {}
            for mname, amask in (("solvent", acc), ("tight", acc_t)):
                sites = find_sites(values, amask, origin, delta, job["n_sites"], job["site_sep"])
                for k, (xyz, v) in enumerate(sites, 1):
                    near = {}
                    if ptree is not None:
                        d, i = ptree.query(xyz, k=1)
                        near = {"nearest_P_chain": p_orig[i][0], "nearest_P_resnum": p_orig[i][1],
                                "nearest_P_resname": p_orig[i][2], "nearest_P_dist": float(d)}
                    out["site_rows"].append({"salt_M": salt_M, "mask": mname, "rank": k, "x": xyz[0], "y": xyz[1],
                                             "z": xyz[2], "phi": v, **near})
                stree[mname] = cKDTree(np.array([x[0] for x in sites])) if sites else None
            for k, ion in enumerate(job["ions"], 1):
                q = np.array([ion["x"], ion["y"], ion["z"]], float)
                phi = pipe.trilinear_interpolate(values, origin, delta, dims, q)
                row = {"salt_M": salt_M, "ion": k, "resname": ion["resname"], "x": q[0], "y": q[1], "z": q[2],
                       "phi_at_ion": phi,
                       "pct_more_negative_tight": float((acc_t_vals < phi).mean()) if np.isfinite(phi) and acc_t_vals.size else np.nan,
                       "pct_more_negative_solvent": float((acc_vals < phi).mean()) if np.isfinite(phi) and acc_vals.size else np.nan}
                if ptree is not None:
                    d, i = ptree.query(q, k=1)
                    row.update({"nearest_P_chain": p_orig[i][0], "nearest_P_resnum": p_orig[i][1],
                                "nearest_P_resname": p_orig[i][2], "nearest_P_dist": float(d)})
                for mname in ("solvent", "tight"):
                    if stree.get(mname) is not None:
                        d, i = stree[mname].query(q, k=1)
                        row[f"dist_to_nearest_site_{mname}"], row[f"nearest_site_rank_{mname}"] = float(d), int(i) + 1
                    else:
                        row[f"dist_to_nearest_site_{mname}"], row[f"nearest_site_rank_{mname}"] = np.nan, np.nan
                out["ion_rows"].append(row)
    except Exception as exc:  # noqa: BLE001
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"[:500]
        log.warning("salt %s mM failed: %s", job["salt_mM"], out["error"])
        log.debug(traceback.format_exc())
    return out


# ----------------------------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------------------------
def read_ions(work: Path, resnames: set, raw_pdb: Path | None, dedupe: float = 0.5) -> list[dict]:
    ions = []
    csv = work / "ions.csv"
    if csv.exists() and csv.stat().st_size > 5:
        d = pd.read_csv(csv)
        if "resname" in d:
            d = d[d["resname"].astype(str).str.upper().isin(resnames)]
            if "occupancy" in d:
                d = d[~(d["occupancy"] < 0.5)]
            ions += [{"resname": str(r.resname).upper(), "x": float(r.x), "y": float(r.y), "z": float(r.z)}
                     for r in d.itertuples(index=False)]
    if raw_pdb is not None and Path(raw_pdb).exists():           # e.g. to add the F- ligand, which ions.csv lacks
        for ln in Path(raw_pdb).read_text().splitlines():
            if ln.startswith("HETATM") and ln[17:20].strip().upper() in resnames:
                ions.append({"resname": ln[17:20].strip().upper(), "x": float(ln[30:38]), "y": float(ln[38:46]),
                             "z": float(ln[46:54])})
    keep = []
    for ion in ions:                                              # merge duplicates (altlocs, file overlap)
        p = np.array([ion["x"], ion["y"], ion["z"]])
        if not any(ion["resname"] == k["resname"] and np.linalg.norm(p - np.array([k["x"], k["y"], k["z"]])) < dedupe
                   for k in keep):
            keep.append(ion)
    return keep


def read_orig_map(work: Path) -> dict:
    f = work / "residue_renumber_map.csv"
    if not f.exists():
        log.warning("%s not found: PDB numbering unknown, using the pipeline's residue numbers", f)
        return {}
    m = pd.read_csv(f)
    return {(r.chain, int(r.synthetic_resnum)): (r.original_chain, int(r.original_resnum)) for r in m.itertuples(index=False)}


# ----------------------------------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------------------------------
def short(resname) -> str:
    r = str(resname).strip()
    return r[1] if len(r) == 2 and r[0] == "R" and r[1] in "ACGU" else r


def tag(row) -> str:
    return f"{short(row.resname)}{int(row.original_resnum)}"


def analyze(outdir: Path, critical: list[int], work: Path, chain: str | None) -> str:
    R = pd.read_csv(outdir / "salt_scan_residues.csv.gz")
    if chain:
        R = R[R["original_chain"] == chain]
    R["critical"] = R["original_resnum"].isin(critical)
    P = R[(R["center"] == "P") & (R["radius"] > 0)].copy()
    salts = sorted(P["salt_M"].unique())
    radii = sorted(P["radius"].unique())
    ref_salt = salts[int(np.argmin(np.abs(np.array(salts) - REF_SALT_M)))]
    hi_salt = salts[-1]
    lines = []
    say = lines.append
    say(f"SALT SCAN SUMMARY  ({len(salts)} salts, {len(radii)} radii; critical = PDB residues {critical})")
    say(f"chains analysed: {sorted(P['original_chain'].unique())}; phosphorus-bearing residues: {P.groupby(['original_chain','original_resnum']).ngroups}\n")

    # 0. reproduction check
    base_csv = work / "per_residue_potential.csv"
    if base_csv.exists() and np.isclose(salts, REF_SALT_M).any():
        base = pd.read_csv(base_csv)
        say("0. REPRODUCTION CHECK: C1' sphere means at 0.15 M against the stored labels (should agree to ~1e-6 with the")
        say("   pipeline's default grid; a run with --fine-padding / --coarse-padding differs by the boundary/grid effect, which is itself informative)")
        for rad in (4.0, 8.0, 12.0):
            col = f"potential_mean_kT_e_{rad:g}A"
            sub = R[(R["center"] == "C1p") & np.isclose(R["radius"], rad) & np.isclose(R["salt_M"], ref_salt)]
            if col in base and not sub.empty:
                mm = sub.merge(base[["chain", "resnum", col]], on=["chain", "resnum"])
                say(f"   {rad:g} A: n = {len(mm)}, max |new - stored| = {np.nanmax(np.abs(mm['mean_sphere'] - mm[col])):.2e} kT/e")
        say("")

    def rank_table(salt, rad):
        s = P[np.isclose(P["salt_M"], salt) & np.isclose(P["radius"], rad)].dropna(subset=["mean_sphere"]).copy()
        s["rank"] = s["mean_sphere"].rank(method="min")           # 1 = most negative
        return s.sort_values("rank")

    # 1. ranking
    say("1. WHERE ARE THE MOST NEGATIVE PHOSPHATES? (P-centred sphere mean; rank 1 = most negative)")
    for rad in PAPER_RADII:
        for salt in (ref_salt, hi_salt):
            t = rank_table(salt, rad)
            if t.empty:
                continue
            crit = t[t["critical"]]
            say(f"   r = {rad:g} A, salt = {salt:g} M (n = {len(t)} residues)")
            say("      top 10: " + ", ".join(f"{tag(r)}({r.mean_sphere:.1f})" for r in t.head(10).itertuples()))
            say("      critical ranks: " + ", ".join(f"{tag(r)}={int(r.rank)}" for r in crit.itertuples()))
    say("")

    # 2. salt dependence
    say("2. SALT DEPENDENCE: slope of the sphere mean against ln(salt), and the fraction of the 10 mM potential left at 1 M")
    lns = np.log(np.array(salts))
    rows = []
    for rad in PAPER_RADII:
        sub = P[np.isclose(P["radius"], rad)]
        for (ch, rn), g in sub.groupby(["original_chain", "original_resnum"]):
            g = g.sort_values("salt_M")
            if g["mean_sphere"].notna().sum() < 3:                    # slopes need at least 3 salts
                continue
            slope = np.polyfit(np.log(g["salt_M"]), g["mean_sphere"], 1)[0]
            first, last = g["mean_sphere"].iloc[0], g["mean_sphere"].iloc[-1]
            rows.append({"radius": rad, "chain": ch, "resnum": rn, "resname": g["resname"].iloc[0], "critical": bool(g["critical"].iloc[0]),
                         "slope_per_ln_salt": slope, "kept_fraction": last / first if abs(first) > 1e-9 else np.nan,
                         "phi_low": first, "phi_high": last})
    S = pd.DataFrame(rows, columns=["radius", "chain", "resnum", "resname", "critical", "slope_per_ln_salt",
                                    "kept_fraction", "phi_low", "phi_high"])
    S.to_csv(outdir / "salt_dependence_by_residue.csv", index=False)
    if S.empty:
        say("   (fewer than 3 salts: no slopes computed)")
    for rad in PAPER_RADII:
        s = S[S["radius"] == rad]
        a, b = s[s["critical"]], s[~s["critical"]]
        if len(a) >= 2 and len(b) >= 2:
            for col, lab in (("slope_per_ln_salt", "slope (kT/e per ln salt)"), ("kept_fraction", "fraction kept at high salt")):
                p = stats.mannwhitneyu(a[col].dropna(), b[col].dropna()).pvalue
                say(f"   r = {rad:g} A  {lab:<30s} critical median {a[col].median():+.3f} (n={len(a)})  others median {b[col].median():+.3f} (n={len(b)})  Mann-Whitney p = {p:.3g}")
    say("   (every potential changes with salt; the question is whether the critical phosphates change MORE or are more resilient)\n")

    # 3. robustness over radius
    say("3. ROBUSTNESS OVER RADIUS: median rank of the critical phosphates, and how many are among the 6 most negative")
    say("   radius   salt=ref: median rank / in top 6      salt=1 M: median rank / in top 6")
    rs = ref_salt
    for rad in radii:
        cells = []
        for salt in (rs, hi_salt):
            t = rank_table(salt, rad)
            crit = t[t["critical"]]
            cells.append(f"{crit['rank'].median():5.1f} / {int((crit['rank'] <= 6).sum())}" if len(crit) else "n/a")
        say(f"   {rad:6g}   {cells[0]:<36s}{cells[1]}")
    say("")

    # 4. Boltzmann weight
    say("4. CRUDE DIVALENT-CATION PREFERENCE: log <exp(-2 phi)> over SOLVENT-accessible points within 5 A of P (higher = more favourable)")
    say("   The paper's simulations (Fig. 2A) put the most Mg2+ around A6 and G42, then G5-G8 and U41-G43.")
    for salt in sorted({rs, hi_salt}):
        s = P[np.isclose(P["salt_M"], salt) & np.isclose(P["radius"], 5.0)].dropna(subset=["logboltz2_sphere"]).copy()
        if s.empty:
            continue
        s["rank"] = s["logboltz2_sphere"].rank(ascending=False, method="min")
        s = s.sort_values("rank")
        say(f"   salt = {salt:g} M: top 8: " + ", ".join(f"{tag(r)}({r.logboltz2_sphere:.1f})" for r in s.head(8).itertuples()))
        say("      critical ranks: " + ", ".join(f"{tag(r)}={int(r.rank)}" for r in s[s['critical']].itertuples()))
    say("")

    # 5. crystal ions
    ion_f, site_f = outdir / "salt_scan_ions.csv", outdir / "salt_scan_sites.csv"
    I = pd.read_csv(ion_f) if ion_f.exists() else pd.DataFrame()
    if not I.empty:
        say("5. CRYSTAL IONS (ions were NOT in the calculation). 'tight' percentile = share of grid points >= 1.8 A from every heavy atom")
        say("   centre that have a MORE negative potential (0 % = deepest point of all); the fair comparison for inner-sphere positions.")
        for salt in sorted({salts[0], rs, hi_salt}):
            sub = I[np.isclose(I["salt_M"], salt)]
            say(f"   salt = {salt:g} M")
            for r in sub.itertuples():
                near = f"{short(r.nearest_P_resname)}{int(r.nearest_P_resnum)} ({r.nearest_P_dist:.1f} A)" if hasattr(r, "nearest_P_resnum") else "?"
                say(f"      {r.resname} #{r.ion} (nearest P {near}): phi = {r.phi_at_ion:8.2f} kT/e, tight percentile {100 * r.pct_more_negative_tight:5.1f} %, "
                    f"nearest tight-region site {r.dist_to_nearest_site_tight:.1f} A (rank {int(r.nearest_site_rank_tight) if np.isfinite(r.nearest_site_rank_tight) else -1}), "
                    f"nearest solvent site {r.dist_to_nearest_site_solvent:.1f} A")
        say("")
    if site_f.exists():
        Z = pd.read_csv(site_f)
        say("6. DEEPEST CANDIDATE CATION SITES (potential minima) and the phosphate nearest to each")
        for mname, lab in (("solvent", "solvent-accessible region (outside r_atom + 2.0 A)"), ("tight", "tight region (>= 1.8 A from atom centres)")):
            for salt in sorted({rs, hi_salt}):
                sub = Z[(Z["mask"] == mname) & np.isclose(Z["salt_M"], salt)].head(8)
                say(f"   {lab}, salt = {salt:g} M: " + "; ".join(f"#{int(r.rank)} {r.phi:.1f} kT/e near {short(r.nearest_P_resname)}{int(r.nearest_P_resnum)} ({r.nearest_P_dist:.1f} A)" for r in sub.itertuples()))
        say("")
    text = "\n".join(lines)
    (outdir / "salt_scan_summary.txt").write_text(text + "\n")
    try:
        make_figures(P, S, I, outdir, critical, rs, hi_salt)
    except Exception as exc:  # noqa: BLE001
        log.warning("figures failed: %s", exc)
    return text


def make_figures(P, S, I, outdir, critical, ref_salt, hi_salt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2))
    rad = PAPER_RADII[0]
    p5 = P[np.isclose(P["radius"], rad)]
    # A heat map
    piv = p5.pivot_table(index="salt_M", columns="original_resnum", values="mean_sphere")
    im = ax[0, 0].imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
    step = 1 if len(piv.index) <= 8 else 3
    ax[0, 0].set_yticks(range(0, len(piv.index), step))
    ax[0, 0].set_yticklabels([f"{v * 1000:g}" for v in piv.index[::step]])
    ax[0, 0].set_xticks(range(0, len(piv.columns), 4))
    ax[0, 0].set_xticklabels(piv.columns[::4])
    for j, c in enumerate(piv.columns):
        if c in critical:
            ax[0, 0].axvline(j, color="w", lw=0.4, alpha=0.7)
    ax[0, 0].set_xlabel("PDB residue number (white lines = critical phosphates)")
    ax[0, 0].set_ylabel("salt (mM; rows equally spaced, not to scale)")
    ax[0, 0].set_title(f"A  mean potential within {rad:g} Å of P (kT/e)")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.04)
    # B curves
    crit_ids = sorted(set(p5[p5["critical"]]["original_resnum"]))
    oth = p5[~p5["critical"]].pivot_table(index="salt_M", columns="original_resnum", values="mean_sphere")
    ax[0, 1].fill_between(oth.index * 1000, oth.quantile(0.1, axis=1), oth.quantile(0.9, axis=1), color="0.85", label="other phosphates, 10–90 %")
    ax[0, 1].plot(oth.index * 1000, oth.median(axis=1), color="0.4", label="other phosphates, median")
    cmap = plt.get_cmap("tab10")
    for k, rn in enumerate(crit_ids):
        g = p5[p5["original_resnum"] == rn].sort_values("salt_M")
        ax[0, 1].plot(g["salt_M"] * 1000, g["mean_sphere"], color=cmap(k), lw=1.4, label=f"{g['resname'].iloc[0]}{rn}")
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlabel("monovalent salt (mM)")
    ax[0, 1].set_ylabel("potential (kT/e)")
    ax[0, 1].set_title(f"B  potential against salt, {rad:g} Å sphere around P")
    ax[0, 1].legend(frameon=False, fontsize=6, ncol=2)
    # C slopes
    s5 = S[S["radius"] == rad].sort_values("resnum")
    ax[0, 2].bar(range(len(s5)), s5["slope_per_ln_salt"], color=["tab:red" if c else "0.7" for c in s5["critical"]])
    ax[0, 2].set_xticks(range(0, len(s5), 4))
    ax[0, 2].set_xticklabels(s5["resnum"].iloc[::4])
    ax[0, 2].set_xlabel("PDB residue number (red = critical)")
    ax[0, 2].set_ylabel("d potential / d ln(salt)  (kT/e)")
    ax[0, 2].set_title("C  salt response of each phosphate")
    # D ion percentile
    if not I.empty:
        for (rn, k), g in I.groupby(["resname", "ion"]):
            g = g.sort_values("salt_M")
            ax[1, 0].plot(g["salt_M"] * 1000, 100 * g["pct_more_negative_tight"], marker="o", ms=2.5, label=f"{rn} #{k}")
        ax[1, 0].set_xscale("log")
        ax[1, 0].set_xlabel("salt (mM)")
        ax[1, 0].set_ylabel("% of tight-accessible points more negative")
        ax[1, 0].set_title("D  crystal ions: how deep is the potential there?")
        ax[1, 0].legend(frameon=False, fontsize=6)
    # E radius robustness
    rr = []
    for rd in sorted(P["radius"].unique()):
        for salt, lab in ((ref_salt, f"{ref_salt * 1000:g} mM"), (hi_salt, f"{hi_salt * 1000:g} mM")):
            s = P[np.isclose(P["salt_M"], salt) & np.isclose(P["radius"], rd)].dropna(subset=["mean_sphere"]).copy()
            s["rank"] = s["mean_sphere"].rank(method="min")
            rr.append({"radius": rd, "salt": lab, "median_rank": s[s["critical"]]["rank"].median()})
    rr = pd.DataFrame(rr)
    for lab, g in rr.groupby("salt"):
        ax[1, 1].plot(g["radius"], g["median_rank"], marker="o", ms=3, label=lab)
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_xlabel("sphere radius around P (Å)")
    ax[1, 1].set_ylabel("median rank of critical phosphates (1 = most negative)")
    ax[1, 1].set_title("E  robustness over radius")
    ax[1, 1].legend(frameon=False, fontsize=6)
    # F radial profile of the critical vs other (shell means) at the reference salt
    sh = P[np.isclose(P["salt_M"], ref_salt)]
    for flag, lab, col in ((True, "critical", "tab:red"), (False, "others", "0.5")):
        g = sh[sh["critical"] == flag].groupby("radius")["mean_shell"].median()
        ax[1, 2].plot(g.index, g.values, marker="o", ms=3, color=col, label=lab)
    ax[1, 2].set_xlabel("outer radius of the shell (Å)")
    ax[1, 2].set_ylabel("median shell-mean potential (kT/e)")
    ax[1, 2].set_title(f"F  radial profile around P at {ref_salt * 1000:g} mM")
    ax[1, 2].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(outdir / "fig_salt_scan.png", dpi=300)
    fig.savefig(outdir / "fig_salt_scan.pdf")
    plt.close(fig)
    log.info("wrote %s", outdir / "fig_salt_scan.png")


# ----------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-id", default="4ENC")
    ap.add_argument("--labels-dir", type=Path, default=Path("/data/rna_electrostats/phi_res"))
    ap.add_argument("--pipeline-module", default="~/projects/codes/rna_mg/rna_electrostatics_pipeline_v2.py")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--salts-mM", type=float, nargs="+", default=None, help="default: 10..200 step 10, then 300..1000 step 100")
    ap.add_argument("--radii", type=float, nargs="+", default=DEFAULT_RADII)
    ap.add_argument("--critical-resnums", type=int, nargs="+", default=DEFAULT_CRITICAL, help="PDB residue numbers")
    ap.add_argument("--chain", default=None, help="analyse only this ORIGINAL chain (default: all chains)")
    ap.add_argument("--ion-resnames", nargs="+", default=["MG"], help="crystal ions to check (ions.csv / --raw-pdb)")
    ap.add_argument("--raw-pdb", type=Path, default=None, help="optional raw PDB with HETATM ions, e.g. to add F (use with --ion-resnames MG F)")
    ap.add_argument("--min-atom-dist", type=float, default=1.8, help="closest approach (A) of a cation to a heavy RNA atom")
    ap.add_argument("--max-atom-dist", type=float, default=8.0)
    ap.add_argument("--probe", type=float, default=2.0, help="ion radius (A) for the solvent-accessible region: outside r_atom + probe (the pipeline's APBS ion radius is 2.0)")
    ap.add_argument("--n-sites", type=int, default=15)
    ap.add_argument("--site-sep", type=float, default=3.0)
    ap.add_argument("--fine-padding", type=float, default=None,
                    help="override the pipeline's 15 A fine-grid padding (low salt: Debye length 30 A at 10 mM; try 25 to test)")
    ap.add_argument("--coarse-padding", type=float, default=None, help="override the pipeline's 40 A coarse-grid padding")
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--keep-dx", action="store_true", help="keep every potential grid (about 30 MB each)")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()

    pid = a.pdb_id.upper()
    work = a.labels_dir / "work" / pid
    outdir = a.outdir or Path(f"/data/rna_electrostats/salt_scan_{pid}")
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / "per_salt"
    cache.mkdir(exist_ok=True)
    salts = [int(s) if float(s).is_integer() else s for s in (a.salts_mM or DEFAULT_SALTS_MM)]

    if not a.analyze_only:
        pqr = work / f"{pid.lower()}.pqr"
        if not pqr.exists():
            raise SystemExit(f"ERROR: {pqr} not found")
        ions = read_ions(work, {x.upper() for x in a.ion_resnames}, a.raw_pdb)
        log.info("%s: %d salts, %d radii, %d crystal ions to check (%s)", pid, len(salts), len(a.radii), len(ions),
                 ", ".join(sorted({i['resname'] for i in ions})) or "none")
        jobs = []
        for s in salts:
            if (cache / f"salt_{s:g}mM.pkl").exists():
                continue
            jobs.append({"salt_mM": s, "pqr": str(pqr), "pipeline_module": a.pipeline_module, "radii": a.radii,
                         "orig_map": read_orig_map(work), "ions": ions, "min_atom_dist": a.min_atom_dist,
                         "max_atom_dist": a.max_atom_dist, "probe": a.probe, "n_sites": a.n_sites, "site_sep": a.site_sep,
                         "outdir": str(outdir), "tmp_dir": a.tmp_dir, "keep_dx": a.keep_dx,
                         "fine_padding": a.fine_padding, "coarse_padding": a.coarse_padding})
        log.info("%d solves to run (%d cached)", len(jobs), len(salts) - len(jobs))
        if jobs:
            with ProcessPoolExecutor(max_workers=a.n_workers) as ex:
                futs = {ex.submit(solve_salt, j): j["salt_mM"] for j in jobs}
                for n, f in enumerate(as_completed(futs), 1):
                    r = f.result()
                    if r["status"] == "ok":
                        (cache / f"salt_{futs[f]:g}mM.pkl").write_bytes(pickle.dumps(r))
                    log.info("[%d/%d] %g mM: %s %s", n, len(jobs), futs[f], r["status"], r["solver"] or r["error"])

    res, sites, ions_rows, solvers = [], [], [], {}
    for f in sorted(cache.glob("salt_*mM.pkl")):
        r = pickle.loads(f.read_bytes())
        res += r["res_rows"]
        sites += r["site_rows"]
        ions_rows += r["ion_rows"]
        solvers[r["salt_M"]] = r["solver"]
    if not res:
        raise SystemExit("no results: check the log above")
    pd.DataFrame(res).sort_values(["salt_M", "chain", "resnum", "center", "radius"]).to_csv(outdir / "salt_scan_residues.csv.gz", index=False)
    pd.DataFrame(sites).to_csv(outdir / "salt_scan_sites.csv", index=False)
    pd.DataFrame(ions_rows).to_csv(outdir / "salt_scan_ions.csv", index=False)
    fb = [s for s, v in solvers.items() if v != "nonlinear"]
    if fb:
        log.warning("linearised fallback was used at salts (M): %s", sorted(fb))
    text = analyze(outdir, a.critical_resnums, work, a.chain)
    print("\n" + text)
    print(f"files in {outdir}: salt_scan_residues.csv.gz, salt_scan_sites.csv, salt_scan_ions.csv, salt_dependence_by_residue.csv, "
          f"salt_scan_summary.txt, fig_salt_scan.png/.pdf")


if __name__ == "__main__":
    sys.exit(main())