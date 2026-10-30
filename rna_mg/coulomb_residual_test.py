"""
The COULOMB-RESIDUAL TEST.

Question
--------
Your labels come from a nonlinear Poisson-Boltzmann (PB) solve. A large part of any PB
potential is plain screened Coulomb physics: add up the atomic charges, divide by distance,
damp with the salt. A language model (or a structure feature) can only be credited with
"knowing electrostatics" for what is left AFTER that arithmetic has been done.

    residual = PB label  -  best (calibrated) point-charge screened-Coulomb estimate

This script
  (1) builds the point-charge baseline from the SAME .pqr files that went into APBS
      (charges are the AMBER partial charges; the PQR holds RNA only -- your pipeline strips
      ions, waters and proteins before pdb2pqr, so nothing else contributes);
  (2) reports how much of each PB label that baseline already explains (label-side test,
      no embeddings needed);
  (3) writes a table that centered_latent_manifold.py reads (--coulomb-table) to run the
      LM-side test: can the embeddings predict the residual?

The baseline, in plain words
----------------------------
For a residue we ask "what would the potential be if every atom were a point charge?" in
three flavours, because PB with a low-dielectric interior (pdie=1) and a high-dielectric
solvent (sdie=80) sits between them:
    vac   : bare Coulomb, dielectric = pdie, no salt screening
    dhS   : Debye-Hueckel, dielectric = sdie (water), screened by the salt (Debye length ~8 A at 0.15 M)
    dhP   : dielectric = pdie, screened by the salt
Each flavour is evaluated for the five labels:
    C1' point, phosphorus point        potential AT the atom (excluding that atom's own charge)
    4, 8, 12 A sphere means            EXACT average of the point-charge potential over a ball
                                       centred on the residue's reference atom (C1'), via closed-form
                                       shell-theorem formulas -- the same quantity your grid averaging
                                       approximates, with no grid and no singularity.
Defaults mirror your ApbsSettings: I = 0.150 M, pdie = 1, sdie = 80, T = 300 K.

What this baseline CANNOT do (so the residual is not "pure non-Coulomb"):
  * no dielectric boundary / solvent-exclusion effects (the real PB has these; they are
    partly what the residual contains -- they are geometric, i.e. burial-like);
  * no ion-accessible surface, no nonlinear saturation;
  * APBS smooths charges onto a grid and its point values at an atom include self-field
    artefacts; we exclude the atom's own charge.
A residual predicted by burial/DSSR features therefore means "PB beyond point charges =
geometry". A residual predicted by the LM BEYOND geometry is the interesting case.

Usage
-----
    python coulomb_residual_test.py \
        --labels-dir /data/rna_electrostats/phi_res \
        --out /data/rna_electrostats/coulomb_baseline.csv.gz \
        --structure-controls /data/rna_electrostats/structure_controls.csv.gz \
        --dssr-controls /data/rna_electrostats/dssr_controls.csv.gz
        --workers 10
    # optional, adds the "does Coulomb add anything beyond structure?" rows to the report:
        --structure-controls /data/rna_electrostats/structure_controls.csv.gz \
        --dssr-controls /data/rna_electrostats/dssr_controls.csv.gz
    # re-run only the report on an existing table:   --report-only
"""

from __future__ import annotations

import argparse
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("coulomb_residual_test")

KEYS = ["pdb_id", "chain", "seq_index"]
QTY = ["c1p", "p", "m4", "m8", "m12"]  # the five labels
LABEL_COL = {"c1p": "potential_at_c1prime_kT_e", "p": "potential_at_phosphorus_kT_e",
             "m4": "potential_mean_kT_e_4A", "m8": "potential_mean_kT_e_8A",
             "m12": "potential_mean_kT_e_12A"}
RADII = {"m4": 4.0, "m8": 8.0, "m12": 12.0}
COULOMB_EV_ANGSTROM = 14.3996454784  # e^2 / (4 pi eps0) in eV*A
KB_EV = 8.617333262e-5
NA_PER_A3_PER_M = 6.02214076e-4      # molecules per A^3 at 1 mol/L


# ----------------------------------------------------------------------
# PQR reading
# ----------------------------------------------------------------------

def read_pqr(path: Path) -> pd.DataFrame:
    """chain, resnum, resname, atom_name, x, y, z, q  (fixed-width first, whitespace fallback --
    same layout rules as split_pqr_line in your pipeline)."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            rec = None
            if len(line.rstrip("\n")) >= 69 and line[11] == " " and line[16] == " " and line[20] == " ":
                try:
                    rec = (line[21], int(line[22:26]), line[17:20].strip(), line[12:16].strip(),
                           float(line[30:38]), float(line[38:46]), float(line[46:54]), float(line[54:62]))
                except ValueError:
                    rec = None
            if rec is None:
                p = re.sub(r"(\d\.\d+)(-\d+\.\d+)", r"\1 \2", line).split()
                try:
                    if len(p) == 11:
                        rec = (p[4], int(p[5]), p[3], p[2], float(p[6]), float(p[7]), float(p[8]), float(p[9]))
                    elif len(p) == 10 and len(p[4]) == 5 and p[4][1:].isdigit():
                        rec = (p[4][0], int(p[4][1:]), p[3], p[2], float(p[5]), float(p[6]),
                               float(p[7]), float(p[8]))
                except ValueError:
                    rec = None
            if rec is not None:
                rows.append(rec)
    return pd.DataFrame(rows, columns=["chain", "resnum", "resname", "atom_name", "x", "y", "z", "q"])


# ----------------------------------------------------------------------
# Point-charge physics
# ----------------------------------------------------------------------

def bjerrum_A(T: float) -> float:
    """e^2/(4 pi eps0 kT) in A for dielectric 1."""
    return COULOMB_EV_ANGSTROM / (KB_EV * T)


def debye_kappa(I_molar: float, sdie: float, T: float) -> float:
    """Inverse Debye length (1/A) for a 1:1 salt in a solvent of dielectric sdie."""
    return float(np.sqrt(8.0 * np.pi * (bjerrum_A(T) / sdie) * I_molar * NA_PER_A3_PER_M))


def ball_mean_kernel(d: np.ndarray, R: float, kappa: float) -> np.ndarray:
    """Mean over a ball of radius R of exp(-kappa s)/s, for a unit source at distance d from the
    ball centre (s = distance from the source). Exact (shell theorem for the Yukawa kernel).
    Both the inside (d<R) and outside (d>=R) cases are finite -- no singularity."""
    d = np.maximum(d, 1e-6)
    if kappa < 1e-8:  # bare Coulomb
        return np.where(d >= R, 1.0 / d, (3.0 * R * R - d * d) / (2.0 * R ** 3))
    k = kappa
    g = 3.0 / R ** 3 * (R * np.cosh(k * R) / k ** 2 - np.sinh(k * R) / k ** 3)
    out = g * np.exp(-k * d) / d
    inside = d < R
    if np.any(inside):
        di = d[inside]
        A = di * np.cosh(k * di) / k ** 2 - np.sinh(k * di) / k ** 3
        B = (np.sinh(k * di) / (k * di)) * (np.exp(-k * di) * (1 + k * di)
                                              - np.exp(-k * R) * (1 + k * R)) / k ** 2
        out[inside] = 3.0 / R ** 3 * (A * np.exp(-k * di) / di + B)
    return out


def coulomb_sums(centers: np.ndarray, xyz: np.ndarray, q: np.ndarray, kappa: float,
                 radii: dict, point: bool, chunk: int = 128) -> dict:
    """For each centre: the (unit-prefactor) sum over atoms of q * kernel.
       point=True  -> key 'point': q * exp(-k r)/r, atoms closer than 0.3 A (the atom itself) excluded
       radii={'m4':4,...} -> sphere means (all atoms included)."""
    n = len(centers)
    out = {"point": np.full(n, np.nan)} if point else {}
    for name in radii:
        out[name] = np.full(n, np.nan)
    valid = ~np.isnan(centers).any(axis=1)
    idx_valid = np.flatnonzero(valid)
    for s in range(0, len(idx_valid), chunk):
        ii = idx_valid[s:s + chunk]
        diff = centers[ii][:, None, :] - xyz[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=2))
        if point:
            selfmask = D < 0.3                      # the atom itself: leave its own charge out
            Dp = np.where(selfmask, 1.0, D)
            K = np.where(selfmask, 0.0, np.exp(-kappa * Dp) / Dp)
            out["point"][ii] = K @ q
        for name, R in radii.items():
            res = np.zeros(len(ii))
            far = D >= R
            Kfar = np.where(far, np.exp(-kappa * np.where(far, D, 1.0)) / np.where(far, D, 1.0), 0.0)
            if kappa < 1e-8:
                g = 1.0
            else:
                g = 3.0 / R ** 3 * (R * np.cosh(kappa * R) / kappa ** 2 - np.sinh(kappa * R) / kappa ** 3)
            res += g * (Kfar @ q)
            r_idx, a_idx = np.nonzero(~far)
            if len(r_idx):
                vals = ball_mean_kernel(D[r_idx, a_idx], R, kappa) * q[a_idx]
                res += np.bincount(r_idx, weights=vals, minlength=len(ii))
            out[name][ii] = res
    return out


def family_table(fams: list) -> list:
    """[(name, prefactor_kT_per_e_per_A, kappa)] -- the prefactor multiplies sum q*kernel."""
    return fams


def residue_reference_points(atoms: pd.DataFrame, keys: pd.DataFrame):
    """Per labelled residue: C1' xyz, P xyz, and the sphere centre (C1', else C1*, else P, else mean),
    matching residue_representative_atom in your pipeline."""
    c1 = atoms[atoms["atom_name"].isin(["C1'", "C1*"])].drop_duplicates(["chain", "resnum", "resname"])
    pp = atoms[atoms["atom_name"] == "P"].drop_duplicates(["chain", "resnum", "resname"])
    mean = atoms.groupby(["chain", "resnum", "resname"])[["x", "y", "z"]].mean().reset_index()
    k = ["chain", "resnum", "resname"]
    m = keys[k].merge(c1[k + ["x", "y", "z"]].rename(columns={"x": "c1x", "y": "c1y", "z": "c1z"}), on=k, how="left")
    m = m.merge(pp[k + ["x", "y", "z"]].rename(columns={"x": "px", "y": "py", "z": "pz"}), on=k, how="left")
    m = m.merge(mean.rename(columns={"x": "mx", "y": "my", "z": "mz"}), on=k, how="left")
    c1xyz = m[["c1x", "c1y", "c1z"]].to_numpy(float)
    pxyz = m[["px", "py", "pz"]].to_numpy(float)
    mxyz = m[["mx", "my", "mz"]].to_numpy(float)
    center = c1xyz.copy()
    miss = np.isnan(center).any(axis=1)
    center[miss] = np.where(np.isnan(pxyz[miss]), mxyz[miss], pxyz[miss])
    return c1xyz, pxyz, center


def process_structure(job):
    labels_dir, pdb_id, include_flagged, fam_specs = job
    from rna_embedding_probe_v2 import load_structure_labels
    labels_dir = Path(labels_dir)
    summary = {"pdb_id": pdb_id, "n_atoms": 0, "total_charge": np.nan, "n_residues": 0}
    try:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None or labels.empty:
            return pd.DataFrame(), summary
        pqr = labels_dir / "work" / pdb_id.upper() / f"{pdb_id.lower()}.pqr"
        if not pqr.exists():
            return pd.DataFrame(), summary
        atoms = read_pqr(pqr)
        if atoms.empty:
            return pd.DataFrame(), summary
        xyz, q = atoms[["x", "y", "z"]].to_numpy(float), atoms["q"].to_numpy(float)
        summary.update(n_atoms=len(atoms), total_charge=float(q.sum()),
                       n_residues=atoms.groupby(["chain", "resnum", "resname"]).ngroups)

        k = ["chain", "resnum", "resname"]
        keys = labels[k + ["seq_index", "split"] + list(LABEL_COL.values())].drop_duplicates(k)
        keys = keys.reset_index(drop=True)
        c1xyz, pxyz, center = residue_reference_points(atoms, keys)

        out = keys[["chain", "seq_index", "split"]].copy()
        out.insert(0, "pdb_id", pdb_id)
        for qty, col in LABEL_COL.items():
            out[f"lab_{qty}"] = keys[col].to_numpy(float)
        kappa_cache: dict[float, dict] = {}
        for fam, pref, kappa in fam_specs:
            if kappa not in kappa_cache:
                kappa_cache[kappa] = {
                    "c1p": coulomb_sums(c1xyz, xyz, q, kappa, {}, point=True)["point"],
                    "p": coulomb_sums(pxyz, xyz, q, kappa, {}, point=True)["point"],
                    **coulomb_sums(center, xyz, q, kappa, RADII, point=False),
                }
            for qty in QTY:
                out[f"dh_{fam}_{qty}"] = pref * kappa_cache[kappa][qty]
        return out, summary
    except Exception as exc:
        logging.getLogger("coulomb_residual_test").warning("%s failed: %s", pdb_id, exc)
        return pd.DataFrame(), summary


def make_family_specs(I_list, pdie, sdie, T):
    lb = bjerrum_A(T)
    specs = [("vac", lb / pdie, 0.0)]
    for I in I_list:
        kap = debye_kappa(I, sdie, T)
        specs.append((f"dhS_I{I:g}", lb / sdie, kap))
        specs.append((f"dhP_I{I:g}", lb / pdie, kap))
    return specs


# ----------------------------------------------------------------------
# Label-side report
# ----------------------------------------------------------------------

def within_chain_center(M: np.ndarray, chain_ids: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(M)
    return (df - df.groupby(np.asarray(chain_ids)).transform("mean")).to_numpy()


def load_extra_controls(tbl: pd.DataFrame, paths: list):
    cols = []
    for p in paths:
        c = pd.read_csv(p, dtype={"pdb_id": str, "chain": str})
        c["seq_index"] = c["seq_index"].astype(int)
        c = c.drop_duplicates(KEYS)
        new = [x for x in c.columns if x not in KEYS]
        tbl = tbl.merge(c[KEYS + new], on=KEYS, how="left")
        cols += new
    return tbl, cols


def report(tbl: pd.DataFrame, fam_names: list, outdir: Path, control_paths: list):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    ctrl_cols = []
    if control_paths:
        tbl, ctrl_cols = load_extra_controls(tbl, control_paths)
        before = len(tbl)
        tbl = tbl.dropna(subset=ctrl_cols).reset_index(drop=True)
        log.info("Rows with all controls: %d of %d", len(tbl), before)
    tbl = tbl.dropna(subset=[f"lab_{q}" for q in QTY]).reset_index(drop=True)
    chain = (tbl["pdb_id"] + "::" + tbl["chain"]).to_numpy()
    split = tbl["split"].to_numpy()
    tr, te = split == "train", split == "test"
    log.info("Report on %d residues (%d train, %d test)", len(tbl), int(tr.sum()), int(te.sum()))

    rows = []
    for qty in QTY:
        y = tbl[f"lab_{qty}"].to_numpy(float)
        yc = within_chain_center(y[:, None], chain)[:, 0]
        dh_cols = [f"dh_{f}_{qty}" for f in fam_names]
        D = tbl[dh_cols].to_numpy(float)
        Dc = within_chain_center(D, chain)

        # single flavours: plain correlation
        for f, col in zip(fam_names, dh_cols):
            r_raw = float(np.corrcoef(tbl[col][te], y[te])[0, 1])
            r_cen = float(np.corrcoef(Dc[te][:, fam_names.index(f)], yc[te])[0, 1])
            rows.append({"target": qty, "model": f"corr_{f}", "raw_test_r": r_raw, "centered_test_r": r_cen})

        # calibrated: all flavours together, fit on train, scored on test
        raw_fit = LinearRegression().fit(D[tr], y[tr])
        cen_fit = LinearRegression().fit(Dc[tr], yc[tr])
        r2_raw = float(r2_score(y[te], raw_fit.predict(D[te])))
        r2_cen = float(r2_score(yc[te], cen_fit.predict(Dc[te])))
        row = {"target": qty, "model": "calibrated_coulomb", "raw_test_r2": r2_raw, "centered_test_r2": r2_cen}
        log.info("%-4s calibrated point-charge baseline explains: raw R2 = %.3f | within-chain-centered R2 = %.3f",
                 qty, r2_raw, r2_cen)
        rows.append(row)

        if ctrl_cols:  # does Coulomb add anything BEYOND the structure features?
            C = tbl[ctrl_cols].to_numpy(float)
            Cc = within_chain_center(C, chain)

            def gbm(X):
                m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                                  min_samples_leaf=200, early_stopping=False, random_state=0)
                return float(r2_score(yc[te], m.fit(X[tr], yc[tr]).predict(X[te])))

            a = gbm(np.column_stack([C, Cc]))
            b = gbm(np.column_stack([C, Cc, D, Dc]))
            c = gbm(np.column_stack([D, Dc]))
            rows.append({"target": qty, "model": "controls_only_gbm", "centered_test_r2": a})
            rows.append({"target": qty, "model": "coulomb_only_gbm", "centered_test_r2": c})
            rows.append({"target": qty, "model": "controls_plus_coulomb_gbm", "centered_test_r2": b})
            log.info("%-4s centered R2: controls %.3f | coulomb %.3f | controls+coulomb %.3f  "
                     "(Coulomb adds %.3f beyond structure)", qty, a, c, b, b - a)
    res = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(outdir / "coulomb_residual_report.csv", index=False)
    log.info("Wrote %s", outdir / "coulomb_residual_report.csv")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ionic-strengths", type=float, nargs="+", default=[0.15],
                    help="Molar. Several values let the calibrated fit choose the effective screening.")
    ap.add_argument("--pdie", type=float, default=1.0)
    ap.add_argument("--sdie", type=float, default=80.0)
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--structure-controls", type=Path, default=None)
    ap.add_argument("--dssr-controls", type=Path, default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--limit-structures", type=int, default=None)
    args = ap.parse_args()

    specs = make_family_specs(args.ionic_strengths, args.pdie, args.sdie, args.temp)
    fam_names = [s[0] for s in specs]
    log.info("Baseline families: %s", ", ".join(f"{n} (kappa={k:.4f}/A)" for n, _, k in specs))

    if not args.report_only:
        pdb_ids = sorted(pd.read_csv(args.labels_dir / "all_residue_potentials.csv",
                                     low_memory=False)["pdb_id"].unique())
        if args.limit_structures:
            pdb_ids = pdb_ids[: args.limit_structures]
        jobs = [(str(args.labels_dir), p, args.include_flagged, specs) for p in pdb_ids]
        frames, sums = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, (df, sm) in enumerate(pool.map(process_structure, jobs, chunksize=2), 1):
                sums.append(sm)
                if not df.empty:
                    frames.append(df)
                if n % 50 == 0:
                    log.info("  %d / %d structures done", n, len(jobs))
        if not frames:
            raise RuntimeError("no rows produced -- check .pqr files under labels_dir/work/<PDB_ID>/")
        tbl = pd.concat(frames, ignore_index=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_csv(args.out, index=False, compression="gzip" if str(args.out).endswith(".gz") else None)
        pd.DataFrame(sums).to_csv(args.out.with_name(args.out.name.split(".")[0] + "_structure_summary.csv"),
                                  index=False)
        log.info("Wrote %s (%d residues)", args.out, len(tbl))
    else:
        tbl = pd.read_csv(args.out, dtype={"pdb_id": str, "chain": str})

    paths = [p for p in (args.structure_controls, args.dssr_controls) if p is not None]
    report(tbl, fam_names, args.out.parent / "coulomb_report", paths)


if __name__ == "__main__":
    main()
