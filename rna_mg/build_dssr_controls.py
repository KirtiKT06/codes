"""
Build DSSR-based STRUCTURE CONTROLS for every labelled residue.

What this does, in order
------------------------
1. For each structure in all_residue_potentials.csv, finds the ORIGINAL deposited file
   (<pdb-dir>/<id>.pdb or .cif; lower- or upper-case names are both tried).
2. Runs DSSR on it:      x3dna-dssr -i=<file> --json -o=<json-dir>/<ID>.json
   The JSON is CACHED. A second run re-uses it (use --force to redo, --parse-only to
   skip DSSR entirely and just re-parse the JSON that is already there).
3. Parses the JSON into per-residue features (listed below).
4. Joins them onto your label rows. The join key is the DEPOSITED numbering:
        label table:  original_chain, original_resnum, insertion_code
        DSSR:         chain.resname<resnum>[^icode]    e.g.  A.G1,  A.2MG10,  A.PSU55
   (your pipeline renumbers residues internally; the label tables keep the original numbers
   exactly so that DSSR output can be joined -- see renumber_pdb_residues in the pipeline.)
5. Writes one CSV keyed  pdb_id, chain, seq_index  (chain = your INTERNAL chain letter, the
   same key build_structure_controls.py uses), so it plugs straight into
   centered_latent_manifold.py --dssr-controls.

Features (all 0/1 unless noted). They describe the REAL 3D structure, not a prediction:
  d_paired            in any base pair (any Leontis-Westhof class)
  d_n_pairs           number of pairs the residue is in (a base triple has 2)   [count]
  d_wc_pair           in a canonical cWW G-C / A-U pair
  d_wobble_pair       in a cWW G-U / U-G pair
  d_cww_other         in a cWW pair that is neither WC nor wobble
  d_noncww_pair       in a non-cWW pair (Hoogsteen, sugar edge, trans ...)  = tertiary-type pairing
  d_in_stem           in a DSSR stem (helix without bulges)
  d_stem_len          number of pairs in that stem                            [count, 0 if none]
  d_stem_pos_end      distance (in pairs) from the nearest stem end           [count, 0 if none]
  d_hairpin / d_bulge / d_iloop / d_junction
                      UNPAIRED residue inside that loop type
  d_ss_segment        in a single-stranded segment (includes 5'/3' tails)
  d_loop_closing      paired residue that closes a loop
  d_loop_size         number of residues in the loop it belongs to           [count, 0 if none]
  d_in_multiplet      in a base triple / quadruple / ...
  d_n_stacks          number of DSSR stacks containing the residue            [count]
  d_c3p_endo          sugar pucker C3'-endo (A-form-like)
  d_c2p_endo          sugar pucker C2'-endo
  d_syn               glycosidic bond syn
  d_splay             splay angle (degrees)                                   [float, 0 if missing]

PARSER STATUS: written against the JSON keys visible in your screenshot (pairs[], multiplets[],
helices[], stems[]) and against the DSSR 2.9 documentation for the rest (nts[], hairpins[],
bulges[], iloops[], junctions[], ssSegments[], stacks[]). Every lookup is defensive (.get),
so an unexpected key gives a column of zeros, not a crash. The run therefore prints, for
the whole dataset, how many residues got each feature switched on: an all-zero column means
the key name differs in your DSSR build -- send me one JSON and I will fix that line.

Usage
-----
    python build_dssr_controls.py \
        --labels-dir /data/rna_electrostats/phi_res \
        --pdb-dir    /data/rna_electrostats/phi_res/raw_pdb \
        --dssr-bin   ~/Downloads/dssr-basic-linuxMacWindows-v2.9.4/dssr-basic-linux-v2.9.4/x3dna-dssr \
        --json-dir   /data/rna_electrostats/phi_res/dssr_json \
        --out        /data/rna_electrostats/dssr_controls.csv.gz \
        --workers 8
        --limit-structures 10
    # first test:  add  --limit-structures 10
    # re-parse only (no DSSR):  add  --parse-only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_dssr_controls")

DSSR_COLS = [
    "d_paired", "d_n_pairs", "d_wc_pair", "d_wobble_pair", "d_cww_other", "d_noncww_pair",
    "d_in_stem", "d_stem_len", "d_stem_pos_end",
    "d_hairpin", "d_bulge", "d_iloop", "d_junction", "d_ss_segment", "d_loop_closing", "d_loop_size",
    "d_in_multiplet", "d_n_stacks",
    "d_c3p_endo", "d_c2p_endo", "d_syn", "d_splay",
]
OUT_COLS = ["pdb_id", "chain", "seq_index"] + DSSR_COLS

WC = {"G-C", "C-G", "A-U", "U-A", "A-T", "T-A"}
WOBBLE = {"G-U", "U-G", "G-T", "T-G"}

# 'A.G1', 'A.2MG10', 'A.PSU55', 'B.G10^A', '1:A.G1' (model prefix), chain may be several chars
NT_RE = re.compile(r"^(?:\d+:)?(?P<chain>[^.]+)\.(?P<res>.+?)(?P<num>-?\d+)(?:\^(?P<ic>\S))?$")


def parse_nt_id(s: str):
    """'A.2MG10' -> ('A', 10, '') ; 'A.G10^A' -> ('A', 10, 'A'). None if unparseable."""
    if not isinstance(s, str):
        return None
    m = NT_RE.match(s.strip())
    if not m:
        return None
    return m.group("chain"), int(m.group("num")), (m.group("ic") or "")


def split_nt_list(s) -> list:
    """DSSR lists nucleotides as one comma-separated string ('A.G1,A.C2,...')."""
    if isinstance(s, list):
        return [x for x in s if isinstance(x, str)]
    if isinstance(s, str):
        return [x.strip() for x in s.split(",") if x.strip()]
    return []


def _section(js: dict, *names) -> list:
    """First of the given key names that exists and is a list (DSSR spelling varies by version)."""
    for n in names:
        v = js.get(n)
        if isinstance(v, list):
            return v
    return []


def _f(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------
# JSON -> per-residue features
# ----------------------------------------------------------------------

def parse_dssr_json(js: dict) -> pd.DataFrame:
    """One row per DSSR nucleotide, keyed by (o_chain, o_resnum, o_icode) = DEPOSITED numbering."""
    feats: dict[tuple, dict] = {}

    def blank():
        return {c: 0.0 for c in DSSR_COLS}

    def get(key):
        if key is None:
            return None
        if key not in feats:
            feats[key] = blank()
        return feats[key]

    # --- the residue universe -------------------------------------------------
    nts = _section(js, "nts")
    for nt in nts:
        row = get(parse_nt_id(nt.get("nt_id", "")))
        if row is None:
            continue
        pucker = str(nt.get("puckering", "") or nt.get("sugar_class", ""))
        row["d_c3p_endo"] = float("C3'-endo" in pucker)
        row["d_c2p_endo"] = float("C2'-endo" in pucker)
        row["d_syn"] = float(str(nt.get("glyco_bond", "")).lower() == "syn")
        row["d_splay"] = _f(nt.get("splay_angle"))
    have_nts = bool(nts)

    # --- base pairs -----------------------------------------------------------
    pairs = _section(js, "pairs")
    pair_by_index = {}
    paired_keys = set()
    for p in pairs:
        k1, k2 = parse_nt_id(p.get("nt1", "")), parse_nt_id(p.get("nt2", ""))
        pair_by_index[p.get("index")] = (k1, k2)
        bp = str(p.get("bp", "")).upper()
        lw = str(p.get("LW", ""))
        for k in (k1, k2):
            r = get(k)
            if r is None:
                continue
            paired_keys.add(k)
            r["d_paired"] = 1.0
            r["d_n_pairs"] += 1.0
            if lw == "cWW":
                if bp in WC:
                    r["d_wc_pair"] = 1.0
                elif bp in WOBBLE:
                    r["d_wobble_pair"] = 1.0
                else:
                    r["d_cww_other"] = 1.0
            elif lw:
                r["d_noncww_pair"] = 1.0

    # --- stems (helix without bulges): length and distance from the stem end ----------
    for st in _section(js, "stems"):
        plist = st.get("pairs", [])
        keyed = []
        for e in plist:
            if isinstance(e, dict):
                keyed.append((parse_nt_id(e.get("nt1", "")), parse_nt_id(e.get("nt2", ""))))
            elif e in pair_by_index:
                keyed.append(pair_by_index[e])
        n = len(keyed) or int(_f(st.get("num_pairs"), 0))
        for i, (k1, k2) in enumerate(keyed):
            for k in (k1, k2):
                r = get(k)
                if r is None:
                    continue
                r["d_in_stem"] = 1.0
                r["d_stem_len"] = max(r["d_stem_len"], float(n))
                end = float(min(i, n - 1 - i))
                r["d_stem_pos_end"] = end if r["d_stem_pos_end"] == 0 else min(r["d_stem_pos_end"], end)

    # --- loops ----------------------------------------------------------------
    loop_sections = [("d_hairpin", ("hairpins",)), ("d_bulge", ("bulges",)),
                     ("d_iloop", ("iloops",)), ("d_junction", ("junctions",))]
    for col, names in loop_sections:
        for lp in _section(js, *names):
            ids = split_nt_list(lp.get("nts_long"))
            size = float(lp.get("num_nts") or len(ids))
            for s in ids:
                k = parse_nt_id(s)
                r = get(k)
                if r is None:
                    continue
                if k in paired_keys:
                    r["d_loop_closing"] = 1.0
                else:
                    r[col] = 1.0
                r["d_loop_size"] = max(r["d_loop_size"], size)
    for sg in _section(js, "ssSegments", "ss_segments"):
        for s in split_nt_list(sg.get("nts_long")):
            r = get(parse_nt_id(s))
            if r is not None and parse_nt_id(s) not in paired_keys:
                r["d_ss_segment"] = 1.0

    # --- multiplets and stacks ---------------------------------------------------
    for mp in _section(js, "multiplets"):
        for s in split_nt_list(mp.get("nts_long")):
            r = get(parse_nt_id(s))
            if r is not None:
                r["d_in_multiplet"] = 1.0
    for stk in _section(js, "stacks"):
        for s in split_nt_list(stk.get("nts_long")):
            r = get(parse_nt_id(s))
            if r is not None:
                r["d_n_stacks"] += 1.0

    rows = [{"o_chain": k[0], "o_resnum": k[1], "o_icode": k[2], **v} for k, v in feats.items()]
    df = pd.DataFrame(rows, columns=["o_chain", "o_resnum", "o_icode"] + DSSR_COLS)
    df.attrs["have_nts"] = have_nts
    return df


# ----------------------------------------------------------------------
# Running DSSR
# ----------------------------------------------------------------------

def find_structure_file(pdb_dir: Path, pdb_id: str):
    for name in (pdb_id.lower(), pdb_id.upper(), pdb_id):
        for ext in (".pdb", ".cif", ".ent", ".pdb1", ".mmcif"):
            p = pdb_dir / f"{name}{ext}"
            if p.exists():
                return p
    return None


def run_dssr(dssr_bin: str, struct: Path, json_out: Path, timeout: int, extra: list, force: bool):
    """Run DSSR (or reuse the cached JSON). Returns (ok, message)."""
    if json_out.exists() and json_out.stat().st_size > 0 and not force:
        return True, "cached"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    # DSSR drops helper files (dssr-*.pdb, dssr-torsions.txt ...) in the working directory,
    # so run it in a throw-away folder.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_json = Path(tmp) / "out.json"
        cmd = [dssr_bin, f"-i={struct.resolve()}", "--json", f"-o={tmp_json}", *extra]
        try:
            res = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
        except OSError as exc:
            return False, f"cannot execute DSSR: {exc}"
        if not tmp_json.exists() or tmp_json.stat().st_size == 0:
            return False, f"no JSON written (exit {res.returncode}): {(res.stderr or res.stdout)[-200:].strip()}"
        json_out.write_bytes(tmp_json.read_bytes())
    return True, "ran"


def _pick(df: pd.DataFrame, name: str):
    """Label tables hold the deposited numbering in `original_*`; the merge in
    load_structure_labels can leave a `_idxmap` twin."""
    for c in (name, f"{name}_idxmap"):
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan] * len(df), index=df.index)


def process_structure(job):
    (labels_dir, pdb_id, pdb_dir, dssr_bin, json_dir, timeout, extra, force, parse_only,
     include_flagged) = job
    from rna_embedding_probe_v2 import load_structure_labels  # imported here: worker processes
    report = {"pdb_id": pdb_id, "status": "", "n_label_rows": 0, "n_matched": 0, "n_resname_mismatch": 0}
    empty = pd.DataFrame(columns=OUT_COLS)
    try:
        labels = load_structure_labels(Path(labels_dir), pdb_id, include_flagged)
        if labels is None or labels.empty:
            report["status"] = "no labels"
            return empty, report
        report["n_label_rows"] = len(labels)

        json_path = Path(json_dir) / f"{pdb_id.upper()}.json"
        if not parse_only:
            struct = find_structure_file(Path(pdb_dir), pdb_id)
            if struct is None:
                report["status"] = "no structure file"
                return empty, report
            ok, msg = run_dssr(dssr_bin, struct, json_path, timeout, extra, force)
            if not ok:
                report["status"] = f"dssr failed: {msg}"
                return empty, report
        elif not json_path.exists():
            report["status"] = "no cached json"
            return empty, report

        with open(json_path) as fh:
            js = json.load(fh)
        feats = parse_dssr_json(js)
        if feats.empty:
            report["status"] = "json had no nucleotides"
            return empty, report

        key = pd.DataFrame({
            "chain": labels["chain"].to_numpy(),
            "seq_index": labels["seq_index"].astype(int).to_numpy(),
            "o_chain": _pick(labels, "original_chain").astype(str).to_numpy(),
            "o_resnum": pd.to_numeric(_pick(labels, "original_resnum"), errors="coerce").to_numpy(),
            "o_icode": _pick(labels, "insertion_code").fillna("").astype(str).str.strip().to_numpy(),
        })
        key = key.dropna(subset=["o_resnum"]).copy()
        key["o_resnum"] = key["o_resnum"].astype(int)
        key = key.drop_duplicates(["chain", "seq_index"])
        merged = key.merge(feats, on=["o_chain", "o_resnum", "o_icode"], how="inner")
        if not feats.attrs.get("have_nts", True):
            report["status"] = "ok (json has no 'nts': unannotated residues are missing, not zero)"
        else:
            report["status"] = "ok"
        report["n_matched"] = len(merged)
        merged["pdb_id"] = pdb_id
        return merged[OUT_COLS], report
    except Exception as exc:  # one bad structure must not kill a long run
        report["status"] = f"error: {exc}"
        return empty, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--pdb-dir", type=Path, required=True, help="Folder with the ORIGINAL .pdb/.cif files")
    ap.add_argument("--dssr-bin", default="x3dna-dssr", help="Path to the x3dna-dssr executable")
    ap.add_argument("--json-dir", type=Path, required=True, help="Where the per-structure DSSR JSON is cached")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600, help="Seconds allowed per DSSR run")
    ap.add_argument("--dssr-extra", nargs="*", default=[], help="Extra DSSR options, e.g. --more")
    ap.add_argument("--force", action="store_true", help="Re-run DSSR even when a cached JSON exists")
    ap.add_argument("--parse-only", action="store_true", help="Never call DSSR; parse cached JSON only")
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--limit-structures", type=int, default=None)
    args = ap.parse_args()

    pdb_ids = sorted(pd.read_csv(args.labels_dir / "all_residue_potentials.csv",
                                 low_memory=False)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]
    log.info("DSSR controls for %d structures, %d workers (parse_only=%s)",
             len(pdb_ids), args.workers, args.parse_only)

    jobs = [(str(args.labels_dir), p, str(args.pdb_dir), str(Path(args.dssr_bin).expanduser()),
             str(args.json_dir), args.timeout, args.dssr_extra, args.force, args.parse_only,
             args.include_flagged) for p in pdb_ids]
    frames, reports = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (df, rep) in enumerate(pool.map(process_structure, jobs, chunksize=2), 1):
            reports.append(rep)
            if not df.empty:
                frames.append(df)
            if n % 50 == 0:
                log.info("  %d / %d structures done", n, len(jobs))

    rep = pd.DataFrame(reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rep_path = args.out.with_name(args.out.name.split(".")[0] + "_join_report.csv")
    rep.to_csv(rep_path, index=False)
    log.info("Status counts:\n%s", rep["status"].str.split(":").str[0].value_counts().to_string())
    n_lab, n_match = int(rep["n_label_rows"].sum()), int(rep["n_matched"].sum())
    log.info("Matched %d of %d label rows to DSSR nucleotides (%.1f%%). Per-structure report: %s",
             n_match, n_lab, 100.0 * n_match / max(n_lab, 1), rep_path)
    if not frames:
        raise RuntimeError("No DSSR rows produced -- see the join report for per-structure reasons")
    out = pd.concat(frames, ignore_index=True)

    # Sanity table: how many residues switch each feature on. An all-zero row = key name mismatch.
    on = {c: float((out[c] > 0).mean()) for c in DSSR_COLS}
    log.info("Fraction of residues with feature > 0:\n%s",
             pd.Series(on).round(3).to_string())
    dead = [c for c, v in on.items() if v == 0.0]
    if dead:
        log.warning("These features are ALL ZERO and probably need a key-name fix: %s", dead)
    if n_match / max(n_lab, 1) < 0.8:
        log.warning("Under 80%% of label rows matched. Likely causes: chain naming for .cif files, "
                    "or modified residues DSSR does not list. Look at the join report.")

    out.to_csv(args.out, index=False, compression="gzip" if str(args.out).endswith(".gz") else None)
    log.info("Wrote %s (%d rows)", args.out, len(out))


if __name__ == "__main__":
    main()
