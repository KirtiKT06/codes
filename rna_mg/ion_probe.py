#!/usr/bin/env python
"""
Do frozen RNA language-model embeddings predict which phosphates sit next to a Mg2+ ion, FROM SEQUENCE ALONE?

TWO STAGES (run from ~/projects/codes/rna_mg so the audit modules can be imported)

  1. embed : extract embeddings once per model and cache the chosen layer(s) for EVERY labelled residue, with the same
             builder logic and the same residue set as the audit (rna_embedding_probe_v2.load_structure_labels + FASTA).
        python ion_probe.py embed --model ernierna                     # layer 11 (the audit's layer)
        python ion_probe.py embed --model ernierna --random-init       # random-weights control, seed 0
        python ion_probe.py embed --model rnafm
        python ion_probe.py embed --model rnafm --random-init

  2. probe : leave-family-out probes (the audit's own 5 family folds, via latent_controls.build_family_split) with
        python ion_probe.py probe \
            --caches ernierna=/data/rna_electrostats/ion_probe/emb_ernierna_L11.npz \
                     ernierna_rand=/data/rna_electrostats/ion_probe/emb_ernierna_rand0_L11.npz \
                     rnafm=/data/rna_electrostats/ion_probe/emb_rnafm_L11.npz \
                     rnafm_rand=/data/rna_electrostats/ion_probe/emb_rnafm_rand0_L11.npz

WHAT IS SCORED (positives = residues whose P is within 5 A of a Mg2+; table ion_labels_5A.csv.gz from ion_site_pilot.py)
  <name>__ion     ridge on the embedding (chain-centred, standardised) -> the ion label, one probe per family fold
  <name>__phos    ridge on the embedding -> log phos_screened (the dense electrostatic-like label), used as a site score
  seq_ctx         gradient boosting on letters (+-2), position and chain length          (sequence only)
  seq_vienna      seq_ctx + ViennaRNA pairing (structure_controls)                        (sequence only, 2D structure)
  oracle_struct   gradient boosting on TRUE-structure features (packing, DSSR, phosphate crowding, PB labels): the ceiling
  phos_screened   raw phosphate-only screened potential (no training)                     (true structure, free)
METRICS (chain by chain, then averaged within structure, bootstrapped over structures; as in ion_site_pilot.py)
  AUROC, R-precision enrichment, ion hit@10%: the share of ions with at least one neighbouring phosphate among the top
  10% scored residues of the chain (the residue-level analogue of the Top-N success rate used by Metal3DRNA/RMSIF).

DECISIVE COMPARISONS (printed at the end): <name>__ion minus seq_vienna, minus the random-weights twin, and minus the
oracle. If the embedding does not beat seq_vienna, the embeddings add nothing for this task beyond sequence-derived
pairing; if it beats its random twin, pretraining matters; the oracle shows how much room is left.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("ion_probe")

DATA = Path("/data/rna_electrostats")
AUDIT_LAYER = {"ernierna": 11, "rnafm": 11, "rnabert": 3, "splicebert": 2}
LETTERS = ["A", "C", "G", "U"]
STRUCT_COLS = ["is_paired_mfe", "prob_paired", "n_neighbors_5A", "n_neighbors_8A", "n_tertiary_neighbors_5A",
               "dist_nearest_P_other", "burial_proxy"]


def add_code_dir(code_dir: str):
    p = str(Path(code_dir).expanduser())
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------------------------------------------------------------------------------------------
# stage 1: embedding cache
# ----------------------------------------------------------------------------------------------------------------
def cmd_embed(a):
    add_code_dir(a.code_dir)
    import torch
    import rna_embedding_probe_v2 as rep
    if a.random_init:
        from latent_controls import install_random_init_loader
        install_random_init_loader(a.seed)
    layers = a.layers or [AUDIT_LAYER[a.model]]
    device = a.device if (torch.cuda.is_available() or a.device == "cpu") else "cpu"
    pdb_ids = sorted(pd.read_csv(a.labels_dir / "all_residue_potentials.csv", usecols=["pdb_id"],
                                 low_memory=False)["pdb_id"].unique())
    if a.limit_structures:
        pdb_ids = pdb_ids[: a.limit_structures]
    tokenizer, model, n_layers, model_max_len = rep.load_model(a.model, device)
    eff = max(1, min(a.max_len, model_max_len - 2))
    log.info("%s%s: layers %s of %d, effective max length %d, %d structures", a.model,
             f" RANDOM seed {a.seed}" if a.random_init else "", layers, n_layers, eff, len(pdb_ids))
    Xs = {L: [] for L in layers}
    meta = {k: [] for k in ("pdb_id", "chain", "resnum", "seq_index", "split", "base")}
    for n, pdb_id in enumerate(pdb_ids, 1):
        labels = rep.load_structure_labels(a.labels_dir, pdb_id, a.include_flagged)
        if labels is None:
            continue
        seqs = rep.load_fasta_by_chain(a.labels_dir, pdb_id)
        for chain, rows in labels.groupby("chain"):
            seq = seqs.get(chain)
            if not seq:
                continue
            emb, _ = rep.extract_layer_embeddings(tokenizer, model, seq, device, max_len=eff)
            if emb is None:
                continue
            valid = rows[rows["seq_index"] < emb.shape[1]]
            if valid.empty:
                continue
            idx = valid["seq_index"].to_numpy()
            for L in layers:
                Xs[L].append(np.asarray(emb[L, idx, :], dtype=np.float32).astype(np.float16))
            meta["pdb_id"].append(np.full(len(valid), pdb_id))
            meta["chain"].append(np.full(len(valid), str(chain)))
            meta["resnum"].append(valid["resnum"].to_numpy().astype(np.int32))
            meta["seq_index"].append(valid["seq_index"].to_numpy().astype(np.int32))
            meta["split"].append(valid["split"].to_numpy().astype(str) if "split" in valid.columns else np.full(len(valid), "train"))
            meta["base"].append(valid["base"].to_numpy().astype(str))
        if n % 100 == 0:
            log.info("%d / %d structures", n, len(pdb_ids))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    m = {k: np.concatenate(v) for k, v in meta.items()}
    tag = f"{a.model}{'_rand' + str(a.seed) if a.random_init else ''}"
    for L in layers:
        X = np.concatenate(Xs[L], axis=0)
        if not np.isfinite(X.astype(np.float32)).all():
            raise RuntimeError(f"non-finite values in layer {L}")
        out = a.out_dir / f"emb_{tag}_L{L}.npz"
        np.savez(out, X=X, **m)
        log.info("wrote %s: %d residues, dim %d, %.2f GB", out, X.shape[0], X.shape[1], X.nbytes / 1e9)


# ----------------------------------------------------------------------------------------------------------------
# stage 2: probes
# ----------------------------------------------------------------------------------------------------------------
def load_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    meta = pd.DataFrame({k: z[k] for k in ("pdb_id", "chain", "resnum", "seq_index", "split", "base")})
    meta["pdb_id"] = meta["pdb_id"].astype(str)
    meta["chain"] = meta["chain"].astype(str)
    return meta, z["X"]


def within_chain_center(X: np.ndarray, cid: np.ndarray, n_chains: int) -> np.ndarray:
    cnt = np.bincount(cid, minlength=n_chains).astype(np.float64)
    out = np.empty_like(X, dtype=np.float32)
    for j0 in range(0, X.shape[1], 128):                         # column blocks keep memory low
        blk = X[:, j0:j0 + 128].astype(np.float32)
        s = np.zeros((n_chains, blk.shape[1]), np.float64)
        np.add.at(s, cid, blk)
        out[:, j0:j0 + 128] = blk - (s / np.maximum(cnt, 1)[:, None]).astype(np.float32)[cid]
    return out


def chain_center_vec(v: np.ndarray, cid: np.ndarray, n_chains: int) -> np.ndarray:
    ok = np.isfinite(v)
    s = np.bincount(cid[ok], weights=v[ok], minlength=n_chains)
    c = np.bincount(cid[ok], minlength=n_chains)
    return v - (s / np.maximum(c, 1))[cid]


def auroc(score, pos):
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(score, method="average")
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def boot_ci(per_struct: pd.Series, n_boot: int, seed: int):
    v = per_struct.dropna().to_numpy(float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(v)


def mean_chain_auc(df: pd.DataFrame, col: str, target: str = "pos_MG") -> float:
    v = []
    for _, g in df.groupby(["pdb_id", "chain"], sort=False):
        pos = g[target].to_numpy(bool)
        s = g[col].to_numpy(float)
        if len(g) >= 8 and pos.sum() >= 1 and (~pos).sum() >= 3 and np.isfinite(s).all():
            v.append(auroc(s, pos))
    return float(np.mean(v)) if v else np.nan


def ridge_scores(Xc, y, cid_unused, tr, va, te, alphas_rel, D, score_name, target):
    """Standardise on the training rows, choose alpha on the validation rows by mean within-chain AUROC, refit on
    train+val, predict test. Xc is chain-centred; y is chain-centred and finite on tr/va."""
    mu = Xc[tr].mean(axis=0)
    sd = Xc[tr].std(axis=0) + 1e-6

    def design(idx):
        return (Xc[idx] - mu) / sd

    Xtr, ytr = design(tr), y[tr]
    G = (Xtr.T @ Xtr).astype(np.float64)
    b = (Xtr.T @ ytr).astype(np.float64)
    lam, V = np.linalg.eigh(G)
    Vb = V.T @ b
    best, best_a = -np.inf, alphas_rel[0]
    if len(va):
        Xva = design(va)
        tmp = D.loc[va, ["pdb_id", "chain", target]].copy()
        for ar in alphas_rel:
            w = V @ (Vb / (lam + ar * len(tr)))
            tmp["s"] = Xva @ w.astype(np.float32)
            auc = mean_chain_auc(tmp, "s", target if target in ("pos_MG",) else "pos_MG")
            if np.isfinite(auc) and auc > best:
                best, best_a = auc, ar
    trva = np.concatenate([tr, va])
    Xa = design(trva)
    Ga = (Xa.T @ Xa).astype(np.float64)
    ba = (Xa.T @ y[trva]).astype(np.float64)
    w = np.linalg.solve(Ga + best_a * len(trva) * np.eye(Ga.shape[0]), ba)
    return (design(te) @ w.astype(np.float32)), best_a


# ----------------------------------------------------------------------------------------------------------------
def build_features(M: pd.DataFrame, struct: pd.DataFrame, dssr: pd.DataFrame | None):
    """Sequence-only and oracle feature matrices (DataFrames aligned to M)."""
    key = ["pdb_id", "chain", "seq_index"]
    F = M[key + ["base"]].copy()
    g = M.groupby(["pdb_id", "chain"])["seq_index"]
    mx = g.transform("max").to_numpy()
    cl = g.transform("size").to_numpy()
    F["pos_rel"] = M["seq_index"].to_numpy() / np.maximum(mx, 1)
    F["log_len"] = np.log(cl)
    F["d5"] = np.minimum(M["seq_index"].to_numpy(), 60) / 60.0
    F["d3"] = np.minimum(mx - M["seq_index"].to_numpy(), 60) / 60.0
    base_map = M.set_index(key)["base"]
    for d in (-2, -1, 0, 1, 2):
        idx = pd.MultiIndex.from_arrays([M["pdb_id"], M["chain"], M["seq_index"] + d])
        b = base_map.reindex(idx).to_numpy()
        for L in LETTERS:
            F[f"b{d:+d}_{L}"] = (b == L).astype(np.float32)
    seq_cols = [c for c in F.columns if c.startswith("b") and "_" in c] + ["pos_rel", "log_len", "d5", "d3"]
    S = M[key].merge(struct, on=key, how="left")
    vien = pd.DataFrame(index=F.index)
    for d in (-2, -1, 0, 1, 2):
        pp = S.set_index(key)["prob_paired"]
        idx = pd.MultiIndex.from_arrays([M["pdb_id"], M["chain"], M["seq_index"] + d])
        vien[f"pp{d:+d}"] = pp.reindex(idx).to_numpy()
    vien["paired_mfe"] = S["is_paired_mfe"].to_numpy()
    seq_ctx = F[seq_cols].reset_index(drop=True)
    seq_vienna = pd.concat([seq_ctx, vien.reset_index(drop=True)], axis=1)
    ora = pd.concat([S[STRUCT_COLS].reset_index(drop=True), seq_ctx[[c for c in seq_ctx.columns if c.startswith("b+0")]]], axis=1)
    if dssr is not None:
        D_ = M[key].merge(dssr, on=key, how="left")
        ora = pd.concat([ora, D_.drop(columns=key).reset_index(drop=True)], axis=1)
    return seq_ctx, seq_vienna, ora


def ion_neighbors(D: pd.DataFrame, labels_dir: Path, cutoff: float, min_occ: float = 0.5):
    """{pdb_id: [ [(chain, resnum), ...] per Mg ion ]}: P-neighbours of every kept Mg2+ (same definition as the label)."""
    out = {}
    for pid, g in D.groupby("pdb_id"):
        f = labels_dir / "work" / pid / "ions.csv"
        if not f.exists() or f.stat().st_size < 5:
            continue
        try:
            ions = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if "resname" not in ions:
            continue
        ions = ions[ions["resname"].astype(str).str.upper() == "MG"]
        if "occupancy" in ions:
            ions = ions[~(ions["occupancy"] < min_occ)]
        if "low_resolution_flag" in ions:
            ions = ions[~ions["low_resolution_flag"].astype(str).str.lower().isin(["true", "1"])]
        if ions.empty:
            continue
        pts = ions[["x", "y", "z"]].to_numpy(float)
        keep = []
        for p in pts:
            if not any(np.linalg.norm(p - q) < 0.5 for q in keep):
                keep.append(p)
        tree = cKDTree(g[["x", "y", "z"]].to_numpy(float))
        ch, rn = g["chain"].to_numpy(), g["resnum"].to_numpy()
        out[pid] = [[(ch[i], int(rn[i])) for i in tree.query_ball_point(p, cutoff)] for p in keep]
    return out


def evaluate(D: pd.DataFrame, scores: list[str], nb: dict, min_p: int, min_neg: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (pid, ch), g in D.groupby(["pdb_id", "chain"], sort=False):
        pos = g["pos_MG"].to_numpy(bool)
        if len(g) < min_p or pos.sum() < 1 or (~pos).sum() < min_neg:
            continue
        rn_index = {int(r): i for i, r in enumerate(g["resnum"].to_numpy())}
        ions_here = []
        for nbrs in nb.get(pid, []):
            idx = [rn_index[r] for c, r in nbrs if c == ch and r in rn_index]
            if idx:
                ions_here.append(idx)
        for name in scores:
            s = g[name].to_numpy(float)
            ok = np.isfinite(s)
            if ok.sum() < min_p or pos[ok].sum() < 1 or (~pos[ok]).sum() < min_neg:
                continue
            if not ok.all():
                continue
            m = int(pos.sum())
            order = np.argsort(-(s + 1e-9 * rng.standard_normal(len(s))), kind="stable")
            rprec = float(pos[order[:m]].mean())
            k10 = max(1, int(np.ceil(0.10 * len(s))))
            top = set(order[:k10].tolist())
            hit = float(np.mean([any(i in top for i in idx) for idx in ions_here])) if ions_here else np.nan
            rows.append({"pdb_id": pid, "chain": ch, "score": name, "n_res": len(s), "n_pos": m, "auroc": auroc(s, pos),
                         "enrichment": rprec / pos.mean(), "hit10": hit,
                         "length_bin": "<=100" if len(g) <= 100 else ("101-300" if len(g) <= 300 else ">300"),
                         "method": g["method"].iloc[0]})
    return pd.DataFrame(rows)


def cmd_probe(a):
    add_code_dir(a.code_dir)
    from latent_controls import build_family_split, load_family_table
    from sklearn.ensemble import HistGradientBoostingClassifier

    caches = dict(kv.split("=", 1) for kv in a.caches)
    names = list(caches)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    metas, Xraw = {}, {}
    for nm, p in caches.items():
        metas[nm], Xraw[nm] = load_cache(Path(p))
        log.info("%s: %d residues, dim %d", nm, len(metas[nm]), Xraw[nm].shape[1])
    # common rows across caches
    key = ["pdb_id", "chain", "resnum"]
    common = metas[names[0]][key].copy()
    for nm in names[1:]:
        common = common.merge(metas[nm][key], on=key, how="inner")
    for nm in names:
        m = metas[nm].reset_index(drop=True)
        m["row"] = np.arange(len(m))
        sub = common.merge(m, on=key, how="left")
        if len(sub) != len(metas[nm]):
            log.warning("%s: restricted to %d rows shared by all caches (of %d)", nm, len(sub), len(m))
        metas[nm] = sub
    M = metas[names[0]].reset_index(drop=True)
    for nm in names[1:]:
        assert (metas[nm][key].to_numpy() == M[key].to_numpy()).all()
    Xs = {nm: Xraw[nm][metas[nm]["row"].to_numpy()] for nm in names}
    del Xraw
    n = len(M)
    chains, cid = np.unique(M["pdb_id"] + "::" + M["chain"], return_inverse=True)
    n_chains = len(chains)

    lab = pd.read_csv(a.labels_table, low_memory=False, dtype={"pdb_id": str, "chain": str})
    keep_lab = ["pdb_id", "chain", "resnum", "pos_MG", "phos_screened", "phos_n8", "phos_n12", "pb_c1p", "pb_m4", "pb_m8",
                "pb_m12", "x", "y", "z", "method", "resolution_A"]
    D = M.merge(lab[keep_lab], on=key, how="left")
    assert len(D) == n
    D["has_label"] = D["pos_MG"].notna()
    D["pos_MG"] = D["pos_MG"].fillna(False).astype(bool)
    log.info("rows with ion labels: %d of %d; Mg positives %d", int(D["has_label"].sum()), n, int(D["pos_MG"].sum()))

    struct = pd.read_csv(a.structure_controls, usecols=["pdb_id", "chain", "seq_index"] + STRUCT_COLS, dtype={"pdb_id": str, "chain": str})
    dssr = pd.read_csv(a.dssr_controls, dtype={"pdb_id": str, "chain": str}) if a.dssr_controls and Path(a.dssr_controls).exists() else None
    seq_ctx, seq_vienna, oracle = build_features(M, struct, dssr)
    oracle = pd.concat([oracle, D[["phos_screened", "phos_n8", "phos_n12", "pb_c1p", "pb_m4", "pb_m8", "pb_m12"]].reset_index(drop=True)], axis=1)
    fam = load_family_table(Path(a.family_table))
    pdb_g, chain_g, splits = M["pdb_id"].to_numpy(), (M["pdb_id"] + "::" + M["chain"]).to_numpy(), M["split"].to_numpy()

    # centred embeddings (once per cache) and centred targets
    Xc = {nm: within_chain_center(Xs[nm], cid, n_chains) for nm in names}
    y_ion = chain_center_vec(np.where(D["has_label"], D["pos_MG"].astype(float), np.nan), cid, n_chains)
    y_phos = chain_center_vec(np.where(D["has_label"], np.log(D["phos_screened"].to_numpy(float) + 0.05), np.nan), cid, n_chains)

    out_cols = {}
    for nm in names:
        out_cols[f"{nm}__ion"] = np.full(n, np.nan)
        out_cols[f"{nm}__phos"] = np.full(n, np.nan)
    for c in ("seq_ctx", "seq_vienna", "oracle_struct"):
        out_cols[c] = np.full(n, np.nan)
    if a.extras:
        for c in ("seq_pos", "seq_base", "seq_vienna_ridge"):
            out_cols[c] = np.full(n, np.nan)
        for nm in names:
            out_cols[f"{nm}__gbm"] = np.full(n, np.nan)
            out_cols[f"{nm}__gbm_seq"] = np.full(n, np.nan)
        pos_cols = ["pos_rel", "log_len", "d5", "d3"]
        seq_pos = seq_ctx[pos_cols]
        seq_base = seq_ctx[[c for c in seq_ctx.columns if c not in pos_cols]]
        Xseq_c = within_chain_center(seq_vienna.fillna(0.0).to_numpy(np.float32), cid, n_chains)
    in_test = np.zeros(n, bool)
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for k in range(a.folds):
        new_splits, keep, _, _ = build_family_split(pdb_g, chain_g, splits, fam, a.folds, k, "drop")
        new_splits, keep = np.asarray(new_splits), np.asarray(keep)
        has = D["has_label"].to_numpy()
        te = np.where((new_splits == "test") & keep & has)[0]
        va = np.where((new_splits == "val") & keep & has)[0]
        tr = np.where((new_splits == "train") & keep & has)[0]
        in_test[te] = True
        log.info("fold %d: train %d, val %d, test %d residues (%d test chains)", k, len(tr), len(va), len(te),
                 len(np.unique(cid[te])))
        for nm in names:
            for tag, yv, tgt in (("ion", y_ion, "pos_MG"), ("phos", y_phos, "pos_MG")):
                s, al = ridge_scores(Xc[nm], yv, None, tr, va, te, alphas, D, f"{nm}__{tag}", tgt)
                out_cols[f"{nm}__{tag}"][te] = s
                log.info("  %s__%s alpha %.0e", nm, tag, al)
        ytr = D["pos_MG"].to_numpy()[np.concatenate([tr, va])].astype(int)
        trva = np.concatenate([tr, va])
        for cname, Fm in (("seq_ctx", seq_ctx), ("seq_vienna", seq_vienna), ("oracle_struct", oracle)):
            clf = HistGradientBoostingClassifier(max_iter=a.gbm_iter, learning_rate=0.1, max_leaf_nodes=31,
                                                 early_stopping=False, random_state=a.seed)
            clf.fit(Fm.iloc[trva].to_numpy(np.float32), ytr)
            out_cols[cname][te] = clf.predict_proba(Fm.iloc[te].to_numpy(np.float32))[:, 1]
        if a.extras:
            from sklearn.decomposition import PCA
            for cname, Fm in (("seq_pos", seq_pos), ("seq_base", seq_base)):
                clf = HistGradientBoostingClassifier(max_iter=a.gbm_iter, learning_rate=0.1, max_leaf_nodes=31,
                                                     early_stopping=False, random_state=a.seed)
                clf.fit(Fm.iloc[trva].to_numpy(np.float32), ytr)
                out_cols[cname][te] = clf.predict_proba(Fm.iloc[te].to_numpy(np.float32))[:, 1]
            s_, al_ = ridge_scores(Xseq_c, y_ion, None, tr, va, te, alphas, D, "seq_vienna_ridge", "pos_MG")
            out_cols["seq_vienna_ridge"][te] = s_
            rng = np.random.default_rng(a.seed + k)
            sub = rng.choice(tr, size=min(len(tr), 100_000), replace=False)
            for nm in names:
                mu_ = Xc[nm][tr].mean(axis=0)
                sd_ = Xc[nm][tr].std(axis=0) + 1e-6
                pca = PCA(n_components=a.pca_dim, svd_solver="randomized", random_state=a.seed).fit((Xc[nm][sub] - mu_) / sd_)
                Ptr = pca.transform((Xc[nm][trva] - mu_) / sd_).astype(np.float32)
                Pte = pca.transform((Xc[nm][te] - mu_) / sd_).astype(np.float32)
                for tag, Atr, Ate in (("gbm", Ptr, Pte),
                                      ("gbm_seq", np.hstack([Ptr, seq_vienna.iloc[trva].to_numpy(np.float32)]),
                                       np.hstack([Pte, seq_vienna.iloc[te].to_numpy(np.float32)]))):
                    clf = HistGradientBoostingClassifier(max_iter=a.gbm_iter, learning_rate=0.1, max_leaf_nodes=31,
                                                         early_stopping=False, random_state=a.seed)
                    clf.fit(Atr, ytr)
                    out_cols[f"{nm}__{tag}"][te] = clf.predict_proba(Ate)[:, 1]
                log.info("  %s: PCA-%d + boosting heads done", nm, a.pca_dim)
    for c, v in out_cols.items():
        D[c] = v
    D["phos_screened_raw"] = D["phos_screened"]
    D["pb_m12_neg"] = -D["pb_m12"]
    score_cols = [c for c in out_cols] + ["phos_screened_raw", "pb_m12_neg"]
    E = D[in_test & D["has_label"].to_numpy()].copy()
    log.info("evaluating %d held-out residues in %d chains", len(E), E.groupby(["pdb_id", "chain"]).ngroups)
    nb = ion_neighbors(E, a.labels_dir, a.cutoff)
    Mx = evaluate(E, score_cols, nb, a.min_p, a.min_neg, a.seed)
    Mx.to_csv(a.out_dir / "ion_probe_chain_metrics.csv", index=False)
    E[["pdb_id", "chain", "resnum"] + score_cols].to_csv(a.out_dir / "ion_probe_predictions.csv.gz", index=False)
    text = summarize(Mx, names, a.n_boot, a.seed)
    from scipy.stats import spearmanr
    diag = ["\nSANITY: within-chain Spearman between the embedding's predicted phos label and the true log phos_screened (held-out chains)"]
    E["_lp"] = np.log(E["phos_screened"].to_numpy(float) + 0.05)
    for nm in names:
        r = []
        for _, g in E.groupby(["pdb_id", "chain"]):
            if len(g) >= 8 and g[f"{nm}__phos"].notna().all():
                r.append(spearmanr(g[f"{nm}__phos"], g["_lp"])[0])
        diag.append(f"  {nm:<14s} mean {np.nanmean(r):.3f}  median {np.nanmedian(r):.3f}  (chains {len(r)})")
    text = text + "\n" + "\n".join(diag)
    (a.out_dir / "ion_probe_summary.txt").write_text(text + "\n")
    print(text)
    try:
        make_figure(Mx, names, a.out_dir, a.n_boot, a.seed)
    except Exception as exc:  # noqa: BLE001
        log.warning("figure failed: %s", exc)
    print(f"\nfiles in {a.out_dir}: ion_probe_summary.txt, ion_probe_chain_metrics.csv, ion_probe_predictions.csv.gz, fig_ion_probe.png")


def summarize(Mx: pd.DataFrame, names: list[str], n_boot: int, seed: int) -> str:
    L = []
    say = L.append
    say("ION-SITE PROBE: Mg2+ positives, leave-family-out folds (the audit's), held-out chains only")
    say(f"chains evaluated: {Mx[['pdb_id','chain']].drop_duplicates().shape[0]}   structures: {Mx['pdb_id'].nunique()}")
    say(f"\n{'score':<26s}{'AUROC (structure mean) [95% CI]':<36s}{'enrichment':<14s}{'ion hit@10%':<12s}")
    order = ([f"{n}__{t}" for n in names for t in ("ion", "phos", "gbm", "gbm_seq")]
             + ["seq_pos", "seq_base", "seq_ctx", "seq_vienna", "seq_vienna_ridge", "oracle_struct", "phos_screened_raw", "pb_m12_neg"])
    piv = Mx.pivot_table(index=["pdb_id", "chain"], columns="score", values="auroc")
    for s in order:
        if s not in Mx["score"].values:
            continue
        sub = Mx[Mx["score"] == s]
        am, al, ah, nn = boot_ci(sub.groupby("pdb_id")["auroc"].mean(), n_boot, seed)
        em = sub.groupby("pdb_id")["enrichment"].mean().mean()
        hm = sub.groupby("pdb_id")["hit10"].mean().mean()
        say(f"{s:<26s}{am:.3f} [{al:.3f}, {ah:.3f}] (n={nn:<4d}) {em:5.2f}         {hm:5.2f}")

    def paired(a_, b_):
        if a_ not in piv.columns or b_ not in piv.columns:
            return None
        d = (piv[a_] - piv[b_]).dropna().groupby(level=0).mean()
        dm, dl, dh, nn = boot_ci(d, n_boot, seed)
        verdict = f"{a_} better" if dl > 0 else (f"{b_} better" if dh < 0 else "no clear difference")
        return f"  {a_:<22s} - {b_:<22s} {dm:+.3f} [{dl:+.3f}, {dh:+.3f}]  (n={nn})  {verdict}"
    say("\nDECISIVE PAIRED COMPARISONS (AUROC difference over structures, 95% bootstrap CI)")
    for n_ in names:
        if n_.endswith("_rand"):
            continue
        for a_, b_ in ((f"{n_}__gbm_seq", "seq_vienna"), (f"{n_}__gbm_seq", f"{n_}_rand__gbm_seq"), (f"{n_}__gbm", f"{n_}_rand__gbm"),
                       (f"{n_}__gbm", "seq_vienna"), (f"{n_}__ion", "seq_vienna_ridge"),
                       (f"{n_}__ion", "seq_vienna"), (f"{n_}__ion", "seq_ctx"), (f"{n_}__ion", f"{n_}_rand__ion"),
                       (f"{n_}__ion", f"{n_}__phos"), (f"{n_}__phos", "seq_vienna"), (f"{n_}__ion", "oracle_struct")):
            r = paired(a_, b_)
            if r:
                say(r)
    for a_, b_ in (("seq_vienna", "seq_ctx"), ("seq_ctx", "seq_pos"), ("seq_ctx", "seq_base"), ("oracle_struct", "phos_screened_raw")):
        r = paired(a_, b_)
        if r:
            say(r)
    say("\nSUBGROUPS (structure-mean AUROC)")
    for col in ("length_bin", "method"):
        for lev, g in Mx.groupby(col):
            if g["pdb_id"].nunique() < 5:
                continue
            cells = []
            for s in [f"{n}__ion" for n in names if not n.endswith("_rand")] + ["seq_vienna", "oracle_struct"]:
                v = g[g["score"] == s].groupby("pdb_id")["auroc"].mean().mean()
                cells.append(f"{s} {v:.3f}")
            say(f"  {col} = {str(lev):<20s} (n structures {g['pdb_id'].nunique():4d})  " + "  ".join(cells))
    return "\n".join(L)


def make_figure(Mx, names, out, n_boot, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    order = [f"{n}__{t}" for n in names for t in ("ion", "phos")] + ["seq_ctx", "seq_vienna", "oracle_struct", "phos_screened_raw"]
    order = [s for s in order if s in Mx["score"].values]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, s in enumerate(order):
        m, lo, hi, _ = boot_ci(Mx[Mx["score"] == s].groupby("pdb_id")["auroc"].mean(), n_boot, seed)
        ax[0].errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=4, capsize=2,
                       color="C3" if "rand" in s else ("C0" if "__" in s else "0.3"))
    ax[0].axhline(0.5, color="k", lw=0.5, ls="--")
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels(order, rotation=50, ha="right")
    ax[0].set_ylabel("within-chain AUROC, held-out families (95% CI)")
    ax[0].set_title("A  which score flags Mg2+ sites?")
    piv = Mx.pivot_table(index=["pdb_id", "chain"], columns="score", values="auroc")
    emb = [n for n in names if not n.endswith("_rand")]
    for k, n_ in enumerate(emb):
        if f"{n_}__ion" in piv and "seq_vienna" in piv:
            ax[1].scatter(piv["seq_vienna"], piv[f"{n_}__ion"], s=6, alpha=0.5, label=n_, color=f"C{k}")
    ax[1].plot([0, 1], [0, 1], color="k", lw=0.5)
    ax[1].set_xlabel("AUROC, sequence + ViennaRNA baseline")
    ax[1].set_ylabel("AUROC, embedding probe")
    ax[1].set_title("B  chain by chain (above the line = embedding better)")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_ion_probe.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("embed", "probe"):
        p = sp.add_parser(name)
        p.add_argument("--labels-dir", type=Path, default=DATA / "phi_res")
        p.add_argument("--code-dir", default="~/projects/codes/rna_mg")
        p.add_argument("--out-dir", type=Path, default=DATA / "ion_probe")
        p.add_argument("--seed", type=int, default=0)
    e = sp.choices["embed"]
    e.add_argument("--model", required=True, choices=list(AUDIT_LAYER))
    e.add_argument("--layers", type=int, nargs="*", default=None, help="default: the audit's layer for this model")
    e.add_argument("--random-init", action="store_true")
    e.add_argument("--device", default="cuda")
    e.add_argument("--max-len", type=int, default=1024)
    e.add_argument("--include-flagged", action="store_true")
    e.add_argument("--limit-structures", type=int, default=None)
    pr = sp.choices["probe"]
    pr.add_argument("--caches", nargs="+", required=True, help="name=path/to/emb_*.npz; name 'X_rand' marks the random twin of X")
    pr.add_argument("--labels-table", type=Path, default=DATA / "ion_pilot" / "ion_labels_5A.csv.gz")
    pr.add_argument("--structure-controls", type=Path, default=DATA / "structure_controls.csv.gz")
    pr.add_argument("--dssr-controls", type=Path, default=DATA / "dssr_controls.csv.gz")
    pr.add_argument("--family-table", type=Path, default=DATA / "controls" / "families_kmer_t0.5.csv")
    pr.add_argument("--folds", type=int, default=5)
    pr.add_argument("--cutoff", type=float, default=5.0)
    pr.add_argument("--min-p", type=int, default=8)
    pr.add_argument("--min-neg", type=int, default=3)
    pr.add_argument("--gbm-iter", type=int, default=200)
    pr.add_argument("--extras", action="store_true",
                    help="fairness controls: position-only / letters-only baselines, a LINEAR sequence baseline, and a NONLINEAR "
                         "(PCA-64 + boosting) head on the embeddings, alone and combined with seq_vienna")
    pr.add_argument("--pca-dim", type=int, default=64)
    pr.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    {"embed": cmd_embed, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())