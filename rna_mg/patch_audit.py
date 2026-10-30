#!/usr/bin/env python
"""
Patch centered_latent_manifold.py in place (a backup is written first).

    python patch_audit.py ~/projects/codes/rna_mg/centered_latent_manifold.py

Two additions, nothing else changes:

 1. The audit now also writes  audit_chain_stats.csv.gz  (per test chain: pdb, chain, n, sse, sst, for every
    predictor set and target). This is what lets audit_extras.py do chain-length stratification and PAIRED
    bootstrap comparisons between runs offline, with no refitting.

 2. New flags for a POWER ANALYSIS (spike-in):
        --spike-fractions 0.005 0.01 0.02 0.05
        --spike-type linear interaction
        --spike-targets potential_mean_kT_e_12A
        --spike-n-boot 100
    A synthetic signal that depends ONLY on the LM embedding, and has been made unpredictable from the controls,
    is added to the (centered) label with a known share f of the label variance. The audit is rerun on that label.
    If `unique_lm` comes back close to f, the audit could have seen an effect of size f; if it comes back much
    smaller, a measured 'unique_lm ~ 0' is weaker evidence. Output: spike_in_summary.csv and one sub-folder per
    (type, target, f). Requires --audit.

The patcher refuses to run if any anchor text is not found exactly once (i.e. if your file differs from the one
this was written against), so it cannot half-apply.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def sub(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"ABORT: anchor found {n}x (need exactly 1): {old.strip()[:90]!r}\n"
                         "Your centered_latent_manifold.py differs from the version this patch targets; nothing written.")
    return text.replace(old, new)


CHAIN_STATS = '''def chain_stats(y, pred, chain_ids, pdb_ids) -> pd.DataFrame:
    """Per-CHAIN sums (pdb, chain, n, sse, sst). Saved by the audit so later analyses (chain-length bins,
    paired bootstraps between runs) can be done offline without refitting anything."""
    df = pd.DataFrame({"pdb": pdb_ids, "chain": chain_ids, "se": (y - pred) ** 2, "s2": y ** 2})
    return df.groupby(["pdb", "chain"]).agg(sse=("se", "sum"), sst=("s2", "sum"), n=("se", "size")).reset_index()


def r2_micro_macro(stats: pd.DataFrame, w: np.ndarray):'''

SPIKE_FN = '''def run_spike_in(H, Y_cen, blocks, splits, chain_groups, pdb_groups, targets_all, args):
    """Power analysis. Adds to the label a synthetic signal that (i) is a function of the LM embedding only and
    (ii) has been made unpredictable from the controls, with a known share f of the final label variance, then
    reruns the audit and reports how much `unique_lm` it recovers."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    train, test = splits == "train", splits == "test"
    scaler = StandardScaler().fit(H[train])
    pca = PCA(n_components=min(args.pca_dim, H.shape[1]), random_state=args.seed).fit(scaler.transform(H[train]))
    Hp = pca.transform(scaler.transform(H))
    Hp = Hp / Hp[train].std(0)
    names = [n for n in ("F", "struct", "dssr", "ctx", "coul") if n in blocks]
    M = np.column_stack([blocks[n] for n in names])
    Xc = np.column_stack([M, within_chain_center(M, chain_groups)])
    k = max(1, min(16, Hp.shape[1] // 2))
    all_rows = []
    for stype in args.spike_type:
        rng = np.random.default_rng(args.seed + 17)
        w1, w2 = rng.normal(size=k), rng.normal(size=k)
        if stype == "linear":
            s = Hp[:, :k] @ w1                                   # a probe that is linear in the PCs can find it
        else:
            s = np.tanh(Hp[:, :k] @ w1) * (Hp[:, k:2 * k] @ w2)  # an interaction: needs a nonlinear probe
        s = within_chain_center(s[:, None], chain_groups)[:, 0]
        gb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                           min_samples_leaf=200, early_stopping=False, random_state=args.seed)
        s_perp = s - gb.fit(Xc[train], s[train]).predict(Xc)    # remove whatever the controls can predict
        r2_ctrl = 1.0 - float(np.var(s_perp[test]) / max(np.var(s[test]), 1e-12))
        s_perp = s_perp / s_perp[train].std()
        log.info("SPIKE-IN (%s): controls explained %.1f%% of the raw synthetic signal before it was removed",
                 stype, 100 * r2_ctrl)
        for t in args.spike_targets:
            j = targets_all.index(t)
            sd = float(Y_cen[train, j].std())
            for f in args.spike_fractions:
                Y_s = Y_cen.copy()
                Y_s[:, j] = Y_cen[:, j] + sd * np.sqrt(f / (1.0 - f)) * s_perp   # signal = share f of total variance
                sub = args.outdir / f"spike_{stype}_{t}_f{f:g}"
                run_audit(H, Y_s, blocks, splits, chain_groups, pdb_groups, targets_all, [t], sub,
                          seed=args.seed, n_boot=args.spike_n_boot, pca_dim=args.pca_dim)
                P = pd.read_csv(sub / "audit_variance_partition.csv")
                P = P[P.component == "unique_lm"].assign(injected=f, spike_type=stype,
                                                         control_explained_raw_signal=r2_ctrl)
                all_rows.append(P)
                for _, r in P.iterrows():
                    log.info("  spike %-11s %-26s f=%-6g %-24s %-5s recovered unique_lm = %+.4f [%+.4f, %+.4f]",
                             stype, t, f, r.controls, r.weighting, r.value, r.lo, r.hi)
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.outdir / "spike_in_summary.csv", index=False)
    log.info("SPIKE-IN written to %s", args.outdir / "spike_in_summary.csv")


def main():'''

SPIKE_ARGS = '''    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--spike-fractions", type=float, nargs="*", default=[],
                    help="Power analysis: inject a synthetic LM-only signal with these shares of the label variance "
                         "and report how much unique_lm the audit recovers (requires --audit)")
    ap.add_argument("--spike-type", nargs="+", default=["linear"], choices=["linear", "interaction"])
    ap.add_argument("--spike-targets", nargs="+", default=["potential_mean_kT_e_12A"])
    ap.add_argument("--spike-n-boot", type=int, default=100)'''


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1]).expanduser()
    text = path.read_text()
    if "run_spike_in" in text:
        raise SystemExit("already patched")
    t = text
    t = sub(t, "def r2_micro_macro(stats: pd.DataFrame, w: np.ndarray):", CHAIN_STATS)
    t = sub(t, "    rows, part_rows = [], []\n", "    rows, part_rows, chain_rows = [], [], []\n")
    t = sub(t, "        stats = {k: structure_stats(y[test], p, chain_t, pdb_t) for k, p in preds.items()}\n",
            "        stats = {k: structure_stats(y[test], p, chain_t, pdb_t) for k, p in preds.items()}\n"
            "        for k, p in preds.items():\n"
            "            chain_rows.append(chain_stats(y[test], p, chain_t, pdb_t).assign(target=t, predictors=k))\n")
    t = sub(t, '    pd.DataFrame(rows).to_csv(outdir / "audit_probe_r2.csv", index=False)\n',
            '    outdir.mkdir(parents=True, exist_ok=True)\n'
            '    pd.DataFrame(rows).to_csv(outdir / "audit_probe_r2.csv", index=False)\n'
            '    pd.concat(chain_rows, ignore_index=True).to_csv(outdir / "audit_chain_stats.csv.gz", index=False,\n'
            '                                                    compression="gzip")\n')
    t = sub(t, '    ap.add_argument("--pca-dim", type=int, default=64)', SPIKE_ARGS)
    t = sub(t, "def main():", SPIKE_FN)
    t = sub(t, "                  args.outdir, seed=args.seed, n_boot=args.n_boot, pca_dim=args.pca_dim)\n",
            "                  args.outdir, seed=args.seed, n_boot=args.n_boot, pca_dim=args.pca_dim)\n"
            "        if args.spike_fractions:\n"
            "            run_spike_in(H, Y_cen, bl, splits, chain_groups, pdb_groups, ALL_TARGETS, args)\n")
    compile(t, str(path), "exec")                       # syntax check before anything is written
    shutil.copy(path, path.with_suffix(".py.bak"))
    path.write_text(t)
    print(f"patched {path}\nbackup: {path.with_suffix('.py.bak')}")


if __name__ == "__main__":
    main()
