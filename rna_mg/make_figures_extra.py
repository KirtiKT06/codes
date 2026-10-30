#!/usr/bin/env python
"""
Figures for the new analyses. Imports styling/helpers from make_figures.py (keep both files in one folder).

    python make_figures_extra.py --root /data/rna_electrostats --out figures

  fig8_controls_grid_micro / _macro   pretrained vs random-init vs leave-family-out, all four models, with error bars
                                      (single run: 95% bootstrap CI; several runs: mean with min-max). Prefers the
                                      layer-matched runs in <root>/controls_matched/, falls back to the old folders.
  fig10_power                         spike-in power analysis: injected share of variance vs what the audit recovers
  fig11_length                        shared / unique-LM variance by chain-length bin, pretrained vs random
  fig12_paired                        pretrained MINUS random on identical test chains, with paired bootstrap CIs
  fig13_family_paired                 the same on pooled held-out-family folds (needs run_family_analysis.sh)
  fig7_benchmark, fig9_mg             benchmark fidelity vs lDDT with the nulls (--bench-dir) and Mg2+ sensitivity (--mg-dir)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_figures import (MODELS, MODEL_LABEL, MODEL_COLOR, TARGET_LABEL, AUDIT_TARGETS, save)  # noqa: E402

RUN_COLOR = {"pretrained": "#0072B2", "random-init": "#999999", "leave-family-out": "#D55E00"}
PART = "audit_variance_partition.csv"


# ----------------------------------------------------------------------------------------------
def collect_runs(root: Path, model: str, fam_thr: str) -> tuple[pd.DataFrame, dict]:
    """All partition tables for one model, labelled by run kind. Returns (table, provenance)."""
    cm, v3 = root / "controls_matched", root / "centered_latent_v3"
    spec = {
        "pretrained": ([cm / "pretrained" / model / PART], [v3 / model / PART]),
        "random-init": (sorted(cm.glob(f"random_s*/{model}/{PART}")), [root / "controls/random_init" / model / PART]),
        "leave-family-out": (sorted(cm.glob(f"family_t{fam_thr}_fold*/{model}/{PART}")),
                             sorted((root / "controls/family" / model).glob(f"fold*/{PART}"))),
    }
    frames, prov = [], {}
    for run, (matched, old) in spec.items():
        paths = [p for p in matched if p.exists()]
        prov[run] = "matched" if paths else "old"
        if not paths:
            paths = [p for p in old if p.exists()]
        for i, p in enumerate(paths):
            frames.append(pd.read_csv(p).assign(model=model, run=run, rep=i))
        if not paths:
            prov[run] = "missing"
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), prov


def fig_controls_grid(root: Path, outdir: Path, controls: str, weighting: str, fam_thr: str):
    frames, prov_all = [], {}
    for m in MODELS:
        d, prov = collect_runs(root, m, fam_thr)
        prov_all[m] = prov
        if not d.empty:
            frames.append(d)
    if not frames:
        print("  [skip] fig8 grid: nothing found")
        return
    d = pd.concat(frames)
    d = d[(d.controls == controls) & (d.weighting == weighting) & d.component.isin(["shared", "unique_lm"])]
    models = [m for m in MODELS if m in set(d.model)]
    runs = [r for r in RUN_COLOR if r in set(d.run)]
    fig, axes = plt.subplots(2, len(AUDIT_TARGETS), figsize=(2.9 * len(AUDIT_TARGETS), 4.7), sharey="row")
    w = 0.26
    for ri, comp in enumerate(["shared", "unique_lm"]):
        for j, t in enumerate(AUDIT_TARGETS):
            ax = axes[ri, j]
            for mi, m in enumerate(models):
                for ki, run in enumerate(runs):
                    g = d[(d.model == m) & (d.run == run) & (d.target == t) & (d.component == comp)]
                    if g.empty:
                        continue
                    x = mi + (ki - (len(runs) - 1) / 2) * w
                    y = g.value.mean()
                    if len(g) == 1:
                        lo, hi = float(g.lo.iloc[0]), float(g.hi.iloc[0])      # bootstrap CI
                    else:
                        lo, hi = g.value.min(), g.value.max()                   # spread across runs
                    ax.bar(x, y, width=w * 0.92, color=RUN_COLOR[run],
                           label=run if (ri == 0 and j == 0 and mi == 0) else None)
                    ax.errorbar(x, y, yerr=[[max(y - lo, 0)], [max(hi - y, 0)]], fmt="none", ecolor="k", lw=0.6, capsize=1.2)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=40, ha="right")
            if ri == 0:
                ax.set_title(TARGET_LABEL.get(t, t))
            if j == 0:
                ax.set_ylabel(("shared with controls" if comp == "shared" else "unique to LM") +
                              f"\n(fraction of variance, {weighting})")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.035))
    tag = "; ".join(f"{r}: {'layer-matched' if all(prov_all[m].get(r) == 'matched' for m in models) else 'see note'}"
                    for r in runs)
    fig.text(0.5, 0.0, f"error bars: single run = 95% bootstrap CI, several runs (random seeds / family folds) = min–max.  {tag}",
             ha="center", fontsize=6, color="0.3")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, outdir, f"fig8_controls_grid_{weighting}")
    print("   provenance:", {m: prov_all[m] for m in models})


# ----------------------------------------------------------------------------------------------
def fig_power(root: Path, outdir: Path, controls: str):
    frames = []
    for m in MODELS:
        f = root / "controls_matched" / "pretrained" / m / "spike_in_summary.csv"
        if f.exists():
            frames.append(pd.read_csv(f).assign(model=m))
    if not frames:
        print("  [skip] fig10: no spike_in_summary.csv under controls_matched/pretrained/<model>/")
        return
    d = pd.concat(frames)
    d = d[d.controls == controls]
    types = [t for t in ("linear", "interaction") if t in set(d.spike_type)]
    targets = list(dict.fromkeys(d.target))
    fig, axes = plt.subplots(len(types), len(targets), figsize=(2.9 * len(targets), 2.7 * len(types)), squeeze=False,
                             sharex=True)
    for i, st in enumerate(types):
        for j, t in enumerate(targets):
            ax = axes[i, j]
            top = d.injected.max()
            ax.plot([0, top], [0, top], color="0.7", lw=0.8, ls=":", label="ideal (recovered = injected)")
            for m in MODELS:
                for w, ls in (("micro", "-"), ("macro", "--")):
                    g = d[(d.model == m) & (d.spike_type == st) & (d.target == t) & (d.weighting == w)].sort_values("injected")
                    if g.empty:
                        continue
                    ax.plot(g.injected, g.value, ls, color=MODEL_COLOR[m], marker="o", ms=2.5, lw=1,
                            label=(MODEL_LABEL[m] if (w == "micro" and i == 0 and j == 0) else None))
                    if w == "micro":
                        ax.fill_between(g.injected, g.lo, g.hi, color=MODEL_COLOR[m], alpha=0.15, lw=0)
            ax.set_title(f"{st} signal; {TARGET_LABEL.get(t, t)}", fontsize=7.5)
            if i == len(types) - 1:
                ax.set_xlabel("injected share of label variance")
            if j == 0:
                ax.set_ylabel("recovered unique-LM variance")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper left")
    axes[0, -1].plot([], [], "k-", label="micro")
    axes[0, -1].plot([], [], "k--", label="macro")
    axes[0, -1].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    save(fig, outdir, "fig10_power")


# ----------------------------------------------------------------------------------------------
def _bin_key(label: str) -> float:
    nums = re.findall(r"\d+", label)
    return float(nums[0]) + (1e6 if label.startswith(">") else 0)


def fig_length(root: Path, outdir: Path, controls: str):
    f = root / "controls_matched" / "analysis" / "length_partition.csv"
    if not f.exists():
        print(f"  [skip] fig11: {f} not found (run run_analysis_matched.sh)")
        return
    d = pd.read_csv(f)
    d = d[(d.controls == controls) & (d.weighting == "micro") & d.component.isin(["shared", "unique_lm"])
          & (d.length_bin != "all")]
    d["kind"] = np.where(d.run == "pretrained", "pretrained", np.where(d.run.str.startswith("random"), "random", "other"))
    d = d[d.kind != "other"]
    bins = sorted(set(d.length_bin), key=_bin_key)
    targets = [t for t in AUDIT_TARGETS if t in set(d.target)]
    models = [m for m in MODELS if m in set(d.model)]
    fig, axes = plt.subplots(2, len(targets), figsize=(2.9 * len(targets), 4.6), squeeze=False)
    for ri, comp in enumerate(["shared", "unique_lm"]):
        for j, t in enumerate(targets):
            ax = axes[ri, j]
            for m in models:
                for kind, ls in (("pretrained", "-"), ("random", "--")):
                    g = d[(d.model == m) & (d.target == t) & (d.component == comp) & (d.kind == kind)]
                    if g.empty:
                        continue
                    a = g.groupby("length_bin").agg(v=("value", "mean"), lo=("lo", "mean"), hi=("hi", "mean"))
                    a = a.reindex(bins)
                    x = np.arange(len(bins))
                    ax.plot(x, a.v.values, ls, color=MODEL_COLOR[m], marker="o", ms=3, lw=1,
                            label=f"{MODEL_LABEL[m]} {kind}" if (ri == 0 and j == 0) else None)
                    if kind == "pretrained":
                        ax.fill_between(x, a.lo.values, a.hi.values, color=MODEL_COLOR[m], alpha=0.12, lw=0)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels(bins, rotation=30)
            if ri == 0:
                ax.set_title(TARGET_LABEL.get(t, t))
            if ri == 1:
                ax.set_xlabel("residues per test chain (bin)")
            if j == 0:
                ax.set_ylabel(("shared with controls" if comp == "shared" else "unique to LM") + "\n(fraction, micro)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, outdir, "fig11_length")


# ----------------------------------------------------------------------------------------------
def fig_paired(root: Path, outdir: Path, controls: str):
    f = root / "controls_matched" / "analysis" / "paired_differences.csv"
    if not f.exists():
        print(f"  [skip] fig12: {f} not found (run run_analysis_matched.sh)")
        return
    d = pd.read_csv(f)
    d = d[d.controls == controls]
    targets = [t for t in AUDIT_TARGETS if t in set(d.target)]
    models = [m for m in MODELS if m in set(d.tag)]
    fig, axes = plt.subplots(2, len(targets), figsize=(2.9 * len(targets), 4.4), squeeze=False, sharey="row")
    for ri, comp in enumerate(["shared", "unique_lm"]):
        for j, t in enumerate(targets):
            ax = axes[ri, j]
            for mi, m in enumerate(models):
                for wi, (w, mk) in enumerate((("micro", "o"), ("macro", "s"))):
                    g = d[(d.tag == m) & (d.target == t) & (d.component == comp) & (d.weighting == w)]
                    if g.empty:
                        continue
                    r = g.iloc[0]
                    x = mi + (wi - 0.5) * 0.25
                    ax.errorbar(x, r.diff_a_minus_b, yerr=[[r.diff_a_minus_b - r.lo], [r.hi - r.diff_a_minus_b]],
                                fmt=mk, ms=4, color=MODEL_COLOR[m], mfc=MODEL_COLOR[m] if w == "micro" else "white",
                                capsize=1.5, lw=0.9)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=40, ha="right")
            if ri == 0:
                ax.set_title(TARGET_LABEL.get(t, t))
            if j == 0:
                ax.set_ylabel(f"pretrained − random\n({'shared' if comp == 'shared' else 'unique-LM'} variance)")
    axes[0, -1].plot([], [], "ko", ms=4, label="micro (all residues equal)")
    axes[0, -1].plot([], [], "ks", ms=4, mfc="white", label="macro (all chains equal)")
    h, l = axes[0, -1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, outdir, "fig12_paired")


def fig_family_paired(root: Path, outdir: Path, controls: str):
    f = root / "controls_matched" / "analysis" / "family_paired_differences.csv"
    if not f.exists():
        print(f"  [skip] fig13: {f} not found (run run_family_analysis.sh)")
        return
    d = pd.read_csv(f)
    d = d[d.controls == controls].copy()
    d["model"] = d.tag.str.split("_t").str[0]
    targets = [t for t in AUDIT_TARGETS if t in set(d.target)]
    models = [m for m in MODELS if m in set(d.model)]
    fig, axes = plt.subplots(2, len(targets), figsize=(2.9 * len(targets), 4.4), squeeze=False, sharey="row")
    for ri, comp in enumerate(["shared", "unique_lm"]):
        for j, t in enumerate(targets):
            ax = axes[ri, j]
            for mi, m in enumerate(models):
                for wi, (w, mk) in enumerate((("micro", "o"), ("macro", "s"))):
                    g = d[(d.model == m) & (d.target == t) & (d.component == comp) & (d.weighting == w)]
                    if g.empty:
                        continue
                    r = g.iloc[0]
                    x = mi + (wi - 0.5) * 0.25
                    ax.errorbar(x, r["diff"], yerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt=mk, ms=4,
                                color=MODEL_COLOR[m], mfc=MODEL_COLOR[m] if w == "micro" else "white", capsize=1.5, lw=0.9)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=40, ha="right")
            if ri == 0:
                ax.set_title(TARGET_LABEL.get(t, t))
            if j == 0:
                ax.set_ylabel(f"pretrained − random\n({'shared' if comp == 'shared' else 'unique-LM'} variance)")
    axes[0, -1].plot([], [], "ko", ms=4, label="micro (all residues equal)")
    axes[0, -1].plot([], [], "ks", ms=4, mfc="white", label="macro (all chains equal)")
    h, l = axes[0, -1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.0, "pooled over five held-out-family folds, random-init on the same folds; 95% bootstrap CI over structures",
             ha="center", fontsize=6, color="0.3")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, outdir, "fig13_family_paired")


# ----------------------------------------------------------------------------------------------
# fig7 (benchmark) and fig9 (Mg2+): self-contained, so they do not depend on make_figures.py being current
# ----------------------------------------------------------------------------------------------
def find_bench(root: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    for name in ("bench_apbs_v2", "bench_coulomb_v2", "bench_out"):
        if (root / name / "chain_metrics_with_excess.csv").exists() or (root / name / "chain_metrics.csv").exists():
            return root / name
    return root / "bench_apbs_v2"


def fig_benchmark(metrics: Path, excess: Path | None, outdir: Path):
    from scipy import stats
    d = pd.read_csv(metrics)
    if "status" in d:
        d = d[d.status == "ok"]
    show = [f"pred_pearson_{q}" for q in ("c1p", "m4", "m8", "m12") if f"pred_pearson_{q}" in d.columns]
    if not show or "lddt_c1p" not in d:
        print("  [skip] fig7: expected columns lddt_c1p / pred_pearson_* not found")
        return
    have_ex = excess is not None and excess.exists()
    fig, axes = plt.subplots(1, len(show) + (1 if have_ex else 0), figsize=(2.7 * (len(show) + have_ex), 3.1))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, show):
        ok = d[["lddt_c1p", c]].dropna()
        ax.scatter(ok.lddt_c1p, ok[c], s=7, alpha=0.6, color="#0072B2", lw=0)
        rho = stats.spearmanr(ok.lddt_c1p, ok[c]).statistic
        ax.text(0.04, 0.96, f"Spearman ρ = {rho:.2f}\nn = {len(ok)} chains", transform=ax.transAxes, fontsize=6.5, va="top")
        ax.axhline(0.8, color="0.6", lw=0.5, ls="--")
        okb = ok.assign(b=pd.cut(ok.lddt_c1p, [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01], right=False))
        med = okb.groupby("b", observed=True).agg(x=("lddt_c1p", "median"), y=(c, "median"))
        ax.plot(med.x, med.y, "-D", color="#D55E00", ms=3, lw=1, label="median per lDDT bin")
        ax.set_xlabel("C1′-lDDT of prediction")
        ax.set_ylabel("within-chain Pearson with native map")
        ax.set_title(TARGET_LABEL.get(c.replace("pred_pearson_", ""), c), fontsize=8)
        ax.set_ylim(top=1.0, bottom=-0.25)
        ax.set_xlim(0.2, 1.0)
        if c == show[0]:
            ax.legend(frameon=False, fontsize=6, loc="lower right")
    if have_ex:
        e = pd.read_csv(excess)
        ax = axes[-1]
        kinds = list(dict.fromkeys(e["null"]))
        qtys = list(dict.fromkeys(e["qty"]))
        w = 0.8 / max(len(qtys), 1)
        for qi, q in enumerate(qtys):
            r = e[e.qty == q].set_index("null")
            y = np.array([r.median_excess.get(k, np.nan) for k in kinds])
            lo = np.array([r.ci_lo.get(k, np.nan) for k in kinds])
            hi = np.array([r.ci_hi.get(k, np.nan) for k in kinds])
            xs = np.arange(len(kinds)) + (qi - (len(qtys) - 1) / 2) * w
            ax.errorbar(xs, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="o", ms=3.5, capsize=1.5,
                        lw=0.8, label=TARGET_LABEL.get(q, q))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(kinds)))
        nice = {"smooth": "smooth\n(RMSD-matched)", "jitter": "jitter\n(RMSD-matched)",
                "smoothL": "smooth\n(lDDT-matched)", "jitterL": "jitter\n(lDDT-matched)"}
        ax.set_xticklabels([nice.get(k, k) for k in kinds], rotation=30, fontsize=6)
        ax.set_ylabel("prediction − matched null (median, 95% CI)")
        ax.set_title("vs distortion nulls", fontsize=8)
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, outdir, "fig7_benchmark")


def _bin_edges(s: str):
    nums = re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", str(s))
    return float(nums[0]), float(nums[1])


def fig_mg(path: Path, outdir: Path):
    d = pd.read_csv(path)
    variants = list(dict.fromkeys(d.variant))
    charge = [v for v in variants if v.startswith("mg_r")]
    ctrl = [v for v in variants if v.startswith("mg0_r")]
    if not charge:
        print("  [skip] fig9: no Mg variant in file")
        return
    cv = charge[0]
    d = d.assign(lo=[_bin_edges(b)[0] for b in d.bin], hi=[_bin_edges(b)[1] for b in d.bin])
    bins = d[["lo", "hi"]].drop_duplicates().sort_values("lo")
    labels = [f"{lo:g}–{hi:g}" if hi < 100 else f">{lo:g}" for lo, hi in zip(bins.lo, bins.hi)]
    x = np.arange(len(bins))
    qtys = [q for q in ("c1p", "m4", "m8", "m12") if q in set(d.qty)]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    ax = axes[0]
    for q, colr in zip(qtys, ["#999999", "#56B4E9", "#0072B2", "#D55E00"]):
        r = d[(d.variant == cv) & (d.qty == q)].set_index("lo").reindex(bins.lo)
        ax.plot(x, 100 * r.rms_change_over_label_sd.values, "-o", ms=3, lw=1.1, color=colr, label=TARGET_LABEL.get(q, q))
    ax.axhline(100, color="k", lw=0.5, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_xlabel("distance to nearest Mg$^{2+}$ (Å)")
    ax.set_ylabel("rms change of the label\n(% of the label's own sd)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=6)
    ax.text(-0.16, 1.04, "A", transform=ax.transAxes, fontsize=10, fontweight="bold")
    ax = axes[1]
    q = "m8" if "m8" in qtys else qtys[0]
    r1 = d[(d.variant == cv) & (d.qty == q)].set_index("lo").reindex(bins.lo)
    ax.plot(x, r1.rms_change.values, "-o", ms=3, lw=1.1, color="#0072B2", label="with Mg$^{2+}$ charge")
    if ctrl:
        r0 = d[(d.variant == ctrl[0]) & (d.qty == q)].set_index("lo").reindex(bins.lo)
        ax.plot(x, r0.rms_change.values, "-s", ms=3, lw=1.1, color="#999999", label="same cavity, charge 0 (control)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_xlabel("distance to nearest Mg$^{2+}$ (Å)")
    ax.set_ylabel(f"rms change, {TARGET_LABEL.get(q, q)} (kT/e)")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=6)
    ax.text(-0.16, 1.04, "B", transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, outdir, "fig9_mg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/data/rna_electrostats"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--controls", default="controls_geom_coulomb")
    ap.add_argument("--family-threshold", default="0.5")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--bench-dir", type=Path, default=None, help="folder with chain_metrics*.csv for fig7 (default bench_apbs_v2)")
    ap.add_argument("--mg-dir", type=Path, default=None, help="folder with mg_effect_by_distance.csv (default <root>/mg_full)")
    a = ap.parse_args()
    want = lambda k: a.only is None or any(k.startswith(o) for o in a.only)  # noqa: E731
    if want("fig8"):
        for w in ("micro", "macro"):
            fig_controls_grid(a.root, a.out, a.controls, w, a.family_threshold)
    if want("fig10"):
        fig_power(a.root, a.out, a.controls)
    if want("fig11"):
        fig_length(a.root, a.out, a.controls)
    if want("fig12"):
        fig_paired(a.root, a.out, a.controls)
    if want("fig7"):
        bd = find_bench(a.root, a.bench_dir)
        m = bd / "chain_metrics_with_excess.csv"
        if not m.exists():
            m = bd / "chain_metrics.csv"
            print(f"  [note] fig7: run `python predicted_structure_benchmark.py summarize --outdir {bd}` for the null panel")
        if m.exists():
            fig_benchmark(m, bd / "excess_vs_null.csv", a.out)
        else:
            print(f"  [skip] fig7: no chain_metrics*.csv in {bd}")
    if want("fig9"):
        mg = (a.mg_dir or a.root / "mg_full") / "mg_effect_by_distance.csv"
        if mg.exists():
            fig_mg(mg, a.out)
        else:
            print(f"  [skip] fig9: {mg} not found")
    if want("fig13"):
        fig_family_paired(a.root, a.out, a.controls)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
