#!/usr/bin/env python
"""
Collects the data needed to verify four items in the project guide. Read-only; needs only pandas.

    python verify_items.py > verify_items.txt 2>&1

then send verify_items.txt.

  1. cross-model agreement values (the guide's table was read off a figure)
  2. Mg2+ distance bins and rms change per bin (the guide only lists the first three edges)
  3. which label the within-chain shuffle control (R2 0.30-0.35) was run on
  4. RNABERT: PLS test R2 against the ridge ceiling for C1' (PLS looked ~10x lower)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/rna_electrostats")
V3 = ROOT / "centered_latent_v3"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ------------------------------------------------------------------------------------------ 1
section("1. CROSS-MODEL AGREEMENT: mean_canonical_corr by pair and condition")
try:
    files = sorted((V3 / "cross_model").glob("cross_model_agreement*.csv"))
    print("files:", [f.name for f in files])
    for f in files:
        d = pd.read_csv(f)
        d["pair"] = d.model_a + "-" + d.model_b
        extra = {c: d[c].iloc[0] for c in ("controls", "residualizer", "variant") if c in d.columns}
        print(f"\n{f.name}   {extra}")
        print(d.pivot_table(index="pair", columns="condition", values="mean_canonical_corr").round(3).to_string())
except Exception as e:  # noqa: BLE001
    print("FAILED:", repr(e))

# ------------------------------------------------------------------------------------------ 2
section("2. Mg2+ EFFECT BY DISTANCE (mg_full/mg_effect_by_distance.csv)")
try:
    f = ROOT / "mg_full" / "mg_effect_by_distance.csv"
    d = pd.read_csv(f)
    print("columns:", list(d.columns))
    print("variants:", list(dict.fromkeys(d.variant)))
    cv = [v for v in dict.fromkeys(d.variant) if str(v).startswith("mg_r")]
    cv = cv[0] if cv else d.variant.iloc[0]
    x = d[d.variant == cv].copy()
    if "rms_change_over_label_sd" in x:
        x["pct_of_label_sd"] = (100 * x.rms_change_over_label_sd).round(0)
    keep = [c for c in ("qty", "bin", "n", "rms_change", "pct_of_label_sd") if c in x.columns]
    print(f"\nvariant = {cv}")
    print(x[keep].to_string(index=False))
    print("\nbin labels in order:", list(dict.fromkeys(x["bin"])))
except Exception as e:  # noqa: BLE001
    print("FAILED:", repr(e))

# ------------------------------------------------------------------------------------------ 3
section("3. WITHIN-CHAIN SHUFFLE CONTROL (priority results): which label, which R2")
try:
    base = ROOT / "priority_results_c1prime"
    print("exists:", base.exists())
    if base.exists():
        print("top level:", sorted(p.name for p in base.iterdir())[:40])
        hits = sorted(list(base.rglob("*negative*")) + list(base.rglob("*shuf*")))
        print("negative-control files:", [str(h.relative_to(base)) for h in hits][:40])
        for h in hits[:12]:
            if h.suffix == ".csv":
                print(f"\n--- {h.relative_to(base)}")
                print(pd.read_csv(h).head(12).to_string(index=False))
    for other in sorted(ROOT.glob("priority_results*")):
        if other != base:
            print("\nother priority folder:", other.name, sorted(p.name for p in other.iterdir())[:20])
    for p in sorted(ROOT.rglob("probe_*results*.csv"))[:5]:
        print("\nprobe results file:", p)
except Exception as e:  # noqa: BLE001
    print("FAILED:", repr(e))

# ------------------------------------------------------------------------------------------ 4
section("4. PLS test R2 vs ridge ceiling (centered_base_resid), all models")
try:
    for m in ("rnabert", "rnafm", "splicebert", "ernierna"):
        f = V3 / m / "centered_base_resid" / "pls_dimensionality_sweep.csv"
        d = pd.read_csv(f)
        d = d[d.target.isin(["potential_at_c1prime_kT_e", "potential_mean_kT_e_12A"])]
        piv = d.pivot_table(index="n_components", columns="target", values="test_r2").round(3)
        ceil = d.groupby("target").single_target_ceiling_r2.first().round(3)
        print(f"\n{m}: single-target ridge ceiling = {ceil.to_dict()}")
        print(piv.tail(4).to_string())
        print("embedding dimension is not stored here; n_components tried:", sorted(d.n_components.unique()))
except Exception as e:  # noqa: BLE001
    print("FAILED:", repr(e))
print("\nDONE")
