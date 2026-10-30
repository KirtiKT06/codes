#!/usr/bin/env python
"""
Put several audit_variance_partition.csv files side by side (e.g. pretrained vs random-init vs
leave-family-out folds) so the control comparison is one table, not a hunt through folders.

    python compare_audit_runs.py \
        pretrained=/data/rna_electrostats/centered_latent_v3/ernierna/audit_variance_partition.csv \
        random=/data/rna_electrostats/random_init/ernierna/audit_variance_partition.csv \
        --controls controls_geom_coulomb --weighting micro

Several files can share a label (e.g. family=.../fold0/... family=.../fold1/...): their values are
averaged and the min-max across files is shown, which is how leave-family-out folds are pooled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=path/to/audit_variance_partition.csv")
    ap.add_argument("--controls", default="controls_geom_coulomb")
    ap.add_argument("--weighting", default="micro", choices=["micro", "macro"])
    ap.add_argument("--components", nargs="+", default=["unique_lm", "shared", "unique_controls", "unexplained"])
    ap.add_argument("--out", type=Path, default=None, help="optional CSV")
    a = ap.parse_args()

    frames = []
    for item in a.runs:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"'{item}' must look like label=path")
        d = pd.read_csv(path)
        d = d[(d.controls == a.controls) & (d.weighting == a.weighting) & d.component.isin(a.components)]
        if d.empty:
            raise SystemExit(f"{path}: no rows for controls={a.controls}, weighting={a.weighting}")
        d = d.assign(run=label)
        frames.append(d)
    allr = pd.concat(frames)
    g = allr.groupby(["target", "component", "run"])["value"].agg(["mean", "min", "max", "count"]).reset_index()

    lines = []
    for target, gt in g.groupby("target"):
        lines.append(f"\n{target}  [{a.controls}, {a.weighting}]")
        for comp in a.components:
            row = []
            for run in dict.fromkeys(r.partition("=")[0] for r in a.runs):
                x = gt[(gt.component == comp) & (gt.run == run)]
                if x.empty:
                    continue
                x = x.iloc[0]
                txt = f"{x['mean']:+.3f}" + (f" [{x['min']:+.3f}..{x['max']:+.3f}, n={int(x['count'])}]"
                                              if x["count"] > 1 else "")
                row.append(f"{run}: {txt}")
            lines.append(f"  {comp:<16} " + "   ".join(row))
    print("\n".join(lines))
    if a.out:
        g.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
