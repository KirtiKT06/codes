#!/usr/bin/env python
"""
Fixes two layout problems in make_figures.py (a backup is written first):
  fig2_partition : legend overlapped the x tick labels of the first panel -> figure-level legend at the bottom
  fig4_ladder    : legend overlapped the bars, and the first y label was cut off -> legend at the bottom, short y labels
Refuses to run if an anchor is not found exactly once.
    python patch_figures.py make_figures.py
"""
import shutil
import sys
from pathlib import Path


def sub(t, old, new):
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"ABORT: anchor found {n}x (need 1): {old.strip()[:80]!r}; nothing written")
    return t.replace(old, new)


p = Path(sys.argv[1] if len(sys.argv) > 1 else "make_figures.py").expanduser()
t = p.read_text()
t = sub(t, '''    axes[0, 0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.0), ncol=1, fontsize=6)
''', '''    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0))
''')
t = sub(t, '''    fig.tight_layout()
    save(fig, outdir, "fig2_partition")''', '''    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, outdir, "fig2_partition")''')
t = sub(t, '''                ax.set_ylabel(f"{w} $R^2$ (within-chain centered)")''', '''                ax.set_ylabel(f"{w} $R^2$")''')
t = sub(t, '''    axes[0, 0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    save(fig, outdir, "fig4_ladder")''', '''    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, -0.02, "R² of the within-chain-centred label on held-out structures", ha="center", fontsize=6, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, outdir, "fig4_ladder")''')
compile(t, str(p), "exec")
shutil.copy(p, p.with_suffix(".py.bak"))
p.write_text(t)
print("patched", p, "backup:", p.with_suffix(".py.bak"))
